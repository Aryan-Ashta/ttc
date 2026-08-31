"""
Phase 4 -- Z3 Equivalence Proof.

Scope: Not the whole conv layer -- just the
innermost kernel, one MAC-accumulate-requantize sequence, K=9 (a 3x3,
cin=1 kernel, exactly what this model actually has). Proven exhaustively
over the full int8/uint8 domain via Z3 bit-vector + floating-point
reasoning; the whole-network claim is an INDUCTIVE argument (this exact
kernel runs 4*6*6=144 times with different concrete weights/bias/inputs
each time -- proving it correct for ALL symbolic weights/bias/inputs
covers every one of those 144 concrete instantiations as a special case).

Two independently-derived encodings of the SAME kernel:
  - spec_kernel(): transliterated from reference.py's conv2d_quant +
    _requantize (Phase 0's validated spec).
  - codegen_kernel(): transliterated from model.c's actual C statements
    (Phase 3's generated code) -- line by line, including the exact
    32-bit `long` semantics real ARM hardware has (not the 64-bit `long`
    the x86-64 host build used in Phase 3's test_model_host).
  - buggy_codegen_kernel(): codegen_kernel with ONE deliberately
    introduced bug (sign-extending the uint8 activation instead of
    zero-extending it -- exactly the qint8-vs-quint8 confusion Phase 0
    ran into with the real qnnpack backend). Used to demonstrate Z3
    actually catching a real counterexample, not just confirming what
    was already known correct.
"""
import json
import os
import sys

from z3 import (
    BitVec, BitVecVal, BitVecSort, Solver, sat, unsat,
    ZeroExt, SignExt, Extract, If,
    FPVal, Float64, RNE, fpSignedToFP, fpMul, fpRoundToIntegral, fpToSBV,
)

K = 9  # 3x3 kernel, cin=1 -- exactly this model's conv.

# Same default-path resolution as reference.py/ir.py.
_DEFAULT_QPARAMS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "qparams.json")
)


def load_model_constants(path=None):
    if path is None:
        path = _DEFAULT_QPARAMS if os.path.exists(_DEFAULT_QPARAMS) else "qparams.json"
    with open(path) as f:
        p = json.load(f)
    multiplier = p["input_scale"] * p["weight_scale"] / p["output_scale"]
    return {
        "input_zero_point": p["input_zero_point"],
        "output_zero_point": p["output_zero_point"],
        "multiplier": multiplier,
    }


def requantize_fp(acc_bv32, multiplier: float, output_zp: int, long_bits: int):
    """The float64-multiply + round-half-to-even + clamp step, shared by
    every version below. long_bits models the width of C's `long`."""
    acc_double = fpSignedToFP(RNE(), acc_bv32, Float64())
    mult_double = FPVal(multiplier, Float64())
    scaled = fpMul(RNE(), acc_double, mult_double)
    rounded = fpRoundToIntegral(RNE(), scaled)
    q_long = fpToSBV(RNE(), rounded, BitVecSort(long_bits))
    q = q_long + BitVecVal(output_zp, long_bits)
    zero = BitVecVal(0, long_bits)
    max255 = BitVecVal(255, long_bits)
    q_clamped = If(q < zero, zero, If(q > max255, max255, q))
    return Extract(7, 0, q_clamped)  # truncate to uint8 -- safe, already clamped to [0,255]


def spec_kernel(x, w, bias, input_zp, multiplier, output_zp, long_bits=64):
    """Direct transliteration of reference.py:
        x32 = x_u8.astype(np.int32) - x_zp        # ZERO-extend (x is uint8)
        w32 = w_int8.astype(np.int32)              # SIGN-extend (w is int8)
        acc = sum(x32*w32) + bias_int32
        <requantize>
    """
    acc = SignExt(long_bits - 32, bias)
    for i in range(K):
        xi = ZeroExt(long_bits - 8, x[i]) - BitVecVal(input_zp, long_bits)
        wi = SignExt(long_bits - 8, w[i])
        acc = acc + xi * wi
    acc32 = _trunc32(acc) if long_bits > 32 else acc
    return requantize_fp(acc32, multiplier, output_zp, long_bits)


def _trunc32(bv64):
    return Extract(31, 0, bv64)


def codegen_kernel(x, w, bias, input_zp, multiplier, output_zp, long_bits=32):
    """Line-by-line transliteration of model.c's op0_conv2d + requantize:
        int32_t acc = conv1_bias[oc];
        for (...) {
            int32_t x = (int32_t)in[...] - INPUT_ZP;   // uint8_t* -> ZERO-extend
            int32_t w = conv1_weight[...];               // int8_t, SIGN-extend
            acc += x * w;
        }
        ... requantize(acc, multiplier, output_zp) with `long` = 32 bits on
        the real ARM target (RP2040), NOT the 64-bit `long` the x86-64 host
        build happens to use.
    """
    return requantize_fp(codegen_accumulator(x, w, bias, input_zp), multiplier, output_zp, long_bits)


def codegen_accumulator(x, w, bias, input_zp):
    """Just the int32 accumulator part of codegen_kernel, split out so it
    can be compared directly (pure bitvector, no floating point) against
    codegen_accumulator_optimized below"""
    acc = bias  # already BitVec32
    for i in range(K):
        xi = ZeroExt(24, x[i]) - BitVecVal(input_zp, 32)
        wi = SignExt(24, w[i])
        acc = acc + xi * wi  # 32-bit multiply-add, wraps on overflow like real int32_t
    return acc


def buggy_codegen_kernel(x, w, bias, input_zp, multiplier, output_zp, long_bits=32):
    """Same as codegen_kernel, except the activation byte is SIGN-extended
    instead of zero-extended, as if `in` had been declared `const
    int8_t *` instead of `const uint8_t *`. A realistic bug."""
    acc = bias
    for i in range(K):
        xi = SignExt(24, x[i]) - BitVecVal(input_zp, 32)  # BUG: should be ZeroExt
        wi = SignExt(24, w[i])
        acc = acc + xi * wi
    return requantize_fp(acc, multiplier, output_zp, long_bits)


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------

def make_symbols():
    x = [BitVec(f"x{i}", 8) for i in range(K)]   # uint8 activations
    w = [BitVec(f"w{i}", 8) for i in range(K)]   # int8 weights
    bias = BitVec("bias", 32)
    return x, w, bias


def codegen_kernel_optimized(x, w, bias, input_zp, multiplier, output_zp, long_bits=32):
    """Transliteration of Phase 5's op0_conv2d (emit_conv_op_optimized in
    codegen.py): weight bytes packed into ceil(K/4) uint32_t words (zero-
    padded), then unpacked via the shift-based sign-extension trick,
    THEN used in a fully-unrolled MAC. Modeling the packing step too is the point,
    an error in the shift amounts, the padding, or a little/big-endian mixup
    would be a real bug distinct from the activation-sign-extension bug.
    """
    return requantize_fp(codegen_accumulator_optimized(x, w, bias, input_zp),
                          multiplier, output_zp, long_bits)


def codegen_accumulator_optimized(x, w, bias, input_zp):
    """Just the int32 accumulator part of codegen_kernel_optimized:
    pack/unpack/MAC, no floating point. Compared directly against
    codegen_accumulator instead of going through requantize_fp again.
    """
    K = len(w)
    K_padded = ((K + 3) // 4) * 4
    num_words = K_padded // 4

    # Pack, exactly like emit_packed_weight_array does at codegen time.
    # Assumes a little-endian target (true for RP2040 and for the x86-64
    # host used to develop this) -- a big-endian target would need the
    # byte order in this loop reversed.
    words = []
    for word_idx in range(num_words):
        word = BitVecVal(0, 32)
        for b in range(4):
            flat = word_idx * 4 + b
            byte_val = w[flat] if flat < K else BitVecVal(0, 8)
            word = word | (ZeroExt(24, byte_val) << (8 * b))
        words.append(word)

    # Unpack, exactly like the generated C's shift-based sign extension:
    # ((int32_t)(packed << shift)) >> 24.
    w_unpacked = []
    for flat in range(K):
        word_idx, byte_in_word = divmod(flat, 4)
        shift = 24 - 8 * byte_in_word
        w_unpacked.append((words[word_idx] << shift) >> 24)

    acc = bias
    for i in range(K):
        xi = ZeroExt(24, x[i]) - BitVecVal(input_zp, 32)
        acc = acc + xi * w_unpacked[i]
    return acc


def prove_optimized_equivalence(consts, timeout_ms=60000):
    """Proves spec_kernel == codegen_kernel_optimized by decomposing the proof

    Decomposed into two steps:
    1. codegen_accumulator_optimized(x,w,bias) == codegen_accumulator(x,w,bias)
       for ALL x,w,bias -- pure bitvector (pack + unpack + MAC, no FP),
       fast. This is the part that could actually hide a new bug (wrong
       shift amount, wrong padding, endianness mixup) distinct from
       anything Phase 4 already checked.
    2. Composition, not SMT: codegen_kernel_optimized(x,w,bias,...) =
       requantize_fp(codegen_accumulator_optimized(x,w,bias), ...) by
       definition, codegen_kernel(x,w,bias,...) =
       requantize_fp(codegen_accumulator(x,w,bias), ...) by definition,
       and requantize_fp is a plain deterministic function -- so step 1's
       result (equal inputs to equal-domain fn) gives equal outputs
       WITHOUT needing Z3 to re-derive it, and codegen_kernel was already
       proven == spec_kernel (Phase 4's first proof, still true, nothing
       about that changed). Equality is transitive: optimized == naive
       == spec.
    """
    x, w, bias = make_symbols()
    acc_naive = codegen_accumulator(x, w, bias, consts["input_zero_point"])
    acc_opt = codegen_accumulator_optimized(x, w, bias, consts["input_zero_point"])

    s = Solver()
    s.set("timeout", timeout_ms)
    s.add(acc_naive != acc_opt)
    print("--- codegen_accumulator_optimized vs. codegen_accumulator (pure bitvector, no FP) ---")
    result = s.check()
    if result == unsat:
        print("UNSAT: the packed/unpacked accumulator is bit-identical to the naive one, "
              "for every possible x[9]/w[9]/bias.")
        print("=> By composition with requantize_fp (deterministic) and Phase 4's original")
        print("   spec_kernel == codegen_kernel proof: spec_kernel == codegen_kernel_optimized.")
        return True
    elif result == sat:
        m = s.model()
        print("SAT: found a case where packing/unpacking does NOT reproduce the naive")
        print("accumulator -- there's a real bug in the packed-load optimization.")
        print(m)
        return False
    print(f"UNKNOWN: {result}")
    return None


def prove_equivalent(name, kernel_a, kernel_b, consts, timeout_ms=180000):
    x, w, bias = make_symbols()
    out_a = kernel_a(x, w, bias, consts["input_zero_point"], consts["multiplier"], consts["output_zero_point"])
    out_b = kernel_b(x, w, bias, consts["input_zero_point"], consts["multiplier"], consts["output_zero_point"])

    s = Solver()
    s.set("timeout", timeout_ms)
    s.add(out_a != out_b)  # ask Z3 to find a DISAGREEMENT

    print(f"--- {name} ---")
    result = s.check()
    if result == unsat:
        print("UNSAT: proven equal for ALL uint8 x[9], int8 w[9], int32 bias.")
        return True, None
    elif result == sat:
        m = s.model()
        xs = [m.evaluate(xi, model_completion=True).as_long() for xi in x]
        ws = [m.evaluate(wi, model_completion=True).as_long() for wi in w]
        # w is stored as an unsigned bitvector value 0..255; reinterpret as signed int8.
        ws_signed = [v - 256 if v >= 128 else v for v in ws]
        b = m.evaluate(bias, model_completion=True).as_long()
        if b >= 2**31:
            b -= 2**32
        print("SAT: found a counterexample where the two kernels disagree.")
        print(f"  x (uint8) = {xs}")
        print(f"  w (int8)  = {ws_signed}")
        print(f"  bias (int32) = {b}")
        return False, (xs, ws_signed, b)
    else:
        print(f"UNKNOWN (Z3 gave up / timeout): {s.reason_unknown() if hasattr(s, 'reason_unknown') else ''}")
        return None, None


def prove_no_overflow_risk(consts, timeout_ms=60000):
    """Separate check that the host's 64-bit `long` vs. the ARM target's 32-bit `long` never
    actually matters for this kernel: prove the pre-clamp int64 value fpToSBV would produce
    always fits in int32 range, for every possible x/w/bias."""
    x, w, bias = make_symbols()
    acc = bias
    for i in range(K):
        xi = ZeroExt(24, x[i]) - BitVecVal(consts["input_zero_point"], 32)
        wi = SignExt(24, w[i])
        acc = acc + xi * wi
    acc_double = fpSignedToFP(RNE(), acc, Float64())
    scaled = fpMul(RNE(), acc_double, FPVal(consts["multiplier"], Float64()))
    rounded = fpRoundToIntegral(RNE(), scaled)
    q64 = fpToSBV(RNE(), rounded, BitVecSort(64)) + BitVecVal(consts["output_zero_point"], 64)

    s = Solver()
    s.set("timeout", timeout_ms)
    # Ask Z3: does q64 ever fall outside what a 32-bit `long` can hold?
    s.add((q64 > BitVecVal(2**31 - 1, 64)) | (q64 < BitVecVal(-(2**31), 64)))
    print("--- overflow-risk check: does the host/target `long`-width difference ever matter? ---")
    result = s.check()
    if result == unsat:
        print("UNSAT: the pre-clamp value NEVER exceeds int32 range for any x/w/bias "
              "with this model's multiplier -- the 32-bit-vs-64-bit `long` difference "
              "between the ARM target and the x86-64 host build is provably immaterial "
              "for this kernel.")
        return True
    elif result == sat:
        m = s.model()
        print("SAT: found inputs where the 32-bit and 64-bit `long` casts would actually "
              "differ -- this IS a real host-vs-target risk for this kernel. Counterexample:")
        print(m)
        return False
    print("UNKNOWN")
    return None


def prove_relu_and_maxpool(consts, timeout_ms=10000):
    """Bonus: proof of ReLU and MaxPool """
    from z3 import ULT, UGT
    output_zp = consts["output_zero_point"]

    # ReLU: spec is np.maximum(x, zp); model.c is `v < zp ? zp : v`.
    v = BitVec("v", 8)
    zp = BitVecVal(output_zp, 8)
    spec_relu = If(UGT(zp, v), zp, v)          # max(v, zp) via unsigned compare
    codegen_relu = If(ULT(v, zp), zp, v)        # transliteration of `v < zp ? zp : v`
    s = Solver()
    s.set("timeout", timeout_ms)
    s.add(spec_relu != codegen_relu)
    relu_ok = s.check() == unsat
    print(f"--- bonus: ReLU spec vs. codegen --- {'UNSAT (proven equal)' if relu_ok else s.check()}")

    # MaxPool (2x2): spec is window.max(); model.c is a running `if (v>m) m=v`.
    a, b, c, d = (BitVec(n, 8) for n in "abcd")
    spec_max = If(UGT(b, a), b, a)
    spec_max = If(UGT(c, spec_max), c, spec_max)
    spec_max = If(UGT(d, spec_max), d, spec_max)
    # codegen: m starts at a (the (0,0) window element per model.c), then
    # `if (v > m) m = v` for b, c, d in turn -- same unsigned compare.
    m = a
    for v_ in (b, c, d):
        m = If(UGT(v_, m), v_, m)
    s2 = Solver()
    s2.set("timeout", timeout_ms)
    s2.add(spec_max != m)
    pool_ok = s2.check() == unsat
    print(f"--- bonus: MaxPool spec vs. codegen --- {'UNSAT (proven equal)' if pool_ok else s2.check()}")

    return relu_ok, pool_ok


def main():
    consts = load_model_constants()
    print(f"Model constants (from qparams.json): {consts}\n")

    ok1, _ = prove_equivalent("spec vs. codegen (correct)", spec_kernel, codegen_kernel, consts)
    print()
    ok2, counterexample = prove_equivalent("spec vs. codegen (BUGGY: sign-extended activation)",
                                            spec_kernel, buggy_codegen_kernel, consts)
    print()
    ok3 = prove_no_overflow_risk(consts)
    print()
    ok4, ok5 = prove_relu_and_maxpool(consts)
    print()
    ok6 = prove_optimized_equivalence(consts)
    print()

    print("=" * 70)
    print("Summary:")
    print(f"  spec == codegen (correct)      : {'PROVEN EQUAL (UNSAT)' if ok1 is True else ('MISMATCH FOUND' if ok1 is False else 'INCONCLUSIVE (timeout)')}")
    print(f"  spec == codegen (buggy)        : {'PROVEN EQUAL (unexpected!)' if ok2 is True else ('COUNTEREXAMPLE FOUND (expected)' if ok2 is False else 'INCONCLUSIVE (timeout)')}")
    print(f"  32-bit/64-bit `long` immaterial: {'PROVEN (UNSAT)' if ok3 is True else ('RISK FOUND' if ok3 is False else 'INCONCLUSIVE (timeout)')}")
    print(f"  ReLU spec == codegen (bonus)   : {'PROVEN (UNSAT)' if ok4 else 'MISMATCH/INCONCLUSIVE'}")
    print(f"  MaxPool spec == codegen (bonus): {'PROVEN (UNSAT)' if ok5 else 'MISMATCH/INCONCLUSIVE'}")
    print(f"  spec == codegen_optimized      : {'PROVEN EQUAL (UNSAT)' if ok6 is True else ('MISMATCH FOUND' if ok6 is False else 'INCONCLUSIVE (timeout)')}")

    if ok1 is True and ok2 is False and ok3 is True and ok4 and ok5 and ok6 is True:
        print("\nAll expected outcomes confirmed: the real Phase 3 codegen is proven")
        print("equivalent to the Phase 0 spec, and Z3 successfully catches the")
        print("deliberately-injected sign-extension bug with a concrete counterexample.")
        return 0
    print("\nUNEXPECTED outcome -- see above, something needs investigating.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
