"""
verify.py proves spec_kernel == codegen_kernel symbolically. This test
checks that Z3's spec_kernel encoding produces the exact same
conv output pixel reference.py's own conv2d_quant produces.
"""
import os
import sys
import numpy as np
from z3 import Solver, sat, BitVec, BitVecVal, simplify

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import reference as ref
from verify import spec_kernel, codegen_kernel, load_model_constants, K


def test_spec_kernel_matches_reference_py():
    params = ref.load_params()
    consts = load_model_constants()

    rng = np.random.default_rng(7)
    x_u8 = rng.integers(0, 256, size=(1, 8, 8), dtype=np.uint8)  # (Cin, H, W)

    # Ground truth: reference.py's actual conv output for output channel 0,
    # pixel (oy=0, ox=0), BEFORE relu/pool.
    w_int8 = np.array(params["weight_int8"], dtype=np.int8)
    bias_fp32 = np.array(params["bias_fp32"], dtype=np.float32)
    conv_out = ref.conv2d_quant(x_u8, params["input_zero_point"], w_int8, bias_fp32,
                                 params["input_scale"], params["weight_scale"],
                                 params["output_scale"], params["output_zero_point"])
    expected = int(conv_out[0, 0, 0])

    # The same 9 input values / 9 weights / bias, fed into Z3's spec_kernel
    # as CONCRETE BitVecVals (K=9 flattened 3x3 patch for oc=0, oy=0, ox=0).
    patch = x_u8[:, 0:3, 0:3].flatten()          # (Cin=1,3,3) -> 9 values, matches the C loop order (ic,ky,kx)
    kernel = w_int8[0].flatten()                  # oc=0 -> (Cin=1,3,3) -> 9 values, same order
    x_vals = [BitVecVal(int(v), 8) for v in patch]
    w_vals = [BitVecVal(int(v), 8) for v in kernel]
    bias_val = BitVecVal(int(np.rint(bias_fp32[0] / (params["input_scale"] * params["weight_scale"]))), 32)

    out_expr = spec_kernel(x_vals, w_vals, bias_val,
                            consts["input_zero_point"], consts["multiplier"], consts["output_zero_point"])
    z3_result = simplify(out_expr)
    z3_value = z3_result.as_long()

    print(f"reference.py conv_out[0,0,0] = {expected}")
    print(f"Z3 spec_kernel(...)          = {z3_value}")
    assert z3_value == expected, "Z3's spec_kernel encoding does NOT match reference.py's real behavior!"
    print("PASS: Z3's spec_kernel encoding matches reference.py.")


def test_codegen_kernel_matches_reference_py_too():
    """Same check for codegen_kernel, should also match, since verify.py
    already proved spec_kernel == codegen_kernel for ALL inputs."""
    params = ref.load_params()
    consts = load_model_constants()

    rng = np.random.default_rng(99)
    x_u8 = rng.integers(0, 256, size=(1, 8, 8), dtype=np.uint8)

    w_int8 = np.array(params["weight_int8"], dtype=np.int8)
    bias_fp32 = np.array(params["bias_fp32"], dtype=np.float32)
    conv_out = ref.conv2d_quant(x_u8, params["input_zero_point"], w_int8, bias_fp32,
                                 params["input_scale"], params["weight_scale"],
                                 params["output_scale"], params["output_zero_point"])
    expected = int(conv_out[2, 1, 2])  # a different (oc,oy,ox) this time, for variety

    oc, oy, ox = 2, 1, 2
    patch = x_u8[:, oy:oy + 3, ox:ox + 3].flatten()
    kernel = w_int8[oc].flatten()
    x_vals = [BitVecVal(int(v), 8) for v in patch]
    w_vals = [BitVecVal(int(v), 8) for v in kernel]
    bias_val = BitVecVal(int(np.rint(bias_fp32[oc] / (params["input_scale"] * params["weight_scale"]))), 32)

    out_expr = codegen_kernel(x_vals, w_vals, bias_val,
                               consts["input_zero_point"], consts["multiplier"], consts["output_zero_point"])
    z3_value = simplify(out_expr).as_long()

    print(f"reference.py conv_out[{oc},{oy},{ox}] = {expected}")
    print(f"Z3 codegen_kernel(...)             = {z3_value}")
    assert z3_value == expected
    print("PASS: Z3's codegen_kernel encoding matches reference.py on a different example.")


if __name__ == "__main__":
    test_spec_kernel_matches_reference_py()
    print()
    test_codegen_kernel_matches_reference_py_too()
