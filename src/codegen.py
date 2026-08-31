"""
Phase 3 -- Naive Codegen (C, no optimization).

Walks the IR using the memory plam and creates a single
model.h/model.c pair: a static arena, weights baked into .rodata, one
 triple-nested-loop function per op, and a requantize() that
matches reference.py's _requantize.
"""
import numpy as np

import ir
import planner as pl


def c_array_literal(arr: np.ndarray) -> str:
    """Recursively formats a numpy array as a nested C brace initializer."""
    if arr.ndim == 0:
        return str(int(arr))
    if arr.ndim == 1:
        return "{" + ", ".join(str(int(x)) for x in arr) + "}"
    return "{" + ", ".join(c_array_literal(sub) for sub in arr) + "}"


def c_dims(shape) -> str:
    return "".join(f"[{d}]" for d in shape)


# ---------------------------------------------------------------------------
# Per-op C emission
# ---------------------------------------------------------------------------

def emit_conv_op(op: ir.ConvOp, g: ir.Graph, idx: int) -> str:
    in_t, out_t = g.tensors[op.input], g.tensors[op.output]
    cin, in_h, in_w = in_t.shape
    cout, out_h, out_w = out_t.shape
    k = op.kernel_size
    multiplier = op.output_scale and (in_t.scale * g.weights[op.weight].scale / op.output_scale)

    return f"""\
static void op{idx}_conv2d(void) {{
    const uint8_t *in = &arena[OFFSET_{op.input}];
    uint8_t *out = &arena[OFFSET_{op.output}];
    for (int oc = 0; oc < {cout}; oc++) {{
        for (int oy = 0; oy < {out_h}; oy++) {{
            for (int ox = 0; ox < {out_w}; ox++) {{
                int32_t acc = {op.bias}[oc];
                for (int ic = 0; ic < {cin}; ic++) {{
                    for (int ky = 0; ky < {k}; ky++) {{
                        for (int kx = 0; kx < {k}; kx++) {{
                            int32_t x = (int32_t)in[ic*{in_h}*{in_w} + (oy+ky)*{in_w} + (ox+kx)]
                                        - {op.input_zero_point};
                            int32_t w = {op.weight}[oc][ic][ky][kx];
                            acc += x * w;
                        }}
                    }}
                }}
                out[oc*{out_h}*{out_w} + oy*{out_w} + ox] =
                    requantize(acc, {multiplier!r}, {op.output_zero_point});
            }}
        }}
    }}
}}"""


def emit_conv_op_optimized(op: ir.ConvOp, g: ir.Graph, idx: int) -> str:
    """Phase 5:
    1. Packed loads for weights, Each output channel's
       K=cin*k*k weight bytes are read as ceil(K/4) uint32_t words
       instead of K individual signed-byte loads, then unpacked
       with shift-based sign extension (M0+ has no SXTB16).
    2. That unpacking happens ONCE PER OUTPUT CHANNEL, hoisted out of the
       oy/ox loop entirely, confirmed by disassembly that plain -O2
       does not do this on its own: every one of the 4*6*6*9=1296 MACs
       in the naive version re-reads its weight byte from memory, even
       though each channel only has 9 weight values reused
       across all 36 output pixels.
    """
    in_t, out_t = g.tensors[op.input], g.tensors[op.output]
    cin, in_h, in_w = in_t.shape
    cout, out_h, out_w = out_t.shape
    k = op.kernel_size
    K = cin * k * k
    multiplier = in_t.scale * g.weights[op.weight].scale / op.output_scale
    K_padded = ((K + 3) // 4) * 4
    num_words = K_padded // 4

    lines = [f"static void op{idx}_conv2d(void) {{"]
    lines.append(f"    const uint8_t *in = &arena[OFFSET_{op.input}];")
    lines.append(f"    uint8_t *out = &arena[OFFSET_{op.output}];")
    lines.append(f"    const uint32_t *wpacked = (const uint32_t *){op.weight}_packed;")
    lines.append(f"    for (int oc = 0; oc < {cout}; oc++) {{")

    # --- unpack this channel's K weight bytes, once, via packed word reads ---
    lines.append(f"        /* unpack {K} weight bytes ({num_words} packed uint32_t reads) once per channel,")
    lines.append(f"           manual sign-extension via shift trick (M0+ has no SXTB16) */")
    lines.append(f"        int32_t w[{K}];")
    for word_idx in range(num_words):
        lines.append(f"        {{")
        lines.append(f"            uint32_t packed = wpacked[oc*{num_words} + {word_idx}];")
        for byte_in_word in range(4):
            flat = word_idx * 4 + byte_in_word
            if flat >= K:
                break
            shift = 24 - 8 * byte_in_word
            lines.append(f"            w[{flat}] = ((int32_t)(packed << {shift})) >> 24;")
        lines.append(f"        }}")

    lines.append(f"        for (int oy = 0; oy < {out_h}; oy++) {{")
    lines.append(f"            for (int ox = 0; ox < {out_w}; ox++) {{")
    lines.append(f"                int32_t acc = {op.bias}[oc];")
    lines.append(f"                /* fully unrolled K={K} MAC -- no loop counter, no branch */")
    flat = 0
    for ic in range(cin):
        for ky in range(k):
            for kx in range(k):
                lines.append(
                    f"                acc += ((int32_t)in[{ic}*{in_h}*{in_w} + (oy+{ky})*{in_w} + (ox+{kx})]"
                    f" - {op.input_zero_point}) * w[{flat}];"
                )
                flat += 1
    lines.append(f"                out[oc*{out_h}*{out_w} + oy*{out_w} + ox] =")
    lines.append(f"                    requantize(acc, {multiplier!r}, {op.output_zero_point});")
    lines.append(f"            }}")
    lines.append(f"        }}")
    lines.append(f"    }}")
    lines.append(f"}}")
    return "\n".join(lines)


def emit_packed_weight_array(op: ir.ConvOp, g: ir.Graph) -> str:
    """Emits `{weight}_packed`: the same weight bytes as `{weight}`, but
    flattened per-channel and zero-padded to a multiple of 4 bytes so
    they can be read as aligned uint32_t words. `__attribute__((aligned(4)))`
    makes that alignment a guarantee, not a hope."""
    w = g.weights[op.weight]
    cout = w.array.shape[0]
    K = int(np.prod(w.array.shape[1:]))
    K_padded = ((K + 3) // 4) * 4
    flat = w.array.reshape(cout, K)
    padded = np.zeros((cout, K_padded), dtype=np.uint8)
    padded[:, :K] = flat.astype(np.uint8)  # reinterpret int8 bit pattern as uint8 for storage
    rows = ", ".join(c_array_literal(row) for row in padded)
    return (f"static const uint8_t {op.weight}_packed[{cout}][{K_padded}] "
            f"__attribute__((aligned(4))) = {{{rows}}};")


def emit_relu_op(op: ir.ReluOp, g: ir.Graph, idx: int) -> str:
    in_t = g.tensors[op.input]
    n = in_t.num_elements
    # Deliberately doesn't assume in-place: reads OFFSET_{input}, writes
    # OFFSET_{output}. If Phase 2 aliased them, those offsets are equal
    # and this is in-place "for free" -- if a future graph didn't alias
    # this op, the same code is still correct, since ReLU is elementwise
    # (out[i] depends only on in[i], no cross-element hazard either way).
    return f"""\
static void op{idx}_relu(void) {{
    const uint8_t *in = &arena[OFFSET_{op.input}];
    uint8_t *out = &arena[OFFSET_{op.output}];
    for (int i = 0; i < {n}; i++) {{
        uint8_t v = in[i];
        out[i] = v < {op.zero_point} ? {op.zero_point} : v;
    }}
}}"""


def emit_maxpool_op(op: ir.MaxPoolOp, g: ir.Graph, idx: int) -> str:
    in_t, out_t = g.tensors[op.input], g.tensors[op.output]
    c, in_h, in_w = in_t.shape
    _, out_h, out_w = out_t.shape
    p, s = op.pool_size, op.stride
    return f"""\
static void op{idx}_maxpool2d(void) {{
    const uint8_t *in = &arena[OFFSET_{op.input}];
    uint8_t *out = &arena[OFFSET_{op.output}];
    for (int c = 0; c < {c}; c++) {{
        for (int oy = 0; oy < {out_h}; oy++) {{
            for (int ox = 0; ox < {out_w}; ox++) {{
                uint8_t m = in[c*{in_h}*{in_w} + (oy*{s})*{in_w} + (ox*{s})];
                for (int py = 0; py < {p}; py++) {{
                    for (int px = 0; px < {p}; px++) {{
                        uint8_t v = in[c*{in_h}*{in_w} + (oy*{s}+py)*{in_w} + (ox*{s}+px)];
                        if (v > m) m = v;
                    }}
                }}
                out[c*{out_h}*{out_w} + oy*{out_w} + ox] = m;
            }}
        }}
    }}
}}"""


EMITTERS = {
    ir.ConvOp: emit_conv_op,
    ir.ReluOp: emit_relu_op,
    ir.MaxPoolOp: emit_maxpool_op,
}

OP_CALL_NAME = {
    ir.ConvOp: "conv2d",
    ir.ReluOp: "relu",
    ir.MaxPoolOp: "maxpool2d",
}


# ---------------------------------------------------------------------------
# Top-level codegen
# ---------------------------------------------------------------------------

def generate(g: ir.Graph, result: pl.PlanResult, optimize: bool = False, func_name: str = "model_run",
             header_filename: str = None) -> tuple[str, str]:
    """Returns (model_h_text, model_c_text). optimize=True swaps in the
    Phase 5 conv2d (packed weight loads, hoisted-per-channel unpacking,
    fully unrolled K-loop) instead of the naive Phase 3 version.
    func_name lets two variants be compiled into the same binary without a
    symbol clash."""
    input_name = g.input_names[0]
    output_name = g.output_names[0]
    input_bytes = g.tensors[input_name].size_bytes
    output_bytes = g.tensors[output_name].size_bytes

    # ---- model.h ----
    header = f"""\
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

#define MODEL_ARENA_BYTES {result.total_bytes}
#define MODEL_INPUT_BYTES {input_bytes}
#define MODEL_OUTPUT_BYTES {output_bytes}

/* input: {g.tensors[input_name].shape} uint8, scale={g.tensors[input_name].scale}, zero_point={g.tensors[input_name].zero_point}
 * output: {g.tensors[output_name].shape} uint8, scale={g.tensors[output_name].scale}, zero_point={g.tensors[output_name].zero_point}
 * Generated by codegen.py ({'OPTIMIZED' if optimize else 'naive'} conv2d) from the Phase 0
 * quantized model (qparams.json) and the Phase 2 memory plan. Do not
 * edit by hand -- edit the graph / planner and regenerate. */
void {func_name}(const uint8_t *input, uint8_t *output);

#endif /* MODEL_H */
"""

    # ---- model.c ----
    # header_filename must match whatever the caller actually writes the
    # header out as (main() below picks model.h vs model_opt.h) -- this
    # was previously hardcoded to "model.h" unconditionally, which only
    # "worked" for the optimized variant by coincidence (both headers
    # happen to define identical macro values), not by correct design.
    inc_name = header_filename or ("model_opt.h" if optimize else "model.h")
    lines = []
    lines.append(f'#include "{inc_name}"')
    lines.append("#include <math.h>")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append(f"static uint8_t arena[MODEL_ARENA_BYTES];")
    lines.append("")

    # Arena offsets, straight from Phase 2's memory map.
    lines.append("/* Byte offsets into arena[], from Phase 2's memory planner. */")
    for name in g.tensors:
        lines.append(f"#define OFFSET_{name} {result.memory_map[name]}")
    lines.append("")

    # Weights, baked into flash (.rodata -- these are `static const`).
    # In optimized mode the plain per-element weight array for any conv op
    # is never referenced (only its packed version below is), so skip
    # emitting it -- avoids an unused-variable warning on an otherwise
    # clean build.
    packed_weight_names = {op.weight for op in g.ops if isinstance(op, ir.ConvOp)} if optimize else set()
    lines.append("/* Weights & biases, baked in as flash constants. */")
    for name, w in g.weights.items():
        if name in packed_weight_names:
            continue
        c_type = {"int8": "int8_t", "int32": "int32_t"}[str(w.array.dtype)]
        lines.append(f"static const {c_type} {name}{c_dims(w.array.shape)} = "
                      f"{c_array_literal(w.array)};")
    if optimize:
        lines.append("/* Phase 5: same weight bytes, padded to a multiple of 4 per channel and")
        lines.append(" * 4-byte aligned, so they can be read as packed uint32_t words. */")
        for op in g.ops:
            if isinstance(op, ir.ConvOp):
                lines.append(emit_packed_weight_array(op, g))
    lines.append("")

    # Requantize -- must match reference.py's _requantize exactly: a
    # float64 multiply, then round-HALF-TO-EVEN via nearbyint() under the
    # default FE_TONEAREST rounding mode (matches numpy's np.rint), then
    # clamp to uint8. See phase0/reference.py's _requantize docstring and
    # test_tie_rounding.py for why round-half-to-even was chosen and how
    # far that choice was actually verified.
    lines.append("""\
static uint8_t requantize(int32_t acc, double multiplier, int32_t zero_point) {
    double scaled = (double)acc * multiplier;
    long q = (long)nearbyint(scaled) + zero_point;
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return (uint8_t)q;
}
""")

    # One function per op.
    conv_emitter = emit_conv_op_optimized if optimize else emit_conv_op
    emitters = dict(EMITTERS)
    emitters[ir.ConvOp] = conv_emitter
    for idx, op in enumerate(g.ops):
        emitter = emitters[type(op)]
        lines.append(emitter(op, g, idx))
        lines.append("")

    # model_run: copy input into the arena, run every op in order, copy
    # the final output back out.
    lines.append(f"void {func_name}(const uint8_t *input, uint8_t *output) {{")
    lines.append(f"    memcpy(&arena[OFFSET_{input_name}], input, MODEL_INPUT_BYTES);")
    for idx, op in enumerate(g.ops):
        call = OP_CALL_NAME[type(op)]
        lines.append(f"    op{idx}_{call}();")
    lines.append(f"    memcpy(output, &arena[OFFSET_{output_name}], MODEL_OUTPUT_BYTES);")
    lines.append("}")
    lines.append("")

    return header, "\n".join(lines)


def main():
    import sys
    optimize = "--optimize" in sys.argv
    func_name = "model_run"
    for arg in sys.argv:
        if arg.startswith("--func-name="):
            func_name = arg.split("=", 1)[1]
    g = ir.build_graph()
    result = pl.plan(g)
    header, source = generate(g, result, optimize=optimize, func_name=func_name)
    out_c = "model_opt.c" if optimize else "model.c"
    out_h = "model_opt.h" if optimize else "model.h"
    with open(out_h, "w") as f:
        f.write(header)
    with open(out_c, "w") as f:
        f.write(source)
    print(f"wrote {out_h}, {out_c} (func_name={func_name})")
    print(f"arena: {result.total_bytes} bytes, input: {g.tensors[g.input_names[0]].size_bytes} bytes, "
          f"output: {g.tensors[g.output_names[0]].size_bytes} bytes")


if __name__ == "__main__":
    main()
