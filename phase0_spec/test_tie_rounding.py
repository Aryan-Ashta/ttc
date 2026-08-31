"""
Phase 0 rigor check: random inputs (test_reference.py) essentially never
hit an exact .5 tie in the requantization multiply, so they can't tell
round-half-to-even from round-half-away-from-zero apart. This script
constructs a real input that DOES produce an exact tie at one conv output
pixel (by zeroing out 8 of the 9 kernel taps' contribution and sweeping
the 9th tap's input value through all 256 uint8 values), feeds it through
the REAL quantized model to get ground truth for that pixel, and checks
which rounding rule the real qnnpack backend actually uses there.
"""
import numpy as np
import torch

from build_model import build_and_quantize, IN_CH, IN_H, IN_W, KSIZE, OUT_CH

torch.backends.quantized.engine = "qnnpack"


def find_tie(w_int8, bias_int32, multiplier, zp):
    """For each (out_channel, kernel_y, kernel_x), sweep one input tap over
    0..255 with all other taps pinned at the zero-point (contributing 0),
    and look for an input value where acc*multiplier lands on an exact
    .5 boundary. Returns the first (oc, ky, kx, v, acc, scaled) found."""
    for oc in range(OUT_CH):
        for ky in range(KSIZE):
            for kx in range(KSIZE):
                w = int(w_int8[oc, 0, ky, kx])
                if w == 0:
                    continue
                for v in range(256):
                    acc = w * (v - zp) + int(bias_int32[oc])
                    scaled = acc * multiplier
                    frac = scaled - np.floor(scaled)
                    if abs(frac - 0.5) < 1e-9:
                        return oc, ky, kx, v, acc, scaled
    return None


def main():
    qmodel, params = build_and_quantize()
    qmodel.eval()

    w_int8 = np.array(params["weight_int8"], dtype=np.int64)
    input_scale = params["input_scale"]
    input_zp = params["input_zero_point"]
    weight_scale = params["weight_scale"]
    output_scale = params["output_scale"]
    output_zp = params["output_zero_point"]
    bias_scale = input_scale * weight_scale
    bias_int32 = np.rint(np.array(params["bias_fp32"], dtype=np.float64) / bias_scale).astype(np.int64)
    multiplier = bias_scale / output_scale

    # Before even trying to construct a real input: is an exact .5 tie
    # reachable at all within this kernel's realistic accumulator range?
    # Bound: |acc| <= cin*k*k*127*127 + max|bias_int32| for K=9 (3x3, cin=1).
    cin, k = 1, KSIZE
    max_acc = cin * k * k * 127 * 127 + int(np.abs(bias_int32).max())
    acc_range = np.arange(-max_acc, max_acc + 1, dtype=np.int64)
    scaled_range = acc_range.astype(np.float64) * multiplier
    frac_range = scaled_range - np.floor(scaled_range)
    ties_in_domain = np.where(np.abs(frac_range - 0.5) < 1e-9)[0]
    print(f"multiplier = {multiplier!r}")
    print(f"realistic accumulator range for this kernel: [-{max_acc}, {max_acc}] "
          f"({len(acc_range)} integers)")
    print(f"exact .5 ties found in that ENTIRE range: {len(ties_in_domain)}")

    if len(ties_in_domain) == 0:
        print()
        print("=> No exact .5 tie is reachable by ANY input to this kernel. "
              "round-half-to-even and round-half-away-from-zero are therefore "
              "PROVABLY IDENTICAL for every input this model can ever see -- "
              "this particular ambiguity is moot for this specific model/scale "
              "combination (a generic multiplier's dyadic structure just doesn't "
              "line up with an integer accumulator to produce an exact tie in "
              "this range). Kept round-half-to-even in reference.py anyway "
              "since it's numpy's/IEEE-754's default and the most likely match "
              "for qnnpack's C++ if a future model (different scales) does "
              "reach a tie.")
        return

    found = find_tie(w_int8, bias_int32, multiplier, input_zp)
    if found is None:
        print("Ties exist in the accumulator domain in principle, but none were "
              "reachable by sweeping single taps -- would need a multi-tap "
              "search to construct a realizable example.")
        return

    oc, ky, kx, v, acc, scaled = found
    print(f"Found exact tie: out_channel={oc}, tap=({ky},{kx}), input_value={v}")
    print(f"  acc={acc}, acc*multiplier={scaled!r} (fractional part exactly .5)")

    round_even = int(np.rint(scaled)) + output_zp
    round_away = int(np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)) + output_zp
    round_even = int(np.clip(round_even, 0, 255))
    round_away = int(np.clip(round_away, 0, 255))
    print(f"  round-half-to-even predicts:      {round_even}")
    print(f"  round-half-away-from-zero predicts: {round_away}")

    # Build the real image: zero_point everywhere except the one tap that
    # lands under kernel position (ky,kx) for output pixel (0,0), which we
    # set to v.
    x_u8 = np.full((1, IN_CH, IN_H, IN_W), input_zp, dtype=np.uint8)
    x_u8[0, 0, ky, kx] = v
    x_float = torch.from_numpy(((x_u8.astype(np.float32) - input_zp) * input_scale))

    with torch.no_grad():
        q = qmodel.quant(x_float)
        assert q.int_repr().numpy().astype(np.uint8)[0, 0, ky, kx] == v, \
            "requantization of our constructed float didn't round-trip to v"
        conv_out = qmodel.conv(q)

    oracle_val = int(conv_out.int_repr().numpy()[0, oc, 0, 0])
    print(f"  REAL qnnpack output at that pixel: {oracle_val}")

    if oracle_val == round_even and oracle_val != round_away:
        print("=> CONFIRMED: qnnpack uses round-half-to-even (banker's rounding).")
    elif oracle_val == round_away and oracle_val != round_even:
        print("=> CONFIRMED: qnnpack uses round-half-away-from-zero.")
    elif oracle_val == round_even == round_away:
        print("=> Both rules predict the same value here (bad luck of clamping); "
              "inconclusive, would need another tie point.")
    else:
        print("=> Oracle matches NEITHER rule -- reference.py's requantize model is wrong "
              "in some other way and needs to be revisited.")


if __name__ == "__main__":
    main()
