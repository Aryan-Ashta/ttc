"""
Phase 0:

Pure numpy implementation of the quantized Conv2D(1->4,3x3)
-> ReLU -> MaxPool(2x2) pipeline that build_model.py
produces via torch's static PTQ.
"""
import json
import os
import numpy as np

# Resolve qparams.json relative to this file's location (src/../data/) by
# default, so it's found regardless of the caller's working directory --
# falls back to a plain cwd-relative "qparams.json" if that's what's there
# (keeps explicit-path callers and any old invocation style working too).
_DEFAULT_QPARAMS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "qparams.json")
)


def load_params(path=None):
    if path is None:
        path = _DEFAULT_QPARAMS if os.path.exists(_DEFAULT_QPARAMS) else "qparams.json"
    with open(path) as f:
        return json.load(f)


def quantize_input(x_float, input_scale, input_zero_point):
    """float32 -> uint8, matching torch's QuantStub (round-half-to-even,
    clamp to [0,255])."""
    q = np.rint(x_float / input_scale).astype(np.int64) + input_zero_point
    return np.clip(q, 0, 255).astype(np.uint8)


def _requantize(acc_int32, multiplier, zero_point):
    """int32 accumulator -> uint8 output."""
    scaled = acc_int32.astype(np.float64) * multiplier
    q = np.rint(scaled).astype(np.int64) + zero_point
    return np.clip(q, 0, 255).astype(np.uint8)


def conv2d_quant(x_u8, x_zp, w_int8, bias_fp32, input_scale, weight_scale,
                  output_scale, output_zp):
    """
    x_u8: (Cin, H, W) uint8, x_zp: python int, the input's zero_point.
    w_int8: (Cout, Cin, K, K) int8, symmetric (zero_point == 0).
    bias_fp32: (Cout,) float32 -- the *unquantized* bias PyTorch stores;
        quantized internally with scale = input_scale * weight_scale
        (zero_point=0), the same fixed-point domain as the zero-point-
        subtracted int32 accumulator, so it can be added before
        requantization.
    Returns: (Cout, H-K+1, W-K+1) uint8, the requantized conv output.
    """
    cin, h, w = x_u8.shape
    cout, cin_w, k, k2 = w_int8.shape
    assert cin == cin_w and k == k2
    out_h, out_w = h - k + 1, w - k + 1

    # Subtract the zero-point BEFORE the dot product -- this is the one
    # piece of arithmetic that symmetric-everywhere quantization would
    # have avoided, and the reason quint8 activations are more work than
    # qint8 ones would have been.
    x32 = x_u8.astype(np.int32) - x_zp
    w32 = w_int8.astype(np.int32)  # weight zero_point is 0, nothing to subtract

    bias_scale = input_scale * weight_scale
    bias_int32 = np.rint(np.asarray(bias_fp32, dtype=np.float64) / bias_scale).astype(np.int64)

    acc = np.zeros((cout, out_h, out_w), dtype=np.int64)
    for oc in range(cout):
        for oy in range(out_h):
            for ox in range(out_w):
                patch = x32[:, oy:oy + k, ox:ox + k]
                acc[oc, oy, ox] = int(np.sum(patch * w32[oc])) + bias_int32[oc]

    multiplier = bias_scale / output_scale  # == (input_scale*weight_scale)/output_scale
    return _requantize(acc.astype(np.int32), multiplier, output_zp)


def relu_quant(x_u8, zero_point):
    """Quantized ReLU: clamp at the quantized representation of real 0.0,
    which is `zero_point` (not 0, now that activations are quint8)."""
    return np.maximum(x_u8.astype(np.int32), zero_point).astype(np.uint8)


def maxpool2d_quant(x_u8, pool=2, stride=2):
    """Quantized MaxPool2d: max is monotonic under the (scale, zero_point)
    affine map (scale > 0), so it's computed directly in the uint8 domain."""
    c, h, w = x_u8.shape
    out_h, out_w = h // pool, w // pool
    out = np.empty((c, out_h, out_w), dtype=np.uint8)
    for oy in range(out_h):
        for ox in range(out_w):
            window = x_u8[:, oy * stride:oy * stride + pool, ox * stride:ox * stride + pool]
            out[:, oy, ox] = window.reshape(c, -1).max(axis=1)
    return out


def run_reference(x_u8, params=None):
    """uint8 (Cin,H,W) input -> uint8 (Cout,H',W') output."""
    if params is None:
        params = load_params()

    w_int8 = np.array(params["weight_int8"], dtype=np.int8)
    bias_fp32 = np.array(params["bias_fp32"], dtype=np.float32)
    input_scale = params["input_scale"]
    input_zp = params["input_zero_point"]
    weight_scale = params["weight_scale"]
    output_scale = params["output_scale"]
    output_zp = params["output_zero_point"]

    assert params["weight_zero_point"] == 0, "weights must be symmetric"

    conv_out = conv2d_quant(x_u8, input_zp, w_int8, bias_fp32,
                             input_scale, weight_scale, output_scale, output_zp)
    relu_out = relu_quant(conv_out, zero_point=output_zp)
    pool_out = maxpool2d_quant(relu_out, pool=2, stride=2)
    return pool_out
