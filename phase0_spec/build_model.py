"""
Phase 0 step 1-3: build Conv2D(1->4,3x3) -> ReLU -> MaxPool(2x2), quantize it
with static PTQ (qnnpack backend, per-tensor symmetric weights), and dump the
exact integer arithmetic (int8 weights, int32-equivalent bias, scales,
zero-points) to a JSON file for reference.py.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, QConfig
from torch.ao.quantization.observer import MinMaxObserver

torch.manual_seed(0)
torch.backends.quantized.engine = "qnnpack"

IN_H, IN_W = 8, 8          # small hand-sized input so int8 arrays are tiny
KSIZE = 3
IN_CH, OUT_CH = 1, 4


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(IN_CH, OUT_CH, KSIZE, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dequant(x)
        return x


def build_and_quantize():
    """
    Builds TinyNet, runs static PTQ (qnnpack, per-tensor symmetric qint8
    for both weights and activations), and returns (qmodel, params_dict).
    """
    model = TinyNet()
    model.eval()

    # Give the conv some non-trivial, reproducible weights/bias instead of
    # torch's default init, so the numbers in the writeup are stable.
    with torch.no_grad():
        model.conv.weight.copy_(
            torch.empty(OUT_CH, IN_CH, KSIZE, KSIZE).uniform_(-1.0, 1.0)
        )
        model.conv.bias.copy_(torch.empty(OUT_CH).uniform_(-0.5, 0.5))

    # --- Static PTQ config: per-tensor, qnnpack backend.
    # First attempt used qint8 (symmetric) for activations too, to match
    # the doc's "symmetric, per-tensor" ask literally -- but qnnpack's
    # actual compiled conv2d kernel rejects that at run time:
    #   RuntimeError: quantized::conv(qnnpack): Expected activation data
    #   type QUInt8 but got QInt8
    # i.e. the real backend hard-requires unsigned quint8 activations.
    # Phase 0's whole point is "use PyTorch's actual arithmetic as the
    # oracle," so we follow the kernel's real requirement rather than
    # force a scheme it doesn't support: quint8 (possibly-asymmetric)
    # activations, qint8 per-tensor SYMMETRIC weights (zero_point=0,
    # which qnnpack does require/guarantee for weights).
    act_observer = MinMaxObserver.with_args(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    )
    weight_observer = MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
    )
    qconfig = QConfig(activation=act_observer, weight=weight_observer)
    model.qconfig = qconfig

    torch.ao.quantization.prepare(model, inplace=True)

    # Calibrate with random data covering the input range we care about.
    calib_inputs = [torch.empty(1, IN_CH, IN_H, IN_W).uniform_(-2.0, 2.0) for _ in range(200)]
    with torch.no_grad():
        for x in calib_inputs:
            model(x)

    qmodel = torch.ao.quantization.convert(model, inplace=False)
    qmodel.eval()

    # --- Pull out the exact integer arithmetic PyTorch will run. ---
    qconv = qmodel.conv  # torch.ao.nn.quantized.Conv2d

    w_qtensor = qconv.weight()          # quantized int8 weight tensor
    w_scale = w_qtensor.q_scale()
    w_zp = w_qtensor.q_zero_point()
    assert w_zp == 0, "expected symmetric weight quantization (zero_point=0)"
    w_int8 = w_qtensor.int_repr().numpy().astype(np.int8)

    bias_fp32 = qconv.bias().detach().numpy().astype(np.float32)

    in_scale = float(qmodel.quant.scale[0])
    in_zp = int(qmodel.quant.zero_point[0])

    out_scale = float(qconv.scale)
    out_zp = int(qconv.zero_point)

    params = {
        "in_ch": IN_CH, "out_ch": OUT_CH, "ksize": KSIZE,
        "in_h": IN_H, "in_w": IN_W,
        "weight_int8": w_int8.tolist(),
        "weight_scale": w_scale,
        "weight_zero_point": w_zp,
        "bias_fp32": bias_fp32.tolist(),
        "input_scale": in_scale,
        "input_zero_point": in_zp,
        "output_scale": out_scale,
        "output_zero_point": out_zp,
    }

    return qmodel, params


def main():
    qmodel, params = build_and_quantize()

    # Writes to the shared data/ directory (sibling of every phaseN_*
    # folder) -- this is THE tool that produces data/qparams.json; every
    # other phase reads it from there via src/reference.py's and
    # src/ir.py's default path resolution.
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "qparams.json")
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"wrote {out_path}")

    print("input : scale=%.8f zero_point=%d (dtype=qint8)" % (params["input_scale"], params["input_zero_point"]))
    print("weight: scale=%.8f zero_point=%d" % (params["weight_scale"], params["weight_zero_point"]))
    print("conv out (post-requant): scale=%.8f zero_point=%d" % (params["output_scale"], params["output_zero_point"]))
    print("weight int8 shape:", np.array(params["weight_int8"]).shape)
    print("bias fp32:", params["bias_fp32"])

    # Sanity-inspect what ReLU/MaxPool do to scale/zero-point in this backend.
    print("\nqmodel structure:")
    print(qmodel)
    print("relu scale/zp:", getattr(qmodel.relu, "scale", None), getattr(qmodel.relu, "zero_point", None))
    print("pool scale/zp:", getattr(qmodel.pool, "scale", None), getattr(qmodel.pool, "zero_point", None))


if __name__ == "__main__":
    main()
