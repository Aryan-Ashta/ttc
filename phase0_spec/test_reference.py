"""
Phase 0 deliverable: test script comparing reference.py's output against
the REAL quantized PyTorch model's output on N random inputs. Must match
bit-for-bit (exact int8 arrays) or Phase 0 is not done -- everything
downstream (C codegen, Z3 proof) assumes this spec is exactly right.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from build_model import build_and_quantize, IN_CH, IN_H, IN_W
import reference as ref

torch.backends.quantized.engine = "qnnpack"


def oracle_forward(qmodel, x_float):
    """Run the real quantized model up to (but not including) dequant, and
    return the raw uint8 output array -- this is ground truth."""
    with torch.no_grad():
        q = qmodel.quant(x_float)
        q = qmodel.conv(q)
        q = qmodel.relu(q)
        q = qmodel.pool(q)
    return q.int_repr().numpy().astype(np.uint8)[0]  # drop batch dim -> (C,H,W)


def main(n=500, seed=1234):
    qmodel, params = build_and_quantize()
    qmodel.eval()

    rng = np.random.default_rng(seed)
    mismatches = []

    for i in range(n):
        x_float = torch.from_numpy(
            rng.uniform(-2.0, 2.0, size=(1, IN_CH, IN_H, IN_W)).astype(np.float32)
        )

        # Ground truth int8 output straight from the real quantized model.
        oracle_out = oracle_forward(qmodel, x_float)

        # Our uint8 input, derived the identical way torch's QuantStub does
        # it (so both pipelines start from the exact same uint8 numbers).
        with torch.no_grad():
            x_q = qmodel.quant(x_float)
        x_u8 = x_q.int_repr().numpy().astype(np.uint8)[0]

        our_out = ref.run_reference(x_u8, params)

        if not np.array_equal(our_out, oracle_out):
            mismatches.append((i, x_u8, our_out, oracle_out))

    print(f"Tested {n} random inputs.")
    if not mismatches:
        print("PASS: reference.py matches the real quantized model bit-for-bit on all inputs.")
        return 0

    print(f"FAIL: {len(mismatches)}/{n} mismatches. First mismatch:")
    idx, x_u8, our_out, oracle_out = mismatches[0]
    print("input uint8:\n", x_u8)
    print("our output:\n", our_out)
    print("oracle output:\n", oracle_out)
    print("diff:\n", our_out.astype(int) - oracle_out.astype(int))
    return 1


if __name__ == "__main__":
    sys.exit(main())
