"""
Emits test_vectors.h: N random uint8 inputs and their expected uint8
outputs, computed by reference.py. test_model.c
includes this and checks model_run() against it byte-for-byte.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import reference as ref
import ir


def main(n=500, seed=42, path="test_vectors.h"):
    g = ir.build_graph()
    params = ref.load_params()
    in_shape = g.tensors[g.input_names[0]].shape
    out_shape = g.tensors[g.output_names[0]].shape

    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, 256, size=(n,) + in_shape, dtype=np.uint8)
    outputs = np.empty((n,) + out_shape, dtype=np.uint8)
    for i in range(n):
        outputs[i] = ref.run_reference(inputs[i], params)

    flat_in = inputs.reshape(n, -1)
    flat_out = outputs.reshape(n, -1)

    with open(path, "w") as f:
        f.write("#ifndef TEST_VECTORS_H\n#define TEST_VECTORS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_TEST_VECTORS {n}\n")
        f.write(f"#define TEST_INPUT_BYTES {flat_in.shape[1]}\n")
        f.write(f"#define TEST_OUTPUT_BYTES {flat_out.shape[1]}\n\n")

        f.write(f"static const uint8_t test_inputs[NUM_TEST_VECTORS][TEST_INPUT_BYTES] = {{\n")
        for row in flat_in:
            f.write("  {" + ", ".join(str(int(v)) for v in row) + "},\n")
        f.write("};\n\n")

        f.write(f"static const uint8_t test_expected[NUM_TEST_VECTORS][TEST_OUTPUT_BYTES] = {{\n")
        for row in flat_out:
            f.write("  {" + ", ".join(str(int(v)) for v in row) + "},\n")
        f.write("};\n\n")

        f.write("#endif /* TEST_VECTORS_H */\n")

    print(f"wrote {path}: {n} vectors, input={flat_in.shape[1]}B, output={flat_out.shape[1]}B")


if __name__ == "__main__":
    main()
