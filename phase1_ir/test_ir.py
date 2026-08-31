"""
Phase 1 tests:
  1. The IR's declared tensor shapes/dtypes matches what
     reference.py computes at runtime
  2. Graph.validate() catches broken graphs
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ir
import reference as ref


def test_ir_shapes_match_reference_runtime_shapes():
    g = ir.build_graph()
    params = ref.load_params()

    x_u8 = np.random.default_rng(0).integers(0, 256, size=g.tensors["input"].shape, dtype=np.uint8)
    out = ref.run_reference(x_u8, params)

    assert out.shape == g.tensors["pool_out"].shape, (
        f"IR says pool_out shape is {g.tensors['pool_out'].shape}, "
        f"reference.py produced {out.shape}"
    )
    assert out.dtype == np.uint8
    assert g.tensors["pool_out"].dtype == "uint8"
    print("PASS: IR-declared shapes match reference.py's runtime output shape.")


def test_ir_scales_match_qparams():
    g = ir.build_graph()
    params = ref.load_params()
    assert g.tensors["input"].scale == params["input_scale"]
    assert g.tensors["input"].zero_point == params["input_zero_point"]
    assert g.tensors["pool_out"].scale == params["output_scale"]
    assert g.tensors["pool_out"].zero_point == params["output_zero_point"]
    print("PASS: IR-declared scales match qparams.json.")


def test_validate_catches_use_before_def():
    g = ir.Graph()
    g.add_input(ir.TensorInfo(name="x", shape=(1, 4, 4), dtype="uint8"))
    # Reference a tensor ("ghost") that was never produced or declared input.
    bad_op = ir.ReluOp(input="ghost", output="y", zero_point=0)
    try:
        g.add_op(bad_op, ir.TensorInfo(name="y", shape=(1, 4, 4), dtype="uint8"))
        raised = False
    except ValueError:
        raised = True
    assert raised, "add_op should have rejected a reference to an undefined tensor"
    print("PASS: use-before-def is rejected.")


def test_validate_catches_double_definition():
    g = ir.Graph()
    g.add_input(ir.TensorInfo(name="x", shape=(1, 4, 4), dtype="uint8"))
    g.add_op(ir.ReluOp(input="x", output="y", zero_point=0),
              ir.TensorInfo(name="y", shape=(1, 4, 4), dtype="uint8"))
    # Second op ALSO claims to produce "y" -- breaks the SSA single-def property.
    g.add_op(ir.ReluOp(input="x", output="y", zero_point=0),
              ir.TensorInfo(name="y", shape=(1, 4, 4), dtype="uint8"))
    try:
        g.validate()
        raised = False
    except ValueError:
        raised = True
    assert raised, "validate() should have caught 'y' being produced twice"
    print("PASS: double-definition is rejected by validate().")


def test_live_ranges_straight_line():

    g = ir.build_graph()
    assert g.tensors["input"].first_def == -1 and g.tensors["input"].last_use == 0
    assert g.tensors["conv_out"].first_def == 0 and g.tensors["conv_out"].last_use == 1
    assert g.tensors["relu_out"].first_def == 1 and g.tensors["relu_out"].last_use == 2
    assert g.tensors["pool_out"].first_def == 2 and g.tensors["pool_out"].last_use == 2
    print("PASS: live ranges are exactly what the straight-line chain implies.")


if __name__ == "__main__":
    test_ir_shapes_match_reference_runtime_shapes()
    test_ir_scales_match_qparams()
    test_validate_catches_use_before_def()
    test_validate_catches_double_definition()
    test_live_ranges_straight_line()
    print("\nAll Phase 1 tests passed.")
