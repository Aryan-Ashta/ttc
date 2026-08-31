"""
Phase 2 tests.
  1. Independent check that the produced memory map has no illegal
     overlap
  2. find_aliases() picks exactly the op it should
  3. Allocator unit tests on small SYNTHETIC graphs, shows the allocator generalizes
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ir
import planner as pl


def independent_overlap_check(g: ir.Graph, memory_map: dict) -> list:
    """Re-implemented from the live-range table directly to catch bugs"""
    aliases = pl.find_aliases(g)
    names = list(g.tensors.keys())
    violations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = g.tensors[names[i]], g.tensors[names[j]]
            root_a = pl.resolve_alias_root(a.name, aliases)
            root_b = pl.resolve_alias_root(b.name, aliases)
            if root_a == root_b:
                continue  # intentional aliasing, not a bug
            time_ok = not (a.first_def <= b.last_use and b.first_def <= a.last_use)
            oa, ob = memory_map[a.name], memory_map[b.name]
            space_ok = not (oa < ob + b.size_bytes and ob < oa + a.size_bytes)
            if not (time_ok or space_ok):
                violations.append((a.name, b.name))
    return violations


def test_real_model_has_no_overlap():
    g = ir.build_graph()
    result = pl.plan(g)
    violations = independent_overlap_check(g, result.memory_map)
    assert violations == [], f"illegal overlaps found: {violations}"
    print("PASS: real model's memory map has no illegal overlaps.")


def test_aliasing_picks_exactly_relu():
    g = ir.build_graph()
    aliases = pl.find_aliases(g)
    assert aliases == {"relu_out": "conv_out"}, aliases
    print("PASS: aliasing picked exactly {relu_out: conv_out}")


def test_arena_smaller_than_naive_sum():
    g = ir.build_graph()
    result = pl.plan(g)
    naive = sum(t.size_bytes for t in g.tensors.values())
    assert result.total_bytes < naive
    print(f"PASS: planned arena ({result.total_bytes}B) < naive sum ({naive}B).")


def _make_chain_graph(shapes_dtypes):
    """Builds a synthetic straight-line graph of ReLU ops (any op works
    for a generic allocator test) with the given [(shape, dtype), ...]
    per tensor, tensor0 -> tensor1 -> ... i.e. num_ops = len - 1."""
    g = ir.Graph()
    names = [f"t{i}" for i in range(len(shapes_dtypes))]
    shape0, dtype0 = shapes_dtypes[0]
    g.add_input(ir.TensorInfo(name=names[0], shape=shape0, dtype=dtype0))
    for i in range(1, len(shapes_dtypes)):
        shape, dtype = shapes_dtypes[i]
        g.add_op(ir.ReluOp(input=names[i - 1], output=names[i], zero_point=0),
                  ir.TensorInfo(name=names[i], shape=shape, dtype=dtype))
    g.add_output(names[-1])
    g.validate()
    ir.compute_live_ranges(g)
    return g


def test_synthetic_all_same_shape_aliases_everything():
    """A pure ReLU chain, all same shape/dtype: every op should alias,
    so the WHOLE thing should collapse into a single buffer."""
    g = _make_chain_graph([((4, 4), "uint8")] * 5)
    result = pl.plan(g)
    offsets = set(result.memory_map.values())
    assert len(offsets) == 1, f"expected one shared buffer, got offsets {result.memory_map}"
    assert result.total_bytes == 16  # 4*4*1 byte, single buffer, no duplication
    print("PASS: an all-same-shape ReLU chain collapses into a single 16-byte buffer.")


def test_synthetic_disjoint_lifetimes_reuse_space():
    """Two same-size, non-aliasable (different dtype so find_aliases
    skips them) tensors with NON-overlapping lifetimes should be able to
    share the same offset."""
    g = ir.Graph()
    g.add_input(ir.TensorInfo(name="a", shape=(4,), dtype="uint8"))
    g.add_op(ir.MaxPoolOp(input="a", output="b", pool_size=1, stride=1),
              ir.TensorInfo(name="b", shape=(4,), dtype="uint8"))
    g.add_op(ir.MaxPoolOp(input="b", output="c", pool_size=1, stride=1),
              ir.TensorInfo(name="c", shape=(4,), dtype="uint8"))
    g.add_output("c")
    g.validate()
    ir.compute_live_ranges(g)
    # a: [-1,0], b: [0,1], c: [1,1]. a and c never overlap in time (a ends
    # at op0, c starts at op1) -- expect them to share an offset.
    result = pl.plan(g)
    assert result.memory_map["a"] == result.memory_map["c"], (
        "expected disjoint-lifetime tensors 'a' and 'c' to share space to save SRAM"
    )
    print("PASS: disjoint-lifetime tensors reuse the same memory offset.")


def test_time_overlap_and_space_overlap_edge_cases():
    # Touching-but-not-overlapping space ranges: [0,64) and [64,128) don't overlap.
    assert pl.space_overlap(0, 64, 64, 64) is False
    # Any actual byte shared: overlap.
    assert pl.space_overlap(0, 64, 63, 64) is True
    # Touching op indices ARE considered overlapping in time (closed interval).
    a = pl.BufferGroup(root="x", members=["x"], size=1, start=0, end=1)
    b = pl.BufferGroup(root="y", members=["y"], size=1, start=1, end=2)
    assert pl.time_overlap(a, b) is True
    c = pl.BufferGroup(root="z", members=["z"], size=1, start=2, end=3)
    assert pl.time_overlap(a, c) is False
    print("PASS: time_overlap/space_overlap edge cases behave as documented.")


if __name__ == "__main__":
    test_real_model_has_no_overlap()
    test_aliasing_picks_exactly_relu()
    test_arena_smaller_than_naive_sum()
    test_synthetic_all_same_shape_aliases_everything()
    test_synthetic_disjoint_lifetimes_reuse_space()
    test_time_overlap_and_space_overlap_edge_cases()
    print("\nAll Phase 2 tests passed.")
