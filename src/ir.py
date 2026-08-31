"""
Phase 1 — IR and Graph Representation.

A minimal SSA-ish IR:
Two kinds of tensors:
  - "activation" tensors (TensorInfo): the input, and everything an op
    produces. These live in the dynamic SRAM arena.
  - "constant" tensors (WeightInfo): conv weights and biases. These are
    baked into flash (.rodata) at codegen time and are not allocated a
    live SRAM range.

The op set matches exactly what Phase 0 validated bit-for-bit against
the quantized PyTorch model: quint8 activations (real zero-points),
int8 symmetric weights, int32 bias already expressed in the
zero-point-subtracted fixed-point domain (bias_scale = in_scale *
w_scale), and ReLU/MaxPool as scale/zero-point-preserving passthroughs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Same default-path resolution as reference.py -- see its comment for why.
_DEFAULT_QPARAMS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "qparams.json")
)


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------

@dataclass
class TensorInfo:
    """An activation tensor: something an op produces and a later op
    consumes. Lives in the dynamic SRAM arena."""
    name: str
    shape: Tuple[int, ...]          # (C, H, W) -- no batch dim, MVP is single-image
    dtype: str                      # 'uint8' | 'int32' (int32 only ever transient, pre-requant)
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    is_graph_input: bool = False
    is_graph_output: bool = False

    # Filled in by compute_live_ranges(), not at construction time.
    first_def: Optional[int] = None   # op index that produces this tensor (-1 == graph input, available before op 0)
    last_use: Optional[int] = None    # last op index that consumes this tensor

    @property
    def dtype_size(self) -> int:
        return {"uint8": 1, "int8": 1, "int32": 4}[self.dtype]

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> int:
        return self.num_elements * self.dtype_size


@dataclass
class WeightInfo:
    """A constant tensor (conv weight or bias). No live range, it lives in
    flash for the whole program."""
    name: str
    array: np.ndarray               # int8 weight or int32 bias
    scale: Optional[float] = None   # None for bias (bias is already int32 in the fixed-point domain, no separate scale needed downstream)
    zero_point: int = 0             # weights/bias are always symmetric (zp=0) in this scheme

    @property
    def dtype(self) -> str:
        return str(self.array.dtype)

    @property
    def size_bytes(self) -> int:
        return int(self.array.nbytes)


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

@dataclass
class ConvOp:
    input: str
    output: str
    weight: str                 # name of a WeightInfo (int8, shape (Cout,Cin,K,K))
    bias: str                   # name of a WeightInfo (int32, shape (Cout,))
    stride: int
    kernel_size: int
    input_zero_point: int       # subtracted from input before the dot product (Phase 0 finding: real, non-zero)
    output_scale: float         # requantization target scale
    output_zero_point: int

    def input_names(self) -> List[str]:
        return [self.input]

    def output_names(self) -> List[str]:
        return [self.output]

    def op_name(self) -> str:
        return "Conv2d"


@dataclass
class ReluOp:
    input: str
    output: str
    zero_point: int             # clamp floor is zero_point, not 0 (Phase 0 finding)

    def input_names(self) -> List[str]:
        return [self.input]

    def output_names(self) -> List[str]:
        return [self.output]

    def op_name(self) -> str:
        return "ReLU"


@dataclass
class MaxPoolOp:
    input: str
    output: str
    pool_size: int
    stride: int

    def input_names(self) -> List[str]:
        return [self.input]

    def output_names(self) -> List[str]:
        return [self.output]

    def op_name(self) -> str:
        return "MaxPool2d"


Op = ConvOp | ReluOp | MaxPoolOp


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)
    weights: Dict[str, WeightInfo] = field(default_factory=dict)
    ops: List[Op] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    # -- construction helpers --------------------------------------------

    def add_input(self, t: TensorInfo) -> None:
        t.is_graph_input = True
        t.first_def = -1
        self.tensors[t.name] = t
        self.input_names.append(t.name)

    def add_output(self, name: str) -> None:
        self.tensors[name].is_graph_output = True
        self.output_names.append(name)

    def add_weight(self, w: WeightInfo) -> None:
        self.weights[w.name] = w

    def add_op(self, op: Op, output_tensor: TensorInfo) -> int:
        """Appends op, registers its output tensor, returns the op's index
        (its position in self.ops -- this IS the SSA "instruction number"
        live ranges are measured in)."""
        for name in op.input_names():
            if name not in self.tensors:
                raise ValueError(f"op {op.op_name()} references undefined tensor '{name}'")
        idx = len(self.ops)
        self.ops.append(op)
        output_tensor.first_def = idx
        self.tensors[output_tensor.name] = output_tensor
        return idx

    # -- validation ---------------------------------------------------

    def validate(self) -> None:
        """Cheap sanity checks: every op's declared output shape/dtype is
        internally consistent, and every tensor is produced exactly once
        (true SSA property) except graph inputs."""
        produced = set(self.input_names)
        for i, op in enumerate(self.ops):
            for name in op.input_names():
                if name not in produced:
                    raise ValueError(f"op {i} ({op.op_name()}) uses '{name}' before it's produced")
            for name in op.output_names():
                if name in produced:
                    raise ValueError(f"tensor '{name}' produced more than once (breaks SSA) at op {i}")
                produced.add(name)
        for name in self.output_names:
            if name not in produced:
                raise ValueError(f"declared graph output '{name}' is never produced")


# ---------------------------------------------------------------------------
# Live-range computation
# ---------------------------------------------------------------------------

def compute_live_ranges(g: Graph) -> None:
    """Fills in first_def/last_use on every TensorInfo in g.tensors
    first_def: -1 for graph inputs (available before op 0), else the
        index of the op that produces it (set already by add_op).
    last_use: the index of the last op that *consumes* it, OR, for a
        graph output, pinned to stay alive through the final op,
        represented here as the index of the last op in the graph.
    """
    last_op_idx = len(g.ops) - 1

    for t in g.tensors.values():
        t.last_use = t.first_def if t.first_def is not None and t.first_def >= 0 else -1

    for i, op in enumerate(g.ops):
        for name in op.input_names():
            t = g.tensors[name]
            t.last_use = max(t.last_use if t.last_use is not None else -1, i)

    for name in g.output_names:
        t = g.tensors[name]
        t.last_use = max(t.last_use, last_op_idx)


def print_live_range_table(g: Graph) -> None:
    print("Ops:")
    for i, op in enumerate(g.ops):
        ins = ", ".join(op.input_names())
        print(f"  [{i}] {op.op_name():10s} {ins:>10s} -> {op.output_names()[0]}")

    print("\nActivation tensors (live in the dynamic SRAM arena):")
    header = f"  {'name':10s} {'shape':12s} {'dtype':6s} {'bytes':>7s} {'scale':>10s} {'zp':>5s} {'first_def':>9s} {'last_use':>8s} {'flags'}"
    print(header)
    for name, t in g.tensors.items():
        flags = []
        if t.is_graph_input:
            flags.append("INPUT")
        if t.is_graph_output:
            flags.append("OUTPUT")
        scale_str = f"{t.scale:.6f}" if t.scale is not None else "-"
        print(f"  {t.name:10s} {str(t.shape):12s} {t.dtype:6s} {t.size_bytes:7d} "
              f"{scale_str:>10s} {t.zero_point!s:>5s} {t.first_def!s:>9s} {t.last_use!s:>8s} "
              f"{','.join(flags)}")

    print("\nConstant tensors (flash, no live range):")
    print(f"  {'name':10s} {'shape':16s} {'dtype':6s} {'bytes':>7s}")
    for name, w in g.weights.items():
        print(f"  {w.name:10s} {str(w.array.shape):16s} {w.dtype:6s} {w.size_bytes:7d}")

    total_flash = sum(w.size_bytes for w in g.weights.values())
    print(f"\nTotal constant (flash) bytes: {total_flash}")


# ---------------------------------------------------------------------------
# Build the concrete Phase 0 model graph from qparams.json
# ---------------------------------------------------------------------------

def build_graph(qparams_path: str = None) -> Graph:
    if qparams_path is None:
        qparams_path = _DEFAULT_QPARAMS if os.path.exists(_DEFAULT_QPARAMS) else "qparams.json"
    with open(qparams_path) as f:
        p = json.load(f)

    in_ch, out_ch, k = p["in_ch"], p["out_ch"], p["ksize"]
    in_h, in_w = p["in_h"], p["in_w"]
    conv_h, conv_w = in_h - k + 1, in_w - k + 1   # stride 1, no padding
    pool_h, pool_w = conv_h // 2, conv_w // 2

    g = Graph()

    g.add_input(TensorInfo(
        name="input", shape=(in_ch, in_h, in_w), dtype="uint8",
        scale=p["input_scale"], zero_point=p["input_zero_point"],
    ))

    g.add_weight(WeightInfo(
        name="conv1_weight",
        array=np.array(p["weight_int8"], dtype=np.int8),
        scale=p["weight_scale"], zero_point=p["weight_zero_point"],
    ))
    bias_scale = p["input_scale"] * p["weight_scale"]
    bias_int32 = np.rint(np.array(p["bias_fp32"], dtype=np.float64) / bias_scale).astype(np.int32)
    g.add_weight(WeightInfo(name="conv1_bias", array=bias_int32, scale=None, zero_point=0))

    g.add_op(
        ConvOp(
            input="input", output="conv_out", weight="conv1_weight", bias="conv1_bias",
            stride=1, kernel_size=k,
            input_zero_point=p["input_zero_point"],
            output_scale=p["output_scale"], output_zero_point=p["output_zero_point"],
        ),
        TensorInfo(name="conv_out", shape=(out_ch, conv_h, conv_w), dtype="uint8",
                   scale=p["output_scale"], zero_point=p["output_zero_point"]),
    )

    g.add_op(
        ReluOp(input="conv_out", output="relu_out", zero_point=p["output_zero_point"]),
        TensorInfo(name="relu_out", shape=(out_ch, conv_h, conv_w), dtype="uint8",
                   scale=p["output_scale"], zero_point=p["output_zero_point"]),
    )

    g.add_op(
        MaxPoolOp(input="relu_out", output="pool_out", pool_size=2, stride=2),
        TensorInfo(name="pool_out", shape=(out_ch, pool_h, pool_w), dtype="uint8",
                   scale=p["output_scale"], zero_point=p["output_zero_point"]),
    )

    g.add_output("pool_out")

    g.validate()
    compute_live_ranges(g)
    return g


def main():
    g = build_graph()
    print_live_range_table(g)

    # A couple of sanity assertions worth stating out loud rather than
    # leaving implicit: overlap detection here is the exact thing Phase 2
    # will need to get right at scale.
    assert g.tensors["input"].first_def == -1
    assert g.tensors["input"].last_use == 0          # only consumed by op 0 (Conv)
    assert g.tensors["conv_out"].first_def == 0
    assert g.tensors["conv_out"].last_use == 1        # produced by op0, consumed by op1 (ReLU)
    assert g.tensors["relu_out"].last_use == 2         # consumed by op2 (MaxPool)
    assert g.tensors["pool_out"].is_graph_output
    print("\nSanity checks passed: live ranges match the expected straight-line chain.")


if __name__ == "__main__":
    main()
