"""
Phase 2 diagram: a Gantt-style chart of the arena. X-axis = op index
(time), Y-axis = byte offset.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ir
import planner as pl

OP_LABELS = {0: "op0: Conv2d", 1: "op1: ReLU", 2: "op2: MaxPool2d"}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def render(g: ir.Graph, result: pl.PlanResult, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, grp in enumerate(sorted(result.groups, key=lambda x: x.offset)):
        color = COLORS[i % len(COLORS)]
        x0 = grp.start - 0.4
        width = (grp.end - grp.start) + 0.8
        ax.add_patch(mpatches.Rectangle(
            (x0, grp.offset), width, grp.size,
            facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85,
        ))
        ax.text(grp.start + (grp.end - grp.start) / 2, grp.offset + grp.size / 2,
                f"{grp.label()}\n{grp.size}B", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    num_ops = len(g.ops)
    for op_idx in range(num_ops):
        ax.axvline(op_idx - 0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(num_ops - 1 + 0.5, color="gray", linestyle=":", linewidth=0.8)

    xticks = list(range(-1, num_ops))
    xticklabels = ["input\n(before op0)"] + [OP_LABELS.get(i, f"op{i}") for i in range(num_ops)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8)

    ax.axhline(result.total_bytes, color="red", linestyle="--", linewidth=1)
    ax.text(num_ops - 1 + 0.65, result.total_bytes + result.total_bytes * 0.03,
            f"arena top = {result.total_bytes}B", color="red", fontsize=8, va="bottom")

    ax.set_xlim(-1.6, num_ops - 1 + 1.2)
    ax.set_ylim(0, result.total_bytes * 1.25)
    ax.set_xlabel("time (op index)")
    ax.set_ylabel("byte offset in arena[]")
    ax.set_title(f"SRAM arena plan -- {result.total_bytes} bytes total "
                 f"(vs. {sum(t.size_bytes for t in g.tensors.values())}B naive, "
                 f"{result.order_used} strategy)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    g = ir.build_graph()
    result = pl.plan(g)
    render(g, result, "memory_map.png")


if __name__ == "__main__":
    main()
