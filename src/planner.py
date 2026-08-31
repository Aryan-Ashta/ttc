"""
Phase 2 -- Static Memory Planner.

Maps every activation tensor in the IR graph to a byte
offset in one global `uint8_t arena[N]` array, with zero overlap between
any two tensors whose live ranges intersect in time.

Pipeline:
  1. find_aliases()        -- "output aliases input" for elementwise,
                               shape/dtype-preserving ops (ReLU). This is
                               where most of the SRAM savings come from.
  2. build_buffer_groups()  -- merges aliased tensors into single buffers
                               with a combined live range.
  3. greedy_allocate()      -- first-fit placement, tried with two
                               different sort orders (size-descending,
                               start-time-ascending).
  4. check_no_illegal_overlap() -- a plain-Python (pre-Z3) sanity pass:
                               no two DISTINCT buffers may share space
                               while their live ranges intersect in time.
                               Phase 4 will prove this properly with Z3;
                               this is the cheap net that catches bugs
                               right now, while building the planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import ir

RP2040_SRAM_BYTES = 264 * 1024


# ---------------------------------------------------------------------------
# Step 1: alias analysis
# ---------------------------------------------------------------------------

def find_aliases(g: ir.Graph) -> Dict[str, str]:
    """Returns {output_tensor_name: input_tensor_name} for every op whose
    output can safely overwrite its input's buffer in place.

    Rule: only ops whose output has the SAME shape and dtype as its input qualify.
    In this op set that's ReLU (elementwise, shape-preserving).
    """
    aliases: Dict[str, str] = {}
    for op in g.ops:
        if isinstance(op, ir.ReluOp):
            in_t = g.tensors[op.input]
            out_t = g.tensors[op.output]
            if in_t.shape == out_t.shape and in_t.dtype == out_t.dtype:
                aliases[op.output] = op.input
    return aliases


def resolve_alias_root(name: str, aliases: Dict[str, str]) -> str:
    """Follows an alias chain to the tensor that actually owns memory."""
    while name in aliases:
        name = aliases[name]
    return name


# ---------------------------------------------------------------------------
# Step 2: buffer groups (aliased tensors merged into one physical buffer)
# ---------------------------------------------------------------------------

@dataclass
class BufferGroup:
    root: str                    # tensor name that "owns" the memory (first-defined)
    members: List[str]           # all tensor names sharing this buffer, root first
    size: int                    # bytes
    start: int                   # merged first_def (when the memory is first written)
    end: int                     # merged last_use (last op that reads ANY member)
    offset: Optional[int] = None  # filled in by greedy_allocate()

    def label(self) -> str:
        if len(self.members) == 1:
            return self.root
        return f"{self.root}(={'='.join(self.members[1:])})"


def build_buffer_groups(g: ir.Graph, aliases: Dict[str, str]) -> List[BufferGroup]:
    aliased_outputs = set(aliases.keys())
    groups: Dict[str, BufferGroup] = {}

    for name, t in g.tensors.items():
        if name in aliased_outputs:
            continue  # folded into its root's group below
        groups[name] = BufferGroup(root=name, members=[name], size=t.size_bytes,
                                    start=t.first_def, end=t.last_use)

    for out_name, in_name in aliases.items():
        root = resolve_alias_root(in_name, aliases)
        grp = groups[root]
        out_t = g.tensors[out_name]
        grp.members.append(out_name)
        grp.size = max(grp.size, out_t.size_bytes)
        grp.end = max(grp.end, out_t.last_use)
        # grp.start is untouched: the memory's lifetime starts when the
        # ROOT first writes it, not when an alias later overwrites it.

    return list(groups.values())


# ---------------------------------------------------------------------------
# Step 3: interval overlap + greedy first-fit allocator
# ---------------------------------------------------------------------------

def time_overlap(a: BufferGroup, b: BufferGroup) -> bool:
    """Closed-interval, any-shared-op-index overlap."""
    return a.start <= b.end and b.start <= a.end


def space_overlap(offset_a: int, size_a: int, offset_b: int, size_b: int) -> bool:
    """Half-open byte ranges: [offset, offset+size). Touching at a
    boundary is NOT an overlap."""
    return offset_a < offset_b + size_b and offset_b < offset_a + size_a


def greedy_allocate(groups: List[BufferGroup], order: str) -> Tuple[List[BufferGroup], int]:
    """First-fit placement. order: 'size_desc' or 'start_time'.
    Mutates and returns the groups (each gets .offset set) plus the total
    arena size (bytes)."""
    if order == "size_desc":
        seq = sorted(groups, key=lambda x: (-x.size, x.root))
    elif order == "start_time":
        seq = sorted(groups, key=lambda x: (x.start, x.root))
    else:
        raise ValueError(f"unknown order {order!r}")

    placed: List[BufferGroup] = []
    for grp in seq:
        offset = 0
        while True:
            conflict = next(
                (p for p in placed
                 if time_overlap(grp, p) and space_overlap(offset, grp.size, p.offset, p.size)),
                None,
            )
            if conflict is None:
                break
            offset = conflict.offset + conflict.size  # hop past it, re-scan from the top
        grp.offset = offset
        placed.append(grp)

    total = max((g.offset + g.size for g in placed), default=0)
    return placed, total


# ---------------------------------------------------------------------------
# Step 4: sanity check (plain Python; Z3 does the formal version in Phase 4)
# ---------------------------------------------------------------------------

def check_no_illegal_overlap(groups: List[BufferGroup]) -> List[Tuple[str, str]]:
    """Between DISTINCT buffer groups only -- aliasing within a group is
    intentional (same address, that's the point) and must not be flagged."""
    violations = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            if time_overlap(a, b) and space_overlap(a.offset, a.size, b.offset, b.size):
                violations.append((a.root, b.root))
    return violations


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    groups: List[BufferGroup]
    memory_map: Dict[str, int]   # every activation tensor name -> byte offset
    total_bytes: int
    order_used: str
    comparison: Dict[str, int]   # order name -> total bytes, for both strategies tried


def plan(g: ir.Graph) -> PlanResult:
    aliases = find_aliases(g)
    base_groups = build_buffer_groups(g, aliases)

    comparison = {}
    best_order, best_groups, best_total = None, None, None
    for order in ("size_desc", "start_time"):
        # greedy_allocate mutates the BufferGroup objects it's given, so
        # give each trial its own fresh copies.
        trial_groups = [BufferGroup(g2.root, list(g2.members), g2.size, g2.start, g2.end)
                         for g2 in base_groups]
        placed, total = greedy_allocate(trial_groups, order)
        comparison[order] = total
        if best_total is None or total < best_total:
            best_order, best_groups, best_total = order, placed, total

    violations = check_no_illegal_overlap(best_groups)
    if violations:
        raise RuntimeError(f"allocator produced overlapping buffers: {violations}")

    memory_map: Dict[str, int] = {}
    for grp in best_groups:
        for member in grp.members:
            memory_map[member] = grp.offset

    assert best_total < RP2040_SRAM_BYTES, (
        f"arena size {best_total} exceeds RP2040 SRAM ({RP2040_SRAM_BYTES} bytes)"
    )

    return PlanResult(groups=best_groups, memory_map=memory_map, total_bytes=best_total,
                       order_used=best_order, comparison=comparison)


def print_plan(g: ir.Graph, result: PlanResult) -> None:
    print("Allocator strategy comparison (total arena bytes):")
    for order, total in result.comparison.items():
        marker = "  <- used" if order == result.order_used else ""
        print(f"  {order:12s} {total:5d} bytes{marker}")

    print(f"\nChosen: {result.order_used}, total arena = {result.total_bytes} bytes "
          f"(RP2040 has {RP2040_SRAM_BYTES} bytes SRAM)")

    print("\nBuffer groups:")
    print(f"  {'buffer':30s} {'offset':>7s} {'size':>5s} {'[start,end]':>12s}")
    for grp in sorted(result.groups, key=lambda x: x.offset):
        print(f"  {grp.label():30s} {grp.offset:7d} {grp.size:5d} "
              f"[{grp.start:3d},{grp.end:3d}]")

    print("\nFull memory map (every activation tensor -> offset):")
    for name, t in g.tensors.items():
        print(f"  {name:10s} -> offset {result.memory_map[name]:4d}  "
              f"(size {t.size_bytes} bytes)")

    aliased = {out: inp for out, inp in find_aliases(g).items()}
    if aliased:
        print("\nAliasing applied (SRAM saved by not double-allocating):")
        for out_name, in_name in aliased.items():
            saved = g.tensors[out_name].size_bytes
            print(f"  {out_name} writes in-place into {in_name}'s buffer -- saves {saved} bytes")

    naive_total = sum(t.size_bytes for t in g.tensors.values())
    print(f"\nNaive (no reuse at all, sum of every tensor) would be {naive_total} bytes.")
    print(f"Planned arena is {result.total_bytes} bytes "
          f"({naive_total - result.total_bytes} bytes saved, "
          f"{100*(naive_total-result.total_bytes)/naive_total:.0f}% reduction).")


def main():
    g = ir.build_graph()
    result = plan(g)
    print_plan(g, result)


if __name__ == "__main__":
    main()
