# Tiny Tensor Compiler

A quantized-CNN compiler stack, built phase by phase from a validated
PyTorch spec down to bit-exact C running on a Raspberry Pi Pico
(RP2040 / Cortex-M0+), with a formal Z3 equivalence proof between the
spec and the generated code, and a measured (disassembly-verified)
optimization pass.

## Pipeline

```
qparams.json (data/)                 <- extracted from a real PyTorch PTQ model
      |
      v
reference.py (src/)                  <- the spec: pure numpy, uint8 in -> uint8 out
      |                                  validated bit-exact against the real
      |                                  quantized PyTorch model, 2000/2000
      v
ir.py (src/)                         <- graph IR, live-range analysis
      |
      v
planner.py (src/)                    <- static SRAM allocator (alias analysis +
      |                                  greedy first-fit), 46% smaller than naive
      v
codegen.py (src/)                    <- IR + memory plan -> model.c / model.h
      |
      +---------------------------+
      |                           |
      v                           v
verify.py (src/)            phaseN_*/pico_project/
Z3: spec == C, exhaustive   real, buildable RP2040 firmware
over the full int8/uint8    (.uf2 included, prebuilt)
domain, incl. a caught bug
```

## Results at a glance

| Metric | Value |
|---|---|
| Reference spec vs. real quantized PyTorch model | **2000/2000** bit-exact |
| SRAM arena, naive vs. planned | 388B → **208B** (46% smaller) |
| Generated C vs. spec, on host | **500/500** bit-exact |
| Z3 proof: C matches spec | **UNSAT** (proven for all 2^144 input combinations, not sampled) |
| Z3 proof: injected bug caught | **SAT**, concrete counterexample in ~76s |
| Optimized vs. naive conv2d, dynamic instructions/output-pixel | 137 → **45** (−67%, hand-verified from ARM disassembly) |
| Z3 proof: optimized C still matches spec | **UNSAT** (re-proven after optimizing, decomposed proof) |

## Repo layout

- **`src/`** — the five shared modules everything else imports. Nothing
  phase-specific lives here.
- **`data/qparams.json`** — the one artifact `phase0_spec/build_model.py`
  produces and every later phase reads.
- **`phase0_spec/`** through **`phase5_optimization/`** — one folder per
  phase, containing only what's specific to that phase (tests, generated
  C, Pico firmware projects)

## Quickstart

```bash
# Python deps
pip install torch numpy z3-solver matplotlib

# ARM cross-compiler (for Phases 3/5's cross-compile + Pico firmware)
apt-get install gcc-arm-none-eabi cmake

# Re-run any phase's checks, e.g.:
cd phase2_memory_planner && python3 test_planner.py
cd phase4_z3_proof && python3 ../src/verify.py

# Regenerate the C from scratch:
cd phase3_codegen && python3 ../src/codegen.py && gcc -O2 -o t test_model.c model.c -lm && ./t

# Flash a Pico: BOOTSEL + plug in USB, drag on the .uf2:
phase3_codegen/pico_project/prebuilt/ttc_phase3.uf2       # naive, MVP milestone
phase5_optimization/pico_project/prebuilt/ttc_phase5.uf2  # naive vs. optimized benchmark
```

## Devlog
### Phase 0 — Spec: 
Planned to use symmetric int8 for both weights and activations. PyTorch's actual backend rejected that, activations turned out to need unsigned quint8 with a zero-point. Fixed it, then checked whether the rounding-mode choice (round-half-to-even) even mattered, and proved it didn't for this model. Validated 2000/2000 bit-exact against the real quantized model.

### Phase 1 — IR: 
Clean build. Main decision: keep weights and activations as separate types, since only activations need memory-planning info. Built tests that check the IR against output shapes, rather than internal consistency.

### Phase 2 — Memory Planner: 
Only let ReLU alias its input buffer in-place, not MaxPool, the latter changes shape and isn't safely provable in general. Result: 46% SRAM reduction (388B → 208B). Tested the allocator on synthetic graphs too to confirm it generalizes.

### Phase 3 — Codegen: 
Generated C matched the Python spec bit-for-exact on host and cross-compiled cleanly for the real chip. Built a flashable firmware image, and then ran it on real hardware to double-check it worked.

### Phase 4 — Formal Proof: 
Used Z3 to prove the C code is mathematically identical to the spec across every possible input (2^144 combinations). Then deliberately broke the code (reintroducing the same signed/unsigned bug from Phase 0) to confirm Z3 could actually catch it. It did, but took 76 seconds, slower than the correctness proof.

### Phase 5 — Optimization: 
Checked the compiler's output first and found it was wasting work by reloading the same weights repeatedly. Fixed that with packed loads and loop unrolling, cutting instructions per pixel by 67%. Re-running the Phase 4 proof against the optimized code timed out at first, which solved by breaking the proof into two smaller pieces.

## GenAI Statement
Generative AI was used in the creation of this project to: (1) teach me the concepts behind the project, (2) proofread code, and (3) assist in debugging.

## Future Work
I plan to move forward with this project by actually applying it to robotics with the RP2040.
