# Sobel Optimization Report

## Benchmark Setup
- Image size: `4096 x 4096` grayscale (`SIZE` constant shared by every implementation).
- Timing: wall-clock interval using `clock_gettime(CLOCK_MONOTONIC_RAW)` inside `sobel()`.
- Data set: `input.grey` as source, results written to `output_sobel.grey`, validated against `golden.grey` (every run reported PSNR = `inf`, matching the reference image).
- Toolchain: Intel `icx`, evaluated with `-O0` (no auto-optimisation) and `-ffast-math` (aggressive math transformations).
- Automation: `script.sh` rebuilds each variant under both flag sets, executes the binaries, and appends timings to `run_results.txt`. `plot_results.py` consumes that log to produce execution-time and speedup charts (`run_results.png` and `run_results_speedup.png`).

## Timing Summary

| Variant | -O0 time (s) | -ffast-math time (s) |
| --- | ---: | ---: |
| `sobel_orig` | 1.61868 | 0.400537 |
| `sobe_loop_interchange` | 1.16464 | 0.112307 |
| `sobe_loop_unrolling` | 1.18138 | 0.176520 |
| `sobe_loop_fusion` | 1.14022 | 0.145559 |
| `sobe_inlining` | 0.854159 | 0.149554 |
| `sobe_loop_invariant` | 0.865810 | 0.150815 |
| `sobe_cse` | 0.906098 | 0.142253 |
| `sobe_strength_reduction` | 0.206763 | 0.090328 |
| `sobe_compiler_assist` | 0.204297 | 0.035874 |

## Optimisation-by-optimisation Analysis

### Loop Interchange — `sobe_loop_interchange.c`
- Change: swapped the loop order so rows are traversed outermost, aligning accesses with row-major storage.
- Impact: cache-friendly traversal trims ~28% off the baseline at `-O0` (1.62 → 1.16 s) and ~72% with `-ffast-math` (0.40 → 0.11 s).

### Loop Unrolling — `sobe_loop_unrolling.c`
- Change: unrolled the inner loop by 4, with a scalar cleanup for remaining columns.
- Impact: Slight slowdown relative to loop interchange; the extra convolution calls and register pressure outweigh reduced loop overhead for this stencil.

### Loop Fusion — `sobe_loop_fusion.c`
- Change: merged the convolution and PSNR loops so each pixel is processed once and its error accumulated immediately.
- Impact: Recovers the unrolling regression and edges past the interchange baseline thanks to halved memory traffic.

### Function Inlining — `sobe_inlining.c`
- Change: inlined `convolution2D()` via a macro, keeping computation local to `sobel()` and exposing more optimisation opportunities.
- Impact: Large win at `-O0` (1.14 → 0.85 s) by removing call overhead and enabling scalar optimisations. High-O optimisation already inlines the helper, so `-ffast-math` barely shifts.

### Loop Invariant Code Motion — `sobe_loop_invariant.c`
- Change: hoisted row pointers (`upper`, `middle`, `lower`) outside the column loop, reducing repeated address arithmetic.
- Impact: Neutral to mildly positive; minor ALU savings leave runtime roughly unchanged at both flag levels.

### Common Subexpression Elimination — `sobe_cse.c`
- Change: cached the nine neighbouring pixel values before applying the Sobel weights, avoiding duplicate loads inside the macro.
- Impact: No significant change; after the first access, neighbours are hot in cache, so the extra temporaries mostly shift register usage.

### Strength Reduction — `sobe_strength_reduction.c`
- Change: replaced the repeated `pow()` calls with integer multiplications and direct squared-magnitude accumulation.
- Impact: The breakthrough optimisation—runtime drops to 0.21 s (`-O0`) and 0.09 s (`-ffast-math`), eliminating expensive transcendental math.

### Compiler Assistance — `sobe_compiler_assist.c`
- Change: marked filters `static const` and sobel buffers/pointers `restrict`, preserving the strength-reduction improvements while helping vectorisation.
- Impact: Small improvement at `-O0`, but a large one under `-ffast-math` (0.09 → 0.036 s) because the compiler can now assume non-aliasing and keep values in registers.

Best observed speedup: 11.17× achieved by `sobe_compiler_assist` under `-ffast-math` compiler flags.

## Takeaways
- Memory-locality fixes (interchange, fusion) are valuable early steps before heavier arithmetic changes.
- Manual unrolling and CSE do not guarantee gains; they can cost registers without complementary vector-friendly structure.
- Removing high-cost math (`pow`) is the pivotal optimisation, unlocking ~8× better runtime at `-O0`.
- Communicating invariants to the compiler (`restrict`, `const`) compounds improvements when aggressive optimisation flags are available.
