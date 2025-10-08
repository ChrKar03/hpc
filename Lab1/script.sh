#!/usr/bin/env bash
set -euo pipefail

executables=(
  sobel_orig
  sobe_loop_interchange
  sobe_loop_unrolling
  sobe_loop_fusion
  sobe_inlining
  sobe_loop_invariant
  sobe_cse
  sobe_strength_reduction
  sobe_compiler_assist
)

compiler_flags=("-O0" "-ffast-math -mavx2")
results_file="run_results.txt"

> "${results_file}"

for flag in "${compiler_flags[@]}"; do
  echo "=== Building with ${flag} ===" | tee -a "${results_file}"
  make clean
  make CFLAGS="-Wall ${flag}" "${executables[@]}"

  for exe in "${executables[@]}"; do
    echo "--- ${exe} (CFLAGS=${flag}) ---" | tee -a "${results_file}"
    "./${exe}" >> "${results_file}" 2>&1
    echo >> "${results_file}"
  done
done
