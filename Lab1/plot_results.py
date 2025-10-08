#!/usr/bin/env python3
import argparse
import collections
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


FLAG_HEADER_RE = re.compile(r"^=== Building with (.+) ===$")
EXEC_HEADER_RE = re.compile(r"^--- (\S+) \(CFLAGS=([^)]+)\) ---$")
DEFAULT_INPUT = "run_results.txt"
DEFAULT_OUTPUT = "run_results.png"


def parse_results(path: pathlib.Path):
	results = collections.OrderedDict()
	current_flag = None
	current_exec = None

	for raw_line in path.read_text().splitlines():
		line = raw_line.strip()
		if not line:
			continue

		flag_match = FLAG_HEADER_RE.match(line)
		if flag_match:
			current_flag = flag_match.group(1)
			results.setdefault(current_flag, collections.OrderedDict())
			current_exec = None
			continue

		exec_match = EXEC_HEADER_RE.match(line)
		if exec_match:
			current_exec = exec_match.group(1)
			continue

		if line.startswith("Total time") and current_flag and current_exec:
			time_token = line.split("=", 1)[1].split()[0]
			results[current_flag][current_exec] = float(time_token)
			current_exec = None

	return results


def plot(results, output_path: pathlib.Path, show: bool):
	flags = list(results.keys())
	if not flags:
		raise ValueError("No results found to plot.")

	variants = list(results[flags[0]].keys())

	x = np.arange(len(variants))
	width = 0.8 / max(len(flags), 1)

	fig, ax = plt.subplots(figsize=(12, 6))
	for index, flag in enumerate(flags):
		times = [results[flag].get(variant, np.nan) for variant in variants]
		offset = width * (index - (len(flags) - 1) / 2)
		ax.bar(x + offset, times, width=width, label=flag)

	ax.set_xticks(x)
	ax.set_xticklabels(variants, rotation=45, ha="right")
	ax.set_ylabel("Execution time (seconds)")
	ax.set_title("Sobel Variants Execution Time")
	ax.grid(axis="y", linestyle="--", alpha=0.4)
	ax.legend(title="Compiler flags")
	fig.tight_layout()

	fig.savefig(output_path, dpi=200)

	# Speedup plot relative to sobel_orig for each flag
	speedup_fig, speedup_ax = plt.subplots(figsize=(12, 6))
	for index, flag in enumerate(flags):
		flag_results = results[flag]
		baseline_exec = variants[0]
		baseline_time = flag_results.get(baseline_exec)
		if baseline_time is None or baseline_time == 0.0:
			continue
		speedups = [baseline_time / flag_results.get(variant, np.nan) for variant in variants]
		offset = width * (index - (len(flags) - 1) / 2)
		speedup_ax.bar(x + offset, speedups, width=width, label=flag)

	speedup_ax.set_xticks(x)
	speedup_ax.set_xticklabels(variants, rotation=45, ha="right")
	speedup_ax.set_ylabel("Speedup vs sobel_orig")
	speedup_ax.set_title("Sobel Variants Speedup")
	speedup_ax.grid(axis="y", linestyle="--", alpha=0.4)
	speedup_ax.legend(title="Compiler flags")
	speedup_fig.tight_layout()

	speedup_path = output_path.with_name(output_path.stem + "_speedup" + output_path.suffix)
	speedup_fig.savefig(speedup_path, dpi=200)

	if show:
		plt.show()


def main():
	parser = argparse.ArgumentParser(description="Plot Sobel optimisation timing results.")
	parser.add_argument("--input", type=pathlib.Path, default=DEFAULT_INPUT, help="Path to run_results.txt")
	parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT, help="Output image file")
	parser.add_argument("--show", action="store_true", help="Display the plot interactively")
	args = parser.parse_args()

	results = parse_results(args.input)
	plot(results, args.output, args.show)


if __name__ == "__main__":
	main()
