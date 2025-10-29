#!/usr/bin/env python3
import os
import re
import math
import matplotlib.pyplot as plt

# === Configuration ===
LOG_DIR = "runs/logs"   # Default log path used by test_omp.sh
SAVE_PLOTS = True       # Save plots as PNG
SHOW_PLOTS = True       # Show plots interactively

# === Regex patterns ===
seq_time_pattern = re.compile(r"Computation timing\s*=\s*([\d.]+)")
omp_thread_pattern = re.compile(r"Threads\s*=\s*(\d+)")
omp_time_pattern = re.compile(r"Computation timing\s*=\s*([\d.]+)")

def parse_seq_time(log_dir):
    seq_path = os.path.join(log_dir, "run_seq.log")
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequential log not found: {seq_path}")
    with open(seq_path, "r") as f:
        text = f.read()
    times = [float(x) for x in seq_time_pattern.findall(text)]
    if not times:
        raise ValueError("No sequential timing entries found.")
    seq_avg = sum(times) / len(times)
    return seq_avg

def parse_omp_logs_raw(log_dir):
    """
    Returns:
      data: dict[int -> list[float]]
            Example: {1: [7.11, 7.09, ...], 4: [2.14, 2.13, ...], ...}
    """
    data = {}
    for fname in sorted(os.listdir(log_dir)):
        if not fname.startswith("run_t") or not fname.endswith(".log"):
            continue
        path = os.path.join(log_dir, fname)
        with open(path, "r") as f:
            text = f.read()

        thr_match = omp_thread_pattern.search(text)
        times = [float(x) for x in omp_time_pattern.findall(text)]
        if thr_match and times:
            t = int(thr_match.group(1))
            data[t] = times
    return dict(sorted(data.items(), key=lambda kv: kv[0]))

def summarize_from_raw(seq_time, data):
    """
    From per-run times, compute avg/std per thread and derive speedup/efficiency.
    Returns:
      summary: list of dicts with keys:
        threads, avg_time, std_time, speedup, efficiency
    """
    summary = []
    for t, times in data.items():
        n = len(times)
        avg = sum(times) / n
        sumsq = sum(x*x for x in times)
        std = math.sqrt(sumsq / n - avg*avg) if n > 0 else float("nan")
        sp = seq_time / avg
        eff = sp / t * 100.0
        summary.append({
            "threads": t,
            "avg_time": avg,
            "std_time": std,
            "speedup": sp,
            "efficiency": eff
        })
    return summary

def plot_speedup_efficiency(seq_time, summary):
    threads = [row["threads"] for row in summary]
    avg_times = [row["avg_time"] for row in summary]
    speedup = [row["speedup"] for row in summary]
    efficiency = [row["efficiency"] for row in summary]

    print("\n=== Performance Summary ===")
    print(f"Sequential average time: {seq_time:.4f} s")
    print(f"{'Threads':>8} | {'Avg Time (s)':>12} | {'Std (s)':>10} | {'Speedup':>8} | {'Efficiency (%)':>15}")
    print("-" * 68)
    for row in summary:
        print(f"{row['threads']:>8} | {row['avg_time']:>12.4f} | {row['std_time']:>10.4f} | "
              f"{row['speedup']:>8.2f} | {row['efficiency']:>15.2f}")

    # --- Speedup line plot ---
    plt.figure(figsize=(8,5))
    plt.plot(threads, speedup, marker='o', label="Measured Speedup")
    plt.plot(threads, threads, "--", label="Ideal Linear Speedup")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title("K-Means OpenMP Speedup vs Threads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig("speedup.png", dpi=200)
    if SHOW_PLOTS: plt.show()

    # --- Efficiency line plot ---
    plt.figure(figsize=(8,5))
    plt.plot(threads, efficiency, marker='s', label="Efficiency (%)")
    plt.xlabel("Number of Threads")
    plt.ylabel("Parallel Efficiency (%)")
    plt.title("K-Means OpenMP Efficiency vs Threads")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig("efficiency.png", dpi=200)
    if SHOW_PLOTS: plt.show()

def plot_boxplots(seq_time, data):
    """
    Boxplots:
      1) Computation time per thread
      2) Speedup per thread (computed per run vs seq_avg)
    """
    threads = list(data.keys())
    times_lists = [data[t] for t in threads]
    speedup_lists = [[seq_time / tm for tm in lst] for lst in times_lists]

    # --- Times boxplot ---
    plt.figure(figsize=(9,5))
    plt.boxplot(times_lists, labels=[str(t) for t in threads], showmeans=True)
    plt.xlabel("Number of Threads")
    plt.ylabel("Computation Time (s)")
    plt.title("K-Means OpenMP: Computation Time Distribution per Thread Count")
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig("times_boxplot.png", dpi=200)
    if SHOW_PLOTS: plt.show()

    # --- Speedup boxplot ---
    plt.figure(figsize=(9,5))
    plt.boxplot(speedup_lists, labels=[str(t) for t in threads], showmeans=True)
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (vs sequential average)")
    plt.title("K-Means OpenMP: Speedup Distribution per Thread Count")
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout()
    if SAVE_PLOTS: plt.savefig("speedup_boxplot.png", dpi=200)
    if SHOW_PLOTS: plt.show()

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' not found.")
        return

    seq_time = parse_seq_time(LOG_DIR)
    data = parse_omp_logs_raw(LOG_DIR)
    if not data:
        print("No OpenMP run logs found in", LOG_DIR)
        return

    summary = summarize_from_raw(seq_time, data)
    plot_speedup_efficiency(seq_time, summary)
    plot_boxplots(seq_time, data)

if __name__ == "__main__":
    main()
