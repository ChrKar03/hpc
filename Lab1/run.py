import subprocess
import time
import sys
import matplotlib.pyplot as plt

labels = {
    "sobel_orig": "Original",
    "sobel_loop_interchange": "Loop Interchange",
    "sobel_loop_unrolling": "Loop Unrolling",
    "sobel_loop_fusion": "Loop Fusion",
    "sobel_function_inlining": "Function Inlining",
    "sobel_loop_invariant": "Loop Invariant",
    "sobel_cse": "Common Subexpression Elimination",
    "sobel_strength_elimination": "Strength Elimination",
    "sobel_compiler_assist": "Compiler Assist"
}

executables = list(labels.keys())
executables_fast = [f"{key}_fast" for key in labels.keys()]

N_RUNS = 10

def run_executables(executables_list):
    results = {}
    for exe in executables_list:
        all_times = []
        for _ in range(N_RUNS):
            process = subprocess.Popen(f"./{exe}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = process.communicate()
            data = stdout.decode()
            data = data.splitlines()[0]
            data = data.split('=')[1].strip().split()[0]
            data = float(data)
            all_times.append(data)
        avg_time = sum(all_times) / len(all_times)
        std_time = (sum((x - avg_time) ** 2 for x in all_times) / len(all_times)) ** 0.5
        results[exe] = (avg_time, std_time)
        print(f"Executable:     {exe}")
        print(f"Execution Time: {avg_time} ± {std_time} seconds")
        print("-" * 40)
    return results

if len(sys.argv) > 1 and sys.argv[1] == "run":
    print("\nRunning standard executables...")
    standard_results = run_executables(executables)

    print("\nRunning fast executables...")
    fast_results = run_executables(executables_fast)

    print("All executions completed.")

    # Save results to a text file
    with open("execution_times.txt", "w") as f:
        f.write("Standard Executables:\n")
        for exe, (avg, std) in standard_results.items():
            f.write(f"{exe}: {avg} ± {std} seconds\n")
        f.write("\nFast Executables:\n")
        for exe, (avg, std) in fast_results.items():
            f.write(f"{exe}: {avg} ± {std} seconds\n")
else:
    # Read results from the text file and plot
    try:
        with open("execution_times.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("No execution times found. Please run the script with 'run' argument first.")
        sys.exit(1)

    # Parse the results
    standard_results = {}
    fast_results = {}
    current_dict = None
    for line in lines:
        line = line.strip()
        if line == "Standard Executables:":
            current_dict = standard_results
        elif line == "Fast Executables:":
            current_dict = fast_results
        elif line:
            exe, rest = line.split(":")
            splt = rest.strip().split("±")
            avg_time = float(splt[0].strip())
            std_time = float(splt[1].strip().split()[0])
            current_dict[exe] = (avg_time, std_time)

# Prepare data for plotting
standard_avgs = [standard_results[f"{label}"][0] for label in labels]
fast_avgs = [fast_results[f"{label}_fast"][0] for label in labels]

standard_stds = [standard_results[f"{label}"][1] for label in labels]
fast_stds = [fast_results[f"{label}_fast"][1] for label in labels]

x = range(len(labels))
width = 0.35

# Plotting with error bars
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width/2 for i in x], standard_avgs, width,
       yerr=standard_stds, label='Standard', color='#1f77b4', capsize=5)
ax.bar([i + width/2 for i in x], fast_avgs, width,
       yerr=fast_stds, label='Fast', color='#ff7f0e', capsize=5)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Execution Times of Sobel Implementations')
ax.set_xticks(x)
ax.set_xticklabels(list(labels.values()), rotation=30, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig("execution_times.png")
plt.show()

# Plotting speedup
original_standard_time = standard_results["sobel_orig"][0]
original_fast_time = fast_results["sobel_orig_fast"][0]

standard_speedups = [original_standard_time / standard_results[f"{label}"][0] for label in labels]
fast_speedups = [original_fast_time / fast_results[f"{label}_fast"][0] for label in labels]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width/2 for i in x], standard_speedups, width,
       label='Standard', color='#1f77b4')
ax.bar([i + width/2 for i in x], fast_speedups, width,
       label='Fast', color='#ff7f0e')
ax.set_ylabel('Speedup')
ax.set_title('Speedup of Sobel Implementations Compared to Original')
ax.set_xticks(x)
ax.set_xticklabels(list(labels.values()), rotation=30, ha='right')
ax.set_yticklabels([f"{int(tick)}x" for tick in ax.get_yticks()])
ax.legend()
plt.tight_layout()
plt.savefig("speedup.png")
plt.show()