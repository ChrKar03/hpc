#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./test_omp.sh [options]

Runs the sequential baseline and the OpenMP k-means implementation for a set of
thread counts, captures timing information (averaged over N runs), and checks
that the OpenMP results match the sequential output.

Options:
  -i, --input PATH      Input dataset (default: Image_data/color100.txt)
  -k, --clusters K      Number of clusters (default: 4)
      --threshold VAL   Convergence threshold (default: 0.001)
  -T, --threads "LIST"  Space or comma separated list of OpenMP threads
                        (default: "1 2 4 8")
  -b, --binary          Treat the input as binary (-b flag for the executables)
  -a, --atomic          Enable the atomic accumulation path in omp_main (-a)
      --outdir DIR      Base directory for run artifacts and logs (default: logs)
  -r, --runs N          Number of runs to average (default: 10)
  -h, --help            Show this help and exit

Example:
  ./test_omp.sh -i Image_data/texture100.txt -k 4 --threads "1 4 8" -r 5
EOF
}

INPUT="Image_data/texture17695.bin"
CLUSTERS=2000
THRESHOLD=0.001
THREADS="1 4 8 14 28 56"
IS_BINARY=0
USE_ATOMIC=0
OUTDIR="runs"
ROUNDS=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT="$2"
            shift 2
            ;;
        -k|--clusters)
            CLUSTERS="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -T|--threads)
            THREADS="$2"
            shift 2
            ;;
        -b|--binary)
            IS_BINARY=1
            shift
            ;;
        -a|--atomic)
            USE_ATOMIC=1
            shift
            ;;
        --outdir)
            OUTDIR="$2"
            shift 2
            ;;
        -r|--runs)
            ROUNDS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

THREADS="${THREADS//,/ }"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to run this script." >&2
    exit 1
fi

if [[ ! -x ./omp_main || ! -x ./seq_main ]]; then
    echo "Building seq_main and omp_main..."
    make seq_main omp_main
else
    echo "Ensuring seq_main and omp_main are up to date..."
    make seq_main omp_main
fi

# Create a clear separation: logs go to a distinguishable logs directory,
# all other run artifacts (cluster/membership copies) go to artifacts subdirs.
rm -rf "$OUTDIR"
LOG_DIR="$OUTDIR/logs"
ARTIFACTS_DIR="$OUTDIR/artifacts"
mkdir -p "$LOG_DIR"
mkdir -p "$ARTIFACTS_DIR"

INPUT_ABS="$(python3 - "$INPUT" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

if [[ ! -f "$INPUT_ABS" ]]; then
    echo "Input file not found: $INPUT" >&2
    exit 1
fi

SEQ_ART_DIR="$ARTIFACTS_DIR/seq"
mkdir -p "$SEQ_ART_DIR"

echo "Running sequential baseline on $INPUT (K=$CLUSTERS) for $ROUNDS run(s)..."
seq_cmd=(./seq_main -o -n "$CLUSTERS" -i "$INPUT" -t "$THRESHOLD")
if (( IS_BINARY == 1 )); then
    seq_cmd+=(-b)
fi

SEQ_LOG="$LOG_DIR/run_seq.log"
: > "$SEQ_LOG"

# Run sequential baseline ROUNDS times, append logs into the same file.
for ((i=1;i<=ROUNDS;i++)); do
    {
        echo "=== SEQ RUN $i ==="
        "${seq_cmd[@]}"
    } >> "$SEQ_LOG" 2>&1

    CLUSTER_FILE="${INPUT}.cluster_centres"
    MEMBERSHIP_FILE="${INPUT}.membership"

    if [[ ! -f "$CLUSTER_FILE" || ! -f "$MEMBERSHIP_FILE" ]]; then
        echo "Expected output files not found after sequential run #$i." >&2
        exit 1
    fi

    # Save the first run as the canonical baseline artifacts
    if [[ $i -eq 1 ]]; then
        CLUSTER_BASENAME="$(basename "$CLUSTER_FILE")"
        MEMBERSHIP_BASENAME="$(basename "$MEMBERSHIP_FILE")"
        cp "$CLUSTER_FILE" "$SEQ_ART_DIR/$CLUSTER_BASENAME"
        cp "$MEMBERSHIP_FILE" "$SEQ_ART_DIR/$MEMBERSHIP_BASENAME"
    fi
done

# Compute average sequential computation time from the aggregated log
SEQ_TIME="$(awk '/Computation timing/ {sum+=$(NF-1); n++} END{ if(n) printf("%.6g", sum/n); else print "NA"}' "$SEQ_LOG")"

# Detect numCoords from the sequential log (first occurrence)
NUM_COORDS="$(awk '/numCoords/ {print $NF; exit}' "$SEQ_LOG")"
if [[ -z "${NUM_COORDS:-}" ]]; then
    echo "Error: Unable to detect numCoords from $SEQ_LOG" >&2
    exit 1
fi

declare -a SUMMARY_THREADS=()
declare -a SUMMARY_TIMES=()
declare -a SUMMARY_STD=()
declare -a SUMMARY_MEMBERSHIP=()
declare -a SUMMARY_CENTROIDS=()

STATUS=0

for T in $THREADS; do
    RUN_ART_DIR="$ARTIFACTS_DIR/omp_t${T}"
    mkdir -p "$RUN_ART_DIR"
    echo "Running OpenMP version with $T thread(s) for $ROUNDS run(s)..."

    omp_cmd=(./omp_main -o -n "$CLUSTERS" -i "$INPUT" -t "$THRESHOLD" -p "$T")
    if (( IS_BINARY == 1 )); then
        omp_cmd+=(-b)
    fi
    if (( USE_ATOMIC == 1 )); then
        omp_cmd+=(-a)
    fi

    OMP_LOG="$LOG_DIR/run_t${T}.log"
    : > "$OMP_LOG"

    declare -a RUN_TIMES=()
    MEMBERSHIP_ALL_MATCH=1
    DIFF_RUNS=()
    MAX_CENTROID_DIFF="0"
    ANY_CENTROID_ERROR=0

    for ((i=1;i<=ROUNDS;i++)); do
        {
            echo "=== OMP T=${T} RUN ${i} ==="
            OMP_NUM_THREADS="$T" "${omp_cmd[@]}"
        } >> "$OMP_LOG" 2>&1

        CLUSTER_FILE="${INPUT}.cluster_centres"
        MEMBERSHIP_FILE="${INPUT}.membership"

        if [[ ! -f "$CLUSTER_FILE" || ! -f "$MEMBERSHIP_FILE" ]]; then
            echo "Missing output files after OpenMP run #$i with $T threads." >&2
            STATUS=1
            MEMBERSHIP_ALL_MATCH=0
            DIFF_RUNS+=("$i")
            continue
        fi

        RUN_SUBDIR="$RUN_ART_DIR/run${i}"
        mkdir -p "$RUN_SUBDIR"
        cp "$CLUSTER_FILE" "$RUN_SUBDIR/$(basename "$CLUSTER_FILE")"
        cp "$MEMBERSHIP_FILE" "$RUN_SUBDIR/$(basename "$MEMBERSHIP_FILE")"

        run_time="$(awk '/Computation timing/ {print $(NF-1)}' "$OMP_LOG" | tail -n1)"
        if [[ -z "$run_time" ]]; then
            run_time="NA"
        fi
        RUN_TIMES+=("$run_time")

        if diff -q "$SEQ_ART_DIR/$MEMBERSHIP_BASENAME" "$RUN_SUBDIR/$MEMBERSHIP_BASENAME" >/dev/null 2>&1; then
            :
        else
            MEMBERSHIP_ALL_MATCH=0
            DIFF_RUNS+=("$i")
            STATUS=1
        fi

        # --- centroid diffs (supports both text and binary centroid files) ---
        CENTROID_DIFF="$(python3 - "$SEQ_ART_DIR/$CLUSTER_BASENAME" "$RUN_SUBDIR/$CLUSTER_BASENAME" "$CLUSTERS" "$NUM_COORDS" <<'PY'
import sys, math, array

ref_path, cmp_path, k_str, c_str = sys.argv[1:5]
K = int(k_str)
C = int(c_str)

def try_parse_float_list(tokens):
    vals = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError:
            return None
    return vals

def load_text_centroids(path, K, C):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                toks = s.split()
                # try all floats
                vals = try_parse_float_list(toks)
                if vals is not None:
                    if len(vals) == C:
                        rows.append(vals)
                        continue
                    if len(vals) == C + 1:
                        rows.append(vals[1:])
                        continue
                    # malformed line, skip
                    continue
                # try skipping first token
                if len(toks) > 1:
                    vals2 = try_parse_float_list(toks[1:])
                    if vals2 is not None and len(vals2) == C:
                        rows.append(vals2)
        if len(rows) == K and all(len(r) == C for r in rows):
            return rows
        return None
    except Exception:
        return None

def load_binary_centroids(path, K, C):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None
    # exact fits for float32 and float64
    need_f32 = K * C * 4
    need_f64 = K * C * 8
    if len(data) == need_f32:
        arr = array.array("f")
        arr.frombytes(data)
        # assume little-endian platform; if not, byteswap
        if arr.itemsize != 4:  # extremely rare
            return None
        rows = [arr[i*C:(i+1)*C] for i in range(K)]
        return [list(map(float, r)) for r in rows]
    if len(data) == need_f64:
        arr = array.array("d")
        arr.frombytes(data)
        if arr.itemsize != 8:
            return None
        rows = [arr[i*C:(i+1)*C] for i in range(K)]
        return [list(map(float, r)) for r in rows]
    return None

def load_centroids(path, K, C):
    # text first
    rows = load_text_centroids(path, K, C)
    if rows is not None:
        return rows
    # then binary
    rows = load_binary_centroids(path, K, C)
    return rows

ref = load_centroids(ref_path, K, C)
cmp = load_centroids(cmp_path, K, C)
if ref is None or cmp is None:
    print("parse-error")
    sys.exit(0)
if len(ref) != len(cmp) or any(len(a)!=len(b) for a,b in zip(ref, cmp)):
    print("shape-mismatch")
    sys.exit(0)

maxdiff = 0.0
for ra, rb in zip(ref, cmp):
    for a, b in zip(ra, rb):
        d = abs(a - b)
        if d > maxdiff:
            maxdiff = d
print(f"{maxdiff:.6g}")
PY
)"
        if [[ -z "$CENTROID_DIFF" ]]; then
            CENTROID_DIFF="NA"
            ANY_CENTROID_ERROR=1
            STATUS=1
        fi
        case "$CENTROID_DIFF" in
            parse-error|shape-mismatch)
                ANY_CENTROID_ERROR=1
                STATUS=1
                ;;
            *)
                if [[ "$CENTROID_DIFF" != "NA" ]]; then
                    greater="$(awk -v a="$CENTROID_DIFF" -v b="$MAX_CENTROID_DIFF" 'BEGIN{print (a>b)?"1":"0"}')"
                    if [[ "$greater" == "1" ]]; then
                        MAX_CENTROID_DIFF="$CENTROID_DIFF"
                    fi
                fi
                ;;
        esac

    done

    # === Compute average and standard deviation ===
    read avg_time std_time <<< "$(printf "%s\n" "${RUN_TIMES[@]}" | awk '{
        if($0 ~ /NA/) next;
        sum+=$1; sumsq+=$1*$1; n++
    } END {
        if(n>0) {
            mean=sum/n;
            stddev=sqrt((sumsq/n)-(mean*mean));
            printf("%.6g %.6g", mean, stddev);
        } else print "NA NA"
    }')"

    if [[ $MEMBERSHIP_ALL_MATCH -eq 1 ]]; then
        MEMBERSHIP_STATUS="match"
    else
        MEMBERSHIP_STATUS="DIFF (runs: ${DIFF_RUNS[*]})"
    fi

    SUMMARY_THREADS+=("$T")
    SUMMARY_TIMES+=("$avg_time")
    SUMMARY_STD+=("$std_time")
    SUMMARY_MEMBERSHIP+=("$MEMBERSHIP_STATUS")
    SUMMARY_CENTROIDS+=("${MAX_CENTROID_DIFF:-NA}")
done

# === Final summary output ===
echo
printf "%-10s %-15s %-15s %-25s %-20s\n" "threads" "avg_time(s)" "std_dev(s)" "membership" "max_centroid_delta"
printf "%-10s %-15s %-15s %-25s %-20s\n" "seq" "$SEQ_TIME" "-" "baseline" "-"
for idx in "${!SUMMARY_THREADS[@]}"; do
    printf "%-10s %-15s %-15s %-25s %-20s\n" \
        "${SUMMARY_THREADS[$idx]}" \
        "${SUMMARY_TIMES[$idx]}" \
        "${SUMMARY_STD[$idx]}" \
        "${SUMMARY_MEMBERSHIP[$idx]}" \
        "${SUMMARY_CENTROIDS[$idx]}"
done

ARTIFACTS_ABS="$(python3 - "$ARTIFACTS_DIR" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
LOG_DIR_ABS="$(python3 - "$LOG_DIR" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

echo
echo "Log files stored in: $LOG_DIR_ABS"
echo "Other run artifacts (cluster & membership copies) stored in: $ARTIFACTS_ABS"

exit "$STATUS"
