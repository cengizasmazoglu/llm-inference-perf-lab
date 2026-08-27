#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEED="${SEED:-0}"

GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

PROFILE_DELAY_S="${PROFILE_DELAY_S:-2}"
PROFILE_DURATION_S="${PROFILE_DURATION_S:-2}"

WAVES_PER_POINT="${WAVES_PER_POINT:-8}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-420}"

EXPERIMENT_ID="${EXPERIMENT_ID:-004c_nsys_interactive_$(date -u +%Y%m%dT%H%M%SZ)}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-vllm/results/raw/$EXPERIMENT_ID}"

SESSION="vllm_nsys_${RANDOM}_$$"

BENCH_PID=""

mkdir -p "$EXPERIMENT_DIR"

if [[ -n "${EXPORT_DIR:-}" ]]; then
    FINAL_EXPORT_DIR="$EXPORT_DIR"
elif [[ -n "${RUNPOD_POD_ID:-}" && -d /workspace ]]; then
    FINAL_EXPORT_DIR="/workspace"
else
    FINAL_EXPORT_DIR=""
fi


cleanup() {
    set +e

    if [[ -n "${BENCH_PID:-}" ]]; then
        kill "$BENCH_PID" 2>/dev/null || true
        wait "$BENCH_PID" 2>/dev/null || true
    fi

    nsys shutdown \
        --session="$SESSION" \
        --kill=sigterm \
        >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM


echo "========================================"
echo "Experiment 004C"
echo "Interactive Nsight Systems Comparison"
echo "========================================"

echo
echo "Nsight version:"
nsys --version

echo
echo "Resolving model revision..."

MODEL_REVISION="$(
    python vllm/scripts/06_resolve_model_revision.py "$MODEL"
)"

echo "$MODEL_REVISION" \
    > "$EXPERIMENT_DIR/model-revision.txt"

cat > "$EXPERIMENT_DIR/experiment-config.txt" <<EOF
experiment=004c_nsys_interactive
model=$MODEL
model_revision=$MODEL_REVISION
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
seed=$SEED
gpu_memory_utilization=$GPU_UTIL
max_model_len=$MAX_MODEL_LEN
prefix_caching=off
profile_delay_s=$PROFILE_DELAY_S
profile_duration_s=$PROFILE_DURATION_S
collection_control=nsys_interactive_cli
cuda_profiler_api=false
EOF


RUN_ID="$EXPERIMENT_ID" \
RUN_DIR="$EXPERIMENT_DIR" \
bash vllm/scripts/00_check_env.sh


echo
echo "Launching vLLM inside interactive Nsight session..."

export VLLM_WORKER_MULTIPROC_METHOD=spawn

nsys launch \
    --session-new="$SESSION" \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --show-output=true \
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --generation-config vllm \
        --revision "$MODEL_REVISION" \
        --no-enable-prefix-caching


echo
echo "Waiting for vLLM health..."

READY=0

for _ in $(seq 1 "$SERVER_READY_TIMEOUT_S"); do

    if curl -fsS \
        "http://127.0.0.1:${PORT}/health" \
        >/dev/null 2>&1; then

        READY=1
        break
    fi

    sleep 1
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: server did not become healthy."
    exit 1
fi

echo "Server healthy."


echo
echo "Running unprofiled warmup..."

WARMUP_DIR="$EXPERIMENT_DIR/_warmup"

RUN_ID="${EXPERIMENT_ID}_warmup" \
RUN_DIR="$WARMUP_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
NUM_PROMPTS=64 \
NUM_WARMUPS=0 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=64 \
INPUT_LEN="$INPUT_LEN" \
OUTPUT_LEN="$OUTPUT_LEN" \
SEED="$SEED" \
bash vllm/scripts/02_bench_random_baseline.sh


validate_report() {
    local report="$1"
    local stats="$2"

    echo
    echo "Validating CUDA trace:"
    echo "$report"

    nsys stats \
        --report cuda_gpu_kern_sum \
        --report cuda_api_sum \
        "$report" \
        > "$stats" 2>&1

    cat "$stats"

    if grep -qi \
        'does not contain CUDA kernel data' \
        "$stats"; then

        echo
        echo "ERROR: report contains no CUDA kernel data."
        exit 1
    fi

    if ! grep -q \
        'CUDA GPU Kernel Summary' \
        "$stats"; then

        echo
        echo "ERROR: CUDA GPU Kernel Summary missing."
        exit 1
    fi

    echo
    echo "CUDA TRACE VALID."
}


run_profile_point() {
    local C="$1"
    local TAG="$2"
    local NUM_PROMPTS="$3"

    local PADDED_C
    PADDED_C="$(printf '%03d' "$C")"

    local POINT_DIR="$EXPERIMENT_DIR/$TAG"
    local REPORT_BASE="$POINT_DIR/nsys"

    mkdir -p "$POINT_DIR"

    cat > "$POINT_DIR/point-meta.txt" <<EOF
max_concurrency=$C
num_prompts=$NUM_PROMPTS
profile_delay_s=$PROFILE_DELAY_S
profile_duration_s=$PROFILE_DURATION_S
collection_control=nsys_interactive_cli
profiling_run=true
clean_performance_claim=false
EOF

    echo
    echo "========================================"
    echo "$TAG"
    echo "Concurrency: $C"
    echo "Prompts:     $NUM_PROMPTS"
    echo "========================================"

    RUN_ID="${EXPERIMENT_ID}_${TAG}" \
    RUN_DIR="$POINT_DIR" \
    MODEL="$MODEL" \
    PORT="$PORT" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    NUM_WARMUPS=0 \
    REQUEST_RATE=inf \
    MAX_CONCURRENCY="$C" \
    INPUT_LEN="$INPUT_LEN" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    SEED="$SEED" \
    bash vllm/scripts/02_bench_random_baseline.sh \
        > "$POINT_DIR/benchmark-driver.log" 2>&1 &

    BENCH_PID=$!

    sleep "$PROFILE_DELAY_S"

    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
        echo "ERROR: benchmark ended before collection began."
        wait "$BENCH_PID" || true
        BENCH_PID=""
        exit 1
    fi

    echo "Starting external Nsight collection..."

    nsys start \
        --session="$SESSION" \
        --output="$REPORT_BASE"

    sleep "$PROFILE_DURATION_S"

    echo "Stopping external Nsight collection..."

    nsys stop \
        --session="$SESSION"

    set +e
    wait "$BENCH_PID"
    BENCH_STATUS=$?
    set -e

    BENCH_PID=""

    if [[ "$BENCH_STATUS" -ne 0 ]]; then
        echo "ERROR: benchmark failed."
        exit "$BENCH_STATUS"
    fi

    REPORT="$(
        find "$POINT_DIR" \
            -maxdepth 1 \
            -type f \
            -name '*.nsys-rep' \
            | sort -V \
            | head -1
    )"

    if [[ -z "$REPORT" ]]; then
        echo "ERROR: Nsight report not generated."
        exit 1
    fi

    validate_report \
        "$REPORT" \
        "$POINT_DIR/nsys-stats.txt"
}


echo
echo "========================================"
echo "vLLM PROFILING SMOKE"
echo "========================================"

run_profile_point \
    128 \
    "_profile_smoke_c128" \
    512

echo
echo "vLLM interactive profiling smoke PASSED."


echo
echo "Running real diagnostic regions..."

run_profile_point \
    64 \
    "concurrency_064" \
    $((64 * WAVES_PER_POINT))

run_profile_point \
    128 \
    "concurrency_128" \
    $((128 * WAVES_PER_POINT))

run_profile_point \
    256 \
    "concurrency_256" \
    $((256 * WAVES_PER_POINT))


echo
echo "Shutting down Nsight/vLLM session..."

nsys shutdown \
    --session="$SESSION" \
    --kill=sigterm \
    || true


echo
echo "Validating final experiment structure..."

POINT_REPORT_COUNT="$(
    find "$EXPERIMENT_DIR" \
        -type f \
        -path '*/concurrency_*/*.nsys-rep' \
        | wc -l
)"

POINT_BENCHMARK_COUNT="$(
    find "$EXPERIMENT_DIR" \
        -type f \
        -path '*/concurrency_*/benchmark.json' \
        | wc -l
)"

echo "diagnostic_nsys_reports=$POINT_REPORT_COUNT"
echo "diagnostic_benchmarks=$POINT_BENCHMARK_COUNT"

if [[ "$POINT_REPORT_COUNT" -ne 3 ]]; then
    echo "ERROR: expected 3 diagnostic Nsight reports."
    exit 1
fi

if [[ "$POINT_BENCHMARK_COUNT" -ne 3 ]]; then
    echo "ERROR: expected 3 diagnostic benchmark files."
    exit 1
fi


echo
echo "Packaging..."

bash vllm/scripts/07_pack_run.sh \
    "$EXPERIMENT_DIR"

ARCHIVE="$(basename "$EXPERIMENT_DIR").tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

if [[ -n "$FINAL_EXPORT_DIR" ]]; then
    cp "$ARCHIVE" "$CHECKSUM" \
        "$FINAL_EXPORT_DIR/"
fi


echo
echo "========================================"
echo "EXPERIMENT 004C COMPLETE"
echo "========================================"
echo
echo "The benchmark numbers in this experiment"
echo "are diagnostic only."
echo
echo "Canonical performance remains Experiment 003."
echo

if [[ -n "$FINAL_EXPORT_DIR" ]]; then
    echo "Download:"
    echo "$FINAL_EXPORT_DIR/$ARCHIVE"
    echo "$FINAL_EXPORT_DIR/$CHECKSUM"
fi

echo
echo "SAFE TO DOWNLOAD ARTIFACTS AND TERMINATE POD."