#!/usr/bin/env bash
set -euo pipefail


MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION is required}"

PORT="${PORT:-8000}"

GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

SEED="${SEED:-0}"

WAVES_PER_POINT="${WAVES_PER_POINT:-2}"
IDLE_GAP_S="${IDLE_GAP_S:-8}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-420}"

RUN_DIR="${RUN_DIR:-/workspace/006_nsys_short_regimes}"

NSYS_BIN="${NSYS_BIN:-nsys}"

NSYS_PID=""


mkdir -p "$RUN_DIR"

rm -f \
    "$RUN_DIR/client_windows.tsv" \
    "$RUN_DIR/experiment-meta.txt"


timestamp_monotonic_ns() {
    python -c \
        'import time; print(time.monotonic_ns())'
}


timestamp_utc() {
    date --iso-8601=ns
}


record_window() {
    local regime="$1"
    local edge="$2"

    printf '%s\t%s\t%s\t%s\n' \
        "$regime" \
        "$edge" \
        "$(timestamp_monotonic_ns)" \
        "$(timestamp_utc)" \
        >> "$RUN_DIR/client_windows.tsv"
}


stop_profile() {
    if [[ -z "${NSYS_PID:-}" ]]; then
        return
    fi

    if kill -0 "$NSYS_PID" 2>/dev/null; then
        echo
        echo "Stopping Nsight collection..."

        # For `nsys profile`, SIGINT behaves like Ctrl+C:
        # stop collection, finalize the report, and terminate
        # the target according to --kill.
        kill -INT "$NSYS_PID" 2>/dev/null || true

        for _ in $(seq 1 120); do
            if ! kill -0 "$NSYS_PID" 2>/dev/null; then
                break
            fi

            sleep 1
        done

        if kill -0 "$NSYS_PID" 2>/dev/null; then
            echo "Nsight did not exit after SIGINT; sending TERM."
            kill -TERM "$NSYS_PID" 2>/dev/null || true
        fi
    fi

    wait "$NSYS_PID" 2>/dev/null || true

    NSYS_PID=""
}


cleanup() {
    stop_profile
}


trap cleanup EXIT INT TERM


run_benchmark() {
    local regime="$1"
    local concurrency="$2"
    local num_prompts="$3"

    local point_dir
    point_dir="$RUN_DIR/concurrency_$(printf '%03d' "$concurrency")"

    echo
    echo "========================================"
    echo "$regime"
    echo "========================================"
    echo "concurrency=$concurrency"
    echo "num_prompts=$num_prompts"
    echo "waves=$WAVES_PER_POINT"

    record_window \
        "$regime" \
        "START"

    RUN_ID="nsys_${regime}" \
    RUN_DIR="$point_dir" \
    MODEL="$MODEL" \
    PORT="$PORT" \
    NUM_PROMPTS="$num_prompts" \
    NUM_WARMUPS=0 \
    REQUEST_RATE=inf \
    MAX_CONCURRENCY="$concurrency" \
    INPUT_LEN="$INPUT_LEN" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    SEED="$SEED" \
    bash vllm/scripts/02_bench_random_baseline.sh

    record_window \
        "$regime" \
        "END"

    echo
    echo "$regime complete."

    sleep "$IDLE_GAP_S"
}


echo "========================================"
echo "Experiment 006"
echo "Short Nsight concurrency regimes"
echo "========================================"

echo "MODEL=$MODEL"
echo "MODEL_REVISION=$MODEL_REVISION"
echo "RUN_DIR=$RUN_DIR"
echo "WAVES_PER_POINT=$WAVES_PER_POINT"
echo


if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
    echo "ERROR: Nsight Systems not found: $NSYS_BIN"
    exit 1
fi


if curl -fsS \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null 2>&1; then

    echo "ERROR: port $PORT already has a healthy server."
    exit 1
fi


{
    echo "experiment=006_nsys_short_regimes"
    echo "git_sha=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "model_revision=$MODEL_REVISION"
    echo "gpu_util=$GPU_UTIL"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "input_len=$INPUT_LEN"
    echo "output_len=$OUTPUT_LEN"
    echo "seed=$SEED"
    echo "waves_per_point=$WAVES_PER_POINT"
    echo "idle_gap_s=$IDLE_GAP_S"

    echo
    echo "=== NSIGHT ==="
    "$NSYS_BIN" --version

    echo
    echo "=== VLLM ==="
    python - <<'PY'
import vllm
print(vllm.__version__)
PY

    echo
    echo "=== TORCH ==="
    python - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
PY

    echo
    echo "=== GPU ==="
    nvidia-smi \
        --query-gpu=name,uuid,driver_version \
        --format=csv,noheader
} > "$RUN_DIR/experiment-meta.txt"


export VLLM_WORKER_MULTIPROC_METHOD=spawn


echo "Starting vLLM under full-session Nsight profiling..."


"$NSYS_BIN" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --cuda-event-trace=false \
    --kill=sigterm \
    --force-overwrite=true \
    --output="$RUN_DIR/server_full" \
    vllm serve "$MODEL" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --generation-config vllm \
        --revision "$MODEL_REVISION" \
        --no-enable-prefix-caching \
    > "$RUN_DIR/server.log" 2>&1 &

NSYS_PID=$!


echo "Nsight PID: $NSYS_PID"
echo "Waiting for server..."


READY=0

for _ in $(seq 1 "$SERVER_READY_TIMEOUT_S"); do
    if curl -fsS \
        "http://127.0.0.1:${PORT}/health" \
        >/dev/null 2>&1; then

        READY=1
        break
    fi

    if ! kill -0 "$NSYS_PID" 2>/dev/null; then
        echo "ERROR: Nsight/server exited during startup."
        tail -n 120 "$RUN_DIR/server.log" || true
        exit 1
    fi

    sleep 1
done


if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: server readiness timeout."
    tail -n 120 "$RUN_DIR/server.log" || true
    exit 1
fi


echo "Server ready."


echo
echo "========================================"
echo "Warmup"
echo "========================================"


RUN_ID="nsys_warmup" \
RUN_DIR="$RUN_DIR/warmup" \
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


sleep "$IDLE_GAP_S"


run_benchmark \
    "C64" \
    64 \
    $((64 * WAVES_PER_POINT))


run_benchmark \
    "C128" \
    128 \
    $((128 * WAVES_PER_POINT))


run_benchmark \
    "C256" \
    256 \
    $((256 * WAVES_PER_POINT))


stop_profile


if [[ ! -f "$RUN_DIR/server_full.nsys-rep" ]]; then
    echo "ERROR: Nsight report was not generated."
    exit 1
fi


echo
echo "========================================"
echo "Nsight report generated"
echo "========================================"

ls -lh \
    "$RUN_DIR/server_full.nsys-rep"


RUN_NAME="$(basename "$RUN_DIR")"
RUN_PARENT="$(dirname "$RUN_DIR")"

ARCHIVE="$RUN_PARENT/${RUN_NAME}.tar.gz"
CHECKSUM="${ARCHIVE}.sha256"


echo
echo "Packaging run..."


tar -czf "$ARCHIVE" \
    -C "$RUN_PARENT" \
    "$RUN_NAME"


sha256sum "$ARCHIVE" \
    > "$CHECKSUM"


echo
echo "========================================"
echo "Experiment complete"
echo "========================================"

ls -lh \
    "$ARCHIVE" \
    "$CHECKSUM"
