#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:?MODEL_REVISION is required}"

PORT="${PORT:-8000}"

GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-1024}"
SEED="${SEED:-0}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-420}"

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"

mkdir -p "$RUN_DIR"

SERVER_PID=""

cleanup() {
    set +e

    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi

            sleep 1
        done

        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

echo "========================================"
echo "Nsight full-profile target"
echo "========================================"
echo "model=$MODEL"
echo "revision=$MODEL_REVISION"
echo "concurrency=$MAX_CONCURRENCY"
echo "prompts=$NUM_PROMPTS"
echo

export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Starting vLLM..."

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --generation-config vllm \
    --revision "$MODEL_REVISION" \
    --no-enable-prefix-caching \
    > "$RUN_DIR/server.log" 2>&1 &

SERVER_PID=$!

echo "$SERVER_PID" > "$RUN_DIR/server.pid"

echo "Waiting for server health..."

READY=0

for _ in $(seq 1 "$SERVER_READY_TIMEOUT_S"); do
    if curl -fsS \
        "http://127.0.0.1:${PORT}/health" \
        >/dev/null 2>&1; then

        READY=1
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server exited during startup."
        tail -n 100 "$RUN_DIR/server.log" || true
        exit 1
    fi

    sleep 1
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: server readiness timeout."
    exit 1
fi

echo "Server healthy."

echo
echo "Warmup..."

RUN_ID="nsys_full_profile_warmup" \
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

echo
echo "Measured diagnostic workload..."

RUN_ID="nsys_full_profile_c${MAX_CONCURRENCY}" \
RUN_DIR="$RUN_DIR/benchmark" \
MODEL="$MODEL" \
PORT="$PORT" \
NUM_PROMPTS="$NUM_PROMPTS" \
NUM_WARMUPS=0 \
REQUEST_RATE=inf \
MAX_CONCURRENCY="$MAX_CONCURRENCY" \
INPUT_LEN="$INPUT_LEN" \
OUTPUT_LEN="$OUTPUT_LEN" \
SEED="$SEED" \
bash vllm/scripts/02_bench_random_baseline.sh

echo
echo "Diagnostic workload complete."

cleanup
SERVER_PID=""

echo "Target complete."