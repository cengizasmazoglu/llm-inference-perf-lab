#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

MIN_PROMPTS="${MIN_PROMPTS:-128}"
WAVES_PER_POINT="${WAVES_PER_POINT:-8}"

SEED="${SEED:-0}"

GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32 64}"

SWEEP_ID="${SWEEP_ID:-002_concurrency_saturation_$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${SWEEP_DIR:-vllm/results/raw/$SWEEP_ID}"

SERVER_PID=""
TELEMETRY_PID=""

mkdir -p "$SWEEP_DIR"

cleanup() {
  set +e

  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi

  if [[ -n "${SERVER_PID:-}" ]]; then
    echo
    echo "Stopping vLLM server..."
    kill -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "========================================"
echo "Experiment 002"
echo "Concurrency Saturation Sweep"
echo "========================================"
echo "SWEEP_ID=$SWEEP_ID"
echo "SWEEP_DIR=$SWEEP_DIR"
echo "MODEL=$MODEL"
echo "CONCURRENCIES=$CONCURRENCIES"
echo "MIN_PROMPTS=$MIN_PROMPTS"
echo "WAVES_PER_POINT=$WAVES_PER_POINT"
echo

echo "Resolving exact Hugging Face revision..."

MODEL_REVISION="$(
  python vllm/scripts/06_resolve_model_revision.py "$MODEL"
)"

export MODEL_REVISION

echo "$MODEL_REVISION" > "$SWEEP_DIR/model-revision.txt"

echo "MODEL_REVISION=$MODEL_REVISION"
echo

cat > "$SWEEP_DIR/sweep-config.txt" <<EOF
experiment=002_concurrency_saturation
model=$MODEL
model_revision=$MODEL_REVISION
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
min_prompts=$MIN_PROMPTS
waves_per_point=$WAVES_PER_POINT
request_rate=inf
seed=$SEED
gpu_memory_utilization=$GPU_UTIL
max_model_len=$MAX_MODEL_LEN
prefix_caching=off
concurrencies=$CONCURRENCIES
EOF

echo "Step 1: Recording environment..."

RUN_ID="$SWEEP_ID" \
RUN_DIR="$SWEEP_DIR" \
bash vllm/scripts/00_check_env.sh

echo
echo "Step 2: Starting one persistent vLLM server..."

RUN_ID="$SWEEP_ID" \
RUN_DIR="$SWEEP_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
GPU_UTIL="$GPU_UTIL" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MODEL_REVISION="$MODEL_REVISION" \
PREFIX_CACHING=off \
setsid bash vllm/scripts/01_serve_baseline.sh &

SERVER_PID=$!

echo "$SERVER_PID" > "$SWEEP_DIR/server.pid"

echo
echo "Waiting for server..."

READY=0

for _ in $(seq 1 180); do
  if curl -fsS \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null 2>&1; then
    READY=1
    break
  fi

  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "Server did not become healthy."
  exit 1
fi

echo "Server healthy."

echo
echo "Step 3: Server warmup..."

WARMUP_DIR="$SWEEP_DIR/_warmup"

RUN_ID="${SWEEP_ID}_warmup" \
RUN_DIR="$WARMUP_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
NUM_PROMPTS=8 \
NUM_WARMUPS=0 \
REQUEST_RATE=inf \
MAX_CONCURRENCY=4 \
INPUT_LEN="$INPUT_LEN" \
OUTPUT_LEN="$OUTPUT_LEN" \
SEED="$SEED" \
bash vllm/scripts/02_bench_random_baseline.sh

echo
echo "Warmup complete."
echo
echo "Step 4: Running concurrency sweep..."

for C in $CONCURRENCIES; do
  PADDED_C="$(printf '%03d' "$C")"

  POINT_ID="${SWEEP_ID}_c${PADDED_C}"
  POINT_DIR="$SWEEP_DIR/concurrency_${PADDED_C}"

  POINT_PROMPTS=$((C * WAVES_PER_POINT))

  if (( POINT_PROMPTS < MIN_PROMPTS )); then
    POINT_PROMPTS="$MIN_PROMPTS"
  fi

  mkdir -p "$POINT_DIR"

  cp "$SWEEP_DIR/server-config.txt" \
     "$POINT_DIR/server-config.txt"

  cp "$SWEEP_DIR/server-command.txt" \
     "$POINT_DIR/server-command.txt"

  cp "$SWEEP_DIR/model-revision.txt" \
     "$POINT_DIR/model-revision.txt"

  cat > "$POINT_DIR/point-meta.txt" <<EOF
sweep_id=$SWEEP_ID
point_id=$POINT_ID
max_concurrency=$C
num_prompts=$POINT_PROMPTS
target_concurrency_waves=$WAVES_PER_POINT
request_rate=inf
seed=$SEED
prefix_caching=off
EOF

  echo
  echo "----------------------------------------"
  echo "Concurrency $C"
  echo "Prompts: $POINT_PROMPTS"
  echo "----------------------------------------"

  RUN_DIR="$POINT_DIR" \
  bash vllm/scripts/04_collect_gpu_metrics.sh &

  TELEMETRY_PID=$!

  date -u +"%Y-%m-%dT%H:%M:%S.%NZ" \
    > "$POINT_DIR/benchmark-started-at-utc.txt"

  set +e

  RUN_ID="$POINT_ID" \
  RUN_DIR="$POINT_DIR" \
  MODEL="$MODEL" \
  PORT="$PORT" \
  NUM_PROMPTS="$POINT_PROMPTS" \
  NUM_WARMUPS=0 \
  REQUEST_RATE=inf \
  MAX_CONCURRENCY="$C" \
  INPUT_LEN="$INPUT_LEN" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  SEED="$SEED" \
  bash vllm/scripts/02_bench_random_baseline.sh

  BENCH_STATUS=$?

  set -e

  date -u +"%Y-%m-%dT%H:%M:%S.%NZ" \
    > "$POINT_DIR/benchmark-finished-at-utc.txt"

  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
  TELEMETRY_PID=""

  if [[ "$BENCH_STATUS" -ne 0 ]]; then
    echo "Benchmark failed at concurrency $C"
    exit "$BENCH_STATUS"
  fi

  echo "Concurrency $C complete."
done

echo
echo "========================================"
echo "Concurrency sweep complete"
echo "========================================"
echo "SWEEP_ID=$SWEEP_ID"
echo "Results: $SWEEP_DIR"