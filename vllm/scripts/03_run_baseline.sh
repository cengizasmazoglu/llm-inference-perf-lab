#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

RUN_ID="${RUN_ID:-001_vllm_baseline_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/${RUN_ID}}"

export MODEL
export PORT
export GPU_UTIL
export MAX_MODEL_LEN
export RUN_ID
export RUN_DIR

mkdir -p "$RUN_DIR"

echo "========================================"
echo "vLLM Baseline Experiment"
echo "========================================"
echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "MODEL=$MODEL"
echo

echo "Resolving exact Hugging Face model revision..."

MODEL_REVISION="$(
  python vllm/scripts/06_resolve_model_revision.py "$MODEL"
)"

export MODEL_REVISION

echo "MODEL_REVISION=$MODEL_REVISION"

printf '%s\n' "$MODEL_REVISION" \
  > "$RUN_DIR/model-revision.txt"

echo
echo "Step 1/5: Collecting environment..."

bash vllm/scripts/00_check_env.sh


cleanup() {
  local exit_code=$?

  trap - EXIT INT TERM

  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
      echo
      echo "Stopping GPU telemetry..."
      kill "$TELEMETRY_PID" 2>/dev/null || true
      wait "$TELEMETRY_PID" 2>/dev/null || true
    fi
  fi

  if [[ -n "${SERVER_PID:-}" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo
      echo "Stopping vLLM server..."
      kill -- "-$SERVER_PID" 2>/dev/null || true
    fi
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM


echo
echo "Step 2/5: Starting vLLM server..."

setsid bash vllm/scripts/01_serve_baseline.sh &
SERVER_PID=$!

echo "$SERVER_PID" > "$RUN_DIR/server.pid"

echo "Waiting for vLLM health endpoint..."

READY=0
DEADLINE=$((SECONDS + HEALTH_TIMEOUT))

while (( SECONDS < DEADLINE )); do

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "ERROR: vLLM server exited before becoming healthy."
    tail -50 "$RUN_DIR/server.log" 2>/dev/null || true
    exit 1
  fi

  if curl -fsS \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null 2>&1; then

    READY=1
    break
  fi

  sleep 2
done


if [[ "$READY" -ne 1 ]]; then
  echo
  echo "ERROR: vLLM server did not become healthy."
  tail -50 "$RUN_DIR/server.log" 2>/dev/null || true
  exit 1
fi

echo "vLLM is healthy."


echo
echo "Step 3/5: Starting GPU telemetry..."

bash vllm/scripts/04_collect_gpu_metrics.sh &
TELEMETRY_PID=$!

echo "$TELEMETRY_PID" > "$RUN_DIR/gpu-telemetry.pid"

sleep 2


echo
echo "Step 4/5: Running benchmark..."

bash vllm/scripts/02_bench_random_baseline.sh


echo
echo "Stopping GPU telemetry..."

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
TELEMETRY_PID=""


echo
echo "Step 5/5: Creating experiment manifest..."

python vllm/scripts/05_make_manifest.py "$RUN_DIR"


echo
echo "========================================"
echo "Experiment complete"
echo "========================================"
echo "RUN_ID=$RUN_ID"
echo "Results: $RUN_DIR"