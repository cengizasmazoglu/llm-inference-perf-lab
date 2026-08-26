#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

WAVES_PER_POINT="${WAVES_PER_POINT:-8}"
SEED="${SEED:-0}"

GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-420}"
POINT_COOLDOWN_S="${POINT_COOLDOWN_S:-2}"

EXPERIMENT_ID="${EXPERIMENT_ID:-003_saturation_repeatability_$(date -u +%Y%m%dT%H%M%SZ)}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-vllm/results/raw/$EXPERIMENT_ID}"

SERVER_PID=""
TELEMETRY_PID=""

if [[ -n "${EXPORT_DIR:-}" ]]; then
  FINAL_EXPORT_DIR="$EXPORT_DIR"
elif [[ -n "${RUNPOD_POD_ID:-}" && -d /workspace ]]; then
  FINAL_EXPORT_DIR="/workspace"
else
  FINAL_EXPORT_DIR=""
fi

mkdir -p "$EXPERIMENT_DIR"

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
echo "Experiment 003"
echo "Saturation Repeatability"
echo "========================================"
echo "EXPERIMENT_ID=$EXPERIMENT_ID"
echo "MODEL=$MODEL"
echo "WAVES_PER_POINT=$WAVES_PER_POINT"
echo "SEED=$SEED"
echo

echo "Resolving exact Hugging Face revision..."

MODEL_REVISION="$(
  python vllm/scripts/06_resolve_model_revision.py "$MODEL"
)"

export MODEL_REVISION

echo "$MODEL_REVISION" \
  > "$EXPERIMENT_DIR/model-revision.txt"

cat > "$EXPERIMENT_DIR/experiment-config.txt" <<EOF
experiment=003_saturation_repeatability
model=$MODEL
model_revision=$MODEL_REVISION
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
waves_per_point=$WAVES_PER_POINT
request_rate=inf
seed=$SEED
gpu_memory_utilization=$GPU_UTIL
max_model_len=$MAX_MODEL_LEN
prefix_caching=off
server_ready_timeout_s=$SERVER_READY_TIMEOUT_S
point_cooldown_s=$POINT_COOLDOWN_S
replicate_1_order=64 128 256
replicate_2_order=128 256 64
replicate_3_order=256 64 128
EOF

echo
echo "Step 1: Recording environment..."

RUN_ID="$EXPERIMENT_ID" \
RUN_DIR="$EXPERIMENT_DIR" \
bash vllm/scripts/00_check_env.sh

echo
echo "Step 2: Starting persistent vLLM server..."

RUN_ID="$EXPERIMENT_ID" \
RUN_DIR="$EXPERIMENT_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
GPU_UTIL="$GPU_UTIL" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MODEL_REVISION="$MODEL_REVISION" \
PREFIX_CACHING=off \
setsid bash vllm/scripts/01_serve_baseline.sh &

SERVER_PID=$!

echo "$SERVER_PID" \
  > "$EXPERIMENT_DIR/server.pid"

echo
echo "Waiting for server health..."

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
echo "Step 3: Warmup..."

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

echo
echo "Warmup complete."

run_point() {
  local REP="$1"
  local POSITION="$2"
  local C="$3"
  local ORDER="$4"

  local PADDED_REP
  local PADDED_C
  local POINT_PROMPTS
  local REP_DIR
  local POINT_DIR
  local POINT_ID
  local BENCH_STATUS

  PADDED_REP="$(printf '%02d' "$REP")"
  PADDED_C="$(printf '%03d' "$C")"

  POINT_PROMPTS=$((C * WAVES_PER_POINT))

  REP_DIR="$EXPERIMENT_DIR/rep_${PADDED_REP}"
  POINT_DIR="$REP_DIR/concurrency_${PADDED_C}"

POINT_ID="${EXPERIMENT_ID}_r${PADDED_REP}_c${PADDED_C}"

  mkdir -p "$POINT_DIR"

  cat > "$POINT_DIR/point-meta.txt" <<EOF
experiment_id=$EXPERIMENT_ID
point_id=$POINT_ID
replicate=$REP
sequence_position=$POSITION
replicate_order=$ORDER
max_concurrency=$C
num_prompts=$POINT_PROMPTS
waves_per_point=$WAVES_PER_POINT
request_rate=inf
seed=$SEED
prefix_caching=off
EOF

  echo
  echo "----------------------------------------"
  echo "Replicate $REP / position $POSITION"
  echo "Concurrency: $C"
  echo "Prompts:     $POINT_PROMPTS"
  echo "----------------------------------------"

  if ! curl -fsS \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null; then

    echo "ERROR: server unhealthy before point."
    exit 1
  fi

  RUN_DIR="$POINT_DIR" \
  SAMPLE_INTERVAL_S=0.5 \
  bash vllm/scripts/04_collect_gpu_metrics.sh &

  TELEMETRY_PID=$!

  sleep 0.25

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

  kill "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
  TELEMETRY_PID=""

  if [[ "$BENCH_STATUS" -ne 0 ]]; then
    echo "ERROR: benchmark failed."
    exit "$BENCH_STATUS"
  fi

  python \
    vllm/scripts/09_extract_benchmark_gpu_window.py \
    "$POINT_DIR"

  echo "Point complete."

  sleep "$POINT_COOLDOWN_S"
}

echo
echo "Step 4: Balanced repeatability experiment..."

for REP in 1 2 3; do

  case "$REP" in
    1)
      ORDER="64 128 256"
      ;;
    2)
      ORDER="128 256 64"
      ;;
    3)
      ORDER="256 64 128"
      ;;
  esac

  POSITION=0

  for C in $ORDER; do
    POSITION=$((POSITION + 1))

    run_point \
      "$REP" \
      "$POSITION" \
      "$C" \
      "$ORDER"
  done
done

echo
echo "Step 5: Validating experiment..."

EXPECTED_POINTS=9

BENCHMARK_COUNT="$(
  find "$EXPERIMENT_DIR" \
    -type f \
    -path '*/rep_*/concurrency_*/benchmark.json' \
    | wc -l
)"

GPU_SUMMARY_COUNT="$(
  find "$EXPERIMENT_DIR" \
    -type f \
    -path '*/rep_*/concurrency_*/gpu-benchmark-summary.json' \
    | wc -l
)"

echo "Expected benchmark points: $EXPECTED_POINTS"
echo "Benchmark files:          $BENCHMARK_COUNT"
echo "GPU summaries:            $GPU_SUMMARY_COUNT"

if [[ "$BENCHMARK_COUNT" -ne "$EXPECTED_POINTS" ]]; then
  echo "ERROR: incomplete benchmark set."
  exit 1
fi

if [[ "$GPU_SUMMARY_COUNT" -ne "$EXPECTED_POINTS" ]]; then
  echo "ERROR: incomplete telemetry set."
  exit 1
fi

echo
echo "Creating experiment inventory..."

{
  echo "experiment_id=$EXPERIMENT_ID"
  echo "expected_points=$EXPECTED_POINTS"
  echo "benchmark_files=$BENCHMARK_COUNT"
  echo "gpu_summaries=$GPU_SUMMARY_COUNT"
  echo

  echo "benchmark_artifacts:"

  find "$EXPERIMENT_DIR" \
    -type f \
    -path '*/rep_*/concurrency_*/benchmark.json' \
    | sort

  echo
  echo "gpu_summary_artifacts:"

  find "$EXPERIMENT_DIR" \
    -type f \
    -path '*/rep_*/concurrency_*/gpu-benchmark-summary.json' \
    | sort

} > "$EXPERIMENT_DIR/experiment-inventory.txt"

echo
echo "Step 6: Packaging..."

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
echo "EXPERIMENT COMPLETE"
echo "========================================"
echo
echo "Experiment:"
echo "$EXPERIMENT_DIR"
echo
echo "Archive:"
echo "$ARCHIVE"
echo
echo "Checksum:"
echo "$CHECKSUM"

if [[ -n "$FINAL_EXPORT_DIR" ]]; then
  echo
  echo "Download:"
  echo "$FINAL_EXPORT_DIR/$ARCHIVE"
  echo "$FINAL_EXPORT_DIR/$CHECKSUM"
fi

echo
echo "SAFE TO DOWNLOAD ARTIFACTS AND TERMINATE POD."