#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEED="${SEED:-0}"

GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

CONCURRENCIES="${CONCURRENCIES:-64 128 256}"
WAVES_PER_POINT="${WAVES_PER_POINT:-8}"

PROFILE_DELAY_S="${PROFILE_DELAY_S:-2}"
PROFILE_DURATION_S="${PROFILE_DURATION_S:-2}"

SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-420}"
FINALIZE_TIMEOUT_S="${FINALIZE_TIMEOUT_S:-180}"

EXPERIMENT_ID="${EXPERIMENT_ID:-004_nsys_saturation_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-vllm/results/raw/$EXPERIMENT_ID}"
NSYS_DIR="$EXPERIMENT_DIR/nsys"

NSYS_PID=""
BENCH_PID=""

if [[ -n "${EXPORT_DIR:-}" ]]; then
  FINAL_EXPORT_DIR="$EXPORT_DIR"
elif [[ -n "${RUNPOD_POD_ID:-}" && -d /workspace ]]; then
  FINAL_EXPORT_DIR="/workspace"
else
  FINAL_EXPORT_DIR=""
fi

mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$NSYS_DIR"

cleanup() {
  set +e

  if [[ -n "${BENCH_PID:-}" ]]; then
    kill "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true
  fi

  if [[ -n "${NSYS_PID:-}" ]]; then
    if kill -0 "$NSYS_PID" 2>/dev/null; then
      kill -INT "$NSYS_PID" 2>/dev/null || true
      wait "$NSYS_PID" 2>/dev/null || true
    fi
  fi
}

trap cleanup EXIT INT TERM

echo "========================================"
echo "Experiment 004"
echo "Nsight Systems Saturation Comparison"
echo "========================================"
echo "EXPERIMENT_ID=$EXPERIMENT_ID"
echo "CONCURRENCIES=$CONCURRENCIES"
echo "PROFILE_DURATION_S=$PROFILE_DURATION_S"
echo

echo "Checking Nsight Systems..."

if ! command -v nsys >/dev/null 2>&1; then
  echo
  echo "ERROR: nsys is not installed."
  echo "Do not continue profiling."
  exit 2
fi

nsys --version \
  | tee "$EXPERIMENT_DIR/nsys-version.txt"

echo
echo "Resolving exact model revision..."

MODEL_REVISION="$(
  python vllm/scripts/06_resolve_model_revision.py "$MODEL"
)"

export MODEL_REVISION

echo "$MODEL_REVISION" \
  > "$EXPERIMENT_DIR/model-revision.txt"

cat > "$EXPERIMENT_DIR/experiment-config.txt" <<EOF
experiment=004_nsys_saturation
model=$MODEL
model_revision=$MODEL_REVISION
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
seed=$SEED
gpu_memory_utilization=$GPU_UTIL
max_model_len=$MAX_MODEL_LEN
prefix_caching=off
concurrencies=$CONCURRENCIES
waves_per_point=$WAVES_PER_POINT
profile_delay_s=$PROFILE_DELAY_S
profile_duration_s=$PROFILE_DURATION_S
profiler=nsight_systems
EOF

echo
echo "Recording environment..."

RUN_ID="$EXPERIMENT_ID" \
RUN_DIR="$EXPERIMENT_DIR" \
bash vllm/scripts/00_check_env.sh

echo
echo "Starting vLLM under Nsight Systems..."

export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_ID="$EXPERIMENT_ID" \
RUN_DIR="$EXPERIMENT_DIR" \
MODEL="$MODEL" \
PORT="$PORT" \
GPU_UTIL="$GPU_UTIL" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MODEL_REVISION="$MODEL_REVISION" \
PREFIX_CACHING=off \
PROFILER=cuda \
setsid nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:3:defer \
  --kill=sigterm \
  --output="$NSYS_DIR/vllm_profile" \
  bash vllm/scripts/01_serve_baseline.sh \
  > "$EXPERIMENT_DIR/nsys-launch.log" 2>&1 &

NSYS_PID=$!

echo "$NSYS_PID" \
  > "$EXPERIMENT_DIR/nsys.pid"

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

  if ! kill -0 "$NSYS_PID" 2>/dev/null; then
    echo "ERROR: Nsight/vLLM process exited during startup."
    tail -n 100 "$EXPERIMENT_DIR/nsys-launch.log" || true
    exit 1
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

echo
echo "Warmup complete."

CAPTURE_INDEX=0

for C in $CONCURRENCIES; do
  CAPTURE_INDEX=$((CAPTURE_INDEX + 1))

  PADDED_C="$(printf '%03d' "$C")"

  POINT_ID="${EXPERIMENT_ID}_c${PADDED_C}"
  POINT_DIR="$EXPERIMENT_DIR/concurrency_${PADDED_C}"

  NUM_PROMPTS=$((C * WAVES_PER_POINT))

  mkdir -p "$POINT_DIR"

  cat > "$POINT_DIR/point-meta.txt" <<EOF
experiment_id=$EXPERIMENT_ID
capture_index=$CAPTURE_INDEX
max_concurrency=$C
num_prompts=$NUM_PROMPTS
profile_delay_s=$PROFILE_DELAY_S
profile_duration_s=$PROFILE_DURATION_S
profiling_run=true
clean_performance_claim=false
EOF

  echo
  echo "----------------------------------------"
  echo "Profile capture $CAPTURE_INDEX"
  echo "Concurrency: $C"
  echo "Prompts:     $NUM_PROMPTS"
  echo "----------------------------------------"

  if ! curl -fsS \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null; then

    echo "ERROR: server unhealthy before profile point."
    exit 1
  fi

  set +e

  RUN_ID="$POINT_ID" \
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

  set -e

  sleep "$PROFILE_DELAY_S"

  if ! kill -0 "$BENCH_PID" 2>/dev/null; then
    echo "ERROR: benchmark ended before profile capture started."
    wait "$BENCH_PID" || true
    BENCH_PID=""
    exit 1
  fi

  echo "Starting Nsight capture..."

  curl -fsS \
    -X POST \
    "http://127.0.0.1:${PORT}/start_profile" \
    > "$POINT_DIR/start-profile-response.txt"

  sleep "$PROFILE_DURATION_S"

  echo "Stopping Nsight capture..."

  curl -fsS \
    -X POST \
    "http://127.0.0.1:${PORT}/stop_profile" \
    > "$POINT_DIR/stop-profile-response.txt"

  set +e
  wait "$BENCH_PID"
  BENCH_STATUS=$?
  set -e

  BENCH_PID=""

  if [[ "$BENCH_STATUS" -ne 0 ]]; then
    echo "ERROR: benchmark failed at concurrency $C."
    exit "$BENCH_STATUS"
  fi

  echo "Concurrency $C diagnostic run complete."

  sleep 2
done

echo
echo "All capture ranges complete."
echo "Stopping Nsight session and finalizing reports..."

kill -INT "$NSYS_PID" 2>/dev/null || true

FINALIZED=0

for _ in $(seq 1 "$FINALIZE_TIMEOUT_S"); do
  if ! kill -0 "$NSYS_PID" 2>/dev/null; then
    FINALIZED=1
    break
  fi

  sleep 1
done

if [[ "$FINALIZED" -ne 1 ]]; then
  echo "ERROR: Nsight did not finalize within timeout."
  exit 1
fi

wait "$NSYS_PID" 2>/dev/null || true
NSYS_PID=""

echo
echo "Checking Nsight reports..."

mapfile -t REPORTS < <(
  find "$NSYS_DIR" \
    -maxdepth 1 \
    -type f \
    -name '*.nsys-rep' \
    | sort -V
)

EXPECTED_REPORTS=3
ACTUAL_REPORTS="${#REPORTS[@]}"

echo "Expected reports: $EXPECTED_REPORTS"
echo "Actual reports:   $ACTUAL_REPORTS"

if [[ "$ACTUAL_REPORTS" -ne "$EXPECTED_REPORTS" ]]; then
  echo "ERROR: expected three Nsight reports."
  find "$NSYS_DIR" -maxdepth 1 -type f -print
  exit 1
fi

echo
echo "Creating profile mapping and summaries..."

CONCURRENCY_ARRAY=(64 128 256)

{
  printf "capture_index\tconcurrency\treport\n"

  for INDEX in "${!REPORTS[@]}"; do
    REPORT="${REPORTS[$INDEX]}"
    C="${CONCURRENCY_ARRAY[$INDEX]}"

    printf "%d\t%d\t%s\n" \
      "$((INDEX + 1))" \
      "$C" \
      "$REPORT"
  done
} > "$EXPERIMENT_DIR/profile-map.tsv"

VALID_CUDA_REPORTS=0

for REPORT in "${REPORTS[@]}"; do
  STEM="${REPORT%.nsys-rep}"
  STATS="${STEM}_stats.txt"

  echo
  echo "Generating stats:"
  echo "$REPORT"

  nsys stats \
    --report cuda_gpu_kern_sum \
    --report cuda_api_sum \
    "$REPORT" \
    > "$STATS" 2>&1

  cat "$STATS"

  if grep -qi \
    'does not contain CUDA kernel data' \
    "$STATS"; then

    echo
    echo "ERROR: Nsight report contains no CUDA kernel data:"
    echo "$REPORT"
    exit 1
  fi

  if ! grep -q \
    'CUDA GPU Kernel Summary' \
    "$STATS"; then

    echo
    echo "ERROR: CUDA GPU Kernel Summary missing:"
    echo "$REPORT"
    exit 1
  fi

  VALID_CUDA_REPORTS=$((VALID_CUDA_REPORTS + 1))
done

echo
echo "Validated CUDA reports: $VALID_CUDA_REPORTS"

if [[ "$VALID_CUDA_REPORTS" -ne 3 ]]; then
  echo "ERROR: expected three valid CUDA traces."
  exit 1
fi

echo
echo "Validating benchmark artifacts..."

BENCHMARK_COUNT="$(
  find "$EXPERIMENT_DIR" \
    -type f \
    -path '*/concurrency_*/benchmark.json' \
    | wc -l
)"

echo "Benchmark files: $BENCHMARK_COUNT"

if [[ "$BENCHMARK_COUNT" -ne 3 ]]; then
  echo "ERROR: expected three diagnostic benchmark files."
  exit 1
fi

echo
echo "Packaging experiment..."

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
echo "IMPORTANT:"
echo "These are PROFILING runs."
echo "Do not use their throughput/latency as canonical benchmark results."
echo
echo "Profile mapping:"
cat "$EXPERIMENT_DIR/profile-map.tsv"
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