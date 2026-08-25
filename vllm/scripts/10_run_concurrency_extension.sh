#!/usr/bin/env bash
set -euo pipefail

CONCURRENCIES="${CONCURRENCIES:-64 128 256 512}"
MIN_PROMPTS="${MIN_PROMPTS:-128}"
WAVES_PER_POINT="${WAVES_PER_POINT:-8}"

SWEEP_ID="${SWEEP_ID:-002_concurrency_saturation_hi_$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_DIR="${SWEEP_DIR:-vllm/results/raw/$SWEEP_ID}"

if [[ -n "${EXPORT_DIR:-}" ]]; then
  FINAL_EXPORT_DIR="$EXPORT_DIR"
elif [[ -n "${RUNPOD_POD_ID:-}" && -d /workspace ]]; then
  FINAL_EXPORT_DIR="/workspace"
else
  FINAL_EXPORT_DIR=""
fi

export CONCURRENCIES
export MIN_PROMPTS
export WAVES_PER_POINT
export SWEEP_ID
export SWEEP_DIR

echo "========================================"
echo "Experiment 002B"
echo "High-Concurrency Saturation Extension"
echo "========================================"
echo "SWEEP_ID=$SWEEP_ID"
echo "CONCURRENCIES=$CONCURRENCIES"
echo "WAVES_PER_POINT=$WAVES_PER_POINT"
echo

echo "Running sweep..."

bash vllm/scripts/08_run_concurrency_sweep.sh

echo
echo "Generating benchmark-aligned GPU telemetry..."

for POINT_DIR in "$SWEEP_DIR"/concurrency_*; do
  if [[ ! -f "$POINT_DIR/benchmark.json" ]]; then
    continue
  fi

  echo
  echo "Aligning:"
  echo "$POINT_DIR"

  python \
    vllm/scripts/09_extract_benchmark_gpu_window.py \
    "$POINT_DIR"
done

echo
echo "Validating benchmark count..."

EXPECTED_COUNT="$(
  wc -w <<< "$CONCURRENCIES"
)"

ACTUAL_COUNT="$(
  find "$SWEEP_DIR" \
    -mindepth 2 \
    -maxdepth 2 \
    -path '*/concurrency_*/benchmark.json' \
    -type f \
    | wc -l
)"

echo "Expected: $EXPECTED_COUNT"
echo "Actual:   $ACTUAL_COUNT"

if [[ "$ACTUAL_COUNT" -ne "$EXPECTED_COUNT" ]]; then
  echo
  echo "ERROR: incomplete sweep."
  exit 1
fi

echo
echo "Creating sweep inventory..."

{
  echo "sweep_id=$SWEEP_ID"
  echo "concurrencies=$CONCURRENCIES"
  echo "expected_points=$EXPECTED_COUNT"
  echo "completed_points=$ACTUAL_COUNT"
  echo
  echo "benchmark_files:"

  find "$SWEEP_DIR" \
    -path '*/concurrency_*/benchmark.json' \
    -type f \
    | sort

  echo
  echo "aligned_gpu_summaries:"

  find "$SWEEP_DIR" \
    -path '*/concurrency_*/gpu-benchmark-summary.json' \
    -type f \
    | sort

} > "$SWEEP_DIR/sweep-inventory.txt"

echo
echo "Packaging experiment..."

bash vllm/scripts/07_pack_run.sh \
  "$SWEEP_DIR"

ARCHIVE="$(basename "$SWEEP_DIR").tar.gz"
CHECKSUM="${ARCHIVE}.sha256"

if [[ -n "$FINAL_EXPORT_DIR" ]]; then
  echo
  echo "Copying downloadable artifacts to:"
  echo "$FINAL_EXPORT_DIR"

  cp "$ARCHIVE" "$CHECKSUM" \
    "$FINAL_EXPORT_DIR/"
fi

echo
echo "========================================"
echo "EXPERIMENT COMPLETE"
echo "========================================"
echo
echo "Sweep:"
echo "$SWEEP_DIR"
echo
echo "Archive:"
echo "$ARCHIVE"
echo
echo "Checksum:"
echo "$CHECKSUM"

if [[ -n "$FINAL_EXPORT_DIR" ]]; then
  echo
  echo "Download from:"
  echo "$FINAL_EXPORT_DIR/$ARCHIVE"
  echo "$FINAL_EXPORT_DIR/$CHECKSUM"
fi

echo
echo "SAFE TO DOWNLOAD ARTIFACTS AND TERMINATE POD."
echo