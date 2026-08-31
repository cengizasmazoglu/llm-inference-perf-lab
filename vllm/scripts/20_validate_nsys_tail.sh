#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: 20_validate_nsys_tail.sh RUN_DIR}"
DB="$RUN_DIR/server_full.sqlite"
WINDOW="$RUN_DIR/post_canary_window.tsv"
OUT="$RUN_DIR/trace_integrity.txt"

[[ -f "$DB" ]] || { echo "TRACE_INTEGRITY_FAILED: missing sqlite"; exit 2; }
[[ -f "$WINDOW" ]] || { echo "TRACE_INTEGRITY_FAILED: missing canary window"; exit 2; }

CANARY_START_NS="$(awk -F '\t' '$1=="START_NS"{print $2}' "$WINDOW")"
SESSION_NS="$(sqlite3 "$DB" 'SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1;')"

GPU_PID="$(sqlite3 "$DB" 'SELECT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid ORDER BY COUNT(*) DESC LIMIT 1;')"
LAST_CUDA_REL_NS="$(sqlite3 "$DB" "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=$GPU_PID;")"

LAST_CUDA_EPOCH_NS=$((SESSION_NS + LAST_CUDA_REL_NS))
DELTA_NS=$((LAST_CUDA_EPOCH_NS - CANARY_START_NS))

{
    echo "gpu_global_pid=$GPU_PID"
    echo "canary_start_epoch_ns=$CANARY_START_NS"
    echo "last_cuda_epoch_ns=$LAST_CUDA_EPOCH_NS"
    echo "last_cuda_minus_canary_start_ns=$DELTA_NS"
} | tee "$OUT"

if (( DELTA_NS >= 0 )); then
    echo "TRACE_INTEGRITY_OK" | tee -a "$OUT"
    exit 0
fi

echo "TRACE_INTEGRITY_FAILED" | tee -a "$OUT"
exit 2
