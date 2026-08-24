#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/${RUN_ID}}"

INTERVAL="${GPU_TELEMETRY_INTERVAL:-1}"

mkdir -p "$RUN_DIR"

OUT="$RUN_DIR/gpu-metrics.csv"

cat > "$RUN_DIR/gpu-telemetry-config.txt" <<EOF
INTERVAL_SECONDS=$INTERVAL
FIELDS=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem
EOF

echo "timestamp,index,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_c,sm_clock_mhz,memory_clock_mhz" \
  > "$OUT"

while true; do

  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    >> "$OUT"

  sleep "$INTERVAL"

done