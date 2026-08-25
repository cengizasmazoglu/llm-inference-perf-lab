#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-0.5}"

mkdir -p "$RUN_DIR"

OUTPUT="$RUN_DIR/gpu-metrics.csv"

cat > "$RUN_DIR/gpu-telemetry-config.txt" <<EOF
collector=nvidia-smi
sample_interval_s=$SAMPLE_INTERVAL_S
clock=time.perf_counter
output=$OUTPUT
EOF

echo "Collecting GPU telemetry..."
echo "Interval: ${SAMPLE_INTERVAL_S}s"
echo "Output: $OUTPUT"

exec python - "$OUTPUT" "$SAMPLE_INTERVAL_S" <<'PY'
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone


output_path = sys.argv[1]
interval = float(sys.argv[2])

query = [
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
]

command = [
    "nvidia-smi",
    "--query-gpu=" + ",".join(query),
    "--format=csv,noheader,nounits",
]

header = [
    "timestamp",
    "perf_counter_s",
    "index",
    "name",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
]

with open(output_path, "w", newline="", buffering=1) as f:
    writer = csv.writer(f)
    writer.writerow(header)

    while True:
        sample_clock = time.perf_counter()

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        for line in result.stdout.strip().splitlines():
            fields = [
                value.strip()
                for value in line.split(",")
            ]

            writer.writerow(
                [
                    timestamp,
                    f"{sample_clock:.9f}",
                    *fields,
                ]
            )

        f.flush()

        elapsed = time.perf_counter() - sample_clock
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)
PY