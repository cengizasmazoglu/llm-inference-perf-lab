#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def quantile_linear(values, q):
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    fraction = pos - lo
    return values[lo] + (values[hi] - values[lo]) * fraction


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "point_dir",
        type=Path,
    )

    args = parser.parse_args()

    point = args.point_dir

    benchmark_path = point / "benchmark.json"
    telemetry_path = point / "gpu-metrics.csv"

    benchmark = json.loads(
        benchmark_path.read_text()
    )

    with telemetry_path.open(newline="") as f:
        reader = csv.DictReader(f)

        if (
            not reader.fieldnames
            or "perf_counter_s" not in reader.fieldnames
        ):
            raise RuntimeError(
                "Telemetry does not contain perf_counter_s."
            )

        fieldnames = reader.fieldnames
        telemetry = list(reader)

    starts = [
        float(x)
        for x in benchmark["start_times"]
    ]

    if not starts:
        raise RuntimeError(
            "benchmark.json contains no start_times."
        )

    benchmark_start = min(starts)

    benchmark_duration = float(
        benchmark["duration"]
    )

    benchmark_end = (
        benchmark_start
        + benchmark_duration
    )

    active = [
        row
        for row in telemetry
        if (
            float(row["perf_counter_s"])
            >= benchmark_start
            and float(row["perf_counter_s"])
            <= benchmark_end
        )
    ]

    if not active:
        raise RuntimeError(
            "No telemetry samples fall inside "
            "the benchmark window."
        )

    output_csv = (
        point
        / "gpu-metrics-benchmark.csv"
    )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(active)

    def column(name):
        return [
            float(row[name])
            for row in active
        ]

    gpu_util = column(
        "utilization_gpu_pct"
    )
    memory_activity = column(
        "utilization_memory_pct"
    )
    vram = column(
        "memory_used_mib"
    )
    power = column(
        "power_draw_w"
    )
    temperature = column(
        "temperature_c"
    )
    sm_clock = column(
        "sm_clock_mhz"
    )

    summary = {
        "alignment_clock":
            "time.perf_counter",

        "benchmark_start_perf_counter_s":
            benchmark_start,

        "benchmark_end_perf_counter_s":
            benchmark_end,

        "benchmark_duration_s":
            benchmark_duration,

        "telemetry_samples_total":
            len(telemetry),

        "telemetry_samples_benchmark":
            len(active),

        "gpu_util_mean_pct":
            mean(gpu_util),

        "gpu_util_p95_pct":
            quantile_linear(
                gpu_util,
                0.95,
            ),

        "gpu_util_max_pct":
            max(gpu_util),

        "memory_activity_mean_pct":
            mean(memory_activity),

        "vram_mean_mib":
            mean(vram),

        "vram_max_mib":
            max(vram),

        "power_mean_w":
            mean(power),

        "power_max_w":
            max(power),

        "temperature_max_c":
            max(temperature),

        "sm_clock_mean_mhz":
            mean(sm_clock),
    }

    summary_path = (
        point
        / "gpu-benchmark-summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"Benchmark-aligned telemetry: "
        f"{len(active)}/{len(telemetry)} samples"
    )

    print(
        f"GPU util mean: "
        f"{summary['gpu_util_mean_pct']:.2f}%"
    )

    print(
        f"Saved: {output_csv}"
    )

    print(
        f"Saved: {summary_path}"
    )


if __name__ == "__main__":
    main()
