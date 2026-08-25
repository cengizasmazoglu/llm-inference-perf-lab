#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


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

    telemetry = pd.read_csv(
        telemetry_path
    )

    if "perf_counter_s" not in telemetry:
        raise RuntimeError(
            "Telemetry does not contain perf_counter_s."
        )

    starts = [
        float(x)
        for x in benchmark["start_times"]
    ]

    if not starts:
        raise RuntimeError(
            "benchmark.json contains no start_times."
        )

    #
    # vLLM request start_times use time.perf_counter().
    # Our telemetry collector now uses the same clock.
    #
    # The main benchmark begins immediately before requests
    # are scheduled, so min(start_times) provides the
    # request-active start on the shared monotonic clock.
    #
    benchmark_start = min(starts)

    benchmark_duration = float(
        benchmark["duration"]
    )

    benchmark_end = (
        benchmark_start
        + benchmark_duration
    )

    active = telemetry[
        (
            telemetry["perf_counter_s"]
            >= benchmark_start
        )
        & (
            telemetry["perf_counter_s"]
            <= benchmark_end
        )
    ].copy()

    if active.empty:
        raise RuntimeError(
            "No telemetry samples fall inside "
            "the benchmark window."
        )

    output_csv = (
        point
        / "gpu-metrics-benchmark.csv"
    )

    active.to_csv(
        output_csv,
        index=False,
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
            int(len(telemetry)),

        "telemetry_samples_benchmark":
            int(len(active)),

        "gpu_util_mean_pct":
            float(
                active[
                    "utilization_gpu_pct"
                ].mean()
            ),

        "gpu_util_p95_pct":
            float(
                active[
                    "utilization_gpu_pct"
                ].quantile(0.95)
            ),

        "gpu_util_max_pct":
            float(
                active[
                    "utilization_gpu_pct"
                ].max()
            ),

        "memory_activity_mean_pct":
            float(
                active[
                    "utilization_memory_pct"
                ].mean()
            ),

        "vram_mean_mib":
            float(
                active[
                    "memory_used_mib"
                ].mean()
            ),

        "vram_max_mib":
            float(
                active[
                    "memory_used_mib"
                ].max()
            ),

        "power_mean_w":
            float(
                active[
                    "power_draw_w"
                ].mean()
            ),

        "power_max_w":
            float(
                active[
                    "power_draw_w"
                ].max()
            ),

        "temperature_max_c":
            float(
                active[
                    "temperature_c"
                ].max()
            ),

        "sm_clock_mean_mhz":
            float(
                active[
                    "sm_clock_mhz"
                ].mean()
            ),
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