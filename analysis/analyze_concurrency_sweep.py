#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text())


def legacy_gpu_summary(point_dir: Path, benchmark: dict):
    """
    Backward-compatible approximation for Experiment 002.

    Old telemetry did not contain perf_counter_s, so reconstruct
    the benchmark window from vLLM's benchmark date + duration.

    The benchmark date has second-level precision, so this method
    is approximate and should not be used for new experiments.
    """
    path = point_dir / "gpu-metrics.csv"

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y/%m/%d %H:%M:%S.%f",
        utc=True,
    )

    end = datetime.strptime(
        benchmark["date"],
        "%Y%m%d-%H%M%S",
    ).replace(tzinfo=timezone.utc)

    start = end - timedelta(
        seconds=float(benchmark["duration"])
    )

    active = df[
        (df["timestamp"] >= start)
        & (
            df["timestamp"]
            <= end + timedelta(seconds=1)
        )
    ].copy()

    if active.empty:
        return {
            "telemetry_alignment":
                "legacy_wallclock_approx",
            "telemetry_samples": 0,
        }

    return {
        "telemetry_alignment":
            "legacy_wallclock_approx",

        "telemetry_samples":
            len(active),

        "gpu_util_mean_pct":
            active[
                "utilization_gpu_pct"
            ].mean(),

        "gpu_util_p95_pct":
            active[
                "utilization_gpu_pct"
            ].quantile(0.95),

        "gpu_util_max_pct":
            active[
                "utilization_gpu_pct"
            ].max(),

        "memory_activity_mean_pct":
            active[
                "utilization_memory_pct"
            ].mean(),

        "vram_mean_mib":
            active[
                "memory_used_mib"
            ].mean(),

        "vram_max_mib":
            active[
                "memory_used_mib"
            ].max(),

        "power_mean_w":
            active[
                "power_draw_w"
            ].mean(),

        "power_max_w":
            active[
                "power_draw_w"
            ].max(),

        "sm_clock_mean_mhz":
            active[
                "sm_clock_mhz"
            ].mean(),

        "temperature_max_c":
            active[
                "temperature_c"
            ].max(),
    }


def analyze_gpu(point_dir: Path, benchmark: dict):
    """
    New experiments should already contain a benchmark-aligned
    telemetry summary generated from the shared perf_counter clock.
    """
    summary_path = (
        point_dir
        / "gpu-benchmark-summary.json"
    )

    if summary_path.exists():
        summary = read_json(summary_path)

        return {
            "telemetry_alignment":
                summary.get(
                    "alignment_clock",
                    "time.perf_counter",
                ),

            "telemetry_samples":
                summary.get(
                    "telemetry_samples_benchmark"
                ),

            "gpu_util_mean_pct":
                summary.get(
                    "gpu_util_mean_pct"
                ),

            "gpu_util_p95_pct":
                summary.get(
                    "gpu_util_p95_pct"
                ),

            "gpu_util_max_pct":
                summary.get(
                    "gpu_util_max_pct"
                ),

            "memory_activity_mean_pct":
                summary.get(
                    "memory_activity_mean_pct"
                ),

            "vram_mean_mib":
                summary.get(
                    "vram_mean_mib"
                ),

            "vram_max_mib":
                summary.get(
                    "vram_max_mib"
                ),

            "power_mean_w":
                summary.get(
                    "power_mean_w"
                ),

            "power_max_w":
                summary.get(
                    "power_max_w"
                ),

            "sm_clock_mean_mhz":
                summary.get(
                    "sm_clock_mean_mhz"
                ),

            "temperature_max_c":
                summary.get(
                    "temperature_max_c"
                ),
        }

    return legacy_gpu_summary(
        point_dir,
        benchmark,
    )


def analyze_point(point_dir: Path):
    benchmark = read_json(
        point_dir / "benchmark.json"
    )

    concurrency = int(
        point_dir.name.split("_")[-1]
    )

    row = {
        "concurrency":
            concurrency,

        "num_prompts":
            benchmark["num_prompts"],

        "completed":
            benchmark["completed"],

        "failed":
            benchmark["failed"],

        "duration_s":
            benchmark["duration"],

        "request_throughput":
            benchmark["request_throughput"],

        "output_throughput":
            benchmark["output_throughput"],

        "total_token_throughput":
            benchmark["total_token_throughput"],

        "max_concurrent_requests":
            benchmark[
                "max_concurrent_requests"
            ],

        "mean_ttft_ms":
            benchmark["mean_ttft_ms"],

        "p50_ttft_ms":
            benchmark["p50_ttft_ms"],

        "p95_ttft_ms":
            benchmark["p95_ttft_ms"],

        "p99_ttft_ms":
            benchmark["p99_ttft_ms"],

        "mean_tpot_ms":
            benchmark["mean_tpot_ms"],

        "p95_tpot_ms":
            benchmark["p95_tpot_ms"],

        "p99_tpot_ms":
            benchmark["p99_tpot_ms"],

        "mean_e2el_ms":
            benchmark["mean_e2el_ms"],

        "p95_e2el_ms":
            benchmark["p95_e2el_ms"],

        "p99_e2el_ms":
            benchmark["p99_e2el_ms"],

        "input_tokens_mean":
            np.mean(
                benchmark["input_lens"]
            ),

        "output_tokens_mean":
            np.mean(
                benchmark["output_lens"]
            ),
    }

    row.update(
        analyze_gpu(
            point_dir,
            benchmark,
        )
    )

    return row


def add_scaling_metrics(df):
    df = (
        df.sort_values("concurrency")
        .reset_index(drop=True)
    )

    first_throughput = (
        df.loc[
            0,
            "output_throughput",
        ]
    )

    df["throughput_vs_first"] = (
        df["output_throughput"]
        / first_throughput
    )

    df["throughput_gain_pct"] = (
        df["output_throughput"]
        .pct_change()
        * 100
    )

    max_throughput = (
        df["output_throughput"].max()
    )

    df["pct_of_max_throughput"] = (
        df["output_throughput"]
        / max_throughput
        * 100
    )

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sweep_dir",
        type=Path,
    )

    args = parser.parse_args()

    point_dirs = sorted(
        p
        for p in args.sweep_dir.glob(
            "concurrency_*"
        )
        if (
            p / "benchmark.json"
        ).exists()
    )

    if not point_dirs:
        raise RuntimeError(
            "No concurrency benchmark points found."
        )

    rows = [
        analyze_point(point)
        for point in point_dirs
    ]

    df = pd.DataFrame(rows)

    df = add_scaling_metrics(df)

    columns = [
        "concurrency",
        "num_prompts",
        "request_throughput",
        "output_throughput",
        "pct_of_max_throughput",
        "throughput_gain_pct",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2el_ms",
        "gpu_util_mean_pct",
        "telemetry_samples",
        "vram_max_mib",
        "power_mean_w",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print()
    print(
        "=== CONCURRENCY SATURATION CURVE ==="
    )
    print()

    print(
        df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=== TELEMETRY ALIGNMENT ===")
    print()

    for _, row in df.iterrows():
        print(
            f"C={int(row['concurrency']):4d}: "
            f"{row.get('telemetry_alignment')}"
        )

    print()
    print("=== SCALING SUMMARY ===")
    print()

    max_row = df.loc[
        df["output_throughput"].idxmax()
    ]

    print(
        "Maximum measured output throughput: "
        f"{max_row['output_throughput']:.2f} tok/s "
        f"at concurrency "
        f"{int(max_row['concurrency'])}"
    )

    candidates = df[
        df["pct_of_max_throughput"] >= 90
    ]

    if not candidates.empty:
        first_90 = candidates.iloc[0]

        print(
            "First measured point reaching "
            ">=90% of measured max: "
            f"C={int(first_90['concurrency'])} "
            f"({first_90['pct_of_max_throughput']:.2f}%)"
        )

    print()
    print(
        "Reminder: the highest measured point "
        "is not necessarily true saturation."
    )

    out_dir = Path(
        "vllm/results/processed/concurrency"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        out_dir
        / f"{args.sweep_dir.name}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()