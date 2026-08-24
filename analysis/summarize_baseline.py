#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


RUN_METRICS = [
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "max_concurrent_requests",

    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",

    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",

    "mean_itl_ms",
    "p50_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",

    "mean_e2el_ms",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
]


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_benchmark_end(date_str: str) -> datetime:
    # vLLM result format:
    # 20260824-135708
    return datetime.strptime(date_str, "%Y%m%d-%H%M%S")


def analyze_gpu_telemetry(run_dir: Path, benchmark: dict):
    path = run_dir / "gpu-metrics.csv"

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y/%m/%d %H:%M:%S.%f",
    )

    benchmark_end = parse_benchmark_end(
        benchmark["date"]
    )

    benchmark_start = (
        benchmark_end
        - timedelta(
            seconds=float(benchmark["duration"])
        )
    )

    # benchmark["date"] has only second precision.
    # Keep a small tolerance at the end so we don't
    # accidentally lose the final telemetry sample.
    mask = (
        (df["timestamp"] >= benchmark_start)
        & (
            df["timestamp"]
            <= benchmark_end + timedelta(seconds=1)
        )
    )

    active = df.loc[mask].copy()

    if active.empty:
        return {
            "telemetry_rows_total": len(df),
            "telemetry_rows_benchmark": 0,
        }

    return {
        "telemetry_rows_total": len(df),
        "telemetry_rows_benchmark": len(active),

        "gpu_util_mean_pct":
            active["utilization_gpu_pct"].mean(),

        "gpu_util_p95_pct":
            active["utilization_gpu_pct"].quantile(0.95),

        "gpu_util_max_pct":
            active["utilization_gpu_pct"].max(),

        "gpu_memory_activity_mean_pct":
            active["utilization_memory_pct"].mean(),

        "vram_mean_mib":
            active["memory_used_mib"].mean(),

        "vram_max_mib":
            active["memory_used_mib"].max(),

        "power_mean_w":
            active["power_draw_w"].mean(),

        "power_max_w":
            active["power_draw_w"].max(),

        "temperature_max_c":
            active["temperature_c"].max(),

        "sm_clock_mean_mhz":
            active["sm_clock_mhz"].mean(),

        "memory_clock_mean_mhz":
            active["memory_clock_mhz"].mean(),

        "estimated_benchmark_start":
            benchmark_start.isoformat(),

        "benchmark_end":
            benchmark_end.isoformat(),
    }


def analyze_arrivals(benchmark: dict):
    starts = np.asarray(
        benchmark.get("start_times", []),
        dtype=float,
    )

    if len(starts) < 2:
        return {}

    gaps = np.diff(starts)

    mean_gap = gaps.mean()
    std_gap = gaps.std(ddof=1)

    cv = (
        std_gap / mean_gap
        if mean_gap > 0
        else np.nan
    )

    return {
        "interarrival_mean_ms":
            mean_gap * 1000,

        "interarrival_std_ms":
            std_gap * 1000,

        "interarrival_cv":
            cv,

        "interarrival_min_ms":
            gaps.min() * 1000,

        "interarrival_p50_ms":
            np.percentile(gaps, 50) * 1000,

        "interarrival_p95_ms":
            np.percentile(gaps, 95) * 1000,

        "interarrival_max_ms":
            gaps.max() * 1000,
    }


def analyze_lengths(benchmark: dict):
    inputs = np.asarray(
        benchmark.get("input_lens", []),
        dtype=float,
    )

    outputs = np.asarray(
        benchmark.get("output_lens", []),
        dtype=float,
    )

    result = {}

    if len(inputs):
        result.update(
            {
                "actual_input_mean_tokens":
                    inputs.mean(),

                "actual_input_min_tokens":
                    inputs.min(),

                "actual_input_max_tokens":
                    inputs.max(),
            }
        )

    if len(outputs):
        result.update(
            {
                "actual_output_mean_tokens":
                    outputs.mean(),

                "actual_output_min_tokens":
                    outputs.min(),

                "actual_output_max_tokens":
                    outputs.max(),
            }
        )

    return result


def analyze_run(run_dir: Path):
    benchmark = load_json(
        run_dir / "benchmark.json"
    )

    manifest = load_json(
        run_dir / "manifest.json"
    )

    row = {
        "run_id": run_dir.name,

        "git_commit":
            manifest["source"]["git_commit"],

        "git_dirty":
            manifest["source"]["git_dirty"],

        "vllm":
            manifest["software"]["vllm"],

        "pytorch":
            manifest["software"]["pytorch"],

        "pytorch_cuda":
            manifest["software"]["pytorch_cuda"],

        "gpu":
            manifest["hardware"]["gpus"][0]["name"],

        "model":
            manifest["model"]["name"],

        "model_sha":
            manifest["model"][
                "resolved_huggingface_sha"
            ],

        "configured_request_rate":
            benchmark["request_rate"],

        "configured_burstiness":
            benchmark["burstiness"],

        "benchmark_duration_s":
            benchmark["duration"],

        "actual_input_tokens_total":
            benchmark["total_input_tokens"],

        "actual_output_tokens_total":
            benchmark["total_output_tokens"],

        "failed_requests":
            benchmark["failed"],
    }

    for metric in RUN_METRICS:
        row[metric] = benchmark.get(metric)

    row.update(
        analyze_gpu_telemetry(
            run_dir,
            benchmark,
        )
    )

    row.update(
        analyze_arrivals(benchmark)
    )

    row.update(
        analyze_lengths(benchmark)
    )

    return row


def coefficient_of_variation(series):
    mean = series.mean()

    if mean == 0:
        return np.nan

    return (
        series.std(ddof=1)
        / mean
        * 100
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
    )

    args = parser.parse_args()

    rows = [
        analyze_run(run_dir)
        for run_dir in args.run_dirs
    ]

    df = pd.DataFrame(rows)

    print("\n=== RUN-LEVEL RESULTS ===\n")

    display_columns = [
        "run_id",
        "request_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2el_ms",
        "max_concurrent_requests",
        "gpu_util_mean_pct",
        "gpu_util_p95_pct",
        "vram_max_mib",
        "interarrival_mean_ms",
        "interarrival_cv",
        "actual_input_mean_tokens",
    ]

    available = [
        column
        for column in display_columns
        if column in df.columns
    ]

    print(
        df[available].to_string(
            index=False,
        )
    )

    print("\n=== REPEATABILITY ===\n")

    stability_metrics = [
        "request_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2el_ms",
        "gpu_util_mean_pct",
    ]

    summary_rows = []

    for metric in stability_metrics:

        if metric not in df:
            continue

        series = pd.to_numeric(
            df[metric],
            errors="coerce",
        ).dropna()

        if not len(series):
            continue

        summary_rows.append(
            {
                "metric": metric,
                "mean": series.mean(),
                "std": (
                    series.std(ddof=1)
                    if len(series) > 1
                    else np.nan
                ),
                "min": series.min(),
                "max": series.max(),
                "cv_pct":
                    coefficient_of_variation(series)
                    if len(series) > 1
                    else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n=== EXPERIMENT IDENTITY CHECK ===\n")

    identity_columns = [
        "git_commit",
        "git_dirty",
        "vllm",
        "pytorch",
        "pytorch_cuda",
        "gpu",
        "model",
        "model_sha",
    ]

    for column in identity_columns:
        values = df[column].dropna().unique()

        status = (
            "OK"
            if len(values) == 1
            else "MISMATCH"
        )

        print(
            f"{column:20s} "
            f"{status:8s} "
            f"{list(values)}"
        )


if __name__ == "__main__":
    main()