#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path):
    return json.loads(path.read_text())


def load_point(point_dir: Path):
    benchmark = read_json(
        point_dir / "benchmark.json"
    )

    gpu_summary = read_json(
        point_dir / "gpu-benchmark-summary.json"
    )

    meta = {}

    for line in (
        point_dir / "point-meta.txt"
    ).read_text().splitlines():

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        meta[key] = value

    return {
        "replicate":
            int(meta["replicate"]),

        "sequence_position":
            int(meta["sequence_position"]),

        "concurrency":
            int(meta["max_concurrency"]),

        "num_prompts":
            int(meta["num_prompts"]),

        "request_throughput":
            benchmark["request_throughput"],

        "output_throughput":
            benchmark["output_throughput"],

        "total_token_throughput":
            benchmark["total_token_throughput"],

        "mean_ttft_ms":
            benchmark["mean_ttft_ms"],

        "p99_ttft_ms":
            benchmark["p99_ttft_ms"],

        "mean_tpot_ms":
            benchmark["mean_tpot_ms"],

        "p99_tpot_ms":
            benchmark["p99_tpot_ms"],

        "mean_e2el_ms":
            benchmark["mean_e2el_ms"],

        "p99_e2el_ms":
            benchmark["p99_e2el_ms"],

        "gpu_util_mean_pct":
            gpu_summary["gpu_util_mean_pct"],

        "gpu_util_p95_pct":
            gpu_summary["gpu_util_p95_pct"],

        "power_mean_w":
            gpu_summary["power_mean_w"],

        "vram_max_mib":
            gpu_summary["vram_max_mib"],

        "telemetry_samples":
            gpu_summary[
                "telemetry_samples_benchmark"
            ],
    }


def cv_pct(series):
    mean = series.mean()

    if mean == 0:
        return np.nan

    return (
        series.std(ddof=1)
        / mean
        * 100.0
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment_dir",
        type=Path,
    )

    args = parser.parse_args()

    points = sorted(
        args.experiment_dir.glob(
            "rep_*/concurrency_*"
        )
    )

    rows = [
        load_point(point)
        for point in points
        if (
            point / "benchmark.json"
        ).exists()
    ]

    if len(rows) != 9:
        raise RuntimeError(
            f"Expected 9 points, found {len(rows)}"
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        [
            "replicate",
            "sequence_position",
        ]
    ).reset_index(drop=True)

    print()
    print("=== INDIVIDUAL MEASUREMENTS ===")
    print()

    display = [
        "replicate",
        "sequence_position",
        "concurrency",
        "output_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2el_ms",
        "gpu_util_mean_pct",
        "power_mean_w",
    ]

    print(
        df[display].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("=== REPEATABILITY BY CONCURRENCY ===")
    print()

    metrics = [
        "output_throughput",
        "request_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "mean_e2el_ms",
        "gpu_util_mean_pct",
        "power_mean_w",
    ]

    summary_rows = []

    for concurrency, group in df.groupby(
        "concurrency"
    ):

        for metric in metrics:
            series = group[metric]

            summary_rows.append(
                {
                    "concurrency":
                        concurrency,

                    "metric":
                        metric,

                    "mean":
                        series.mean(),

                    "std":
                        series.std(ddof=1),

                    "min":
                        series.min(),

                    "max":
                        series.max(),

                    "cv_pct":
                        cv_pct(series),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    key_metrics = summary[
        summary["metric"].isin(
            [
                "output_throughput",
                "mean_ttft_ms",
                "p99_ttft_ms",
                "mean_tpot_ms",
                "mean_e2el_ms",
            ]
        )
    ]

    print(
        key_metrics.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("=== OUTPUT THROUGHPUT SUMMARY ===")
    print()

    throughput = (
        df.groupby("concurrency")[
            "output_throughput"
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    throughput["cv_pct"] = (
        df.groupby("concurrency")[
            "output_throughput"
        ]
        .apply(cv_pct)
    )

    print(
        throughput.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print()
    print("=== PAIRED C128 vs C256 ===")
    print()

    pivot = df.pivot(
        index="replicate",
        columns="concurrency",
        values="output_throughput",
    )

    if 128 in pivot and 256 in pivot:
        comparison = pd.DataFrame(
            {
                "c128_tok_s":
                    pivot[128],

                "c256_tok_s":
                    pivot[256],
            }
        )

        comparison["difference_tok_s"] = (
            comparison["c128_tok_s"]
            - comparison["c256_tok_s"]
        )

        comparison["c128_advantage_pct"] = (
            comparison["difference_tok_s"]
            / comparison["c256_tok_s"]
            * 100.0
        )

        print(
            comparison.to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )

        print()

        mean_advantage = (
            comparison[
                "c128_advantage_pct"
            ].mean()
        )

        print(
            "Mean paired C128 advantage over C256: "
            f"{mean_advantage:.3f}%"
        )

    print()
    print("=== POSITION CHECK ===")
    print()

    position_summary = (
        df.groupby(
            "sequence_position"
        )["output_throughput"]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print(
        position_summary.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    out_dir = Path(
        "vllm/results/processed/"
        "saturation_repeatability"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_output = (
        out_dir
        / f"{args.experiment_dir.name}.csv"
    )

    summary_output = (
        out_dir
        / f"{args.experiment_dir.name}_summary.csv"
    )

    df.to_csv(
        raw_output,
        index=False,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print()
    print(f"Saved: {raw_output}")
    print(f"Saved: {summary_output}")


if __name__ == "__main__":
    main()