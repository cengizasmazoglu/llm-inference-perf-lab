#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS_MS = [100, 250, 500, 1000]


def load_run(run_dir: Path) -> pd.DataFrame:
    data = json.loads(
        (run_dir / "benchmark.json").read_text()
    )

    starts = np.asarray(
        data["start_times"],
        dtype=float,
    )

    ttfts_ms = (
        np.asarray(
            data["ttfts"],
            dtype=float,
        )
        * 1000.0
    )

    if len(starts) != len(ttfts_ms):
        raise ValueError(
            "start_times and ttfts have different lengths"
        )

    order = np.argsort(starts)

    starts = starts[order]
    ttfts_ms = ttfts_ms[order]

    start_offset_ms = (
        starts - starts[0]
    ) * 1000.0

    previous_gap_ms = np.full(
        len(starts),
        np.nan,
    )

    if len(starts) > 1:
        previous_gap_ms[1:] = (
            np.diff(starts) * 1000.0
        )

    result = pd.DataFrame(
        {
            "request_index": np.arange(
                len(starts)
            ),
            "start_time_monotonic": starts,
            "start_offset_ms": start_offset_ms,
            "previous_gap_ms": previous_gap_ms,
            "ttft_ms": ttfts_ms,
        }
    )

    for window_ms in WINDOWS_MS:
        window_s = window_ms / 1000.0

        counts = []

        for i, t in enumerate(starts):
            left = np.searchsorted(
                starts,
                t - window_s,
                side="left",
            )

            # Only requests that arrived before this one.
            counts.append(i - left)

        result[
            f"arrivals_prev_{window_ms}ms"
        ] = counts

    return result


def spearman(x, y):
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if len(frame) < 3:
        return np.nan

    return (
        frame["x"]
        .rank()
        .corr(frame["y"].rank())
    )


def analyze_run(run_dir: Path):
    df = load_run(run_dir)

    print()
    print("=" * 70)
    print(run_dir.name)
    print("=" * 70)

    print("\n=== ARRIVAL → TTFT ASSOCIATIONS ===\n")

    gap_corr = spearman(
        df["previous_gap_ms"],
        df["ttft_ms"],
    )

    print(
        "Spearman(previous inter-arrival gap, TTFT): "
        f"{gap_corr:.4f}"
    )

    for window_ms in WINDOWS_MS:
        column = (
            f"arrivals_prev_{window_ms}ms"
        )

        corr = spearman(
            df[column],
            df["ttft_ms"],
        )

        print(
            f"Spearman(arrivals in previous "
            f"{window_ms:4d} ms, TTFT): "
            f"{corr:.4f}"
        )

    print("\n=== HIGHEST TTFT REQUESTS ===\n")

    columns = [
        "request_index",
        "start_offset_ms",
        "previous_gap_ms",
        "arrivals_prev_100ms",
        "arrivals_prev_250ms",
        "arrivals_prev_500ms",
        "arrivals_prev_1000ms",
        "ttft_ms",
    ]

    top = (
        df.sort_values(
            "ttft_ms",
            ascending=False,
        )
        .head(10)
    )

    print(
        top[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n=== ARRIVAL STATISTICS ===\n")

    gaps = df[
        "previous_gap_ms"
    ].dropna()

    print(
        f"mean gap:   {gaps.mean():.2f} ms"
    )
    print(
        f"std gap:    {gaps.std(ddof=1):.2f} ms"
    )
    print(
        f"CV:         "
        f"{gaps.std(ddof=1) / gaps.mean():.4f}"
    )
    print(
        f"min gap:    {gaps.min():.2f} ms"
    )
    print(
        f"median gap: {gaps.median():.2f} ms"
    )
    print(
        f"max gap:    {gaps.max():.2f} ms"
    )

    out_dir = Path(
        "vllm/results/processed/"
        "arrival_ttft"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        out_dir /
        f"{run_dir.name}.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"Per-request analysis saved: "
        f"{csv_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
    )

    args = parser.parse_args()

    for run_dir in args.run_dirs:
        analyze_run(run_dir)


if __name__ == "__main__":
    main()