#!/usr/bin/env python3

import argparse
import json
import sqlite3
import statistics
from dataclasses import dataclass, replace
from pathlib import Path


FA2_KERNEL = "flash_fwd_splitkv_kernel"
MODEL_LAYERS = 36


@dataclass(frozen=True)
class Regime:
    name: str
    start_ns: int
    end_ns: int
    steady_start_step: int
    steady_end_step: int
    benchmark_dir: str
    output_tokens: int = 0


# 007B trace provenance.
#
# start_ns/end_ns are benchmark main-workload windows in the Nsight/CUPTI
# timestamp domain. They were established during offline 007B trace alignment.
# The trace contains no harness/regime NVTX markers and exports no monotonic
# clock anchor that would allow benchmark.json start_times to be converted
# automatically into CUPTI timestamps. Do not silently recompute these values.
#
# steady_start_step/steady_end_step are explicit empirically selected plateau
# windows used for the steady mixed-FA2 comparison.
REGIMES = [
    Regime(
        "C64",
        103077332756,
        107219161224,
        20,
        31,
        "concurrency_064",
    ),
    Regime(
        "C128",
        132574338977,
        139571274627,
        38,
        67,
        "concurrency_128",
    ),
    Regime(
        "C256",
        165702428116,
        179881395115,
        74,
        137,
        "concurrency_256",
    ),
]



def load_benchmark_metadata(artifacts_root: Path, regime: Regime):
    path = artifacts_root / regime.benchmark_dir / "benchmark.json"

    if not path.is_file():
        raise FileNotFoundError(f"{regime.name}: missing {path}")

    data = json.loads(path.read_text())

    expected_concurrency = int(regime.name.removeprefix("C"))

    if data["max_concurrency"] != expected_concurrency:
        raise RuntimeError(
            f"{regime.name}: expected max_concurrency="
            f"{expected_concurrency}, got {data['max_concurrency']}"
        )

    if data["failed"] != 0:
        raise RuntimeError(
            f"{regime.name}: benchmark contains {data['failed']} failures"
        )

    if data["completed"] != data["num_prompts"]:
        raise RuntimeError(
            f"{regime.name}: completed={data['completed']} but "
            f"num_prompts={data['num_prompts']}"
        )

    input_lens = data["input_lens"]
    output_lens = data["output_lens"]
    start_times = data["start_times"]

    if len(input_lens) != data["completed"]:
        raise RuntimeError(
            f"{regime.name}: input_lens count does not match completed"
        )

    if len(output_lens) != data["completed"]:
        raise RuntimeError(
            f"{regime.name}: output_lens count does not match completed"
        )

    if len(start_times) != data["completed"]:
        raise RuntimeError(
            f"{regime.name}: start_times count does not match completed"
        )

    if sum(input_lens) != data["total_input_tokens"]:
        raise RuntimeError(
            f"{regime.name}: input token total does not match input_lens"
        )

    if sum(output_lens) != data["total_output_tokens"]:
        raise RuntimeError(
            f"{regime.name}: output token total does not match output_lens"
        )

    return {
        "path": path,
        "num_prompts": data["num_prompts"],
        "completed": data["completed"],
        "failed": data["failed"],
        "max_concurrency": data["max_concurrency"],
        "duration_s": data["duration"],
        "total_input_tokens": data["total_input_tokens"],
        "total_output_tokens": data["total_output_tokens"],
        "min_input_len": min(input_lens),
        "max_input_len": max(input_lens),
        "mean_input_len": statistics.mean(input_lens),
        "min_output_len": min(output_lens),
        "max_output_len": max(output_lens),
    }

def load_mixed_fa2_steps(conn: sqlite3.Connection, regime: Regime):
    rows = conn.execute(
        """
        SELECT k.start, k.end, k.gridY
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s
          ON s.id = k.shortName
        WHERE s.value = ?
          AND k.gridX = 9
          AND k.gridZ = 16
          AND k.start >= ?
          AND k.start < ?
        ORDER BY k.start
        """,
        (FA2_KERNEL, regime.start_ns, regime.end_ns),
    ).fetchall()

    if not rows:
        raise RuntimeError(f"{regime.name}: no mixed FA2 kernels found")

    if len(rows) % MODEL_LAYERS != 0:
        raise RuntimeError(
            f"{regime.name}: {len(rows)} launches is not divisible "
            f"by {MODEL_LAYERS} model layers"
        )

    steps = []

    for step_id, offset in enumerate(range(0, len(rows), MODEL_LAYERS)):
        group = rows[offset : offset + MODEL_LAYERS]

        batches = [row[2] for row in group]

        if min(batches) != max(batches):
            raise RuntimeError(
                f"{regime.name} step {step_id}: inconsistent gridY "
                f"within 36-layer forward: {min(batches)}..{max(batches)}"
            )

        avg_kernel_us = statistics.mean(
            (end_ns - start_ns) / 1_000.0
            for start_ns, end_ns, _ in group
        )

        steps.append(
            {
                "step": step_id,
                "batch": batches[0],
                "avg_kernel_us": avg_kernel_us,
            }
        )

    return steps


def summarize_steady(regime: Regime, steps):
    steady = [
        row
        for row in steps
        if regime.steady_start_step
        <= row["step"]
        <= regime.steady_end_step
    ]

    expected = regime.steady_end_step - regime.steady_start_step + 1
    if len(steady) != expected:
        raise RuntimeError(
            f"{regime.name}: expected {expected} steady steps, "
            f"found {len(steady)}"
        )

    return {
        "name": regime.name,
        "total_steps": len(steps),
        "steady_range": (
            regime.steady_start_step,
            regime.steady_end_step,
        ),
        "steady_steps": len(steady),
        "mean_batch": statistics.mean(row["batch"] for row in steady),
        "mean_kernel_us": statistics.mean(
            row["avg_kernel_us"] for row in steady
        ),
    }




def classify_kernel(name: str) -> str:
    lower = name.lower()

    if "gemm" in lower:
        return "gemm"

    if "flash_fwd_splitkv" in lower:
        return "splitkv_attention"

    if "triton" in lower:
        return "triton_other"

    return "other"



def summarize_fa2_geometry(conn: sqlite3.Connection, regime: Regime):
    rows = conn.execute(
        """
        SELECT k.gridX, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s
          ON s.id = k.shortName
        WHERE s.value = ?
          AND k.start >= ?
          AND k.start < ?
          AND k.gridX IN (1, 9)
        """,
        (FA2_KERNEL, regime.start_ns, regime.end_ns),
    ).fetchall()

    result = {}

    for grid_x in (1, 9):
        selected = [
            (start_ns, end_ns)
            for gx, start_ns, end_ns in rows
            if gx == grid_x
        ]

        total_ns = sum(
            end_ns - start_ns
            for start_ns, end_ns in selected
        )

        result[grid_x] = {
            "calls": len(selected),
            "total_ms": total_ns / 1e6,
            "us_per_output_token": (
                total_ns / 1e3 / regime.output_tokens
            ),
            "avg_kernel_us": (
                total_ns / 1e3 / len(selected)
                if selected else 0.0
            ),
        }

    return result

def summarize_kernel_families(conn: sqlite3.Connection, regime: Regime):
    rows = conn.execute(
        """
        SELECT s.value, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s
          ON s.id = k.shortName
        WHERE k.start >= ?
          AND k.start < ?
        """,
        (regime.start_ns, regime.end_ns),
    ).fetchall()

    totals_ns = {
        "gemm": 0,
        "splitkv_attention": 0,
        "triton_other": 0,
        "other": 0,
    }

    calls = {key: 0 for key in totals_ns}

    for name, start_ns, end_ns in rows:
        family = classify_kernel(name)
        totals_ns[family] += end_ns - start_ns
        calls[family] += 1

    return {
        family: {
            "calls": calls[family],
            "total_ms": totals_ns[family] / 1e6,
            "us_per_output_token": (
                totals_ns[family] / 1e3 / regime.output_tokens
            ),
        }
        for family in totals_ns
    }

def summarize_gpu_busy(conn: sqlite3.Connection, regime: Regime):
    rows = conn.execute(
        """
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start < ?
          AND end > ?
        ORDER BY start
        """,
        (regime.end_ns, regime.start_ns),
    ).fetchall()

    intervals = [
        (max(start, regime.start_ns), min(end, regime.end_ns))
        for start, end in rows
        if min(end, regime.end_ns) > max(start, regime.start_ns)
    ]

    busy_ns = 0
    if intervals:
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                busy_ns += current_end - current_start
                current_start, current_end = start, end

        busy_ns += current_end - current_start

    wall_ns = regime.end_ns - regime.start_ns

    return {
        "name": regime.name,
        "kernel_count": len(rows),
        "wall_s": wall_ns / 1e9,
        "busy_s": busy_ns / 1e9,
        "busy_pct": 100.0 * busy_ns / wall_ns,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Nsight Systems regime behavior for vLLM 007B."
    )
    parser.add_argument(
        "sqlite",
        help="Nsight Systems SQLite export, e.g. server_full.sqlite",
    )
    parser.add_argument(
        "--artifacts-root",
        required=True,
        type=Path,
        help=(
            "007B artifact directory containing concurrency_064/, "
            "concurrency_128/, and concurrency_256/"
        ),
    )
    args = parser.parse_args()

    benchmark_metadata = {}
    regimes = []

    for regime in REGIMES:
        metadata = load_benchmark_metadata(args.artifacts_root, regime)

        trace_duration_s = (regime.end_ns - regime.start_ns) / 1e9
        duration_error_s = abs(trace_duration_s - metadata["duration_s"])

        if duration_error_s > 1e-6:
            raise RuntimeError(
                f"{regime.name}: trace window duration "
                f"{trace_duration_s:.9f}s does not match benchmark.json "
                f"duration {metadata['duration_s']:.9f}s "
                f"(error={duration_error_s:.9f}s)"
            )

        benchmark_metadata[regime.name] = metadata
        regimes.append(
            replace(
                regime,
                output_tokens=metadata["total_output_tokens"],
            )
        )

    conn = sqlite3.connect(args.sqlite)

    try:
        summaries = []

        for regime in regimes:
            steps = load_mixed_fa2_steps(conn, regime)
            summaries.append(summarize_steady(regime, steps))

        print("Benchmark provenance from benchmark.json")
        print("=" * 88)
        print(
            f"{'Regime':<8}"
            f"{'Prompts':>10}"
            f"{'Duration s':>14}"
            f"{'Output tok':>14}"
            f"{'Input len':>16}"
            f"{'Mean input':>14}"
        )

        for regime in regimes:
            row = benchmark_metadata[regime.name]
            input_range = (
                f"{row['min_input_len']}-{row['max_input_len']}"
                if row["min_input_len"] != row["max_input_len"]
                else str(row["min_input_len"])
            )

            print(
                f"{regime.name:<8}"
                f"{row['num_prompts']:>10}"
                f"{row['duration_s']:>14.3f}"
                f"{row['total_output_tokens']:>14}"
                f"{input_range:>16}"
                f"{row['mean_input_len']:>14.2f}"
            )

        print()
        print("Trace-window provenance")
        print("=" * 88)
        print(
            "Nsight benchmark windows are explicit 007B alignment metadata; "
            "benchmark.json start_times are in a different monotonic clock "
            "domain and the trace contains no harness NVTX clock bridge."
        )

        print()
        print("Mixed/prefill-containing FA2 steady regimes")
        print("=" * 72)
        print(
            f"{'Regime':<8}"
            f"{'All steps':>12}"
            f"{'Steady':>12}"
            f"{'Mean batch':>16}"
            f"{'Kernel us':>16}"
        )

        for row in summaries:
            lo, hi = row["steady_range"]
            steady_label = f"{lo}-{hi}"

            print(
                f"{row['name']:<8}"
                f"{row['total_steps']:>12}"
                f"{steady_label:>12}"
                f"{row['mean_batch']:>16.2f}"
                f"{row['mean_kernel_us']:>16.2f}"
            )

        c128 = next(x for x in summaries if x["name"] == "C128")
        c256 = next(x for x in summaries if x["name"] == "C256")

        batch_ratio = c256["mean_batch"] / c128["mean_batch"]
        kernel_ratio = (
            c256["mean_kernel_us"] / c128["mean_kernel_us"]
        )

        print()
        print("C128 -> C256 steady-regime ratios")
        print("=" * 72)
        print(f"Attention batch ratio : {batch_ratio:.4f}x")
        print(f"FA2 kernel-time ratio : {kernel_ratio:.4f}x")

        print()
        print("GPU kernel-active time")
        print("=" * 72)
        print(
            f"{'Regime':<8}"
            f"{'Kernels':>12}"
            f"{'Wall s':>14}"
            f"{'Busy s':>14}"
            f"{'Busy %':>14}"
        )

        for regime in regimes:
            row = summarize_gpu_busy(conn, regime)
            print(
                f"{row['name']:<8}"
                f"{row['kernel_count']:>12}"
                f"{row['wall_s']:>14.3f}"
                f"{row['busy_s']:>14.3f}"
                f"{row['busy_pct']:>13.2f}%"
            )


        print()
        print("Summed kernel duration per output token")
        print("=" * 72)
        print(
            f"{'Family':<22}"
            f"{'C64 us/tok':>16}"
            f"{'C128 us/tok':>16}"
            f"{'C256 us/tok':>16}"
        )

        family_results = {
            regime.name: summarize_kernel_families(conn, regime)
            for regime in regimes
        }

        for family in (
            "gemm",
            "splitkv_attention",
            "triton_other",
            "other",
        ):
            print(
                f"{family:<22}"
                f"{family_results['C64'][family]['us_per_output_token']:>16.2f}"
                f"{family_results['C128'][family]['us_per_output_token']:>16.2f}"
                f"{family_results['C256'][family]['us_per_output_token']:>16.2f}"
            )


        print()
        print("FA2 main-kernel geometry decomposition")
        print("=" * 72)
        print(
            f"{'Path':<22}"
            f"{'C64 us/tok':>16}"
            f"{'C128 us/tok':>16}"
            f"{'C256 us/tok':>16}"
        )

        fa2_results = {
            regime.name: summarize_fa2_geometry(conn, regime)
            for regime in regimes
        }

        for grid_x, label in (
            (1, "gridX=1 decode-oriented"),
            (9, "gridX=9 mixed/prefill"),
        ):
            print(
                f"{label:<22}"
                f"{fa2_results['C64'][grid_x]['us_per_output_token']:>16.2f}"
                f"{fa2_results['C128'][grid_x]['us_per_output_token']:>16.2f}"
                f"{fa2_results['C256'][grid_x]['us_per_output_token']:>16.2f}"
            )

        print()
        print("FA2 launch counts")
        print("=" * 72)
        print(
            f"{'Path':<22}"
            f"{'C64':>12}"
            f"{'C128':>12}"
            f"{'C256':>12}"
        )

        for grid_x, label in (
            (1, "gridX=1"),
            (9, "gridX=9"),
        ):
            print(
                f"{label:<22}"
                f"{fa2_results['C64'][grid_x]['calls']:>12}"
                f"{fa2_results['C128'][grid_x]['calls']:>12}"
                f"{fa2_results['C256'][grid_x]['calls']:>12}"
            )


        print()
        print("C128 -> C256 kernel-family delta attribution")
        print("=" * 72)
        print(
            f"{'Family':<22}"
            f"{'C128 us/tok':>16}"
            f"{'C256 us/tok':>16}"
            f"{'Delta':>16}"
        )

        total_c128 = 0.0
        total_c256 = 0.0

        for family in (
            "gemm",
            "splitkv_attention",
            "triton_other",
            "other",
        ):
            c128_value = family_results["C128"][family]["us_per_output_token"]
            c256_value = family_results["C256"][family]["us_per_output_token"]
            delta = c256_value - c128_value

            total_c128 += c128_value
            total_c256 += c256_value

            print(
                f"{family:<22}"
                f"{c128_value:>16.2f}"
                f"{c256_value:>16.2f}"
                f"{delta:>+16.2f}"
            )

        print("-" * 72)
        print(
            f"{'TOTAL':<22}"
            f"{total_c128:>16.2f}"
            f"{total_c256:>16.2f}"
            f"{total_c256 - total_c128:>+16.2f}"
        )

        mixed_c128 = fa2_results["C128"][9]["us_per_output_token"]
        mixed_c256 = fa2_results["C256"][9]["us_per_output_token"]

        decode_c128 = fa2_results["C128"][1]["us_per_output_token"]
        decode_c256 = fa2_results["C256"][1]["us_per_output_token"]

        print()
        print("FA2 path-specific C128 -> C256 deltas")
        print("=" * 72)
        print(
            f"decode-oriented gridX=1 : "
            f"{decode_c128:.2f} -> {decode_c256:.2f} "
            f"({decode_c256 - decode_c128:+.2f} us/tok)"
        )
        print(
            f"mixed/prefill gridX=9   : "
            f"{mixed_c128:.2f} -> {mixed_c256:.2f} "
            f"({mixed_c256 - mixed_c128:+.2f} us/tok)"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
