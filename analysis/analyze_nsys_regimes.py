#!/usr/bin/env python3

import csv
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# A long GPU-idle interval separates distinct pieces of work such as
# vLLM bench's initial single-request probe from the actual benchmark.
#
# This is NOT used as a performance metric. It is only used to partition
# the broad client invocation window into candidate GPU-active islands.
ISLAND_BREAK_NS = 500_000_000  # 500 ms


REGIMES = (
    "C64",
    "C128",
    "C256",
)


def percentile(values, p):
    if not values:
        return float("nan")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = (
        position - lower
    )

    return (
        values[lower]
        * (1.0 - fraction)
        + values[upper]
        * fraction
    )


def iso_to_epoch_ns(text):
    match = re.fullmatch(
        r"(.+?)[,.](\d+)"
        r"(Z|[+-]\d\d:\d\d)",
        text,
    )

    if not match:
        raise ValueError(
            "Unsupported timestamp: "
            f"{text}"
        )

    base = match.group(1)
    fractional = match.group(2)
    timezone = match.group(3)

    if timezone == "Z":
        timezone = "+00:00"

    dt = datetime.fromisoformat(
        base + timezone
    )

    fractional_ns = int(
        fractional[:9].ljust(
            9,
            "0",
        )
    )

    return (
        int(dt.timestamp())
        * 1_000_000_000
        + fractional_ns
    )


def load_client_windows(path):
    windows = {}

    with path.open() as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()

            if not raw_line:
                continue

            parts = raw_line.split()

            if len(parts) != 4:
                raise ValueError(
                    "Unexpected client window "
                    f"row: {raw_line!r}"
                )

            (
                regime,
                edge,
                monotonic_ns,
                utc_text,
            ) = parts

            windows.setdefault(
                regime,
                {},
            )[edge] = {
                "monotonic_ns":
                    int(monotonic_ns),

                "utc_text":
                    utc_text,

                "epoch_ns":
                    iso_to_epoch_ns(
                        utc_text
                    ),
            }

    for regime in REGIMES:
        if regime not in windows:
            raise RuntimeError(
                f"Missing {regime} "
                "from client_windows.tsv"
            )

        if (
            "START"
            not in windows[regime]
            or "END"
            not in windows[regime]
        ):
            raise RuntimeError(
                f"Incomplete {regime} "
                "window"
            )

    return windows


def resolve_string(
    value,
    string_ids,
):
    if value is None:
        return ""

    if isinstance(
        value,
        int,
    ):
        return string_ids.get(
            value,
            str(value),
        )

    return str(value)


def kernel_family(name):
    lowered = name.lower()

    if (
        "reshape_and_cache"
        in lowered
        or "kv_cache"
        in lowered
        or "kvcache"
        in lowered
        or "cache_flash"
        in lowered
    ):
        return "kv_cache"

    if (
        "flash" in lowered
        or "attention" in lowered
    ):
        return "attention"

    if (
        "gemm" in lowered
        or "gemv" in lowered
        or "cutlass" in lowered
        or "cublas" in lowered
    ):
        return "gemm"

    if "triton" in lowered:
        return "triton_fused"

    if (
        "elementwise"
        in lowered
        or "vectorized"
        in lowered
        or "fillfunctor"
        in lowered
    ):
        return "elementwise"

    return "other"


def merge_overlapping_intervals(
    intervals,
):
    if not intervals:
        return []

    intervals = sorted(
        intervals
    )

    merged = [
        [
            intervals[0][0],
            intervals[0][1],
        ]
    ]

    for start, end in intervals[1:]:
        current = merged[-1]

        if start <= current[1]:
            current[1] = max(
                current[1],
                end,
            )

        else:
            merged.append(
                [
                    start,
                    end,
                ]
            )

    return merged


def split_into_islands(
    events,
):
    if not events:
        return []

    events = sorted(
        events,
        key=lambda event:
            event["start"],
    )

    islands = []
    current = [
        events[0]
    ]

    previous_end = (
        events[0]["end"]
    )

    for event in events[1:]:
        gap_ns = (
            event["start"]
            - previous_end
        )

        if gap_ns > ISLAND_BREAK_NS:
            islands.append(
                current
            )

            current = [
                event
            ]

        else:
            current.append(
                event
            )

        previous_end = max(
            previous_end,
            event["end"],
        )

    islands.append(
        current
    )

    return islands


def island_span_s(events):
    if not events:
        return 0.0

    return (
        max(
            event["end"]
            for event in events
        )
        - min(
            event["start"]
            for event in events
        )
    ) / 1e9


def choose_main_island(
    islands,
    benchmark_duration_s,
):
    if not islands:
        raise RuntimeError(
            "No GPU islands available"
        )

    if (
        benchmark_duration_s
        is not None
        and benchmark_duration_s > 0
    ):
        return min(
            islands,
            key=lambda island:
                abs(
                    island_span_s(
                        island
                    )
                    - benchmark_duration_s
                ),
        )

    return max(
        islands,
        key=island_span_s,
    )


def deep_find(
    obj,
    candidates,
):
    if isinstance(
        obj,
        dict,
    ):
        for key in candidates:
            if key in obj:
                return obj[key]

        for value in obj.values():
            result = deep_find(
                value,
                candidates,
            )

            if result is not None:
                return result

    elif isinstance(
        obj,
        list,
    ):
        for value in obj:
            result = deep_find(
                value,
                candidates,
            )

            if result is not None:
                return result

    return None


def load_benchmark(path):
    with path.open() as handle:
        data = json.load(
            handle
        )

    successful_requests = deep_find(
        data,
        (
            "successful_requests",
            "completed",
            "num_successful_requests",
        ),
    )

    failed_requests = deep_find(
        data,
        (
            "failed_requests",
            "failed",
            "num_failed_requests",
        ),
    )

    input_tokens = deep_find(
        data,
        (
            "total_input_tokens",
        ),
    )

    output_tokens = deep_find(
        data,
        (
            "total_generated_tokens",
            "total_output_tokens",
        ),
    )

    duration_s = deep_find(
        data,
        (
            "benchmark_duration",
            "duration",
        ),
    )

    output_throughput = deep_find(
        data,
        (
            "output_token_throughput",
            "output_throughput",
        ),
    )

    return {
        "successful_requests":
            successful_requests,

        "failed_requests":
            failed_requests,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "benchmark_duration_s":
            duration_s,

        "output_throughput":
            output_throughput,
    }


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Wrote {path}"
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: "
            "analyze_nsys_regimes.py "
            "EXP_DIR"
        )

    exp_dir = Path(
        sys.argv[1]
    )

    db_path = (
        exp_dir
        / "server_full.sqlite"
    )

    windows_path = (
        exp_dir
        / "client_windows.tsv"
    )

    if not db_path.exists():
        raise SystemExit(
            f"Missing {db_path}"
        )

    if not windows_path.exists():
        raise SystemExit(
            f"Missing {windows_path}"
        )

    windows = load_client_windows(
        windows_path
    )

    benchmark_paths = {
        "C64":
            exp_dir
            / "concurrency_064"
            / "benchmark.json",

        "C128":
            exp_dir
            / "concurrency_128"
            / "benchmark.json",

        "C256":
            exp_dir
            / "concurrency_256"
            / "benchmark.json",
    }

    benchmarks = {}

    for regime, path in (
        benchmark_paths.items()
    ):
        if not path.exists():
            raise RuntimeError(
                f"Missing {path}"
            )

        benchmarks[regime] = (
            load_benchmark(
                path
            )
        )

    con = sqlite3.connect(
        str(db_path)
    )

    string_ids = dict(
        con.execute(
            """
            SELECT
                id,
                value
            FROM StringIds
            """
        ).fetchall()
    )

    processes = {}

    for (
        global_pid,
        pid,
        name,
    ) in con.execute(
        """
        SELECT
            globalPid,
            pid,
            name
        FROM PROCESSES
        """
    ):
        processes[
            global_pid
        ] = {
            "pid":
                pid,

            "name":
                resolve_string(
                    name,
                    string_ids,
                ),
        }

    kernel_counts = dict(
        con.execute(
            """
            SELECT
                globalPid,
                COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY globalPid
            """
        ).fetchall()
    )

    print(
        "\n=== GPU KERNEL PROCESSES ==="
    )

    ranked_processes = sorted(
        kernel_counts.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    for (
        global_pid,
        count,
    ) in ranked_processes:
        info = processes.get(
            global_pid,
            {},
        )

        print(
            f"globalPid={global_pid} "
            f"pid={info.get('pid')} "
            f"kernels={count} "
            f"name={info.get('name')}"
        )

    engine_candidates = [
        global_pid
        for (
            global_pid,
            _,
        ) in ranked_processes
        if "enginecor"
        in (
            processes
            .get(
                global_pid,
                {},
            )
            .get(
                "name",
                "",
            )
            .lower()
        )
    ]

    if engine_candidates:
        engine_global_pid = (
            engine_candidates[0]
        )

        selection_reason = (
            "EngineCore process name"
        )

    elif ranked_processes:
        engine_global_pid = (
            ranked_processes[0][0]
        )

        selection_reason = (
            "largest CUDA kernel count"
        )

    else:
        raise RuntimeError(
            "No CUDA kernel events found"
        )

    engine_info = processes.get(
        engine_global_pid,
        {},
    )

    print(
        "\n=== SELECTED GPU PROCESS ==="
    )

    print(
        f"globalPid="
        f"{engine_global_pid}"
    )

    print(
        f"pid="
        f"{engine_info.get('pid')}"
    )

    print(
        f"name="
        f"{engine_info.get('name')}"
    )

    print(
        f"selection_reason="
        f"{selection_reason}"
    )

    session_row = con.execute(
        """
        SELECT
            utcEpochNs,
            utcTime,
            localTime
        FROM TARGET_INFO_SESSION_START_TIME
        LIMIT 1
        """
    ).fetchone()

    if not session_row:
        raise RuntimeError(
            "Missing "
            "TARGET_INFO_SESSION_START_TIME"
        )

    session_epoch_ns = int(
        session_row[0]
    )

    print(
        "\n=== NSIGHT SESSION ==="
    )

    print(
        f"utcEpochNs="
        f"{session_epoch_ns}"
    )

    print(
        f"utcTime="
        f"{session_row[1]}"
    )

    print(
        f"localTime="
        f"{session_row[2]}"
    )

    kernel_minmax = con.execute(
        """
        SELECT
            MIN(start),
            MAX(end)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE globalPid = ?
        """,
        (
            engine_global_pid,
        ),
    ).fetchone()

    print(
        "\n=== ENGINECORE KERNEL "
        "TIMESTAMP RANGE ==="
    )

    print(
        f"min={kernel_minmax[0]}"
    )

    print(
        f"max={kernel_minmax[1]}"
    )

    candidate_windows = {
        "session_relative_utc":
            {},

        "host_monotonic":
            {},
    }

    for (
        regime,
        edges,
    ) in windows.items():
        candidate_windows[
            "session_relative_utc"
        ][regime] = (
            edges[
                "START"
            ][
                "epoch_ns"
            ]
            - session_epoch_ns,

            edges[
                "END"
            ][
                "epoch_ns"
            ]
            - session_epoch_ns,
        )

        candidate_windows[
            "host_monotonic"
        ][regime] = (
            edges[
                "START"
            ][
                "monotonic_ns"
            ],

            edges[
                "END"
            ][
                "monotonic_ns"
            ],
        )

    alignment_scores = {}

    for (
        mode,
        regime_windows,
    ) in candidate_windows.items():
        score = 0

        for (
            start_ns,
            end_ns,
        ) in regime_windows.values():
            count = con.execute(
                """
                SELECT COUNT(*)
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE globalPid = ?
                  AND end > ?
                  AND start < ?
                """,
                (
                    engine_global_pid,
                    start_ns,
                    end_ns,
                ),
            ).fetchone()[0]

            score += count

        alignment_scores[
            mode
        ] = score

    print(
        "\n=== CLOCK ALIGNMENT TEST ==="
    )

    for (
        mode,
        score,
    ) in alignment_scores.items():
        print(
            f"{mode}: "
            f"{score} overlapping kernels"
        )

    alignment_mode = max(
        alignment_scores,
        key=alignment_scores.get,
    )

    if (
        alignment_scores[
            alignment_mode
        ]
        == 0
    ):
        raise RuntimeError(
            "Could not align client "
            "windows with Nsight "
            "kernel timestamps"
        )

    print(
        "\nUsing alignment: "
        f"{alignment_mode}"
    )

    regime_windows = (
        candidate_windows[
            alignment_mode
        ]
    )

    summaries = []
    family_rows = []
    top_kernel_rows = []
    island_rows = []

    for regime in REGIMES:
        benchmark = (
            benchmarks[
                regime
            ]
        )

        (
            outer_start_ns,
            outer_end_ns,
        ) = regime_windows[
            regime
        ]

        rows = con.execute(
            """
            SELECT
                start,
                end,
                graphNodeId,
                demangledName,
                shortName,
                mangledName
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE globalPid = ?
              AND end > ?
              AND start < ?
            ORDER BY start
            """,
            (
                engine_global_pid,
                outer_start_ns,
                outer_end_ns,
            ),
        ).fetchall()

        outer_events = []

        for (
            start_ns,
            end_ns,
            graph_node_id,
            demangled_name,
            short_name,
            mangled_name,
        ) in rows:
            clipped_start_ns = max(
                start_ns,
                outer_start_ns,
            )

            clipped_end_ns = min(
                end_ns,
                outer_end_ns,
            )

            if (
                clipped_end_ns
                <= clipped_start_ns
            ):
                continue

            kernel_name = ""

            for candidate in (
                demangled_name,
                short_name,
                mangled_name,
            ):
                resolved = (
                    resolve_string(
                        candidate,
                        string_ids,
                    )
                )

                if resolved:
                    kernel_name = (
                        resolved
                    )
                    break

            if not kernel_name:
                kernel_name = (
                    "<unknown>"
                )

            outer_events.append(
                {
                    "start":
                        clipped_start_ns,

                    "end":
                        clipped_end_ns,

                    "duration_ns":
                        clipped_end_ns
                        - clipped_start_ns,

                    "name":
                        kernel_name,

                    "family":
                        kernel_family(
                            kernel_name
                        ),

                    "graph_node":
                        (
                            graph_node_id
                            not in (
                                None,
                                0,
                            )
                        ),
                }
            )

        if not outer_events:
            print(
                f"\n{regime}: "
                "NO KERNEL EVENTS"
            )

            continue

        islands = split_into_islands(
            outer_events
        )

        benchmark_duration_s = (
            benchmark[
                "benchmark_duration_s"
            ]
        )

        main_island = (
            choose_main_island(
                islands,
                benchmark_duration_s,
            )
        )

        for (
            island_index,
            island,
        ) in enumerate(
            islands,
            start=1,
        ):
            island_rows.append(
                {
                    "regime":
                        regime,

                    "island_index":
                        island_index,

                    "kernel_count":
                        len(island),

                    "span_s":
                        island_span_s(
                            island
                        ),

                    "selected":
                        (
                            island
                            is main_island
                        ),
                }
            )

        events = main_island

        intervals = [
            (
                event["start"],
                event["end"],
            )
            for event in events
        ]

        merged = (
            merge_overlapping_intervals(
                intervals
            )
        )

        active_start_ns = (
            merged[0][0]
        )

        active_end_ns = (
            merged[-1][1]
        )

        active_span_ns = (
            active_end_ns
            - active_start_ns
        )

        busy_union_ns = sum(
            end_ns - start_ns
            for (
                start_ns,
                end_ns,
            ) in merged
        )

        gaps_ns = [
            (
                merged[index][0]
                - merged[
                    index - 1
                ][1]
            )
            for index in range(
                1,
                len(merged),
            )
            if (
                merged[index][0]
                > merged[
                    index - 1
                ][1]
            )
        ]

        durations_ns = [
            event[
                "duration_ns"
            ]
            for event in events
        ]

        summed_kernel_ns = sum(
            durations_ns
        )

        family_time_ns = (
            defaultdict(int)
        )

        family_count = (
            defaultdict(int)
        )

        kernel_time_ns = (
            defaultdict(int)
        )

        kernel_count = (
            defaultdict(int)
        )

        graph_kernel_count = 0

        for event in events:
            family_time_ns[
                event["family"]
            ] += (
                event[
                    "duration_ns"
                ]
            )

            family_count[
                event["family"]
            ] += 1

            kernel_time_ns[
                event["name"]
            ] += (
                event[
                    "duration_ns"
                ]
            )

            kernel_count[
                event["name"]
            ] += 1

            if event[
                "graph_node"
            ]:
                graph_kernel_count += 1

        failed_requests = (
            benchmark[
                "failed_requests"
            ]
        )

        if failed_requests is None:
            failed_requests = 0

        performance_valid = (
            failed_requests == 0
        )

        output_tokens = (
            benchmark[
                "output_tokens"
            ]
        )

        if (
            performance_valid
            and output_tokens
        ):
            busy_ns_per_output_token = (
                busy_union_ns
                / output_tokens
            )

        else:
            busy_ns_per_output_token = (
                None
            )

        summary = {
            "regime":
                regime,

            "performance_valid":
                performance_valid,

            "successful_requests":
                benchmark[
                    "successful_requests"
                ],

            "failed_requests":
                failed_requests,

            "input_tokens":
                benchmark[
                    "input_tokens"
                ],

            "output_tokens":
                output_tokens,

            "benchmark_duration_s":
                benchmark_duration_s,

            "output_throughput":
                benchmark[
                    "output_throughput"
                ],

            "outer_client_window_s":
                (
                    outer_end_ns
                    - outer_start_ns
                )
                / 1e9,

            "candidate_gpu_islands":
                len(islands),

            "selected_gpu_island_s":
                active_span_ns
                / 1e9,

            "kernel_count":
                len(events),

            "kernel_launch_rate_per_s":
                (
                    len(events)
                    / (
                        active_span_ns
                        / 1e9
                    )
                ),

            "summed_kernel_time_s":
                summed_kernel_ns
                / 1e9,

            "kernel_busy_union_s":
                busy_union_ns
                / 1e9,

            # This means "there was at least one
            # traced kernel active" and must NOT
            # be interpreted as SM utilization.
            "kernel_present_fraction_pct":
                (
                    100.0
                    * busy_union_ns
                    / active_span_ns
                ),

            "kernel_duration_mean_us":
                (
                    sum(
                        durations_ns
                    )
                    / len(
                        durations_ns
                    )
                    / 1e3
                ),

            "kernel_duration_p50_us":
                (
                    percentile(
                        durations_ns,
                        50,
                    )
                    / 1e3
                ),

            "kernel_duration_p95_us":
                (
                    percentile(
                        durations_ns,
                        95,
                    )
                    / 1e3
                ),

            "kernel_duration_p99_us":
                (
                    percentile(
                        durations_ns,
                        99,
                    )
                    / 1e3
                ),

            "gpu_gap_p50_us":
                (
                    percentile(
                        gaps_ns,
                        50,
                    )
                    / 1e3
                    if gaps_ns
                    else 0.0
                ),

            "gpu_gap_p95_us":
                (
                    percentile(
                        gaps_ns,
                        95,
                    )
                    / 1e3
                    if gaps_ns
                    else 0.0
                ),

            "gpu_gap_p99_us":
                (
                    percentile(
                        gaps_ns,
                        99,
                    )
                    / 1e3
                    if gaps_ns
                    else 0.0
                ),

            "graph_kernel_pct":
                (
                    100.0
                    * graph_kernel_count
                    / len(events)
                ),

            "busy_ns_per_output_token":
                (
                    busy_ns_per_output_token
                ),
        }

        summaries.append(
            summary
        )

        for family in sorted(
            family_time_ns,
            key=family_time_ns.get,
            reverse=True,
        ):
            family_rows.append(
                {
                    "regime":
                        regime,

                    "family":
                        family,

                    "kernel_count":
                        family_count[
                            family
                        ],

                    "kernel_time_s":
                        (
                            family_time_ns[
                                family
                            ]
                            / 1e9
                        ),

                    "kernel_time_share_pct":
                        (
                            100.0
                            * family_time_ns[
                                family
                            ]
                            / summed_kernel_ns
                        ),
                }
            )

        top_kernel_names = sorted(
            kernel_time_ns,
            key=kernel_time_ns.get,
            reverse=True,
        )[:20]

        for (
            rank,
            kernel_name,
        ) in enumerate(
            top_kernel_names,
            start=1,
        ):
            top_kernel_rows.append(
                {
                    "regime":
                        regime,

                    "rank":
                        rank,

                    "kernel":
                        kernel_name,

                    "family":
                        kernel_family(
                            kernel_name
                        ),

                    "count":
                        kernel_count[
                            kernel_name
                        ],

                    "time_s":
                        (
                            kernel_time_ns[
                                kernel_name
                            ]
                            / 1e9
                        ),

                    "share_pct":
                        (
                            100.0
                            * kernel_time_ns[
                                kernel_name
                            ]
                            / summed_kernel_ns
                        ),
                }
            )

    output_dir = (
        exp_dir
        / "analysis"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    write_csv(
        output_dir
        / "regime_summary.csv",
        summaries,
    )

    write_csv(
        output_dir
        / "kernel_family_summary.csv",
        family_rows,
    )

    write_csv(
        output_dir
        / "top_kernels.csv",
        top_kernel_rows,
    )

    write_csv(
        output_dir
        / "gpu_islands.csv",
        island_rows,
    )

    print(
        "\n=== REGIME SUMMARY ==="
    )

    for summary in summaries:
        print()
        print(
            f"{summary['regime']}:"
        )

        keys = (
            "performance_valid",
            "successful_requests",
            "failed_requests",
            "output_tokens",
            "output_throughput",
            "benchmark_duration_s",
            "outer_client_window_s",
            "candidate_gpu_islands",
            "selected_gpu_island_s",
            "kernel_count",
            "kernel_launch_rate_per_s",
            "kernel_busy_union_s",
            "kernel_present_fraction_pct",
            "kernel_duration_p50_us",
            "kernel_duration_p95_us",
            "gpu_gap_p50_us",
            "gpu_gap_p95_us",
            "graph_kernel_pct",
            "busy_ns_per_output_token",
        )

        for key in keys:
            print(
                f"  {key}: "
                f"{summary[key]}"
            )

        if not summary[
            "performance_valid"
        ]:
            print(
                "  NOTE: failed requests "
                "were present; throughput "
                "and per-token normalization "
                "must be treated as "
                "diagnostic only."
            )

    print(
        "\n=== KERNEL FAMILY MIX ==="
    )

    current_regime = None

    for row in family_rows:
        if (
            row["regime"]
            != current_regime
        ):
            current_regime = (
                row["regime"]
            )

            print(
                f"\n{current_regime}:"
            )

        print(
            "  "
            f"{row['family']:14s} "
            f"{row['kernel_time_share_pct']:7.2f}% "
            "time, "
            f"{row['kernel_count']} kernels"
        )

    print(
        "\n=== GPU ISLANDS ==="
    )

    current_regime = None

    for row in island_rows:
        if (
            row["regime"]
            != current_regime
        ):
            current_regime = (
                row["regime"]
            )

            print(
                f"\n{current_regime}:"
            )

        marker = (
            "*"
            if row["selected"]
            else " "
        )

        print(
            f" {marker} island "
            f"{row['island_index']}: "
            f"{row['span_s']:.3f}s, "
            f"{row['kernel_count']} kernels"
        )

    con.close()


if __name__ == "__main__":
    main()
