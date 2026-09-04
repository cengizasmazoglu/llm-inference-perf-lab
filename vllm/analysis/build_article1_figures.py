#!/usr/bin/env python3

import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("vllm/results/article1")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CANONICAL = ROOT / "canonical_repeatability.csv"
CLEANROOM = ROOT / "cleanroom_repeatability.csv"
MECHANISM = ROOT / "mechanism_summary.csv"


def read_repeatability(path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "rep": row["rep"],
                "concurrency": int(row["concurrency"]),
                "completed": int(row["completed"]),
                "failed": int(row["failed"]),
                "output_throughput": float(row["output_throughput"]),
                "mean_ttft_ms": float(row["mean_ttft_ms"]),
                "p99_ttft_ms": float(row["p99_ttft_ms"]),
                "mean_tpot_ms": float(row["mean_tpot_ms"]),
                "p99_tpot_ms": float(row["p99_tpot_ms"]),
            })
    return rows


def read_mechanism(path):
    values = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            values[row["metric"]] = {
                64: float(row["C64"]),
                128: float(row["C128"]),
                256: float(row["C256"]),
                "unit": row["unit"],
            }
    return values


def by_concurrency(rows, key):
    out = {}
    for c in (64, 128, 256):
        out[c] = [r[key] for r in rows if r["concurrency"] == c]
    return out


def means(rows, key):
    grouped = by_concurrency(rows, key)
    return [statistics.mean(grouped[c]) for c in (64, 128, 256)]


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


canonical = read_repeatability(CANONICAL)
cleanroom = read_repeatability(CLEANROOM)
mechanism = read_mechanism(MECHANISM)

concurrencies = [64, 128, 256]

# 1. Throughput knee + repeatability.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for rep in sorted({r["rep"] for r in canonical}):
    subset = sorted(
        (r for r in canonical if r["rep"] == rep),
        key=lambda r: r["concurrency"],
    )
    ax.plot(
        [r["concurrency"] for r in subset],
        [r["output_throughput"] for r in subset],
        marker="o",
        linewidth=1.2,
        alpha=0.45,
        label=rep,
    )
ax.plot(
    concurrencies,
    means(canonical, "output_throughput"),
    marker="o",
    linewidth=2.8,
    label="mean",
)
ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("Output throughput (tok/s)")
ax.set_title("Canonical saturation sweep: throughput knee")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "01_throughput_knee_repeatability")


# 2. TTFT dominance.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.plot(
    concurrencies,
    means(canonical, "mean_ttft_ms"),
    marker="o",
    linewidth=2.0,
    label="Mean TTFT",
)
ax.plot(
    concurrencies,
    means(canonical, "p99_ttft_ms"),
    marker="o",
    linewidth=2.0,
    label="P99 TTFT",
)
ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("TTFT (ms)")
ax.set_title("TTFT worsens sharply beyond the throughput knee")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "02_ttft_by_concurrency")


# 3. TPOT dominance.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.plot(
    concurrencies,
    means(canonical, "mean_tpot_ms"),
    marker="o",
    linewidth=2.0,
    label="Mean TPOT",
)
ax.plot(
    concurrencies,
    means(canonical, "p99_tpot_ms"),
    marker="o",
    linewidth=2.0,
    label="P99 TPOT",
)
ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("TPOT (ms)")
ax.set_title("Token generation latency is also worse at C256")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "03_tpot_by_concurrency")


# 4. Clean-room reproduction.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.plot(
    concurrencies,
    means(canonical, "output_throughput"),
    marker="o",
    linewidth=2.0,
    label="Canonical",
)
ax.plot(
    concurrencies,
    means(cleanroom, "output_throughput"),
    marker="o",
    linewidth=2.0,
    label="Independent clean-room reproduction",
)
ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("Mean output throughput (tok/s)")
ax.set_title("The regime ordering survives independent reproduction")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "04_cleanroom_reproduction")


# 5. Kernel-family normalized cost.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for metric, label in (
    ("gemm_us_per_output_token", "GEMM"),
    ("splitkv_attention_us_per_output_token", "Split-KV attention"),
    ("triton_other_us_per_output_token", "Triton other"),
    ("other_us_per_output_token", "Other"),
):
    ax.plot(
        concurrencies,
        [mechanism[metric][c] for c in concurrencies],
        marker="o",
        linewidth=2.0,
        label=label,
    )
ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("Summed kernel duration (µs/output token)")
ax.set_title("Post-knee cost increase is localized to attention")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "05_kernel_family_cost")


# 6. FA2 path decomposition.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for metric, label in (
    ("fa2_decode_us_per_output_token", "gridX=1 decode-oriented"),
    ("fa2_mixed_prefill_us_per_output_token", "gridX=9 mixed/prefill"),
):
    ax.plot(
        concurrencies,
        [mechanism[metric][c] for c in concurrencies],
        marker="o",
        linewidth=2.2,
        label=label,
    )
ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("FA2 kernel duration (µs/output token)")
ax.set_title("The expensive path is mixed/prefill FA2, not decode FA2")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "06_fa2_path_decomposition")


# 7. Steady mixed/prefill population vs kernel time.
fig, ax = plt.subplots(figsize=(7.2, 4.5))
xs = [mechanism["steady_mixed_batch"][c] for c in concurrencies]
ys = [mechanism["steady_mixed_kernel_us"][c] for c in concurrencies]
ax.plot(xs, ys, marker="o", linewidth=2.0)
for c, x, y in zip(concurrencies, xs, ys):
    ax.annotate(
        f"C{c}",
        (x, y),
        textcoords="offset points",
        xytext=(6, 6),
    )
ax.set_xlabel("Steady mixed/prefill sequence population")
ax.set_ylabel("Mean flash_fwd_splitkv_kernel duration (µs)")
ax.set_title("C128→C256 doubles both sustained population and kernel time")
ax.grid(alpha=0.2)
save(fig, "07_mixed_fa2_steady_scaling")


# Text summary used for article drafting.
can_tp = by_concurrency(canonical, "output_throughput")
clean_tp = by_concurrency(cleanroom, "output_throughput")

def cv_pct(values):
    return 100.0 * statistics.stdev(values) / statistics.mean(values)

can_adv = [
    100.0 * (a - b) / b
    for a, b in zip(can_tp[128], can_tp[256])
]
clean_adv = [
    100.0 * (a - b) / b
    for a, b in zip(clean_tp[128], clean_tp[256])
]

summary = f"""Canonical means (tok/s):
C64  = {statistics.mean(can_tp[64]):.2f}
C128 = {statistics.mean(can_tp[128]):.2f}
C256 = {statistics.mean(can_tp[256]):.2f}

Canonical CV:
C64  = {cv_pct(can_tp[64]):.3f}%
C128 = {cv_pct(can_tp[128]):.3f}%
C256 = {cv_pct(can_tp[256]):.3f}%

Canonical paired C128 advantage over C256:
{statistics.mean(can_adv):.3f}%

Clean-room means (tok/s):
C64  = {statistics.mean(clean_tp[64]):.2f}
C128 = {statistics.mean(clean_tp[128]):.2f}
C256 = {statistics.mean(clean_tp[256]):.2f}

Clean-room paired C128 advantage over C256:
{statistics.mean(clean_adv):.3f}%

007B:
GPU busy C128→C256 = {mechanism["gpu_busy_pct"][128]:.2f}% → {mechanism["gpu_busy_pct"][256]:.2f}%
steady mixed batch C128→C256 = {mechanism["steady_mixed_batch"][128]:.2f} → {mechanism["steady_mixed_batch"][256]:.2f}
steady mixed kernel C128→C256 = {mechanism["steady_mixed_kernel_us"][128]:.2f} → {mechanism["steady_mixed_kernel_us"][256]:.2f} µs
mixed/prefill FA2 C128→C256 = {mechanism["fa2_mixed_prefill_us_per_output_token"][128]:.2f} → {mechanism["fa2_mixed_prefill_us_per_output_token"][256]:.2f} µs/output-token
"""
(ROOT / "article1_numeric_summary.txt").write_text(summary)
print(summary)
print(f"Figures written to: {FIG}")
