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

# Remove obsolete publication figures so stale files cannot be mistaken
# for the final Article #1 figure set.
obsolete_stems = (
    "01_throughput_knee_repeatability",
    "02_ttft_by_concurrency",
    "03_tpot_by_concurrency",
    "04_cleanroom_reproduction",
    "05_kernel_family_cost",
    "06_fa2_path_decomposition",
    "07_mixed_fa2_steady_scaling",
    "02_latency_dominated",
    "03_gpu_cost_localization",
    "04_mixed_fa2_steady_scaling",
    "S1_cleanroom_reproduction",
)

for stem in obsolete_stems:
    for ext in ("png", "svg"):
        (FIG / f"{stem}.{ext}").unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Figure 1. Canonical throughput knee + repeatability.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))

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

canonical_tp_means = means(canonical, "output_throughput")

ax.plot(
    concurrencies,
    canonical_tp_means,
    marker="o",
    linewidth=2.8,
    label="mean",
)

mean_drop_pct = (
    100.0
    * (canonical_tp_means[1] - canonical_tp_means[2])
    / canonical_tp_means[1]
)

ax.annotate(
    f"C128 → C256: −{mean_drop_pct:.2f}% mean throughput\n"
    "C128 > C256 in 3/3 repeats",
    xy=(256, canonical_tp_means[2]),
    xycoords="data",
    xytext=(0.57, 0.70),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
    ha="center",
)

ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("Output throughput (tok/s)")
ax.set_title("Canonical saturation sweep: throughput knee")
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

save(fig, "01_throughput_knee_repeatability")


# ---------------------------------------------------------------------
# Figure 2. C256 is dominated on latency: TTFT + TPOT.
# ---------------------------------------------------------------------
fig, (ax_ttft, ax_tpot) = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.5),
)

# TTFT panel.
ax_ttft.plot(
    concurrencies,
    means(canonical, "mean_ttft_ms"),
    marker="o",
    linewidth=2.0,
    label="Mean TTFT",
)
ax_ttft.plot(
    concurrencies,
    means(canonical, "p99_ttft_ms"),
    marker="o",
    linewidth=2.0,
    label="P99 TTFT",
)

ax_ttft.set_xlabel("Maximum concurrency")
ax_ttft.set_ylabel("Latency (ms)")
ax_ttft.set_title("TTFT")
ax_ttft.set_xticks(concurrencies)
ax_ttft.legend(frameon=False)
ax_ttft.grid(axis="y", alpha=0.2)


# TPOT panel.
ax_tpot.plot(
    concurrencies,
    means(canonical, "mean_tpot_ms"),
    marker="o",
    linewidth=2.0,
    label="Mean TPOT",
)
ax_tpot.plot(
    concurrencies,
    means(canonical, "p99_tpot_ms"),
    marker="o",
    linewidth=2.0,
    label="P99 TPOT",
)

ax_tpot.set_xlabel("Maximum concurrency")
ax_tpot.set_ylabel("Latency (ms)")
ax_tpot.set_title("TPOT")
ax_tpot.set_xticks(concurrencies)
ax_tpot.legend(frameon=False)
ax_tpot.grid(axis="y", alpha=0.2)

save(fig, "02_latency_dominated")


# ---------------------------------------------------------------------
# Figure 3. Localize the post-knee GPU-cost increase.
# Left: C128→C256 kernel-family delta.
# Right: FA2 path decomposition.
# ---------------------------------------------------------------------
fig, (ax_family, ax_fa2) = plt.subplots(
    1,
    2,
    figsize=(11.2, 4.7),
)

kernel_families = (
    ("gemm_us_per_output_token", "GEMM"),
    ("splitkv_attention_us_per_output_token", "Split-KV attention"),
    ("triton_other_us_per_output_token", "Triton other"),
    ("other_us_per_output_token", "Other"),
)

family_labels = []
family_deltas = []

for metric, label in kernel_families:
    family_labels.append(label)
    family_deltas.append(
        mechanism[metric][256] - mechanism[metric][128]
    )

bars = ax_family.barh(
    range(len(family_labels)),
    family_deltas,
)

ax_family.axvline(0, linewidth=1.0)
ax_family.set_yticks(
    range(len(family_labels)),
    labels=family_labels,
)
ax_family.invert_yaxis()

delta_limit = max(abs(v) for v in family_deltas)
ax_family.set_xlim(
    -delta_limit * 1.35,
    delta_limit * 1.35,
)

for bar, value in zip(bars, family_deltas):
    ax_family.text(
        value + (0.45 if value >= 0 else -0.45),
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.2f}",
        va="center",
        ha="left" if value >= 0 else "right",
    )

ax_family.set_xlabel(
    "C128→C256 Δ summed kernel duration\n"
    "(µs/output token)"
)
ax_family.set_title("Kernel-family localization")
ax_family.grid(axis="x", alpha=0.2)


for metric, label in (
    (
        "fa2_decode_us_per_output_token",
        "gridX=1 decode-oriented",
    ),
    (
        "fa2_mixed_prefill_us_per_output_token",
        "gridX=9 mixed/prefill-containing",
    ),
):
    ax_fa2.plot(
        concurrencies,
        [mechanism[metric][c] for c in concurrencies],
        marker="o",
        linewidth=2.2,
        label=label,
    )

ax_fa2.set_xlabel("Maximum concurrency")
ax_fa2.set_ylabel(
    "Summed FA2 duration\n"
    "(µs/output token)"
)
ax_fa2.set_title("FA2 path decomposition")
ax_fa2.set_xticks(concurrencies)
ax_fa2.legend(frameon=False)
ax_fa2.grid(axis="y", alpha=0.2)

save(fig, "03_gpu_cost_localization")


# ---------------------------------------------------------------------
# Figure 4. Sustained mixed/prefill population vs kernel duration.
#
# IMPORTANT: scatter only. Do not connect C64/C128/C256 because doing so
# would visually imply a general linear scaling law that we do not claim.
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 4.8))

xs = [
    mechanism["steady_mixed_batch"][c]
    for c in concurrencies
]
ys = [
    mechanism["steady_mixed_kernel_us"][c]
    for c in concurrencies
]

ax.scatter(xs, ys, s=70)

for c, x, y in zip(concurrencies, xs, ys):
    ax.annotate(
        f"C{c}",
        (x, y),
        textcoords="offset points",
        xytext=(7, 7),
    )

population_ratio = (
    mechanism["steady_mixed_batch"][256]
    / mechanism["steady_mixed_batch"][128]
)

kernel_ratio = (
    mechanism["steady_mixed_kernel_us"][256]
    / mechanism["steady_mixed_kernel_us"][128]
)

ax.annotate(
    f"C128 → C256\n"
    f"{population_ratio:.4f}× population\n"
    f"{kernel_ratio:.4f}× kernel duration",
    xy=(xs[2], ys[2]),
    xycoords="data",
    xytext=(0.58, 0.48),
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->"},
    ha="center",
)

ax.set_xlabel("Steady mixed/prefill sequence population")
ax.set_ylabel("Mean mixed/prefill FA2 kernel duration (µs)")
ax.set_title(
    "C128→C256: sustained population and kernel duration both ~2×"
)
ax.grid(alpha=0.2)

save(fig, "04_mixed_fa2_steady_scaling")


# ---------------------------------------------------------------------
# Supporting figure only — not one of the four main article figures.
# ---------------------------------------------------------------------
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
    label="Fresh-host clean-room reproduction",
)

ax.set_xlabel("Maximum concurrency")
ax.set_ylabel("Mean output throughput (tok/s)")
ax.set_title(
    "Clean-room reproduction preserves the regime ordering"
)
ax.set_xticks(concurrencies)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

save(fig, "S1_cleanroom_reproduction")


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
