#!/usr/bin/env python3
"""Generate the benchmark chart PNG for the README."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "benchmarks" / "metrics_history.csv"
OUTPUT = ROOT / "assets" / "benchmark-chart-v2.png"


def load_entries() -> list[tuple[str, float, str]]:
    entries: list[tuple[str, float, str]] = []
    with HISTORY.open(newline="") as f:
        for row in csv.DictReader(f):
            runtime = row["runtime"]
            mode = row["mode"]
            decode_tok_s = float(row["decode_tok_s"])

            if "mxfp4" in mode:
                label = "HRM-mlx 4-bit"
            elif runtime == "HRM-mlx":
                label = "HRM-mlx BF16"
            elif runtime == "PyTorch MPS":
                label = "PyTorch MPS BF16"
            else:
                continue

            color = "#5f8ff0" if label == "HRM-mlx 4-bit" else "#454b68"
            entries.append((label, decode_tok_s, color))

    return sorted(entries, key=lambda entry: entry[1], reverse=True)


entries = load_entries()
labels = [entry[0] for entry in entries]
values = [entry[1] for entry in entries]
colors = [entry[2] for entry in entries]

BG = "#0d1117"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
})

fig, ax = plt.subplots(figsize=(8.2, 4.2))
fig.subplots_adjust(left=0.28, right=0.90, top=0.83, bottom=0.08)

bars = ax.barh(range(len(entries)), values, height=0.54, color=colors, edgecolor="none")

for i, (bar, val) in enumerate(zip(bars, values)):
    is_ours = labels[i] == "HRM-mlx 4-bit"
    ax.text(
        val + 1.0,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}",
        va="center",
        fontsize=10.5,
        fontweight="bold" if is_ours else "normal",
        color="#ffffff" if is_ours else "#8b95a5",
        fontfamily="monospace",
    )

ax.set_yticks(range(len(entries)))
tick_labels = ax.set_yticklabels(labels, fontsize=10.0, color="#c0c8d4")
for tick, label in zip(tick_labels, labels):
    if label == "HRM-mlx 4-bit":
        tick.set_color("#ffffff")
        tick.set_fontweight("bold")

ax.invert_yaxis()
ax.set_xlim(0, 60)
ax.xaxis.set_visible(False)
ax.spines[:].set_visible(False)
ax.tick_params(left=False, bottom=False)

fig.text(
    0.5,
    0.95,
    "Throughput on HRM-Text-1B",
    fontsize=14,
    fontweight="bold",
    color="#e6eaf0",
    ha="center",
)
fig.text(
    0.5,
    0.88,
    "decode tok/s  ·  512-token prompt, 128-token generation  ·  MacBook Pro M4 Max",
    fontsize=9.5,
    color="#6b7585",
    ha="center",
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
print(f"Saved to {OUTPUT}")
plt.close()
