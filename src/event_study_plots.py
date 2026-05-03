#!/usr/bin/env python3
"""Shared plotting helpers for the per-event study scripts.

Two charts per event, single panel each (figsize ~ 7×5):
  - Skew chart:  t0 + t1 puts/calls dots-only, linear y-axis 0..160%
  - Decomp chart: F1..F6 bars + ΔVIX_actual / Σ reference lines

Locked colour and styling conventions (carried over from the prior
combined chart):
  t0 puts  #1f4e79 (dark blue)     t0 calls  #6699cc (light blue)
  t1 puts  #c65d00 (dark orange)   t1 calls  #ff9933 (light orange)
  Scatter alpha 0.7, marker size s=18.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SKEW_COLORS = {
    "t0_put":  "#1f4e79",
    "t0_call": "#6699cc",
    "t1_put":  "#c65d00",
    "t1_call": "#ff9933",
}
SKEW_MARKER_SIZE = 18
SKEW_ALPHA = 0.7


def plot_skew_panel(ax, t0_res: dict, t1_res: dict,
                    t0_str: str, t1_str: str, title: str) -> None:
    """Render the t0/t1 put-and-call skew dots on a single Axes.

    Linear y-axis, ticks every 32% from 0 to 160. CBOE Exhibit 17 style:
    dots only, no connecting line, no spline."""
    series = [
        (t0_res["put_skew_30d"],  f"Put {t0_str}",  SKEW_COLORS["t0_put"]),
        (t0_res["call_skew_30d"], f"Call {t0_str}", SKEW_COLORS["t0_call"]),
        (t1_res["put_skew_30d"],  f"Put {t1_str}",  SKEW_COLORS["t1_put"]),
        (t1_res["call_skew_30d"], f"Call {t1_str}", SKEW_COLORS["t1_call"]),
    ]
    for skew, label, color in series:
        if not skew:
            continue
        ks = sorted(skew.keys())
        ys = [skew[k] for k in ks]
        ax.scatter(ks, ys, s=SKEW_MARKER_SIZE, color=color,
                   alpha=SKEW_ALPHA, label=label, zorder=2)

    ax.set_yticks([0, 32, 64, 96, 128, 160])
    ax.set_yticklabels(["0%", "32%", "64%", "96%", "128%", "160%"])
    ax.set_ylim(0, 160)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="major", axis="y")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def plot_decomp_panel(ax, rec: dict, dvix: float, sum_factors: float) -> None:
    """Render the F1..F6 bars + ΔVIX/Σ reference lines on a single Axes."""
    labels = ["F1\nSticky", "F2\nParallel", "F3\nPut Skew",
              "F4\nCall Skew", "F5\nDownside\nConv", "F6\nUpside\nConv"]
    values = [rec["F1"], rec["F2"], rec["F3"], rec["F4"], rec["F5"], rec["F6"]]
    colors = ["#7f7f7f", "#4682b4", "#8b0000",
              "#006400", "#cd5c5c", "#90ee90"]

    bars = ax.bar(labels, values, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(dvix, color="black", lw=1.5, ls="-",
               label=f"ΔVIX actual = {dvix:+.2f}")
    ax.axhline(sum_factors, color="black", lw=1.5, ls="--",
               label=f"Sum of factors = {sum_factors:+.2f}")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)

    ax.set_title(
        f"{rec['name']}: ΔVIX = {dvix:+.2f},  Σ = {sum_factors:+.2f}",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Vol-points")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 3 if v >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=8,
        )


def make_skew_chart(rec: dict, out_path: str) -> None:
    """Single-event skew chart written to out_path."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plot_skew_panel(ax, rec["_t0_res"], rec["_t1_res"],
                    rec["t0"], rec["t1"], rec["name"])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_decomp_chart(rec: dict, dvix: float, sum_factors: float,
                      out_path: str) -> None:
    """Single-event decomposition bar chart written to out_path."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plot_decomp_panel(ax, rec, dvix, sum_factors)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
