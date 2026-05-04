"""Plot F1-F6 breakdown from vix_decomposition.csv (output of vix_decomposition.py)."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "output", "vix_decomposition.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "vix_decomposition_chart.png")


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    dates = df["date"].to_numpy()

    f1 = df["sticky_strike_contribution"].to_numpy()
    f2 = df["parallel_contribution"].to_numpy()
    f3 = df["put_skew_contribution"].to_numpy()
    f4 = df["call_skew_contribution"].to_numpy()
    f5 = df["put_convexity_contribution"].to_numpy()
    f6 = df["call_convexity_contribution"].to_numpy()
    dvix = df["delta_vix"].to_numpy()
    sumf = (f1 + f2 + f3 + f4 + f5 + f6)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(dates, f1, label="F1 Sticky Strike", lw=1.4)
    ax.plot(dates, f2, label="F2 Parallel Shift", lw=1.4)
    ax.plot(dates, f3, label="F3 Put Skew", lw=1.4)
    ax.plot(dates, f4, label="F4 Call Skew", lw=1.4)
    ax.plot(dates, f5, label="F5 Put Convexity", lw=1.4, color="purple")
    ax.plot(dates, f6, label="F6 Call Convexity", lw=1.4, color="brown")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Contribution (vol-pts)")
    ax.set_title("F1–F6 daily contributions (vix_decomposition.py)")
    ax.legend(loc="best", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(dates, dvix, label="ΔVIX (synthetic)", color="black", lw=1.8)
    ax.plot(dates, sumf, label="Σ F1..F6", color="red", lw=1.2, linestyle="--")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ΔVIX (vol-pts)")
    ax.set_title("ΔVIX vs Σ F1..F6")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle("VIX Decomposition – Q4 2023 (vix_decomposition.py 7-factor → F1..F6 + residual)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
