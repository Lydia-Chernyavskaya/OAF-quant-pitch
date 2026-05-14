#!/usr/bin/env python3
"""3-panel VIX-analysis & decomposition charts for the three pipelines.

For each of:
    1. vix_decomposition.py            (OLS-projection pipeline)
    2. vix_pipeline_variance.py        (variance-space scalar perturbation)
    3. vix_pipeline_CBOE_bucket_nd2.py (CBOE bucket recomputation, N(d2))

…this script reads the corresponding output CSV under src/output/ and
produces a chart with three vertically-stacked panels:

    Panel 1 — VIX Computed vs VIX Actual (line plot)
    Panel 2 — Decomposition factors F1..F4 (sticky / parallel / put / call)
    Panel 3 — Decomposition factors F5..F6 (downside conv / upside conv)

Each chart is titled
    "VIX Analysis & Decomposition (2023, <method-name>)"

The script also prints (and writes) a macro breakdown table comparing
each method's VIX_computed series against VIX_actual on the 2023 sample:

    method, n_days, mean_err, rmse, mae, std_resid, corr, max_abs_err

Output files (src/output/):
    vix_decomposition_OLS_panels.png
    vix_decomposition_variance_panels.png
    vix_decomposition_CBOE_bucket_panels.png
    vix_decomposition_method_comparison.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_DIR = HERE / "output"
MARKET_PARQUET = PROJECT_ROOT / "data" / "processed" / "market_data.parquet"


# ============================================================================
# Per-pipeline CSV adapters
# Each adapter returns a tidy DataFrame indexed by date with the columns:
#     VIX_computed, VIX_actual,
#     F1, F2, F3, F4, F5, F6
# F1..F6 may be NaN on the first row (no t-1 available) — that's expected.
# ============================================================================

def _load_actual_vix() -> pd.Series:
    """VIX_actual series from the cached yfinance market data parquet."""
    mkt = pd.read_parquet(MARKET_PARQUET)
    mkt.index = pd.to_datetime(mkt.index).normalize()
    return mkt["vix_close"].rename("VIX_actual")


def load_OLS(csv_path: Path) -> pd.DataFrame:
    """OLS pipeline (vix_decomposition.csv).

    The OLS CSV does not carry VIX_actual; we merge it from the cached
    market_data.parquet. The 'computed' VIX is vix_bs_t (the BS-from-IV
    reconstruction at the end of the chain).
    """
    raw = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    raw.index = raw.index.normalize()

    out = pd.DataFrame(index=raw.index)
    out["VIX_computed"] = raw["vix_bs_t"]
    out["F1"] = raw["sticky_strike_contribution"]
    out["F2"] = raw["parallel_contribution"]
    out["F3"] = raw["put_skew_contribution"]
    out["F4"] = raw["call_skew_contribution"]
    out["F5"] = raw["put_convexity_contribution"]
    out["F6"] = raw["call_convexity_contribution"]

    actual = _load_actual_vix()
    out["VIX_actual"] = actual.reindex(out.index)
    return out


def load_variance(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    raw.index = raw.index.normalize()
    out = pd.DataFrame(index=raw.index)
    out["VIX_computed"] = raw["VIX_computed"]
    out["VIX_actual"]   = raw["VIX_actual"]
    for f in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        out[f] = raw[f]
    return out


def load_CBOE_bucket(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    raw.index = raw.index.normalize()
    out = pd.DataFrame(index=raw.index)
    out["VIX_computed"] = raw["VIX_computed"]
    out["VIX_actual"]   = raw["VIX_actual"]
    out["F1"] = raw["F1_sticky_strike"]
    out["F2"] = raw["F2_parallel_shift"]
    out["F3"] = raw["F3_put_skew_grad"]
    out["F4"] = raw["F4_call_skew_grad"]
    out["F5"] = raw["F5_downside_conv"]
    out["F6"] = raw["F6_upside_conv"]
    return out


# ============================================================================
# Chart
# ============================================================================

F14_COLORS = {"F1": "steelblue", "F2": "darkorange",
              "F3": "darkgreen",  "F4": "crimson"}
F56_COLORS = {"F5": "purple", "F6": "saddlebrown"}


def make_panels(df: pd.DataFrame, method_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ── Panel 1: VIX Computed vs Actual ─────────────────────────────────
    ax = axes[0]
    ax.plot(df.index, df["VIX_computed"],
            label="VIX Computed", color="steelblue", lw=1.8)
    ax.plot(df.index, df["VIX_actual"],
            label="VIX Actual", color="darkorange", lw=1.2, alpha=0.85)
    ax.set_ylabel("VIX")
    ax.set_title("VIX Computed vs Actual")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 2: F1..F4 ────────────────────────────────────────────────
    ax = axes[1]
    for label, col in [("F1 Sticky Strike",   "F1"),
                       ("F2 Parallel Shift",  "F2"),
                       ("F3 Put Skew Grad",   "F3"),
                       ("F4 Call Skew Grad",  "F4")]:
        ax.plot(df.index, df[col], label=label,
                color=F14_COLORS[col], lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Vol Change (pts)")
    ax.set_title("Decomposition Factors F1–F4")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ── Panel 3: F5..F6 ────────────────────────────────────────────────
    ax = axes[2]
    for label, col in [("F5 Downside Conv", "F5"),
                       ("F6 Upside Conv",   "F6")]:
        ax.plot(df.index, df[col], label=label,
                color=F56_COLORS[col], lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Vol Change (pts)")
    ax.set_title("Decomposition Factors F5–F6")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle(f"VIX Analysis & Decomposition (2023, {method_label})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Macro breakdown
# ============================================================================

def macro_stats(df: pd.DataFrame, method_label: str) -> dict:
    """Closeness of VIX_computed vs VIX_actual on the 2023 sample.

    Reports:
        n_days       — paired non-NaN count
        mean_err     — mean(computed − actual)        (bias, vol-pts)
        rmse         — sqrt(mean((c − a)²))           (vol-pts)
        mae          — mean(|c − a|)                  (vol-pts)
        std_resid    — sample std of (c − a)          (vol-pts)
        corr         — Pearson corr(c, a)
        max_abs_err  — max |c − a|                    (vol-pts)
        actual_std   — std of VIX_actual over the window (for context)
    """
    paired = df[["VIX_computed", "VIX_actual"]].dropna()
    c = paired["VIX_computed"].to_numpy()
    a = paired["VIX_actual"].to_numpy()
    err = c - a

    return {
        "method":      method_label,
        "n_days":      int(len(paired)),
        "mean_err":    float(err.mean()),
        "rmse":        float(np.sqrt((err ** 2).mean())),
        "mae":         float(np.abs(err).mean()),
        "std_resid":   float(err.std(ddof=1)),
        "corr":        float(np.corrcoef(c, a)[0, 1]),
        "max_abs_err": float(np.abs(err).max()),
        "actual_std":  float(a.std(ddof=1)),
    }


# ============================================================================
# Main
# ============================================================================

PIPELINES = [
    {
        "label":   "vix_decomposition.py (OLS projection)",
        "slug":    "OLS",
        "csv":     OUTPUT_DIR / "vix_decomposition.csv",
        "loader":  load_OLS,
    },
    {
        "label":   "vix_pipeline_variance.py",
        "slug":    "variance",
        "csv":     OUTPUT_DIR / "vix_decomposition_variance.csv",
        "loader":  load_variance,
    },
    {
        "label":   "vix_pipeline_CBOE_bucket_nd2.py",
        "slug":    "CBOE_bucket",
        "csv":     OUTPUT_DIR / "vix_decomposition_CBOE_bucket_nd2.csv",
        "loader":  load_CBOE_bucket,
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict] = []

    for p in PIPELINES:
        if not p["csv"].exists():
            print(f"  ! missing {p['csv']} — skipping {p['slug']}")
            continue

        print(f"\n[{p['slug']}] reading {p['csv'].name}")
        df = p["loader"](p["csv"])
        print(f"  rows: {len(df)}   "
              f"date range: {df.index.min().date()} → {df.index.max().date()}")

        png = OUTPUT_DIR / f"vix_decomposition_{p['slug']}_panels.png"
        make_panels(df, p["label"], png)
        print(f"  → {png}")

        s = macro_stats(df, p["label"])
        stats_rows.append(s)
        print(f"  vs actual: n={s['n_days']}  "
              f"mean_err={s['mean_err']:+.3f}  "
              f"RMSE={s['rmse']:.3f}  "
              f"MAE={s['mae']:.3f}  "
              f"std_resid={s['std_resid']:.3f}  "
              f"corr={s['corr']:.4f}  "
              f"max|err|={s['max_abs_err']:.3f}")

    if not stats_rows:
        raise SystemExit("No CSVs found — nothing to plot.")

    stats_df = pd.DataFrame(stats_rows)
    csv_path = OUTPUT_DIR / "vix_decomposition_method_comparison.csv"
    stats_df.to_csv(csv_path, index=False)
    print(f"\n→ {csv_path}")

    print("\n" + "=" * 110)
    print("MACRO BREAKDOWN — VIX_computed vs VIX_actual (2023)")
    print("=" * 110)
    fmt_cols = ["method", "n_days", "mean_err", "rmse", "mae",
                "std_resid", "corr", "max_abs_err", "actual_std"]
    print(stats_df[fmt_cols].to_string(
        index=False,
        formatters={
            "mean_err":    lambda x: f"{x:+.3f}",
            "rmse":        lambda x: f"{x:.3f}",
            "mae":         lambda x: f"{x:.3f}",
            "std_resid":   lambda x: f"{x:.3f}",
            "corr":        lambda x: f"{x:.4f}",
            "max_abs_err": lambda x: f"{x:.3f}",
            "actual_std":  lambda x: f"{x:.3f}",
        },
    ))
    print("=" * 110)
    print("All errors are in vol-points. corr is Pearson(VIX_computed, VIX_actual).")
    print("actual_std = std-dev of VIX_actual over the same window (context).")


if __name__ == "__main__":
    main()
