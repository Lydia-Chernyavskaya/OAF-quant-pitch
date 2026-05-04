#!/usr/bin/env python3
"""Event-study driver for the OLS-projection VIX decomposition pipeline.

Generates four event PNGs that visually mirror
event_<name>_variance_decomp_chart.png and
event_<name>_CBOE_bucket_decomp_chart.png — same six bars (F1 Sticky /
F2 Parallel / F3 Put Skew / F4 Call Skew / F5 Downside Conv /
F6 Upside Conv), same colour palette, same ΔVIX/Σ reference lines.

The OLS pipeline (src/vix_decomposition.py) closes exactly:
    F1 + F2 + F3 + F4 + F5 + F6 + higher_order  ==  ΔVIX_actual

We do NOT add a 7th bar for higher_order; instead the visible gap
between the solid ΔVIX line and the dashed Σ (sum of F1..F6) line IS
higher_order, which we annotate as small dark-grey text.

Outputs (src/output/):
    event_volmageddon_OLS_decomp_chart.png
    event_feb2020_OLS_decomp_chart.png
    event_covid_OLS_decomp_chart.png
    event_svb_OLS_decomp_chart.png
    event_OLS_summary.csv
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from src.iv_surface import build_daily_surfaces  # noqa: E402
from src.vix_decomposition import decompose_one_day  # noqa: E402
from spx_local_loader import load_spx_options  # noqa: E402

log = logging.getLogger("event_OLS_decomp")


DATA_DIR = os.path.expanduser("~/data/spx_eod")
OUTPUT_DIR = HERE / "output"


# Hard-coded reference VIX closes (cross-checked in the per-event
# scripts under src/event_study_*.py). These are *informational*: the
# OLS pipeline closes its chain against the BS-reconstructed ΔVIX
# (vix_bs_t − vix0), not the market close, so this dict is only used
# to print a side-by-side market-vs-reconstructed comparison.
REFERENCE_VIX_CLOSE = {
    "2018-02-02": 17.31,
    "2018-02-05": 37.32,
    "2020-02-21": 17.08,
    "2020-02-24": 25.03,
    "2020-03-13": 57.83,
    "2020-03-16": 82.69,
    "2023-03-15": 26.14,
    "2023-03-16": 22.99,
}


EVENTS = [
    {"slug": "volmageddon", "name": "Volmageddon",
     "t0": "2018-02-02", "t1": "2018-02-05"},
    {"slug": "feb2020",    "name": "Feb 2020 onset",
     "t0": "2020-02-21", "t1": "2020-02-24"},
    {"slug": "covid",      "name": "Covid crash",
     "t0": "2020-03-13", "t1": "2020-03-16"},
    {"slug": "svb",        "name": "SVB / Credit Suisse",
     "t0": "2023-03-15", "t1": "2023-03-16"},
]


# ============================================================================
# Wide → long conversion (mirrors data_ingestion._parse_single_txt)
# ============================================================================

def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert spx_local_loader wide chain to the long format expected by
    iv_surface.build_daily_surfaces.

    Output columns: date, expiration, dte, strike, underlying_last,
                    option_type, implied_vol, bid, ask, mid
    """
    df = wide.copy()
    df = df.rename(columns={
        "QUOTE_DATE":      "date",
        "EXPIRE_DATE":     "expiration",
        "DTE":             "dte",
        "STRIKE":          "strike",
        "UNDERLYING_LAST": "underlying_last",
    })

    for col in ["dte", "strike", "underlying_last",
                "C_IV", "C_BID", "C_ASK",
                "P_IV", "P_BID", "P_ASK"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    id_cols = ["date", "expiration", "dte", "strike", "underlying_last"]

    calls = df[id_cols + ["C_IV", "C_BID", "C_ASK"]].rename(columns={
        "C_IV": "implied_vol", "C_BID": "bid", "C_ASK": "ask",
    })
    calls["option_type"] = "C"

    puts = df[id_cols + ["P_IV", "P_BID", "P_ASK"]].rename(columns={
        "P_IV": "implied_vol", "P_BID": "bid", "P_ASK": "ask",
    })
    puts["option_type"] = "P"

    long_df = pd.concat([calls, puts], ignore_index=True)
    long_df["mid"] = (long_df["bid"] + long_df["ask"]) / 2.0
    return long_df


def _apply_quality_gates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that would corrupt the surface fit. Mirrors the gates
    used by data_ingestion._apply_sanity_checks but stripped to the
    minimum needed to keep build_daily_surfaces happy.
    """
    n0 = len(df)

    # Restrict to the CBOE VIX DTE window.
    df = df[(df["dte"] >= config.MIN_DTE) & (df["dte"] <= config.MAX_DTE)]

    # Drop missing/garbage IVs.
    df = df[df["implied_vol"].between(config.MIN_IV, config.MAX_IV, inclusive="neither")]

    # Drop crossed / non-positive markets.
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]

    # Drop minimum-bid-too-low (illiquid).
    df = df[df["bid"] >= config.MIN_BID]

    # Drop excessively wide spreads.
    mid = (df["bid"] + df["ask"]) / 2.0
    rel = (df["ask"] - df["bid"]) / mid.replace(0.0, np.nan)
    df = df[rel <= config.MAX_RELATIVE_SPREAD]

    log.info("  quality gates: kept %d / %d rows (%.1f%%)",
             len(df), n0, 100.0 * len(df) / max(n0, 1))
    return df.reset_index(drop=True)


# ============================================================================
# Chart
# ============================================================================

# Bar palette — matches src/event_study_plots.py:plot_decomp_panel and the
# variance / CBOE_bucket reference PNGs.
BAR_LABELS = ["F1\nSticky", "F2\nParallel", "F3\nPut Skew",
              "F4\nCall Skew", "F5\nDownside\nConv", "F6\nUpside\nConv"]
BAR_COLORS = ["#7f7f7f", "#4682b4", "#8b0000",
              "#006400", "#cd5c5c", "#90ee90"]


def make_decomp_chart(name: str, values: list[float],
                      delta_vix: float, sum_f1_f6: float,
                      higher_order: float, out_path: Path) -> None:
    """Render the F1..F6 bar chart with ΔVIX (solid) and Σ (dashed) lines
    plus a small higher_order text annotation in the gap between them.

    figsize / dpi chosen to match the reference 1034×729 PNGs.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    bars = ax.bar(BAR_LABELS, values, color=BAR_COLORS,
                  edgecolor="black", linewidth=1.0)

    ax.axhline(delta_vix, color="black", lw=1.5, ls="-",
               label=f"ΔVIX actual = {delta_vix:+.2f}")
    ax.axhline(sum_f1_f6, color="black", lw=1.5, ls="--",
               label=f"Sum of factors = {sum_f1_f6:+.2f}")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)

    ax.set_title(
        f"{name}: ΔVIX = {delta_vix:+.2f},  Σ = {sum_f1_f6:+.2f}",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("Vol-points")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Bar value labels (2dp, signed, above positive / below negative).
    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:+.2f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, v),
            xytext=(0, 3 if v >= 0 else -12),
            textcoords="offset points",
            ha="center", fontsize=8,
        )

    # higher_order annotation in the gap between ΔVIX and Σ lines.
    # White bbox so the text stays readable when the gap is tiny and the
    # solid/dashed lines visually overlap.
    ax.annotate(
        f"higher_order = {higher_order:+.2f}",
        xy=(2.5, (delta_vix + sum_f1_f6) / 2.0),
        ha="center", va="center",
        fontsize=9, color="#404040",
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white", edgecolor="none", alpha=0.85),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Per-event runner
# ============================================================================

def _run_one_event(event: dict) -> dict:
    """Run the OLS decomposition for one event and write the chart.

    Returns a flat record suitable for stacking into event_OLS_summary.csv.
    """
    name = event["name"]
    slug = event["slug"]
    t0_str, t1_str = event["t0"], event["t1"]

    print(f"\n[{name}] loading {t0_str} / {t1_str} …")
    wide = load_spx_options(data_dir=DATA_DIR, quote_dates=[t0_str, t1_str])

    loaded = sorted(wide["QUOTE_DATE"].dt.strftime("%Y-%m-%d").unique())
    for needed in (t0_str, t1_str):
        if needed not in loaded:
            raise RuntimeError(
                f"[{name}] required date {needed} not in loaded chain. "
                f"Loaded: {loaded}"
            )

    long_df = _wide_to_long(wide)
    long_df = _apply_quality_gates(long_df)

    # Build IV surfaces for both event days.
    surfaces = build_daily_surfaces(long_df)
    date_t0 = pd.Timestamp(t0_str)
    date_t1 = pd.Timestamp(t1_str)
    if date_t0 not in surfaces or date_t1 not in surfaces:
        raise RuntimeError(
            f"[{name}] surface build failed: have keys "
            f"{sorted(d.date() for d in surfaces.keys())}"
        )

    # OLS decomposition. Note decompose_one_day's argument naming is
    # confusing: date_t = "today" (event t1), date_t1 = "yesterday" (event t0).
    vix_synth_dummy = pd.Series(dtype=float)  # only used for diagnostic
    decomp = decompose_one_day(date_t1, date_t0, surfaces, vix_synth_dummy)
    if decomp is None:
        raise RuntimeError(f"[{name}] decompose_one_day returned None")

    f1 = decomp["sticky_strike_contribution"]
    f2 = decomp["parallel_contribution"]
    f3 = decomp["put_skew_contribution"]
    f4 = decomp["call_skew_contribution"]
    f5 = decomp["put_convexity_contribution"]
    f6 = decomp["call_convexity_contribution"]
    higher_order = decomp["higher_order_contribution"]
    sum_f1_f6 = f1 + f2 + f3 + f4 + f5 + f6
    delta_vix = decomp["delta_vix"]
    identity = (sum_f1_f6 + higher_order) - delta_vix

    # ── Chart ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUTPUT_DIR / f"event_{slug}_OLS_decomp_chart.png"
    make_decomp_chart(name, [f1, f2, f3, f4, f5, f6],
                      delta_vix, sum_f1_f6, higher_order, chart_path)
    print(f"  → {chart_path}")

    # ── Sanity print (kept consistent with the variance harness style) ──
    market_dvix = (REFERENCE_VIX_CLOSE.get(t1_str, float("nan"))
                   - REFERENCE_VIX_CLOSE.get(t0_str, float("nan")))
    print(f"  ΔVIX (BS-reconstructed) = {delta_vix:+.3f}  "
          f"ΔVIX (market close)     = {market_dvix:+.2f}")
    print(f"  F1={f1:+.3f}  F2={f2:+.3f}  F3={f3:+.3f}  "
          f"F4={f4:+.3f}  F5={f5:+.3f}  F6={f6:+.3f}")
    print(f"  Σ(F1..F6)     = {sum_f1_f6:+.3f}   "
          f"higher_order = {higher_order:+.3f}")
    print(f"  identity (Σ + h.o. − ΔVIX) = {identity:+.2e}  "
          f"(must be < 1e-6)")

    return {
        "event":          name,
        "t0_date":        t0_str,
        "t1_date":        t1_str,
        "ΔVIX_actual":    delta_vix,
        "F1":             f1,
        "F2":             f2,
        "F3":             f3,
        "F4":             f4,
        "F5":             f5,
        "F6":             f6,
        "sum_F1_F6":      sum_f1_f6,
        "higher_order":   higher_order,
        "identity_check": identity,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-8s  %(name)s – %(message)s",
    )

    records = [_run_one_event(ev) for ev in EVENTS]

    df = pd.DataFrame.from_records(records)
    csv_path = OUTPUT_DIR / "event_OLS_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n→ {csv_path}")

    # Final identity check.
    max_id = df["identity_check"].abs().max()
    print(f"\nMax |identity_check| across all 4 events = {max_id:.2e}")
    if max_id >= 1e-6:
        raise SystemExit(
            f"FAIL: identity_check {max_id:.2e} ≥ 1e-6 — chain did not close."
        )
    print("OK: chain closes for all four events (|residual| < 1e-6).\n")

    print("=" * 100)
    print("EVENT OLS SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("=" * 100)


if __name__ == "__main__":
    main()
