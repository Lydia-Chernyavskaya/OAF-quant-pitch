#!/usr/bin/env python3
"""
Event-study runner for Volmageddon (2018-02-02 → 2018-02-05) and the
Covid crash (2020-03-13 → 2020-03-16). Uses the N(d2) hybrid pipeline
in vix_pipeline_local_nd2.py for the F1-F6 decomposition.

Outputs:
  src/output/event_studies_nd2_decomp.csv        — one row per event
  src/output/event_studies_nd2_skew_chart.png    — 2-panel skew comparison
  src/output/event_studies_nd2_decomp_chart.png  — 2-panel F1-F6 bar chart
"""

from __future__ import annotations

import os
import sys
import math
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Local imports — sibling modules in src/
sys.path.insert(0, os.path.dirname(__file__))
from vix_pipeline_local_nd2 import (  # noqa: E402
    compute_vix_for_snapshot,
    build_30day_skew,
    run_decomposition,
    fetch_cboe_vix_historical,
)
from spx_local_loader import (  # noqa: E402
    load_spx_options,
    fetch_rfr,
    fetch_vix_actual,
)


EVENTS = [
    {"name": "Volmageddon", "t0": "2018-02-02", "t1": "2018-02-05"},
    {"name": "Covid crash", "t0": "2020-03-13", "t1": "2020-03-16"},
]


def build_snapshot_for_day(day_df: pd.DataFrame, quote_dt: pd.Timestamp,
                           rfr_decimal: float, vix_actual: float | None) -> dict:
    """Build the {date, payload} dict that compute_vix_for_snapshot expects.

    payload["rfr"] is stored in PERCENT to match the existing convention
    (vix_pipeline_local_nd2.main divides by 100 again before passing to
    compute_vix_for_snapshot).
    """
    optionchain: dict[str, list[dict]] = {}
    for expire_dt, exp_df in day_df.groupby("EXPIRE_DATE", sort=True):
        if pd.Timestamp(expire_dt).date() <= pd.Timestamp(quote_dt).date():
            continue
        expiry_str = pd.Timestamp(expire_dt).strftime("%Y-%m-%d")
        sub = exp_df.sort_values("STRIKE")
        cbid = pd.to_numeric(sub["C_BID"], errors="coerce").fillna(0.0).to_numpy()
        cask = pd.to_numeric(sub["C_ASK"], errors="coerce").fillna(0.0).to_numpy()
        pbid = pd.to_numeric(sub["P_BID"], errors="coerce").fillna(0.0).to_numpy()
        pask = pd.to_numeric(sub["P_ASK"], errors="coerce").fillna(0.0).to_numpy()
        strikes = sub["STRIKE"].to_numpy()
        rows = [
            {"strike": float(K), "cbid": float(cb), "cask": float(ca),
             "pbid": float(pb), "pask": float(pa)}
            for K, cb, ca, pb, pa in zip(strikes, cbid, cask, pbid, pask)
        ]
        optionchain[expiry_str] = rows

    spot = float(day_df["UNDERLYING_LAST"].iloc[0])
    rfr_pct = float(rfr_decimal) * 100.0 if rfr_decimal is not None else 0.0
    vix_spot = float(vix_actual) if vix_actual is not None else None

    return {
        "date": pd.Timestamp(quote_dt).strftime("%Y-%m-%d"),
        "payload": {
            "SPX": {"spot": spot, "optionchain": optionchain},
            "VIX": {"spot": vix_spot},
            "rfr": rfr_pct,
        },
    }


def process_day(snap: dict, cboe_vix: dict) -> dict:
    """Run compute_vix_for_snapshot + build_30day_skew on one snapshot.
    Returns the result dict augmented with put_skew_30d / call_skew_30d.
    """
    payload = snap["payload"]
    spot = payload["SPX"]["spot"]
    rfr_decimal = float(payload.get("rfr", 0.0)) / 100.0  # convention: payload is percent
    optionchain = payload["SPX"]["optionchain"]
    snap_date = datetime.strptime(snap["date"], "%Y-%m-%d").date()

    res = compute_vix_for_snapshot(spot, rfr_decimal, optionchain, snap_date)
    if res is None:
        raise RuntimeError(f"compute_vix_for_snapshot returned None for {snap['date']}")

    # CBOE VIX preferred over data-source VIX, but accept either.
    if snap["date"] in cboe_vix:
        res["vix_actual"] = cboe_vix[snap["date"]]
    elif payload.get("VIX", {}).get("spot") is not None:
        res["vix_actual"] = payload["VIX"]["spot"]

    put_skew, call_skew = build_30day_skew(
        res["chain1_df"], res["chain2_df"],
        res["DTE1"], res["DTE2"],
        spot, res["F"], res["F2"], rfr_decimal,
    )
    res["put_skew_30d"] = put_skew
    res["call_skew_30d"] = call_skew
    return res


def run_event(event: dict, data_dir: str, cboe_vix: dict) -> dict:
    """Load t0/t1 only, decompose, return a flat record."""
    t0_str, t1_str = event["t0"], event["t1"]
    print(f"\n[{event['name']}] loading {t0_str} and {t1_str} ...")
    chain = load_spx_options(data_dir=data_dir, quote_dates=[t0_str, t1_str])

    loaded_dates = sorted(chain["QUOTE_DATE"].dt.strftime("%Y-%m-%d").unique())
    for needed in (t0_str, t1_str):
        if needed not in loaded_dates:
            raise RuntimeError(
                f"[{event['name']}] required date {needed} not in loaded chain. "
                f"Loaded dates: {loaded_dates}"
            )

    # rfr / vix_actual lookups (one round-trip each, both days at once)
    timestamps = [pd.Timestamp(d) for d in (t0_str, t1_str)]
    rfr_lookup = fetch_rfr(timestamps)
    vix_lookup = fetch_vix_actual(timestamps)

    snaps = {}
    for day_str in (t0_str, t1_str):
        day_dt = pd.Timestamp(day_str)
        day_df = chain[chain["QUOTE_DATE"] == day_dt]
        rfr_decimal = rfr_lookup.get(day_dt)
        vix_val = vix_lookup.get(day_dt)
        snap = build_snapshot_for_day(day_df, day_dt, rfr_decimal, vix_val)
        snaps[day_str] = snap

    t0_res = process_day(snaps[t0_str], cboe_vix)
    t1_res = process_day(snaps[t1_str], cboe_vix)

    decomp = run_decomposition(t0_res, t1_res)
    if decomp is None:
        raise RuntimeError(f"[{event['name']}] run_decomposition returned None")

    sum_factors = (
        decomp.factor1_sticky_strike
        + decomp.factor2_parallel_shift
        + decomp.factor3_put_skew_grad
        + decomp.factor4_call_skew_grad
        + decomp.factor5_downside_conv
        + decomp.factor6_upside_conv
    )
    vix_t0 = t0_res.get("vix_actual") or t0_res["vix_computed"]
    vix_t1 = t1_res.get("vix_actual") or t1_res["vix_computed"]
    dvix = vix_t1 - vix_t0

    return {
        "name": event["name"],
        "t0": t0_str,
        "t1": t1_str,
        "SPX_t0": t0_res["spot"],
        "SPX_t1": t1_res["spot"],
        "VIX_t0": vix_t0,
        "VIX_t1": vix_t1,
        "dVIX": dvix,
        "F1": decomp.factor1_sticky_strike,
        "F2": decomp.factor2_parallel_shift,
        "F3": decomp.factor3_put_skew_grad,
        "F4": decomp.factor4_call_skew_grad,
        "F5": decomp.factor5_downside_conv,
        "F6": decomp.factor6_upside_conv,
        "sum_factors": sum_factors,
        "residual": sum_factors - dvix,
        "_t0_res": t0_res,
        "_t1_res": t1_res,
    }


# ─── plotting ─────────────────────────────────────────────────────────────

def _plot_skew_panel(ax, t0_res: dict, t1_res: dict, t0_str: str, t1_str: str,
                     title: str):
    """One panel of the skew chart: t0/t1 puts and calls scatter + spline,
    log-y, fixed ticks at 32/64/96/128%."""
    series = [
        (t0_res["put_skew_30d"],  f"Put {t0_str}",  "#1f4e79", 12),
        (t0_res["call_skew_30d"], f"Call {t0_str}", "#6699cc", 12),
        (t1_res["put_skew_30d"],  f"Put {t1_str}",  "#c65d00", 12),
        (t1_res["call_skew_30d"], f"Call {t1_str}", "#ff9933", 12),
    ]
    for skew, label, color, msize in series:
        if not skew:
            continue
        ks = sorted(skew.keys())
        ys = [skew[k] for k in ks]
        ax.scatter(ks, ys, s=msize, color=color, alpha=0.6, zorder=2)
        if len(ks) >= 4:
            cs = CubicSpline(ks, ys, bc_type="natural")
            xs = np.linspace(ks[0], ks[-1], 400)
            ax.plot(xs, cs(xs), lw=1.5, color=color, alpha=0.9, label=label, zorder=3)
        else:
            ax.plot(ks, ys, lw=1.5, color=color, alpha=0.9, label=label, zorder=3)

    ax.set_yscale("log")
    ax.set_yticks([32, 64, 96, 128])
    ax.set_yticklabels(["32%", "64%", "96%", "128%"])
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, which="major", axis="y")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def make_skew_chart(records: list[dict], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rec in zip(axes, records):
        _plot_skew_panel(
            ax, rec["_t0_res"], rec["_t1_res"],
            rec["t0"], rec["t1"], rec["name"],
        )
    fig.suptitle("SPX 30-Day Skew: Volmageddon vs Covid Crash", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_decomp_panel(ax, rec: dict):
    labels = ["F1\nSticky", "F2\nParallel", "F3\nPut Skew",
              "F4\nCall Skew", "F5\nDownside\nConv", "F6\nUpside\nConv"]
    values = [rec["F1"], rec["F2"], rec["F3"], rec["F4"], rec["F5"], rec["F6"]]
    colors = ["#7f7f7f", "#4682b4", "#8b0000", "#006400", "#cd5c5c", "#90ee90"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(rec["dVIX"], color="black", lw=1.5, ls="-",
               label=f"ΔVIX actual = {rec['dVIX']:+.2f}")
    ax.axhline(rec["sum_factors"], color="black", lw=1.5, ls="--",
               label=f"Sum of factors = {rec['sum_factors']:+.2f}")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    title = (f"{rec['name']}: ΔVIX = {rec['dVIX']:+.2f},  "
             f"Σ = {rec['sum_factors']:+.2f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Vol-points")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    # Numeric labels above each bar
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3 if v >= 0 else -12),
                    textcoords="offset points",
                    ha="center", fontsize=8)


def make_decomp_chart(records: list[dict], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rec in zip(axes, records):
        _plot_decomp_panel(ax, rec)
    fig.suptitle("VIX Decomposition: F1-F6 Contributions", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    data_dir = os.path.expanduser("~/data/spx_eod")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching CBOE historical VIX (for actual values)...")
    cboe_vix = fetch_cboe_vix_historical()
    print(f"  got {len(cboe_vix)} CBOE VIX records")

    records = [run_event(ev, data_dir, cboe_vix) for ev in EVENTS]

    # CSV: drop the internal _t0_res / _t1_res keys before writing
    csv_cols = ["name", "t0", "t1", "SPX_t0", "SPX_t1", "VIX_t0", "VIX_t1",
                "dVIX", "F1", "F2", "F3", "F4", "F5", "F6",
                "sum_factors", "residual"]
    csv_path = os.path.join(output_dir, "event_studies_nd2_decomp.csv")
    pd.DataFrame([{k: r[k] for k in csv_cols} for r in records]).to_csv(
        csv_path, index=False
    )
    print(f"\nDecomposition CSV → {csv_path}")

    skew_path = os.path.join(output_dir, "event_studies_nd2_skew_chart.png")
    make_skew_chart(records, skew_path)
    print(f"Skew chart       → {skew_path}")

    decomp_path = os.path.join(output_dir, "event_studies_nd2_decomp_chart.png")
    make_decomp_chart(records, decomp_path)
    print(f"Decomp chart     → {decomp_path}")

    # ── Text summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("EVENT-STUDY SUMMARY (N(d2) hybrid pipeline)")
    print("=" * 78)
    for r in records:
        d_spx = r["SPX_t1"] - r["SPX_t0"]
        d_spx_pct = d_spx / r["SPX_t0"] * 100.0
        print(f"\n{r['name']}  ({r['t0']} → {r['t1']})")
        print(f"  SPX:  {r['SPX_t0']:.2f} → {r['SPX_t1']:.2f}  "
              f"(Δ = {d_spx:+.2f}, {d_spx_pct:+.2f}%)")
        print(f"  VIX:  {r['VIX_t0']:.2f} → {r['VIX_t1']:.2f}  "
              f"(ΔVIX_actual = {r['dVIX']:+.2f})")
        print(f"  F1 Sticky strike:    {r['F1']:+.3f}")
        print(f"  F2 Parallel shift:   {r['F2']:+.3f}")
        print(f"  F3 Put shoulder:     {r['F3']:+.3f}")
        print(f"  F4 Call shoulder:    {r['F4']:+.3f}")
        print(f"  F5 Put wing:         {r['F5']:+.3f}")
        print(f"  F6 Call wing:        {r['F6']:+.3f}")
        print(f"  Σ F1..F6:            {r['sum_factors']:+.3f}")
        print(f"  Residual (Σ − ΔVIX): {r['residual']:+.3f}")
    print()
    print("Note: the hybrid pipeline does NOT telescope exactly — F1/F2 are "
          "single-strike vol-pt scalars while F3-F6 are VIX-pt impacts of "
          "bucket-swap counterfactuals. The residual is the F2-vs-F3-F6 "
          "attribution gap (mixed units), not a bug.")


if __name__ == "__main__":
    main()
