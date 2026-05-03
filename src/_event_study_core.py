#!/usr/bin/env python3
"""Shared core for the per-event study scripts.

Per-event scripts hold event-specific config (EVENT dict, source-cited
REFERENCE_VIX_CLOSE) and call run_event() here to do the loading,
decomposition, IV summary, CSV writing, and chart generation.
"""

from __future__ import annotations

import os
import sys
import math
from datetime import datetime

import pandas as pd

# Local sibling imports
sys.path.insert(0, os.path.dirname(__file__))
from vix_pipeline_local_nd2 import (  # noqa: E402
    compute_vix_for_snapshot,
    build_30day_skew,
    run_decomposition,
    fetch_cboe_vix_historical,
    get_vol_at_strike,
)
from spx_local_loader import (  # noqa: E402
    load_spx_options,
    fetch_rfr,
    fetch_vix_actual,
)
from event_study_plots import make_skew_chart, make_decomp_chart  # noqa: E402


DATA_DIR = os.path.expanduser("~/data/spx_eod")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _build_optionchain(day_df: pd.DataFrame, qd: pd.Timestamp) -> dict:
    """Group day_df by EXPIRE_DATE into the shape compute_vix_for_snapshot expects."""
    optionchain: dict[str, list[dict]] = {}
    for expire_dt, exp_df in day_df.groupby("EXPIRE_DATE", sort=True):
        if pd.Timestamp(expire_dt).date() <= pd.Timestamp(qd).date():
            continue
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
        optionchain[pd.Timestamp(expire_dt).strftime("%Y-%m-%d")] = rows
    return optionchain


def _process_day(snap: dict, cboe_vix: dict, reference_vix: dict) -> dict:
    """compute_vix_for_snapshot + build_30day_skew + vix_actual lookup.

    Priority for vix_actual: hard-coded reference (deterministic) → live
    CBOE CSV → data-source VIX.spot.
    """
    payload = snap["payload"]
    spot = payload["SPX"]["spot"]
    rfr_decimal = float(payload.get("rfr", 0.0)) / 100.0  # payload is percent
    optionchain = payload["SPX"]["optionchain"]
    snap_date = datetime.strptime(snap["date"], "%Y-%m-%d").date()

    res = compute_vix_for_snapshot(spot, rfr_decimal, optionchain, snap_date)
    if res is None:
        raise RuntimeError(f"compute_vix_for_snapshot returned None for {snap['date']}")

    if snap["date"] in reference_vix:
        res["vix_actual"] = reference_vix[snap["date"]]
    elif snap["date"] in cboe_vix:
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


def _build_snapshot_for_day(day_df: pd.DataFrame, quote_dt: pd.Timestamp,
                            rfr_decimal: float, vix_actual) -> dict:
    spot = float(day_df["UNDERLYING_LAST"].iloc[0])
    rfr_pct = float(rfr_decimal) * 100.0 if rfr_decimal is not None else 0.0
    vix_spot = float(vix_actual) if vix_actual is not None else None
    return {
        "date": pd.Timestamp(quote_dt).strftime("%Y-%m-%d"),
        "payload": {
            "SPX": {"spot": spot, "optionchain": _build_optionchain(day_df, quote_dt)},
            "VIX": {"spot": vix_spot},
            "rfr": rfr_pct,
        },
    }


def _check_reference_against_live(reference_vix: dict, cboe_vix: dict) -> None:
    """If the live CBOE feed and our hard-coded reference differ by > 0.10
    on any of this event's dates, print a warning so the user can decide
    whether to update the hard-coded value. If the live feed is empty,
    fall back silently."""
    if not cboe_vix:
        return
    for d, ref_val in reference_vix.items():
        live_val = cboe_vix.get(d)
        if live_val is None:
            continue
        if abs(float(live_val) - float(ref_val)) > 0.10:
            print(f"  ⚠ live CBOE close for {d} = {live_val:.2f} differs from "
                  f"hard-coded reference {ref_val:.2f} by "
                  f"{abs(live_val - ref_val):.2f} vol-pt — consider updating.")


def run_event(event: dict, reference_vix: dict, slug: str) -> dict:
    """Load t0/t1, decompose, write four output files, return a flat record.

    `slug`: short string used in output filenames, e.g. 'volmageddon'.
    """
    t0_str, t1_str = event["t0"], event["t1"]
    print(f"\n[{event['name']}] loading {t0_str} and {t1_str} …")
    chain = load_spx_options(data_dir=DATA_DIR, quote_dates=[t0_str, t1_str])

    loaded_dates = sorted(chain["QUOTE_DATE"].dt.strftime("%Y-%m-%d").unique())
    for needed in (t0_str, t1_str):
        if needed not in loaded_dates:
            raise RuntimeError(
                f"[{event['name']}] required date {needed} not in loaded chain. "
                f"Loaded dates: {loaded_dates}"
            )

    timestamps = [pd.Timestamp(d) for d in (t0_str, t1_str)]
    rfr_lookup = fetch_rfr(timestamps)
    vix_lookup = fetch_vix_actual(timestamps)
    cboe_vix = fetch_cboe_vix_historical()
    _check_reference_against_live(reference_vix, cboe_vix)

    snaps: dict[str, dict] = {}
    for day_str in (t0_str, t1_str):
        day_dt = pd.Timestamp(day_str)
        day_df = chain[chain["QUOTE_DATE"] == day_dt]
        rfr_decimal = rfr_lookup.get(day_dt)
        vix_val = vix_lookup.get(day_dt)
        snaps[day_str] = _build_snapshot_for_day(day_df, day_dt, rfr_decimal, vix_val)

    t0_res = _process_day(snaps[t0_str], cboe_vix, reference_vix)
    t1_res = _process_day(snaps[t1_str], cboe_vix, reference_vix)

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

    rec = {
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Decomp CSV ──────────────────────────────────────────────────────
    csv_cols = ["name", "t0", "t1", "SPX_t0", "SPX_t1", "VIX_t0", "VIX_t1",
                "dVIX", "F1", "F2", "F3", "F4", "F5", "F6",
                "sum_factors", "residual"]
    decomp_csv = os.path.join(OUTPUT_DIR, f"event_{slug}_decomp.csv")
    pd.DataFrame([{k: rec[k] for k in csv_cols}]).to_csv(decomp_csv, index=False)
    print(f"  → {decomp_csv}")

    # ── IV summary CSV (Warren's table) ─────────────────────────────────
    # Textbook CBOE F1/F2 evaluated on full put+call combined skew dicts.
    # These may differ slightly from the decomp.csv F1/F2 above (which
    # come from run_decomposition); both are valid — iv_summary uses the
    # direct skew lookups Warren asked for.
    skew_t0_full = {**t0_res["put_skew_30d"], **t0_res["call_skew_30d"]}
    skew_t1_full = {**t1_res["put_skew_30d"], **t1_res["call_skew_30d"]}
    S_t0 = float(t0_res["spot"]); S_t1 = float(t1_res["spot"])
    atm_iv_t0      = get_vol_at_strike(skew_t0_full, S_t0)
    atm_iv_t1      = get_vol_at_strike(skew_t1_full, S_t1)
    iv_t0_at_S_t1  = get_vol_at_strike(skew_t0_full, S_t1)
    f1_textbook = iv_t0_at_S_t1 - atm_iv_t0
    f2_textbook = atm_iv_t1 - iv_t0_at_S_t1

    iv_row = {
        "event": event["name"],
        "t0": t0_str,
        "t1": t1_str,
        "S_t0": S_t0,
        "S_t1": S_t1,
        "ATM_IV_t0": atm_iv_t0,
        "ATM_IV_t1": atm_iv_t1,
        "IV_t0_at_S_t1": iv_t0_at_S_t1,
        "F1_sticky_strike": f1_textbook,
        "F2_parallel": f2_textbook,
    }
    iv_csv = os.path.join(OUTPUT_DIR, f"event_{slug}_iv_summary.csv")
    pd.DataFrame([iv_row]).to_csv(iv_csv, index=False)
    print(f"  → {iv_csv}")

    # ── Charts ──────────────────────────────────────────────────────────
    skew_path = os.path.join(OUTPUT_DIR, f"event_{slug}_skew_chart.png")
    make_skew_chart(rec, skew_path)
    print(f"  → {skew_path}")

    decomp_path = os.path.join(OUTPUT_DIR, f"event_{slug}_decomp_chart.png")
    make_decomp_chart(rec, dvix, sum_factors, decomp_path)
    print(f"  → {decomp_path}")

    # ── Terminal summary ────────────────────────────────────────────────
    d_spx_pct = (S_t1 - S_t0) / S_t0 * 100.0
    print()
    print(f"{event['name']} ({t0_str} → {t1_str})")
    print(f"  S_t0 = {S_t0:.2f}   S_t1 = {S_t1:.2f}   ΔSPX = {d_spx_pct:+.2f}%")
    print(f"  VIX_t0 = {vix_t0:.2f}   VIX_t1 = {vix_t1:.2f}   "
          f"ΔVIX_actual = {dvix:+.2f}")
    print()
    print(f"  ATM IV at t0 (σ_t0(S_t0))           = {atm_iv_t0:6.2f}%")
    print(f"  ATM IV at t1 (σ_t1(S_t1))           = {atm_iv_t1:6.2f}%")
    print(f"  IV at t0 of new spot (σ_t0(S_t1))   = {iv_t0_at_S_t1:6.2f}%")
    print()
    print(f"  F1 sticky strike  = {f1_textbook:+.2f} vol-pts  "
          f"(= σ_t0(S_t1) − σ_t0(S_t0))")
    print(f"  F2 parallel shift = {f2_textbook:+.2f} vol-pts  "
          f"(= σ_t1(S_t1) − σ_t0(S_t1))")
    print()
    print(f"  Decomposition (run_decomposition):")
    print(f"    F1={decomp.factor1_sticky_strike:+.3f}  "
          f"F2={decomp.factor2_parallel_shift:+.3f}  "
          f"F3={decomp.factor3_put_skew_grad:+.3f}")
    print(f"    F4={decomp.factor4_call_skew_grad:+.3f}  "
          f"F5={decomp.factor5_downside_conv:+.3f}  "
          f"F6={decomp.factor6_upside_conv:+.3f}")
    print(f"    Σ = {sum_factors:+.3f}   "
          f"residual (Σ − ΔVIX) = {sum_factors - dvix:+.3f}")

    return rec
