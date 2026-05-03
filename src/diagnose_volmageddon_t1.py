#!/usr/bin/env python3
"""
Diagnostic: why does Volmageddon t1 (2018-02-05) call wing truncate at
~K=2840 in the event-study skew chart, while t0 (2018-02-02) reaches ~K=3080?

Steps for each day's near-expiry chain (chain1_df):
  1. Print strike range (min/max/count).
  2. Find K_max_in_skew = largest strike currently in call_skew_30d.
  3. For every strike K > K_max_in_skew up to the chain's max:
     dump cbid, cask, cmid, pbid, pask, and the build_30day_skew rejection
     reason on the call side.
  4. Compare t0 vs t1 across the same strike range to determine whether the
     gap is bid-side (MM pulled quotes), strike-list (Monday lists fewer
     strikes), or solver-side (bs_iv failed).

Writes the report to src/output/volmageddon_t1_call_wing_diag.txt.
"""

from __future__ import annotations

import os
import sys
import math
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from spx_local_loader import load_spx_options, fetch_rfr  # noqa: E402
from vix_pipeline_local_nd2 import (  # noqa: E402
    compute_vix_for_snapshot,
    build_30day_skew,
    bs_iv,
    _is_zero_bid,
)


T0 = "2018-02-02"
T1 = "2018-02-05"
DATA_DIR = os.path.expanduser("~/data/spx_eod")
OUT_PATH = os.path.join(os.path.dirname(__file__), "output",
                       "volmageddon_t1_call_wing_diag.txt")


def build_optionchain(day_df: pd.DataFrame, qd: pd.Timestamp) -> dict:
    oc: dict[str, list[dict]] = {}
    for exp_dt, exp_df in day_df.groupby("EXPIRE_DATE", sort=True):
        if pd.Timestamp(exp_dt).date() <= pd.Timestamp(qd).date():
            continue
        sub = exp_df.sort_values("STRIKE")
        cbid = pd.to_numeric(sub["C_BID"], errors="coerce").fillna(0.0).to_numpy()
        cask = pd.to_numeric(sub["C_ASK"], errors="coerce").fillna(0.0).to_numpy()
        pbid = pd.to_numeric(sub["P_BID"], errors="coerce").fillna(0.0).to_numpy()
        pask = pd.to_numeric(sub["P_ASK"], errors="coerce").fillna(0.0).to_numpy()
        ks = sub["STRIKE"].to_numpy()
        oc[pd.Timestamp(exp_dt).strftime("%Y-%m-%d")] = [
            {"strike": float(K), "cbid": float(cb), "cask": float(ca),
             "pbid": float(pb), "pask": float(pa)}
            for K, cb, ca, pb, pa in zip(ks, cbid, cask, pbid, pask)
        ]
    return oc


def reason_for_call_exclusion(near_row, far_row, K, F_near, F_far,
                              T_near, T_far, dte_near, dte_far, rfr) -> str:
    """Reproduce build_30day_skew's call-side filter logic and return why
    K was rejected (or 'PASSED' if it should have been admitted)."""
    if K <= 0:
        return "K<=0"

    # Per-side bid validity (CALL side only)
    if _is_zero_bid(near_row, "call"):
        return "near call bid = 0"
    if far_row is None:
        return "K not in far chain (intersection requirement)"
    if _is_zero_bid(far_row, "call"):
        return "far call bid = 0"

    cmid_n = float(near_row.get("cmid", float("nan")))
    cmid_f = float(far_row.get("cmid", float("nan")))
    if math.isnan(cmid_n) or math.isnan(cmid_f):
        return f"cmid NaN (near={cmid_n}, far={cmid_f})"
    if cmid_n <= 0 or cmid_f <= 0:
        return f"cmid <= 0 (near={cmid_n}, far={cmid_f})"
    MIN_MID = 0.05
    if cmid_n <= MIN_MID or cmid_f <= MIN_MID:
        return f"cmid <= {MIN_MID} bid floor (near={cmid_n:.3f}, far={cmid_f:.3f})"

    iv_n = bs_iv(cmid_n, F_near, K, T_near, rfr, is_call=True)
    if iv_n <= 0:
        return f"bs_iv near failed (cmid={cmid_n:.3f}, F={F_near:.2f}, T={T_near:.4f})"
    iv_f = bs_iv(cmid_f, F_far, K, T_far, rfr, is_call=True)
    if iv_f <= 0:
        return f"bs_iv far failed (cmid={cmid_f:.3f}, F={F_far:.2f}, T={T_far:.4f})"

    var_n = (iv_n / 100.0) ** 2 * T_near
    var_f = (iv_f / 100.0) ** 2 * T_far
    w1 = (dte_far - 30.0) / (dte_far - dte_near)
    w2 = (30.0 - dte_near) / (dte_far - dte_near)
    var30 = w1 * var_n + w2 * var_f
    if var30 <= 0:
        return f"var30 <= 0 (var_n={var_n:.4e}, var_f={var_f:.4e}, w1={w1:.3f}, w2={w2:.3f})"

    return f"PASSED (iv_n={iv_n:.2f}%, iv_f={iv_f:.2f}%, var30={var30:.4e})"


def collect_day_state(day_str: str, ev_chain: pd.DataFrame, rfr_lookup: dict):
    """Return a dict with per-day state needed for the diagnostic."""
    day_dt = pd.Timestamp(day_str)
    day_df = ev_chain[ev_chain["QUOTE_DATE"] == day_dt]
    spot = float(day_df["UNDERLYING_LAST"].iloc[0])
    rfr_decimal = rfr_lookup[day_dt]
    oc = build_optionchain(day_df, day_dt)
    res = compute_vix_for_snapshot(
        spot, rfr_decimal, oc, datetime.strptime(day_str, "%Y-%m-%d").date()
    )
    df_near = res["chain1_df"]
    df_far = res["chain2_df"]
    F_near = res["F"]; F_far = res["F2"]; T_near = res["T1"]; T_far = res["T2"]
    dte_near = res["DTE1"]; dte_far = res["DTE2"]
    put_skew, call_skew = build_30day_skew(
        df_near, df_far, dte_near, dte_far, spot, F_near, F_far, rfr_decimal,
    )
    near_by_K = {float(K): row for K, row in
                 zip(df_near["strike"], df_near.to_dict("records"))}
    far_by_K = {float(K): row for K, row in
                zip(df_far["strike"], df_far.to_dict("records"))}
    return {
        "day": day_str,
        "spot": spot,
        "rfr": rfr_decimal,
        "F_near": F_near, "F_far": F_far,
        "T_near": T_near, "T_far": T_far,
        "dte_near": dte_near, "dte_far": dte_far,
        "df_near": df_near, "df_far": df_far,
        "near_by_K": near_by_K, "far_by_K": far_by_K,
        "put_skew": put_skew, "call_skew": call_skew,
    }


def section_header(out, title: str):
    line = "=" * 78
    out.append(line)
    out.append(title)
    out.append(line)


def main():
    out: list[str] = []
    print(f"Loading {T0}, {T1} ...")
    ev_chain = load_spx_options(data_dir=DATA_DIR, quote_dates=[T0, T1])
    rfr_lookup = fetch_rfr([pd.Timestamp(T0), pd.Timestamp(T1)])

    s0 = collect_day_state(T0, ev_chain, rfr_lookup)
    s1 = collect_day_state(T1, ev_chain, rfr_lookup)

    for s in (s0, s1):
        section_header(out, f"{s['day']} chain summary")
        out.append(f"  spot={s['spot']:.2f}  F_near={s['F_near']:.2f}  "
                   f"T_near={s['T_near']:.4f} ({s['dte_near']}d)  "
                   f"T_far={s['T_far']:.4f} ({s['dte_far']}d)")
        out.append(f"  near chain: {len(s['df_near'])} strikes "
                   f"[{s['df_near']['strike'].min():.0f} .. "
                   f"{s['df_near']['strike'].max():.0f}]")
        out.append(f"  far chain:  {len(s['df_far'])} strikes "
                   f"[{s['df_far']['strike'].min():.0f} .. "
                   f"{s['df_far']['strike'].max():.0f}]")
        out.append(f"  call_skew_30d: {len(s['call_skew'])} strikes  "
                   f"max K = {max(s['call_skew']) if s['call_skew'] else 'EMPTY'}")
        out.append(f"  put_skew_30d:  {len(s['put_skew'])} strikes  "
                   f"min K = {min(s['put_skew']) if s['put_skew'] else 'EMPTY'}")
        out.append("")

    # Common strike range for the call-wing comparison: from min(t0_max_in_skew, t1_max_in_skew)
    # up to max(t0_near_max, t1_near_max).
    K_max_t0 = max(s0["call_skew"]) if s0["call_skew"] else 0
    K_max_t1 = max(s1["call_skew"]) if s1["call_skew"] else 0
    chain_max_t0 = float(s0["df_near"]["strike"].max())
    chain_max_t1 = float(s1["df_near"]["strike"].max())
    K_lo = min(K_max_t0, K_max_t1)
    K_hi = max(chain_max_t0, chain_max_t1)
    section_header(out, f"Call wing comparison: strikes K > {K_lo:.0f} up to {K_hi:.0f}")
    out.append(f"  K_max in call_skew_30d  : t0={K_max_t0:.0f}  t1={K_max_t1:.0f}")
    out.append(f"  K_max in near chain     : t0={chain_max_t0:.0f}  t1={chain_max_t1:.0f}")
    out.append("")

    # Build the strike list to inspect: union of t0+t1 near strikes above K_lo
    inspect_strikes = sorted(
        set(float(K) for K in s0["df_near"]["strike"] if K > K_lo)
        | set(float(K) for K in s1["df_near"]["strike"] if K > K_lo)
    )

    for s, label in ((s0, "t0=" + T0), (s1, "t1=" + T1)):
        section_header(out, f"Per-strike call rejection trace — {label}")
        header = (f"  {'K':>7}  {'cbid':>7} {'cask':>7} {'cmid':>7}  "
                  f"{'pbid':>7} {'pask':>7}  reason")
        out.append(header)
        out.append("  " + "-" * (len(header) - 2))
        for K in inspect_strikes:
            near_row = s["near_by_K"].get(K)
            far_row = s["far_by_K"].get(K)
            if near_row is None:
                row_str = (f"  {K:>7.0f}  {'-':>7} {'-':>7} {'-':>7}  "
                           f"{'-':>7} {'-':>7}  K not in NEAR chain")
                out.append(row_str)
                continue
            cbid = float(near_row.get("cbid", 0) or 0)
            cask = float(near_row.get("cask", 0) or 0)
            cmid = float(near_row.get("cmid", float("nan")))
            pbid = float(near_row.get("pbid", 0) or 0)
            pask = float(near_row.get("pask", 0) or 0)
            cmid_disp = "nan" if math.isnan(cmid) else f"{cmid:.3f}"
            reason = reason_for_call_exclusion(
                near_row, far_row, K, s["F_near"], s["F_far"],
                s["T_near"], s["T_far"], s["dte_near"], s["dte_far"], s["rfr"],
            )
            out.append(f"  {K:>7.0f}  {cbid:>7.3f} {cask:>7.3f} {cmid_disp:>7}  "
                       f"{pbid:>7.3f} {pask:>7.3f}  {reason}")
        out.append("")

    # Side-by-side comparison: which strikes admitted on t0 but rejected on t1?
    section_header(out, "t0-vs-t1 admission table (CALL side)")
    out.append(f"  {'K':>7}  {'t0':>20}  {'t1':>20}")
    out.append("  " + "-" * 56)
    for K in inspect_strikes:
        t0_in = K in s0["call_skew"]
        t1_in = K in s1["call_skew"]
        if t0_in and not t1_in:
            t0r = "ADMITTED"
            t1r = reason_for_call_exclusion(
                s1["near_by_K"].get(K), s1["far_by_K"].get(K), K,
                s1["F_near"], s1["F_far"], s1["T_near"], s1["T_far"],
                s1["dte_near"], s1["dte_far"], s1["rfr"],
            ) if s1["near_by_K"].get(K) is not None else "NOT IN t1 NEAR"
            out.append(f"  {K:>7.0f}  {t0r:>20}  {t1r}")
        elif t1_in and not t0_in:
            t1r = "ADMITTED"
            t0r = reason_for_call_exclusion(
                s0["near_by_K"].get(K), s0["far_by_K"].get(K), K,
                s0["F_near"], s0["F_far"], s0["T_near"], s0["T_far"],
                s0["dte_near"], s0["dte_far"], s0["rfr"],
            ) if s0["near_by_K"].get(K) is not None else "NOT IN t0 NEAR"
            out.append(f"  {K:>7.0f}  {t0r}  {t1r:>20}")

    # ── Categorise t1 rejection reasons ──────────────────────────────────
    section_header(out, "Summary of t1 call-wing rejection causes")
    bins = {"near_zero_bid": 0, "far_zero_bid": 0, "k_not_in_far": 0,
            "k_not_in_near": 0, "cmid_floor": 0, "iv_solver_fail": 0,
            "var30_neg": 0, "passed": 0, "other": 0}
    for K in inspect_strikes:
        if K <= K_max_t1:  # only count strikes that should extend the wing
            continue
        near_row = s1["near_by_K"].get(K)
        if near_row is None:
            bins["k_not_in_near"] += 1
            continue
        far_row = s1["far_by_K"].get(K)
        reason = reason_for_call_exclusion(
            near_row, far_row, K, s1["F_near"], s1["F_far"],
            s1["T_near"], s1["T_far"], s1["dte_near"], s1["dte_far"], s1["rfr"],
        )
        if "near call bid = 0" in reason:
            bins["near_zero_bid"] += 1
        elif "far call bid = 0" in reason:
            bins["far_zero_bid"] += 1
        elif "K not in far chain" in reason:
            bins["k_not_in_far"] += 1
        elif "bid floor" in reason:
            bins["cmid_floor"] += 1
        elif "bs_iv" in reason and "failed" in reason:
            bins["iv_solver_fail"] += 1
        elif "var30 <= 0" in reason:
            bins["var30_neg"] += 1
        elif "PASSED" in reason:
            bins["passed"] += 1
        else:
            bins["other"] += 1
    for k, v in bins.items():
        out.append(f"  {k:<20}: {v}")

    # ── Recommendation ───────────────────────────────────────────────────
    section_header(out, "Recommendation")
    dominant = max(bins, key=bins.get)
    if dominant in ("near_zero_bid", "far_zero_bid"):
        rec = ("(A) Genuine market-data limitation: market makers pulled "
               "quotes on the call wing once vol exploded. The skew chart "
               "is correctly truncating. No code change needed; add a "
               "deck caption noting MM behaviour on Monday post-XIV blow-up.")
    elif dominant == "k_not_in_near":
        rec = ("(C) Strikes are absent from the t1 chain itself. Nothing "
               "to fix in the pipeline.")
    elif dominant == "k_not_in_far":
        rec = ("Mixed: t1 near chain lists these strikes but the far chain "
               "doesn't, so the intersection-based build_30day_skew drops "
               "them. Consider relaxing to allow near-only IV with a "
               "single-expiry vol caveat — out of scope here.")
    elif dominant == "iv_solver_fail":
        rec = ("(B) bs_iv is failing in the high-vol regime. Try widening "
               "the brentq bracket from [1e-6, 5.0] to [1e-6, 10.0] for "
               "high-vol days. Test on the diagnostic strikes first.")
    elif dominant == "cmid_floor":
        rec = ("Most rejected strikes have mids at the bid-floor "
               "(< 0.05). These carry no σ information; the floor filter "
               "is correctly dropping them.")
    else:
        rec = ("Mixed cause — see per-strike trace above. No single fix "
               "covers the majority of rejected strikes.")
    out.append("  " + rec)
    out.append("")

    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\nReport saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
