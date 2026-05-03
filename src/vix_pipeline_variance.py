#!/usr/bin/env python3
"""
VIX Analysis (variance-space chain construction).

Sibling of vix_pipeline_local.py. Computes the same 6-factor VIX
decomposition but as a sequential surface-construction chain in
variance space, rather than parallel counterfactuals with
vol-space scalar subtractions.

    Σ_0 (t0 chain at t0 spot)
     → Σ_1 (sticky strike: same chain at t1 spot)
     → Σ_2 (parallel: every strike's variance bumped by Δvar_atm)
     → Σ_3 (put shoulder: t1 IVs swapped on Σ_2)
     → Σ_4 (call shoulder)
     → Σ_5 (put wing)
     → Σ_6 (call wing)

    F_k = VIX(Σ_k) − VIX(Σ_{k-1})  for k = 1..6
    sum(F1..F6) = VIX_6 − VIX_0 ≈ ΔVIX

Output paths use the suffix "_variance" so this pipeline does not
overwrite vix_pipeline_local.py's CSVs:
    src/output/vix_decomposition_variance.csv
    src/output/vix_decomposition_variance_local.csv
    src/output/vix_decomposition_variance_chart.png
"""

from __future__ import annotations
import os
import json
import math
import argparse
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import scipy.stats as _ss  # _ss.norm used throughout; always available before any function def
norm = _ss.norm  # module-level alias so functions can use norm directly

# Module-level smoothing knob plumbed in from --hybrid-smoothing CLI arg.
HYBRID_SMOOTHING_SIGMA = 0.0


@dataclass
class VIXDecomposition:
    total_vix_change: float
    factor1_sticky_strike: float
    factor2_parallel_shift: float
    factor3_put_skew_grad: float
    factor4_call_skew_grad: float
    factor5_downside_conv: float
    factor6_upside_conv: float


# BLACK-SCHOLES IV (copied from /tmp/tastytrade-bot/methods.py)
def _bs_call(F: float, K: float, T: float, sigma: float, rfr: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0) * math.exp(-rfr * T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-rfr * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def _bs_put(F: float, K: float, T: float, sigma: float, rfr: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(K - F, 0.0) * math.exp(-rfr * T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-rfr * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def bs_iv(price: float, F: float, K: float, T: float, rfr: float,
          is_call: bool = True) -> float:
    """Return implied volatility in % given option price, forward, strike, T, rfr."""
    if price < 1e-8 or T <= 0:
        return 0.0
    def objective(sigma):
        if is_call:
            return _bs_call(F, K, T, sigma, rfr) - price
        else:
            return _bs_put(F, K, T, sigma, rfr) - price
    try:
        iv = brentq(objective, 1e-6, 5.0, maxiter=500)
    except ValueError:
        iv = 0.0
    return iv * 100.0



# VIX COMPUTATION
def build_chain_df(optionchain_list: list) -> pd.DataFrame:
    """Convert raw optionchain list rows into a clean DataFrame with mid prices."""
    rows = []
    for row in optionchain_list:
        strike = float(row.get("strike", 0))
        cbid = float(row.get("cbid", 0) or 0)
        cask = float(row.get("cask", 0) or 0)
        pbid = float(row.get("pbid", 0) or 0)
        pask = float(row.get("pask", 0) or 0)
        rows.append({
            "strike": strike,
            "cmid": (cbid + cask) / 2 if (cbid > 0 and cask > 0) else float("nan"),
            "pmid": (pbid + pask) / 2 if (pbid > 0 and pask > 0) else float("nan"),
            "cbid": cbid, "cask": cask, "pbid": pbid, "pask": pask,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("strike").reset_index(drop=True)
    return df

def compute_forward(df: pd.DataFrame, spot: float, rfr: float, T: float) -> float:
    """Compute model-free forward price via put-call parity at ATM forward strike.

    Finds K_atmf = strike minimizing |cmid - pmid| (model-free ATM),
    then F = K_atmf + exp(rT) * (cmid - pmid).
    """
    if "cmid" not in df.columns or "pmid" not in df.columns:
        return spot
    pc_diff = (df["cmid"] - df["pmid"]).abs()
    if pc_diff.empty:
        return spot
    idx = pc_diff.idxmin()
    K_atmf = float(df.loc[idx, "strike"])
    cmid = float(df.loc[idx, "cmid"])
    pmid = float(df.loc[idx, "pmid"])
    if math.isnan(cmid): cmid = 0.0
    if math.isnan(pmid): pmid = 0.0
    return K_atmf + math.exp(rfr * T) * (cmid - pmid)

def _is_zero_bid(row, side: str) -> bool:
    """Cboe per-side zero-bid check.
    side='put'  -> put bid == 0
    side='call' -> call bid == 0
    Used by the CBOE zero-bid truncation rule and by per-strike
    validity checks in skew construction and bucket selection.
    """
    if side == 'put':
        return (row.get("pbid", 0) or 0) == 0
    return (row.get("cbid", 0) or 0) == 0


def compute_vix_variance(df: pd.DataFrame, F: float, rfr: float, T: float) -> tuple[float, dict[float, float], float]:
    """
    Compute variance σ² for a single expiry using the full CBOE formula.

    σ² = (2/T) × Σ[ΔKᵢ/Kᵢ² × e^(RT) × Q(Kᵢ)] − (1/T) × [F/K₀ − 1]²

    The e^(RT) factor un-discounts option mids to forward measure
    (Cboe whitepaper Exhibit 4).

    CBOE zero-bid truncation (per-side): starting from ATM (K₀), walk
    outward. On the put side (left of K₀) two consecutive zero-put-bid
    strikes terminate the walk. On the call side (right of K₀) two
    consecutive zero-call-bid strikes terminate the walk.

    Returns (total_var, contrib_dict, forward_adj) where:
      - contrib_dict[K] = (2/T) * (dK/K²) * e^(RT) * Q for each valid strike
      - total_var = sum(contrib_dict.values()) - forward_adj (clamped >= 0)
      - forward_adj = (1/T) * ((F/K0) - 1)²  (for full VIX; skip for buckets)
    """
    raw = df[["strike", "cmid", "pmid", "cbid", "cask", "pbid", "pask"]].to_dict("records")
    if not raw:
        return (0.0, {}, 0.0)

    K0_strike = max((r["strike"] for r in raw if r["strike"] <= F), default=None)
    if K0_strike is None:
        return (0.0, {}, 0.0)

    idx_atm = next(i for i, r in enumerate(raw) if r["strike"] == K0_strike)

    valid = [True] * len(raw)

    for i in range(idx_atm - 1, -1, -1):
        if _is_zero_bid(raw[i], 'put') and _is_zero_bid(raw[i + 1], 'put'):
            valid[i] = False
            break
        valid[i] = True

    for i in range(idx_atm + 1, len(raw)):
        if _is_zero_bid(raw[i], 'call') and _is_zero_bid(raw[i - 1], 'call'):
            valid[i] = False
            break
        valid[i] = True

    rows = [r for i, r in enumerate(raw) if valid[i]]
    if len(rows) < 2:
        return (0.0, {}, 0.0)

    strikes = np.array([r["strike"] for r in rows])
    cmids   = np.array([r["cmid"] for r in rows])
    pmids   = np.array([r["pmid"] for r in rows])

    Q = np.empty_like(strikes, dtype=float)
    for i, K in enumerate(strikes):
        if K < K0_strike:
            Q[i] = pmids[i] if not math.isnan(pmids[i]) else 0.0
        elif K > K0_strike:
            Q[i] = cmids[i] if not math.isnan(cmids[i]) else 0.0
        else:
            c = cmids[i] if not math.isnan(cmids[i]) else 0.0
            p = pmids[i] if not math.isnan(pmids[i]) else 0.0
            Q[i] = (c + p) / 2.0

    n = len(strikes)
    dK = np.zeros(n, dtype=float)
    dK[0]    = strikes[1] - strikes[0]
    dK[-1]   = strikes[-1] - strikes[-2]
    dK[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0

    contrib_dict = {}
    for i in range(n):
        K = strikes[i]
        if K > 0 and Q[i] > 0:
            contrib_dict[K] = (2.0 / T) * (dK[i] / (K ** 2)) * math.exp(rfr * T) * Q[i]

    forward_adj = (1.0 / T) * ((F / K0_strike - 1) ** 2)
    total_var = sum(contrib_dict.values()) - forward_adj
    return (max(total_var, 0.0), contrib_dict, forward_adj)


def get_strikes_in_delta_bucket(
    chain_df: pd.DataFrame,
    spot: float,
    T: float,
    rfr: float,
    delta_lo: float,
    delta_hi: float,
    side: str,
    lower_excl: bool = False,
    upper_excl: bool = False,
) -> set[float]:
    """
    Return strikes whose N(d1) delta falls in the [delta_lo, delta_hi]
    interval (boundary inclusivity controlled by lower_excl / upper_excl).

    For each strike K in chain_df:
      1. Use bs_iv to get sigma from the option price (pmid for puts, cmid for calls)
      2. Compute d1 = (ln(spot/K) + 0.5*sigma²*T) / (sigma*sqrt(T))
      3. delta = N(d1) - 1 for puts, N(d1) for calls
      4. Filter to the chosen interval. Per-side bid validity: skip
         strikes whose relevant-side bid is zero (Cboe spec — no
         cross-side fallback).

    Returns set of strikes in the delta bucket.
    """
    df = chain_df.sort_values("strike").reset_index(drop=True)
    strikes_in = set()
    sqrt_T = math.sqrt(T)

    for _, row in df.iterrows():
        K = float(row["strike"])
        if K <= 0:
            continue

        if _is_zero_bid(row.to_dict(), side):
            continue
        if side == 'put':
            price = float(row["pmid"])
        else:
            price = float(row["cmid"])
        if math.isnan(price) or price <= 0:
            continue

        sigma_iv = bs_iv(price, spot, K, T, rfr, is_call=(side == 'call'))
        if sigma_iv <= 0:
            sigma_iv = 20.0  # fallback

        sigma = sigma_iv / 100.0
        d1 = (math.log(spot / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        if side == 'put':
            delta_val = norm.cdf(d1) - 1.0
        else:
            delta_val = norm.cdf(d1)

        lo_ok = (delta_val > delta_lo) if lower_excl else (delta_val >= delta_lo)
        hi_ok = (delta_val < delta_hi) if upper_excl else (delta_val <= delta_hi)
        if lo_ok and hi_ok:
            strikes_in.add(K)

    return strikes_in


def find_nearest_expiries(optionchain: dict, snapshot_date: date,
                          target_dte: int = 30):
    """
    Find two expiries that STRADDLE target_dte days from snapshot.
    Returns (near_exp, near_dte), (next_exp, next_dte) where near_dte <= target_dte < next_dte.
    Falls back to two closest if no perfect straddle exists.
    """
    results = []
    for exp_str in optionchain.keys():
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_dt - snapshot_date).days
        if dte > 0:
            results.append((exp_str, dte))
    results.sort(key=lambda x: x[1])

    if len(results) < 2:
        return results[:2] if results else []

    # Find where target_dte falls in the sorted DTE list
    # We want near <= target < next (bracket the target)
    near = None
    far = None
    for i, (exp_str, dte) in enumerate(results):
        if dte <= target_dte:
            near = (exp_str, dte)
        elif dte > target_dte and near is not None:
            far = (exp_str, dte)
            break
        elif dte > target_dte and near is None:
            # target is before all expiries - use two shortest
            near = (exp_str, dte)
            far = results[i + 1] if i + 1 < len(results) else results[-1]
            break

    if near is None:
        # All expiries are > target_dte
        near = results[0]
        far = results[1]
    elif far is None:
        # No expiry found > target_dte - use two longest
        near = results[-2]
        far = results[-1]

    return [near, far]

def compute_vix_for_snapshot(spot: float, rfr: float,
                               optionchain: dict,
                               snapshot_date: date) -> dict | None:
    """
    Compute 30-day constant-maturity VIX for a single snapshot.

    Returns dict with keys:
        date, spot, vix_computed, near_exp, next_exp, DTE1, DTE2,
        IV_30d, sigma_30d, rfr, K_atm1, K_atm2,
        chain1_df, chain2_df, F, T1, T2, vix_actual
    """
    # ── Find two closest expiries to 30 days ───────────────────────────────
    expiry_pair = find_nearest_expiries(optionchain, snapshot_date, target_dte=30)
    if len(expiry_pair) < 2:
        return None

    exp1_str, dte1 = expiry_pair[0]
    exp2_str, dte2 = expiry_pair[1]
    if dte1 <= 0 or dte2 <= 0 or dte1 == dte2:
        return None

    # ── Build DataFrames for each expiry ────────────────────────────────────
    df1 = build_chain_df(optionchain[exp1_str])
    df2 = build_chain_df(optionchain[exp2_str])

    T1 = dte1 / 365.0
    T2 = dte2 / 365.0

    # ── ATM strikes ─────────────────────────────────────────────────────────
    K_atm1 = float(df1.loc[(df1["strike"] - spot).abs().idxmin(), "strike"])
    K_atm2 = float(df2.loc[(df2["strike"] - spot).abs().idxmin(), "strike"])

    # ── Forward prices ──────────────────────────────────────────────────────
    F1 = compute_forward(df1, K_atm1, rfr, T1)
    F2 = compute_forward(df2, K_atm2, rfr, T2)

    # ── Compute FULL CBOE variance for each expiry ─────────────────────────
    var1, _, _ = compute_vix_variance(df1, F1, rfr, T1)
    var2, _, _ = compute_vix_variance(df2, F2, rfr, T2)

    if var1 <= 0 or var2 <= 0:
        return None

    # ── Two-expiry constant-maturity formula ─────────────────────────────────
    # Cboe constant-maturity formula: interpolate total variance T·σ² in
    # time, then divide by T30 to re-annualize (whitepaper Exhibit A2).
    T30 = 30.0 / 365.0

    var_30d = (T1 * var1 * (T2 - T30) + T2 * var2 * (T30 - T1)) / (T30 * (T2 - T1))
    if var_30d < 0:
        return None

    vix_computed = 100.0 * math.sqrt(var_30d)

    # ── 30d blended vol (sigma_30d) ────────────────────────────────────────
    # This is the ATM-equivalent vol used throughout the decomposition.
    sigma_30d = math.sqrt(var_30d)
    IV_30d = sigma_30d * 100.0   # convert to percentage for compatibility

    return {
        "date": snapshot_date.isoformat(),
        "spot": spot,
        "vix_computed": vix_computed,
        "near_exp": exp1_str,
        "next_exp": exp2_str,
        "DTE1": dte1,
        "DTE2": dte2,
        "IV_30d": IV_30d,
        "sigma_30d": sigma_30d,
        "rfr": rfr,
        "K_atm1": K_atm1,
        "K_atm2": K_atm2,
        "chain1_df": df1,
        "chain2_df": df2,
        "F": F1,
        "F2": F2,
        "T1": T1,
        "T2": T2,
        "vix_actual": None,   # filled in later
    }

# SKEW & INTERPOLATION
def build_30day_skew(df_near: pd.DataFrame, df_far: pd.DataFrame,
                     dte_near: int, dte_far: int,
                     spot: float, F_near: float, F_far: float,
                     rfr: float) -> tuple[dict[float, float], dict[float, float]]:
    """
    Build 30-day interpolated put and call skews from near/far expiry chains.

    For each strike K in the union of near+far strikes:
    1. Compute IV from option price at that strike for near-expiry
    2. Compute IV from option price at that strike for far-expiry
    3. Interpolate variance to 30-day: Var30(K) = w1*Var_near(K) + w2*Var_far(K)
    4. Convert to vol: σ30(K) = √(Var30(K) × 365/30) × 100

    Returns (put_skew_30d, call_skew_30d) where each is strike→vol dict.
    """
    T_near = dte_near / 365.0
    T_far = dte_far / 365.0
    T30 = 30.0 / 365.0

    # Interpolation weights (same for all strikes)
    w1 = (dte_far - 30.0) / (dte_far - dte_near)
    w2 = (30.0 - dte_near) / (dte_far - dte_near)

    # ATM strike for classification (use near-expiry ATM)
    K_atm_near = float(df_near.loc[(df_near["strike"] - spot).abs().idxmin(), "strike"])

    # Union of all strikes
    all_strikes = sorted(set(df_near["strike"].tolist()) | set(df_far["strike"].tolist()))

    put_skew_30d = {}
    call_skew_30d = {}

    for K in all_strikes:
        if K <= 0:
            continue

        side_str = 'put' if K < K_atm_near else 'call'
        is_put_near = (side_str == 'put')

        # ── Near-expiry: find nearest strike and compute IV ─────────────────
        near_strikes = df_near["strike"].values
        idx_near = np.argmin(np.abs(near_strikes - K))
        K_near_nearest = near_strikes[idx_near]
        near_row = df_near[df_near["strike"] == K_near_nearest].iloc[0]

        # Per-side validity: skip if relevant bid is zero
        if _is_zero_bid(near_row.to_dict(), side_str):
            continue
        price_near = float(near_row["pmid"] if side_str == 'put' else near_row["cmid"])
        if math.isnan(price_near) or price_near <= 0:
            continue
        iv_near = bs_iv(price_near, F_near, K_near_nearest, T_near, rfr,
                        is_call=not is_put_near)
        if iv_near <= 0:
            continue

        # ── Far-expiry: find nearest strike and compute IV ──────────────────
        far_strikes = df_far["strike"].values
        idx_far = np.argmin(np.abs(far_strikes - K))
        K_far_nearest = far_strikes[idx_far]
        far_row = df_far[df_far["strike"] == K_far_nearest].iloc[0]

        if _is_zero_bid(far_row.to_dict(), side_str):
            continue
        price_far = float(far_row["pmid"] if side_str == 'put' else far_row["cmid"])
        if math.isnan(price_far) or price_far <= 0:
            continue
        iv_far = bs_iv(price_far, F_far, K_far_nearest, T_far, rfr,
                       is_call=not is_put_near)
        if iv_far <= 0:
            continue

        # ── Variance interpolation ─────────────────────────────────────────
        var_near = (iv_near / 100.0) ** 2 * T_near
        var_far = (iv_far / 100.0) ** 2 * T_far
        var30 = w1 * var_near + w2 * var_far

        if var30 <= 0:
            continue

        # Convert to 30-day vol (%)
        vol30 = math.sqrt(var30 / T30) * 100.0

        # ── Classify into put vs call skew ─────────────────────────────────
        if K < K_atm_near:
            put_skew_30d[K] = vol30
        else:
            call_skew_30d[K] = vol30

    return put_skew_30d, call_skew_30d

def get_vol_at_strike(skew_dict: dict[float, float], target_strike: float) -> float:
    """
    Cubic spline interpolation of vol at target_strike from a skew dict (strike→vol).
    Returns edge values if target is outside the strike range.
    """
    if not skew_dict:
        return 0.0
    strikes = sorted(skew_dict.keys())
    vols = [skew_dict[k] for k in strikes]

    if target_strike <= strikes[0]:
        return vols[0]
    if target_strike >= strikes[-1]:
        return vols[-1]

    # Fit cubic spline and evaluate at target strike
    cs = CubicSpline(strikes, vols, bc_type="natural")
    return float(cs(target_strike))

def _signed_delta(K: float, S: float, vol30: float, T30: float, side: str) -> float:
    """Signed delta using N(d1): put = N(d1) - 1, call = N(d1)."""
    sigma = vol30 / 100.0
    sqrt_T30 = math.sqrt(T30)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T30) / (sigma * sqrt_T30)
    if side == 'put':
        return norm.cdf(d1) - 1.0
    else:
        return norm.cdf(d1)


def compute_vix_from_chains(df_near, df_far, F_near, F_far,
                            T1, T2, r):
    """Run compute_vix_variance on each chain, then Cboe two-expiry
    interpolation (linear-in-total-variance). Returns VIX in
    points, or None if either variance is non-positive."""

    var_n, _, _ = compute_vix_variance(df_near, F_near, r, T1)
    var_f, _, _ = compute_vix_variance(df_far,  F_far,  r, T2)
    if var_n <= 0 or var_f <= 0 or T2 == T1:
        return None
    T30 = 30.0 / 365.0
    var_30 = (T1 * var_n * (T2 - T30) + T2 * var_f * (T30 - T1)) \
             / (T30 * (T2 - T1))
    if var_30 < 0:
        return None
    return 100.0 * math.sqrt(var_30)


def apply_uniform_variance_bump(df, F, T, r, var_bump_annual):
    """Strike-independent parallel shift in variance space.

    For every strike: imply IV from the existing mid, convert
    IV → annualized variance σ², add var_bump_annual to that,
    convert back to IV, BS-reprice. The same annualized variance
    bump is applied at every strike — this is the natural
    'parallel' shift in the VIX formula's integration measure.

    Differs from a vol-pt parallel shift (σ + Δσ): the vol-space
    version makes the variance change strike-dependent because
    (σ+Δσ)² − σ² = 2σΔσ + Δσ². The variance-space version (this)
    bumps every strike's variance by the same amount.

    Strikes whose IV cannot be implied (NaN mid, IV solver fails)
    are left unchanged. Strikes whose bumped variance would be
    non-positive are also left unchanged.
    """

    out = df.copy()
    for i, row in out.iterrows():
        K = float(row["strike"])
        if K <= 0:
            continue

        for side, mid_col, pricer in [
            ("call", "cmid", _bs_call),
            ("put",  "pmid", _bs_put),
        ]:
            mid = float(row.get(mid_col, float("nan")))
            if math.isnan(mid) or mid <= 0:
                continue
            iv_pct = bs_iv(mid, F, K, T, r,
                           is_call=(side == "call"))
            if iv_pct <= 0:
                continue
            iv = iv_pct / 100.0
            var_old = iv * iv * T
            var_new = var_old + var_bump_annual * T
            if var_new <= 0:
                continue
            iv_new = math.sqrt(var_new / T)
            new_price = pricer(F, K, T, iv_new, r)
            if np.isfinite(new_price) and new_price > 0:
                out.at[i, mid_col] = new_price
    return out


def apply_bucket_swap(df, bucket_strikes, t1_iv_spline, S_old,
                      F, T, r, side, m_lo, m_hi):
    """Overwrite mids at bucket_strikes using t1's IV looked up
    at moneyness K/S_old, BS-priced under t0's frame (F, T, r).

    No level-shift adjustment — the caller has already applied any
    parallel/shoulder shift to the input chain.

    side='put'  updates pmid via _bs_put.
    side='call' updates cmid via _bs_call.
    Strikes outside [m_lo, m_hi] are left alone.
    """

    out = df.copy()
    if side == "put":
        bs_price = _bs_put
        mid_col  = "pmid"
    else:
        bs_price = _bs_call
        mid_col  = "cmid"

    for K in bucket_strikes:
        m = K / S_old
        if m < m_lo or m > m_hi:
            continue
        iv = float(t1_iv_spline(m))
        if iv <= 0 or not np.isfinite(iv):
            continue
        sigma = iv / 100.0
        new_price = bs_price(F, K, T, sigma, r)
        if not np.isfinite(new_price) or new_price <= 0:
            continue
        out.loc[out["strike"] == K, mid_col] = new_price
    return out


def run_decomposition(prev: dict, curr: dict) -> VIXDecomposition | None:
    """
    6-factor VIX decomposition via sequential variance-space chain
    (Cboe Aug 2025 framework).

    Builds intermediate volatility surfaces and computes each factor
    as a true VIX-pt difference between consecutive surfaces:

        Σ_0  t0 chain at t0 spot                       → VIX_0
        Σ_1  t0 chain, forward scaled to t1 spot       → F1 = VIX_1 − VIX_0
        Σ_2  Σ_1 with uniform variance bump +Δvar_atm  → F2 = VIX_2 − VIX_1
        Σ_3  Σ_2 with put-shoulder t1 IV swap          → F3 = VIX_3 − VIX_2
        Σ_4  Σ_3 with call-shoulder t1 IV swap         → F4 = VIX_4 − VIX_3
        Σ_5  Σ_4 with put-wing t1 IV swap              → F5 = VIX_5 − VIX_4
        Σ_6  Σ_5 with call-wing t1 IV swap             → F6 = VIX_6 − VIX_5

    By construction sum(F1..F6) = VIX_6 − VIX_0 ≈ ΔVIX. Residual
    sources:
      (a) Near-ATM strikes outside any bucket whose t1 IV differs
          from Σ_2's parallel-shifted approximation.
      (b) Σ_1 forward-scaling approximation
          (F_new ≈ F_old × S_new/S_old).

    F2 is computed in variance space: Δvar_annual = σ_new(S_new)² −
    σ_old(S_new)², applied uniformly to every strike's variance.
    This differs from Cboe's worked-example arithmetic, which treats
    the parallel shift as a vol-pt scalar at one strike. The
    variance-space convention is internally consistent with the rest
    of the chain (which computes F-factors as differences of
    VIX = √variance values) and ensures the chain closes to ΔVIX.
    Numerical F2 values may differ from Cboe's tool by ~5–10% on
    big-move days; the F-factor interpretations are unchanged.

    Bucket boundaries (Cboe whitepaper Exhibit 11):
        F3 put shoulder       delta in [-.45, -.15)
        F4 call shoulder      delta in ( .15,  .45]
        F5 put wing           delta in [-.15, -.01)
        F6 call wing          delta in ( .01,  .15]

    Bucket strikes for each Σ_k are identified from that step's INPUT
    chain (df1_2 for Σ_3, df1_3 for Σ_4, etc.), so boundary strikes
    near ±0.15 and ±0.45 are classified using the post-shift IV.
    """
    put_old  = prev.get("put_skew_30d", {})
    put_new  = curr.get("put_skew_30d", {})
    call_old = prev.get("call_skew_30d", {})
    call_new = curr.get("call_skew_30d", {})

    if not put_old or not put_new or not call_old or not call_new:
        return None

    S_old = prev["spot"]
    S_new = curr["spot"]
    F_old      = prev["F"]
    F_far_old  = prev["F2"]
    T1 = prev["T1"]
    T2 = prev["T2"]
    r  = prev["rfr"]
    df1 = prev["chain1_df"]
    df2 = prev["chain2_df"]

    # Direction-appropriate side for F1/F2 vol-pt lookups
    if S_new < S_old:
        skew_old_use, skew_new_use = put_old, put_new
    else:
        skew_old_use, skew_new_use = call_old, call_new

    vol_old_at_S_new = get_vol_at_strike(skew_old_use, S_new)
    vol_old_at_S_old = get_vol_at_strike(skew_old_use, S_old)
    vol_new_at_S_new = get_vol_at_strike(skew_new_use, S_new)

    # ── Σ_0 baseline ──────────────────────────────────────────────
    VIX_0 = prev["vix_computed"]

    # ── Σ_1 sticky strike ────────────────────────────────────────
    # Same chain, but evaluate VIX with forward scaled to new spot.
    # Approximation: F_new ≈ F_old * (S_new / S_old). A purely
    # correct Σ_1 would re-solve put-call parity at S_new on the
    # unchanged t0 chain, but no row has changed so parity solution
    # degenerates. Proportional scaling captures leading-order spot
    # effect cleanly.
    spot_ratio  = S_new / S_old
    F_new_S_new = F_old * spot_ratio
    F_far_S_new = F_far_old * spot_ratio

    VIX_1 = compute_vix_from_chains(df1, df2,
                                    F_new_S_new, F_far_S_new,
                                    T1, T2, r)
    if VIX_1 is None:
        VIX_1 = VIX_0
    F1 = VIX_1 - VIX_0

    # ── Σ_2 parallel shift in VARIANCE space ─────────────────────
    # Δvar_annual = σ_new(S_new)² − σ_old(S_new)²  [decimal vol²]
    # Apply uniformly to every strike's variance on both chains.
    sigma_new_dec = vol_new_at_S_new / 100.0
    sigma_old_dec = vol_old_at_S_new / 100.0
    var_bump_annual = sigma_new_dec**2 - sigma_old_dec**2

    df1_2 = apply_uniform_variance_bump(df1, F_old,     T1, r,
                                        var_bump_annual)
    df2_2 = apply_uniform_variance_bump(df2, F_far_old, T2, r,
                                        var_bump_annual)

    VIX_2 = compute_vix_from_chains(df1_2, df2_2,
                                    F_new_S_new, F_far_S_new,
                                    T1, T2, r)
    if VIX_2 is None:
        VIX_2 = VIX_1
        F2 = 0.0
    else:
        F2 = VIX_2 - VIX_1

    # ── Build t1 IV spline (used for all bucket swaps) ───────────
    t1_full_skew = {**put_new, **call_new}
    sorted_K = sorted(t1_full_skew.keys())
    if len(sorted_K) < 4:
        # Cannot build spline — return F1, F2 only, zero out F3-F6
        VIX_actual_old = prev.get("vix_actual") or VIX_0
        VIX_actual_new = curr.get("vix_actual") or curr.get("vix_computed", 0.0)
        return VIXDecomposition(
            total_vix_change=VIX_actual_new - VIX_actual_old,
            factor1_sticky_strike=F1,
            factor2_parallel_shift=F2,
            factor3_put_skew_grad=0.0,
            factor4_call_skew_grad=0.0,
            factor5_downside_conv=0.0,
            factor6_upside_conv=0.0,
        )

    sigma_smooth = HYBRID_SMOOTHING_SIGMA
    m_t1  = np.array([K / S_new for K in sorted_K])
    iv_t1 = np.array([t1_full_skew[K] for K in sorted_K], dtype=float)
    if sigma_smooth and sigma_smooth > 0:
        iv_t1 = gaussian_filter1d(iv_t1, sigma=float(sigma_smooth))
    t1_iv_spline = CubicSpline(m_t1, iv_t1, bc_type="natural")
    m_lo, m_hi = m_t1[0], m_t1[-1]

    # ── Σ_3 put shoulder: [-0.45, -0.15) ─────────────────────────
    ps_strikes = get_strikes_in_delta_bucket(
        df1_2, S_old, T1, r,
        delta_lo=-0.45, delta_hi=-0.15, side="put",
        lower_excl=False, upper_excl=True,
    )
    df1_3 = apply_bucket_swap(df1_2, ps_strikes, t1_iv_spline,
                              S_old, F_old, T1, r, "put",
                              m_lo, m_hi)
    df2_3 = apply_bucket_swap(df2_2, ps_strikes, t1_iv_spline,
                              S_old, F_far_old, T2, r, "put",
                              m_lo, m_hi)
    VIX_3 = compute_vix_from_chains(df1_3, df2_3,
                                    F_new_S_new, F_far_S_new,
                                    T1, T2, r)
    if VIX_3 is None:
        VIX_3 = VIX_2
        F3 = 0.0
    else:
        F3 = VIX_3 - VIX_2

    # ── Σ_4 call shoulder: (0.15, 0.45] ──────────────────────────
    cs_strikes = get_strikes_in_delta_bucket(
        df1_3, S_old, T1, r,
        delta_lo=0.15, delta_hi=0.45, side="call",
        lower_excl=True, upper_excl=False,
    )
    df1_4 = apply_bucket_swap(df1_3, cs_strikes, t1_iv_spline,
                              S_old, F_old, T1, r, "call",
                              m_lo, m_hi)
    df2_4 = apply_bucket_swap(df2_3, cs_strikes, t1_iv_spline,
                              S_old, F_far_old, T2, r, "call",
                              m_lo, m_hi)
    VIX_4 = compute_vix_from_chains(df1_4, df2_4,
                                    F_new_S_new, F_far_S_new,
                                    T1, T2, r)
    if VIX_4 is None:
        VIX_4 = VIX_3
        F4 = 0.0
    else:
        F4 = VIX_4 - VIX_3

    # ── Σ_5 put wing: [-0.15, -0.01) ─────────────────────────────
    pw_strikes = get_strikes_in_delta_bucket(
        df1_4, S_old, T1, r,
        delta_lo=-0.15, delta_hi=-0.01, side="put",
        lower_excl=False, upper_excl=True,
    )
    df1_5 = apply_bucket_swap(df1_4, pw_strikes, t1_iv_spline,
                              S_old, F_old, T1, r, "put",
                              m_lo, m_hi)
    df2_5 = apply_bucket_swap(df2_4, pw_strikes, t1_iv_spline,
                              S_old, F_far_old, T2, r, "put",
                              m_lo, m_hi)
    VIX_5 = compute_vix_from_chains(df1_5, df2_5,
                                    F_new_S_new, F_far_S_new,
                                    T1, T2, r)
    if VIX_5 is None:
        VIX_5 = VIX_4
        F5 = 0.0
    else:
        F5 = VIX_5 - VIX_4

    # ── Σ_6 call wing: (0.01, 0.15] ──────────────────────────────
    cw_strikes = get_strikes_in_delta_bucket(
        df1_5, S_old, T1, r,
        delta_lo=0.01, delta_hi=0.15, side="call",
        lower_excl=True, upper_excl=False,
    )
    df1_6 = apply_bucket_swap(df1_5, cw_strikes, t1_iv_spline,
                              S_old, F_old, T1, r, "call",
                              m_lo, m_hi)
    df2_6 = apply_bucket_swap(df2_5, cw_strikes, t1_iv_spline,
                              S_old, F_far_old, T2, r, "call",
                              m_lo, m_hi)
    VIX_6 = compute_vix_from_chains(df1_6, df2_6,
                                    F_new_S_new, F_far_S_new,
                                    T1, T2, r)
    if VIX_6 is None:
        VIX_6 = VIX_5
        F6 = 0.0
    else:
        F6 = VIX_6 - VIX_5

    # ── Ground truth ──────────────────────────────────────────────
    VIX_actual_old = prev.get("vix_actual") or VIX_0
    VIX_actual_new = curr.get("vix_actual") or curr.get("vix_computed", 0.0)
    total = VIX_actual_new - VIX_actual_old

    return VIXDecomposition(
        total_vix_change=total,
        factor1_sticky_strike=F1,
        factor2_parallel_shift=F2,
        factor3_put_skew_grad=F3,
        factor4_call_skew_grad=F4,
        factor5_downside_conv=F5,
        factor6_upside_conv=F6,
    )

# CBOE DATA
def fetch_cboe_vix_historical():
    """Fetch VIX daily closing values from CBOE CSV."""
    try:
        import urllib.request
        url = ("https://cdn.cboe.com/api/globalbenchmarks/indices/"
               "benchmark-values/VIX_History.csv")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")
        lines = text.strip().split("\n")
        records = {}
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                date_str = parts[0].strip()
                try:
                    records[date_str] = float(parts[1].strip())
                except ValueError:
                    pass
        return records
    except Exception as e:
        return {}


def fetch_snapshots_2026(data_dir: str | None = None):
    """
    Adapt OptionsDX local 2023 EOD data into the Supabase-style snapshot list
    that main() / compute_vix_for_snapshot() consume.

    rfr convention: spx_local_loader.fetch_rfr returns DECIMAL (^IRX/100). The
    snapshot payload field "rfr" must be PERCENT, because main() does
    `rfr = float(payload["rfr"]) / 100.0`. We therefore multiply by 100 here.

    The imports of load_spx_options / fetch_rfr / fetch_vix_actual are
    function-local because spx_local_loader.py imports from this module at
    module top — a top-level import here would create a circular-import
    deadlock.
    """
    from spx_local_loader import load_spx_options, fetch_rfr, fetch_vix_actual

    if data_dir is None:
        data_dir = os.path.expanduser("~/data/spx_eod")

    chain = load_spx_options(data_dir=data_dir)
    dates = sorted(chain["QUOTE_DATE"].unique())
    rfr_lookup = fetch_rfr(dates)
    vix_lookup = fetch_vix_actual(dates)

    snapshots = []
    n_done = 0
    for quote_dt, day_df in chain.groupby("QUOTE_DATE", sort=True):
        quote_date_str = pd.Timestamp(quote_dt).strftime("%Y-%m-%d")
        spot = float(day_df["UNDERLYING_LAST"].iloc[0])

        optionchain = {}
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

        if not optionchain:
            continue

        rfr_decimal = rfr_lookup.get(pd.Timestamp(quote_dt))
        rfr_pct = float(rfr_decimal) * 100.0 if rfr_decimal is not None else 0.0

        vix_val = vix_lookup.get(pd.Timestamp(quote_dt))
        vix_spot = float(vix_val) if vix_val is not None else None

        snapshots.append({
            "date": quote_date_str,
            "payload": {
                "SPX": {"spot": spot, "optionchain": optionchain},
                "VIX": {"spot": vix_spot},
                "rfr": rfr_pct,
            },
        })

        n_done += 1
        if n_done % 25 == 0:
            print(f"  fetch_snapshots_2026: {n_done} days built...")

    print(f"fetch_snapshots_2026: built {len(snapshots)} snapshots from {data_dir}")
    return snapshots


# MAIN
def main():
    parser = argparse.ArgumentParser(description="VIX decomposition pipeline")
    parser.add_argument(
        "--hybrid-smoothing", type=float, default=0.0,
        help="Gaussian smoothing sigma (in IV-percent samples) applied to t1's "
             "(moneyness, IV) curve before splining for the F3-F6 hybrid swap. "
             "0.0 disables smoothing.",
    )
    args, _ = parser.parse_known_args()
    global HYBRID_SMOOTHING_SIGMA
    HYBRID_SMOOTHING_SIGMA = float(args.hybrid_smoothing)
    print(f"Hybrid smoothing sigma: {HYBRID_SMOOTHING_SIGMA}\n")

    print("Loading SPX EOD snapshots from local OptionsDX archive...")
    snapshots = fetch_snapshots_2026()
    print(f"  Retrieved {len(snapshots)} snapshots\n")

    if not snapshots:
        print("No data found. Exiting.")
        return

    # CBOE historical VIX
    print("Fetching CBOE VIX historical data for validation...")
    cboe_vix = fetch_cboe_vix_historical()
    print(f"  Got {len(cboe_vix)} CBOE VIX records\n")

    # Compute VIX for each snapshot
    results = []
    skipped = 0

    for snap in snapshots:
        snap_date_str = snap["date"]
        snap_date = datetime.strptime(snap_date_str, "%Y-%m-%d").date()
        payload = snap["payload"]

        spot = payload.get("SPX", {}).get("spot")
        rfr_raw = payload.get("rfr", 0.0)
        if rfr_raw is None:
            rfr_raw = 0.0
        rfr = float(rfr_raw) / 100.0  # convert percentage to decimal

        optionchain = payload.get("SPX", {}).get("optionchain", {})
        vix_actual = payload.get("VIX", {}).get("spot") if isinstance(payload.get("VIX"), dict) else None

        if not optionchain or not spot:
            print(f"  Skipping {snap_date_str} -- missing data")
            skipped += 1
            continue

        vix_result = compute_vix_for_snapshot(spot, rfr, optionchain, snap_date)
        if vix_result is None:
            print(f"  Skipping {snap_date_str} -- VIX computation failed")
            skipped += 1
            continue

        # Fill in CBOE actual VIX
        date_key = snap_date_str
        if date_key in cboe_vix:
            vix_result["vix_actual"] = cboe_vix[date_key]
        elif vix_actual is not None:
            vix_result["vix_actual"] = vix_actual

        # ── Build 30-day interpolated skews from near+far expiries ────────
        df_near = vix_result["chain1_df"]
        df_far = vix_result["chain2_df"]
        dte_near = vix_result["DTE1"]
        dte_far = vix_result["DTE2"]
        F_near = vix_result["F"]
        F_far = vix_result["F2"]
        put_skew, call_skew = build_30day_skew(
            df_near, df_far, dte_near, dte_far, spot, F_near, F_far, rfr)
        vix_result["put_skew_30d"] = put_skew
        vix_result["call_skew_30d"] = call_skew

        results.append(vix_result)
        print(f"  {snap_date_str}: SPX={spot:,.2f}, "
              f"VIX_comp={vix_result['vix_computed']:.2f}, "
              f"VIX_actual={vix_result.get('vix_actual', 'N/A')}, "
              f"NearExp={vix_result['near_exp']}({vix_result['DTE1']}d), "
              f"NextExp={vix_result['next_exp']}({vix_result['DTE2']}d), "
              f"σ30d={vix_result['sigma_30d']*100:.2f}%")

    print(f"\nProcessed {len(results)} valid dates, {skipped} skipped\n")

    # ── VIX Decomposition (from 2nd date onwards) ──────────────────────────
    decompositions = [None]  # placeholder for index 0 (no prev date)
    print(f"\n{'Date':<12} {'F1':>8} {'F2':>8} {'F3h':>8} {'F4h':>8} {'F5h':>8} {'F6h':>8} "
          f"{'sum':>8} {'dVIXact':>8} {'resid':>8}")
    print("-" * 86)
    residual_rows = []  # (date, F1..F6, sum, dVIX_actual, residual) — actual only
    for i in range(1, len(results)):
        try:
            decomp = run_decomposition(results[i - 1], results[i])
            decompositions.append(decomp)
            results[i]["decomp"] = decomp
        except Exception as e:
            print(f"  Decomposition failed for {results[i]['date']}: {e}")
            import traceback
            traceback.print_exc()
            decompositions.append(None)
            continue

        if decomp is None:
            continue

        prev_actual = results[i - 1].get("vix_actual")
        curr_actual = results[i].get("vix_actual")
        if prev_actual is not None and curr_actual is not None:
            d_vix_actual = curr_actual - prev_actual
        else:
            d_vix_actual = results[i]["vix_computed"] - results[i - 1]["vix_computed"]

        sum_f = (decomp.factor1_sticky_strike + decomp.factor2_parallel_shift
                 + decomp.factor3_put_skew_grad + decomp.factor4_call_skew_grad
                 + decomp.factor5_downside_conv + decomp.factor6_upside_conv)
        residual = sum_f - d_vix_actual

        date_short = results[i]["date"][:10]
        print(f"{date_short:<12} "
              f"{decomp.factor1_sticky_strike:>8.3f} "
              f"{decomp.factor2_parallel_shift:>8.3f} "
              f"{decomp.factor3_put_skew_grad:>8.3f} "
              f"{decomp.factor4_call_skew_grad:>8.3f} "
              f"{decomp.factor5_downside_conv:>8.3f} "
              f"{decomp.factor6_upside_conv:>8.3f} "
              f"{sum_f:>8.3f} {d_vix_actual:>8.3f} {residual:>8.3f}")

        residual_rows.append({
            "date": date_short,
            "F1": decomp.factor1_sticky_strike,
            "F2": decomp.factor2_parallel_shift,
            "F3": decomp.factor3_put_skew_grad,
            "F4": decomp.factor4_call_skew_grad,
            "F5": decomp.factor5_downside_conv,
            "F6": decomp.factor6_upside_conv,
            "sum": sum_f,
            "d_vix_actual": d_vix_actual,
            "residual": residual,
        })

    # ── Hybrid summary stats + CSV ─────────────────────────────────────────
    if residual_rows:
        resid_arr = np.array([r["residual"] for r in residual_rows])
        f3_arr = np.array([r["F3"] for r in residual_rows])
        f4_arr = np.array([r["F4"] for r in residual_rows])
        f5_arr = np.array([r["F5"] for r in residual_rows])
        f6_arr = np.array([r["F6"] for r in residual_rows])
        print("\n" + "=" * 70)
        print(f"VARIANCE-CHAIN SUMMARY (smoothing_sigma={HYBRID_SMOOTHING_SIGMA})")
        print("=" * 70)
        print(f"  N days:            {len(residual_rows)}")
        print(f"  mean(residual):    {resid_arr.mean():+.4f}")
        print(f"  std(residual):     {resid_arr.std(ddof=0):.4f}")
        print(f"  max|residual|:     {np.abs(resid_arr).max():.4f}")
        print(f"  mean(F3_hybrid):   {f3_arr.mean():+.4f}")
        print(f"  mean(F4_hybrid):   {f4_arr.mean():+.4f}")
        print(f"  mean(F5_hybrid):   {f5_arr.mean():+.4f}")
        print(f"  mean(F6_hybrid):   {f6_arr.mean():+.4f}")

    # Write hybrid CSV
    output_dir_hybrid = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir_hybrid, exist_ok=True)
    hybrid_csv_path = os.path.join(output_dir_hybrid, "vix_decomposition_variance.csv")
    hybrid_rows = []
    for i, res in enumerate(results):
        decomp = decompositions[i] if i > 0 else None
        date_short = res["date"][:10]
        prev_actual = results[i - 1].get("vix_actual") if i > 0 else None
        curr_actual = res.get("vix_actual")
        if i > 0 and prev_actual is not None and curr_actual is not None:
            d_vix = curr_actual - prev_actual
            d_vix_source = "actual"
        elif i > 0:
            d_vix = res["vix_computed"] - results[i - 1]["vix_computed"]
            d_vix_source = "computed"
        else:
            d_vix = None
            d_vix_source = None
        row = {
            "date": date_short,
            "SPX": res["spot"],
            "VIX_computed": res["vix_computed"],
            "VIX_actual": curr_actual,
            "delta_VIX": d_vix,
            "delta_vix_source": d_vix_source,
            "F1": decomp.factor1_sticky_strike if decomp else None,
            "F2": decomp.factor2_parallel_shift if decomp else None,
            "F3": decomp.factor3_put_skew_grad if decomp else None,
            "F4": decomp.factor4_call_skew_grad if decomp else None,
            "F5": decomp.factor5_downside_conv if decomp else None,
            "F6": decomp.factor6_upside_conv if decomp else None,
        }
        if decomp:
            row["sum_factors"] = (decomp.factor1_sticky_strike
                                  + decomp.factor2_parallel_shift
                                  + decomp.factor3_put_skew_grad
                                  + decomp.factor4_call_skew_grad
                                  + decomp.factor5_downside_conv
                                  + decomp.factor6_upside_conv)
            row["residual"] = (row["sum_factors"] - d_vix) if d_vix is not None else None
        else:
            row["sum_factors"] = None
            row["residual"] = None
        hybrid_rows.append(row)

    pd.DataFrame(hybrid_rows).to_csv(hybrid_csv_path, index=False)
    print(f"\nHybrid decomposition CSV saved to {hybrid_csv_path}")

    # ── Build output table ─────────────────────────────────────────────────
    hdr = (f"{'Date':<12} {'SPX_Spot':>10} {'VIX_Comp':>10} "
           f"{'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'F6':>8} "
           f"{'VIX_Actual':>10}")
    sep = "=" * len(hdr)

    print(sep)
    print(hdr)
    print(sep)

    table_lines = []
    all_lines = []

    for i, res in enumerate(results):
        decomp = decompositions[i] if i > 0 else None
        date_str = res["date"][:10]
        spx_s = f"{res['spot']:>10,.2f}"
        vix_c = f"{res['vix_computed']:>10.2f}"

        if decomp:
            f1s = f"{decomp.factor1_sticky_strike:>8.2f}"
            f2s = f"{decomp.factor2_parallel_shift:>8.2f}"
            f3s = f"{decomp.factor3_put_skew_grad:>8.2f}"
            f4s = f"{decomp.factor4_call_skew_grad:>8.2f}"
            f5s = f"{decomp.factor5_downside_conv:>8.2f}"
            f6s = f"{decomp.factor6_upside_conv:>8.2f}"
        else:
            f1s = f2s = f3s = f4s = f5s = f6s = f"{'--':>8}"

        va = (f"{res.get('vix_actual', 'N/A'):>10.2f}"
              if res.get("vix_actual") is not None
              else f"{'N/A':>10}")

        line = f"{date_str:<12} {spx_s} {vix_c} {f1s} {f2s} {f3s} {f4s} {f5s} {f6s} {va}"
        table_lines.append(line)
        all_lines.append(line)
        print(line)

    print(sep)

    # ── Save decomposition CSV ─────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    decomp_csv_path = os.path.join(output_dir, "vix_decomposition_variance_local.csv")
    decomp_rows = []
    for i, res in enumerate(results):
        decomp = decompositions[i] if i > 0 else None
        row = {
            "date": res["date"][:10],
            "SPX_spot": res["spot"],
            "VIX_computed": res["vix_computed"],
            "VIX_actual": res.get("vix_actual"),
        }
        if decomp:
            row["F1_sticky_strike"] = decomp.factor1_sticky_strike
            row["F2_parallel_shift"] = decomp.factor2_parallel_shift
            row["F3_put_skew_grad"] = decomp.factor3_put_skew_grad
            row["F4_call_skew_grad"] = decomp.factor4_call_skew_grad
            row["F5_downside_conv"] = decomp.factor5_downside_conv
            row["F6_upside_conv"] = decomp.factor6_upside_conv
            row["sum_factors"] = sum([
                decomp.factor1_sticky_strike,
                decomp.factor2_parallel_shift,
                decomp.factor3_put_skew_grad,
                decomp.factor4_call_skew_grad,
                decomp.factor5_downside_conv,
                decomp.factor6_upside_conv,
            ])
            prev_actual = results[i - 1].get("vix_actual")
            curr_actual = res.get("vix_actual")
            if prev_actual is not None and curr_actual is not None:
                d_vix = curr_actual - prev_actual
                d_vix_source = "actual"
            else:
                d_vix = res["vix_computed"] - results[i - 1]["vix_computed"]
                d_vix_source = "computed"
            row["delta_vix"] = d_vix
            row["delta_vix_source"] = d_vix_source
            row["diff_computed"] = row["sum_factors"] - d_vix
        decomp_rows.append(row)
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(decomp_csv_path, index=False)
    print(f"\nDecomposition CSV saved to {decomp_csv_path}")

    # ── CBOE Comparison Summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CBOE COMPARISON SUMMARY")
    print("=" * 70)

    valid_pairs = [
        (r["vix_computed"], r["vix_actual"])
        for r in results if r.get("vix_actual") is not None
    ]

    if valid_pairs:
        errors = [comp - actual for comp, actual in valid_pairs]
        abs_errors = [abs(e) for e in errors]
        print(f"  Dates compared:              {len(valid_pairs)}")
        print(f"  Mean error (Comp - Actual): {np.mean(errors):+.3f}")
        print(f"  Mean |error|:               {np.mean(abs_errors):.3f}")
        print(f"  Max |error|:                {max(abs_errors):.3f}")
        print(f"  Min |error|:                {min(abs_errors):.3f}")
        print()
        print(f"  {'Date':<12} {'Computed':>10} {'Actual':>10} {'Error':>10}")
        print("  " + "-" * 44)
        for r in results:
            if r.get("vix_actual") is not None:
                err = r["vix_computed"] - r["vix_actual"]
                print(f"  {r['date'][:10]:<12} {r['vix_computed']:>10.2f} "
                      f"{r['vix_actual']:>10.2f} {err:>+10.3f}")
    else:
        print("  No CBOE actual VIX values available in the dataset.")
        print("  Note: The payload contains VIX spot values from the data source.")
        print("  CBOE live decomposition tool:")
        print("  https://www.cboe.com/en/tradable-products/vix/vix-decomposition/")

    # ── Save to file ───────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "vix_results_variance.txt")
    with open(output_path, "w") as f:
        f.write("VIX Analysis Results -- 2026+\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total dates: {len(results)}  |  Skipped: {skipped}\n")
        f.write(f"CBOE VIX historical records: {len(cboe_vix)}\n\n")
        f.write(sep + "\n")
        f.write(hdr + "\n")
        f.write(sep + "\n")
        for line in all_lines:
            f.write(line + "\n")
        f.write(sep + "\n")

        f.write("\n## CBOE Comparison Summary\n")
        f.write("=" * 70 + "\n")
        if valid_pairs:
            f.write(f"Dates compared:              {len(valid_pairs)}\n")
            f.write(f"Mean error (Computed-Actual): {np.mean(errors):+.3f}\n")
            f.write(f"Mean |error|:               {np.mean(abs_errors):.3f}\n")
            f.write(f"Max |error|:                {max(abs_errors):.3f}\n")
            f.write(f"Min |error|:                {min(abs_errors):.3f}\n")
            f.write(f"\n{'Date':<12} {'Computed':>10} {'Actual':>10} {'Error':>10}\n")
            f.write("  " + "-" * 44 + "\n")
            for r in results:
                if r.get("vix_actual") is not None:
                    err = r["vix_computed"] - r["vix_actual"]
                    f.write(f"  {r['date'][:10]:<12} {r['vix_computed']:>10.2f} "
                            f"{r['vix_actual']:>10.2f} {err:>+10.3f}\n")
        else:
            f.write("No CBOE actual VIX values in dataset.\n")
            f.write("CBOE tool: https://www.cboe.com/en/tradable-products/vix/vix-decomposition/\n")

        f.write("\n## Column Descriptions\n")
        f.write("- VIX_Computed: 30-day constant-maturity VIX using 2-expiry linear interpolation\n")
        f.write("- F1 (Sticky Strike):     ATM vol change from SPX spot move holding skew fixed\n")
        f.write("- F2 (Parallel Shift):    Full surface level change at same strike\n")
        f.write("- F3 (Put Skew Gradient): Put shoulder (30-delta) change net of F2\n")
        f.write("- F4 (Call Skew Gradient):Call shoulder (30-delta) change net of F2\n")
        f.write("- F5 (Downside Convexity):Put wing (10-delta) convexity change net of F2+F3\n")
        f.write("- F6 (Upside Convexity):  Call wing (10-delta) convexity change net of F2+F4\n")
        f.write("- VIX_Actual:             CBOE published VIX (or payload VIX spot where available)\n")
        f.write("\n## Methodology Notes\n")
        f.write("- F1-F6 use SINGLE-STRIKE representative approximation (30-delta put/call, 10-delta put/call)\n")
        f.write("  Full F3 requires VIX recomputed after adjusting ALL 15-45 delta put prices\n")
        f.write("  Full F5 requires VIX recomputed after adjusting ALL 1-15 delta put prices\n")
        f.write("- Delta-to-strike: iterative approach using actual IV from near-term skew at each strike\n")
        f.write("- F1 = OLD_30d_put_vol_at_NEW_spot - OLD_ATM_vol (per whitepaper P13)\n")

    print(f"\nResults saved to {output_path}")

    # ── Generate vix_computed_vs_actual.png ─────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in results]

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Panel 1: VIX Computed vs Actual
        ax = axes[0]
        vix_comp = [r["vix_computed"] for r in results]
        vix_actual = [r.get("vix_actual") for r in results]
        ax.plot(dates, vix_comp, label="VIX Computed", color="steelblue", lw=1.8)
        valid_actual = [(d, v) for d, v in zip(dates, vix_actual) if v is not None]
        if valid_actual:
            d_vals, v_vals = zip(*valid_actual)
            ax.plot(dates, [r.get("vix_actual") for r in results],
                    label="VIX Actual", color="darkorange", lw=1.2, alpha=0.8)
        ax.set_ylabel("VIX")
        ax.set_title("VIX Computed vs Actual")
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: Decomposition factors F1–F4
        ax = axes[1]
        decomp_dates = dates[1:]  # decomposition starts from index 1
        f1_vals = [decompositions[i].factor1_sticky_strike for i in range(1, len(results))]
        f2_vals = [decompositions[i].factor2_parallel_shift for i in range(1, len(results))]
        f3_vals = [decompositions[i].factor3_put_skew_grad for i in range(1, len(results))]
        f4_vals = [decompositions[i].factor4_call_skew_grad for i in range(1, len(results))]

        ax.plot(decomp_dates, f1_vals, label="F1 Sticky Strike", lw=1.4)
        ax.plot(decomp_dates, f2_vals, label="F2 Parallel Shift", lw=1.4)
        ax.plot(decomp_dates, f3_vals, label="F3 Put Skew Grad", lw=1.4)
        ax.plot(decomp_dates, f4_vals, label="F4 Call Skew Grad", lw=1.4)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("Vol Change (pts)")
        ax.set_title("Decomposition Factors F1–F4")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 3: Decomposition factors F5–F6
        ax = axes[2]
        f5_vals = [decompositions[i].factor5_downside_conv for i in range(1, len(results))]
        f6_vals = [decompositions[i].factor6_upside_conv for i in range(1, len(results))]

        ax.plot(decomp_dates, f5_vals, label="F5 Downside Conv", lw=1.4, color="purple")
        ax.plot(decomp_dates, f6_vals, label="F6 Upside Conv", lw=1.4, color="brown")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("Vol Change (pts)")
        ax.set_title("Decomposition Factors F5–F6")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle("VIX Analysis & Decomposition (2026+)", fontsize=13)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "vix_decomposition_variance_chart.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Chart generation failed (non-critical): {e}")

if __name__ == "__main__":
    main()
