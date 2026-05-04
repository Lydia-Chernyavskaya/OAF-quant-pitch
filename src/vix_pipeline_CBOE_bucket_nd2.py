#!/usr/bin/env python3
"""
VIX Analysis — CBOE bucket-recomputation method (single-strike scalar
applied uniformly across the bucket), N(d2) variant.

Strict reproduction of CBOE's methodology prose and worked examples from
the Aug 2025 whitepaper (pp 18, 20, 22). For each of F3–F6, every strike
on t1's chain inside the relevant delta bucket is patched with
sigma_t0(K) + excess_scalar, where

    excess_scalar = (sigma_t1(K_rep) − sigma_t0(K_rep)) − control

is computed at ONE representative strike K_rep (30Δ for F3/F4, 10Δ for
F5/F6). Both put and call mids are patched under PCP so the patch is
visible to compute_vix_variance regardless of which side of t0's K0 each
bucket strike falls on. VIX is then recomputed once per factor. F1/F2
use strike-aligned IV reads at the model-free ATM strike K_atmf (where
|cmid − pmid| is minimized via put-call parity), the same K0 anchor
used internally by the VIX² formula's forward adjustment.

This implementation matches CBOE's Yen-Carry Unwind walkthrough numbers
(p20: 1.66 vol-pts at the 30-delta put; p22: 0.66 vol-pts at the
10-delta put after cascading the F3 control).

KNOWN CAVEATS
- F1, F2 and the F3-F6 control terms are in different unit conventions
  (F1, F2 in vol-pts at K_atmf; F3-F6 in VIX-pts from bucket
  recomputation). The cascade subtraction in F5/F6 mixes them per CBOE's
  worked example (p22: 3.43 − 2.77 = 0.66). The residual sum_factors −
  ΔVIX captures this unit slack plus interaction terms.
- N(d2) delta convention (vs the unspecified delta convention in the
  whitepaper) shifts bucket strike coverage toward more-OTM levels. This
  may modestly change which strikes count as "shoulder" vs "wing" but
  doesn't affect the methodology's correctness.
- Accordion handling at the wings (F5/F6): if a t1 bucket strike is
  absent from t0's overall skew dict (zero-bid truncated at t0), we
  linearly interpolate sigma_t0(K) from t0's flanking strikes via
  _interp_iv_at_strike on the COMBINED put∪call dict. Under PCP the
  combined-dict read is consistent regardless of side. CBOE's prose
  acknowledges activation on p21 but doesn't specify the mechanic.
- F1/F2 use the model-free ATM strike K_atmf (|C−P| min via put-call
  parity) rather than round(spot). This is the same anchor the VIX²
  formula uses internally for K0. CBOE's worked example on p18 uses
  round(spot) directly; on SPX the difference is at most 1-2 strikes and
  well below noise floor.
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

def compute_forward(df: pd.DataFrame, spot: float, rfr: float, T: float) -> tuple[float, float]:
    """Compute model-free forward price via put-call parity at ATM forward strike.

    Finds K_atmf = strike minimizing |cmid - pmid| (model-free ATM),
    then F = K_atmf + exp(rT) * (cmid - pmid).

    Returns (F, K_atmf). K_atmf is the K0 anchor that the VIX² formula uses
    internally for the forward adjustment (see compute_vix_variance below);
    we expose it here so F1/F2 reads can use the same anchor.
    """
    if "cmid" not in df.columns or "pmid" not in df.columns:
        return (spot, spot)
    pc_diff = (df["cmid"] - df["pmid"]).abs()
    if pc_diff.empty:
        return (spot, spot)
    idx = pc_diff.idxmin()
    K_atmf = float(df.loc[idx, "strike"])
    cmid = float(df.loc[idx, "cmid"])
    pmid = float(df.loc[idx, "pmid"])
    if math.isnan(cmid): cmid = 0.0
    if math.isnan(pmid): pmid = 0.0
    return (K_atmf + math.exp(rfr * T) * (cmid - pmid), K_atmf)

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
    N(d2) variant — uses risk-neutral probability of ITM rather than
    BS delta. Bucket thresholds retain their numerical values; strike
    coverage shifts vs the N(d1) version.

    Return strikes whose N(d2)-based "delta" falls in the
    [delta_lo, delta_hi] interval (boundary inclusivity controlled by
    lower_excl / upper_excl).

    For each strike K in chain_df:
      1. Use bs_iv to get sigma from the option price (pmid for puts, cmid for calls)
      2. Compute d1 = (ln(spot/K) + 0.5*sigma²*T) / (sigma*sqrt(T))
                 d2 = d1 - sigma*sqrt(T)
      3. delta = N(d2) - 1 for puts, N(d2) for calls
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
        d2 = d1 - sigma * sqrt_T
        if side == 'put':
            delta_val = norm.cdf(d2) - 1.0
        else:
            delta_val = norm.cdf(d2)

        lo_ok = (delta_val > delta_lo) if lower_excl else (delta_val >= delta_lo)
        hi_ok = (delta_val < delta_hi) if upper_excl else (delta_val <= delta_hi)
        if lo_ok and hi_ok:
            strikes_in.add(K)

    return strikes_in


def get_strike_at_delta(chain_df: pd.DataFrame, spot: float, T: float,
                        rfr: float, target_delta: float, side: str) -> float | None:
    """Return the strike with N(d2)-delta closest to target_delta, or None
    if no valid strike found. side='put' (target_delta < 0) or 'call'
    (target_delta > 0). N(d2) variant matching the rest of this pipeline.

    Same per-side bid validity, IV solver, and N(d2) computation as
    get_strikes_in_delta_bucket — just collapsed to a single nearest match.
    """
    df = chain_df.sort_values("strike").reset_index(drop=True)
    sqrt_T = math.sqrt(T)
    best_K = None
    best_diff = float("inf")

    for _, row in df.iterrows():
        K = float(row["strike"])
        if K <= 0:
            continue

        if _is_zero_bid(row.to_dict(), side):
            continue
        price = float(row["pmid"]) if side == 'put' else float(row["cmid"])
        if math.isnan(price) or price <= 0:
            continue

        sigma_iv = bs_iv(price, spot, K, T, rfr, is_call=(side == 'call'))
        if sigma_iv <= 0:
            sigma_iv = 20.0  # fallback (matches get_strikes_in_delta_bucket)

        sigma = sigma_iv / 100.0
        d1 = (math.log(spot / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        if side == 'put':
            delta_val = norm.cdf(d2) - 1.0
        else:
            delta_val = norm.cdf(d2)

        diff = abs(delta_val - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_K = K

    return best_K


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
    # F1, F2 here are forward PRICES (not the F1/F2 decomposition factors —
    # naming kept consistent with the rest of the file). K_atmf1/K_atmf2 are
    # the model-free ATM strikes (|cmid−pmid| min) and serve as the K0 anchor
    # both compute_vix_variance and run_decomposition's F1/F2 reads use.
    F1, K_atmf1 = compute_forward(df1, K_atm1, rfr, T1)
    F2, K_atmf2 = compute_forward(df2, K_atm2, rfr, T2)

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
        "K_atmf_near": K_atmf1,
        "K_atmf_far":  K_atmf2,
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

    Iterates the INTERSECTION of near and far strikes (exact match — no
    nearest-strike fallback). The previous union+nearest-strike convention
    polluted deep-OTM IVs whenever a strike existed in only one chain (the
    other chain's nearest-strike IV is for a different K and corrupts the
    smile, e.g. a K=700 deep-OTM put borrowing IV from a K=1700 row, which
    rendered as a constant ~96% plateau).

    Per-strike filters:
      - Per-side zero-bid (Cboe spec)
      - Mid <= 0 or NaN
      - Mid < 0.05 (bid-floor noise: SPX EOD quotes round to 0.05 and
        deep-OTM mids at the floor carry no σ information; bs_iv is
        ambiguous in that regime)
      - bs_iv solver failure
      - Negative interpolated variance

    For surviving strikes:
      1. Compute IV from option price at that strike for both expiries
      2. Interpolate variance to 30-day: Var30(K) = w1*Var_near + w2*Var_far
      3. Convert to vol: σ30(K) = √(Var30(K) × 365/30) × 100

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

    # Index by strike for O(1) lookup
    near_by_K = {float(K): row for K, row in zip(df_near["strike"], df_near.to_dict("records"))}
    far_by_K  = {float(K): row for K, row in zip(df_far["strike"],  df_far.to_dict("records"))}

    # Intersection: strikes listed in BOTH chains. Exact strike match
    # (no nearest-strike fallback) — see docstring.
    common_strikes = sorted(set(near_by_K.keys()) & set(far_by_K.keys()))

    put_skew_30d = {}
    call_skew_30d = {}

    MIN_MID = 0.05  # bid-floor threshold; mids <= this are noise

    for K in common_strikes:
        if K <= 0:
            continue

        side_str = 'put' if K < K_atm_near else 'call'
        is_put_near = (side_str == 'put')

        near_row = near_by_K[K]
        far_row  = far_by_K[K]

        # Per-side bid validity on both expiries
        if _is_zero_bid(near_row, side_str) or _is_zero_bid(far_row, side_str):
            continue

        price_near = float(near_row["pmid"] if is_put_near else near_row["cmid"])
        price_far  = float(far_row["pmid"]  if is_put_near else far_row["cmid"])
        if math.isnan(price_near) or math.isnan(price_far):
            continue
        if price_near <= MIN_MID or price_far <= MIN_MID:
            continue

        iv_near = bs_iv(price_near, F_near, K, T_near, rfr,
                        is_call=not is_put_near)
        if iv_near <= 0:
            continue
        iv_far = bs_iv(price_far, F_far, K, T_far, rfr,
                       is_call=not is_put_near)
        if iv_far <= 0:
            continue

        var_near = (iv_near / 100.0) ** 2 * T_near
        var_far  = (iv_far  / 100.0) ** 2 * T_far
        var30 = w1 * var_near + w2 * var_far
        if var30 <= 0:
            continue

        vol30 = math.sqrt(var30 / T30) * 100.0

        if K < K_atm_near:
            put_skew_30d[K] = vol30
        else:
            call_skew_30d[K] = vol30

    return put_skew_30d, call_skew_30d

def get_vol_at_strike(skew_dict: dict[float, float], target_strike: float) -> float:
    """
    Nearest-strike lookup. Matches CBOE whitepaper page 18 which uses 5,185
    as 'ATM strike for Aug 5' even though spot was 5,186 — i.e. the IV is
    read at the closest LISTED strike, no interpolation.

    Returns 0.0 if skew_dict is empty.
    """
    if not skew_dict:
        return 0.0
    nearest_K = min(skew_dict.keys(), key=lambda K: abs(K - target_strike))
    return float(skew_dict[nearest_K])

def _signed_delta(K: float, S: float, vol30: float, T30: float, side: str) -> float:
    """N(d2) variant — uses risk-neutral probability of ITM rather than
    BS delta. Bucket thresholds retain their numerical values; strike
    coverage shifts vs the N(d1) version.

    Signed metric using N(d2): put = N(d2) - 1, call = N(d2).
    """
    sigma = vol30 / 100.0
    sqrt_T30 = math.sqrt(T30)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T30) / (sigma * sqrt_T30)
    d2 = d1 - sigma * sqrt_T30
    if side == 'put':
        return norm.cdf(d2) - 1.0
    else:
        return norm.cdf(d2)


def _interp_iv_at_strike(skew_dict: dict[float, float], K: float) -> float | None:
    """Linear interpolation of vol at K from the same-side skew dict.
    Returns None if K is outside the dict's span (no flanking strikes on
    one side). Used for the t0 σ at K_rep when K_rep isn't listed in t0's
    same-side dict (accordion case at the wing strike, p21)."""
    if not skew_dict:
        return None
    if K in skew_dict:
        return float(skew_dict[K])
    sorted_K = sorted(skew_dict.keys())
    if K < sorted_K[0] or K > sorted_K[-1]:
        return None
    # Find flanking strikes
    lo = max(k for k in sorted_K if k <= K)
    hi = min(k for k in sorted_K if k >= K)
    if lo == hi:
        return float(skew_dict[lo])
    w = (K - lo) / (hi - lo)
    return float(skew_dict[lo]) * (1.0 - w) + float(skew_dict[hi]) * w


def compute_factor_bucket(prev: dict, curr: dict,
                          delta_lo: float, delta_hi: float,
                          target_delta: float,
                          side: str,
                          control_subtract: float,
                          lower_excl: bool = False,
                          upper_excl: bool = False) -> float:
    """
    CBOE-strict bucket recomputation matching whitepaper pp 20, 22 exactly.

    The "excess vol-pts" is a SINGLE SCALAR computed at one representative
    strike K_rep on t1's chain (30Δ for shoulders, 10Δ for wings):

        excess_scalar = (σ_t1(K_rep) − σ_t0(K_rep)) − control_subtract

    This scalar is then applied UNIFORMLY to every strike in the bucket
    (t1's bucket strikes, [15Δ, 45Δ] or [1Δ, 15Δ]):

        σ_eff(K) = σ_t0(K) + excess_scalar          for all K in bucket

    Both pmid and cmid are repriced under PCP at σ_eff(K) so the patch
    registers in compute_vix_variance regardless of which side of t0's K0
    each bucket strike falls on. VIX is then recomputed once.

    σ_t0 lookups (for K_rep and for each bucket K) use the COMBINED
    put∪call skew dict — under PCP, IV at strike K is the same whether
    read from put or call surface, and using the combined dict avoids
    dropping bucket strikes that crossed K_atm between t0 and t1.
    Accordion fallback: _interp_iv_at_strike on the combined dict if a
    bucket strike isn't listed in t0's skew at all.

    Arguments:
      delta_lo, delta_hi: signed bucket bounds for selecting strikes on
        t1's chain. Puts: -0.45..-0.15 (F3) / -0.15..-0.01 (F5). Calls:
        +0.15..+0.45 (F4) / +0.01..+0.15 (F6).
      target_delta: signed scalar for K_rep selection. F3: -0.30, F4:
        +0.30, F5: -0.10, F6: +0.10.
      side: 'put' for F3/F5, 'call' for F4/F6.
      control_subtract: F2 for F3/F4. F2+F3 for F5. F2+F4 for F6.
      lower_excl, upper_excl: bucket boundary inclusivity for disjoint
        buckets across F3/F5 and F4/F6.

    Mixed units (vol-pt control_subtract, VIX-pt return) are CBOE's
    convention; reproduces the 3.43 − 2.77 = 0.66 step in the whitepaper
    p22 worked example.
    """
    F_old     = prev["F"]
    F_far_old = prev["F2"]
    T1        = prev["T1"]
    T2        = prev["T2"]
    r         = prev["rfr"]

    # Combined put∪call skew dicts (PCP: σ at K is the same on both sides;
    # using the combined dict eliminates same-side restrictions when bucket
    # strikes cross K_atm between t0 and t1).
    combined_old = {**prev.get("put_skew_30d", {}), **prev.get("call_skew_30d", {})}
    combined_new = {**curr.get("put_skew_30d", {}), **curr.get("call_skew_30d", {})}
    if not combined_old or not combined_new:
        return 0.0

    # ── Step 1: representative strike on t1's near chain ──────────────────
    K_rep = get_strike_at_delta(curr["chain1_df"], curr["spot"], curr["T1"],
                                curr["rfr"], target_delta, side)
    if K_rep is None:
        return 0.0

    # ── Step 2: scalar excess at K_rep only ───────────────────────────────
    sigma_t1_rep = combined_new.get(float(K_rep))
    if sigma_t1_rep is None:
        sigma_t1_rep = _interp_iv_at_strike(combined_new, float(K_rep))
    if sigma_t1_rep is None:
        return 0.0
    sigma_t0_rep = combined_old.get(float(K_rep))
    if sigma_t0_rep is None:
        sigma_t0_rep = _interp_iv_at_strike(combined_old, float(K_rep))
    if sigma_t0_rep is None:
        return 0.0

    excess_scalar = (float(sigma_t1_rep) - float(sigma_t0_rep)) - control_subtract

    # ── Step 3: identify the bucket on t1 ─────────────────────────────────
    bucket = get_strikes_in_delta_bucket(
        curr["chain1_df"], curr["spot"], curr["T1"], curr["rfr"],
        delta_lo=delta_lo, delta_hi=delta_hi, side=side,
        lower_excl=lower_excl, upper_excl=upper_excl,
    )
    if not bucket:
        return 0.0

    # ── Step 4: build (put_price, call_price) patches for each bucket K ──
    # Both sides repriced under PCP at σ_eff(K) so compute_vix_variance
    # picks up the patch regardless of which side of t0's K0 K falls on.
    patches_near: dict[float, tuple[float, float]] = {}
    patches_far:  dict[float, tuple[float, float]] = {}
    for K in sorted(bucket):
        K_f = float(K)
        sigma_t0_K = combined_old.get(K_f)
        if sigma_t0_K is None:
            sigma_t0_K = _interp_iv_at_strike(combined_old, K_f)
        if sigma_t0_K is None:
            continue
        sigma_eff_K = float(sigma_t0_K) + excess_scalar
        if sigma_eff_K <= 0 or not np.isfinite(sigma_eff_K):
            continue
        sigma_dec = sigma_eff_K / 100.0
        new_put_near  = _bs_put( F_old,     K_f, T1, sigma_dec, r)
        new_call_near = _bs_call(F_old,     K_f, T1, sigma_dec, r)
        new_put_far   = _bs_put( F_far_old, K_f, T2, sigma_dec, r)
        new_call_far  = _bs_call(F_far_old, K_f, T2, sigma_dec, r)
        if (np.isfinite(new_put_near) and new_put_near > 0
                and np.isfinite(new_call_near) and new_call_near > 0):
            patches_near[K_f] = (float(new_put_near), float(new_call_near))
        if (np.isfinite(new_put_far) and new_put_far > 0
                and np.isfinite(new_call_far) and new_call_far > 0):
            patches_far[K_f]  = (float(new_put_far),  float(new_call_far))

    if not patches_near and not patches_far:
        return 0.0

    # ── Step 5: apply patches to near and far chains (BOTH pmid + cmid) ───
    def _patched(df_orig: pd.DataFrame,
                 patches: dict[float, tuple[float, float]]) -> pd.DataFrame:
        df = df_orig.copy()
        existing_strikes = set(df["strike"].astype(float))
        synth_rows = []
        for K, (put_price, call_price) in patches.items():
            if K in existing_strikes:
                df.loc[df["strike"] == K, "pmid"] = put_price
                df.loc[df["strike"] == K, "cmid"] = call_price
            else:
                # Synthetic accordion row. Positive-sentinel bid/ask on
                # BOTH sides so per-side bid filter admits the row whether
                # the strike lands above or below K0_old.
                synth = {
                    "strike": K,
                    "pmid": put_price,
                    "cmid": call_price,
                    "pbid": max(put_price * 0.95, 0.05),
                    "pask": put_price * 1.05,
                    "cbid": max(call_price * 0.95, 0.05),
                    "cask": call_price * 1.05,
                }
                synth_rows.append(synth)
        if synth_rows:
            df = pd.concat([df, pd.DataFrame(synth_rows)], ignore_index=True)
            df = df.sort_values("strike").reset_index(drop=True)
        return df

    df_near_patched = _patched(prev["chain1_df"], patches_near)
    df_far_patched  = _patched(prev["chain2_df"], patches_far)

    # ── Step 6: recompute VIX from patched chains ─────────────────────────
    var_near, _, _ = compute_vix_variance(df_near_patched, F_old,     r, T1)
    var_far,  _, _ = compute_vix_variance(df_far_patched,  F_far_old, r, T2)
    if var_near <= 0 or var_far <= 0 or T2 == T1:
        return 0.0
    T30 = 30.0 / 365.0
    var_30 = (T1 * var_near * (T2 - T30) + T2 * var_far * (T30 - T1)) / (T30 * (T2 - T1))
    if var_30 < 0:
        return 0.0
    vix_patched = 100.0 * math.sqrt(var_30)
    return vix_patched - prev["vix_computed"]


def run_decomposition(prev: dict, curr: dict) -> VIXDecomposition | None:
    """
    6-factor VIX decomposition — CBOE bucket-recomputation method
    (Aug 2025 whitepaper, methodology prose on pp 20 and 22).

    F1 / F2 (whitepaper p13–14) are evaluated at the model-free ATM strike
    K_atmf (where |C−P| is minimised via put-call parity), matching the K0
    anchor used by the VIX² formula itself for the (F/K0 − 1)² adjustment.
    K_atmf shifts day to day with spot; the F1 read picks up that shift.
        F1 = σ_old(K_atmf_new) − σ_old(K_atmf_old)        sticky strike
        F2 = σ_new(K_atmf_new) − σ_old(K_atmf_new)        parallel shift
    Identity: F1 + F2 = σ_new(K_atmf_new) − σ_old(K_atmf_old) = ΔATM at K_atmf.

    F3–F6 use the bucket-recomputation method with single-strike scalar
    lift applied uniformly across the bucket — see whitepaper pp 20, 22.
    excess_scalar = (σ_t1(K_rep) − σ_t0(K_rep)) − control, computed at one
    representative strike K_rep (30Δ for F3/F4, 10Δ for F5/F6). The same
    excess_scalar is then applied as σ_t0(K) + excess_scalar to every K
    in the bucket. Bucket strikes from t1's chain; both pmid and cmid
    patched under PCP so the patch registers regardless of which side of
    K0_old each strike falls on.
        F3 put shoulder   bucket [-0.45, -0.15)   K_rep at -0.30   control = F2
        F4 call shoulder  bucket ( 0.15,  0.45]   K_rep at +0.30   control = F2
        F5 put wing       bucket [-0.15, -0.01)   K_rep at -0.10   control = F2 + F3
        F6 call wing      bucket ( 0.01,  0.15]   K_rep at +0.10   control = F2 + F4

    The cascade subtraction is intentionally asymmetric: F5 (put wing)
    subtracts F3 (put shoulder), NOT F4. F6 (call wing) subtracts F4
    (call shoulder). Same-side dependency only.

    Sum F1..F6 ≈ ΔVIX. Residual = unit mismatch between vol-pt (F1, F2)
    and VIX-pt (F3–F6) factors, plus interaction terms.

    Unit note: F2 is a vol-pt amount at a single strike; F3, F4 are
    VIX-pt impacts. CBOE's walkthrough subtracts them in mixed units
    (treating F3 as a vol-pt scalar at the wing — page 22:
    3.43 − 2.77 = 0.66). This is a known approximation that holds well
    when VIX ≈ ATM IV (calm days) and loosens on big-move days; the
    residual is interpretable as the unit mismatch.

    Bucket bounds (±0.15, ±0.45) match the whitepaper's prose. N(d2) is
    used as the delta convention.
    """
    put_old  = prev.get("put_skew_30d", {})
    put_new  = curr.get("put_skew_30d", {})
    call_old = prev.get("call_skew_30d", {})
    call_new = curr.get("call_skew_30d", {})

    if not put_old or not put_new or not call_old or not call_new:
        return None

    K_atmf_old = float(prev["K_atmf_near"])
    K_atmf_new = float(curr["K_atmf_near"])

    # Combined-side dicts for F1/F2 lookups (so K_atmf can land on either
    # the put or call side without edge-clamping artefacts).
    skew_old_full = {**put_old, **call_old}
    skew_new_full = {**put_new, **call_new}

    vol_old_at_K_atmf_new = get_vol_at_strike(skew_old_full, K_atmf_new)
    vol_old_at_K_atmf_old = get_vol_at_strike(skew_old_full, K_atmf_old)
    vol_new_at_K_atmf_new = get_vol_at_strike(skew_new_full, K_atmf_new)

    # F1 sticky strike: σ_old(K_atmf_new) − σ_old(K_atmf_old)  (whitepaper p13)
    F1 = vol_old_at_K_atmf_new - vol_old_at_K_atmf_old

    # F2 parallel shift: σ_new(K_atmf_new) − σ_old(K_atmf_new)  (whitepaper p14)
    F2 = vol_new_at_K_atmf_new - vol_old_at_K_atmf_new

    # Signed delta thresholds: get_strikes_in_delta_bucket uses
    # delta = N(d2) − 1 for puts (range −1..0) and N(d2) for calls (0..1).
    # target_delta selects K_rep (single representative strike) per CBOE
    # whitepaper pp 20, 22 — 30Δ for shoulders, 10Δ for wings.
    F3 = compute_factor_bucket(prev, curr,
                               delta_lo=-0.45, delta_hi=-0.15,
                               target_delta=-0.30, side="put",
                               control_subtract=F2,
                               lower_excl=False, upper_excl=True)   # [-.45, -.15)
    F4 = compute_factor_bucket(prev, curr,
                               delta_lo=+0.15, delta_hi=+0.45,
                               target_delta=+0.30, side="call",
                               control_subtract=F2,
                               lower_excl=True,  upper_excl=False)  # (.15, .45]
    F5 = compute_factor_bucket(prev, curr,
                               delta_lo=-0.15, delta_hi=-0.01,
                               target_delta=-0.10, side="put",
                               control_subtract=F2 + F3,
                               lower_excl=False, upper_excl=True)   # [-.15, -.01)
    F6 = compute_factor_bucket(prev, curr,
                               delta_lo=+0.01, delta_hi=+0.15,
                               target_delta=+0.10, side="call",
                               control_subtract=F2 + F4,
                               lower_excl=True,  upper_excl=False)  # (.01, .15]

    # ── Ground truth ────────────────────────────────────────────────────────
    VIX_old_actual = prev.get("vix_actual") or prev.get("vix_computed", 0.0)
    VIX_new_actual = curr.get("vix_actual") or curr.get("vix_computed", 0.0)
    total = VIX_new_actual - VIX_old_actual

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


def fetch_snapshots_2026(data_dir: str | None = None, year: int | None = 2023):
    """
    Adapt OptionsDX local EOD data into the Supabase-style snapshot list
    that main() / compute_vix_for_snapshot() consume.

    rfr convention: spx_local_loader.fetch_rfr returns DECIMAL (^IRX/100). The
    snapshot payload field "rfr" must be PERCENT, because main() does
    `rfr = float(payload["rfr"]) / 100.0`. We therefore multiply by 100 here.

    `year`: if not None, restrict to QUOTE_DATE values whose calendar year
    matches. Default 2023 preserves the historical pipeline behaviour
    (pre-recursive-glob, only 2023 data was on disk). Pass None to load
    every year present under data_dir.

    The imports of load_spx_options / fetch_rfr / fetch_vix_actual are
    function-local because spx_local_loader.py imports from this module at
    module top — a top-level import here would create a circular-import
    deadlock.
    """
    from spx_local_loader import load_spx_options, fetch_rfr, fetch_vix_actual

    if data_dir is None:
        data_dir = os.path.expanduser("~/data/spx_eod")

    chain = load_spx_options(data_dir=data_dir)
    if year is not None:
        chain = chain[chain["QUOTE_DATE"].dt.year == year].reset_index(drop=True)
        if chain.empty:
            raise ValueError(
                f"No SPX EOD rows found for year={year} under {data_dir}."
            )
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
        help="(deprecated; no effect in the CBOE bucket-recomputation pipeline). "
             "Kept for CLI compatibility with the hybrid pipeline.",
    )
    parser.add_argument(
        "--year", type=int, default=2023,
        help="Calendar year to filter SPX EOD data to (default 2023). "
             "Pass 0 to load every year present under the data dir.",
    )
    args, _ = parser.parse_known_args()
    global HYBRID_SMOOTHING_SIGMA
    HYBRID_SMOOTHING_SIGMA = float(args.hybrid_smoothing)
    year_filter = args.year if args.year and args.year > 0 else None
    print(f"Hybrid smoothing sigma: {HYBRID_SMOOTHING_SIGMA}")
    print(f"Year filter:            {year_filter if year_filter else 'ALL'}\n")

    print("Loading SPX EOD snapshots from local OptionsDX archive...")
    snapshots = fetch_snapshots_2026(year=year_filter)
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
        print(f"CBOE BUCKET-RECOMPUTATION SUMMARY (N(d2))")
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
    hybrid_csv_path = os.path.join(output_dir_hybrid, "vix_decomposition_CBOE_bucket_diagnostic_nd2.csv")
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
    decomp_csv_path = os.path.join(output_dir, "vix_decomposition_CBOE_bucket_nd2.csv")
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

    # ── F1 sign + zero-count diagnostic ──────────────────────────────────
    try:
        diag_df = pd.read_csv(hybrid_csv_path).dropna(subset=["F1"]).copy()
        diag_df["dSPX_pct"] = diag_df["SPX"].pct_change()
        diag_df = diag_df.dropna(subset=["dSPX_pct"]).copy()
        n_zero = int((diag_df["F1"].abs() < 1e-12).sum())
        spot_up = diag_df[diag_df["dSPX_pct"] > 0]
        spot_dn = diag_df[diag_df["dSPX_pct"] < 0]
        corr_f1 = float(diag_df["F1"].corr(diag_df["dSPX_pct"]))
        print("\n" + "=" * 70)
        print("F1 SIGN DIAGNOSTIC")
        print("=" * 70)
        print(f"  F1 == 0 rows:        {n_zero}/{len(diag_df)}")
        print(f"  spot-up days F1<0:   {(spot_up['F1']<0).mean()*100:.1f}% "
              f"({(spot_up['F1']<0).sum()}/{len(spot_up)})")
        print(f"  spot-dn days F1>0:   {(spot_dn['F1']>0).mean()*100:.1f}% "
              f"({(spot_dn['F1']>0).sum()}/{len(spot_dn)})")
        print(f"  corr(F1, dSPX_pct):  {corr_f1:+.4f}")
    except Exception as e:
        print(f"\nF1 sign diagnostic skipped: {e}")

    # ── N(d2) vs N(d1) comparison ────────────────────────────────────────
    try:
        nd1_path = os.path.join(output_dir, "vix_decomposition_hybrid.csv")
        nd2_path = os.path.join(output_dir, "vix_decomposition_hybrid_nd2.csv")
        nd1 = pd.read_csv(nd1_path)
        nd2 = pd.read_csv(nd2_path)
        merged = nd1.merge(nd2, on="date", suffixes=("_nd1", "_nd2"))
        print("\n" + "=" * 70)
        print("N(d2) vs N(d1) factor diff (per-date)")
        print("=" * 70)
        for f in ["F3", "F4", "F5", "F6"]:
            diff = merged[f + "_nd2"] - merged[f + "_nd1"]
            print(f"  {f}: mean diff {diff.mean():+.4f}, "
                  f"std {diff.std():.4f}, "
                  f"max |diff| {diff.abs().max():.4f}")
    except Exception as e:
        print(f"\nN(d2) vs N(d1) comparison skipped: {e}")

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
    output_path = os.path.join(os.path.dirname(__file__), "vix_results_CBOE_bucket_nd2.txt")
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
        f.write("- F3 (Put Skew Gradient): excess vol-pts at the 30-delta put on t1 applied uniformly\n")
        f.write("                          across all puts in the [15Δ, 45Δ] bucket; VIX recomputed.\n")
        f.write("- F4 (Call Skew Gradient):Same on calls (30Δ K_rep, [15Δ, 45Δ] bucket).\n")
        f.write("- F5 (Downside Convexity):Same with 10-delta put K_rep and bucket [1Δ, 15Δ].\n")
        f.write("- F6 (Upside Convexity):  Same with 10-delta call K_rep and bucket [1Δ, 15Δ].\n")
        f.write("- VIX_Actual:             CBOE published VIX (or payload VIX spot where available)\n")
        f.write("\n## Methodology Notes\n")
        f.write("- F3-F6 use SINGLE-STRIKE SCALAR LIFT applied uniformly across the bucket\n")
        f.write("  (CBOE whitepaper Aug 2025, worked examples pp 20, 22): excess_scalar =\n")
        f.write("  (sigma_t1(K_rep) - sigma_t0(K_rep)) - control, then sigma_eff(K) =\n")
        f.write("  sigma_t0(K) + excess_scalar for every K in the bucket. Both pmid and cmid\n")
        f.write("  are repriced under PCP. VIX is recomputed once per factor.\n")
        f.write("- F1/F2 read at the model-free ATM strike K_atmf (|cmid-pmid| min, put-call parity).\n")
        f.write("  Same anchor used internally by the VIX^2 forward-adjustment term.\n")
        f.write("- Delta convention: N(d2) (risk-neutral prob ITM), not N(d1) BS delta.\n")
        f.write("- Accordion handling at wings: t0 sigma at a t1-bucket strike that's missing\n")
        f.write("  from t0's overall (combined put+call) skew dict is linearly interpolated\n")
        f.write("  from flanking strikes via _interp_iv_at_strike.\n")
        f.write("- Cascade subtraction (F5 subtracts F3, F6 subtracts F4) mixes vol-pt and\n")
        f.write("  VIX-pt units per CBOE's worked example (p22: 3.43 - 2.77 = 0.66). The\n")
        f.write("  residual sum_factors - ΔVIX captures this unit slack plus interaction terms.\n")

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

        plot_path = os.path.join(output_dir, "vix_decomposition_CBOE_bucket_nd2_chart.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Chart generation failed (non-critical): {e}")

if __name__ == "__main__":
    main()
