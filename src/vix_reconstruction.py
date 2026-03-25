"""
src/vix_reconstruction.py  –  Step 3: VIX Reconstruction
=========================================================
Implements the official CBOE variance formula to compute a synthetic VIX
from SPX option mid-quote prices.

CBOE formula (per maturity T)
-----------------------------
  σ² = (2/T) × e^{rT} × Σᵢ (ΔKᵢ / Kᵢ²) × Q(Kᵢ)  −  (1/T) × (F/K₀ − 1)²

where:
  Q(Kᵢ) = mid-quote price of the OTM option at strike Kᵢ
             (put for Kᵢ < K₀, call for Kᵢ > K₀, average at K₀)
  F      = forward index level  (from put-call parity, Step 2)
  K₀     = first strike at or below F
  T      = time to expiration in years  (DTE / 365)
  ΔKᵢ   = half-distance to adjacent strikes (trapezoid-rule interval)
             = (K_{i+1} - K_{i-1}) / 2  for inner strikes
             = K₂ - K₁ / K_{n} - K_{n-1}  for edge strikes

30-day interpolation (per CBOE white paper)
--------------------------------------------
VIX uses two expirations that straddle 30 days (near-term T₁ < 30 ≤ T₂):

  VIX² / 10000 = [T₁σ₁²(D₂−30) + T₂σ₂²(30−D₁)] / (D₂−D₁) × (365/30)

This is equivalent to linearly interpolating total variance (T×σ²) to the
30-day horizon, then annualising.

Why use market prices (Q) rather than spline IV?
-------------------------------------------------
The CBOE formula is a model-free estimator — it uses the actual market
mid-quotes without assuming any particular IV model.  Converting IV→price
via Black-Scholes would reintroduce model dependency.  We therefore go back
to the original options DataFrame (from Step 1) and extract the raw prices.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.iv_surface import IVSurface

log = logging.getLogger(__name__)

# Target maturity for VIX (calendar days)
_TARGET_DTE = 30.0
_TARGET_T   = _TARGET_DTE / 365.0


# ============================================================================
# Helper 1 — OTM price selection
# ============================================================================

def _select_otm_prices(
    group: pd.DataFrame,
    k0: float,
) -> pd.DataFrame:
    """
    Select the out-of-the-money option mid-quote price at each available strike.

    Follows the CBOE convention (VIX white paper §1):
      • K < K₀  →  put mid-quote
      • K > K₀  →  call mid-quote
      • K = K₀  →  average of call and put mid-quotes

    Parameters
    ----------
    group : pd.DataFrame
        Options slice for one (date, expiration).
        Required columns: option_type ('C'/'P'), strike, mid.
    k0 : float
        First strike at or below the forward price F.

    Returns
    -------
    pd.DataFrame
        Columns: strike (float), Q (mid-quote price).
        Sorted by strike ascending, deduplicated.
        Only rows with Q > 0 are retained.
    """
    calls = (
        group.loc[group["option_type"] == "C", ["strike", "mid"]]
        .drop_duplicates("strike")
        .set_index("strike")
    )
    puts = (
        group.loc[group["option_type"] == "P", ["strike", "mid"]]
        .drop_duplicates("strike")
        .set_index("strike")
    )

    all_strikes = sorted(
        set(calls.index.tolist()) | set(puts.index.tolist())
    )

    records = []
    for K in all_strikes:
        if K < k0:
            # OTM put
            if K in puts.index:
                Q = float(puts.loc[K, "mid"])
                if Q > 0:
                    records.append({"strike": K, "Q": Q})

        elif K > k0:
            # OTM call
            if K in calls.index:
                Q = float(calls.loc[K, "mid"])
                if Q > 0:
                    records.append({"strike": K, "Q": Q})

        else:
            # K == K₀  →  average
            parts, n = 0.0, 0
            if K in calls.index:
                parts += float(calls.loc[K, "mid"])
                n += 1
            if K in puts.index:
                parts += float(puts.loc[K, "mid"])
                n += 1
            if n > 0 and parts > 0:
                records.append({"strike": K, "Q": parts / n})

    if not records:
        return pd.DataFrame(columns=["strike", "Q"])

    return (
        pd.DataFrame(records)
        .sort_values("strike")
        .reset_index(drop=True)
    )


# ============================================================================
# Helper 2 — ΔK (trapezoid-rule intervals)
# ============================================================================

def _compute_delta_k(strikes: np.ndarray) -> np.ndarray:
    """
    Compute the strike interval ΔKᵢ for each strike in the sorted array.

    The CBOE uses the trapezoid rule:
      • Inner strikes :  ΔKᵢ = (K_{i+1} − K_{i-1}) / 2
      • Left edge     :  ΔK₀ = K₁ − K₀
      • Right edge    :  ΔKₙ = Kₙ − K_{n-1}

    This gives the width of the "bucket" centred on each strike and is
    equivalent to the midpoint rule for unevenly spaced grids.

    Parameters
    ----------
    strikes : np.ndarray
        Strictly increasing 1-D array of strike prices.

    Returns
    -------
    np.ndarray
        ΔK values, same length as *strikes*.
    """
    n  = len(strikes)
    dk = np.empty(n, dtype=float)

    if n == 1:
        # Degenerate — treat as a single-point integral with width 1
        dk[0] = 1.0
        return dk

    if n == 2:
        dk[0] = strikes[1] - strikes[0]
        dk[1] = strikes[1] - strikes[0]
        return dk

    # Inner strikes
    dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    # Edge strikes
    dk[0]    = strikes[1]  - strikes[0]
    dk[-1]   = strikes[-1] - strikes[-2]

    return dk


# ============================================================================
# Helper 3 — CBOE variance formula for one expiration
# ============================================================================

def _compute_sigma_squared(
    group: pd.DataFrame,
    surface: IVSurface,
    r: float = config.RISK_FREE_RATE,
) -> float:
    """
    Compute the CBOE model-free variance estimate σ² for one expiry.

      σ² = (2/T) × e^{rT} × Σᵢ (ΔKᵢ / Kᵢ²) × Q(Kᵢ)
                           − (1/T) × (F/K₀ − 1)²

    Parameters
    ----------
    group : pd.DataFrame
        Options rows for one (date, expiration).
        Requires: option_type, strike, mid.
    surface : IVSurface
        The corresponding surface from Step 2 — used for F, K₀, and T.
        (We do NOT use its spline here; we only read the pre-computed
         forward price and K0.)
    r : float
        Annualised risk-free rate.

    Returns
    -------
    float  – annualised variance σ² (1/year units).
             np.nan if computation fails.

    Notes
    -----
    Including the e^{rT} factor converts the discounted option prices
    (which reflect e^{-rT} discounting to today) to their undiscounted
    "forward-measure" equivalents — matching the theoretical derivation
    of the variance swap rate.
    """
    F   = surface.forward
    k0  = surface.k0
    T   = surface.T

    if T <= 0 or not np.isfinite(F) or not np.isfinite(k0):
        log.warning("Invalid surface parameters: F=%.4f, k0=%.4f, T=%.6f", F, k0, T)
        return np.nan

    # ── Get OTM mid-quote prices ──────────────────────────────────────────────
    otm = _select_otm_prices(group, k0)

    if len(otm) < 2:
        log.warning(
            "[%s / %s] Only %d OTM price(s) — skipping variance computation.",
            surface.date.date(), surface.expiration.date(), len(otm),
        )
        return np.nan

    K = otm["strike"].values.astype(float)
    Q = otm["Q"].values.astype(float)

    # ── Trapezoid-rule intervals ───────────────────────────────────────────────
    dk = _compute_delta_k(K)

    # ── Main sum ──────────────────────────────────────────────────────────────
    #   Σᵢ (ΔKᵢ / Kᵢ²) × e^{rT} × Q(Kᵢ)
    growth   = np.exp(r * T)        # e^{rT}
    main_sum = np.sum((dk / K**2) * growth * Q)

    # ── K₀ adjustment ─────────────────────────────────────────────────────────
    #   (1/T) × (F/K₀ − 1)²
    adj = (F / k0 - 1.0) ** 2

    # ── Full formula ──────────────────────────────────────────────────────────
    sigma2 = (2.0 / T) * main_sum - (1.0 / T) * adj

    if sigma2 < 0:
        # Can happen due to data noise in very low-vol regimes.
        # Floor at zero: a negative variance is non-physical.
        log.debug(
            "[%s / %s] σ²=%.6f < 0 (clamped to 0). "
            "main_sum=%.8f, adj=%.8f",
            surface.date.date(), surface.expiration.date(),
            sigma2, main_sum, adj,
        )
        sigma2 = 0.0

    return float(sigma2)


# ============================================================================
# Helper 4 — 30-day interpolation (CBOE §2)
# ============================================================================

def _interpolate_to_30d(
    sigma2_near: float,
    sigma2_next: float,
    dte_near: float,
    dte_next: float,
    target_dte: float = _TARGET_DTE,
) -> float:
    """
    Interpolate near-term and next-term variance estimates to a
    constant 30-calendar-day maturity.

    CBOE formula (day-unit version)
    --------------------------------
    Let D₁, D₂ = DTE of near and next expirations (calendar days),
    T₁ = D₁/365, T₂ = D₂/365, T₃₀ = 30/365.

    Linearly interpolate the *total* variance (T×σ²) to D₃₀, then
    annualise by dividing by T₃₀:

      σ²₃₀ = [T₁σ₁²(D₂−30) + T₂σ₂²(30−D₁)] / (D₂−D₁) × (365/30)

    Equivalently:  VIX²/10000 = σ²₃₀

    Edge cases
    ----------
    • D₁ = D₂ (only one expiry)  →  scale that expiry's total variance
      directly: σ²₃₀ = σ₁² × T₁ / T₃₀.
    • 30 ≤ D₁ (near-term ≥ 30d)  →  use only near-term, scaled.
    • D₂ ≤ 30 (next-term ≤ 30d)  →  use only next-term, scaled.

    Parameters
    ----------
    sigma2_near, sigma2_next : float  – annualised variance per year.
    dte_near, dte_next       : float  – calendar DTE.
    target_dte               : float  – interpolation target (default 30).

    Returns
    -------
    float  – annualised 30-day variance (VIX²/10000 before √ and ×100).
    """
    T1  = dte_near / 365.0
    T2  = dte_next / 365.0
    T30 = target_dte / 365.0

    D1, D2 = dte_near, dte_next
    D30    = target_dte

    # ── Degenerate: single expiry ─────────────────────────────────────────────
    if abs(D2 - D1) < 0.5:
        # Scale total variance to target maturity
        total_var = sigma2_near * T1
        return float(total_var / T30)

    # ── Near-term already at or beyond 30d ────────────────────────────────────
    if D1 >= D30:
        log.debug(
            "Near-term DTE=%.1f ≥ 30; using near-term only (scaled).", D1
        )
        return float(sigma2_near * T1 / T30)

    # ── Next-term at or below 30d ─────────────────────────────────────────────
    if D2 <= D30:
        log.debug(
            "Next-term DTE=%.1f ≤ 30; using next-term only (scaled).", D2
        )
        return float(sigma2_next * T2 / T30)

    # ── Standard case: D1 < 30 < D2 ──────────────────────────────────────────
    # Weights sum to 1; w_near up-weights the shorter expiry when target is
    # close to it, w_next does the same for the longer expiry.
    w_near = (D2 - D30) / (D2 - D1)   # = fraction of interval above target
    w_next = (D30 - D1) / (D2 - D1)   # = fraction of interval below target

    # Interpolate total variance and annualise
    total_var_30 = T1 * sigma2_near * w_near + T2 * sigma2_next * w_next
    sigma2_30    = total_var_30 / T30

    return float(sigma2_30)


# ============================================================================
# Main per-date VIX computation
# ============================================================================

def compute_synthetic_vix_one_day(
    date: pd.Timestamp,
    options: pd.DataFrame,
    surfaces: Dict[pd.Timestamp, Dict[str, IVSurface]],
    r: float = config.RISK_FREE_RATE,
) -> Tuple[float, dict]:
    """
    Compute the synthetic VIX for a single trading date.

    Parameters
    ----------
    date : pd.Timestamp
    options : pd.DataFrame  –  full cleaned options dataset (from Step 1).
    surfaces : dict         –  {date: {'near': IVSurface, 'next': IVSurface}}
    r : float               –  risk-free rate.

    Returns
    -------
    (vix, diagnostics) : (float, dict)
        vix         – synthetic VIX level (0–100 scale, like the real VIX).
        diagnostics – dict with intermediate values for debugging / Step 4.
            keys: sigma2_near, sigma2_next, dte_near, dte_next,
                  sigma2_30, n_otm_near, n_otm_next,
                  F_near, F_next, k0_near, k0_next
    """
    diag: dict = {}

    # ── Surface lookup ────────────────────────────────────────────────────────
    if date not in surfaces:
        log.warning("[%s] No surfaces found — cannot compute VIX.", date.date())
        return np.nan, diag

    sv   = surfaces[date]
    near = sv.get("near")
    nxt  = sv.get("next")

    if near is None and nxt is None:
        log.warning("[%s] Both near and next are None.", date.date())
        return np.nan, diag

    # ── Get options rows for this date ────────────────────────────────────────
    day_opts = options.loc[options["date"] == date]

    if day_opts.empty:
        log.warning("[%s] No options rows found in dataset.", date.date())
        return np.nan, diag

    # ── Compute σ² for near-term ──────────────────────────────────────────────
    sigma2_near = np.nan
    if near is not None:
        near_group = day_opts.loc[
            day_opts["expiration"] == near.expiration
        ]
        if not near_group.empty:
            sigma2_near = _compute_sigma_squared(near_group, near, r)
            diag["sigma2_near"]  = sigma2_near
            diag["dte_near"]     = near.dte
            diag["F_near"]       = near.forward
            diag["k0_near"]      = near.k0
            diag["n_otm_near"]   = len(
                _select_otm_prices(near_group, near.k0)
            )

    # ── Compute σ² for next-term ──────────────────────────────────────────────
    sigma2_next = np.nan
    if nxt is not None:
        next_group = day_opts.loc[
            day_opts["expiration"] == nxt.expiration
        ]
        if not next_group.empty:
            sigma2_next = _compute_sigma_squared(next_group, nxt, r)
            diag["sigma2_next"] = sigma2_next
            diag["dte_next"]    = nxt.dte
            diag["F_next"]      = nxt.forward
            diag["k0_next"]     = nxt.k0
            diag["n_otm_next"]  = len(
                _select_otm_prices(next_group, nxt.k0)
            )

    # ── Handle missing σ² estimates ───────────────────────────────────────────
    if np.isnan(sigma2_near) and np.isnan(sigma2_next):
        log.warning("[%s] Both σ² estimates are NaN.", date.date())
        return np.nan, diag

    # If one is missing, fall back to the other (scaled to 30d)
    if np.isnan(sigma2_near) and nxt is not None:
        log.debug("[%s] Near-term σ² failed; using next-term only.", date.date())
        sigma2_near = sigma2_next
        dte_near    = nxt.dte
    else:
        dte_near = near.dte if near is not None else _TARGET_DTE

    if np.isnan(sigma2_next) and near is not None:
        log.debug("[%s] Next-term σ² failed; using near-term only.", date.date())
        sigma2_next = sigma2_near
        dte_next    = near.dte
    else:
        dte_next = nxt.dte if nxt is not None else _TARGET_DTE

    # ── Interpolate to 30-day ─────────────────────────────────────────────────
    sigma2_30 = _interpolate_to_30d(sigma2_near, sigma2_next, dte_near, dte_next)
    diag["sigma2_30"] = sigma2_30

    if not np.isfinite(sigma2_30) or sigma2_30 < 0:
        log.warning(
            "[%s] Invalid σ²₃₀=%.6f after interpolation.", date.date(), sigma2_30
        )
        return np.nan, diag

    # ── VIX = 100 × √σ²₃₀ ────────────────────────────────────────────────────
    vix = 100.0 * np.sqrt(sigma2_30)
    diag["vix_synthetic"] = vix

    return float(vix), diag


# ============================================================================
# Full time-series builder
# ============================================================================

def build_vix_series(
    options: pd.DataFrame,
    surfaces: Dict[pd.Timestamp, Dict[str, IVSurface]],
    r: float = config.RISK_FREE_RATE,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute synthetic VIX for every date in the surfaces dictionary.

    Parameters
    ----------
    options  : pd.DataFrame  –  cleaned options from Step 1.
    surfaces : dict          –  built in Step 2.
    r        : float         –  risk-free rate.

    Returns
    -------
    (vix_series, diagnostics_df) : (pd.Series, pd.DataFrame)

        vix_series     – pd.Series, index = date, values = synthetic VIX.
        diagnostics_df – pd.DataFrame with per-date intermediate values
                         (sigma2_near, sigma2_next, dte_near, dte_next,
                          sigma2_30, n_otm_near, n_otm_next).
    """
    log.info("=" * 60)
    log.info("STEP 3 — Computing synthetic VIX")
    log.info("=" * 60)

    vix_records  = {}
    diag_records = {}

    all_dates = sorted(surfaces.keys())
    log.info("Processing %d dates…", len(all_dates))

    for date in all_dates:
        vix, diag = compute_synthetic_vix_one_day(date, options, surfaces, r)
        vix_records[date]  = vix
        diag_records[date] = diag

    vix_series = pd.Series(vix_records, name="vix_synthetic")
    vix_series.index.name = "date"

    diag_df = pd.DataFrame(diag_records).T
    diag_df.index.name = "date"

    n_valid = vix_series.notna().sum()
    log.info(
        "Synthetic VIX computed: %d / %d dates valid. "
        "Range: %.2f – %.2f",
        n_valid, len(all_dates),
        vix_series.min(), vix_series.max(),
    )
    return vix_series, diag_df


# ============================================================================
# Validation against actual VIX
# ============================================================================

def validate_vix(
    vix_synthetic: pd.Series,
    vix_actual: pd.Series,
) -> dict:
    """
    Compare synthetic VIX against the actual CBOE VIX close.

    Validation targets (from the spec)
    ------------------------------------
    • Correlation > 0.95
    • Mean Absolute Error < 1 VIX point

    Parameters
    ----------
    vix_synthetic : pd.Series  – synthetic VIX (date index).
    vix_actual    : pd.Series  – actual VIX closes (date index).

    Returns
    -------
    dict with keys: corr, mae, rmse, bias, pct_within_1pt, passed.
    Logs a warning if either target is not met.
    """
    # Align on common dates
    common = vix_synthetic.index.intersection(vix_actual.index)
    synth  = vix_synthetic.loc[common].dropna()
    actual = vix_actual.loc[common].dropna()
    common = synth.index.intersection(actual.index)
    synth  = synth.loc[common]
    actual = actual.loc[common]

    if len(common) == 0:
        log.error("No overlapping dates between synthetic and actual VIX.")
        return {}

    err  = synth - actual
    corr = float(synth.corr(actual))
    mae  = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    bias = float(err.mean())
    pct1 = float((err.abs() <= 1.0).mean() * 100)

    results = {
        "corr":          corr,
        "mae":           mae,
        "rmse":          rmse,
        "bias":          bias,
        "pct_within_1pt": pct1,
        "n_days":        len(common),
        "passed":        (corr > 0.95) and (mae < 1.0),
    }

    # ── Log results ───────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("VIX RECONSTRUCTION VALIDATION")
    log.info("─" * 60)
    log.info("  Days compared    : %d", len(common))
    log.info("  Correlation      : %.4f  [target > 0.95]  %s",
             corr, "✓" if corr > 0.95 else "✗ BELOW TARGET")
    log.info("  MAE              : %.3f pts  [target < 1.0]  %s",
             mae, "✓" if mae < 1.0 else "✗ ABOVE TARGET")
    log.info("  RMSE             : %.3f pts", rmse)
    log.info("  Bias (synth−act) : %+.3f pts", bias)
    log.info("  Within ±1pt      : %.1f%%", pct1)

    if results["passed"]:
        log.info("  ✓ VALIDATION PASSED")
    else:
        log.warning("  ✗ VALIDATION FAILED — investigate before proceeding")

    return results


# ============================================================================
# Diagnostic print
# ============================================================================

def print_vix_comparison(
    vix_synthetic: pd.Series,
    vix_actual: pd.Series,
    n_rows: int = 20,
) -> None:
    """
    Print a side-by-side comparison table of synthetic vs actual VIX.

    Parameters
    ----------
    vix_synthetic, vix_actual : pd.Series  (date index)
    n_rows : int   –  number of rows to display.
    """
    common = vix_synthetic.index.intersection(vix_actual.index)
    df = pd.DataFrame({
        "Synthetic": vix_synthetic.loc[common],
        "Actual":    vix_actual.loc[common],
    }).dropna()
    df["Error"]   = df["Synthetic"] - df["Actual"]
    df["AbsError"] = df["Error"].abs()

    print("\n" + "=" * 55)
    print("SYNTHETIC VIX vs ACTUAL VIX")
    print("=" * 55)
    print(df.head(n_rows).to_string(float_format="{:.3f}".format))
    print("-" * 55)
    print(f"  Mean synthetic : {df['Synthetic'].mean():.3f}")
    print(f"  Mean actual    : {df['Actual'].mean():.3f}")
    print(f"  Mean error     : {df['Error'].mean():+.3f}")
    print(f"  MAE            : {df['AbsError'].mean():.3f}")
    print(f"  Max abs error  : {df['AbsError'].max():.3f}")
    print("=" * 55)
