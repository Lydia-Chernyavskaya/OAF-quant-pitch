"""
src/vix_decomposition.py  –  Step 4: VIX Decomposition
========================================================
Decomposes daily changes in the synthetic VIX into seven structural
contributions derived from the implied-volatility surface:

  ΔVIX  =  sticky  +  parallel  +  put_skew  +  call_skew
         +  put_convexity  +  call_convexity  +  higher_order

The chain closes exactly — no approximation error.

Mathematical framework
----------------------
Let m_i = (K_i − K₀) / K₀ be normalised moneyness.  On each day t we
have a t-day strike grid K_t and observed IVs IV_t(K_i).

Sticky-strike baseline
  The t-1 spline is evaluated at t-day strikes to obtain IV_ss(K_i):
  the level the surface would have today under the sticky-strike convention.

IV change decomposition (CBOE-weighted OLS)
  ΔIV_i = IV_t(K_i) − IV_ss(K_i)  is projected onto a 5-parameter
  piecewise polynomial with shared intercept (put/call sides are disjoint):

      ΔIV ≈ α                            (uniform level shift — parallel)
           + β_put  · m   (m ≤ 0)       (put-side slope change)
           + β_call · m   (m > 0)       (call-side slope change)
           + γ_put  · m²  (m ≤ 0)       (put-side curvature change)
           + γ_call · m²  (m > 0)       (call-side curvature change)

  OLS weights w_i = ΔK_i / K_i² match the CBOE VIX integrand,
  so the fit minimises VIX-relevant IV error rather than raw IV error.

Sequential bump chain
  Five intermediate IV arrays are built by applying the components one at a
  time.  Because put (m ≤ 0) and call (m > 0) sides act on disjoint strikes,
  contributions are order-independent (additively separable):

    IV₁[i] = IV_ss[i] + α                         [+parallel, all strikes]
    IV₂[i] = IV₁[i]  + β_put  · m_i  (m ≤ 0)    [+put skew]
    IV₃[i] = IV₂[i]  + β_call · m_i  (m > 0)    [+call skew]
    IV₄[i] = IV₃[i]  + γ_put  · m_i² (m ≤ 0)    [+put convexity]
    IV₅[i] = IV₄[i]  + γ_call · m_i² (m > 0)    [+call convexity]
    IV₆[i] = IV_t[i]                              [+higher-order residual]

  Full CBOE VIX formula (with 30-day interpolation) is evaluated at each step:
    VIX₀ (t-1 own frame) → VIX_SS → VIX₁ → VIX₂ → VIX₃ → VIX₄ → VIX₅ → VIX_BS_T

  Contributions telescope exactly:
    sticky_contribution        = VIX_SS  − VIX₀
    parallel_contribution      = VIX₁   − VIX_SS
    put_skew_contribution      = VIX₂   − VIX₁
    call_skew_contribution     = VIX₃   − VIX₂
    put_convexity_contribution = VIX₄   − VIX₃
    call_convexity_contribution= VIX₅   − VIX₄
    higher_order_contribution  = VIX_BS_T − VIX₅  (non-quadratic IV change)
    model_total                = VIX₅   − VIX₀    (quadratic model)
    residual                   = higher_order      (= ΔVIX − model_total)

  skew_contribution       = put_skew + call_skew        (combined, for signals)
  convexity_contribution  = put_convexity + call_convexity

Why Black-Scholes for the bump reconstruction?
----------------------------------------------
The CBOE formula operates on market mid-quote PRICES, not IVs.  When we
build the intermediate IV arrays, we convert IVs to OTM prices via Black-
Scholes.  The slight model-dependency is acceptable here because we care
about the CHANGE in VIX, not its absolute level.

TV quadratic fit (backward compat)
-----------------------------------
A local quadratic IV(m) = a + b·m + c·m² is also fit to both today's and
yesterday's near-term smiles.  The resulting a_t, b_t, c_t, delta_a etc.
are included in the output for use by signals.py (PSMR and SSD signals).
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.iv_surface import IVSurface

log = logging.getLogger(__name__)


# ============================================================================
# Black-Scholes option pricing  (used to convert bumped IVs → prices)
# ============================================================================

def _bs_call(F: float, K: float, sigma: float, T: float, r: float) -> float:
    """
    Black-Scholes European call price using the forward-price form.

      C = e^{−rT} [ F·N(d₁) − K·N(d₂) ]

    where  d₁ = [ln(F/K) + ½σ²T] / (σ√T),  d₂ = d₁ − σ√T

    Parameters
    ----------
    F     : forward price
    K     : strike
    sigma : annualised implied vol (decimal)
    T     : time to expiry in years
    r     : risk-free rate (continuous)

    Returns
    -------
    float  –  call price in the same units as F and K.
    """
    if sigma <= 0 or T <= 0:
        return float(max(F - K, 0.0) * np.exp(-r * T))
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)))


def _bs_put(F: float, K: float, sigma: float, T: float, r: float) -> float:
    """
    Black-Scholes European put price using the forward-price form.

      P = e^{−rT} [ K·N(−d₂) − F·N(−d₁) ]
    """
    if sigma <= 0 or T <= 0:
        return float(max(K - F, 0.0) * np.exp(-r * T))
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def _otm_bs_price(
    F: float, K: float, K0: float, sigma: float, T: float, r: float
) -> float:
    """
    Return the out-of-the-money Black-Scholes price at strike K.

    Convention (mirrors Step 2/3 OTM selection):
      K ≤ K₀  →  put price
      K > K₀  →  call price
    """
    if K <= K0:
        return _bs_put(F, K, sigma, T, r)
    else:
        return _bs_call(F, K, sigma, T, r)


# ============================================================================
# ΔK  (trapezoid-rule strike intervals — identical to Step 3)
# ============================================================================

def _delta_k(strikes: np.ndarray) -> np.ndarray:
    """
    Trapezoid-rule ΔK for a sorted array of strikes.

    Inner: ΔKᵢ = (K_{i+1} − K_{i−1}) / 2
    Edges: ΔK₀ = K₁ − K₀  ;  ΔKₙ = Kₙ − K_{n−1}
    """
    n  = len(strikes)
    dk = np.empty(n, dtype=float)
    if n == 1:
        dk[0] = 1.0
        return dk
    if n == 2:
        dk[:] = strikes[1] - strikes[0]
        return dk
    dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    dk[0]    = strikes[1]  - strikes[0]
    dk[-1]   = strikes[-1] - strikes[-2]
    return dk


# ============================================================================
# VIX from an IV array  (bump reconstruction core)
# ============================================================================

def _sigma2_from_iv_array(
    strikes: np.ndarray,
    ivs: np.ndarray,
    K0: float,
    F: float,
    T: float,
    r: float,
) -> float:
    """
    Return the CBOE annualised variance σ² from an (strike, IV) array via BS.

      σ² = (2/T)·e^{rT}·Σ(ΔK/K²)·Q_BS(K)  −  (1/T)(F/K₀ − 1)²

    Returns σ² (floored at 0) rather than VIX so the result can feed directly
    into the two-tenor 30-day interpolation in _vix_30d_from_bumped_pair.
    """
    Q = np.array(
        [_otm_bs_price(F, K, K0, sigma, T, r)
         for K, sigma in zip(strikes, ivs)],
        dtype=float,
    )
    dk     = _delta_k(strikes)
    growth = np.exp(r * T)
    main   = np.sum((dk / strikes**2) * growth * Q)
    adj    = (F / K0 - 1.0) ** 2
    sigma2 = (2.0 / T) * main - (1.0 / T) * adj
    return float(max(sigma2, 0.0))


# ============================================================================
# Quadratic IV surface fitting  (legacy — superseded by polynomial projection)
# QuadraticParams / fit_quadratic are retained for reference only and are not
# called in the main decomposition path.
# ============================================================================

@dataclass
class QuadraticParams:
    """
    Parameters of the local quadratic IV model.

    Model:  IV(m) = a + b·m + c·m²   where  m = (K − K₀) / K₀

    Full-smile fit (all strikes in ATM window):
        IV(m) = a + b·m + c·m²   where  m = (K − K₀) / K₀

    Side-specific fits (shared intercept a, each half fitted separately):
        IV_put(m)  = a + b_put·m  + c_put·m²   for m ≤ 0  (OTM puts)
        IV_call(m) = a + b_call·m + c_call·m²  for m ≥ 0  (OTM calls)

    Attributes
    ----------
    a        : float  – ATM level (IV at K₀, m = 0).
    b        : float  – full-smile skew (dIV/dm at m = 0).
    c        : float  – full-smile convexity (d²IV/dm² / 2 at m = 0).
    k0       : float  – reference strike K₀ used for normalisation.
    r2       : float  – R² of the full-smile OLS regression.
    n_strikes: int    – total strikes used in the full ATM window.
    b_put    : float or None  – put-side slope  (m ≤ 0); None if <3 put strikes.
    c_put    : float or None  – put-side curvature; None if <3 put strikes.
    b_call   : float or None  – call-side slope (m ≥ 0); None if <3 call strikes.
    c_call   : float or None  – call-side curvature; None if <3 call strikes.
    """
    a: float
    b: float
    c: float
    k0: float
    r2: float
    n_strikes: int
    b_put:  Optional[float] = None
    c_put:  Optional[float] = None
    b_call: Optional[float] = None
    c_call: Optional[float] = None


def fit_quadratic(
    surface: IVSurface,
    window_pct: float = config.QUADRATIC_WINDOW_PCT,
) -> Optional[QuadraticParams]:
    """
    Fit a local quadratic model to the near-ATM portion of the IV smile.

    Model
    -----
    Using normalised moneyness  m = (K − K₀) / K₀  for numerical stability:

        IV(m) = a + b·m + c·m²

    Only strikes within the window  m ∈ [−window_pct, +window_pct]  are used.
    This keeps the fit local, avoiding contamination from the steep put tail
    and the flat call region far from ATM.

    Method
    ------
    Ordinary least squares via numpy.linalg.lstsq (no regularisation needed
    for 10+ points with a 3-parameter polynomial).

    R² is reported to indicate fit quality; the VIX decomposition is most
    meaningful when R² > 0.9 across both days.

    Parameters
    ----------
    surface    : IVSurface  –  built in Step 2.
    window_pct : float      –  half-width of ATM window in moneyness units.
                               Default from config.QUADRATIC_WINDOW_PCT = 0.15.

    Returns
    -------
    QuadraticParams  or  None (if fewer than config.MIN_QUAD_STRIKES found).
    """
    K0 = surface.k0
    K  = surface.strikes
    IV = surface.ivs

    # ── Filter to ATM window ──────────────────────────────────────────────────
    m    = (K - K0) / K0          # normalised moneyness
    mask = np.abs(m) <= window_pct
    K_w  = K[mask]
    IV_w = IV[mask]
    m_w  = m[mask]

    if len(K_w) < config.MIN_QUAD_STRIKES:
        log.warning(
            "[%s / %s] Only %d strikes in ATM window (need %d) — "
            "quadratic fit skipped.",
            surface.date.date(), surface.expiration.date(),
            len(K_w), config.MIN_QUAD_STRIKES,
        )
        return None

    # ── OLS: IV = a + b·m + c·m² ─────────────────────────────────────────────
    # Design matrix  [1, m, m²]
    A = np.column_stack([np.ones_like(m_w), m_w, m_w ** 2])

    # Solve via least-squares  (handles over-determined system cleanly)
    coeffs, _, rank, _ = np.linalg.lstsq(A, IV_w, rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # ── R² ───────────────────────────────────────────────────────────────────
    IV_pred = A @ coeffs
    ss_res  = float(np.sum((IV_w - IV_pred) ** 2))
    ss_tot  = float(np.sum((IV_w - IV_w.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    log.debug(
        "[%s] Quad fit: a=%.4f b=%.4f c=%.4f R²=%.4f n=%d",
        surface.date.date(), a, b, c, r2, len(K_w),
    )

    # ── Side-specific fits: shared intercept a, fit [b, c] per half ──────────
    # OTM puts: m ≤ 0 (at K₀, m = 0, both halves share the point)
    # OTM calls: m ≥ 0
    # Need ≥ 3 strikes per side to fit 2 parameters meaningfully (else None).
    MIN_SIDE = 3

    def _fit_side(m_side: np.ndarray, iv_side: np.ndarray):
        """OLS of (IV − a) on [m, m²] — no intercept; returns (b, c) or None."""
        if len(m_side) < MIN_SIDE:
            return None, None
        y   = iv_side - a
        A_s = np.column_stack([m_side, m_side ** 2])
        coef, _, _, _ = np.linalg.lstsq(A_s, y, rcond=None)
        return float(coef[0]), float(coef[1])

    put_mask  = m_w <= 0
    call_mask = m_w >= 0
    b_put,  c_put  = _fit_side(m_w[put_mask],  IV_w[put_mask])
    b_call, c_call = _fit_side(m_w[call_mask], IV_w[call_mask])

    return QuadraticParams(
        a=a, b=b, c=c, k0=K0, r2=r2, n_strikes=len(K_w),
        b_put=b_put, c_put=c_put, b_call=b_call, c_call=c_call,
    )


# ============================================================================
# Total-variance quadratic surface fitting  (log-moneyness space)
# ============================================================================

@dataclass
class TVParams:
    """
    Parameters of the total-variance quadratic model in log-moneyness space.

    Model:  w(k) = α + β·k + γ·k²
            where  w = T·IV²  (total variance),
                   k = ln(K/F)  (log-moneyness)

    ATM (k = 0): w(0) = α  →  ATM total variance = α,  ATM IV = √(α/T).

    This parameterisation is more natural than the IV-space quadratic because:
      • VIX integrates variance (not IV), so bumps in w-space are additive in
        the CBOE integrand without the non-linear IV→w conversion error.
      • log-moneyness handles the asymmetry of the smile more naturally than
        normalised moneyness, especially for large spot moves.

    Full-smile fit (all strikes in ATM window):
        w(k) = α + β·k + γ·k²

    Side-specific fits (shared intercept α, each half fitted separately):
        w_put(k)  = α + β_put·k  + γ_put·k²   for k ≤ 0  (OTM puts)
        w_call(k) = α + β_call·k + γ_call·k²  for k > 0  (OTM calls)

    Attributes
    ----------
    alpha    : ATM total variance  (w at k = 0  =  T·σ_ATM²).
    beta     : full-smile total-variance skew  (dw/dk at k = 0).
    gamma    : full-smile total-variance curvature  (d²w/dk²/2 at k = 0).
    F        : forward price used for  k = ln(K/F).
    T        : time to expiry (years).
    r2       : R² of the full-smile OLS fit.
    n_strikes: number of strikes used in the fit.
    beta_put / gamma_put   : put-side (k ≤ 0) slope / curvature (or None).
    beta_call / gamma_call : call-side (k > 0) slope / curvature (or None).
    """
    alpha:    float
    beta:     float
    gamma:    float
    F:        float
    T:        float
    r2:       float
    n_strikes: int
    beta_put:   Optional[float] = None
    gamma_put:  Optional[float] = None
    beta_call:  Optional[float] = None
    gamma_call: Optional[float] = None

    @property
    def atm_iv(self) -> float:
        """ATM implied vol = √(α/T).  Backward-compatible with signals.py (a_t)."""
        return float(np.sqrt(max(self.alpha / self.T, 0.0)))


def fit_tv_surface(
    surface: IVSurface,
    window_pct: float = config.QUADRATIC_WINDOW_PCT,
    F_ref: Optional[float] = None,
) -> Optional[TVParams]:
    """
    Fit  w(k) = α + β·k + γ·k²  where  w = T·IV²,  k = ln(K/F).

    Working in total-variance / log-moneyness space keeps the bump method
    consistent with the variance integrand in the CBOE formula and avoids
    the non-linear IV-space rounding error that inflates decomposition residuals.

    Parameters
    ----------
    surface    : IVSurface from Step 2.
    window_pct : half-width of the ATM window in |k| units.
                 Default from config.QUADRATIC_WINDOW_PCT = 0.15  (≈ ±15%).

    Returns
    -------
    TVParams  or  None (if fewer than config.MIN_QUAD_STRIKES strikes in window).
    """
    # F_ref overrides the log-moneyness reference (used to re-fit t-1 surface in
    # the t-day forward frame, which isolates pure shape changes from forward movement)
    F  = F_ref if F_ref is not None else surface.forward
    T  = surface.T
    K  = surface.strikes
    IV = surface.ivs

    k    = np.log(K / F)            # log-moneyness  (k=0 at K=F)
    w    = T * IV ** 2              # total variance
    mask = np.abs(k) <= window_pct  # ATM window
    k_w  = k[mask]
    w_w  = w[mask]

    if len(k_w) < config.MIN_QUAD_STRIKES:
        log.warning(
            "[%s / %s] Only %d strikes in TV window (need %d) — "
            "TV surface fit skipped.",
            surface.date.date(), surface.expiration.date(),
            len(k_w), config.MIN_QUAD_STRIKES,
        )
        return None

    # OLS: w = α + β·k + γ·k²
    A_w = np.column_stack([np.ones_like(k_w), k_w, k_w ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(A_w, w_w, rcond=None)
    alpha, beta, gamma = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # R²
    w_pred = A_w @ coeffs
    ss_res = float(np.sum((w_w - w_pred) ** 2))
    ss_tot = float(np.sum((w_w - w_w.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

    log.debug(
        "[%s] TV fit: α=%.5f β=%.4f γ=%.4f R²=%.4f n=%d",
        surface.date.date(), alpha, beta, gamma, r2, len(k_w),
    )

    # Sided fits: shared α, OLS of (w − α) on [k, k²] — no intercept
    MIN_SIDE = 3

    def _fit_side(k_s: np.ndarray, w_s: np.ndarray):
        if len(k_s) < MIN_SIDE:
            return None, None
        y   = w_s - alpha
        A_s = np.column_stack([k_s, k_s ** 2])
        coef, _, _, _ = np.linalg.lstsq(A_s, y, rcond=None)
        return float(coef[0]), float(coef[1])

    put_mask  = k_w <= 0
    call_mask = k_w >= 0
    beta_put,  gamma_put  = _fit_side(k_w[put_mask],  w_w[put_mask])
    beta_call, gamma_call = _fit_side(k_w[call_mask], w_w[call_mask])

    return TVParams(
        alpha=alpha, beta=beta, gamma=gamma,
        F=F, T=T, r2=r2, n_strikes=len(k_w),
        beta_put=beta_put, gamma_put=gamma_put,
        beta_call=beta_call, gamma_call=gamma_call,
    )




# ============================================================================
# TV-space perturbation helpers  (legacy — not called in current method)
# These functions supported the old total-variance bump approach.  The current
# polynomial projection uses _vix_30d_from_iv_arrays instead.
# ============================================================================

def _vix_30d_with_tv_perturbation(
    surf_near: IVSurface,
    surf_next: Optional[IVSurface],
    pert_w_near: np.ndarray,
    pert_w_next: Optional[np.ndarray],
    r: float,
) -> float:
    """
    Apply total-variance perturbation arrays and return the CBOE 30-day VIX.

    For each surface:
      1.  w_bumped = T·IV² + pert_w   (floored at 0 to avoid complex IVs)
      2.  IV_bumped = max(√(w_bumped/T), BUMP_IV_FLOOR)
      3.  Apply CBOE formula via BS prices  →  σ²
    Then 30-day time-weighted interpolation as in Step 3.

    Working in total-variance space means the perturbation Δw is additive in
    the variance integrand, eliminating the non-linear IV→w conversion error
    that inflated residuals in the IV-space quadratic approach.

    Parameters
    ----------
    surf_near, surf_next : IVSurface at the same date.
    pert_w_near : total-variance (T·IV²) perturbation per strike of surf_near.
    pert_w_next : total-variance perturbation per strike of surf_next (or None).
    r           : risk-free rate.
    """
    # ── Near surface ─────────────────────────────────────────────────────────
    T1            = surf_near.T
    w_base_near   = T1 * surf_near.ivs ** 2
    w_bumped_near = np.maximum(w_base_near + pert_w_near, 0.0)
    ivs_near      = np.maximum(np.sqrt(w_bumped_near / T1), config.BUMP_IV_FLOOR)
    sigma2_near   = _sigma2_from_iv_array(
        surf_near.strikes, ivs_near,
        surf_near.k0, surf_near.forward, T1, r,
    )
    D1  = surf_near.dte
    T30 = 30.0 / 365.0

    if surf_next is None or pert_w_next is None or abs(surf_next.dte - D1) < 0.5:
        sigma2_30 = sigma2_near * T1 / T30
        return float(100.0 * np.sqrt(max(sigma2_30, 0.0)))

    # ── Next surface ─────────────────────────────────────────────────────────
    T2            = surf_next.T
    w_base_next   = T2 * surf_next.ivs ** 2
    w_bumped_next = np.maximum(w_base_next + pert_w_next, 0.0)
    ivs_next      = np.maximum(np.sqrt(w_bumped_next / T2), config.BUMP_IV_FLOOR)
    sigma2_next   = _sigma2_from_iv_array(
        surf_next.strikes, ivs_next,
        surf_next.k0, surf_next.forward, T2, r,
    )
    D2  = surf_next.dte
    D30 = 30.0

    # ── CBOE 30-day interpolation (mirrors Step 3 _interpolate_to_30d) ────────
    if D1 >= D30:
        sigma2_30 = sigma2_near * T1 / T30
    elif D2 <= D30:
        sigma2_30 = sigma2_next * T2 / T30
    else:
        w_n = (D2 - D30) / (D2 - D1)
        w_x = (D30 - D1) / (D2 - D1)
        sigma2_30 = (T1 * sigma2_near * w_n + T2 * sigma2_next * w_x) / T30

    return float(100.0 * np.sqrt(max(sigma2_30, 0.0)))


# (also legacy — see note above)

def _vix_30d_tv_pert_in_frame(
    surf_iv_near:  IVSurface,
    surf_iv_next:  Optional[IVSurface],
    surf_ref_near: IVSurface,
    surf_ref_next: Optional[IVSurface],
    pert_w_near:   np.ndarray,
    pert_w_next:   Optional[np.ndarray],
    r: float,
) -> float:
    """
    Apply a TV perturbation to surf_iv IVs but evaluate the CBOE formula in
    the reference frame (K₀, F, T, DTE) taken from surf_ref.

    This decouples the IV source from the integration reference frame, which
    is essential for the sticky-strike decomposition step:

      • Sticky-strike VIX:   surf_iv = t-1 surface,  surf_ref = t-day surface,
                              pert_w  = 0.
      • Shape-change bumps:  surf_iv = t-1 surface,  surf_ref = t-day surface,
                              pert_w  = Δα + Δβ·k + Δγ·k²  (accumulated).

    For each surface pair (near, next):
      1.  w_bumped = T_ref · IV_iv² + pert_w   (total variance in ref-tenor units)
      2.  IV_bumped = max(√(w_bumped / T_ref), BUMP_IV_FLOOR)
      3.  sigma2   = CBOE(strikes_iv, IV_bumped, K₀_ref, F_ref, T_ref, r)
    Then 30-day interpolation using DTEs from surf_ref.

    Parameters
    ----------
    surf_iv_near/next  : IVSurface providing strikes and base IVs.
    surf_ref_near/next : IVSurface providing K₀, F, T, DTE for CBOE formula.
    pert_w_near/next   : total-variance perturbation arrays (may be zeros).
    r                  : risk-free rate.
    """
    # ── Near surface ─────────────────────────────────────────────────────────
    T1_ref        = surf_ref_near.T
    w_base_near   = T1_ref * surf_iv_near.ivs ** 2
    w_bumped_near = np.maximum(w_base_near + pert_w_near, 0.0)
    ivs_near      = np.maximum(np.sqrt(w_bumped_near / T1_ref), config.BUMP_IV_FLOOR)
    sigma2_near   = _sigma2_from_iv_array(
        surf_iv_near.strikes, ivs_near,
        surf_ref_near.k0, surf_ref_near.forward, T1_ref, r,
    )
    D1  = surf_ref_near.dte
    T30 = 30.0 / 365.0

    if (surf_iv_next is None or surf_ref_next is None
            or pert_w_next is None or abs(surf_ref_next.dte - D1) < 0.5):
        sigma2_30 = sigma2_near * T1_ref / T30
        return float(100.0 * np.sqrt(max(sigma2_30, 0.0)))

    # ── Next surface ─────────────────────────────────────────────────────────
    T2_ref        = surf_ref_next.T
    w_base_next   = T2_ref * surf_iv_next.ivs ** 2
    w_bumped_next = np.maximum(w_base_next + pert_w_next, 0.0)
    ivs_next      = np.maximum(np.sqrt(w_bumped_next / T2_ref), config.BUMP_IV_FLOOR)
    sigma2_next   = _sigma2_from_iv_array(
        surf_iv_next.strikes, ivs_next,
        surf_ref_next.k0, surf_ref_next.forward, T2_ref, r,
    )
    D2  = surf_ref_next.dte
    D30 = 30.0

    # ── CBOE 30-day interpolation ─────────────────────────────────────────────
    if D1 >= D30:
        sigma2_30 = sigma2_near * T1_ref / T30
    elif D2 <= D30:
        sigma2_30 = sigma2_next * T2_ref / T30
    else:
        w_n = (D2 - D30) / (D2 - D1)
        w_x = (D30 - D1) / (D2 - D1)
        sigma2_30 = (T1_ref * sigma2_near * w_n + T2_ref * sigma2_next * w_x) / T30

    return float(100.0 * np.sqrt(max(sigma2_30, 0.0)))


# ============================================================================
# Sticky-strike baseline and polynomial projection helpers
# ============================================================================

def _eval_spline_with_fallback(
    surf: IVSurface,
    K: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the IVSurface spline at each strike in K.

    For strikes within [surf.min_strike, surf.max_strike] the spline is used.
    For strikes outside that range (edge of the t-day surface) the fallback
    IV is returned instead.  This keeps the deep-OTM tails stable when the
    strike grids of two days differ slightly.

    Parameters
    ----------
    surf     : IVSurface  –  the target surface whose spline we evaluate.
    K        : 1-D array of strikes at which to evaluate (from another day).
    fallback : 1-D array of fallback IVs (same shape as K).

    Returns
    -------
    np.ndarray  –  IV values, same shape as K.
    """
    result = fallback.copy()
    lo, hi = surf.min_strike, surf.max_strike
    for i, k in enumerate(K):
        if lo <= k <= hi:
            result[i] = float(surf(k))
    return result


def _vix_30d_from_iv_arrays(
    strikes_near: np.ndarray,
    ivs_near:     np.ndarray,
    surf_ref_near: IVSurface,
    strikes_next:  Optional[np.ndarray],
    ivs_next:      Optional[np.ndarray],
    surf_ref_next: Optional[IVSurface],
    r: float,
) -> float:
    """
    Compute the CBOE 30-day VIX from explicit (strike, IV) arrays.

    The CBOE K₀, F, T, DTE are taken from the surf_ref surfaces, which allows
    using a different strike grid (e.g. t-1 strikes with t-day reference frame)
    while keeping the integration parameters consistent with the t-day surface.

    This is the core primitive for the polynomial projection attribution.  The
    caller builds five intermediate IV arrays by sequentially applying the
    polynomial components (α, β_put·m, β_call·m, γ_put·m², γ_call·m²); the
    resulting VIX difference at each step measures that component's contribution.

    Parameters
    ----------
    strikes_near / ivs_near : explicit (strike, IV) arrays for the near tenor.
    surf_ref_near           : provides K₀, F, T, DTE for the CBOE formula.
    strikes_next / ivs_next : same for the next tenor (None → single-tenor).
    surf_ref_next           : reference for the next tenor (None → skip).
    r                       : risk-free rate.

    Returns
    -------
    float  –  CBOE 30-day VIX (100 × √σ²₃₀).
    """
    sigma2_near = _sigma2_from_iv_array(
        strikes_near, ivs_near,
        surf_ref_near.k0, surf_ref_near.forward, surf_ref_near.T, r,
    )
    D1  = surf_ref_near.dte
    T1  = surf_ref_near.T
    T30 = 30.0 / 365.0

    if (strikes_next is None or ivs_next is None or surf_ref_next is None
            or abs(surf_ref_next.dte - D1) < 0.5):
        sigma2_30 = sigma2_near * T1 / T30
        return float(100.0 * np.sqrt(max(sigma2_30, 0.0)))

    sigma2_next = _sigma2_from_iv_array(
        strikes_next, ivs_next,
        surf_ref_next.k0, surf_ref_next.forward, surf_ref_next.T, r,
    )
    D2  = surf_ref_next.dte
    T2  = surf_ref_next.T
    D30 = 30.0

    if D1 >= D30:
        sigma2_30 = sigma2_near * T1 / T30
    elif D2 <= D30:
        sigma2_30 = sigma2_next * T2 / T30
    else:
        w_n = (D2 - D30) / (D2 - D1)
        w_x = (D30 - D1) / (D2 - D1)
        sigma2_30 = (T1 * sigma2_near * w_n + T2 * sigma2_next * w_x) / T30

    return float(100.0 * np.sqrt(max(sigma2_30, 0.0)))


# ============================================================================
# Single-day decomposition
# ============================================================================

def decompose_one_day(
    date_t:     pd.Timestamp,
    date_t1:    pd.Timestamp,
    surfaces:   Dict[pd.Timestamp, Dict[str, IVSurface]],
    vix_synth:  pd.Series,
    r: float = config.RISK_FREE_RATE,
) -> Optional[dict]:
    """
    Decompose ΔVIX = VIX_t − VIX_{t−1} into parallel, skew, and convexity
    contributions using the numerical bump method.

    Algorithm (spline polynomial projection — asymmetric piecewise fit)
    --------------------------------------------------------------------
    1.  Map t-1 IVs onto the t-day strike grid via the t-1 spline
        (sticky-strike baseline IV_ss).
    2.  Compute the IV change at each t-day strike:
            ΔIV_i = IV_t(K_i) − IV_ss(K_i)
    3.  Fit a 5-parameter piecewise polynomial with shared intercept via
        CBOE-integrand-weighted OLS:
            ΔIV = α + β_put·m  + γ_put·m²   (m ≤ 0, OTM put side)
            ΔIV = α + β_call·m + γ_call·m²  (m > 0, OTM call side)
        m_i = (K_i − K₀)/K₀.  Put/call sides share α (parallel),
        ensuring a unique level shift and continuity at K₀.
    4.  Build five intermediate IV arrays (put/call sides are disjoint
        so contributions are order-independent):
            IV₁[i] = IV_ss[i] + α                         [+parallel, all]
            IV₂[i] = IV₁[i] + β_put·m_i    (m≤0 only)   [+put skew]
            IV₃[i] = IV₂[i] + β_call·m_i   (m>0 only)   [+call skew]
            IV₄[i] = IV₃[i] + γ_put·m_i²   (m≤0 only)   [+put convexity]
            IV₅[i] = IV₄[i] + γ_call·m_i²  (m>0 only)   [+call convexity]
            IV₆[i] = IV_t[i]                              [+higher-order]
    5.  Compute VIX at each surface (full CBOE formula, no linearisation):
            VIX₀ → VIX_SS → VIX₁ → VIX₂ → VIX₃ → VIX₄ → VIX₅ → VIX_BS_T
    6.  Contributions (telescope exactly to ΔVIX):
            sticky         = VIX_SS  − VIX₀
            parallel       = VIX₁   − VIX_SS
            put_skew       = VIX₂   − VIX₁
            call_skew      = VIX₃   − VIX₂
            put_convexity  = VIX₄   − VIX₃
            call_convexity = VIX₅   − VIX₄
            higher_order   = VIX_BS_T − VIX₅
            model_total    = VIX₅   − VIX₀
            residual       = VIX_BS_T − VIX₅  =  higher_order
        Chain closes: model_total + residual = ΔVIX exactly.

    Parameters
    ----------
    date_t, date_t1 : timestamps for day t and t−1.
    surfaces        : dict from Step 2.
    vix_synth       : pd.Series of synthetic VIX from Step 3.
    r               : risk-free rate.

    Returns
    -------
    dict  or  None (if either day's TV surface fit fails or a surface is missing).
    """
    # ── Retrieve surfaces ─────────────────────────────────────────────────────
    sv_t  = surfaces.get(date_t,  {})
    sv_t1 = surfaces.get(date_t1, {})

    surf_t_near  = sv_t.get("near")
    surf_t_next  = sv_t.get("next")
    surf_t1_near = sv_t1.get("near")
    surf_t1_next = sv_t1.get("next")

    if surf_t_near is None or surf_t1_near is None:
        log.warning(
            "Missing near-term surface for %s or %s — skipping decomposition.",
            date_t.date(), date_t1.date(),
        )
        return None

    # ── Fit TV surface for t and t-1 in their own forward frames ─────────────
    # params_t1 in own frame → a_t1 / atm_iv backward-compat with signals.py
    params_t  = fit_tv_surface(surf_t_near)
    params_t1 = fit_tv_surface(surf_t1_near)

    # Re-fit t-1 in the t-day forward frame so that Δα/Δβ/Δγ reflect only pure
    # surface shape changes — the forward movement effect is captured separately
    # in the sticky-strike step.
    params_t1_rf = fit_tv_surface(surf_t1_near, F_ref=surf_t_near.forward)

    if params_t is None or params_t1 is None or params_t1_rf is None:
        log.warning(
            "[%s] TV surface fit failed for t or t−1 — skipping.",
            date_t.date(),
        )
        return None

    # ── Changes in TV parameters (reframed: pure shape, no forward effect) ───
    delta_alpha = params_t.alpha - params_t1_rf.alpha
    delta_beta  = params_t.beta  - params_t1_rf.beta
    delta_gamma = params_t.gamma - params_t1_rf.gamma

    # ATM-IV change (backward compat: each day uses its own forward frame)
    delta_a = params_t.atm_iv - params_t1.atm_iv

    # ── Model-free synthetic ΔVIX (kept for reference / diagnostics) ─────────
    vix_t_synth  = vix_synth.get(date_t)
    vix_t1_synth = vix_synth.get(date_t1)
    delta_vix_synth = (
        float(vix_t_synth) - float(vix_t1_synth)
        if (vix_t_synth is not None and vix_t1_synth is not None
            and not np.isnan(float(vix_t_synth)) and not np.isnan(float(vix_t1_synth)))
        else float("nan")
    )

    # ── Side-specific parameter changes (reframed t-1) ────────────────────────
    delta_beta_put  = (
        params_t.beta_put   - params_t1_rf.beta_put
        if (params_t.beta_put   is not None and params_t1_rf.beta_put   is not None)
        else delta_beta
    )
    delta_gamma_put  = (
        params_t.gamma_put  - params_t1_rf.gamma_put
        if (params_t.gamma_put  is not None and params_t1_rf.gamma_put  is not None)
        else delta_gamma
    )
    delta_beta_call = (
        params_t.beta_call  - params_t1_rf.beta_call
        if (params_t.beta_call  is not None and params_t1_rf.beta_call  is not None)
        else delta_beta
    )
    delta_gamma_call = (
        params_t.gamma_call - params_t1_rf.gamma_call
        if (params_t.gamma_call is not None and params_t1_rf.gamma_call is not None)
        else delta_gamma
    )

    # ── Spline polynomial projection decomposition ────────────────────────────
    # Use the t-day strike grid.  Project ΔIV = IV_t − IV_ss onto {1, m, m²}
    # via CBOE-integrand-weighted OLS (all strikes).  Build four intermediate
    # IV arrays; compute CBOE VIX at each step.  The chain closes exactly:
    #   VIX_BS_T = VIX_SS + parallel + skew + convexity + higher_order
    K_t_near  = surf_t_near.strikes
    IV_t_near = surf_t_near.ivs
    K_t_next  = surf_t_next.strikes if surf_t_next is not None else None
    IV_t_next = surf_t_next.ivs     if surf_t_next is not None else None

    # Sticky-strike IVs: t-1 spline evaluated at t-day strikes
    IV_ss_near = _eval_spline_with_fallback(surf_t1_near, K_t_near, IV_t_near)
    IV_ss_next = (
        _eval_spline_with_fallback(surf_t1_next, K_t_next, IV_t_next)
        if (surf_t1_next is not None and K_t_next is not None)
        else None
    )

    # ── VIX₀, VIX_SS, VIX_BS_T ──────────────────────────────────────────────
    vix0 = _vix_30d_from_iv_arrays(
        surf_t1_near.strikes, surf_t1_near.ivs, surf_t1_near,
        surf_t1_next.strikes if surf_t1_next is not None else None,
        surf_t1_next.ivs     if surf_t1_next is not None else None,
        surf_t1_next, r,
    )
    vix_ss = _vix_30d_from_iv_arrays(
        K_t_near, IV_ss_near, surf_t_near,
        K_t_next, IV_ss_next, surf_t_next, r,
    )
    vix_bs_t = _vix_30d_from_iv_arrays(
        K_t_near, IV_t_near, surf_t_near,
        K_t_next, IV_t_next, surf_t_next, r,
    )
    delta_vix      = vix_bs_t - vix0
    sticky_contrib = vix_ss   - vix0

    # ── CBOE-weighted OLS: asymmetric piecewise fit per tenor ────────────────
    def _poly_project(K, IV_t_arr, IV_ss_arr, K0):
        """
        Fit piecewise polynomial with shared intercept α:
            ΔIV = α + β_put·m  + γ_put·m²   (m ≤ 0, put side)
            ΔIV = α + β_call·m + γ_call·m²  (m > 0, call side)

        Weights: w_i = ΔK_i / K_i²  (CBOE trapezoid-rule weights).
        Put/call sides are disjoint, so their contributions are additive
        regardless of the order they are applied in the chain.

        Returns (alpha, beta_put, gamma_put, beta_call, gamma_call,
                 iv1, iv2, iv3, iv4, iv5) where iv1–iv5 are the five
        intermediate IV arrays, each floored at BUMP_IV_FLOOR.
        """
        dIV      = IV_t_arr - IV_ss_arr
        m        = (K - K0) / K0
        w        = _delta_k(K) / K ** 2
        sw       = np.sqrt(w)
        put_mask = m <= 0
        cal_mask = m >  0

        # 5 columns: [α,  β_put,  γ_put,  β_call,  γ_call]
        X = np.zeros((len(K), 5))
        X[:, 0]          = 1.0
        X[put_mask, 1]   = m[put_mask]
        X[put_mask, 2]   = m[put_mask] ** 2
        X[cal_mask, 3]   = m[cal_mask]
        X[cal_mask, 4]   = m[cal_mask] ** 2

        coeffs, _, _, _ = np.linalg.lstsq(X * sw[:, np.newaxis], dIV * sw,
                                           rcond=None)
        alpha, b_p, g_p, b_c, g_c = coeffs

        fl  = config.BUMP_IV_FLOOR
        iv1 = np.maximum(IV_ss_arr + alpha, fl)
        iv2 = iv1.copy()
        iv2[put_mask] = np.maximum(iv1[put_mask] + b_p * m[put_mask],       fl)
        iv3 = iv2.copy()
        iv3[cal_mask] = np.maximum(iv2[cal_mask] + b_c * m[cal_mask],       fl)
        iv4 = iv3.copy()
        iv4[put_mask] = np.maximum(iv3[put_mask] + g_p * m[put_mask] ** 2,  fl)
        iv5 = iv4.copy()
        iv5[cal_mask] = np.maximum(iv4[cal_mask] + g_c * m[cal_mask] ** 2,  fl)
        return alpha, b_p, g_p, b_c, g_c, iv1, iv2, iv3, iv4, iv5

    a_n, bp_n, gp_n, bc_n, gc_n, iv1_near, iv2_near, iv3_near, iv4_near, iv5_near = (
        _poly_project(K_t_near, IV_t_near, IV_ss_near, surf_t_near.k0)
    )
    if K_t_next is not None and IV_ss_next is not None:
        _, _, _, _, _, iv1_next, iv2_next, iv3_next, iv4_next, iv5_next = (
            _poly_project(K_t_next, IV_t_next, IV_ss_next, surf_t_next.k0)
        )
    else:
        iv1_next = iv2_next = iv3_next = iv4_next = iv5_next = None

    # ── Chain: SS→1(par)→2(put sk)→3(call sk)→4(put cv)→5(call cv)→BS_T ─────
    vix1 = _vix_30d_from_iv_arrays(
        K_t_near, iv1_near, surf_t_near, K_t_next, iv1_next, surf_t_next, r)
    vix2 = _vix_30d_from_iv_arrays(
        K_t_near, iv2_near, surf_t_near, K_t_next, iv2_next, surf_t_next, r)
    vix3 = _vix_30d_from_iv_arrays(
        K_t_near, iv3_near, surf_t_near, K_t_next, iv3_next, surf_t_next, r)
    vix4 = _vix_30d_from_iv_arrays(
        K_t_near, iv4_near, surf_t_near, K_t_next, iv4_next, surf_t_next, r)
    vix5 = _vix_30d_from_iv_arrays(
        K_t_near, iv5_near, surf_t_near, K_t_next, iv5_next, surf_t_next, r)
    # VIX₆ = vix_bs_t (already computed) — chain closes exactly

    parallel_contrib      = vix1     - vix_ss
    put_skew_contrib      = vix2     - vix1
    call_skew_contrib     = vix3     - vix2
    put_convexity_contrib = vix4     - vix3
    cal_convexity_contrib = vix5     - vix4
    higher_order_contrib  = vix_bs_t - vix5
    skew_contrib          = put_skew_contrib + call_skew_contrib
    convexity_contrib     = put_convexity_contrib + cal_convexity_contrib

    # model_total = quadratic model (sticky→vix5); residual = higher_order
    model_total = vix5     - vix0
    residual    = vix_bs_t - vix5

    # ── Explained fraction ────────────────────────────────────────────────────
    if abs(delta_vix) >= 0.05:
        explained_pct = (1.0 - abs(residual) / abs(delta_vix)) * 100.0
    else:
        explained_pct = float("nan")

    return {
        "date":                    date_t,
        # Total BS-consistent ΔVIX
        "delta_vix":               delta_vix,
        # Contributions — chain closes exactly
        "sticky_strike_contribution":  sticky_contrib,
        "parallel_contribution":       parallel_contrib,
        "put_skew_contribution":       put_skew_contrib,
        "call_skew_contribution":      call_skew_contrib,
        "put_convexity_contribution":  put_convexity_contrib,
        "call_convexity_contribution": cal_convexity_contrib,
        "skew_contribution":           skew_contrib,
        "convexity_contribution":      convexity_contrib,
        "higher_order_contribution":   higher_order_contrib,
        # model_total = quadratic model (sticky+…+call_conv); residual = higher_order
        "model_total":    model_total,
        "residual":       residual,
        "explained_pct":  explained_pct,
        # OLS piecewise polynomial coefficients (near tenor)
        "proj_alpha":      float(a_n),
        "proj_beta_put":   float(bp_n),  "proj_gamma_put":  float(gp_n),
        "proj_beta_call":  float(bc_n),  "proj_gamma_call": float(gc_n),
        # TV parameters at t and t−1 (own frames — backward compat with signals.py)
        # a_t / a_t1 = ATM IV in each day's own forward frame
        "a_t":     params_t.atm_iv,   "b_t":    params_t.beta,   "c_t":    params_t.gamma,
        "a_t1":    params_t1.atm_iv,  "b_t1":   params_t1.beta,  "c_t1":   params_t1.gamma,
        "delta_a": delta_a,           "delta_b": delta_beta,      "delta_c": delta_gamma,
        # Raw TV parameters
        "alpha_t":  params_t.alpha,    "beta_t":  params_t.beta,    "gamma_t":  params_t.gamma,
        "alpha_t1": params_t1.alpha,   "beta_t1": params_t1.beta,   "gamma_t1": params_t1.gamma,
        # Reframed t-1 params (in t-day forward frame; Δ = pure shape change)
        "alpha_t1_rf": params_t1_rf.alpha, "beta_t1_rf": params_t1_rf.beta,
        "delta_alpha": delta_alpha,    "delta_beta": delta_beta,    "delta_gamma": delta_gamma,
        # Side-specific TV parameter changes (reframed)
        "delta_beta_put":   delta_beta_put,   "delta_gamma_put":   delta_gamma_put,
        "delta_beta_call":  delta_beta_call,  "delta_gamma_call":  delta_gamma_call,
        # TV fit quality
        "r2_t":  params_t.r2,  "r2_t1": params_t1.r2,
        "n_k_t": params_t.n_strikes,
        # Surface metadata
        "k0_t1":  surf_t1_near.k0,    "F_t1":   surf_t1_near.forward,
        "T_t1":   surf_t1_near.T,     "dte_t1": surf_t1_near.dte,
        # Endpoint and synthetic VIX reference
        "vix_bs_t":        vix_bs_t,
        "delta_vix_synth": delta_vix_synth,
        # Intermediate VIX levels along the projection chain
        "vix0": vix0, "vix_ss": vix_ss,
        "vix1": vix1, "vix2": vix2, "vix3": vix3, "vix4": vix4, "vix5": vix5,
    }


# ============================================================================
# Full time-series decomposition
# ============================================================================

def build_decomposition(
    surfaces:  Dict[pd.Timestamp, Dict[str, IVSurface]],
    vix_synth: pd.Series,
    r: float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Run the decomposition for all consecutive trading-day pairs.

    Parameters
    ----------
    surfaces  : built in Step 2.
    vix_synth : pd.Series of synthetic VIX from Step 3  (date index).
    r         : risk-free rate.

    Returns
    -------
    pd.DataFrame indexed by date, with columns:
        delta_vix, sticky_strike_contribution, parallel_contribution,
        skew_contribution, convexity_contribution, higher_order_contribution,
        model_total, residual, explained_pct,
        proj_alpha, proj_beta, proj_gamma,
        a_t, b_t, c_t, delta_a, delta_b, delta_c,
        r2_t, r2_t1, dte_t1, vix0, vix_ss, vix1, vix2, vix3, vix_bs_t

    Notes
    -----
    The first date in the dataset has no t−1 and is excluded.  With 63
    trading days in Q4 2023 this leaves 62 decomposition rows.
    """
    log.info("=" * 60)
    log.info("STEP 4 — VIX Decomposition")
    log.info("=" * 60)

    # ── Sorted list of available dates ───────────────────────────────────────
    # Use only dates that appear in BOTH surfaces and vix_synth
    all_dates = sorted(
        set(surfaces.keys()) & set(vix_synth.dropna().index)
    )
    n_dates = len(all_dates)
    log.info("Dates available: %d  →  %d decomposition rows", n_dates, n_dates - 1)

    records = []
    skipped = 0

    for i in range(1, n_dates):
        date_t  = all_dates[i]
        date_t1 = all_dates[i - 1]

        result = decompose_one_day(date_t, date_t1, surfaces, vix_synth, r)

        if result is None:
            skipped += 1
        else:
            records.append(result)

    if not records:
        raise RuntimeError(
            "All decomposition days failed.  Check surface and VIX data."
        )

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)

    n_ok = len(df)
    log.info(
        "Decomposition complete: %d days, %d skipped.",
        n_ok, skipped,
    )
    _log_decomposition_stats(df)
    return df


def _log_decomposition_stats(df: pd.DataFrame) -> None:
    """Log summary statistics for the decomposition DataFrame."""
    log.info("─" * 60)
    log.info("DECOMPOSITION STATISTICS")
    log.info("─" * 60)

    # Average fractional contribution (on days when |ΔVIX| > 0.1)
    big_days = df[df["delta_vix"].abs() > 0.1]

    # Only report explained% on days where |ΔVIX| ≥ 0.05 (avoids division noise)
    expl_valid = df["explained_pct"].dropna()

    log.info("  Days decomposed                : %d", len(df))
    log.info("  Mean |ΔVIX|                    : %.3f pts", df["delta_vix"].abs().mean())
    log.info("  Mean explained %% (|ΔVIX|≥0.05): %.1f%%  (%d days)",
             expl_valid.mean() if len(expl_valid) else float("nan"), len(expl_valid))
    log.info("  Mean residual                  : %+.4f pts", df["residual"].mean())
    log.info("  Mean |residual|                : %.4f pts", df["residual"].abs().mean())
    log.info("  Fraction |residual|/|ΔVIX|     : %.1f%%",
             (df["residual"].abs() / df["delta_vix"].abs().clip(0.01)).mean() * 100)
    log.info("  Median R² (TV fit)             : %.4f", df["r2_t"].median())
    log.info("  [Residual] = non-quadratic IV change (higher-order spline component).")

    if len(big_days) > 0:
        dv = big_days["delta_vix"].mean()
        for col, label in [
            ("sticky_strike_contribution",  "  Avg sticky-strike contribution"),
            ("parallel_contribution",       "  Avg parallel contribution"),
            ("put_skew_contribution",       "    → put-skew contribution"),
            ("call_skew_contribution",      "    → call-skew contribution"),
            ("put_convexity_contribution",  "    → put-convexity contribution"),
            ("call_convexity_contribution", "    → call-convexity contribution"),
            ("higher_order_contribution",   "    → higher-order contribution"),
        ]:
            val = big_days[col].mean()
            log.info("%-42s : %+.3f pts (%+.1f%% of ΔVIX)", label, val, val / dv * 100)

    log.info("─" * 60)


# ============================================================================
# Validation  —  check that contributions sum to ΔVIX
# ============================================================================

def validate_decomposition(df: pd.DataFrame) -> dict:
    """
    Verify the accounting identity:

        sticky + parallel + skew + convexity  =  model_total  (exact)
        model_total + residual                =  ΔVIX         (exact)

    residual = higher_order_contribution (non-quadratic IV change).
    The explained_pct measures how much of ΔVIX the quadratic model captures.

    Parameters
    ----------
    df : pd.DataFrame  –  output of build_decomposition().

    Returns
    -------
    dict with keys:
        sum_check_max_error   – max |sticky+parallel+skew+conv − model_total|
                                (must be < 1e-10, purely numerical)
        mean_explained_pct    – mean % of ΔVIX explained by quadratic model
        mean_abs_residual     – mean |higher_order_contribution|
        pct_within_05pt       – % of days where |residual| ≤ 0.5 pts
    """
    # Identity: sticky+parallel+put_sk+call_sk+put_cv+call_cv must equal model_total
    model_recon = (
        df["sticky_strike_contribution"]
        + df["parallel_contribution"]
        + df["put_skew_contribution"]
        + df["call_skew_contribution"]
        + df["put_convexity_contribution"]
        + df["call_convexity_contribution"]
    )
    sum_error = (model_recon - df["model_total"]).abs().max()

    pct_within = ((df["residual"].abs() <= 0.5).mean() * 100)

    result = {
        "sum_check_max_error":  float(sum_error),
        "mean_explained_pct":   float(df["explained_pct"].mean()),
        "mean_abs_residual":    float(df["residual"].abs().mean()),
        "pct_within_05pt":      float(pct_within),
    }

    log.info("DECOMPOSITION VALIDATION")
    log.info("  Sum check max error   : %.2e  (must be <1e-9)", sum_error)
    log.info("  Mean explained %%      : %.1f%%", result["mean_explained_pct"])
    log.info("  Mean |residual|       : %.4f pts", result["mean_abs_residual"])
    log.info("  Days |residual|≤0.5pt : %.1f%%", pct_within)

    if sum_error > 1e-9:
        log.error("ACCOUNTING ERROR: contributions do not sum to model_total!")

    return result


# ============================================================================
# Diagnostic print
# ============================================================================

def print_decomposition_table(df: pd.DataFrame, n_rows: int = 20) -> None:
    """
    Print a human-readable table of daily ΔVIX contributions.

    Columns: date, ΔVIX, parallel, skew, convexity, residual, explained%
    """
    display_cols = [
        "delta_vix",
        "sticky_strike_contribution",
        "parallel_contribution",
        "put_skew_contribution",
        "call_skew_contribution",
        "put_convexity_contribution",
        "call_convexity_contribution",
        "higher_order_contribution",
        "residual",
        "explained_pct",
        "r2_t",
    ]
    short_names = {
        "delta_vix":                   "ΔVIX",
        "sticky_strike_contribution":  "Sticky",
        "parallel_contribution":       "Parallel",
        "put_skew_contribution":       "PutSkew",
        "call_skew_contribution":      "CallSkew",
        "put_convexity_contribution":  "PutConv",
        "call_convexity_contribution": "CallConv",
        "higher_order_contribution":   "HigherOrd",
        "residual":                    "Residual",
        "explained_pct":               "Expl%",
        "r2_t":                        "R²",
    }

    disp = df[display_cols].rename(columns=short_names)

    print("\n" + "=" * 110)
    print("VIX DECOMPOSITION — Daily Contributions (sticky / parallel / skew / convexity / higher-order)")
    print("=" * 110)
    print(disp.head(n_rows).to_string(
        float_format=lambda x: f"{x:+.4f}" if abs(x) < 100 else f"{x:.1f}"
    ))
    print("-" * 110)
    print(f"{'Means':20s}  "
          f"ΔVIX={df['delta_vix'].mean():+.4f}  "
          f"Sticky={df['sticky_strike_contribution'].mean():+.4f}  "
          f"Par={df['parallel_contribution'].mean():+.4f}  "
          f"PutSkew={df['put_skew_contribution'].mean():+.4f}  "
          f"CallSkew={df['call_skew_contribution'].mean():+.4f}  "
          f"PutConv={df['put_convexity_contribution'].mean():+.4f}  "
          f"CallConv={df['call_convexity_contribution'].mean():+.4f}  "
          f"HiOrd={df['higher_order_contribution'].mean():+.4f}  "
          f"Res={df['residual'].mean():+.4f}")
    print(f"{'|Mean|':20s}  "
          f"ΔVIX={df['delta_vix'].abs().mean():.4f}  "
          f"Sticky={df['sticky_strike_contribution'].abs().mean():.4f}  "
          f"Par={df['parallel_contribution'].abs().mean():.4f}  "
          f"PutSkew={df['put_skew_contribution'].abs().mean():.4f}  "
          f"CallSkew={df['call_skew_contribution'].abs().mean():.4f}  "
          f"PutConv={df['put_convexity_contribution'].abs().mean():.4f}  "
          f"CallConv={df['call_convexity_contribution'].abs().mean():.4f}  "
          f"HiOrd={df['higher_order_contribution'].abs().mean():.4f}  "
          f"Res={df['residual'].abs().mean():.4f}")
    print("=" * 110)
