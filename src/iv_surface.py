"""
src/iv_surface.py  –  Step 2: Continuous IV Surface Construction
=================================================================
For each trading day this module:

  1. Selects the near-term and next-term expirations that straddle 30 DTE
     (the two maturities required by the CBOE VIX interpolation formula).
  2. Derives the forward index level F from put-call parity for each expiry.
  3. Identifies K0 (first strike at or below F) — used by the VIX formula.
  4. Selects OTM option implied volatilities at each available strike:
       • puts for K  < K0
       • calls for K > K0
       • average of call and put at K = K0
  5. Fits a cubic spline through (strike, IV) pairs.
  6. Returns an IVSurface object — a callable that maps any K ∈ [min, max]
     to its interpolated IV.  Extrapolation is strictly forbidden.

Naming convention
-----------------
  near-term surface  :  DTE closest to 30 from below (< 30 days)
  next-term surface  :  DTE closest to 30 from above (≥ 30 days)

  When the smallest available DTE is already ≥ 30, both near and next are
  set to the same (only) available expiry so Step 3 can still run.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


# ============================================================================
# IVSurface  —  the callable surface object returned by this module
# ============================================================================

@dataclass
class IVSurface:
    """
    A cubic-spline implied-volatility surface for one (date, expiration) pair.

    After construction the object is callable:

        surface = IVSurface(...)
        iv = surface(4300.0)   # IV at strike 4300

    Attributes
    ----------
    date        : pd.Timestamp  – the trading date
    expiration  : pd.Timestamp  – the option expiration date
    dte         : float  – calendar days to expiration at quote time
    T           : float  – time to expiration in years  (dte / 365)
    forward     : float  – forward index level F derived from put-call parity
    k0          : float  – first strike at or below F  (CBOE K0)
    strikes     : np.ndarray  – sorted array of strikes used in the spline fit
    ivs         : np.ndarray  – corresponding OTM implied volatilities
    _spline     : CubicSpline – the fitted interpolator (private)

    Properties
    ----------
    min_strike, max_strike : bounds of the fitted domain
    """

    date:       pd.Timestamp
    expiration: pd.Timestamp
    dte:        float
    T:          float
    forward:    float
    k0:         float
    strikes:    np.ndarray
    ivs:        np.ndarray
    _spline:    object          # CubicSpline; typed as object for dataclass

    # ------------------------------------------------------------------
    # Domain bounds
    # ------------------------------------------------------------------

    @property
    def min_strike(self) -> float:
        return float(self.strikes[0])

    @property
    def max_strike(self) -> float:
        return float(self.strikes[-1])

    # ------------------------------------------------------------------
    # Evaluation (scalar)
    # ------------------------------------------------------------------

    def __call__(self, K: float) -> float:
        """
        Return interpolated IV at strike K.

        Parameters
        ----------
        K : float  – the strike to evaluate.

        Returns
        -------
        float  – annualised IV (decimal, e.g. 0.18 = 18%).
                 np.nan is returned for any K outside [min_strike, max_strike].

        Notes
        -----
        We enforce a floor of config.MIN_IV (0.001) so that the output is
        always strictly positive — preventing division-by-zero in downstream
        variance calculations.
        """
        if K < self.min_strike or K > self.max_strike:
            return np.nan
        return float(max(self._spline(K), config.MIN_IV))

    # ------------------------------------------------------------------
    # Evaluation (vectorised)
    # ------------------------------------------------------------------

    def iv_array(self, strikes: np.ndarray) -> np.ndarray:
        """
        Vectorised evaluation of IV over an array of strikes.

        Entries outside the fitted domain are returned as np.nan.

        Parameters
        ----------
        strikes : np.ndarray  – 1-D array of strike prices.

        Returns
        -------
        np.ndarray  – corresponding IV values, same length as *strikes*.
        """
        out = np.full(len(strikes), np.nan, dtype=float)
        inside = (strikes >= self.min_strike) & (strikes <= self.max_strike)
        if inside.any():
            vals = self._spline(strikes[inside])
            out[inside] = np.maximum(vals, config.MIN_IV)
        return out

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def to_series(self, num_points: int = 300) -> pd.Series:
        """
        Evaluate the surface on a uniform strike grid.

        Useful for plotting, visual inspection, and comparison against raw
        market quotes.

        Parameters
        ----------
        num_points : int  – number of grid points across [min_strike, max_strike].

        Returns
        -------
        pd.Series  – index = strike, values = IV.
        """
        grid = np.linspace(self.min_strike, self.max_strike, num_points)
        return pd.Series(self.iv_array(grid), index=grid, name="IV")

    def __repr__(self) -> str:
        return (
            f"IVSurface(date={self.date.date()}, "
            f"exp={self.expiration.date()}, "
            f"DTE={self.dte:.0f}, "
            f"F={self.forward:.2f}, "
            f"K0={self.k0:.0f}, "
            f"strikes={len(self.strikes)}, "
            f"K_range=[{self.min_strike:.0f},{self.max_strike:.0f}])"
        )


# ============================================================================
# Forward price  —  derived from put-call parity
# ============================================================================

def _compute_forward_price(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    dte: float,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> Tuple[float, float]:
    """
    Derive the forward index level F and time-to-expiration T using
    put-call parity.

    Theory
    ------
    For European options on a non-dividend paying asset:

        C - P = e^{-rT} * (F - K)

    Rearranging:
        F = K + e^{rT} * (C - P)

    We compute (C - P) for all strikes where both a call and a put exist,
    then find the strike K* where |C - P| is minimised — this is the strike
    closest to the forward (per CBOE methodology).  F is then:

        F = K* + e^{rT} * (C(K*) - P(K*))

    This approach is robust to dividend effects and does not require an
    independent estimate of the dividend yield.

    Parameters
    ----------
    calls, puts : pd.DataFrame
        Must contain columns 'strike' and 'mid'.
    dte : float
        Calendar days to expiration.
    risk_free_rate : float
        Annualised continuously-compounded rate.  Defaults to config value.

    Returns
    -------
    (F, T) : (float, float)
        F  – forward price.
        T  – time to expiration in years (dte / 365).

    Raises
    ------
    ValueError
        If fewer than 2 (call, put) pairs share the same strike.
    """
    T        = dte / 365.0
    grow     = np.exp(risk_free_rate * T)   # e^{rT}
    discount = 1.0 / grow                   # e^{-rT}

    # ── Merge calls and puts on the same strike ───────────────────────────────
    merged = (
        calls[["strike", "mid"]].rename(columns={"mid": "C"})
        .merge(
            puts[["strike", "mid"]].rename(columns={"mid": "P"}),
            on="strike",
            how="inner",
        )
        .copy()
    )

    if len(merged) < 2:
        raise ValueError(
            f"Only {len(merged)} (call, put) pair(s) share the same strike. "
            "Cannot compute forward price reliably."
        )

    merged["C_minus_P"] = merged["C"] - merged["P"]

    # ── CBOE: use the strike with minimum |C - P| ─────────────────────────────
    idx_star = merged["C_minus_P"].abs().idxmin()
    row      = merged.loc[idx_star]

    K_star       = float(row["strike"])
    C_minus_P_K  = float(row["C_minus_P"])

    F = K_star + grow * C_minus_P_K

    log.debug(
        "  Forward: K*=%.0f, C-P=%.4f, r=%.3f%%, T=%.4f yr  →  F=%.2f",
        K_star, C_minus_P_K, risk_free_rate * 100, T, F,
    )
    return float(F), float(T)


# ============================================================================
# OTM option selection  —  puts below K0, calls above K0
# ============================================================================

def _select_otm_iv(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    forward: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Build the (strike, IV) set used in the spline fit using OTM options only.

    CBOE convention (VIX white paper, §1):
      • For K < K0  : use the **put** implied volatility
      • For K > K0  : use the **call** implied volatility
      • At K = K0   : average the call IV and the put IV

    where K0 is the first strike at or below F.

    This convention ensures we use the more liquid, out-of-the-money side
    at every strike and avoids double-counting.

    Parameters
    ----------
    calls, puts : pd.DataFrame
        Must contain columns 'strike' and 'implied_vol'.
    forward : float
        Forward price F derived from put-call parity.

    Returns
    -------
    (k0, strikes_array, ivs_array)
        k0      – first strike at or below F
        strikes – 1-D sorted np.ndarray
        ivs     – corresponding OTM IVs
    """
    # ── Identify K0 ───────────────────────────────────────────────────────────
    # K0 = largest strike ≤ F.
    all_strikes = np.union1d(calls["strike"].values, puts["strike"].values)
    candidates  = all_strikes[all_strikes <= forward]

    if len(candidates) == 0:
        # Entire strike range is above F — use the smallest available strike
        log.warning("All strikes are above F=%.2f; using min_strike as K0.", forward)
        k0 = float(all_strikes.min())
    else:
        k0 = float(candidates.max())

    log.debug("  K0=%.0f  (F=%.2f)", k0, forward)

    # ── Slice OTM puts and calls ──────────────────────────────────────────────
    put_otm  = puts.loc[puts["strike"]  < k0, ["strike", "implied_vol"]].copy()
    call_otm = calls.loc[calls["strike"] > k0, ["strike", "implied_vol"]].copy()

    # ── Handle K0 strike: average call and put if both exist ─────────────────
    call_at_k0 = calls.loc[calls["strike"] == k0, "implied_vol"]
    put_at_k0  = puts.loc[puts["strike"]  == k0, "implied_vol"]

    if len(call_at_k0) > 0 and len(put_at_k0) > 0:
        atm_iv = (float(call_at_k0.iloc[0]) + float(put_at_k0.iloc[0])) / 2.0
    elif len(call_at_k0) > 0:
        atm_iv = float(call_at_k0.iloc[0])
    elif len(put_at_k0) > 0:
        atm_iv = float(put_at_k0.iloc[0])
    else:
        # K0 not available in either side — the OTM arrays will still cover it
        # if there are adjacent strikes on both sides.
        log.debug("  K0=%.0f has no exact call or put quote; skipping K0 row.", k0)
        atm_iv = None

    # ── Assemble the combined OTM IV table ───────────────────────────────────
    parts = [put_otm, call_otm]
    if atm_iv is not None:
        k0_row = pd.DataFrame({"strike": [k0], "implied_vol": [atm_iv]})
        parts  = [put_otm, k0_row, call_otm]

    combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("strike")
        .sort_values("strike")
    )

    strikes_arr = combined["strike"].values.astype(float)
    ivs_arr     = combined["implied_vol"].values.astype(float)

    return k0, strikes_arr, ivs_arr


# ============================================================================
# Spline fitting
# ============================================================================

def _fit_cubic_spline(strikes: np.ndarray, ivs: np.ndarray) -> CubicSpline:
    """
    Fit a natural cubic spline through the (strike, IV) data points.

    Why cubic spline?
    -----------------
    • Passes exactly through every observed quote (no smoothing, s=0).
    • Twice differentiable — the second derivative (convexity w.r.t. strike)
      needed in Step 4 is well-defined.
    • The "not-a-knot" boundary condition (scipy default) avoids artificial
      oscillations at the endpoints.
    • Faster and more stable than UnivariateSpline for moderate point counts.

    Parameters
    ----------
    strikes : np.ndarray  – strictly increasing array of strikes.
    ivs     : np.ndarray  – corresponding IV values (same length).

    Returns
    -------
    CubicSpline object  (callable).

    Raises
    ------
    ValueError  if fewer than 2 (strike, IV) pairs are supplied.
    """
    if len(strikes) < 2:
        raise ValueError(
            f"Need at least 2 (strike, IV) pairs to fit a spline; "
            f"got {len(strikes)}."
        )

    # Ensure strictly increasing strikes (duplicates would break the spline)
    _, unique_idx = np.unique(strikes, return_index=True)
    strikes = strikes[unique_idx]
    ivs     = ivs[unique_idx]

    return CubicSpline(strikes, ivs, bc_type="not-a-knot", extrapolate=False)


# ============================================================================
# Single-expiry surface builder
# ============================================================================

def _build_surface_for_expiry(
    group: pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> Optional[IVSurface]:
    """
    Build an IVSurface for a single (date, expiration) options slice.

    Parameters
    ----------
    group : pd.DataFrame
        Rows for one (date, expiration) pair.
        Required columns: date, expiration, dte, strike, option_type,
                          implied_vol, mid.
    risk_free_rate : float

    Returns
    -------
    IVSurface on success, None if the group fails any quality gate.

    Quality gates
    -------------
    • At least config.MIN_STRIKES_REQUIRED total strikes after OTM selection.
    • At least config.MIN_SURFACE_STRIKES_EACH_SIDE strikes on each side of K0.
    • All IVs strictly positive after OTM selection.
    """
    date       = group["date"].iloc[0]
    expiration = group["expiration"].iloc[0]
    dte        = float(group["dte"].iloc[0])

    tag = f"{date.date()} / exp={expiration.date()} / DTE={dte:.0f}"

    # ── Split calls and puts ──────────────────────────────────────────────────
    calls = group.loc[group["option_type"] == "C", ["strike", "implied_vol", "mid"]]
    puts  = group.loc[group["option_type"] == "P", ["strike", "implied_vol", "mid"]]

    if calls.empty or puts.empty:
        log.warning("[%s] Missing calls or puts — skipping.", tag)
        return None

    # ── Forward price ─────────────────────────────────────────────────────────
    try:
        F, T = _compute_forward_price(calls, puts, dte, risk_free_rate)
    except ValueError as exc:
        log.warning("[%s] Forward price failed: %s — skipping.", tag, exc)
        return None

    if not np.isfinite(F) or F <= 0:
        log.warning("[%s] Invalid forward F=%.4f — skipping.", tag, F)
        return None

    # ── OTM IV selection ──────────────────────────────────────────────────────
    k0, strikes, ivs = _select_otm_iv(calls, puts, F)

    # ── Quality gate: minimum strike count ────────────────────────────────────
    if len(strikes) < config.MIN_STRIKES_REQUIRED:
        log.warning(
            "[%s] Only %d OTM strikes after selection (need %d) — skipping.",
            tag, len(strikes), config.MIN_STRIKES_REQUIRED,
        )
        return None

    # ── Quality gate: each side of K0 must have enough points ─────────────────
    n_put_side  = int((strikes < k0).sum())
    n_call_side = int((strikes > k0).sum())
    if n_put_side < config.MIN_SURFACE_STRIKES_EACH_SIDE:
        log.warning(
            "[%s] Only %d OTM put strikes (need %d) — skipping.",
            tag, n_put_side, config.MIN_SURFACE_STRIKES_EACH_SIDE,
        )
        return None
    if n_call_side < config.MIN_SURFACE_STRIKES_EACH_SIDE:
        log.warning(
            "[%s] Only %d OTM call strikes (need %d) — skipping.",
            tag, n_call_side, config.MIN_SURFACE_STRIKES_EACH_SIDE,
        )
        return None

    # ── Quality gate: no non-positive IVs ────────────────────────────────────
    bad_iv_mask = ivs <= 0
    if bad_iv_mask.any():
        n_bad = bad_iv_mask.sum()
        log.debug("[%s] Dropping %d zero/negative IVs from OTM set.", tag, n_bad)
        strikes = strikes[~bad_iv_mask]
        ivs     = ivs[~bad_iv_mask]
        if len(strikes) < config.MIN_STRIKES_REQUIRED:
            log.warning("[%s] Too few strikes after IV cleanup — skipping.", tag)
            return None

    # ── Fit spline ────────────────────────────────────────────────────────────
    try:
        spline = _fit_cubic_spline(strikes, ivs)
    except ValueError as exc:
        log.warning("[%s] Spline fitting failed: %s — skipping.", tag, exc)
        return None

    surface = IVSurface(
        date       = date,
        expiration = expiration,
        dte        = dte,
        T          = T,
        forward    = F,
        k0         = k0,
        strikes    = strikes,
        ivs        = ivs,
        _spline    = spline,
    )

    log.debug("[%s] %s", tag, repr(surface))
    return surface


# ============================================================================
# Near-term / next-term expiry selection
# ============================================================================

def _select_near_next(
    expiry_surfaces: Dict[pd.Timestamp, IVSurface],
) -> Tuple[Optional[IVSurface], Optional[IVSurface]]:
    """
    Given a dict of {expiration → IVSurface} for a single trading date,
    return (near_term, next_term) following the CBOE VIX convention.

    CBOE definition
    ---------------
    near-term : the expiration with DTE closest to 30 from below (DTE < 30).
                If no expiration with DTE < 30 exists, use the smallest DTE.
    next-term : the expiration with DTE closest to 30 from above (DTE ≥ 30).
                If no expiration with DTE ≥ 30 exists, use the largest DTE.

    When near == next (only one expiration available), Step 3 will use a
    single-maturity VIX estimate with a logged warning.

    Returns
    -------
    (near_surface, next_surface)  –  either may be None if no surface was
    successfully built for that maturity.
    """
    if not expiry_surfaces:
        return None, None

    # Sort by DTE ascending
    sorted_surfaces = sorted(expiry_surfaces.values(), key=lambda s: s.dte)

    below_30 = [s for s in sorted_surfaces if s.dte <  30.0]
    above_30 = [s for s in sorted_surfaces if s.dte >= 30.0]

    near = below_30[-1] if below_30 else sorted_surfaces[0]
    nxt  = above_30[0]  if above_30 else sorted_surfaces[-1]

    return near, nxt


# ============================================================================
# Top-level builder
# ============================================================================

def build_daily_surfaces(
    options: pd.DataFrame,
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> Dict[pd.Timestamp, Dict[str, IVSurface]]:
    """
    Build IV surfaces for every trading date in the options dataset.

    For each date, surfaces are built for ALL expirations in the DTE window,
    then the two CBOE-relevant ones (near-term and next-term) are selected.

    Parameters
    ----------
    options : pd.DataFrame
        Cleaned long-format options output from Step 1
        (data_ingestion.load_options_data).
    risk_free_rate : float
        Annualised continuously-compounded risk-free rate.

    Returns
    -------
    dict : {date → {'near': IVSurface, 'next': IVSurface}}

        Both 'near' and 'next' keys are always present.  If a surface
        could not be built (insufficient data), the value is None.

        Dates where BOTH near and next surfaces failed are excluded from
        the result dict.

    Processing pipeline (per date)
    --------------------------------
    1.  Group by (date, expiration).
    2.  For each (date, expiration) group:
          a.  Compute forward F and K0 via put-call parity.
          b.  Select OTM IV at each strike.
          c.  Fit cubic spline.
    3.  Select near-term and next-term surfaces.
    4.  Store in result dict under the trading date key.
    """
    log.info("=" * 60)
    log.info("STEP 2 — Building IV surfaces")
    log.info("Dates: %d unique trading dates", options["date"].nunique())
    log.info("=" * 60)

    result: Dict[pd.Timestamp, Dict[str, IVSurface]] = {}
    dates_ok = 0
    dates_skipped = 0

    # ── Iterate over all (date, expiration) groups ────────────────────────────
    # We build a surface for every expiry in the DTE window, then select
    # the two that matter for VIX.
    grouped = options.groupby(["date", "expiration"], sort=True)
    n_groups = len(grouped)
    log.info("Processing %d (date, expiration) groups…", n_groups)

    # Collect all surfaces keyed by (date, expiration)
    all_surfaces: Dict[Tuple, IVSurface] = {}
    for (date, expiry), group in grouped:
        surf = _build_surface_for_expiry(group, risk_free_rate)
        if surf is not None:
            all_surfaces[(date, expiry)] = surf

    log.info(
        "Surfaces built: %d / %d groups succeeded",
        len(all_surfaces), n_groups,
    )

    # ── For each trading date select near and next ────────────────────────────
    for date in sorted(options["date"].unique()):
        date = pd.Timestamp(date)

        # Gather all surfaces for this date
        expiry_map: Dict[pd.Timestamp, IVSurface] = {
            expiry: all_surfaces[(date, expiry)]
            for (d, expiry) in all_surfaces
            if d == date
        }

        if not expiry_map:
            log.warning("[%s] No usable surfaces — date skipped.", date.date())
            dates_skipped += 1
            continue

        near, nxt = _select_near_next(expiry_map)

        if near is None and nxt is None:
            log.warning("[%s] Both near and next are None — date skipped.", date.date())
            dates_skipped += 1
            continue

        result[date] = {"near": near, "next": nxt}

        log.debug(
            "[%s] near=%s  next=%s",
            date.date(),
            repr(near) if near else "None",
            repr(nxt)  if nxt  else "None",
        )
        dates_ok += 1

    log.info(
        "Step 2 complete: %d dates with surfaces, %d dates skipped.",
        dates_ok, dates_skipped,
    )
    return result


# ============================================================================
# Diagnostic helper
# ============================================================================

def print_surface_summary(
    surfaces: Dict[pd.Timestamp, Dict[str, IVSurface]],
) -> None:
    """
    Print a human-readable summary of the built surfaces.

    Reports per-date: near/next DTE, forward price, K0, number of strikes,
    and the ATM IV (evaluated at the forward price).
    """
    print("\n" + "=" * 78)
    print("IV SURFACE SUMMARY")
    print("=" * 78)
    fmt = "{:12s} {:6s} {:6s} {:8s} {:6s} {:8s} {:8s} {:6s}"
    print(fmt.format("Date", "Term", "DTE", "Forward", "K0", "ATM IV", "K-range", "#K"))
    print("-" * 78)

    for date in sorted(surfaces.keys()):
        for term in ("near", "next"):
            surf = surfaces[date].get(term)
            if surf is None:
                print(fmt.format(str(date.date()), term, "—", "—", "—", "—", "—", "—"))
                continue
            atm_iv = surf(surf.forward)
            k_range = f"{surf.min_strike:.0f}-{surf.max_strike:.0f}"
            print(fmt.format(
                str(date.date()),
                term,
                f"{surf.dte:.0f}",
                f"{surf.forward:.2f}",
                f"{surf.k0:.0f}",
                f"{atm_iv:.4f}" if np.isfinite(atm_iv) else "nan",
                k_range,
                str(len(surf.strikes)),
            ))

    print("=" * 78)
    print(f"Total dates: {len(surfaces)}")

    # ── ATM IV cross-section (first 5 dates) ─────────────────────────────────
    print("\nATM IV (near-term) first 5 dates:")
    for date in sorted(surfaces.keys())[:5]:
        surf = surfaces[date].get("near")
        if surf:
            atm = surf(surf.forward)
            print(f"  {date.date()}  F={surf.forward:.2f}  ATM_IV={atm:.4f} "
                  f"({atm*100:.2f}%)  DTE={surf.dte:.0f}")
    print("=" * 78)
