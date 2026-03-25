"""
src/signals.py  –  Step 5: Signal Generation
=============================================
Generates three distinct trading signals derived from the VIX decomposition
produced in Step 4.  Each signal exploits a different structural feature of
the implied-volatility surface.

Signal 1 — Parallel Shift Mean Reversion (PSMR)
─────────────────────────────────────────────────
The ATM implied-vol level a_t (the intercept of the local quadratic fit) is
mean-reverting over medium time horizons.  When the rolling z-score of a_t
exceeds ±threshold, implied vol is considered abnormally elevated or depressed
and we take a contrarian position.

    z_t  =  (a_t − μ_{t−W:t}) / σ_{t−W:t}

    signal_parallel = +1  (short vol)  when  z >  +threshold
    signal_parallel = −1  (long vol)   when  z <  −threshold
    signal_parallel =  0  otherwise

Signal 2 — Convexity Dominance (CD)
─────────────────────────────────────
On days when the convexity contribution |convexity_contribution| accounts for
more than θ of the total surface move |model_total|, the market is primarily
repricing tail risk.  The signal direction follows the sign of the convexity
change.

    convexity_ratio = |convexity_contribution| / (|model_total| + ε)

    signal_cd = +1  when  conv_ratio > θ  AND  conv_contribution > 0
    signal_cd = −1  when  conv_ratio > θ  AND  conv_contribution < 0
    signal_cd =  0  otherwise

    +1 : tail premium expanding  → sell tail risk (short downside convexity)
    −1 : tail premium contracting → buy tail risk (long downside convexity)

Three dominance thresholds θ ∈ {0.50, 0.60, 0.70} are evaluated; the best
is selected during walk-forward calibration (Step 8).

Signal 3 — Sticky Strike Deviation (SSD)
──────────────────────────────────────────
Under the sticky-strike convention, when the spot index moves from S_{t−1}
to S_t, IVs at fixed dollar strikes are unchanged.  The new ATM level should
therefore be whatever the old spline gives at the new reference strike K₀_t:

    a_sticky = surf_{t−1}(K₀_t)

    delta_a_obs = a_t − a_{t−1}            (observed ATM-IV change)
    delta_a_ss  = a_sticky − a_{t−1}       (sticky-strike predicted change)
    ratio       = delta_a_obs / delta_a_ss

    ratio > HIGH  →  surface overshot sticky-strike  →  mean reversion
                     signal = +sign(delta_a_obs)
                     (if vol spiked → short vol +1; if vol crashed → long vol −1)

    ratio < LOW   →  surface undershot sticky-strike  →  momentum
                     signal = −sign(delta_a_obs)
                     (if vol slowly rising → long vol −1 to follow; vice versa)

    else          →  no trade → 0

Sign convention throughout
──────────────────────────
    +1  →  SHORT implied vol  (sell VIX / sell straddle)
    −1  →  LONG  implied vol  (buy  VIX / buy  straddle)
     0  →  flat / no position
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.iv_surface import IVSurface

log = logging.getLogger(__name__)


# ============================================================================
# Signal 1 — Parallel Shift Mean Reversion (PSMR)
# ============================================================================

def compute_parallel_signal(
    decomp:    pd.DataFrame,
    window:    int   = config.PARALLEL_ZSCORE_WINDOW,
    threshold: float = config.PARALLEL_ZSCORE_ENTRY,
) -> pd.DataFrame:
    """
    Parallel Shift Mean Reversion (PSMR) signal.

    Rolling z-score of the ATM implied-vol level a_t over *window* trading
    days.  When |z| > threshold we expect reversion and take a contrarian
    position.

    Parameters
    ----------
    decomp    : pd.DataFrame output of build_decomposition() (Step 4).
    window    : rolling look-back window in trading days.
    threshold : |z| threshold that triggers a signal entry.

    Returns
    -------
    pd.DataFrame indexed by date, columns:
        z_parallel      – rolling z-score of a_t
        parallel_signal – {−1, 0, +1}
    """
    a = decomp["a_t"]

    # Allow half-window to fill in shorter datasets (Q4-2023 has only 63 days,
    # so a strict min_periods=60 would leave almost no signals).
    min_p = max(window // 2, 5)
    roll_mean = a.rolling(window, min_periods=min_p).mean()
    roll_std  = a.rolling(window, min_periods=min_p).std()

    z = (a - roll_mean) / roll_std.clip(lower=1e-6)

    signal = pd.Series(0.0, index=decomp.index)
    signal[z >  threshold] =  1.0   # elevated vol → short vol
    signal[z < -threshold] = -1.0   # depressed vol → long vol

    n_valid = z.notna().sum()
    log.info(
        "PSMR: window=%d days (min_periods=%d), threshold=%.1f σ | "
        "valid z-scores=%d/%d | short(+1)=%d  long(−1)=%d  flat(0)=%d",
        window, min_p, threshold,
        n_valid, len(decomp),
        (signal ==  1.0).sum(),
        (signal == -1.0).sum(),
        (signal ==  0.0).sum(),
    )

    return pd.DataFrame({"z_parallel": z, "parallel_signal": signal})


# ============================================================================
# Signal 2 — Convexity Dominance (CD)
# ============================================================================

def compute_convexity_signal(
    decomp:         pd.DataFrame,
    thresholds:     Optional[List[float]] = None,
    min_model_move: float = config.CD_MIN_MODEL_MOVE,
) -> pd.DataFrame:
    """
    Convexity Dominance (CD) signal.

    When the convexity component is the primary driver of the day's surface
    move (|conv_contrib| / |model_total| > θ), the market is repricing tail
    risk.  We take a directional position aligned with the convexity change.

    Parameters
    ----------
    decomp         : pd.DataFrame from build_decomposition().
    thresholds     : list of dominance-ratio thresholds to evaluate.
                     Default: config.CONVEXITY_DOM_THRESHOLD_VALUES.
    min_model_move : skip days where |model_total| < this (avoids noise on
                     quiet days where ratios are dominated by rounding).

    Returns
    -------
    pd.DataFrame indexed by date, columns:
        convexity_ratio             – |conv| / (|model_total| + ε)
        convexity_signal_0.50       – {−1, 0, +1} for θ = 0.50
        convexity_signal_0.60       – {−1, 0, +1} for θ = 0.60
        convexity_signal_0.70       – {−1, 0, +1} for θ = 0.70
    """
    if thresholds is None:
        thresholds = config.CONVEXITY_DOM_THRESHOLD_VALUES

    eps        = 1e-8
    total_abs  = decomp["model_total"].abs()
    conv       = decomp["convexity_contribution"]
    conv_ratio = conv.abs() / (total_abs + eps)
    meaningful = total_abs > min_model_move

    out = pd.DataFrame({"convexity_ratio": conv_ratio}, index=decomp.index)

    for theta in thresholds:
        dominant = conv_ratio > theta
        active   = dominant & meaningful

        sig = pd.Series(0.0, index=decomp.index)
        sig[active] = np.sign(conv[active]).astype(float)

        col = f"convexity_signal_{theta:.2f}"
        out[col] = sig

        log.info(
            "CD (θ=%.2f): active=%d/%d | short(+1)=%d  long(−1)=%d",
            theta, active.sum(), len(decomp),
            (sig ==  1.0).sum(),
            (sig == -1.0).sum(),
        )

    return out


# ============================================================================
# Signal 3 — Sticky Strike Deviation (SSD)
# ============================================================================

def compute_sticky_strike_signal(
    decomp:      pd.DataFrame,
    surfaces:    Dict,
    high:        float = config.STICKY_STRIKE_HIGH,
    low:         float = config.STICKY_STRIKE_LOW,
    min_ss_move: float = config.SSD_MIN_SS_MOVE,
) -> pd.DataFrame:
    """
    Sticky Strike Deviation (SSD) signal.

    Compares the observed ATM-IV change to what the sticky-strike model
    predicts given the spot move.  Overshoots (ratio > high) are faded;
    undershoots (ratio < low) are followed as momentum.

    Parameters
    ----------
    decomp      : pd.DataFrame from build_decomposition().
    surfaces    : Dict[date, {'near': IVSurface, 'next': IVSurface}]
                  from build_daily_surfaces() (Step 2).
    high        : ratio above which we fade the move (mean reversion).
    low         : ratio below which we follow the move (momentum).
    min_ss_move : minimum |delta_a_ss| required to compute the ratio
                  (avoids division by near-zero when spot barely moved).

    Returns
    -------
    pd.DataFrame indexed by date, columns:
        k0_t          – new ATM strike at date t
        a_sticky      – sticky-strike predicted ATM IV  (surf_{t-1}(K₀_t))
        delta_a_ss    – a_sticky − a_{t-1}  (predicted ATM change)
        ssd_ratio     – delta_a_obs / delta_a_ss
        sticky_signal – {−1, 0, +1, NaN}  (NaN when ratio is undefined)
    """
    # Index of sorted surface dates so we can look up the predecessor.
    surf_dates = sorted(surfaces.keys())
    surf_pos   = {d: i for i, d in enumerate(surf_dates)}

    records: List[dict] = []

    for date_t in decomp.index:
        row = decomp.loc[date_t]

        # ── Look up predecessor date ──────────────────────────────────────────
        pos = surf_pos.get(date_t)
        if pos is None or pos == 0:
            records.append(_ssd_nan_row(date_t))
            continue

        date_t1 = surf_dates[pos - 1]
        surf_t  = surfaces[date_t].get("near")
        surf_t1 = surfaces[date_t1].get("near")

        if surf_t is None or surf_t1 is None:
            records.append(_ssd_nan_row(date_t))
            continue

        k0_t        = surf_t.k0
        a_t         = float(row["a_t"])
        a_t1        = float(row["a_t1"])
        delta_a_obs = a_t - a_t1

        # ── Sticky-strike prediction ──────────────────────────────────────────
        # Evaluate the OLD spline at the NEW reference strike.
        # IVSurface.__call__ returns NaN when k0_t is outside the spline domain.
        a_sticky = surf_t1(k0_t)

        if np.isnan(a_sticky):
            log.debug(
                "[%s] K₀_t=%.1f outside old spline range [%.1f, %.1f]"
                " — SSD skipped.",
                date_t.date(), k0_t,
                surf_t1.min_strike, surf_t1.max_strike,
            )
            records.append(_ssd_nan_row(date_t, k0_t=k0_t))
            continue

        delta_a_ss = a_sticky - a_t1

        # ── Guard against near-zero denominator ──────────────────────────────
        if abs(delta_a_ss) < min_ss_move:
            records.append({
                "date": date_t, "k0_t": k0_t,
                "a_sticky": a_sticky, "delta_a_ss": delta_a_ss,
                "ssd_ratio": np.nan, "sticky_signal": np.nan,
            })
            continue

        ratio = delta_a_obs / delta_a_ss

        # ── Skip opposite-sign edge case ──────────────────────────────────────
        # When ratio < 0, obs and ss have opposite signs (anomaly — skip).
        if ratio < 0.0:
            records.append({
                "date": date_t, "k0_t": k0_t,
                "a_sticky": a_sticky, "delta_a_ss": delta_a_ss,
                "ssd_ratio": ratio, "sticky_signal": np.nan,
            })
            continue

        # ── Generate signal ───────────────────────────────────────────────────
        if ratio > high:
            # Overshoot: mean-revert against the observed move.
            # Vol spiked more than sticky-strike expects → short vol (+1).
            # Vol crashed more than sticky-strike expects → long vol (−1).
            sticky_signal = float(np.sign(delta_a_obs))
        elif ratio < low:
            # Undershoot: momentum — follow direction of the slow move.
            # Vol rising below sticky-strike pace → long vol (−1).
            # Vol falling below sticky-strike pace → short vol (+1).
            sticky_signal = float(-np.sign(delta_a_obs))
        else:
            sticky_signal = 0.0

        records.append({
            "date": date_t, "k0_t": k0_t,
            "a_sticky": a_sticky, "delta_a_ss": delta_a_ss,
            "ssd_ratio": ratio, "sticky_signal": sticky_signal,
        })

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)

    valid = df["sticky_signal"].notna()
    sig   = df["sticky_signal"].fillna(0.0)
    log.info(
        "SSD: high=%.1f, low=%.1f | valid=%d/%d | "
        "short(+1)=%d  long(−1)=%d  flat(0)=%d  NaN=%d",
        high, low,
        valid.sum(), len(df),
        (sig ==  1.0).sum(),
        (sig == -1.0).sum(),
        (sig ==  0.0).sum(),
        (~valid).sum(),
    )

    return df


def _ssd_nan_row(date_t: pd.Timestamp, k0_t: float = np.nan) -> dict:
    """Return a record with all SSD diagnostic fields set to NaN."""
    return {
        "date": date_t, "k0_t": k0_t,
        "a_sticky": np.nan, "delta_a_ss": np.nan,
        "ssd_ratio": np.nan, "sticky_signal": np.nan,
    }


# ============================================================================
# Combine all signals into one DataFrame
# ============================================================================

def build_signals(
    decomp:   pd.DataFrame,
    surfaces: Dict,
) -> pd.DataFrame:
    """
    Run all three signal generators and merge into a single DataFrame.

    Parameters
    ----------
    decomp   : pd.DataFrame from build_decomposition() (Step 4).
    surfaces : Dict from build_daily_surfaces() (Step 2).

    Returns
    -------
    pd.DataFrame indexed by date.  Key columns:

    From decomposition (passthrough):
        delta_vix
        a_t, b_t, c_t, delta_a, delta_b, delta_c   (TV backward-compat params)
        sticky_contribution
        parallel_contribution
        put_skew_contribution, call_skew_contribution, skew_contribution
        put_convexity_contribution, call_convexity_contribution, convexity_contribution
        higher_order_contribution
        model_total, residual

    Signal 1 (PSMR):
        z_parallel, parallel_signal

    Signal 2 (CD):
        convexity_ratio,
        convexity_signal_0.50, convexity_signal_0.60, convexity_signal_0.70

    Signal 3 (SSD):
        k0_t, a_sticky, delta_a_ss, ssd_ratio, sticky_signal
    """
    log.info("=" * 60)
    log.info("STEP 5 — Signal Generation")
    log.info("=" * 60)

    # ── Passthrough decomposition columns ─────────────────────────────────────
    passthrough_cols = [
        "delta_vix",
        "delta_a", "delta_b", "delta_c",
        "a_t", "b_t", "c_t",
        "sticky_contribution",
        "parallel_contribution",
        "put_skew_contribution", "call_skew_contribution", "skew_contribution",
        "put_convexity_contribution", "call_convexity_contribution", "convexity_contribution",
        "higher_order_contribution",
        "model_total", "residual",
    ]
    passthrough = decomp[[c for c in passthrough_cols if c in decomp.columns]].copy()

    # ── Compute signals ────────────────────────────────────────────────────────
    psmr = compute_parallel_signal(decomp)
    cd   = compute_convexity_signal(decomp)
    ssd  = compute_sticky_strike_signal(decomp, surfaces)

    # ── Merge ─────────────────────────────────────────────────────────────────
    signals = (passthrough
               .join(psmr, how="left")
               .join(cd,   how="left")
               .join(ssd,  how="left"))

    log.info(
        "Signal frame built: %d rows × %d columns.",
        len(signals), len(signals.columns),
    )
    return signals


# ============================================================================
# Validation
# ============================================================================

def validate_signals(signals: pd.DataFrame) -> dict:
    """
    Report signal statistics and cross-signal correlations.

    Checks
    ------
    • Signal values in {−1, 0, +1} (or NaN for SSD undefined days).
    • No signal is permanently zero (would indicate a broken generator).
    • Log pairwise correlations (diversification check).

    Returns
    -------
    dict with summary counts for downstream use.
    """
    log.info("─" * 60)
    log.info("SIGNAL VALIDATION")
    log.info("─" * 60)

    n = len(signals)

    # Collect per-signal stats
    sig_cols = {
        "PSMR":    "parallel_signal",
        "CD_0.50": "convexity_signal_0.50",
        "CD_0.60": "convexity_signal_0.60",
        "CD_0.70": "convexity_signal_0.70",
        "SSD":     "sticky_signal",
    }

    result = {"n_days": n}
    sig_series: Dict[str, pd.Series] = {}

    for label, col in sig_cols.items():
        if col not in signals.columns:
            continue
        s = signals[col]
        s_filled = s.fillna(0.0)

        n_active = int((s_filled != 0).sum())
        n_short  = int((s_filled ==  1.0).sum())
        n_long   = int((s_filled == -1.0).sum())
        n_nan    = int(s.isna().sum())

        log.info(
            "  %-10s  active=%3d/%3d (%4.0f%%)  short=%3d  long=%3d  NaN=%3d",
            label, n_active, n,
            100 * n_active / max(n, 1),
            n_short, n_long, n_nan,
        )

        if n_active == 0:
            log.warning("  ⚠  %s generated ZERO active signals — check parameters.", label)

        result[f"n_{label}_active"] = n_active
        sig_series[label] = s_filled

    # ── Cross-signal correlations ─────────────────────────────────────────────
    if len(sig_series) > 1:
        corr_df = pd.DataFrame(sig_series).corr()
        log.info("  Signal pairwise correlations:")
        labels = list(sig_series.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                log.info("    %s ↔ %s : %+.3f", a, b, corr_df.loc[a, b])

    log.info("─" * 60)
    return result


# ============================================================================
# Diagnostic print
# ============================================================================

def print_signals_table(signals: pd.DataFrame, n_rows: int = 20) -> None:
    """
    Print a human-readable table showing each day's signals alongside the
    key decomposition quantities.
    """
    desired_cols = [
        "delta_vix",
        "a_t",
        "z_parallel",
        "parallel_signal",
        "convexity_ratio",
        "convexity_signal_0.60",
        "ssd_ratio",
        "sticky_signal",
    ]
    short = {
        "delta_vix":             "ΔVIX",
        "a_t":                   "ATM_IV",
        "z_parallel":            "z_par",
        "parallel_signal":       "PSMR",
        "convexity_ratio":       "ConvRatio",
        "convexity_signal_0.60": "CD_0.60",
        "ssd_ratio":             "SSD_ratio",
        "sticky_signal":         "SSD",
    }

    cols = [c for c in desired_cols if c in signals.columns]
    disp = signals[cols].rename(columns=short)

    def _fmt(x: float) -> str:
        if isinstance(x, float) and np.isnan(x):
            return "   NaN  "
        if abs(x) < 100:
            return f"{x:+8.4f}"
        return f"{x:8.2f}"

    print("\n" + "=" * 90)
    print("VIX DECOMPOSITION — Trading Signals  (Step 5)")
    print("=" * 90)
    print(disp.head(n_rows).to_string(float_format=_fmt))
    print("─" * 90)

    # Signal-count footer
    footer_map = [
        ("parallel_signal",       "PSMR   "),
        ("convexity_signal_0.50", "CD_0.50"),
        ("convexity_signal_0.60", "CD_0.60"),
        ("convexity_signal_0.70", "CD_0.70"),
        ("sticky_signal",         "SSD    "),
    ]
    for col, label in footer_map:
        if col not in signals.columns:
            continue
        s = signals[col].fillna(0.0)
        print(
            f"  {label}  short(+1)={int((s > 0).sum()):3d}  "
            f"long(−1)={int((s < 0).sum()):3d}  "
            f"flat(0)={int((s == 0).sum()):3d}"
        )
    print("=" * 90)
