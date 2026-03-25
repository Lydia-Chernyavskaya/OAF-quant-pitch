"""
src/portfolio.py  –  Step 6: Portfolio Engine
==============================================
Simulates a daily-rebalanced vol strategy for each of the three signal
families produced in Step 5, then builds an equal-weight composite.

Instrument model
─────────────────
Each signal is treated as a position in a 30-day implied-vol product whose
daily P&L is proportional to the change in the synthetic VIX:

    P&L_gross[t]  =  −position[t−1]  ×  ΔVIX_synth[t]  ×  dollar_per_pt

Sign convention (carried over from Step 5):
    position = +1  →  SHORT vol  →  profit when VIX falls   (ΔVIX < 0)
    position = −1  →  LONG  vol  →  profit when VIX rises   (ΔVIX > 0)
    position =  0  →  flat / no trade

dollar_per_pt  = INITIAL_CAPITAL × POSITION_SIZE_NOTIONAL / 100

This gives each strategy a risk of 1% of capital per VIX point, which is
equivalent to holding ~67 VIX futures contracts on a $1 M book
(at VIX ≈ 15, 1 contract = $1,000 / point = $15,000 notional).

Position mechanics
───────────────────
Signal at end of day t → position entered at close of day t.
P&L on day t+1 = −position[t] × ΔVIX[t+1] × dollar_per_pt

In the DataFrame this is implemented as:
    lag_position[t] = position[t−1]          (1-day look-back shift)
    pnl_gross[t]    = −lag_position[t] × ΔVIX[t] × dollar_per_pt

Transaction costs
──────────────────
A cost is incurred whenever the position changes:

    TC[t]  =  |position[t−1] − position[t−2]|  ×  dollar_per_pt  ×  TC_BPS / 10000

    With TC_BPS = 2 and dollar_per_pt = $10,000:  $2 per unit change.

Equal-weight portfolio
───────────────────────
The five individual strategies (PSMR, CD_0.50, CD_0.60, CD_0.70, SSD) are
each run with the full INITIAL_CAPITAL.  The EW composite is then:

    EW_daily_return[t]  =  mean( daily_return_strategy_i[t]  for i in strategies )

This is equivalent to allocating INITIAL_CAPITAL / 5 to each strategy.
"""

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)

# ── Strategy registry ─────────────────────────────────────────────────────────
# Maps human-readable names to the signal column names in the signals DataFrame.
STRATEGY_SIGNALS: OrderedDict = OrderedDict([
    ("PSMR",    "parallel_signal"),
    ("CD_0.50", "convexity_signal_0.50"),
    ("CD_0.60", "convexity_signal_0.60"),
    ("CD_0.70", "convexity_signal_0.70"),
    ("SSD",     "sticky_signal"),
])


# ============================================================================
# Core single-strategy engine
# ============================================================================

def _run_single_strategy(
    signal:       pd.Series,
    vix_synth:    pd.Series,
    dollar_per_pt: float,
    tc_per_unit:  float,
) -> pd.DataFrame:
    """
    Compute daily P&L, drawdown, and return for one signal series.

    Parameters
    ----------
    signal        : pd.Series indexed by date, values in {−1, 0, +1} / NaN.
                    NaN is treated as 0 (flat, no trade).
    vix_synth     : pd.Series of synthetic VIX levels (Step 3 output).
    dollar_per_pt : USD P&L per 1 VIX point per unit of position.
    tc_per_unit   : USD transaction cost per unit of position change.

    Returns
    -------
    pd.DataFrame with columns:
        signal, position, lag_position, delta_vix,
        pnl_gross, tc, pnl_net, daily_return,
        cum_pnl, cum_pnl_pct, drawdown_pct
    """
    # Treat NaN signal as flat
    position = signal.fillna(0.0)

    # Align vix_synth to the signal dates; compute daily VIX change
    vix = vix_synth.reindex(position.index)
    delta_vix = vix.diff()

    # 1-day lag: position entered at close of t is active over [t, t+1]
    lag_position = position.shift(1).fillna(0.0)

    # ── Gross P&L ─────────────────────────────────────────────────────────────
    pnl_gross = (-lag_position * delta_vix * dollar_per_pt).fillna(0.0)
    # delta_vix[0] = NaN (no previous close); with lag_position[0]=0 the P&L
    # is correctly 0, but 0×NaN=NaN in numpy, so we fill explicitly.

    # ── Transaction costs ─────────────────────────────────────────────────────
    # Charge TC on the day the position changes (based on lag_position change,
    # i.e. the day the old position was set).
    pos_change = lag_position.diff().abs()
    # First row: entering from zero
    pos_change.iloc[0] = abs(lag_position.iloc[0])
    tc = pos_change * tc_per_unit

    # ── Net P&L and returns ───────────────────────────────────────────────────
    pnl_net      = pnl_gross - tc
    daily_return = pnl_net / dollar_per_pt / 100  # as a fraction of capital

    cum_pnl     = pnl_net.cumsum()
    cum_pnl_pct = cum_pnl / (dollar_per_pt * 100)   # fraction of capital

    # ── Drawdown ──────────────────────────────────────────────────────────────
    cum_peak       = cum_pnl.cummax()
    drawdown       = cum_pnl - cum_peak           # always ≤ 0  (USD)
    drawdown_pct   = drawdown / (dollar_per_pt * 100)

    return pd.DataFrame({
        "signal":        position,
        "lag_position":  lag_position,
        "vix_synth":     vix,
        "delta_vix":     delta_vix,
        "pnl_gross":     pnl_gross,
        "tc":            tc,
        "pnl_net":       pnl_net,
        "daily_return":  daily_return,
        "cum_pnl":       cum_pnl,
        "cum_pnl_pct":   cum_pnl_pct,
        "drawdown_pct":  drawdown_pct,
    }, index=position.index)


# ============================================================================
# Full portfolio build
# ============================================================================

def build_portfolio(
    signals:          pd.DataFrame,
    vix_synth:        pd.Series,
    initial_capital:  float = config.INITIAL_CAPITAL,
    tc_bps:           float = config.TRANSACTION_COST_BPS,
    position_size:    float = config.POSITION_SIZE_NOTIONAL,
    strategy_signals: OrderedDict = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run every registered strategy through the portfolio engine, then build
    an equal-weight composite.

    Parameters
    ----------
    signals          : pd.DataFrame from build_signals() (Step 5).
    vix_synth        : pd.Series of synthetic VIX from build_vix_series() (Step 3).
    initial_capital  : starting capital in USD.
    tc_bps           : one-way transaction cost in basis points.
    position_size    : leverage multiplier (1.0 = 1× notional).
    strategy_signals : Optional OrderedDict mapping strategy name → signal column.
                       If None, uses the module-level STRATEGY_SIGNALS default.

    Returns
    -------
    dict mapping strategy name → pd.DataFrame (see _run_single_strategy).
    The key "EW" contains the equal-weight composite.

    Notes
    -----
    dollar_per_pt  = initial_capital × position_size / 100
    → At $1 M and 1×: each VIX point = $10,000 P&L (1 % of capital).

    tc_per_unit    = dollar_per_pt × tc_bps / 10,000
    → At $10K/pt and 2 bps: $2 per unit position change.
    """
    if strategy_signals is None:
        strategy_signals = STRATEGY_SIGNALS

    log.info("=" * 60)
    log.info("STEP 6 — Portfolio Simulation")
    log.info("=" * 60)
    log.info(
        "Capital: $%s | TC: %d bps | Size: %.1f×",
        f"{initial_capital:,.0f}", int(tc_bps), position_size,
    )

    dollar_per_pt = initial_capital * position_size / 100.0
    tc_per_unit   = dollar_per_pt * tc_bps / 10_000.0

    log.info(
        "dollar_per_pt = $%s  |  tc_per_unit = $%.2f",
        f"{dollar_per_pt:,.0f}", tc_per_unit,
    )

    results: Dict[str, pd.DataFrame] = {}

    for name, col in strategy_signals.items():
        if col not in signals.columns:
            log.warning("  Signal column '%s' not found — skipping %s.", col, name)
            continue

        df = _run_single_strategy(
            signal        = signals[col],
            vix_synth     = vix_synth,
            dollar_per_pt = dollar_per_pt,
            tc_per_unit   = tc_per_unit,
        )
        results[name] = df

        n_trades = int((df["lag_position"].diff().abs() > 0).sum())
        pnl_str  = f"${df['cum_pnl'].iloc[-1]:+,.0f}"
        log.info("  %-10s  net P&L=%12s  n_trades=%3d", name, pnl_str, n_trades)

    # ── Equal-weight composite ────────────────────────────────────────────────
    if results:
        ret_df = pd.DataFrame(
            {name: df["daily_return"] for name, df in results.items()}
        )
        ew_ret      = ret_df.mean(axis=1)                     # avg daily fraction
        ew_pnl_net  = ew_ret * initial_capital                # scale to $
        ew_cum      = ew_pnl_net.cumsum()
        ew_peak     = ew_cum.cummax()

        results["EW"] = pd.DataFrame({
            "daily_return":  ew_ret,
            "pnl_net":       ew_pnl_net,
            "cum_pnl":       ew_cum,
            "cum_pnl_pct":   ew_cum / initial_capital,
            "drawdown_pct":  (ew_cum - ew_peak) / initial_capital,
        }, index=ew_ret.index)

        ew_pnl_str = f"${results['EW']['cum_pnl'].iloc[-1]:+,.0f}"
        log.info(
            "  %-10s  net P&L=%12s  (avg of %d strategies)",
            "EW", ew_pnl_str, len(results) - 1,
        )

    return results


# ============================================================================
# Performance statistics
# ============================================================================

def _compute_stats(
    df:              pd.DataFrame,
    initial_capital: float,
    td_per_year:     int = config.TRADING_DAYS_PER_YEAR,
    rf_rate:         float = config.RISK_FREE_RATE,
) -> dict:
    """
    Compute summary performance statistics for one strategy DataFrame.

    Returns
    -------
    dict with keys:
        total_pnl, total_return_pct, ann_return_pct,
        sharpe, max_drawdown_pct, win_rate, n_trades, n_days
    """
    ret = df["daily_return"]
    n   = len(ret)

    if n == 0:
        return {k: np.nan for k in [
            "total_pnl", "total_return_pct", "ann_return_pct",
            "sharpe", "max_drawdown_pct", "win_rate", "n_trades", "n_days",
        ]}

    total_pnl       = float(df["cum_pnl"].iloc[-1])
    total_return    = total_pnl / initial_capital
    ann_return      = total_return * (td_per_year / n)

    # Sharpe (annualised, excess over rf_rate / 252)
    daily_rf    = rf_rate / td_per_year
    excess      = ret - daily_rf
    sharpe      = float(
        (excess.mean() / excess.std(ddof=1)) * np.sqrt(td_per_year)
        if excess.std(ddof=1) > 0 else np.nan
    )

    max_dd_pct  = float(df["drawdown_pct"].min())

    # Win rate: fraction of active (non-flat) days with positive net P&L
    if "lag_position" in df.columns:
        active_mask = df["lag_position"].abs() > 0
    else:
        active_mask = df["pnl_net"].abs() > 0
    active_days = df["pnl_net"][active_mask]
    win_rate    = float((active_days > 0).mean()) if len(active_days) > 0 else np.nan

    # Number of trades (position-change events)
    if "lag_position" in df.columns:
        n_trades = int((df["lag_position"].diff().abs() > 0).sum())
    else:
        n_trades = 0

    return {
        "total_pnl":        total_pnl,
        "total_return_pct": total_return * 100,
        "ann_return_pct":   ann_return * 100,
        "sharpe":           sharpe,
        "max_drawdown_pct": max_dd_pct * 100,
        "win_rate":         win_rate * 100 if not np.isnan(win_rate) else np.nan,
        "n_trades":         n_trades,
        "n_days":           n,
    }


def validate_portfolio(
    results:         Dict[str, pd.DataFrame],
    initial_capital: float = config.INITIAL_CAPITAL,
) -> Dict[str, dict]:
    """
    Compute and log performance statistics for each strategy.

    Returns
    -------
    dict mapping strategy name → stats dict.
    """
    log.info("─" * 60)
    log.info("PORTFOLIO VALIDATION")
    log.info("─" * 60)

    all_stats: Dict[str, dict] = {}
    for name, df in results.items():
        s = _compute_stats(df, initial_capital)
        all_stats[name] = s
        sharpe_s = f"{s['sharpe']:+.2f}" if not np.isnan(s["sharpe"]) else "  NaN"
        wr_s     = f"{s['win_rate']:.1f}%" if not np.isnan(s["win_rate"]) else "  NaN"
        log.info(
            "  %-10s  TotPnL=%12s  AnnRet=%+6.1f%%  "
            "Sharpe=%s  MaxDD=%6.1f%%  WinRate=%s  Trades=%3d",
            name,
            f"${s['total_pnl']:+,.0f}",
            s["ann_return_pct"],
            sharpe_s,
            s["max_drawdown_pct"],
            wr_s,
            s["n_trades"],
        )

    log.info("─" * 60)

    # Sanity checks
    for name, df in results.items():
        if "tc" in df.columns and (df["tc"] < 0).any():
            log.error("  [%s] Negative TC detected — bug in cost logic!", name)
        if "cum_pnl" in df.columns and df["cum_pnl"].isna().any():
            log.warning("  [%s] NaN values in cum_pnl.", name)

    return all_stats


# ============================================================================
# Print tables
# ============================================================================

def print_portfolio_table(
    results:         Dict[str, pd.DataFrame],
    initial_capital: float = config.INITIAL_CAPITAL,
    n_rows:          int   = 20,
) -> None:
    """
    Print a side-by-side daily cumulative P&L table (% of capital)
    for all strategies.
    """
    pct_cols = {}
    for name, df in results.items():
        if "cum_pnl_pct" in df.columns:
            pct_cols[name] = (df["cum_pnl_pct"] * 100).round(3)

    if not pct_cols:
        return

    table = pd.DataFrame(pct_cols)

    print("\n" + "=" * 80)
    print("PORTFOLIO — Cumulative P&L (% of capital)")
    print("=" * 80)
    print(table.head(n_rows).to_string(
        float_format=lambda x: f"{x:+7.3f}%"
    ))
    print("─" * 80)
    print(f"Final cumulative P&L:")
    for name, col in pct_cols.items():
        print(f"  {name:10s}  {col.iloc[-1]:+.3f}%  (${col.iloc[-1]/100 * initial_capital:+,.0f})")
    print("=" * 80)


def print_portfolio_summary(
    results:         Dict[str, pd.DataFrame],
    initial_capital: float = config.INITIAL_CAPITAL,
) -> None:
    """
    Print a compact one-line-per-strategy performance summary table.
    """
    rows = []
    for name, df in results.items():
        s = _compute_stats(df, initial_capital)
        rows.append({
            "Strategy":  name,
            "TotalPnL":  f"${s['total_pnl']:+,.0f}",
            "AnnRet%":   f"{s['ann_return_pct']:+.1f}%",
            "Sharpe":    f"{s['sharpe']:+.2f}" if not np.isnan(s['sharpe']) else "  NaN",
            "MaxDD%":    f"{s['max_drawdown_pct']:.1f}%",
            "WinRate%":  f"{s['win_rate']:.1f}%" if not np.isnan(s['win_rate']) else "  NaN",
            "Trades":    str(s['n_trades']),
        })

    tbl = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "=" * 72)
    print("PORTFOLIO SUMMARY — Step 6")
    print("=" * 72)
    print(tbl.to_string())
    print("─" * 72)
    print(f"  Capital: ${initial_capital:,.0f}  |  "
          f"TC: {config.TRANSACTION_COST_BPS} bps  |  "
          f"Size: {config.POSITION_SIZE_NOTIONAL}×  |  "
          f"VIX $/pt: ${initial_capital * config.POSITION_SIZE_NOTIONAL / 100:,.0f}")
    print("=" * 72)
