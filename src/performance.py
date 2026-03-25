"""
src/performance.py  –  Step 7: Performance Metrics
====================================================
Computes a comprehensive set of performance and risk metrics for each
strategy produced in Step 6.

Metrics computed
────────────────
Return metrics:
    total_pnl           – cumulative net P&L in USD
    total_return_pct    – total return as % of initial capital
    ann_return_pct      – annualised return (simple, scaled by 252/n_days)
    ann_vol_pct         – annualised daily-return volatility

Risk-adjusted:
    sharpe              – annualised Sharpe ratio  (excess / vol × √252)
    sortino             – annualised Sortino ratio  (excess / downside_vol × √252)
                          where downside_vol uses only sub-zero excess returns
    calmar              – ann_return / |max_drawdown|  (ratio, not %)

Drawdown:
    max_drawdown_pct    – worst peak-to-trough drawdown (negative, %)
    max_dd_duration     – longest consecutive days below the prior peak

Trade statistics:
    n_trades            – number of position-change events
    win_rate_pct        – % of active days with positive net P&L
    avg_win_usd         – average USD P&L on winning days
    avg_loss_usd        – average USD P&L on losing days (negative)
    profit_factor       – total gains / total losses (> 1 is profitable)

Tail risk (empirical, 1-day horizon):
    var_95_pct          – 5th percentile of daily_return (i.e. 1-day VaR at 95 %)
    cvar_95_pct         – mean of daily_return in worst 5 % of days (ES)

Diagnostic:
    n_days              – total calendar days in the sample
    n_active_days       – days the strategy held a non-zero position

Rolling metrics:
    rolling_sharpe      – pd.Series of rolling Sharpe (window = ROLLING_SHARPE_WINDOW,
                          min_periods = max(5, window//4))

Convention
──────────
    daily_return  in the portfolio DataFrames is already pnl_net / initial_capital
    (a plain fraction, e.g. 0.01 = 1 % of capital).  All metric functions work on
    this series.  Dollar amounts are rescaled using initial_capital.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _max_drawdown_duration(cum_pnl: pd.Series) -> int:
    """
    Return the longest consecutive run of days where cumulative P&L
    was strictly below its running maximum (i.e. in a drawdown).
    """
    cum_max    = cum_pnl.cummax()
    in_drawdown = (cum_pnl < cum_max).astype(int)

    # Label consecutive runs via a cumulative sum of "entering new segment"
    block = (in_drawdown.diff().fillna(in_drawdown) != 0).cumsum()
    durations = in_drawdown.groupby(block).sum()
    return int(durations.max()) if len(durations) > 0 else 0


def _sortino_ratio(
    ret:        pd.Series,
    rf_rate:    float = 0.0,
    td_per_year: int  = 252,
) -> float:
    """
    Annualised Sortino ratio.

        Sortino = (mean_excess × T) / downside_deviation

    where downside_deviation = sqrt( mean( min(excess_t, 0)² ) ) × sqrt(T)
    and T = td_per_year.

    This is the Estrada (2000) semi-variance formulation, which is more
    robust than using only std of losing-day returns.
    """
    daily_rf = rf_rate / td_per_year
    excess   = ret - daily_rf

    semi_var = np.mean(np.minimum(excess, 0.0) ** 2)
    if semi_var <= 0:
        return np.nan

    downside_vol = np.sqrt(semi_var) * np.sqrt(td_per_year)
    ann_excess   = excess.mean() * td_per_year
    return float(ann_excess / downside_vol)


# ============================================================================
# Per-strategy metric computation
# ============================================================================

def compute_metrics(
    df:              pd.DataFrame,
    initial_capital: float = config.INITIAL_CAPITAL,
    td_per_year:     int   = config.TRADING_DAYS_PER_YEAR,
    rf_rate:         float = config.RISK_FREE_RATE,
) -> dict:
    """
    Compute the full metric suite for one strategy DataFrame.

    Parameters
    ----------
    df              : pd.DataFrame from build_portfolio() (Step 6).
                      Must contain 'daily_return', 'cum_pnl', 'pnl_net',
                      'drawdown_pct'.  Optionally 'lag_position'.
    initial_capital : basis for dollar rescaling.
    td_per_year     : trading days per year for annualisation.
    rf_rate         : annual risk-free rate (continuous, decimal).

    Returns
    -------
    dict of scalar metrics (all floats or ints).
    """
    ret = df["daily_return"]
    n   = len(ret)

    if n == 0 or "cum_pnl" not in df.columns:
        return {k: np.nan for k in [
            "total_pnl", "total_return_pct", "ann_return_pct", "ann_vol_pct",
            "sharpe", "sortino", "calmar",
            "max_drawdown_pct", "max_dd_duration",
            "n_trades", "win_rate_pct", "avg_win_usd", "avg_loss_usd",
            "profit_factor", "var_95_pct", "cvar_95_pct",
            "n_days", "n_active_days",
        ]}

    # ── Return metrics ────────────────────────────────────────────────────────
    total_pnl    = float(df["cum_pnl"].iloc[-1])
    total_ret    = total_pnl / initial_capital
    ann_ret      = total_ret * (td_per_year / n)          # simple annualisation
    ann_vol      = float(ret.std(ddof=1) * np.sqrt(td_per_year))

    # ── Risk-adjusted ─────────────────────────────────────────────────────────
    daily_rf = rf_rate / td_per_year
    excess   = ret - daily_rf

    exc_std  = float(excess.std(ddof=1))
    sharpe   = float(
        excess.mean() / exc_std * np.sqrt(td_per_year)
        if exc_std > 1e-12 else np.nan
    )
    sortino  = _sortino_ratio(ret, rf_rate, td_per_year)

    max_dd_pct = float(df["drawdown_pct"].min())           # e.g. −0.046 = −4.6 %
    calmar     = float(
        ann_ret / abs(max_dd_pct)
        if abs(max_dd_pct) > 1e-12 else np.nan
    )

    # ── Drawdown duration ─────────────────────────────────────────────────────
    max_dd_dur = _max_drawdown_duration(df["cum_pnl"])

    # ── Trade statistics ──────────────────────────────────────────────────────
    if "lag_position" in df.columns:
        n_trades    = int((df["lag_position"].diff().abs() > 0).sum())
        active_mask = df["lag_position"].abs() > 0
    else:
        n_trades    = 0
        active_mask = df["pnl_net"].abs() > 0

    pnl_active  = df["pnl_net"][active_mask]
    n_active    = int(active_mask.sum())

    winning = pnl_active[pnl_active > 0]
    losing  = pnl_active[pnl_active < 0]

    win_rate      = float(len(winning) / n_active * 100) if n_active > 0 else np.nan
    avg_win_usd   = float(winning.mean())  if len(winning) > 0 else 0.0
    avg_loss_usd  = float(losing.mean())   if len(losing)  > 0 else 0.0
    profit_factor = float(
        winning.sum() / (-losing.sum())
        if len(losing) > 0 and losing.sum() < 0 else np.nan
    )

    # ── Tail risk (empirical) ─────────────────────────────────────────────────
    var_95  = float(np.percentile(ret, 5))           # 5th percentile (negative)
    cvar_95 = float(ret[ret <= var_95].mean())       # expected shortfall

    return {
        "total_pnl":        total_pnl,
        "total_return_pct": total_ret  * 100,
        "ann_return_pct":   ann_ret    * 100,
        "ann_vol_pct":      ann_vol    * 100,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "calmar":           calmar,
        "max_drawdown_pct": max_dd_pct * 100,
        "max_dd_duration":  max_dd_dur,
        "n_trades":         n_trades,
        "win_rate_pct":     win_rate,
        "avg_win_usd":      avg_win_usd,
        "avg_loss_usd":     avg_loss_usd,
        "profit_factor":    profit_factor,
        "var_95_pct":       var_95     * 100,
        "cvar_95_pct":      cvar_95    * 100,
        "n_days":           n,
        "n_active_days":    n_active,
    }


def compute_rolling_sharpe(
    df:          pd.DataFrame,
    window:      int   = config.ROLLING_SHARPE_WINDOW,
    td_per_year: int   = config.TRADING_DAYS_PER_YEAR,
    rf_rate:     float = config.RISK_FREE_RATE,
) -> pd.Series:
    """
    Rolling annualised Sharpe ratio.

    Uses min_periods = max(5, window // 4) so partial windows at the start
    of the series still produce values rather than NaN for the full *window*.

    Parameters
    ----------
    df     : strategy DataFrame from Step 6 (must contain 'daily_return').
    window : look-back in trading days  (default: ROLLING_SHARPE_WINDOW = 252).

    Returns
    -------
    pd.Series indexed by date, values = rolling Sharpe (annualised).
    """
    daily_rf  = rf_rate / td_per_year
    excess    = df["daily_return"] - daily_rf
    min_p     = max(5, window // 4)

    roll_mean = excess.rolling(window, min_periods=min_p).mean()
    roll_std  = excess.rolling(window, min_periods=min_p).std(ddof=1)

    return (roll_mean / roll_std.clip(lower=1e-12) * np.sqrt(td_per_year)).rename("rolling_sharpe")


# ============================================================================
# Build full report across all strategies
# ============================================================================

def build_performance_report(
    portfolio:       Dict[str, pd.DataFrame],
    initial_capital: float = config.INITIAL_CAPITAL,
    td_per_year:     int   = config.TRADING_DAYS_PER_YEAR,
    rf_rate:         float = config.RISK_FREE_RATE,
) -> pd.DataFrame:
    """
    Compute metrics for every strategy in *portfolio* and return a
    wide DataFrame (strategies as columns, metrics as rows).

    Parameters
    ----------
    portfolio : dict from build_portfolio() (Step 6).

    Returns
    -------
    pd.DataFrame  –  index = metric names, columns = strategy names.
    """
    log.info("=" * 60)
    log.info("STEP 7 — Performance Metrics")
    log.info("=" * 60)

    records: Dict[str, dict] = {}
    for name, df in portfolio.items():
        records[name] = compute_metrics(df, initial_capital, td_per_year, rf_rate)
        log.debug("  %s: Sharpe=%.2f  Sortino=%.2f  Calmar=%.2f",
                  name,
                  records[name]["sharpe"]  if not np.isnan(records[name]["sharpe"])  else 0,
                  records[name]["sortino"] if not np.isnan(records[name]["sortino"]) else 0,
                  records[name]["calmar"]  if not np.isnan(records[name]["calmar"])  else 0)

    report = pd.DataFrame(records)
    log.info("Performance report built: %d metrics × %d strategies.", *report.shape)
    return report


# ============================================================================
# Validation
# ============================================================================

def validate_performance(
    report:      pd.DataFrame,
    min_sharpe:  float = -3.0,
    max_abs_ret: float = 500.0,    # % — sanity cap; > 500% annualised is suspicious
) -> dict:
    """
    Run basic sanity checks on the performance report and log results.

    Checks
    ------
    • Sharpe in a plausible range  (> min_sharpe for all strategies)
    • No infinite profit_factor
    • Annualised return not implausibly large

    Returns
    -------
    dict with 'passed' (bool) and any flagged issues.
    """
    log.info("─" * 60)
    log.info("PERFORMANCE VALIDATION")
    log.info("─" * 60)

    issues = []
    for col in report.columns:
        s = report[col]

        ar = s.get("ann_return_pct", np.nan)
        if not np.isnan(ar) and abs(ar) > max_abs_ret:
            issues.append(f"{col}: ann_return={ar:.1f}% exceeds ±{max_abs_ret}% cap")

        sh = s.get("sharpe", np.nan)
        if not np.isnan(sh) and sh < min_sharpe:
            issues.append(f"{col}: Sharpe={sh:.2f} below floor {min_sharpe}")

        pf = s.get("profit_factor", np.nan)
        if not np.isnan(pf) and np.isinf(pf):
            issues.append(f"{col}: profit_factor is infinite (no losing days — suspicious)")

    if issues:
        for issue in issues:
            log.warning("  ⚠  %s", issue)
    else:
        log.info("  All checks passed.")

    log.info("─" * 60)
    return {"passed": len(issues) == 0, "issues": issues}


# ============================================================================
# Print helpers
# ============================================================================

def print_performance_report(
    report:          pd.DataFrame,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> None:
    """
    Print a transposed performance table: strategies as columns,
    metrics as rows.  Numeric formatting is metric-specific.
    """
    # ── Metric display spec: (label, format_fn) ───────────────────────────────
    metric_fmt = [
        ("total_pnl",        "Total P&L ($)",        lambda x: f"${x:+,.0f}"),
        ("total_return_pct", "Total Return (%)",      lambda x: f"{x:+.2f}%"),
        ("ann_return_pct",   "Ann. Return (%)",       lambda x: f"{x:+.2f}%"),
        ("ann_vol_pct",      "Ann. Volatility (%)",   lambda x: f"{x:.2f}%"),
        ("sharpe",           "Sharpe Ratio",          lambda x: f"{x:+.3f}"),
        ("sortino",          "Sortino Ratio",         lambda x: f"{x:+.3f}"),
        ("calmar",           "Calmar Ratio",          lambda x: f"{x:+.3f}"),
        ("max_drawdown_pct", "Max Drawdown (%)",      lambda x: f"{x:.2f}%"),
        ("max_dd_duration",  "Max DD Duration (d)",   lambda x: f"{int(x)}"),
        ("n_trades",         "N Trades",              lambda x: f"{int(x)}"),
        ("win_rate_pct",     "Win Rate (%)",          lambda x: f"{x:.1f}%"),
        ("avg_win_usd",      "Avg Win ($)",           lambda x: f"${x:+,.0f}"),
        ("avg_loss_usd",     "Avg Loss ($)",          lambda x: f"${x:+,.0f}"),
        ("profit_factor",    "Profit Factor",         lambda x: f"{x:.3f}"),
        ("var_95_pct",       "VaR 95% (1d, %)",       lambda x: f"{x:.3f}%"),
        ("cvar_95_pct",      "CVaR 95% (1d, %)",      lambda x: f"{x:.3f}%"),
        ("n_days",           "N Days",                lambda x: f"{int(x)}"),
        ("n_active_days",    "Active Days",           lambda x: f"{int(x)}"),
    ]

    def _safe_fmt(val, fmt_fn) -> str:
        try:
            if isinstance(val, float) and np.isnan(val):
                return "     NaN"
            if isinstance(val, float) and np.isinf(val):
                return "     Inf"
            return fmt_fn(float(val))
        except Exception:
            return "     ---"

    strategies = list(report.columns)
    col_w = max(12, max(len(s) for s in strategies) + 2)
    label_w = 22

    print("\n" + "=" * (label_w + col_w * len(strategies) + 2))
    print("PERFORMANCE REPORT — Step 7")
    print("=" * (label_w + col_w * len(strategies) + 2))

    # Header
    header = f"{'Metric':{label_w}s}"
    for s in strategies:
        header += f"{s:>{col_w}s}"
    print(header)
    print("─" * (label_w + col_w * len(strategies) + 2))

    for key, label, fmt_fn in metric_fmt:
        if key not in report.index:
            continue
        row_str = f"{label:{label_w}s}"
        for s in strategies:
            val = report.loc[key, s]
            row_str += f"{_safe_fmt(val, fmt_fn):>{col_w}s}"
        print(row_str)

    print("─" * (label_w + col_w * len(strategies) + 2))
    print(f"  Capital: ${initial_capital:,.0f}  |  "
          f"rf={config.RISK_FREE_RATE:.1%}  |  "
          f"TD/yr={config.TRADING_DAYS_PER_YEAR}")
    print("=" * (label_w + col_w * len(strategies) + 2))


def print_rolling_sharpe_table(
    portfolio:   Dict[str, pd.DataFrame],
    window:      int = config.ROLLING_SHARPE_WINDOW,
    n_rows:      int = 20,
) -> None:
    """
    Print a table of rolling Sharpe ratios (one column per strategy).
    """
    roll_df = pd.DataFrame({
        name: compute_rolling_sharpe(df, window=window)
        for name, df in portfolio.items()
        if "daily_return" in df.columns
    })

    print("\n" + "=" * 72)
    print(f"ROLLING SHARPE (window={window} days, min_periods={max(5, window//4)})")
    print("=" * 72)
    print(roll_df.head(n_rows).to_string(
        float_format=lambda x: f"{x:+.3f}" if not np.isnan(x) else "   NaN"
    ))
    print("─" * 72)
    print("Final rolling Sharpe:")
    for col in roll_df.columns:
        last_valid = roll_df[col].dropna()
        val_str = f"{last_valid.iloc[-1]:+.3f}" if len(last_valid) > 0 else "NaN"
        print(f"  {col:10s}  {val_str}")
    print("=" * 72)
