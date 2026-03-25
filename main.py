"""
main.py  –  VIX Decomposition Research Framework  –  Entry Point
=================================================================
Run each step independently by passing the step number as a CLI argument:

    python main.py --step 1      # Data ingestion only
    python main.py --step 2      # IV surface (requires step 1 output)
    python main.py --step all    # Full pipeline (not yet implemented)

For interactive / notebook use, import the module directly:

    from src.data_ingestion import run_ingestion, print_summary
    data = run_ingestion()
    print_summary(data)

Flags
-----
--start     ISO start date (default: config.DEFAULT_START_DATE)
--end       ISO end date   (default: config.DEFAULT_END_DATE)
--refresh   Force re-download / re-parse even if caches exist
--step      Which step to execute  (default: 1)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Project root on path
sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")


def run_step1(start: str, end: str, refresh: bool) -> dict:
    from src.data_ingestion import run_ingestion, print_summary
    data = run_ingestion(start_date=start, end_date=end, force_refresh=refresh)
    print_summary(data)
    return data


def run_step2(data: dict) -> dict:
    from src.iv_surface import build_daily_surfaces, print_surface_summary
    surfaces = build_daily_surfaces(data["options"])
    print_surface_summary(surfaces)
    return surfaces


def run_step4(surfaces: dict, vix_data: dict) -> pd.DataFrame:
    from src.vix_decomposition import (
        build_decomposition, validate_decomposition,
        print_decomposition_table,
    )
    decomp = build_decomposition(surfaces, vix_data["vix_synthetic"])
    validate_decomposition(decomp)
    print_decomposition_table(decomp, n_rows=62)
    return decomp


def run_step5(decomp: pd.DataFrame, surfaces: dict) -> pd.DataFrame:
    from src.signals import build_signals, validate_signals, print_signals_table
    signals = build_signals(decomp, surfaces)
    validate_signals(signals)
    print_signals_table(signals, n_rows=62)
    return signals


def run_step6(signals: pd.DataFrame, vix_data: dict) -> dict:
    from src.portfolio import (
        build_portfolio, validate_portfolio,
        print_portfolio_table, print_portfolio_summary,
    )
    portfolio = build_portfolio(signals, vix_data["vix_synthetic"])
    validate_portfolio(portfolio)
    print_portfolio_table(portfolio, n_rows=62)
    print_portfolio_summary(portfolio)
    return portfolio


def run_step7(portfolio: dict) -> pd.DataFrame:
    from src.performance import (
        build_performance_report, validate_performance,
        print_performance_report, print_rolling_sharpe_table,
    )
    report = build_performance_report(portfolio)
    validate_performance(report)
    print_performance_report(report)
    print_rolling_sharpe_table(portfolio, n_rows=62)
    return report


def run_step3(data: dict, surfaces: dict) -> dict:
    from src.vix_reconstruction import (
        build_vix_series, validate_vix, print_vix_comparison,
    )
    # Align actual VIX index to date-only (no time component)
    mkt = data["market"].copy()
    mkt.index = mkt.index.normalize()

    vix_synth, diag_df = build_vix_series(data["options"], surfaces)
    vix_actual = mkt["vix_close"].rename("vix_actual")

    stats = validate_vix(vix_synth, vix_actual)
    print_vix_comparison(vix_synth, vix_actual, n_rows=63)
    return {"vix_synthetic": vix_synth, "diagnostics": diag_df, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VIX Decomposition Research Framework"
    )
    parser.add_argument("--step",    default="1", help="Step to run (1–7 or 'all')")
    parser.add_argument("--start",   default=config.DEFAULT_START_DATE)
    parser.add_argument("--end",     default=config.DEFAULT_END_DATE)
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download / re-parse")
    args = parser.parse_args()

    step = args.step.strip().lower()

    if step in ("1", "2", "3", "4", "5", "6", "7", "all"):
        data = run_step1(args.start, args.end, args.refresh)

    if step in ("2", "3", "4", "5", "6", "7", "all"):
        surfaces = run_step2(data)

    if step in ("3", "4", "5", "6", "7", "all"):
        vix_data = run_step3(data, surfaces)

    if step in ("4", "5", "6", "7", "all"):
        decomp = run_step4(surfaces, vix_data)

    if step in ("5", "6", "7", "all"):
        signals = run_step5(decomp, surfaces)

    if step in ("6", "7", "all"):
        portfolio = run_step6(signals, vix_data)

    if step in ("7", "all"):
        run_step7(portfolio)

    if step not in ("1", "2", "3", "4", "5", "6", "7", "all"):
        log.error("Step '%s' not yet implemented.  Use --step 1–7.", step)
        sys.exit(1)


if __name__ == "__main__":
    main()
