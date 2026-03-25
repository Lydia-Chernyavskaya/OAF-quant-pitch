"""
src/data_ingestion.py  –  Step 1: Data Ingestion
==================================================
Responsible for:
  A) Downloading SPX and VIX daily closes from yfinance.
  B) Loading SPX end-of-day options data from the raw txt files.
  C) Filtering maturities to the VIX window (23–37 DTE).
  D) Applying data-quality / sanity checks.
  E) Persisting cleaned data as parquet for fast downstream access.

Raw file format (Cboe / LiveVol style, wide)
--------------------------------------------
Each row contains BOTH the call and put for a given
(quote_date, expiry, strike) triple.  Key columns (square-bracketed):

  [QUOTE_DATE]      – trading date (YYYY-MM-DD)
  [EXPIRE_DATE]     – option expiration date
  [DTE]             – calendar days to expiration at quote time
  [STRIKE]          – option strike price
  [UNDERLYING_LAST] – SPX spot price at end of day
  [C_IV]            – call implied volatility (annualised, decimal)
  [C_BID], [C_ASK]  – call best bid / ask
  [P_IV]            – put implied volatility
  [P_BID], [P_ASK]  – put best bid / ask

We melt this wide layout into a long layout:
  one row per (date, expiry, strike, option_type)
with canonical column names required by downstream modules.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── project imports ──────────────────────────────────────────────────────────
# Make sure the project root is on the path when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------
# The raw files wrap column names in square brackets with a leading space,
# e.g. " [QUOTE_DATE]".  This regex strips both the brackets and whitespace.
_BRACKET_RE = re.compile(r"[\[\]\s]")


def _clean_col(name: str) -> str:
    """Strip brackets and surrounding whitespace from a raw header token."""
    return _BRACKET_RE.sub("", name)


# Mapping from raw (cleaned) column names → canonical framework names.
# Only the columns we actually need are listed; everything else is dropped.
_WIDE_TO_CANONICAL = {
    "QUOTE_DATE":       "date",
    "EXPIRE_DATE":      "expiration",
    "DTE":              "dte",
    "STRIKE":           "strike",
    "UNDERLYING_LAST":  "underlying_last",
    # call-side
    "C_IV":  "call_iv",
    "C_BID": "call_bid",
    "C_ASK": "call_ask",
    # put-side
    "P_IV":  "put_iv",
    "P_BID": "put_bid",
    "P_ASK": "put_ask",
}


# ============================================================================
# A)  MARKET DATA DOWNLOAD  (SPX index + VIX index)
# ============================================================================

def download_market_data(
    start: str = config.DEFAULT_START_DATE,
    end: str   = config.DEFAULT_END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download SPX and VIX daily closes via yfinance and cache as parquet.

    Parameters
    ----------
    start, end : str
        ISO date strings defining the download window.
    force_refresh : bool
        If True, ignore any cached file and re-download.

    Returns
    -------
    pd.DataFrame
        Index: trading date (DatetimeIndex, UTC-normalised).
        Columns: ['spx_close', 'vix_close'].

    Notes
    -----
    yfinance returns NaN for non-trading days; we forward-fill a maximum of
    one day to handle minor data gaps, then drop any remaining NaNs so that
    downstream code always receives a complete time series.
    """
    cache_path = config.PROCESSED_MARKET_PATH

    # ── return cached file if it covers the requested window ─────────────────
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        # Check that the cache actually spans [start, end]
        if (
            pd.Timestamp(start) >= cached.index.min()
            and pd.Timestamp(end)  <= cached.index.max()
        ):
            log.info("Market data loaded from cache: %s", cache_path)
            return cached.loc[start:end]

    log.info("Downloading market data from yfinance (%s → %s)…", start, end)

    # ── download both tickers in one call (more efficient) ───────────────────
    raw = yf.download(
        tickers=[config.SPX_TICKER, config.VIX_TICKER],
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits/dividends
        progress=False,
    )

    # yfinance returns a MultiIndex column structure when multiple tickers are
    # requested: level-0 = field (Close, Open, …), level-1 = ticker symbol.
    # We only need the 'Close' field.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"].copy()
    else:
        # Single-ticker fallback (shouldn't happen, but be safe)
        closes = raw[["Close"]].copy()

    # Rename to canonical names
    ticker_to_col = {
        config.SPX_TICKER: "spx_close",
        config.VIX_TICKER: "vix_close",
    }
    closes = closes.rename(columns=ticker_to_col)

    # Ensure index is tz-naive DatetimeIndex (parquet plays nicer)
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes.index.name = "date"

    # ── data quality ─────────────────────────────────────────────────────────
    # Forward-fill at most 1 day (e.g. data feed hiccups), then drop rest
    closes = closes.ffill(limit=1).dropna()

    if closes.empty:
        raise ValueError(
            f"yfinance returned no usable data for [{start}, {end}]. "
            "Check the date range and your internet connection."
        )

    # Validate we got both columns
    for col in ("spx_close", "vix_close"):
        if col not in closes.columns:
            raise KeyError(
                f"Column '{col}' missing from yfinance download. "
                "Ticker mapping may need updating."
            )

    log.info(
        "Market data: %d trading days, SPX %.1f–%.1f, VIX %.1f–%.1f",
        len(closes),
        closes["spx_close"].min(), closes["spx_close"].max(),
        closes["vix_close"].min(), closes["vix_close"].max(),
    )

    # ── persist ──────────────────────────────────────────────────────────────
    closes.to_parquet(cache_path)
    log.info("Market data cached → %s", cache_path)

    return closes


# ============================================================================
# B + C + D)  OPTIONS DATA LOADING, FILTERING, AND SANITY CHECKS
# ============================================================================

def _parse_single_txt(path: Path) -> pd.DataFrame:
    """
    Parse one raw options txt file (wide format) into a canonical long
    DataFrame.

    Parameters
    ----------
    path : Path
        Absolute path to a single monthly txt file.

    Returns
    -------
    pd.DataFrame
        Long-format options data with columns:
          date, expiration, dte, strike, underlying_last,
          option_type ('C' or 'P'), implied_vol, bid, ask

    Implementation detail
    ---------------------
    The raw files use bracketed column names (e.g. [QUOTE_DATE]).  We:
      1. Read the header line, strip brackets, select the columns we need.
      2. Parse only those columns (fast — avoids loading ~25 unused columns).
      3. "Melt" the wide call/put layout into two rows per strike.
      4. Apply basic dtype coercions.
    """
    log.debug("Parsing %s", path.name)

    # ── Step 1: Read header to learn column positions ─────────────────────
    with open(path, "r") as fh:
        raw_header = fh.readline().strip().split(",")

    clean_header = [_clean_col(c) for c in raw_header]

    # Determine which raw columns we actually need
    needed_raw_cols = set(_WIDE_TO_CANONICAL.keys())
    usecols_idx = [
        i for i, c in enumerate(clean_header) if c in needed_raw_cols
    ]

    if not usecols_idx:
        raise ValueError(
            f"No recognised columns found in {path}.  "
            f"First header tokens: {clean_header[:5]}"
        )

    # ── Step 2: Read the CSV using only needed columns ────────────────────
    # We read by positional index (usecols accepts integers) so we can pass
    # the pre-cleaned names directly.
    df = pd.read_csv(
        path,
        header=0,
        usecols=usecols_idx,
        names=None,               # keep raw header for now
        dtype=str,                # read everything as string first for safety
        na_values=["", " ", "NA", "N/A"],
        low_memory=False,
    )

    # Re-apply clean names (read_csv with usecols preserves original names)
    df.columns = [_clean_col(c) for c in df.columns]

    # Drop any columns that aren't in our mapping (safety)
    df = df[[c for c in df.columns if c in _WIDE_TO_CANONICAL]]

    # Rename to canonical
    df = df.rename(columns=_WIDE_TO_CANONICAL)

    # ── Step 3: Coerce data types ─────────────────────────────────────────
    # Dates
    df["date"]        = pd.to_datetime(df["date"],       errors="coerce")
    df["expiration"]  = pd.to_datetime(df["expiration"], errors="coerce")

    # Numerics — coerce to float; invalid entries become NaN
    numeric_cols = [
        "dte", "strike", "underlying_last",
        "call_iv", "call_bid", "call_ask",
        "put_iv",  "put_bid",  "put_ask",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the key identifiers are missing
    df = df.dropna(subset=["date", "expiration", "strike", "dte"])

    # ── Step 4: Melt wide → long ──────────────────────────────────────────
    # Build two separate frames (calls / puts) and concatenate.

    id_cols = ["date", "expiration", "dte", "strike", "underlying_last"]

    call_df = df[id_cols + ["call_iv", "call_bid", "call_ask"]].copy()
    call_df = call_df.rename(columns={
        "call_iv":  "implied_vol",
        "call_bid": "bid",
        "call_ask": "ask",
    })
    call_df["option_type"] = "C"

    put_df = df[id_cols + ["put_iv", "put_bid", "put_ask"]].copy()
    put_df = put_df.rename(columns={
        "put_iv":  "implied_vol",
        "put_bid": "bid",
        "put_ask": "ask",
    })
    put_df["option_type"] = "P"

    long_df = pd.concat([call_df, put_df], ignore_index=True)
    long_df = long_df.sort_values(["date", "expiration", "strike", "option_type"])

    log.debug(
        "  %s → %d rows (wide) → %d rows (long)",
        path.name, len(df), len(long_df),
    )
    return long_df


def _load_raw_options(data_dir: Path) -> pd.DataFrame:
    """
    Discover and concatenate all txt files under *data_dir* recursively.

    Supports any nesting depth (quarter folders, month folders, etc.).

    Returns
    -------
    pd.DataFrame
        Combined long-format DataFrame from all discovered files.
    """
    txt_files = sorted(data_dir.rglob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found under {data_dir}.  "
            "Verify that RAW_DATA_DIR in config.py points to the correct folder."
        )

    log.info("Found %d raw txt file(s) under %s", len(txt_files), data_dir)

    frames = []
    for path in txt_files:
        try:
            frames.append(_parse_single_txt(path))
        except Exception as exc:
            # Log but continue — one bad file shouldn't abort everything
            log.warning("Skipping %s: %s", path.name, exc)

    if not frames:
        raise RuntimeError("All txt files failed to parse.  Check the raw data.")

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Raw options loaded: %d rows, %d unique dates, %d unique expirations",
        len(combined),
        combined["date"].nunique(),
        combined["expiration"].nunique(),
    )
    return combined


def _filter_dte_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only options whose DTE falls within the CBOE VIX window
    [MIN_DTE, MAX_DTE] (default 23–37 calendar days).

    Why this range?
    ---------------
    The CBOE VIX methodology interpolates between two option expirations
    that straddle 30 days.  Using a ±7-day window around 30 ensures we
    always have candidates on both sides.

    Parameters
    ----------
    df : pd.DataFrame  (long format)

    Returns
    -------
    pd.DataFrame  filtered copy
    """
    mask = df["dte"].between(config.MIN_DTE, config.MAX_DTE)
    filtered = df.loc[mask].copy()

    n_removed = len(df) - len(filtered)
    pct_kept  = 100.0 * len(filtered) / len(df) if len(df) else 0.0

    log.info(
        "DTE filter [%d, %d]: kept %d rows (%.1f%%), removed %d",
        config.MIN_DTE, config.MAX_DTE,
        len(filtered), pct_kept, n_removed,
    )
    return filtered


def _apply_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that would corrupt the VIX reconstruction or signal generation.

    Checks applied (in order)
    -------------------------
    1. Zero / negative / missing implied vol.
    2. Implied vol above the hard ceiling (5.0 = 500% ann.)  —  data error.
    3. Missing or inverted bid/ask (bid > ask).
    4. Zero bid AND zero ask  —  effectively no market exists.
    5. Minimum bid threshold  —  below this the option is too illiquid.
    6. Bid–ask spread > MAX_RELATIVE_SPREAD fraction of mid-quote
       (extremely wide spread implies stale or unreliable quote).

    Each check is logged individually so the analyst can identify
    problematic data without re-running the full pipeline.

    Parameters
    ----------
    df : pd.DataFrame  long-format options with implied_vol, bid, ask cols.

    Returns
    -------
    pd.DataFrame  cleaned copy with an integer 'rows_dropped' tracker logged.
    """
    original_len = len(df)

    def _log_drop(reason: str, mask_bad: pd.Series) -> pd.Series:
        """Helper: log count, return keep-mask (inverse of bad mask)."""
        n_bad = mask_bad.sum()
        if n_bad > 0:
            log.info("  Sanity – %-40s  removing %d rows", reason, n_bad)
        return ~mask_bad

    # 1. IV missing or below minimum threshold (config.MIN_IV = 0.001 = 0.1%)
    #    Using the config constant rather than just "> 0" catches near-zero
    #    values that are mathematically valid but numerically problematic later.
    keep = _log_drop(
        f"IV missing or < {config.MIN_IV} ({config.MIN_IV*100:.1f}%)",
        df["implied_vol"].isna() | (df["implied_vol"] < config.MIN_IV),
    )
    df = df.loc[keep].copy()

    # 2. IV above hard ceiling
    keep = _log_drop(
        f"IV > {config.MAX_IV:.0f} (data error)",
        df["implied_vol"] > config.MAX_IV,
    )
    df = df.loc[keep].copy()

    # 3. Bid or ask missing
    keep = _log_drop(
        "bid or ask is NaN",
        df["bid"].isna() | df["ask"].isna(),
    )
    df = df.loc[keep].copy()

    # 4. Inverted market (bid strictly greater than ask)
    keep = _log_drop(
        "inverted market (bid > ask)",
        df["bid"] > df["ask"],
    )
    df = df.loc[keep].copy()

    # 5. Both bid and ask are zero  →  no market at all
    keep = _log_drop(
        "both bid and ask are zero",
        (df["bid"] == 0) & (df["ask"] == 0),
    )
    df = df.loc[keep].copy()

    # 6. Minimum bid threshold
    keep = _log_drop(
        f"bid < {config.MIN_BID} (too illiquid)",
        df["bid"] < config.MIN_BID,
    )
    df = df.loc[keep].copy()

    # 7. Relative bid–ask spread
    mid   = (df["bid"] + df["ask"]) / 2.0
    # Guard against mid == 0 (already filtered above, but be safe)
    rel_spread = np.where(mid > 0, (df["ask"] - df["bid"]) / mid, np.inf)
    keep = _log_drop(
        f"relative spread > {config.MAX_RELATIVE_SPREAD:.0%}",
        pd.Series(rel_spread, index=df.index) > config.MAX_RELATIVE_SPREAD,
    )
    df = df.loc[keep].copy()

    total_removed = original_len - len(df)
    pct_kept = 100.0 * len(df) / original_len if original_len else 0.0
    log.info(
        "Sanity checks complete: kept %d / %d rows (%.1f%%); "
        "removed %d rows total",
        len(df), original_len, pct_kept, total_removed,
    )
    return df


def _compute_midquote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'mid' column = (bid + ask) / 2.

    The CBOE VIX formula uses mid-quote prices Q(K).  Computing it once here
    avoids repeated calculation in every downstream module.
    """
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df


def _check_strike_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Warn (do NOT drop) any trading days where the number of unique strikes
    falls below MIN_STRIKES_REQUIRED after all filters.

    Downstream (Step 2) will raise an exception for such days and skip them.
    This check is informational so the analyst knows early.

    Parameters
    ----------
    df : pd.DataFrame long-format options data.

    Returns
    -------
    Same df (unmodified) — this is a reporting-only function.
    """
    strike_counts = (
        df.groupby(["date", "expiration"])["strike"]
        .nunique()
        .reset_index(name="n_strikes")
    )
    thin = strike_counts.loc[
        strike_counts["n_strikes"] < config.MIN_STRIKES_REQUIRED
    ]
    if len(thin) > 0:
        log.warning(
            "%d (date, expiry) pairs have fewer than %d strikes after cleaning. "
            "These will be skipped in Step 2.",
            len(thin), config.MIN_STRIKES_REQUIRED,
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Thin-surface pairs:\n%s", thin.to_string())
    else:
        log.info(
            "All (date, expiry) pairs have ≥ %d strikes. ✓",
            config.MIN_STRIKES_REQUIRED,
        )
    return df  # pass-through


# ============================================================================
# E)  FULL PIPELINE ENTRY POINT
# ============================================================================

def load_options_data(
    data_dir: Optional[Path] = None,
    force_refresh: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full Step-1 options pipeline: load → filter → clean → cache.

    Parameters
    ----------
    data_dir : Path, optional
        Root folder containing the raw txt files.
        Defaults to config.RAW_DATA_DIR.
    force_refresh : bool
        Re-parse all txt files even if a parquet cache already exists.
    start_date, end_date : str, optional
        ISO date strings to slice the loaded data.  If None, all available
        dates are returned.

    Returns
    -------
    pd.DataFrame
        Clean long-format options data ready for Step 2.
        Columns: date, expiration, dte, strike, underlying_last,
                 option_type, implied_vol, bid, ask, mid

    Processing pipeline
    -------------------
    1. Check parquet cache (fast path).
    2. If cache is stale or missing: parse all raw txt files.
    3. Filter to DTE window [23, 37].
    4. Apply sanity checks (IV validity, liquidity, spread).
    5. Compute mid-quote.
    6. Report strike coverage.
    7. Write parquet cache.
    8. Optionally slice to [start_date, end_date].
    """
    cache_path = config.PROCESSED_OPTIONS_PATH
    data_dir   = data_dir or config.RAW_DATA_DIR

    # ── Fast path: return cached parquet ─────────────────────────────────────
    if cache_path.exists() and not force_refresh:
        log.info("Loading options data from parquet cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        # Ensure dates are datetime (parquet should preserve, but be safe)
        df["date"]       = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        log.info(
            "Cache hit: %d rows, %d unique dates",
            len(df), df["date"].nunique(),
        )
    else:
        # ── Slow path: parse all txt files ───────────────────────────────────
        log.info("Building options dataset from raw txt files…")

        raw   = _load_raw_options(data_dir)
        filt  = _filter_dte_window(raw)
        clean = _apply_sanity_checks(filt)
        clean = _compute_midquote(clean)
        clean = _check_strike_coverage(clean)

        # Persist
        clean.to_parquet(cache_path, index=False)
        log.info("Options data cached → %s (%d rows)", cache_path, len(clean))
        df = clean

    # ── Date slice (optional) ─────────────────────────────────────────────────
    if start_date is not None:
        df = df.loc[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df.loc[df["date"] <= pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError(
            f"No options data remains after date filtering "
            f"[{start_date}, {end_date}].  "
            "Check that the txt files cover this period."
        )

    log.info(
        "Options data ready: %d rows | %d dates | %d unique strikes",
        len(df), df["date"].nunique(), df["strike"].nunique(),
    )
    return df


# ============================================================================
# CONVENIENCE: run both loaders and return a dict  (used by main.py)
# ============================================================================

def run_ingestion(
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str   = config.DEFAULT_END_DATE,
    force_refresh: bool = False,
) -> dict:
    """
    Execute the full Step-1 ingestion pipeline and return both datasets.

    Parameters
    ----------
    start_date, end_date : str
        Date window to work with.
    force_refresh : bool
        Force re-download / re-parse even if caches exist.

    Returns
    -------
    dict with keys:
        'market'  – pd.DataFrame  (date index, spx_close, vix_close)
        'options' – pd.DataFrame  (long-format cleaned options)
    """
    log.info("=" * 60)
    log.info("STEP 1 — Data Ingestion")
    log.info("Window: %s → %s", start_date, end_date)
    log.info("=" * 60)

    market  = download_market_data(start_date, end_date, force_refresh)
    options = load_options_data(
        force_refresh=force_refresh,
        start_date=start_date,
        end_date=end_date,
    )

    # ── Final cross-validation ────────────────────────────────────────────────
    # Check that options dates align with market-data trading calendar
    options_dates = set(options["date"].dt.normalize().unique())
    market_dates  = set(market.index.normalize().unique())
    orphan_dates  = options_dates - market_dates

    if orphan_dates:
        log.warning(
            "%d options dates have no corresponding market data row "
            "(likely non-trading days in raw file): %s … (showing up to 5)",
            len(orphan_dates),
            sorted(orphan_dates)[:5],
        )

    log.info("Step 1 complete.")
    log.info("  Market rows  : %d", len(market))
    log.info("  Options rows : %d", len(options))

    return {"market": market, "options": options}


# ============================================================================
# Diagnostic helper — run standalone to inspect data quality
# ============================================================================

def print_summary(data: dict) -> None:
    """
    Print a compact human-readable summary of the ingested datasets.
    Useful for interactive validation in a notebook or direct script run.
    """
    mkt = data["market"]
    opt = data["options"]

    print("\n" + "=" * 60)
    print("MARKET DATA SUMMARY")
    print("=" * 60)
    print(f"  Date range   : {mkt.index.min().date()} → {mkt.index.max().date()}")
    print(f"  Trading days : {len(mkt)}")
    print(f"  SPX range    : {mkt['spx_close'].min():.1f} – {mkt['spx_close'].max():.1f}")
    print(f"  VIX range    : {mkt['vix_close'].min():.2f} – {mkt['vix_close'].max():.2f}")

    print("\n" + "=" * 60)
    print("OPTIONS DATA SUMMARY")
    print("=" * 60)
    print(f"  Total rows        : {len(opt):,}")
    print(f"  Unique dates      : {opt['date'].nunique()}")
    print(f"  Unique expirations: {opt['expiration'].nunique()}")
    print(f"  Strike range      : {opt['strike'].min():.0f} – {opt['strike'].max():.0f}")
    print(f"  DTE range         : {opt['dte'].min():.0f} – {opt['dte'].max():.0f}")
    print(f"  Calls / Puts      : {(opt['option_type']=='C').sum():,} / {(opt['option_type']=='P').sum():,}")
    print(f"  IV range          : {opt['implied_vol'].min():.4f} – {opt['implied_vol'].max():.4f}")
    print(f"  Mid range         : {opt['mid'].min():.2f} – {opt['mid'].max():.2f}")

    # Per-date strike count distribution
    per_day = opt.groupby("date")["strike"].nunique()
    print(f"\n  Strikes per date  : min={per_day.min()}, "
          f"median={per_day.median():.0f}, max={per_day.max()}")
    thin_days = (per_day < config.MIN_STRIKES_REQUIRED).sum()
    print(f"  Days below min ({config.MIN_STRIKES_REQUIRED} strikes): {thin_days}")
    print("=" * 60)
