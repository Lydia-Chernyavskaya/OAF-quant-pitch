"""
config.py
=========
Central configuration for the VIX Decomposition Research Framework.

All magic numbers, file paths, and tunable parameters live here.
Nothing is hard-coded inside the source modules — they always import from
this file so that changing a single value propagates everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# Where the raw txt files live (organised as downloaded quarter folders)
RAW_DATA_DIR = PROJECT_ROOT / "SPX data"

# Where we write cleaned, ready-to-use parquet files
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Convenience paths for the two processed artefacts
PROCESSED_OPTIONS_PATH = PROCESSED_DATA_DIR / "spx_options.parquet"
PROCESSED_MARKET_PATH  = PROCESSED_DATA_DIR / "market_data.parquet"

# ---------------------------------------------------------------------------
# Market data download (yfinance)
# ---------------------------------------------------------------------------
SPX_TICKER = "^GSPC"   # S&P 500 index
VIX_TICKER = "^VIX"    # CBOE Volatility Index
SPY_TICKER = "SPY"     # ETF used for portfolio simulation

# Historical window to download (overridden to Q4-2023 for the initial build)
DEFAULT_START_DATE = "2023-10-01"
DEFAULT_END_DATE   = "2023-12-31"

# ---------------------------------------------------------------------------
# VIX construction parameters
# ---------------------------------------------------------------------------
# CBOE uses the window 23-37 calendar days to expiration
MIN_DTE = 23
MAX_DTE = 37

# Target maturity for the synthetic VIX (calendar days)
TARGET_DTE = 30

# Minimum number of usable strikes required on a given day before we attempt
# to build a surface.  Fewer than this means the day is unusable.
MIN_STRIKES_REQUIRED = 20

# ---------------------------------------------------------------------------
# Data quality / sanity-check thresholds
# ---------------------------------------------------------------------------
# Minimum implied vol considered valid (avoids division-by-zero and garbage)
MIN_IV = 0.001          # 0.1 %

# Maximum implied vol we accept (anything above is almost certainly a data
# error — 5 = 500 % annualised vol)
MAX_IV = 5.0

# Maximum bid–ask spread as a fraction of mid-quote.  Beyond this the option
# is too illiquid to trust.
MAX_RELATIVE_SPREAD = 0.50   # 50 % of mid

# Minimum bid price (absolute) to consider an option liquid enough
MIN_BID = 0.05

# ---------------------------------------------------------------------------
# Risk-free rate (used in forward price and VIX reconstruction)
# ---------------------------------------------------------------------------
# Annualised continuously-compounded risk-free rate.
# For production use this should be sourced dynamically from FRED (3-month
# T-bill: series TB3MS) or the OIS curve.  For Q4-2023 the Fed Funds target
# was 5.25–5.50%, so 5.3% is a reasonable constant approximation.
RISK_FREE_RATE = 0.045   # 4.5 %

# ---------------------------------------------------------------------------
# IV surface fitting
# ---------------------------------------------------------------------------
# Smoothing condition for the cubic spline (s=0 → exact interpolation).
# Exact interpolation honours the market quotes precisely; set s > 0 only if
# you observe oscillations ("Runge's phenomenon") in the fitted surface.
SPLINE_SMOOTHING = 0.0

# How far outside [min_strike, max_strike] are we allowed to evaluate the
# spline?  We set this to zero to prevent any extrapolation.
SPLINE_EXTRAPOLATION_ALLOWED = False

# Minimum number of (strike, IV) points needed on each side (OTM puts /
# OTM calls) to accept a surface as usable.  Surfaces below this threshold
# are flagged but not discarded — Step 3 will decide whether to skip them.
MIN_SURFACE_STRIKES_EACH_SIDE = 5

# ---------------------------------------------------------------------------
# VIX Decomposition (Step 4)
# ---------------------------------------------------------------------------
# Current method: asymmetric piecewise polynomial projection.
#   ΔIV = IV_t − IV_ss is projected onto {α, β_put·m, γ_put·m², β_call·m,
#   γ_call·m²} via CBOE-integrand-weighted OLS across ALL t-day strikes.
#   No moneyness window is applied to the polynomial projection itself.

# ATM window for the TV quadratic fit  w(k) = α + β·k + γ·k²
# (used only for backward-compat a_t / b_t / c_t passed to signals.py).
QUADRATIC_WINDOW_PCT = 0.15   # ±15 % of K₀  (for TV fit only)

# Minimum strikes in the TV fit window before the fit is considered valid.
MIN_QUAD_STRIKES = 10

# Legacy bucket boundaries — NOT used in the current polynomial projection.
# Kept for reference; the bucket attribution approach was superseded.
#   Parallel : |k| < BUCKET_ATM_WIDTH
#   Put/Call skew : BUCKET_ATM_WIDTH ≤ |k| ≤ BUCKET_SKEW_WIDTH
#   Put/Call conv : |k| > BUCKET_SKEW_WIDTH
BUCKET_ATM_WIDTH  = 0.05   # ±5% log-moneyness  (legacy)
BUCKET_SKEW_WIDTH = 0.20   # ±20% log-moneyness  (legacy)

# For the polynomial projection IV arrays, IV is floored at this value
# after applying a component perturbation that might push it below zero.
BUMP_IV_FLOOR = 0.001   # same as MIN_IV

# ---------------------------------------------------------------------------
# Signal / strategy parameters (initial defaults; calibrated in walk-forward)
# ---------------------------------------------------------------------------
# Rolling window for z-score of parallel shift (days)
PARALLEL_ZSCORE_WINDOW = 60

# Trigger thresholds for each strategy
PARALLEL_ZSCORE_ENTRY  = 2.0   # |z| > this triggers a trade
CONVEXITY_DOM_THRESHOLD_VALUES = [0.50, 0.60, 0.70]  # tested in Step 5
STICKY_STRIKE_HIGH     = 1.5   # ratio above this → mean reversion
STICKY_STRIKE_LOW      = 0.7   # ratio below this → momentum

# Minimum |delta_a_ss| (in IV units) required to compute the SSD ratio.
# When spot barely moves, the sticky-strike predicted change is near zero and
# the ratio is numerically unreliable.  0.002 = 0.2% annualised IV change.
SSD_MIN_SS_MOVE = 0.002

# Minimum |model_total| (VIX points) for the convexity dominance signal.
# Below this, convexity ratios are dominated by rounding noise.
CD_MIN_MODEL_MOVE = 0.05

# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------
INITIAL_CAPITAL    = 1_000_000   # USD
TRANSACTION_COST_BPS = 2         # basis points per one-way trade
POSITION_SIZE_NOTIONAL = 1.0     # 1× notional (long or short)

# ---------------------------------------------------------------------------
# Performance evaluation
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE        = 0.0   # kept at 0 for simplicity; override if needed

# Rolling window for rolling Sharpe output chart
ROLLING_SHARPE_WINDOW = 252

# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------
TRAIN_START = "2014-01-01"
TRAIN_END   = "2019-12-31"
TEST_START  = "2020-01-01"
TEST_END    = "2024-12-31"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"   # Set to "DEBUG" for verbose output during development
