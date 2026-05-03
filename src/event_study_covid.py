#!/usr/bin/env python3
"""Covid crash event study (Friday 2020-03-13 → Monday 2020-03-16).

The largest single-day S&P drop since 1987 (-12.0%); VIX closed at
82.69, the all-time daily-close high.

Outputs in src/output/:
  event_covid_decomp.csv         F1..F6 from run_decomposition
  event_covid_iv_summary.csv     CBOE textbook F1/F2 (Warren's table)
  event_covid_skew_chart.png     t0+t1 puts/calls dots-only
  event_covid_decomp_chart.png   F1..F6 bars + ΔVIX/Σ reference lines
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _event_study_core import run_event  # noqa: E402


EVENT = {
    "name": "Covid crash",
    "t0":   "2020-03-13",
    "t1":   "2020-03-16",
}


# Reference CBOE VIX close values, hard-coded as fallback because
# fetch_cboe_vix_historical() occasionally returns 0 records when the
# CBOE CSV endpoint is rate-limited or down.
#
# Primary source: CBOE VIX_History.csv
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
#
# Cross-checked against:
#   - Macroption VIX historical record:
#     2020-03-16 close = 82.69 (close-to-close +24.86 from prior 57.83)
#     https://www.macroption.com/vix-all-time-high/
#   - Wikipedia, "VIX": 2020-03-16 listed as the all-time daily close high.
#   - FRED VIXCLS series (https://fred.stlouisfed.org/series/VIXCLS)
REFERENCE_VIX_CLOSE = {
    "2020-03-13": 57.83,  # Covid Friday
    "2020-03-16": 82.69,  # Covid Monday — all-time VIX close high
}


def run() -> dict:
    return run_event(EVENT, REFERENCE_VIX_CLOSE, slug="covid")


if __name__ == "__main__":
    run()
