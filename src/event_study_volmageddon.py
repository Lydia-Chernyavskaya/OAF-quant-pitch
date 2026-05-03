#!/usr/bin/env python3
"""Volmageddon event study (Friday 2018-02-02 → Monday 2018-02-05).

Outputs in src/output/:
  event_volmageddon_decomp.csv         F1..F6 from run_decomposition
  event_volmageddon_iv_summary.csv     CBOE textbook F1/F2 (Warren's table)
  event_volmageddon_skew_chart.png     t0+t1 puts/calls dots-only
  event_volmageddon_decomp_chart.png   F1..F6 bars + ΔVIX/Σ reference lines
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _event_study_core import run_event  # noqa: E402


EVENT = {
    "name": "Volmageddon",
    "t0":   "2018-02-02",
    "t1":   "2018-02-05",
}


# Reference CBOE VIX close values, hard-coded as fallback because
# fetch_cboe_vix_historical() occasionally returns 0 records when the
# CBOE CSV endpoint is rate-limited or down.
#
# Primary source: CBOE VIX_History.csv
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
#
# Cross-checked against:
#   - CBOE research paper "After the Volpocalypse" (Feb 2018):
#     2018-02-05 close = 37.32, prior day 17.31
#     https://cdn.cboe.com/resources/education/research_publications/after-the-volpocalypse-market-observation.pdf
#   - FRED VIXCLS series (https://fred.stlouisfed.org/series/VIXCLS)
REFERENCE_VIX_CLOSE = {
    "2018-02-02": 17.31,  # Volmageddon eve (Friday)
    "2018-02-05": 37.32,  # Volmageddon (XIV blow-up Monday)
}


def run() -> dict:
    return run_event(EVENT, REFERENCE_VIX_CLOSE, slug="volmageddon")


if __name__ == "__main__":
    run()
