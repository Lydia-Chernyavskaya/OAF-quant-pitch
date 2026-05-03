#!/usr/bin/env python3
"""Feb 2020 onset event study (Friday 2020-02-21 → Monday 2020-02-24).

The first major Covid-driven selloff. Friday 2020-02-21 was the last
VIX close below 20 before the panic; Monday 2020-02-24's S&P -3.35%
move drove VIX to 25.03.

Outputs in src/output/:
  event_feb2020_decomp.csv         F1..F6 from run_decomposition
  event_feb2020_iv_summary.csv     CBOE textbook F1/F2 (Warren's table)
  event_feb2020_skew_chart.png     t0+t1 puts/calls dots-only
  event_feb2020_decomp_chart.png   F1..F6 bars + ΔVIX/Σ reference lines
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _event_study_core import run_event  # noqa: E402


EVENT = {
    "name": "Feb 2020 onset",
    "t0":   "2020-02-21",
    "t1":   "2020-02-24",
}


# Reference CBOE VIX close values, hard-coded as fallback because
# fetch_cboe_vix_historical() occasionally returns 0 records when the
# CBOE CSV endpoint is rate-limited or down.
#
# Primary source: CBOE VIX_History.csv
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
#
# Cross-checked against:
#   - 2020-02-21 close = 17.08 (last VIX close below 20 before Covid panic;
#     referenced in Gateway Investment Advisers Jan-2021 market perspective:
#     "the last time the VIX posted a daily closing value below 20 was
#     February 21, 2020")
#     https://www.gia.com/wp-content/uploads/2022/03/Gateway-Market-Perspective-January-2021.pdf
#   - 2020-02-24 close = 25.03 (CNBC reported ~25.04 intraday-late on Mon
#     Feb 24 with +46% jump on coronavirus concerns):
#     https://www.cnbc.com/2020/02/24/us-futures-coronavirus-outbreak.html
#   - FRED VIXCLS series (https://fred.stlouisfed.org/series/VIXCLS)
#
# ΔVIX over Fri→Mon = 25.03 − 17.08 = +7.95 vol-points
REFERENCE_VIX_CLOSE = {
    "2020-02-21": 17.08,  # Last VIX close below 20 pre-Covid (Friday)
    "2020-02-24": 25.03,  # First major Covid selloff (Mon, S&P -3.35%)
}


def run() -> dict:
    return run_event(EVENT, REFERENCE_VIX_CLOSE, slug="feb2020")


if __name__ == "__main__":
    run()
