#!/usr/bin/env python3
"""SVB / Credit Suisse event study (Wed 2023-03-15 → Thu 2023-03-16).

The first VOL-DOWN event in the deck. Wednesday saw Credit Suisse stock
crash 24% after Saudi National Bank refused additional capital; the SNB
announced an emergency CHF 50bn liquidity facility late in the day, with
VIX rising ~10% to close at 26.14. Thursday brought Yellen's Senate
testimony reassuring banking-system soundness AND an 11-bank consortium
injecting $30bn deposits into First Republic; SPX rallied and VIX fell
~3 pts to 22.99.

This is the only event in the set with:
  - ΔVIX < 0  (vol fell on relief news)
  - ΔSPX > 0  (relief rally on Yellen / bank rescue)
  - Wed → Thu intraweek (not Fri → Mon)

Useful as a symmetry check that the framework correctly handles a
vol-down move with rebound spot. F1 should be negative (spot-up rolls
up the put-skewed smile to lower IV at the new ATM strike).

Outputs in src/output/:
  event_svb_decomp.csv         F1..F6 from run_decomposition
  event_svb_iv_summary.csv     CBOE textbook F1/F2 (Warren's table)
  event_svb_skew_chart.png     t0+t1 puts/calls dots-only
  event_svb_decomp_chart.png   F1..F6 bars + ΔVIX/Σ reference lines
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _event_study_core import run_event  # noqa: E402


EVENT = {
    "name": "SVB / Credit Suisse",
    "t0":   "2023-03-15",
    "t1":   "2023-03-16",
}


# Reference CBOE VIX close values for the SVB / Credit Suisse event.
# Hard-coded as fallback because fetch_cboe_vix_historical() occasionally
# returns 0 records when the CBOE CSV endpoint is rate-limited or down.
#
# Primary source: CBOE VIX_History.csv
#   https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
#
# Cross-checked against:
#   - 2023-03-15 close = 26.14 (Wed): VIX rose ~10% on Credit Suisse 24%
#     stock crash and Saudi National Bank refusal of additional capital;
#     SNB announced emergency CHF 50bn liquidity facility late in the day.
#     Reporting: https://washingtonexaminer.com/policy/economy/svb-collapse-swiss-government-steady-markets-credit-suisse
#     ("VIX was up nearly 10% around the time markets closed on Wednesday")
#   - 2023-03-16 close = 22.99 (Thu): VIX fell ~3 pts on Yellen Senate
#     testimony reassuring banking system soundness AND 11-bank consortium
#     injecting $30bn deposits into First Republic to halt regional bank
#     contagion.
#     Reporting: https://www.axios.com/2023/03/18/silicon-valley-bank-timeline
#   - FRED VIXCLS series (https://fred.stlouisfed.org/series/VIXCLS)
#
# ΔVIX over Wed→Thu = 22.99 − 26.14 = -3.15 vol-points
# NOTE: this event has NEGATIVE ΔVIX (vol fell on relief news), opposite
# to Volmageddon / Feb2020 / Covid which all had VIX rising. Useful for
# demonstrating the framework handles vol-down moves correctly.
REFERENCE_VIX_CLOSE = {
    "2023-03-15": 26.14,  # Credit Suisse 24% crash, SNB CHF 50bn backstop
    "2023-03-16": 22.99,  # Yellen reassurance + 11 banks inject $30bn
}


def run() -> dict:
    return run_event(EVENT, REFERENCE_VIX_CLOSE, slug="svb")


if __name__ == "__main__":
    run()
