#!/usr/bin/env python3
"""Run all three event studies in sequence.

  python src/run_all_event_studies.py

Produces 12 output files under src/output/:
  event_volmageddon_*  (decomp.csv, iv_summary.csv, skew_chart.png, decomp_chart.png)
  event_feb2020_*      (same four)
  event_covid_*        (same four)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from event_study_volmageddon import run as run_volmageddon  # noqa: E402
from event_study_feb2020 import run as run_feb2020          # noqa: E402
from event_study_covid import run as run_covid              # noqa: E402


if __name__ == "__main__":
    run_volmageddon()
    run_feb2020()
    run_covid()
