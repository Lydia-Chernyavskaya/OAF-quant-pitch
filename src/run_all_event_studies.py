#!/usr/bin/env python3
"""Run all four event studies in sequence.

  python src/run_all_event_studies.py

Produces 16 output files under src/output/:
  event_volmageddon_*  (decomp.csv, iv_summary.csv, skew_chart.png, decomp_chart.png)
  event_feb2020_*      (same four)
  event_svb_*          (same four)
  event_covid_*        (same four)

Order is chronological. SVB / Credit Suisse is the only event with
negative ΔVIX (vol fell on Yellen / bank-rescue relief).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from event_study_volmageddon import run as run_volmageddon  # noqa: E402
from event_study_feb2020 import run as run_feb2020          # noqa: E402
from event_study_svb import run as run_svb                  # noqa: E402
from event_study_covid import run as run_covid              # noqa: E402


def _print_summary_table(records: list[dict]) -> None:
    print()
    print("=" * 100)
    print("EVENT SUMMARY")
    print("=" * 100)
    header = (f"{'Event':<22} {'dSPX%':>7} {'dVIX':>7} "
              f"{'F1':>6} {'F2':>6} {'F3':>6} {'F4':>6} {'F5':>6} {'F6':>6} "
              f"{'Sum':>7} {'Resid':>7}")
    print(header)
    print("-" * len(header))
    for r in records:
        d_spx_pct = (r["SPX_t1"] - r["SPX_t0"]) / r["SPX_t0"] * 100.0
        print(f"{r['name']:<22} "
              f"{d_spx_pct:>+7.2f} {r['dVIX']:>+7.2f} "
              f"{r['F1']:>+6.2f} {r['F2']:>+6.2f} {r['F3']:>+6.2f} "
              f"{r['F4']:>+6.2f} {r['F5']:>+6.2f} {r['F6']:>+6.2f} "
              f"{r['sum_factors']:>+7.2f} {r['residual']:>+7.2f}")
    print("=" * 100)


if __name__ == "__main__":
    records = []
    records.append(run_volmageddon())
    records.append(run_feb2020())
    records.append(run_svb())
    records.append(run_covid())
    _print_summary_table(records)
