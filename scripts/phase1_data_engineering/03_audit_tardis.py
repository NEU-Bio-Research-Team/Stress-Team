"""
Script 03 – Tardis Audit (T1-T15)
====================================
Runs all 15 Tardis BTC data audit checks.

Usage:
    python scripts/03_audit_tardis.py
    python scripts/03_audit_tardis.py --start 2021-05-01 --end 2021-06-01
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import TARDIS_START_DATE, TARDIS_END_DATE, ensure_dirs
from src.audit.tardis_audit import run_full_audit


def main():
    parser = argparse.ArgumentParser(description="Tardis BTC data audit")
    parser.add_argument("--start", default=TARDIS_START_DATE)
    parser.add_argument("--end", default=TARDIS_END_DATE)
    args = parser.parse_args()

    ensure_dirs()
    print("="*60)
    print("Tardis BTC Futures Audit  (T1 – T15)")
    print("="*60)
    results = run_full_audit(start_date=args.start, end_date=args.end)
    print(f"\nTotal checks: {len(results)}")
    passed = sum(1 for r in results if r.get("pass", False))
    print(f"Passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
