"""
Script 01 – WESAD Audit (W1-W12)
==================================
Runs all 12 WESAD dataset audit checks.

Usage:
    python scripts/01_audit_wesad.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import ensure_dirs
from src.audit.wesad_audit import run_full_audit


def main():
    ensure_dirs()
    print("="*60)
    print("WESAD Dataset Audit  (W1 – W12)")
    print("="*60)
    results = run_full_audit()
    print(f"\nTotal checks: {len(results)}")
    passed = sum(1 for r in results if r.get("pass", False))
    print(f"Passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
