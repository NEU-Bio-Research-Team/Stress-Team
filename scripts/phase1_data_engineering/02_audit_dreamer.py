"""
Script 02 – DREAMER Audit (D1-D12)
====================================
Runs all 12 DREAMER dataset audit checks.

Usage:
    python scripts/02_audit_dreamer.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import ensure_dirs
from src.audit.dreamer_audit import run_full_audit


def main():
    ensure_dirs()
    print("="*60)
    print("DREAMER Dataset Audit  (D1 – D12)")
    print("="*60)
    results = run_full_audit()
    print(f"\nTotal checks: {len(results)}")
    passed = sum(1 for r in results if r.get("pass", False))
    print(f"Passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
