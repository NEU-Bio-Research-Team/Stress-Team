"""
Script 06 – Preprocess Tardis BTC
====================================
Run full Tardis preprocessing pipeline:
  Trade aggregation → OHLCV bars → orderbook features → flash crash detection

Usage:
    python scripts/06_preprocess_tardis.py
    python scripts/06_preprocess_tardis.py --start 2021-01-01 --end 2021-12-31
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import TARDIS_START_DATE, TARDIS_END_DATE, ensure_dirs
from src.preprocessing.tardis_preprocess import preprocess_all


def main():
    parser = argparse.ArgumentParser(description="Preprocess Tardis BTC data")
    parser.add_argument("--start", default=TARDIS_START_DATE)
    parser.add_argument("--end", default=TARDIS_END_DATE)
    args = parser.parse_args()

    ensure_dirs()
    print(f"Preprocessing Tardis BTC: {args.start} → {args.end}")
    preprocess_all(start_date=args.start, end_date=args.end)
    print("\nDone. Output in data/processed/tardis/")


if __name__ == "__main__":
    main()
