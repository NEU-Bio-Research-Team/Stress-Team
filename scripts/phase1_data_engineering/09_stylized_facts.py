"""
Script 09 – Stylized Facts Validation
========================================
Compute and validate the 5 core stylized facts from BTC return series.

Usage:
    python scripts/09_stylized_facts.py
    python scripts/09_stylized_facts.py --input data/processed/market_features.csv
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from config.settings import PROCESSED_DIR, ensure_dirs
from src.analysis.stylized_facts import run_all_stylized_facts


def main():
    parser = argparse.ArgumentParser(description="Stylized facts validation")
    parser.add_argument("--input", default=None,
                        help="Path to market features CSV. Default: auto-detect.")
    args = parser.parse_args()

    ensure_dirs()
    print("="*60)
    print("Stylized Facts Validation  (SF-1 → SF-5)")
    print("="*60)

    # Load market data
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = PROCESSED_DIR / "market_features.csv"

    if not input_path.exists():
        # Try loading raw from tardis processed parquets
        tardis_dir = PROCESSED_DIR / "tardis"
        parquets = sorted(tardis_dir.glob("*.parquet"))
        if parquets:
            dfs = [pd.read_parquet(p) for p in parquets]
            df = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(df)} rows from {len(parquets)} parquet files")
        else:
            print(f"ERROR: No market data found at {input_path}")
            print("Run scripts 06 and 07 first.")
            sys.exit(1)
    else:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from {input_path}")

    # Extract returns – look for close price column
    close_col = None
    for candidate in ["close", "Close", "price", "vwap", "VWAP"]:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        print("ERROR: No price column found. Expected 'close', 'price' or 'vwap'.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    prices = df[close_col].dropna().values
    returns = np.diff(np.log(prices + 1e-12))
    print(f"Computing stylized facts on {len(returns)} returns")

    # Volume if available
    volumes = None
    for vol_col in ["volume", "Volume", "total_volume"]:
        if vol_col in df.columns:
            volumes = df[vol_col].dropna().values[:len(returns)]
            break

    run_all_stylized_facts(returns, volumes)


if __name__ == "__main__":
    main()
