"""
Script 07 – Market Feature Extraction
=======================================
Aggregate market features from preprocessed Tardis parquet files.

Usage:
    python scripts/phase1_data_engineering/07_extract_features.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from config.settings import PROCESSED_DIR, ensure_dirs


def extract_market_features():
    """Aggregate market features from preprocessed Tardis parquet files."""
    from src.features.market_features import MARKET_FEATURE_NAMES

    out_dir = PROCESSED_DIR / "tardis"
    parquet_files = sorted(out_dir.glob("*.parquet"))

    if not parquet_files:
        print("  No preprocessed Tardis files found. Run 06_preprocess_tardis.py first.")
        return

    dfs = []
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        df["source_file"] = pq.stem
        dfs.append(df)
        print(f"  {pq.stem}: {len(df)} rows")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out_path = PROCESSED_DIR / "market_features.csv"
        combined.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(combined)} rows)")


def main():
    ensure_dirs()
    print("\n── Market Feature Extraction ──")
    extract_market_features()
    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
