"""
Script 08 – Cross-Dataset Alignment Check
============================================
Validates alignment between physiological and market features.

Usage:
    python scripts/08_alignment_check.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.audit.cross_dataset_alignment import run_full_alignment_check


def main():
    ensure_dirs()
    print("="*60)
    print("Cross-Dataset Alignment Check  (CA-1 → CA-6)")
    print("="*60)

    # Try to load available feature files
    wesad_feats = None
    dreamer_feats = None
    market_feats = None
    stress_labels = None

    wesad_path = PROCESSED_DIR / "wesad_features.csv"
    if wesad_path.exists():
        wesad_feats = pd.read_csv(wesad_path)
        if "label" in wesad_feats.columns:
            # Labels are already binary (0=non-stress, 1=stress) from preprocessing
            stress_labels = wesad_feats["label"].values
        print(f"Loaded WESAD features: {wesad_feats.shape}")
    else:
        print("WESAD features not found. Run 07_extract_features.py first.")

    dreamer_path = PROCESSED_DIR / "dreamer_features.csv"
    if dreamer_path.exists():
        dreamer_feats = pd.read_csv(dreamer_path)
        if stress_labels is None and "stress" in dreamer_feats.columns:
            stress_labels = dreamer_feats["stress"].values
        print(f"Loaded DREAMER features: {dreamer_feats.shape}")
    else:
        print("DREAMER features not found.")

    market_path = PROCESSED_DIR / "market_features.csv"
    if market_path.exists():
        market_feats = pd.read_csv(market_path)
        print(f"Loaded market features: {market_feats.shape}")
    else:
        print("Market features not found.")

    output_dir = REPORTS_DIR.parent / "alignment"
    results = run_full_alignment_check(
        wesad_features=wesad_feats,
        dreamer_features=dreamer_feats,
        market_features=market_feats,
        stress_labels=stress_labels,
        output_dir=output_dir,
    )

    print(f"\nAlignment check complete.")


if __name__ == "__main__":
    main()
