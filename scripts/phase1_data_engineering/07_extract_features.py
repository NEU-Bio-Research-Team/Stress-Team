"""
Script 07 – Feature Extraction (All Datasets)
================================================
Extract features from preprocessed data:
  WESAD: ECG + EDA features per window
  DREAMER: EEG features per trial
  Tardis: Market microstructure features per day

Usage:
    python scripts/07_extract_features.py
    python scripts/07_extract_features.py --dataset wesad
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from config.settings import (
    PROCESSED_DIR, WESAD_EXPECTED_SUBJECTS, WESAD_CHEST_SR,
    WESAD_WRIST_EDA_SR, DREAMER_EEG_SR, ensure_dirs,
)


def extract_wesad_features():
    """Extract ECG + EDA features from preprocessed WESAD .npz files."""
    out_dir = PROCESSED_DIR / "wesad"
    npz_files = sorted(out_dir.glob("*.npz"))

    if not npz_files:
        print("  No preprocessed WESAD files found. Run 04_preprocess_wesad.py first.")
        return

    # Actual feature names matching what preprocessing computes
    RR_NAMES = ["hr_mean", "hr_std", "rmssd", "sdnn"]
    EDA_NAMES = ["eda_mean", "eda_std", "eda_slope"]

    all_rows = []
    for npz_path in npz_files:
        sid = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)

        # WESAD npz keys: ecg_windows, eda_windows, rr_features, eda_features, labels, clean_mask
        rr_feats  = data.get("rr_features",  None)  # (N, 4): hr_mean, hr_std, rmssd, sdnn
        eda_feats = data.get("eda_features", None)  # (N, 3): eda_mean, eda_std, eda_slope
        labels    = data.get("labels",       None)  # (N,)

        if rr_feats is None and eda_feats is None:
            print(f"    Skipping {sid}: no feature arrays found (expected rr_features / eda_features)")
            continue

        n_windows = (rr_feats.shape[0] if rr_feats is not None
                     else eda_feats.shape[0])
        print(f"  {sid}: extracting features from {n_windows} windows")

        rr_names  = RR_NAMES[:rr_feats.shape[1]]   if rr_feats  is not None else []
        eda_names = EDA_NAMES[:eda_feats.shape[1]]  if eda_feats is not None else []

        for i in range(n_windows):
            row = {"subject": sid, "window_idx": i}
            if labels is not None and i < len(labels):
                row["label"] = int(labels[i])
            if rr_feats is not None and i < len(rr_feats):
                for j, name in enumerate(rr_names):
                    row[name] = float(rr_feats[i, j])
            if eda_feats is not None and i < len(eda_feats):
                for j, name in enumerate(eda_names):
                    row[name] = float(eda_feats[i, j])
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = PROCESSED_DIR / "wesad_features.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} rows × {len(df.columns)} cols)")


def extract_dreamer_features():
    """Extract EEG features from preprocessed DREAMER .npz files."""
    from src.features.eeg_features import extract_eeg_features, get_feature_names

    out_dir = PROCESSED_DIR / "dreamer"
    npz_files = sorted(out_dir.glob("*.npz"))

    if not npz_files:
        print("  No preprocessed DREAMER files found. Run 05_preprocess_dreamer.py first.")
        return

    all_rows = []
    for npz_path in npz_files:
        sid = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        print(f"  {sid}: extracting features")

        # DREAMER npz keys: de_features, stress_labels, valence, arousal, dominance
        features = data.get("de_features", data.get("features", None))
        labels   = data.get("stress_labels", data.get("labels", None))

        if features is None:
            print(f"    Skipping {sid}: no de_features array found")
            continue

        for i in range(len(features)):
            row = {"subject": sid, "trial_idx": i}
            if labels is not None and i < len(labels):
                row["stress"] = int(labels[i])
            if features.ndim == 2:
                feat_names = data.get("feature_names", [f"f{j}" for j in range(features.shape[1])])
                for j, name in enumerate(feat_names):
                    row[str(name)] = float(features[i, j])
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = PROCESSED_DIR / "dreamer_features.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} rows × {len(df.columns)} cols)")


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
    parser = argparse.ArgumentParser(description="Feature extraction")
    parser.add_argument("--dataset", choices=["wesad", "dreamer", "market", "all"],
                        default="all")
    args = parser.parse_args()

    ensure_dirs()

    if args.dataset in ("wesad", "all"):
        print("\n── WESAD Feature Extraction ──")
        extract_wesad_features()

    if args.dataset in ("dreamer", "all"):
        print("\n── DREAMER Feature Extraction ──")
        extract_dreamer_features()

    if args.dataset in ("market", "all"):
        print("\n── Market Feature Extraction ──")
        extract_market_features()

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
