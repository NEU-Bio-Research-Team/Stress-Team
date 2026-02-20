"""
Data Loader for Validation Phase
==================================
Unified loading of features and raw signals for all validation experiments.
Handles WESAD (ECG+EDA), DREAMER (EEG DE), and optional market features.

Usage:
    from src.validation.data_loader import load_wesad_features, load_dreamer_features
    X, y, subjects, feature_cols = load_wesad_features()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
from config.settings import PROCESSED_DIR


# ─────────────────────── WESAD ───────────────────────

def load_wesad_features(
    feature_subset: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load WESAD extracted features from CSV.

    Args:
        feature_subset: list of feature names to keep, e.g. ["hr_mean","rmssd","sdnn","hr_std"]
                        If None, all 7 features are loaded.

    Returns:
        X:            (N, D) feature matrix
        y:            (N,)   binary labels (0=non-stress, 1=stress)
        subjects:     (N,)   subject IDs (str, e.g. "S2")
        feature_cols: list of feature column names used
    """
    csv_path = PROCESSED_DIR / "wesad_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"WESAD features not found at {csv_path}. "
            "Run `python scripts/07_extract_features.py` first."
        )

    df = pd.read_csv(csv_path)

    # Extract clean subject IDs
    df["subject"] = df["subject"].astype(str).str.replace("_preprocessed", "", regex=False)

    # Identify feature columns
    all_feature_cols = [c for c in df.columns if c not in ("subject", "window_idx", "label")]
    if feature_subset:
        missing = set(feature_subset) - set(all_feature_cols)
        if missing:
            raise ValueError(f"Features not found in CSV: {missing}")
        feature_cols = feature_subset
    else:
        feature_cols = all_feature_cols

    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int32)
    subjects = df["subject"].values

    return X, y, subjects, feature_cols


def load_wesad_raw_windows() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw ECG + EDA windows from per-subject .npz files.

    Returns:
        ecg_windows: (N, 3500) raw filtered ECG
        eda_windows: (N, 3500) raw filtered EDA
        labels:      (N,) binary stress labels
        subjects:    (N,) subject IDs (str)
    """
    out_dir = PROCESSED_DIR / "wesad"
    npz_files = sorted(out_dir.glob("S*_preprocessed.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No preprocessed .npz files found in {out_dir}. "
            "Run `python scripts/04_preprocess_wesad.py` first."
        )

    all_ecg, all_eda, all_labels, all_subjects = [], [], [], []

    for npz_file in npz_files:
        data = np.load(npz_file)
        subj = npz_file.stem.replace("_preprocessed", "")
        n = len(data["labels"])

        all_ecg.append(data["ecg_windows"])
        all_eda.append(data["eda_windows"])
        all_labels.append(data["labels"])
        all_subjects.extend([subj] * n)

    return (
        np.vstack(all_ecg),
        np.vstack(all_eda),
        np.concatenate(all_labels),
        np.array(all_subjects),
    )


# ─────────────────────── DREAMER ───────────────────────

def load_dreamer_features(
    exclude_subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load DREAMER extracted features from CSV.

    Args:
        exclude_subjects: list of subject IDs to exclude (e.g. ["S10","S17","S21","S23"])

    Returns:
        X:            (N, 70) DE feature matrix
        y:            (N,)    binary stress labels
        subjects:     (N,)    subject IDs (str)
        feature_cols: list of feature column names
    """
    csv_path = PROCESSED_DIR / "dreamer_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"DREAMER features not found at {csv_path}. "
            "Run `python scripts/07_extract_features.py` first."
        )

    df = pd.read_csv(csv_path)
    df["subject"] = df["subject"].astype(str).str.replace("_preprocessed", "", regex=False)

    if exclude_subjects:
        df = df[~df["subject"].isin(exclude_subjects)].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("subject", "trial_idx", "stress")]

    X = df[feature_cols].values.astype(np.float64)
    y = df["stress"].values.astype(np.int32)
    subjects = df["subject"].values

    return X, y, subjects, feature_cols


def load_dreamer_raw_3d(
    exclude_subjects: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DREAMER 3D DE features from per-subject .npz files.

    Returns:
        de_features_3d: (N, 14, 5)  channels × bands
        labels:         (N,) binary stress labels
        subjects:       (N,) subject IDs (str)
    """
    out_dir = PROCESSED_DIR / "dreamer"
    npz_files = sorted(out_dir.glob("S*_preprocessed.npz"))

    if not npz_files:
        raise FileNotFoundError(
            f"No preprocessed .npz files found in {out_dir}. "
            "Run `python scripts/05_preprocess_dreamer.py` first."
        )

    all_feats, all_labels, all_subjects = [], [], []

    for npz_file in npz_files:
        subj = npz_file.stem.replace("_preprocessed", "")
        if exclude_subjects and subj in exclude_subjects:
            continue
        data = np.load(npz_file)
        n = len(data["stress_labels"])

        all_feats.append(data["de_features_3d"])
        all_labels.append(data["stress_labels"])
        all_subjects.extend([subj] * n)

    return (
        np.vstack(all_feats),
        np.concatenate(all_labels),
        np.array(all_subjects),
    )


# ─────────────────────── Utility ───────────────────────

def describe_dataset(X, y, subjects, name="Dataset"):
    """Print summary statistics for a loaded dataset."""
    unique_subj = np.unique(subjects)
    print(f"\n{'='*60}")
    print(f"  {name} Summary")
    print(f"{'='*60}")
    print(f"  Samples:       {len(y):,}")
    print(f"  Features:      {X.shape[1]}")
    print(f"  Subjects:      {len(unique_subj)}")
    print(f"  Stress ratio:  {y.mean():.3f} ({y.sum():,} / {len(y):,})")
    print(f"  Class balance: {1 - y.mean():.1%} non-stress / {y.mean():.1%} stress")
    print(f"  Feature range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  NaN count:     {np.isnan(X).sum()}")
    print(f"{'='*60}\n")
