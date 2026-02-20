"""
WESAD Preprocessing Pipeline
==============================
Phase A of Stage 1: Process WESAD chest ECG + EDA for stress classification.

Pipeline:
    1. Load subject .pkl / respiban data
    2. ECG: bandpass 0.5-40 Hz, R-peak detection (Pan-Tompkins)
    3. EDA: lowpass filter, ACC-based artifact rejection
    4. Window: 5-second non-overlapping
    5. Labels: binary (stress=1 vs non-stress=0) or 4-class
    6. Export: per-subject .npz files → data/processed/wesad/
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    WESAD_RAW_DIR, WESAD_CHEST_SR, WESAD_LABELS,
    WESAD_ECG_BANDPASS, WESAD_WINDOW_SEC, WESAD_IBI_RANGE,
    PROCESSED_DIR,
)
from src.data.wesad_loader import (
    load_all_subjects, WESADSubject, discover_subjects, load_subject,
)
from src.preprocessing.filters import (
    bandpass, lowpass, detect_r_peaks, compute_rr_intervals,
    reject_rr_outliers, eda_lowpass, eda_artifact_mask,
    segment_signal, normalize_zscore,
)


def _make_binary_labels(labels: np.ndarray) -> np.ndarray:
    """Map WESAD labels to binary: stress(2)=1, everything else=0."""
    return (labels == 2).astype(np.int32)


def _make_4class_labels(labels: np.ndarray) -> np.ndarray:
    """Map to 4 classes: 0=baseline(1), 1=stress(2), 2=amusement(3), 3=meditation(4).
    Undefined labels (0, 5-7) → -1 (to be masked out)."""
    mapping = {1: 0, 2: 1, 3: 2, 4: 3}
    out = np.full_like(labels, -1, dtype=np.int32)
    for orig, new in mapping.items():
        out[labels == orig] = new
    return out


def preprocess_subject(
    subj: WESADSubject,
    window_sec: float = WESAD_WINDOW_SEC,
    task: str = "binary",  # "binary" or "4class"
    sr: int = WESAD_CHEST_SR,
) -> Optional[Dict]:
    """
    Full preprocessing pipeline for one WESAD subject.
    
    Returns dict with:
        ecg_windows: (n_windows, window_samples)
        eda_windows: (n_windows, window_samples)
        labels: (n_windows,)
        rr_features: (n_windows, 4) – [hr_mean, hr_std, rmssd, sdnn]
        eda_features: (n_windows, 3) – [eda_mean, eda_std, eda_slope]
        clean_mask: (n_windows,) – True if window is artifact-free
        subject_id: str
    """
    if subj.chest_ecg is None:
        print(f"  [!] {subj.subject_id}: No ECG data – skipping")
        return None

    ecg_raw = subj.chest_ecg.copy()
    n_samples = len(ecg_raw)
    print(f"  [{subj.subject_id}] ECG: {n_samples} samples "
          f"({n_samples / sr:.0f}s)")

    # ── 1) ECG bandpass 0.5-40 Hz ──
    ecg_filtered = bandpass(ecg_raw, WESAD_ECG_BANDPASS[0],
                            WESAD_ECG_BANDPASS[1], sr)

    # ── 2) EDA lowpass ──
    eda_filtered = None
    if subj.chest_eda is not None:
        eda_filtered = eda_lowpass(subj.chest_eda, sr, cutoff=5.0)

    # ── 3) ACC magnitude for artifact mask ──
    acc_mag = None
    clean_mask_full = np.ones(n_samples, dtype=bool)
    if subj.chest_acc is not None and eda_filtered is not None:
        acc_mag = np.sqrt(np.sum(subj.chest_acc ** 2, axis=1))
        clean_mask_full = eda_artifact_mask(eda_filtered, acc_mag)

    # ── 4) Labels ──
    if subj.labels is not None:
        if task == "binary":
            labels_full = _make_binary_labels(subj.labels)
        else:
            labels_full = _make_4class_labels(subj.labels)
    else:
        labels_full = np.zeros(n_samples, dtype=np.int32)

    # ── 5) Windowing ──
    window_samples = int(window_sec * sr)

    ecg_windows = segment_signal(ecg_filtered, window_samples)
    n_windows = ecg_windows.shape[0]

    # EDA windows
    if eda_filtered is not None:
        eda_windows = segment_signal(eda_filtered, window_samples)
        eda_windows = eda_windows[:n_windows]
    else:
        eda_windows = np.zeros((n_windows, window_samples))

    # Label per window: majority vote
    labels_windowed = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        seg_labels = labels_full[start:end]
        if task == "4class":
            valid = seg_labels[seg_labels >= 0]
            if len(valid) > 0:
                labels_windowed[i] = np.bincount(valid).argmax()
            else:
                labels_windowed[i] = -1
        else:
            labels_windowed[i] = int(np.mean(seg_labels) > 0.5)

    # Clean mask per window
    clean_mask = np.ones(n_windows, dtype=bool)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        if end <= len(clean_mask_full):
            clean_mask[i] = clean_mask_full[start:end].mean() > 0.8

    # ── 6) Feature extraction per window ──
    rr_features = np.zeros((n_windows, 4))  # hr_mean, hr_std, rmssd, sdnn
    eda_features = np.zeros((n_windows, 3))  # mean, std, slope

    for i in range(n_windows):
        # ECG → R-peaks → HR/HRV
        ecg_win = ecg_windows[i]
        r_peaks = detect_r_peaks(ecg_win, sr)
        rr = compute_rr_intervals(r_peaks, sr)
        rr = reject_rr_outliers(rr, WESAD_IBI_RANGE[0], WESAD_IBI_RANGE[1])

        if len(rr) > 1:
            hr = 60000.0 / rr  # bpm
            rr_features[i, 0] = np.mean(hr)
            rr_features[i, 1] = np.std(hr)
            rr_features[i, 2] = np.sqrt(np.mean(np.diff(rr) ** 2))  # RMSSD
            rr_features[i, 3] = np.std(rr)  # SDNN

        # EDA features
        eda_win = eda_windows[i]
        eda_features[i, 0] = np.mean(eda_win)
        eda_features[i, 1] = np.std(eda_win)
        # Slope via linear regression
        t = np.arange(len(eda_win))
        if np.std(eda_win) > 0:
            eda_features[i, 2] = np.polyfit(t, eda_win, 1)[0]

    return {
        "subject_id": subj.subject_id,
        "ecg_windows": ecg_windows,
        "eda_windows": eda_windows,
        "labels": labels_windowed,
        "rr_features": rr_features,
        "eda_features": eda_features,
        "clean_mask": clean_mask,
        "window_sec": window_sec,
        "sr": sr,
        "task": task,
    }


def preprocess_all(
    root: Path = WESAD_RAW_DIR,
    output_dir: Optional[Path] = None,
    task: str = "binary",
    subject_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Preprocess all (or a subset of) WESAD subjects and save to output_dir.

    Args:
        subject_ids: Optional list of subject IDs to process (e.g. ['S2','S3']).
                     If None, all discovered subjects are processed.
    Saves per-subject .npz and a combined summary CSV.
    """
    output_dir = output_dir or (PROCESSED_DIR / "wesad")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"WESAD PREPROCESSING (task={task})")
    print("=" * 60)

    subjects = load_all_subjects(root)
    if subject_ids is not None:
        subjects = {sid: subj for sid, subj in subjects.items()
                    if sid in subject_ids}
    all_results = []
    summary_rows = []

    for sid, subj in sorted(subjects.items()):
        result = preprocess_subject(subj, task=task)
        if result is None:
            continue

        # Save per-subject
        out_path = output_dir / f"{sid}_preprocessed.npz"
        np.savez_compressed(
            out_path,
            ecg_windows=result["ecg_windows"],
            eda_windows=result["eda_windows"],
            labels=result["labels"],
            rr_features=result["rr_features"],
            eda_features=result["eda_features"],
            clean_mask=result["clean_mask"],
        )
        print(f"  → Saved {out_path.name} "
              f"({result['ecg_windows'].shape[0]} windows)")

        summary_rows.append({
            "subject_id": sid,
            "n_windows": result["ecg_windows"].shape[0],
            "n_clean": int(result["clean_mask"].sum()),
            "n_stress": int((result["labels"] == 1).sum()) if task == "binary" else "N/A",
            "stress_pct": f"{(result['labels'] == 1).mean() * 100:.1f}%"
                          if task == "binary" else "N/A",
        })
        all_results.append(result)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[wesad] Summary saved → {summary_path}")
    print(summary_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    from config.settings import ensure_dirs
    ensure_dirs()
    preprocess_all(task="binary")
