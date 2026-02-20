"""
DREAMER Preprocessing Pipeline
================================
Phase B of Stage 1: Process DREAMER EEG + ECG for stress proxy classification.

Pipeline:
    1. Load DREAMER.mat
    2. EEG: bandpass 0.1-40 Hz, notch 48-52 Hz
    3. Baseline subtraction (61s baseline per trial)
    4. Map V/A labels → stress proxy (low V + high A)
    5. Differential entropy per frequency band
    6. Window: 1-2 second segments
    7. Export: per-subject .npz → data/processed/dreamer/
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DREAMER_MAT_PATH, DREAMER_EEG_SR, DREAMER_EEG_CHANNELS,
    DREAMER_EEG_BANDPASS, DREAMER_NOTCH_FREQ, DREAMER_NOTCH_BW,
    DREAMER_BASELINE_SEC, DREAMER_STRESS_AROUSAL_THR,
    DREAMER_STRESS_VALENCE_THR, PROCESSED_DIR,
)
from src.data.dreamer_loader import (
    load_dreamer, DREAMERSubject, DREAMERTrial,
    get_stress_labels, get_all_labels,
)
from src.preprocessing.filters import (
    eeg_bandpass, eeg_notch, compute_differential_entropy,
    segment_signal, normalize_zscore, bandpass, lowpass,
)


# Frequency bands for differential entropy
EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}


def preprocess_trial_eeg(
    trial: DREAMERTrial,
    sr: int = DREAMER_EEG_SR,
    subtract_baseline: bool = True,
) -> Optional[Dict]:
    """
    Preprocess one trial's EEG data.
    
    Returns dict with:
        eeg_clean: (n_samples, 14) – filtered EEG
        de_features: (n_windows, 14, 5) – differential entropy per band
        ecg_clean: (n_samples, 2) or None
        stress_label: 0 or 1
        valence, arousal, dominance: int
    """
    if trial.eeg_stimulus is None:
        return None

    eeg_stim = trial.eeg_stimulus.copy()
    n_samples, n_channels = eeg_stim.shape

    # ── 1) Bandpass 0.1-40 Hz ──
    eeg_filtered = eeg_bandpass(eeg_stim, sr,
                                 DREAMER_EEG_BANDPASS[0],
                                 DREAMER_EEG_BANDPASS[1])

    # ── 2) Notch filter 48-52 Hz ──
    eeg_filtered = eeg_notch(eeg_filtered, sr,
                              freq=DREAMER_NOTCH_FREQ,
                              Q=DREAMER_NOTCH_FREQ / DREAMER_NOTCH_BW)

    # ── 3) Baseline subtraction ──
    if subtract_baseline and trial.eeg_baseline is not None:
        baseline = trial.eeg_baseline.copy()
        baseline_filtered = eeg_bandpass(baseline, sr,
                                          DREAMER_EEG_BANDPASS[0],
                                          DREAMER_EEG_BANDPASS[1])
        baseline_filtered = eeg_notch(baseline_filtered, sr,
                                       freq=DREAMER_NOTCH_FREQ,
                                       Q=DREAMER_NOTCH_FREQ / DREAMER_NOTCH_BW)
        baseline_mean = np.mean(baseline_filtered, axis=0, keepdims=True)
        eeg_filtered = eeg_filtered - baseline_mean

    # ── 4) Differential Entropy features ──
    # Window: 1 second = 128 samples
    window_samples = sr  # 1 second
    n_windows = n_samples // window_samples

    de_features = np.zeros((n_windows, n_channels, len(EEG_BANDS)))

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples
        for ch in range(n_channels):
            seg = eeg_filtered[start:end, ch]
            for bi, (band_name, (lo, hi)) in enumerate(EEG_BANDS.items()):
                de_features[w, ch, bi] = compute_differential_entropy(
                    seg, sr, (lo, hi), nperseg=min(128, len(seg))
                )

    # ── 5) ECG preprocessing (if available) ──
    ecg_clean = None
    if trial.ecg_stimulus is not None:
        ecg = trial.ecg_stimulus.copy()
        # Simple bandpass for ECG
        for ch in range(ecg.shape[1]):
            ecg[:, ch] = bandpass(ecg[:, ch], 0.5, 40, sr, order=3)
        ecg_clean = ecg

    return {
        "eeg_clean": eeg_filtered,
        "de_features": de_features,
        "ecg_clean": ecg_clean,
        "stress_label": 1 if trial.is_stress_proxy else 0,
        "valence": trial.valence,
        "arousal": trial.arousal,
        "dominance": trial.dominance,
        "subject_id": trial.subject_id,
        "trial_id": trial.trial_id,
        "n_windows": n_windows,
    }


def preprocess_subject(
    subj: DREAMERSubject,
    sr: int = DREAMER_EEG_SR,
) -> Dict:
    """Preprocess all trials for one subject."""
    all_de = []
    all_labels = []
    all_valence = []
    all_arousal = []
    all_dominance = []
    all_ecg = []
    n_total_windows = 0

    for trial in subj.trials:
        result = preprocess_trial_eeg(trial, sr)
        if result is None:
            continue

        all_de.append(result["de_features"])
        n_win = result["n_windows"]
        all_labels.extend([result["stress_label"]] * n_win)
        all_valence.extend([result["valence"]] * n_win)
        all_arousal.extend([result["arousal"]] * n_win)
        all_dominance.extend([result["dominance"]] * n_win)
        n_total_windows += n_win

        if result["ecg_clean"] is not None:
            all_ecg.append(result["ecg_clean"])

    de_concat = np.vstack(all_de) if all_de else np.array([])
    # Flatten DE: (n_windows, 14*5=70) features
    if len(de_concat.shape) == 3:
        de_flat = de_concat.reshape(de_concat.shape[0], -1)
    else:
        de_flat = de_concat

    return {
        "subject_id": subj.subject_id,
        "de_features": de_flat,           # (n_windows, 70)
        "de_features_3d": de_concat,      # (n_windows, 14, 5)
        "stress_labels": np.array(all_labels),
        "valence": np.array(all_valence),
        "arousal": np.array(all_arousal),
        "dominance": np.array(all_dominance),
        "n_windows": n_total_windows,
        "age": subj.age,
        "gender": subj.gender,
    }


def preprocess_all(
    mat_path: Path = DREAMER_MAT_PATH,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """Preprocess all DREAMER subjects and save."""
    output_dir = output_dir or (PROCESSED_DIR / "dreamer")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DREAMER PREPROCESSING")
    print("=" * 60)

    subjects = load_dreamer(mat_path)
    all_results = []
    summary_rows = []

    for subj in subjects:
        print(f"\n  Processing Subject {subj.subject_id} ...")
        result = preprocess_subject(subj)

        # Save per-subject
        out_path = output_dir / f"S{subj.subject_id:02d}_preprocessed.npz"
        np.savez_compressed(
            out_path,
            de_features=result["de_features"],
            de_features_3d=result["de_features_3d"],
            stress_labels=result["stress_labels"],
            valence=result["valence"],
            arousal=result["arousal"],
            dominance=result["dominance"],
        )
        print(f"    → Saved {out_path.name} ({result['n_windows']} windows)")

        n_stress = int(result["stress_labels"].sum())
        summary_rows.append({
            "subject_id": subj.subject_id,
            "age": subj.age,
            "gender": subj.gender,
            "n_windows": result["n_windows"],
            "n_stress": n_stress,
            "stress_pct": f"{n_stress / max(result['n_windows'], 1) * 100:.1f}%",
            "n_features": result["de_features"].shape[1]
                          if result["de_features"].ndim == 2 else 0,
        })
        all_results.append(result)

    # Summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "preprocessing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[dreamer] Summary → {summary_path}")
    print(summary_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    from config.settings import ensure_dirs
    ensure_dirs()
    preprocess_all()
