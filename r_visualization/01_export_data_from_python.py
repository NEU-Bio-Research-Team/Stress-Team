"""
Export .npz preprocessed data to CSV format for R visualization.
Run this ONCE before executing the R scripts.

Usage: python 01_export_data_from_python.py
"""

import numpy as np
import pandas as pd
import os
import scipy.io
from pathlib import Path

# --- Paths ---
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"
EXPORT = Path(__file__).resolve().parent / "data_export"
EXPORT.mkdir(exist_ok=True)

print("=" * 60)
print("Exporting data for R visualization")
print("=" * 60)

# ============================================================
# 1. WESAD: Export one subject raw + preprocessed signals
# ============================================================
print("\n[1/4] Exporting WESAD raw + preprocessed signals...")

# Find one available WESAD subject
wesad_processed = PROCESSED / "wesad"
wesad_files = sorted(wesad_processed.glob("S*_preprocessed.npz"))

if wesad_files:
    npz_path = wesad_files[0]  # First available subject
    subj_id = npz_path.stem.split("_")[0]
    print(f"  Using subject: {subj_id}")

    data = np.load(npz_path, allow_pickle=True)
    print(f"  Arrays in file: {list(data.keys())}")

    # Export each array
    for key in data.keys():
        arr = data[key]
        if arr.ndim == 1:
            df = pd.DataFrame({key: arr})
        elif arr.ndim == 2:
            cols = [f"{key}_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)
        else:
            # Flatten higher dims
            reshaped = arr.reshape(arr.shape[0], -1)
            cols = [f"{key}_{i}" for i in range(reshaped.shape[1])]
            df = pd.DataFrame(reshaped, columns=cols)

        out_path = EXPORT / f"wesad_{subj_id}_{key}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Exported: {out_path.name} (shape: {arr.shape})")

    # Try to load raw ECG from WESAD extracted data
    wesad_raw_dir = RAW / "wesad" / "WESAD_extracted"
    if wesad_raw_dir.exists():
        import pickle
        subj_dirs = sorted(wesad_raw_dir.glob(f"{subj_id}"))
        if not subj_dirs:
            subj_dirs = sorted(wesad_raw_dir.glob("S*"))
        if subj_dirs:
            pkl_path = subj_dirs[0] / f"{subj_dirs[0].name}.pkl"
            if pkl_path.exists():
                print(f"  Loading raw data from: {pkl_path.name}")
                with open(pkl_path, "rb") as f:
                    raw = pickle.load(f, encoding="latin1")

                # Extract raw chest ECG (first 35000 samples = 50 seconds at 700 Hz)
                n_samples = 35000  # 50 seconds
                if "signal" in raw and "chest" in raw["signal"]:
                    chest = raw["signal"]["chest"]
                    ecg_raw = chest["ECG"][:n_samples].flatten()
                    eda_raw = chest["EDA"][:n_samples].flatten()
                    label_raw = raw["label"][:n_samples].flatten()

                    raw_df = pd.DataFrame({
                        "sample_idx": np.arange(n_samples),
                        "time_s": np.arange(n_samples) / 700.0,
                        "ecg_raw": ecg_raw,
                        "eda_raw": eda_raw,
                        "label": label_raw
                    })
                    raw_df.to_csv(EXPORT / f"wesad_{subj_id}_raw_signals.csv", index=False)
                    print(f"  Exported: wesad_{subj_id}_raw_signals.csv ({n_samples} samples)")

                    # Also export filtered version (apply the same pipeline for demo)
                    from scipy.signal import butter, filtfilt

                    # ECG bandpass 0.5-40 Hz
                    b, a = butter(4, [0.5 / 350, 40 / 350], btype="band")
                    ecg_filt = filtfilt(b, a, ecg_raw)

                    # EDA lowpass 5 Hz
                    b2, a2 = butter(4, 5 / 350, btype="low")
                    eda_filt = filtfilt(b2, a2, eda_raw)

                    filt_df = pd.DataFrame({
                        "sample_idx": np.arange(n_samples),
                        "time_s": np.arange(n_samples) / 700.0,
                        "ecg_filtered": ecg_filt,
                        "eda_filtered": eda_filt,
                        "label": label_raw
                    })
                    filt_df.to_csv(EXPORT / f"wesad_{subj_id}_filtered_signals.csv", index=False)
                    print(f"  Exported: wesad_{subj_id}_filtered_signals.csv")

                    # R-peak detection for demo
                    from scipy.signal import find_peaks
                    b3, a3 = butter(4, [5 / 350, 15 / 350], btype="band")
                    ecg_qrs = filtfilt(b3, a3, ecg_raw)
                    ecg_sq = np.diff(ecg_qrs) ** 2
                    threshold = np.mean(ecg_sq) + 0.3 * np.std(ecg_sq)
                    peaks, _ = find_peaks(ecg_sq, height=threshold, distance=int(0.25 * 700))

                    rr_intervals = np.diff(peaks) / 700.0 * 1000  # ms
                    valid = (rr_intervals >= 250) & (rr_intervals <= 2000)
                    rr_clean = rr_intervals[valid]
                    rr_times = peaks[1:][valid] / 700.0

                    rr_df = pd.DataFrame({
                        "time_s": rr_times,
                        "rr_interval_ms": rr_clean
                    })
                    rr_df.to_csv(EXPORT / f"wesad_{subj_id}_rr_intervals.csv", index=False)
                    print(f"  Exported: wesad_{subj_id}_rr_intervals.csv ({len(rr_clean)} intervals)")
else:
    print("  WARNING: No WESAD preprocessed files found!")

# ============================================================
# 2. DREAMER: Export one subject raw + preprocessed EEG
# ============================================================
print("\n[2/4] Exporting DREAMER raw + preprocessed signals...")

dreamer_processed = PROCESSED / "dreamer"
dreamer_files = sorted(dreamer_processed.glob("S*_preprocessed.npz"))

if dreamer_files:
    npz_path = dreamer_files[0]
    subj_id = npz_path.stem.split("_")[0]
    print(f"  Using subject: {subj_id}")

    data = np.load(npz_path, allow_pickle=True)
    print(f"  Arrays in file: {list(data.keys())}")

    for key in data.keys():
        arr = data[key]
        if arr.ndim == 1:
            df = pd.DataFrame({key: arr})
        elif arr.ndim == 2:
            cols = [f"{key}_{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, columns=cols)
        else:
            reshaped = arr.reshape(arr.shape[0], -1)
            cols = [f"{key}_{i}" for i in range(reshaped.shape[1])]
            df = pd.DataFrame(reshaped, columns=cols)

        out_path = EXPORT / f"dreamer_{subj_id}_{key}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Exported: {out_path.name} (shape: {arr.shape})")

    # Try to load raw DREAMER from .mat
    mat_path = RAW / "dreamer" / "DREAMER.mat"
    if mat_path.exists():
        print(f"  Loading raw DREAMER.mat...")
        try:
            mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
            dreamer_data = mat.get("DREAMER", None)
            if dreamer_data is not None:
                # Extract first subject, first trial raw EEG
                subj_data = dreamer_data["Data"].item()[0]
                trial_data = subj_data["EEG"].item()["stimuli"].item()[0]

                # First trial: first 10 seconds (1280 samples at 128 Hz)
                n_samples = min(1280, trial_data.shape[0])
                eeg_raw = trial_data[:n_samples, :]  # (n_samples, 14 channels)

                channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
                raw_eeg_df = pd.DataFrame(eeg_raw, columns=channels)
                raw_eeg_df.insert(0, "sample_idx", np.arange(n_samples))
                raw_eeg_df.insert(1, "time_s", np.arange(n_samples) / 128.0)
                raw_eeg_df.to_csv(EXPORT / f"dreamer_{subj_id}_raw_eeg.csv", index=False)
                print(f"  Exported: dreamer_{subj_id}_raw_eeg.csv ({n_samples} samples, 14 ch)")

                # Apply filtering for demo
                from scipy.signal import butter, filtfilt, iirnotch

                # Bandpass 0.1-40 Hz
                b, a = butter(4, [0.1 / 64, 40 / 64], btype="band")
                eeg_filt = filtfilt(b, a, eeg_raw, axis=0)

                # Notch 50 Hz
                b_n, a_n = iirnotch(50, 25, fs=128)
                eeg_filt = filtfilt(b_n, a_n, eeg_filt, axis=0)

                filt_eeg_df = pd.DataFrame(eeg_filt, columns=channels)
                filt_eeg_df.insert(0, "sample_idx", np.arange(n_samples))
                filt_eeg_df.insert(1, "time_s", np.arange(n_samples) / 128.0)
                filt_eeg_df.to_csv(EXPORT / f"dreamer_{subj_id}_filtered_eeg.csv", index=False)
                print(f"  Exported: dreamer_{subj_id}_filtered_eeg.csv")

                # Baseline subtraction demo
                baseline_data = subj_data["EEG"].item()["baseline"].item()[0]
                baseline_n = min(61 * 128, baseline_data.shape[0])
                baseline_seg = baseline_data[:baseline_n, :]
                baseline_filt = filtfilt(b, a, baseline_seg, axis=0)
                baseline_filt = filtfilt(b_n, a_n, baseline_filt, axis=0)
                baseline_mean = baseline_filt.mean(axis=0)

                eeg_corrected = eeg_filt - baseline_mean
                corr_eeg_df = pd.DataFrame(eeg_corrected, columns=channels)
                corr_eeg_df.insert(0, "sample_idx", np.arange(n_samples))
                corr_eeg_df.insert(1, "time_s", np.arange(n_samples) / 128.0)
                corr_eeg_df.to_csv(EXPORT / f"dreamer_{subj_id}_baseline_corrected_eeg.csv", index=False)
                print(f"  Exported: dreamer_{subj_id}_baseline_corrected_eeg.csv")

            else:
                print("  WARNING: Could not parse DREAMER.mat structure")
        except Exception as e:
            print(f"  WARNING: Failed to load DREAMER.mat: {e}")
            print("  (Raw EEG plots will use synthetic demo data)")
else:
    print("  WARNING: No DREAMER preprocessed files found!")

# ============================================================
# 3. Copy validation JSONs (already readable by R)
# ============================================================
print("\n[3/4] Validation JSONs already in reports/validation/ (no export needed)")

# ============================================================
# 4. Export preprocessing summaries
# ============================================================
print("\n[4/4] Exporting preprocessing summaries...")

for dataset in ["wesad", "dreamer"]:
    summary_path = PROCESSED / dataset / "preprocessing_summary.csv"
    if summary_path.exists():
        import shutil
        dest = EXPORT / f"{dataset}_preprocessing_summary.csv"
        shutil.copy2(summary_path, dest)
        print(f"  Copied: {dest.name}")

print("\n" + "=" * 60)
print("Export complete! You can now run the R scripts.")
print("=" * 60)
