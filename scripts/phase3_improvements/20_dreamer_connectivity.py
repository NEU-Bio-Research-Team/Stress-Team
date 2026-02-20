"""
Script 20 - DREAMER Connectivity Features
============================================
Phase 3+ : Advisor recommendation

Problem:
  DREAMER DE features (70 = 5 bands x 14 channels) capture per-channel band power
  but lose inter-channel connectivity information. Emotion is encoded in
  NETWORK DYNAMICS, not scalar band power per electrode.

Solution:
  Compute connectivity features from raw EEG:
    1. Coherence (frequency-domain linear coupling)
    2. Phase Locking Value (PLV) (phase synchrony)
    3. Graph centrality from connectivity matrices

  Combined with z-normalized DE features + valence target (best from Script 16).

Expected improvement:
  0.600 -> 0.68-0.75 (advisor estimate)

Usage:
    python scripts/phase3_improvements/20_dreamer_connectivity.py
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from scipy import signal as sig
from scipy.io import loadmat

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import (
    DREAMER_MAT_PATH, DREAMER_EEG_SR, DREAMER_EEG_CHANNELS,
    DREAMER_EEG_BANDPASS, DREAMER_NOTCH_FREQ, DREAMER_NOTCH_BW,
    PROCESSED_DIR, VALIDATION_DIR,
)
from src.preprocessing.filters import eeg_bandpass, eeg_notch


# ═══════════════════════════════════════════════════════════════════════
#  EEG FREQUENCY BANDS
# ═══════════════════════════════════════════════════════════════════════
EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

N_CHANNELS = len(DREAMER_EEG_CHANNELS)  # 14


# ═══════════════════════════════════════════════════════════════════════
#  CONNECTIVITY FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def compute_coherence_matrix(eeg_window, sr, band):
    """
    Compute magnitude-squared coherence between all pairs of channels
    for a specific frequency band.

    Args:
        eeg_window: (n_samples, n_channels) EEG data for one window
        sr: sampling rate
        band: (low_freq, high_freq) tuple

    Returns:
        coh_matrix: (n_channels, n_channels) coherence matrix
    """
    n_ch = eeg_window.shape[1]
    coh_matrix = np.zeros((n_ch, n_ch))
    nperseg = min(sr, eeg_window.shape[0])  # 1 second or window length

    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == j:
                coh_matrix[i, j] = 1.0
                continue
            try:
                freqs, Cxy = sig.coherence(
                    eeg_window[:, i], eeg_window[:, j],
                    fs=sr, nperseg=nperseg,
                )
                # Average coherence in the band
                band_mask = (freqs >= band[0]) & (freqs <= band[1])
                if band_mask.sum() > 0:
                    coh_val = np.mean(Cxy[band_mask])
                else:
                    coh_val = 0.0
            except Exception:
                coh_val = 0.0

            coh_matrix[i, j] = coh_val
            coh_matrix[j, i] = coh_val

    return coh_matrix


def compute_plv_matrix(eeg_window, sr, band):
    """
    Compute Phase Locking Value (PLV) between all pairs of channels
    for a specific frequency band.

    PLV = |mean(exp(j * delta_phase))| averaged over time.

    Args:
        eeg_window: (n_samples, n_channels)
        sr: sampling rate
        band: (low_freq, high_freq)

    Returns:
        plv_matrix: (n_channels, n_channels)
    """
    n_samples, n_ch = eeg_window.shape
    plv_matrix = np.zeros((n_ch, n_ch))

    # Bandpass filter for the band
    nyq = sr / 2.0
    low = max(band[0] / nyq, 0.001)
    high = min(band[1] / nyq, 0.999)

    try:
        b, a = sig.butter(4, [low, high], btype='band')
    except ValueError:
        return plv_matrix

    # Filter and extract analytic signal (Hilbert) for each channel
    phases = np.zeros((n_samples, n_ch))
    for ch in range(n_ch):
        try:
            filtered = sig.filtfilt(b, a, eeg_window[:, ch])
            analytic = sig.hilbert(filtered)
            phases[:, ch] = np.angle(analytic)
        except Exception:
            phases[:, ch] = 0.0

    # Compute PLV for each pair
    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == j:
                plv_matrix[i, j] = 1.0
                continue

            delta_phase = phases[:, i] - phases[:, j]
            plv = np.abs(np.mean(np.exp(1j * delta_phase)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    return plv_matrix


def extract_graph_features(conn_matrix):
    """
    Extract graph-theoretic features from a connectivity matrix.

    Features:
        - Node degree (strength) for each channel
        - Global efficiency
        - Mean clustering coefficient (approximated)

    Args:
        conn_matrix: (n_ch, n_ch) symmetric connectivity matrix

    Returns:
        features: dict with graph features
    """
    n_ch = conn_matrix.shape[0]
    # Zero diagonal for graph metrics
    adj = conn_matrix.copy()
    np.fill_diagonal(adj, 0)

    # Node strength (weighted degree)
    node_strength = adj.sum(axis=1)

    # Global efficiency: E = 1/N(N-1) * sum(1/d_ij) for connected pairs
    # Using connectivity as inverse distance proxy
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = np.where(adj > 0.01, adj, 0)  # threshold for meaningful connection
    global_efficiency = inv_dist.sum() / (n_ch * (n_ch - 1)) if n_ch > 1 else 0

    # Mean connectivity
    triu_vals = adj[np.triu_indices(n_ch, k=1)]
    mean_conn = np.mean(triu_vals) if len(triu_vals) > 0 else 0
    std_conn = np.std(triu_vals) if len(triu_vals) > 0 else 0

    # Asymmetry: left vs right hemisphere
    # Emotiv EPOC layout: AF3,F7,F3,FC5,T7,P7,O1 (left=0..6), O2,P8,T8,FC6,F4,F8,AF4 (right=7..13)
    left_idx = list(range(7))
    right_idx = list(range(7, 14))

    left_strength = node_strength[left_idx].mean() if left_idx else 0
    right_strength = node_strength[right_idx].mean() if right_idx else 0
    laterality = (right_strength - left_strength) / (right_strength + left_strength + 1e-10)

    # Inter-hemisphere connectivity
    inter_conn = 0
    count = 0
    for li in left_idx:
        for ri in right_idx:
            inter_conn += adj[li, ri]
            count += 1
    inter_conn = inter_conn / max(count, 1)

    # Intra-hemisphere connectivity
    intra_left = 0
    left_count = 0
    for i, li in enumerate(left_idx):
        for j, lj in enumerate(left_idx):
            if i < j:
                intra_left += adj[li, lj]
                left_count += 1
    intra_left = intra_left / max(left_count, 1)

    intra_right = 0
    right_count = 0
    for i, ri in enumerate(right_idx):
        for j, rj in enumerate(right_idx):
            if i < j:
                intra_right += adj[ri, rj]
                right_count += 1
    intra_right = intra_right / max(right_count, 1)

    return {
        "node_strength": node_strength,  # (14,) per-channel
        "global_efficiency": global_efficiency,
        "mean_conn": mean_conn,
        "std_conn": std_conn,
        "laterality": laterality,
        "inter_hemisphere": inter_conn,
        "intra_left": intra_left,
        "intra_right": intra_right,
    }


def extract_connectivity_features_for_window(eeg_window, sr):
    """
    Extract full connectivity feature vector for one EEG window.

    For each band (delta, theta, alpha, beta, gamma):
      - Coherence graph features (8 scalars + 14 node strengths)
      - PLV graph features (8 scalars + 14 node strengths)

    Total: 5 bands * 2 methods * (8 + 14) = 220 features

    Args:
        eeg_window: (n_samples, 14) EEG data
        sr: sampling rate

    Returns:
        feature_vector: (n_features,) numpy array
        feature_names: list of feature names
    """
    features = []
    names = []

    for band_name, band_range in EEG_BANDS.items():
        # Coherence
        coh_mat = compute_coherence_matrix(eeg_window, sr, band_range)
        coh_graph = extract_graph_features(coh_mat)

        # Scalar graph features
        for key in ["global_efficiency", "mean_conn", "std_conn",
                     "laterality", "inter_hemisphere", "intra_left", "intra_right"]:
            features.append(coh_graph[key])
            names.append(f"coh_{band_name}_{key}")

        # Per-channel node strength
        for ch_idx, ch_name in enumerate(DREAMER_EEG_CHANNELS):
            features.append(coh_graph["node_strength"][ch_idx])
            names.append(f"coh_{band_name}_strength_{ch_name}")

        # PLV
        plv_mat = compute_plv_matrix(eeg_window, sr, band_range)
        plv_graph = extract_graph_features(plv_mat)

        for key in ["global_efficiency", "mean_conn", "std_conn",
                     "laterality", "inter_hemisphere", "intra_left", "intra_right"]:
            features.append(plv_graph[key])
            names.append(f"plv_{band_name}_{key}")

        for ch_idx, ch_name in enumerate(DREAMER_EEG_CHANNELS):
            features.append(plv_graph["node_strength"][ch_idx])
            names.append(f"plv_{band_name}_strength_{ch_name}")

    return np.array(features, dtype=np.float64), names


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING (from raw DREAMER.mat)
# ═══════════════════════════════════════════════════════════════════════

def load_and_extract_connectivity(window_sec=2, max_windows_per_trial=None):
    """
    Load raw EEG from DREAMER.mat, preprocess, and extract connectivity features.

    Uses 2-second windows (256 samples at 128Hz) for stable connectivity estimation.

    Args:
        window_sec: window size in seconds
        max_windows_per_trial: limit windows per trial (None = all)

    Returns:
        X_conn: (N, n_features) connectivity features
        X_de:   (N, 70) DE features (z-normed within subject)
        y:      (N,) valence labels (best target from Script 16)
        subjects: (N,) subject IDs
        conn_names: list of connectivity feature names
    """
    sr = DREAMER_EEG_SR  # 128

    print(f"  Loading DREAMER.mat ({DREAMER_MAT_PATH}) ...")
    mat = loadmat(str(DREAMER_MAT_PATH), squeeze_me=True)
    dreamer_data = mat["DREAMER"]["Data"].item()
    n_subjects = len(dreamer_data)

    window_samples = window_sec * sr  # 2 * 128 = 256

    all_conn_features = []
    all_de_features = []
    all_labels = []
    all_subjects = []
    conn_names = None

    total_start = time.time()

    for si in range(n_subjects):
        subj_id = f"S{si+1:02d}"
        subj_data = dreamer_data[si]
        eeg_data = subj_data["EEG"].item()
        stimuli = eeg_data["stimuli"].item()
        valence_scores = subj_data["ScoreValence"].item()
        n_trials = len(stimuli)

        subj_conn = []
        subj_de = []
        subj_labels = []

        print(f"    {subj_id}: {n_trials} trials ...", end="", flush=True)

        for ti in range(n_trials):
            trial_eeg = stimuli[ti]  # (n_samples, 14)
            valence = int(valence_scores[ti]) if np.isscalar(valence_scores[ti]) else int(valence_scores[ti])

            # Binary valence: <=3 = low valence (negative/stress-like)
            label = 1 if valence <= 3 else 0

            # Preprocess: bandpass + notch
            eeg_filtered = eeg_bandpass(
                trial_eeg, sr,
                DREAMER_EEG_BANDPASS[0], DREAMER_EEG_BANDPASS[1],
            )
            eeg_filtered = eeg_notch(
                eeg_filtered, sr,
                freq=DREAMER_NOTCH_FREQ,
                Q=DREAMER_NOTCH_FREQ / DREAMER_NOTCH_BW,
            )

            # Baseline subtraction
            baseline_data = eeg_data["baseline"].item()
            if isinstance(baseline_data, np.ndarray) and baseline_data.ndim >= 1:
                if baseline_data.ndim == 1:
                    # Array of trial baselines
                    baseline_trial = baseline_data[ti] if ti < len(baseline_data) else baseline_data[0]
                else:
                    baseline_trial = baseline_data
                if baseline_trial is not None and hasattr(baseline_trial, 'shape') and baseline_trial.ndim == 2:
                    bl_filtered = eeg_bandpass(baseline_trial, sr,
                                               DREAMER_EEG_BANDPASS[0],
                                               DREAMER_EEG_BANDPASS[1])
                    bl_filtered = eeg_notch(bl_filtered, sr,
                                            freq=DREAMER_NOTCH_FREQ,
                                            Q=DREAMER_NOTCH_FREQ / DREAMER_NOTCH_BW)
                    bl_mean = np.mean(bl_filtered, axis=0, keepdims=True)
                    eeg_filtered = eeg_filtered - bl_mean

            # Window the signal
            n_samples_trial = eeg_filtered.shape[0]
            n_windows = n_samples_trial // window_samples

            if max_windows_per_trial is not None:
                n_windows = min(n_windows, max_windows_per_trial)

            for wi in range(n_windows):
                start = wi * window_samples
                end = start + window_samples
                window = eeg_filtered[start:end, :]

                # Connectivity features
                conn_feat, names = extract_connectivity_features_for_window(window, sr)
                subj_conn.append(conn_feat)
                if conn_names is None:
                    conn_names = names

                # DE features (for combination)
                de_feat = np.zeros(len(EEG_BANDS) * N_CHANNELS)
                for bi, (bname, (lo, hi)) in enumerate(EEG_BANDS.items()):
                    for ch in range(N_CHANNELS):
                        seg = window[:, ch]
                        try:
                            freqs_p, psd = sig.welch(seg, fs=sr, nperseg=min(128, len(seg)))
                            band_mask = (freqs_p >= lo) & (freqs_p <= hi)
                            if band_mask.sum() > 0:
                                band_power = np.mean(psd[band_mask])
                                de_val = 0.5 * np.log(2 * np.pi * np.e * max(band_power, 1e-20))
                            else:
                                de_val = 0.0
                        except Exception:
                            de_val = 0.0
                        de_feat[bi * N_CHANNELS + ch] = de_val

                subj_de.append(de_feat)
                subj_labels.append(label)

        if len(subj_conn) == 0:
            print(" skip (no windows)")
            continue

        subj_conn = np.array(subj_conn)
        subj_de = np.array(subj_de)
        subj_labels_arr = np.array(subj_labels)

        # Within-subject z-normalization (key from Script 16)
        subj_conn_mean = subj_conn.mean(axis=0)
        subj_conn_std = subj_conn.std(axis=0)
        subj_conn_std[subj_conn_std < 1e-10] = 1.0
        subj_conn = (subj_conn - subj_conn_mean) / subj_conn_std

        subj_de_mean = subj_de.mean(axis=0)
        subj_de_std = subj_de.std(axis=0)
        subj_de_std[subj_de_std < 1e-10] = 1.0
        subj_de = (subj_de - subj_de_mean) / subj_de_std

        all_conn_features.append(subj_conn)
        all_de_features.append(subj_de)
        all_labels.append(subj_labels_arr)
        all_subjects.extend([subj_id] * len(subj_labels_arr))

        print(f" {len(subj_labels_arr)} windows, {subj_labels_arr.mean():.1%} positive")

    X_conn = np.vstack(all_conn_features)
    X_de = np.vstack(all_de_features)
    y = np.concatenate(all_labels)
    subjects = np.array(all_subjects)

    elapsed = time.time() - total_start
    print(f"\n  Connectivity extraction complete: {X_conn.shape[0]} windows, "
          f"{X_conn.shape[1]} conn features, {X_de.shape[1]} DE features, "
          f"{elapsed:.0f}s")

    return X_conn, X_de, y, subjects, conn_names


# ═══════════════════════════════════════════════════════════════════════
#  LOSOCV EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def losocv_evaluate(X, y, subjects, feature_name="features"):
    """Simple LOSOCV with LogReg + RF."""
    unique_subjects = np.unique(subjects)
    lr_results = []
    rf_results = []

    for test_subj in unique_subjects:
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # NaN safety
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        # LogReg
        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs",
        )
        lr.fit(X_train_s, y_train)
        lr_pred = lr.predict(X_test_s)
        lr_prob = lr.predict_proba(X_test_s)[:, 1]
        lr_results.append({
            "subject": test_subj,
            "bal_acc": float(balanced_accuracy_score(y_test, lr_pred)),
            "f1": float(f1_score(y_test, lr_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, lr_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        })

        # RandomForest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        rf_prob = rf.predict_proba(X_test_s)[:, 1]
        rf_results.append({
            "subject": test_subj,
            "bal_acc": float(balanced_accuracy_score(y_test, rf_pred)),
            "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, rf_prob)) if len(np.unique(y_test)) > 1 else 0.5,
        })

    # Aggregate
    def agg(results):
        return {
            "bal_acc": round(float(np.mean([r["bal_acc"] for r in results])), 4),
            "bal_acc_std": round(float(np.std([r["bal_acc"] for r in results])), 4),
            "f1": round(float(np.mean([r["f1"] for r in results])), 4),
            "auc": round(float(np.mean([r["auc"] for r in results])), 4),
        }

    return {
        "LogisticRegression": {"per_subject": lr_results, "aggregate": agg(lr_results)},
        "RandomForest": {"per_subject": rf_results, "aggregate": agg(rf_results)},
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("#  DREAMER CONNECTIVITY FEATURES")
    print("#  Advisor: Emotion encoded in network dynamics, not band power")
    print("=" * 70)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Extract connectivity features ──
    # Use 2-second windows for stable PLV/coherence estimation
    # Limit windows per trial to keep computation manageable
    X_conn, X_de, y, subjects, conn_names = load_and_extract_connectivity(
        window_sec=2,
        max_windows_per_trial=50,  # ~100s per trial, cap at 50 windows
    )

    print(f"\n  Dataset: {X_conn.shape[0]} windows, "
          f"{len(np.unique(subjects))} subjects")
    print(f"  Connectivity features: {X_conn.shape[1]}")
    print(f"  DE features: {X_de.shape[1]}")
    print(f"  Valence target: {y.mean():.1%} positive (low valence)")

    # Combined features
    X_combined = np.hstack([X_de, X_conn])
    print(f"  Combined features: {X_combined.shape[1]}")

    all_results = {}

    # ── Experiment 1: DE features only (baseline, should match ~0.600) ──
    print("\n" + "=" * 70)
    print("  Experiment 1: DE features only (z-normed, valence)")
    print("=" * 70)
    de_results = losocv_evaluate(X_de, y, subjects, "DE_only")
    all_results["de_only"] = {
        "n_features": X_de.shape[1],
        "description": "DE features only (z-normed within subject)",
        **de_results,
    }
    lr_acc = de_results["LogisticRegression"]["aggregate"]["bal_acc"]
    rf_acc = de_results["RandomForest"]["aggregate"]["bal_acc"]
    print(f"  -> LogReg: {lr_acc:.4f}, RF: {rf_acc:.4f}")

    # ── Experiment 2: Connectivity features only ──
    print("\n" + "=" * 70)
    print("  Experiment 2: Connectivity features only")
    print("=" * 70)
    conn_results = losocv_evaluate(X_conn, y, subjects, "connectivity_only")
    all_results["connectivity_only"] = {
        "n_features": X_conn.shape[1],
        "description": "Connectivity features only (coherence + PLV graph features)",
        **conn_results,
    }
    lr_acc = conn_results["LogisticRegression"]["aggregate"]["bal_acc"]
    rf_acc = conn_results["RandomForest"]["aggregate"]["bal_acc"]
    print(f"  -> LogReg: {lr_acc:.4f}, RF: {rf_acc:.4f}")

    # ── Experiment 3: DE + Connectivity combined ──
    print("\n" + "=" * 70)
    print("  Experiment 3: DE + Connectivity combined")
    print("=" * 70)
    combined_results = losocv_evaluate(X_combined, y, subjects, "combined")
    all_results["de_plus_connectivity"] = {
        "n_features": X_combined.shape[1],
        "description": "DE + connectivity features (z-normed)",
        **combined_results,
    }
    lr_acc = combined_results["LogisticRegression"]["aggregate"]["bal_acc"]
    rf_acc = combined_results["RandomForest"]["aggregate"]["bal_acc"]
    print(f"  -> LogReg: {lr_acc:.4f}, RF: {rf_acc:.4f}")

    # ── Experiment 4: Connectivity only, selected bands ──
    # Beta + gamma connectivity (advisor: emotion in beta oscillations)
    print("\n" + "=" * 70)
    print("  Experiment 4: Beta+Gamma connectivity only")
    print("=" * 70)
    beta_gamma_mask = np.array([
        "beta" in name or "gamma" in name
        for name in conn_names
    ])
    X_bg = X_conn[:, beta_gamma_mask]
    bg_results = losocv_evaluate(X_bg, y, subjects, "beta_gamma_conn")
    all_results["beta_gamma_connectivity"] = {
        "n_features": int(beta_gamma_mask.sum()),
        "description": "Beta + Gamma connectivity features only",
        **bg_results,
    }
    lr_acc = bg_results["LogisticRegression"]["aggregate"]["bal_acc"]
    rf_acc = bg_results["RandomForest"]["aggregate"]["bal_acc"]
    print(f"  -> LogReg: {lr_acc:.4f}, RF: {rf_acc:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DREAMER CONNECTIVITY RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Experiment':<40} {'LR bal_acc':<12} {'RF bal_acc':<12} {'Best':<10}")
    print("-" * 80)

    baselines = {"Script 16 DE (z-norm+valence)": 0.6004}
    for name, acc in baselines.items():
        print(f"  {name:<40} {acc:<12} {'--':<12} {acc:<10}")

    best_overall = 0
    for key, res in all_results.items():
        lr = res["LogisticRegression"]["aggregate"]["bal_acc"]
        rf = res["RandomForest"]["aggregate"]["bal_acc"]
        best = max(lr, rf)
        best_overall = max(best_overall, best)
        print(f"  {key:<40} {lr:<12} {rf:<12} {best:<10}")

    # Verdict
    baseline = 0.6004
    if best_overall > baseline + 0.03:
        verdict = "CONNECTIVITY_HELPS"
        msg = f"Connectivity features improve DREAMER from {baseline:.3f} to {best_overall:.3f}"
    elif best_overall > baseline:
        verdict = "MARGINAL_IMPROVEMENT"
        msg = f"Small improvement: {baseline:.3f} -> {best_overall:.3f}"
    else:
        verdict = "NO_IMPROVEMENT"
        msg = f"Connectivity features do not improve over DE baseline ({baseline:.3f})"

    all_results["_summary"] = {
        "de_baseline": baseline,
        "best_with_connectivity": best_overall,
        "delta": round(best_overall - baseline, 4),
        "verdict": verdict,
        "message": msg,
    }

    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    # ── Save ──
    out_path = VALIDATION_DIR / "dreamer_connectivity_results.json"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
