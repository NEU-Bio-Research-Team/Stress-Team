"""
Script 21 - R-R Interval Sequence Model (WESAD)
==================================================
Phase 3+: Advisor + Thesis Core Hypothesis Test

Core Question:
  The stress signal lives in beat-to-beat TIMING (HRV), not ECG waveform SHAPE.
  If we give a CNN timing information directly (R-R intervals), can it match or
  beat LogReg on handcrafted HRV features?

  This is a THEORY TEST, not a model tweak:
    - If RR-CNN > LogReg: nonlinear temporal patterns exist beyond mean/std/rmssd
    - If RR-CNN ≈ LogReg: handcrafted HRV features are already sufficient
    - If RR-CNN < LogReg: CNN architecture not suited for short sequences

Design:
  1. Extract R-peaks from full continuous ECG (not 5s windows)
  2. Compute R-R interval series per subject
  3. Window R-R series with N-beat windows (default: 30 beats ≈ 25-35s)
  4. Experiment A: 1D-CNN on raw R-R sequences (timing domain)
  5. Experiment B: LogReg on HRV features from same 30-beat windows
  6. Experiment C: LogReg on original 5s-window features (reference)
  7. Inner-CV threshold for CNN (learned from Script 19)

Expected:
  - LogReg 30-beat > LogReg 5s (more R-R intervals = more stable HRV)
  - RR-CNN ≈ LogReg 30-beat (linear features sufficient for d=1.55)

Advisor framing:
  "This is not about building a better classifier. It's about proving that
  representation-signal alignment, not model complexity, determines performance."

Usage:
    python scripts/phase3_improvements/21_rr_interval_model.py
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from config.settings import (
    WESAD_RAW_DIR, WESAD_CHEST_SR, WESAD_ECG_BANDPASS,
    WESAD_IBI_RANGE, VALIDATION_DIR,
)
from src.data.wesad_loader import discover_subjects, load_subject
from src.preprocessing.filters import (
    bandpass, detect_r_peaks, compute_rr_intervals, reject_rr_outliers,
)


# =====================================================================
#  R-R INTERVAL EXTRACTION FROM CONTINUOUS ECG
# =====================================================================

def extract_rr_for_subject(subj, sr=WESAD_CHEST_SR):
    """
    Extract R-R intervals from full continuous ECG recording.

    Returns:
        rr_ms:       (M,) R-R intervals in milliseconds
        rr_times:    (M,) time of each R-R interval (seconds from start)
        labels_full: (N,) per-sample binary labels (stress=1)
    """
    ecg_raw = subj.chest_ecg.copy()

    # Bandpass 0.5-40 Hz
    ecg_filtered = bandpass(ecg_raw, WESAD_ECG_BANDPASS[0],
                            WESAD_ECG_BANDPASS[1], sr)

    # Detect R-peaks on full recording
    r_peaks = detect_r_peaks(ecg_filtered, sr)

    # Compute R-R intervals
    rr_ms = compute_rr_intervals(r_peaks, sr)
    rr_ms = reject_rr_outliers(rr_ms, WESAD_IBI_RANGE[0], WESAD_IBI_RANGE[1])

    # Time of each R-R interval = midpoint between consecutive R-peaks
    # Use the second peak's time as the R-R interval's timestamp
    rr_times = r_peaks[1:len(rr_ms)+1] / sr  # seconds

    # Binary labels
    labels_full = (subj.labels == 2).astype(np.int32)

    return rr_ms, rr_times, labels_full


def window_rr_intervals(rr_ms, rr_times, labels_full, sr,
                         n_beats=30, stride_beats=15):
    """
    Create fixed-length R-R interval windows with labels.

    Args:
        rr_ms:       (M,) R-R intervals in ms
        rr_times:    (M,) time of each interval in seconds
        labels_full: (N,) per-sample labels at sr Hz
        sr:          sampling rate (for label lookup)
        n_beats:     number of R-R intervals per window
        stride_beats: stride in R-R intervals

    Returns:
        windows:  (W, n_beats) R-R interval windows in ms
        labels:   (W,) majority-vote binary labels per window
        hrv_feat: (W, 7) handcrafted HRV features per window
    """
    M = len(rr_ms)
    if M < n_beats:
        return np.array([]).reshape(0, n_beats), np.array([]), np.zeros((0, 7))

    windows = []
    w_labels = []
    hrv_feats = []

    for start in range(0, M - n_beats + 1, stride_beats):
        end = start + n_beats
        rr_win = rr_ms[start:end]
        windows.append(rr_win)

        # Label: majority vote from the time span of this window
        t_start = rr_times[start]
        t_end = rr_times[min(end - 1, len(rr_times) - 1)]
        sample_start = int(t_start * sr)
        sample_end = int(t_end * sr)
        sample_end = min(sample_end, len(labels_full) - 1)
        sample_start = max(0, sample_start)

        if sample_end > sample_start:
            seg_labels = labels_full[sample_start:sample_end]
            label = int(np.mean(seg_labels) > 0.5)
        else:
            label = 0
        w_labels.append(label)

        # Extract HRV features from this R-R window
        hrv = _compute_hrv_features(rr_win)
        hrv_feats.append(hrv)

    return (
        np.array(windows, dtype=np.float32),
        np.array(w_labels, dtype=np.int32),
        np.array(hrv_feats, dtype=np.float64),
    )


def _compute_hrv_features(rr_ms):
    """
    Compute 7 HRV features from an R-R interval window.
    Same features as WESAD pipeline for fair comparison.

    Returns: [hr_mean, hr_std, rmssd, sdnn, pnn50, rr_mean, rr_std]
    """
    if len(rr_ms) < 2:
        return np.zeros(7)

    hr = 60000.0 / rr_ms
    diff_rr = np.diff(rr_ms)

    return np.array([
        np.mean(hr),                                    # hr_mean
        np.std(hr),                                     # hr_std
        np.sqrt(np.mean(diff_rr ** 2)),                 # rmssd
        np.std(rr_ms),                                  # sdnn
        np.sum(np.abs(diff_rr) > 50) / len(diff_rr),   # pnn50 (fraction)
        np.mean(rr_ms),                                 # rr_mean
        np.std(rr_ms),                                  # rr_std
    ])


# =====================================================================
#  1D-CNN FOR R-R INTERVAL SEQUENCES
# =====================================================================

class RRCNN1D(nn.Module):
    """
    Tiny 1D-CNN for R-R interval sequences.
    Input: (batch, 1, n_beats) R-R intervals

    Much smaller than ECG CNN because:
    - Input is 30 values (not 3500)
    - Signal is already in timing domain
    - Only needs to capture temporal structure in beat-to-beat variation
    """
    def __init__(self, n_beats=30, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 30) -> (32, 15)
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            # Block 2: (32, 15) -> (64, 7)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3: (64, 7) -> (64, 3)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Count params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._param_count = (total, trainable)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        return x


class RRBiLSTM(nn.Module):
    """
    Bidirectional LSTM for R-R interval sequences.
    Better suited for sequential temporal data than CNN.
    """
    def __init__(self, n_beats=30, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        total = sum(p.numel() for p in self.parameters())
        self._param_count = (total, total)

    def forward(self, x):
        # x: (batch, 1, n_beats) -> need (batch, n_beats, 1) for LSTM
        x = x.transpose(1, 2)  # (batch, n_beats, 1)
        output, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from both directions
        # h_n: (num_layers*2, batch, hidden) for bidirectional
        h_fwd = h_n[-2]  # last layer forward
        h_bwd = h_n[-1]  # last layer backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, hidden*2)
        return self.classifier(h_cat)


# =====================================================================
#  TRAINING UTILITIES
# =====================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        logits = model(inputs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())
    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold(probs, labels, n_steps=200):
    best_t, best_score = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, n_steps):
        preds = (probs >= t).astype(int)
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t), float(best_score)


# =====================================================================
#  LOSOCV FOR R-R CNN
# =====================================================================

def losocv_rr_cnn(
    model_factory,
    rr_windows,     # (N, n_beats)
    labels,         # (N,)
    subjects,       # (N,)
    model_name="RRCNN1D",
    n_epochs=60,
    batch_size=256,
    lr=1e-3,
    patience=12,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_subjects = np.unique(subjects)
    per_subject = []

    for fold_idx, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        rr_train = rr_windows[train_mask].copy()
        rr_test = rr_windows[test_mask].copy()
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        # In-fold normalization (z-score on train)
        rr_mean = rr_train.mean()
        rr_std = max(rr_train.std(), 1e-10)
        rr_train = (rr_train - rr_mean) / rr_std
        rr_test = (rr_test - rr_mean) / rr_std

        # Tensors: (N, 1, n_beats) for Conv1D
        train_t = torch.FloatTensor(rr_train).unsqueeze(1)
        test_t = torch.FloatTensor(rr_test).unsqueeze(1)

        # DataLoader with balanced sampling
        n_pos = max(y_train.sum(), 1)
        n_neg = max((1 - y_train).sum(), 1)
        sample_weights = np.where(y_train == 1, len(y_train) / (2*n_pos),
                                  len(y_train) / (2*n_neg))
        sampler = WeightedRandomSampler(
            torch.DoubleTensor(sample_weights), len(y_train), replacement=True
        )

        train_ds = TensorDataset(train_t, torch.LongTensor(y_train))
        test_ds = TensorDataset(test_t, torch.LongTensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size*2,
                                 shuffle=False, num_workers=0)

        # Inner val split (1 subject from train) for early stopping + threshold
        train_subjects_pool = unique_subjects[unique_subjects != test_subj]
        val_subj = np.random.choice(train_subjects_pool)
        inner_val_mask = (subjects == val_subj) & train_mask

        rr_val = (rr_windows[inner_val_mask] - rr_mean) / rr_std
        val_t = torch.FloatTensor(rr_val).unsqueeze(1)
        val_ds = TensorDataset(val_t, torch.LongTensor(labels[inner_val_mask]))
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False)

        # Model
        pos_weight = torch.FloatTensor([n_neg / n_pos]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = model_factory().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            # Early stopping on val balanced accuracy
            val_probs, val_labels = evaluate(model, val_loader, device)
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = float(balanced_accuracy_score(val_labels, val_preds))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        epochs_trained = epoch + 1
        if best_state is not None:
            model.load_state_dict(best_state)

        # Test evaluation
        test_probs, test_labels = evaluate(model, test_loader, device)

        # Default threshold
        preds_default = (test_probs >= 0.5).astype(int)
        bal_acc_default = float(balanced_accuracy_score(test_labels, preds_default))

        # Inner-CV threshold
        val_probs2, val_labels2 = evaluate(model, val_loader, device)
        inner_t, _ = find_optimal_threshold(val_probs2, val_labels2)
        preds_inner = (test_probs >= inner_t).astype(int)
        bal_acc_inner = float(balanced_accuracy_score(test_labels, preds_inner))

        # Oracle threshold
        oracle_t, bal_acc_oracle = find_optimal_threshold(test_probs, test_labels)

        # AUC
        try:
            auc = float(roc_auc_score(test_labels, test_probs))
        except ValueError:
            auc = float('nan')

        f1_val = float(f1_score(test_labels, preds_inner, zero_division=0))

        result = {
            "subject": str(test_subj),
            "bal_acc_default": round(bal_acc_default, 4),
            "bal_acc_inner_cv": round(bal_acc_inner, 4),
            "bal_acc_oracle": round(bal_acc_oracle, 4),
            "inner_threshold": round(inner_t, 4),
            "oracle_threshold": round(oracle_t, 4),
            "auc_roc": round(auc, 4),
            "f1": round(f1_val, 4),
            "epochs": epochs_trained,
            "n_test": len(y_test),
            "stress_ratio_test": round(float(y_test.mean()), 3),
        }
        per_subject.append(result)

        print(
            f"  Fold {fold_idx+1:2d}/{len(unique_subjects)} [{test_subj:>4s}]  "
            f"default={bal_acc_default:.3f}  inner={bal_acc_inner:.3f} (t={inner_t:.3f})  "
            f"oracle={bal_acc_oracle:.3f}  auc={auc:.3f}  ep={epochs_trained}"
        )

    # Aggregate
    agg = {
        "bal_acc_default": round(np.mean([r["bal_acc_default"] for r in per_subject]), 4),
        "bal_acc_inner_cv": round(np.mean([r["bal_acc_inner_cv"] for r in per_subject]), 4),
        "bal_acc_oracle": round(np.mean([r["bal_acc_oracle"] for r in per_subject]), 4),
        "auc_roc": round(np.nanmean([r["auc_roc"] for r in per_subject]), 4),
        "f1": round(np.mean([r["f1"] for r in per_subject]), 4),
    }

    return {
        "model": model_name,
        "per_subject": per_subject,
        "aggregate": agg,
    }


# =====================================================================
#  LOSOCV FOR LOGREG ON HRV FEATURES
# =====================================================================

def losocv_logreg_hrv(hrv_features, labels, subjects, name="LogReg-HRV"):
    """Standard LOSOCV with LogReg on HRV features from R-R windows."""
    unique_subjects = np.unique(subjects)
    per_subject = []

    for fold_idx, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        X_train = hrv_features[train_mask].copy()
        X_test = hrv_features[test_mask].copy()
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        scaler = RobustScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
        X_test = np.nan_to_num(scaler.transform(X_test), nan=0.0)

        model = LogisticRegression(
            max_iter=1000, class_weight='balanced', C=1.0, solver='lbfgs'
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        bal_acc = float(balanced_accuracy_score(y_test, preds))
        try:
            auc = float(roc_auc_score(y_test, probs))
        except ValueError:
            auc = float('nan')
        f1_val = float(f1_score(y_test, preds, zero_division=0))

        per_subject.append({
            "subject": str(test_subj),
            "bal_acc": round(bal_acc, 4),
            "auc_roc": round(auc, 4),
            "f1": round(f1_val, 4),
            "n_test": len(y_test),
        })

    agg = {
        "bal_acc": round(np.mean([r["bal_acc"] for r in per_subject]), 4),
        "auc_roc": round(np.nanmean([r["auc_roc"] for r in per_subject]), 4),
        "f1": round(np.mean([r["f1"] for r in per_subject]), 4),
    }

    print(f"\n  {name}: bal_acc={agg['bal_acc']:.4f}, AUC={agg['auc_roc']:.4f}, F1={agg['f1']:.4f}")

    return {"model": name, "per_subject": per_subject, "aggregate": agg}


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("#  R-R INTERVAL SEQUENCE MODEL (WESAD)")
    print("#  Core hypothesis: stress signal is in TIMING, not SHAPE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"#  Device: {gpu_name}")
    print(f"#  PyTorch: {torch.__version__}")
    print("=" * 70)

    # ── Load raw ECG from WESAD ──
    print("\n  Phase 1: Extracting R-R intervals from continuous ECG ...")
    sr = WESAD_CHEST_SR  # 700 Hz
    N_BEATS = 30          # ~25-35 seconds per window
    STRIDE_BEATS = 15     # 50% overlap

    subject_ids = sorted(discover_subjects(WESAD_RAW_DIR))
    print(f"  Found {len(subject_ids)} subjects")

    all_rr_windows = []
    all_hrv_features = []
    all_labels = []
    all_subjects = []
    subj_stats = []

    for sid in subject_ids:
        subj = load_subject(WESAD_RAW_DIR / sid)

        rr_ms, rr_times, labels_full = extract_rr_for_subject(subj, sr)

        windows, w_labels, hrv_feats = window_rr_intervals(
            rr_ms, rr_times, labels_full, sr,
            n_beats=N_BEATS, stride_beats=STRIDE_BEATS,
        )

        if len(windows) == 0:
            print(f"    {sid}: 0 windows (skipped)")
            continue

        all_rr_windows.append(windows)
        all_hrv_features.append(hrv_feats)
        all_labels.append(w_labels)
        all_subjects.extend([sid] * len(windows))

        n_stress = int(w_labels.sum())
        n_total = len(w_labels)
        mean_hr = 60000.0 / np.mean(rr_ms)
        print(f"    {sid}: {len(rr_ms)} R-R intervals -> {n_total} windows, "
              f"stress={n_stress}/{n_total} ({100*n_stress/n_total:.1f}%), "
              f"mean HR={mean_hr:.0f} bpm")

        subj_stats.append({
            "subject": sid,
            "n_rr_total": len(rr_ms),
            "n_windows": n_total,
            "n_stress": n_stress,
            "stress_pct": round(100 * n_stress / n_total, 1),
            "mean_hr_bpm": round(mean_hr, 1),
            "mean_rr_ms": round(float(np.mean(rr_ms)), 1),
        })

    rr_windows = np.vstack(all_rr_windows)
    hrv_features = np.vstack(all_hrv_features)
    labels = np.concatenate(all_labels)
    subjects = np.array(all_subjects)

    print(f"\n  Total: {len(labels)} windows, {len(np.unique(subjects))} subjects")
    print(f"  R-R window shape: {rr_windows.shape}")
    print(f"  HRV features shape: {hrv_features.shape}")
    print(f"  Stress ratio: {labels.mean():.3f} ({labels.sum()}/{len(labels)})")

    all_results = {
        "config": {
            "n_beats": N_BEATS,
            "stride_beats": STRIDE_BEATS,
            "sr": sr,
            "approx_window_sec": f"{N_BEATS * 0.85:.0f}-{N_BEATS * 1.2:.0f}s",
        },
        "subject_stats": subj_stats,
    }

    # ══════════════════════════════════════════════════════════════════
    #  EXPERIMENT A: 1D-CNN on R-R sequences
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Experiment A: RRCNN1D on R-R interval sequences")
    print("=" * 70)

    def rrcnn_factory():
        return RRCNN1D(n_beats=N_BEATS, dropout=0.3)

    # Print model info
    test_model = rrcnn_factory()
    total_p, train_p = test_model._param_count
    print(f"  Model: RRCNN1D, params={total_p:,}")
    del test_model

    cnn_results = losocv_rr_cnn(
        model_factory=rrcnn_factory,
        rr_windows=rr_windows,
        labels=labels,
        subjects=subjects,
        model_name="RRCNN1D",
        n_epochs=60,
        batch_size=256,
        lr=1e-3,
        patience=12,
        device=device,
    )
    all_results["rr_cnn"] = cnn_results

    print(f"\n  RRCNN1D Summary:")
    print(f"    default t=0.5:  bal_acc={cnn_results['aggregate']['bal_acc_default']:.4f}")
    print(f"    inner-CV:       bal_acc={cnn_results['aggregate']['bal_acc_inner_cv']:.4f}")
    print(f"    oracle:         bal_acc={cnn_results['aggregate']['bal_acc_oracle']:.4f}")
    print(f"    AUC-ROC:        {cnn_results['aggregate']['auc_roc']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  EXPERIMENT B: BiLSTM on R-R sequences
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Experiment B: RRBiLSTM on R-R interval sequences")
    print("=" * 70)

    def lstm_factory():
        return RRBiLSTM(n_beats=N_BEATS, hidden_size=32, num_layers=2, dropout=0.3)

    test_model = lstm_factory()
    total_p, _ = test_model._param_count
    print(f"  Model: RRBiLSTM, params={total_p:,}")
    del test_model

    lstm_results = losocv_rr_cnn(
        model_factory=lstm_factory,
        rr_windows=rr_windows,
        labels=labels,
        subjects=subjects,
        model_name="RRBiLSTM",
        n_epochs=60,
        batch_size=256,
        lr=1e-3,
        patience=12,
        device=device,
    )
    all_results["rr_lstm"] = lstm_results

    print(f"\n  RRBiLSTM Summary:")
    print(f"    default t=0.5:  bal_acc={lstm_results['aggregate']['bal_acc_default']:.4f}")
    print(f"    inner-CV:       bal_acc={lstm_results['aggregate']['bal_acc_inner_cv']:.4f}")
    print(f"    oracle:         bal_acc={lstm_results['aggregate']['bal_acc_oracle']:.4f}")
    print(f"    AUC-ROC:        {lstm_results['aggregate']['auc_roc']:.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  EXPERIMENT C: LogReg on HRV from 30-beat windows
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Experiment C: LogReg on HRV features (30-beat windows)")
    print("=" * 70)

    logreg_30beat = losocv_logreg_hrv(
        hrv_features, labels, subjects,
        name="LogReg-HRV-30beat"
    )
    all_results["logreg_30beat"] = logreg_30beat

    # ══════════════════════════════════════════════════════════════════
    #  FINAL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: Representation vs Model Complexity")
    print("=" * 70)

    baselines = {
        "LogReg on 7 HRV (5s window, Phase 2)": {"bal_acc": 0.763, "auc": 0.892},
        "HybridCNN on raw ECG (5s, Phase 3)":   {"bal_acc": 0.691, "auc": 0.876},
        "HybridCNN + threshold (Script 19)":     {"bal_acc": 0.707, "auc": 0.876},
    }

    print(f"\n  {'Model':<50} {'bal_acc':<12} {'AUC':<10}")
    print("-" * 75)

    for name, vals in baselines.items():
        print(f"  {name:<50} {vals['bal_acc']:<12} {vals['auc']:<10}")

    print("-" * 75)

    # New results
    rr_cnn_acc = cnn_results['aggregate']['bal_acc_inner_cv']
    rr_cnn_auc = cnn_results['aggregate']['auc_roc']
    rr_lstm_acc = lstm_results['aggregate']['bal_acc_inner_cv']
    rr_lstm_auc = lstm_results['aggregate']['auc_roc']
    lr_30_acc = logreg_30beat['aggregate']['bal_acc']
    lr_30_auc = logreg_30beat['aggregate']['auc_roc']

    new_results = {
        "RRCNN1D on R-R intervals (30-beat)": {"bal_acc": rr_cnn_acc, "auc": rr_cnn_auc},
        "RRBiLSTM on R-R intervals (30-beat)": {"bal_acc": rr_lstm_acc, "auc": rr_lstm_auc},
        "LogReg on HRV (30-beat windows)": {"bal_acc": lr_30_acc, "auc": lr_30_auc},
    }

    for name, vals in new_results.items():
        print(f"  {name:<50} {vals['bal_acc']:<12} {vals['auc']:<10}")

    # ── Determine verdict ──
    best_dl = max(rr_cnn_acc, rr_lstm_acc)
    best_dl_name = "RRCNN1D" if rr_cnn_acc >= rr_lstm_acc else "RRBiLSTM"
    logreg_5s = 0.763

    if best_dl > logreg_5s + 0.02:
        verdict = "TIMING_DL_WINS"
        msg = (f"R-R interval DL ({best_dl_name}: {best_dl:.3f}) beats "
               f"LogReg 5s ({logreg_5s:.3f}). Nonlinear temporal patterns exist!")
    elif best_dl > logreg_5s - 0.02:
        verdict = "TIMING_DL_COMPARABLE"
        msg = (f"R-R interval DL ({best_dl:.3f}) matches LogReg 5s ({logreg_5s:.3f}). "
               f"Representation alignment works, but linear features are sufficient.")
    elif lr_30_acc > logreg_5s + 0.02:
        verdict = "LONGER_WINDOW_HELPS"
        msg = (f"LogReg 30-beat ({lr_30_acc:.3f}) beats LogReg 5s ({logreg_5s:.3f}). "
               f"Window length was the bottleneck, not representation.")
    else:
        verdict = "HRV_FEATURES_SUFFICIENT"
        msg = (f"LogReg 5s ({logreg_5s:.3f}) remains best. "
               f"7 HRV features from 5s windows are already optimal.")

    all_results["_comparison"] = {
        "logreg_5s_baseline": logreg_5s,
        "best_dl_model": best_dl_name,
        "best_dl_acc": round(best_dl, 4),
        "logreg_30beat_acc": round(lr_30_acc, 4),
        "rr_cnn_acc": round(rr_cnn_acc, 4),
        "rr_lstm_acc": round(rr_lstm_acc, 4),
        "verdict": verdict,
        "message": msg,
    }

    # ── Scientific interpretation ──
    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")

    # Theory test results
    print("\n  Representation-Signal Alignment Test:")
    print(f"    Raw ECG CNN (shape domain):        {0.691:.3f} (Phase 3)")
    print(f"    R-R interval CNN (timing domain):  {rr_cnn_acc:.3f}")
    print(f"    R-R interval LSTM (timing domain): {rr_lstm_acc:.3f}")
    print(f"    LogReg on HRV (timing features):   {lr_30_acc:.3f} (30-beat)")
    print(f"    LogReg on HRV (timing features):   {logreg_5s:.3f} (5s, baseline)")

    if rr_cnn_acc > 0.691 + 0.03:
        print("\n  -> Switching to timing domain improved DL performance.")
        print("     This confirms: representation alignment matters more than model capacity.")
    elif lr_30_acc > logreg_5s + 0.02:
        print("\n  -> Longer windows improved LogReg but not DL.")
        print("     Window length was limiting HRV estimation stability.")
    else:
        print("\n  -> Neither longer windows nor timing-domain DL improved over baseline.")
        print("     5s HRV features are already near-optimal for this signal (d=1.55).")

    # ── Save ──
    out_path = VALIDATION_DIR / "rr_interval_results.json"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
