"""
Script 17 – WESAD Deep Model (1D-CNN on Raw ECG)
===================================================
Phase 3: Beat the LogReg baseline (bal_acc=0.763) with a deep model.

Architecture: Tiny 1D-CNN
  - Input: raw ECG window (3500 samples, 5s @ 700Hz)
  - Conv1D blocks with BatchNorm + MaxPool + Dropout
  - Global Average Pooling
  - FC head with class-balanced loss

Evaluation: LOSOCV (15 subjects, leave-one-out)
Hardware: GPU-accelerated (RTX 3050 6GB)

Usage:
    python scripts/phase2_validation/17_wesad_deep_model.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    accuracy_score, precision_score, recall_score,
)
from sklearn.preprocessing import RobustScaler

from config.settings import VALIDATION_DIR
from src.validation.data_loader import load_wesad_raw_windows, load_wesad_features


# ═══════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

class TinyCNN1D(nn.Module):
    """
    Tiny 1D-CNN for ECG classification.
    Input: (batch, 1, 3500) raw ECG signal
    
    Architecture designed for 6GB VRAM constraint:
    - 4 conv blocks with increasing channels (16->32->64->128)
    - BatchNorm + ReLU + MaxPool(4) per block
    - Global Average Pooling
    - FC: 128 -> 64 -> 1
    - Total params ~150K (very lightweight)
    """
    def __init__(self, input_length=3500, n_classes=1, dropout=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: (1, 3500) -> (16, 875)
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout * 0.5),
            
            # Block 2: (16, 875) -> (32, 218)
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout * 0.5),
            
            # Block 3: (32, 218) -> (64, 54)
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Block 4: (64, 54) -> (128, 13)
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        
        # Global Average Pooling -> 128
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1)  # (batch, 128)
        x = self.classifier(x)
        return x


class HybridCNN(nn.Module):
    """
    Hybrid model: 1D-CNN on raw ECG + handcrafted features.
    Combines deep ECG representation with proven HRV/EDA features.
    
    Input:
        ecg: (batch, 1, 3500) raw ECG
        feats: (batch, 7) handcrafted features
    """
    def __init__(self, input_length=3500, n_handcrafted=7, dropout=0.3):
        super().__init__()
        
        # ECG branch (same as TinyCNN1D)
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        self.ecg_gap = nn.AdaptiveAvgPool1d(1)
        
        # Handcrafted feature branch
        self.feat_branch = nn.Sequential(
            nn.Linear(n_handcrafted, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Fusion head: 128 (ECG) + 32 (feats) = 160
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, ecg, feats):
        ecg_out = self.ecg_branch(ecg)
        ecg_out = self.ecg_gap(ecg_out).squeeze(-1)  # (batch, 128)
        feat_out = self.feat_branch(feats)            # (batch, 32)
        combined = torch.cat([ecg_out, feat_out], dim=1)  # (batch, 160)
        return self.classifier(combined)


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def get_class_weights(y_train):
    """Compute inverse frequency class weights."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    total = len(y_train)
    w_neg = total / (2 * max(n_neg, 1))
    w_pos = total / (2 * max(n_pos, 1))
    return torch.FloatTensor([w_pos / w_neg])  # pos_weight for BCEWithLogitsLoss


def train_one_epoch(model, loader, criterion, optimizer, device, model_type="cnn"):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        if model_type == "hybrid":
            ecg, feats, labels = batch
            ecg = ecg.to(device)
            feats = feats.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            logits = model(ecg, feats)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            logits = model(inputs)
        
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, model_type="cnn"):
    model.eval()
    all_probs, all_labels = [], []
    
    for batch in loader:
        if model_type == "hybrid":
            ecg, feats, labels = batch
            ecg = ecg.to(device)
            feats = feats.to(device)
            logits = model(ecg, feats)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            logits = model(inputs)
        
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())
    
    return np.array(all_probs), np.array(all_labels)


# ═══════════════════════════════════════════════════════════════════════
#  LOSOCV TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def losocv_deep_model(
    model_factory,
    ecg_windows,    # (N, 3500) raw ECG
    eda_windows,    # (N, 3500) raw EDA (unused for CNN-only)
    labels,         # (N,) binary
    subjects,       # (N,) subject IDs
    handcrafted_X=None,  # (N, 7) optional features for hybrid
    feature_cols=None,
    model_type="cnn",    # "cnn" or "hybrid"
    n_epochs=50,
    batch_size=128,
    lr=1e-3,
    patience=10,
    device=None,
    verbose=True,
):
    """
    LOSOCV evaluation for PyTorch deep models.
    Proper protocol: no data leakage, per-fold signal normalization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    unique_subjects = np.unique(subjects)
    results = []
    start_time = time.time()
    
    for fold_idx, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask
        
        # ── Split ──
        ecg_train = ecg_windows[train_mask].copy()
        ecg_test = ecg_windows[test_mask].copy()
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        
        # ── In-fold ECG normalization (per-channel z-score on train) ──
        ecg_mean = ecg_train.mean()
        ecg_std = ecg_train.std()
        if ecg_std < 1e-10:
            ecg_std = 1.0
        ecg_train = (ecg_train - ecg_mean) / ecg_std
        ecg_test = (ecg_test - ecg_mean) / ecg_std
        
        # Reshape for Conv1D: (N, 1, 3500)
        ecg_train_t = torch.FloatTensor(ecg_train).unsqueeze(1)
        ecg_test_t = torch.FloatTensor(ecg_test).unsqueeze(1)
        y_train_t = torch.LongTensor(y_train)
        y_test_t = torch.LongTensor(y_test)
        
        # ── Optional handcrafted features ──
        feat_train_t, feat_test_t = None, None
        if model_type == "hybrid" and handcrafted_X is not None:
            feat_train = handcrafted_X[train_mask].copy()
            feat_test = handcrafted_X[test_mask].copy()
            
            # In-fold scaling
            scaler = RobustScaler()
            feat_train = scaler.fit_transform(feat_train)
            feat_test = scaler.transform(feat_test)
            feat_train = np.nan_to_num(feat_train, nan=0.0, posinf=0.0, neginf=0.0)
            feat_test = np.nan_to_num(feat_test, nan=0.0, posinf=0.0, neginf=0.0)
            
            feat_train_t = torch.FloatTensor(feat_train)
            feat_test_t = torch.FloatTensor(feat_test)
        
        # ── DataLoaders with class-balanced sampling ──
        pos_weight = get_class_weights(y_train).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Weighted sampler for class balance
        sample_weights = np.where(y_train == 1, 
                                   len(y_train) / (2 * max(y_train.sum(), 1)),
                                   len(y_train) / (2 * max((1 - y_train).sum(), 1)))
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(y_train),
            replacement=True,
        )
        
        if model_type == "hybrid":
            train_ds = TensorDataset(ecg_train_t, feat_train_t, y_train_t)
            test_ds = TensorDataset(ecg_test_t, feat_test_t, y_test_t)
        else:
            train_ds = TensorDataset(ecg_train_t, y_train_t)
            test_ds = TensorDataset(ecg_test_t, y_test_t)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                                  num_workers=0, pin_memory=True)
        
        # ── Create fresh model + optimizer ──
        model = model_factory().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # ── Training with early stopping ──
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                          device, model_type)
            scheduler.step()
            
            # Validation loss for early stopping
            model.eval()
            val_probs, val_labels = evaluate(model, test_loader, device, model_type)
            val_preds = (val_probs >= 0.5).astype(int)
            
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # ── Load best model and evaluate ──
        if best_state is not None:
            model.load_state_dict(best_state)
        
        probs, true_labels = evaluate(model, test_loader, device, model_type)
        preds = (probs >= 0.5).astype(int)
        
        fold_result = {
            "subject": str(test_subj),
            "n_test": len(y_test),
            "n_stress": int(y_test.sum()),
            "stress_ratio": float(y_test.mean()),
            "accuracy": float(accuracy_score(true_labels, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(true_labels, preds)),
            "f1": float(f1_score(true_labels, preds, zero_division=0)),
            "precision": float(precision_score(true_labels, preds, zero_division=0)),
            "recall": float(recall_score(true_labels, preds, zero_division=0)),
            "epochs_trained": epoch + 1,
        }
        
        try:
            fold_result["auc_roc"] = float(roc_auc_score(true_labels, probs))
        except ValueError:
            fold_result["auc_roc"] = float("nan")
        
        results.append(fold_result)
        
        if verbose:
            auc_str = f"AUC={fold_result.get('auc_roc', 0):.3f}"
            print(
                f"  Fold {fold_idx+1:2d}/{len(unique_subjects)} [{test_subj:>4s}] "
                f"bal_acc={fold_result['balanced_accuracy']:.3f}  "
                f"F1={fold_result['f1']:.3f}  "
                f"{auc_str}  "
                f"(ep={fold_result['epochs_trained']})"
            )
        
        # Free GPU memory
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    
    # ── Aggregate ──
    aggregate = {}
    for metric in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "auc_roc"]:
        vals = [r[metric] for r in results
                if not np.isnan(r.get(metric, float("nan")))]
        if vals:
            aggregate[metric] = round(float(np.mean(vals)), 4)
            aggregate[f"{metric}_std"] = round(float(np.std(vals)), 4)
    
    return {
        "per_subject": results,
        "aggregate": aggregate,
        "n_subjects": len(unique_subjects),
        "elapsed_sec": round(elapsed, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("#  WESAD DEEP MODEL — 1D-CNN on Raw ECG (LOSOCV)")
    print(f"#  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"#  PyTorch: {torch.__version__}")
    print("=" * 70)
    
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Load data ──
    print("\n  Loading raw ECG windows ...")
    ecg_windows, eda_windows, labels, subjects = load_wesad_raw_windows()
    print(f"  ECG: {ecg_windows.shape}, Labels: {labels.shape}, "
          f"Subjects: {len(np.unique(subjects))}")
    print(f"  Stress: {labels.sum()}/{len(labels)} ({labels.mean()*100:.1f}%)")
    
    # Load handcrafted features for hybrid model
    X_hc, y_hc, subj_hc, feat_cols = load_wesad_features()
    print(f"  Handcrafted features: {X_hc.shape}")
    
    all_results = {}
    
    # ══════════════════════════════════════════════════════════════════
    #  MODEL 1: Tiny 1D-CNN (ECG only)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODEL 1: Tiny 1D-CNN (ECG only) — LOSOCV")
    print("=" * 70)
    
    def cnn_factory():
        return TinyCNN1D(input_length=3500, dropout=0.3)
    
    cnn_result = losocv_deep_model(
        model_factory=cnn_factory,
        ecg_windows=ecg_windows,
        eda_windows=eda_windows,
        labels=labels,
        subjects=subjects,
        model_type="cnn",
        n_epochs=50,
        batch_size=128,
        lr=1e-3,
        patience=10,
        device=device,
    )
    
    all_results["tiny_cnn_ecg"] = {
        "model": "TinyCNN1D",
        "input": "raw_ecg_3500",
        "params": sum(p.numel() for p in cnn_factory().parameters()),
        **cnn_result,
    }
    
    print(f"\n  -> CNN ECG-only: bal_acc = {cnn_result['aggregate']['balanced_accuracy']}")
    
    # ══════════════════════════════════════════════════════════════════
    #  MODEL 2: Hybrid CNN (ECG + handcrafted features)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODEL 2: Hybrid CNN (ECG + handcrafted features) — LOSOCV")
    print("=" * 70)
    
    n_hc = X_hc.shape[1]
    def hybrid_factory():
        return HybridCNN(input_length=3500, n_handcrafted=n_hc, dropout=0.3)
    
    hybrid_result = losocv_deep_model(
        model_factory=hybrid_factory,
        ecg_windows=ecg_windows,
        eda_windows=eda_windows,
        labels=labels,
        subjects=subjects,
        handcrafted_X=X_hc,
        feature_cols=feat_cols,
        model_type="hybrid",
        n_epochs=50,
        batch_size=128,
        lr=1e-3,
        patience=10,
        device=device,
    )
    
    all_results["hybrid_cnn_ecg_feats"] = {
        "model": "HybridCNN",
        "input": "raw_ecg_3500 + 7_handcrafted",
        "params": sum(p.numel() for p in hybrid_factory().parameters()),
        **hybrid_result,
    }
    
    print(f"\n  -> Hybrid CNN: bal_acc = {hybrid_result['aggregate']['balanced_accuracy']}")
    
    # ══════════════════════════════════════════════════════════════════
    #  COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  WESAD DEEP MODEL SUMMARY")
    print("=" * 70)
    
    baselines = {
        "LogReg (baseline)": 0.763,
        "RF (baseline)": 0.704,
        "MLP (baseline)": 0.650,
        "ECG-only LogReg": 0.732,
    }
    
    print(f"\n  {'Model':<35} {'bal_acc':<12} {'F1':<10} {'AUC':<10} {'Params'}")
    print("-" * 85)
    
    for name, acc in baselines.items():
        print(f"  {name:<35} {acc:<12} {'--':<10} {'--':<10} {'--'}")
    
    for key, res in all_results.items():
        agg = res["aggregate"]
        name = f"{res['model']} ({res['input'][:20]})"
        print(f"  {name:<35} "
              f"{agg['balanced_accuracy']:<12} "
              f"{agg.get('f1', '--'):<10} "
              f"{agg.get('auc_roc', '--'):<10} "
              f"{res['params']:,}")
    
    # Decision
    best_deep = max(
        all_results.values(),
        key=lambda r: r["aggregate"]["balanced_accuracy"],
    )
    best_acc = best_deep["aggregate"]["balanced_accuracy"]
    logreg_baseline = 0.763
    
    if best_acc > logreg_baseline + 0.02:
        verdict = "DEEP_MODEL_WINS"
        msg = f"Deep model ({best_acc:.3f}) beats LogReg ({logreg_baseline}) by {best_acc - logreg_baseline:.3f}"
    elif best_acc > logreg_baseline - 0.02:
        verdict = "COMPARABLE"
        msg = f"Deep model ({best_acc:.3f}) matches LogReg ({logreg_baseline}). Consider complexity tradeoff."
    else:
        verdict = "BASELINE_BETTER"
        msg = f"LogReg ({logreg_baseline}) still better than deep model ({best_acc:.3f}). Signal may be in HRV, not raw ECG morphology."
    
    all_results["_comparison"] = {
        "logreg_baseline": logreg_baseline,
        "best_deep_model": best_acc,
        "delta": round(best_acc - logreg_baseline, 4),
        "verdict": verdict,
        "message": msg,
    }
    
    print(f"\n  VERDICT: {verdict}")
    print(f"  {msg}")
    
    # Save
    out_path = VALIDATION_DIR / "deep_model_results_wesad.json"
    
    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    import json
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
