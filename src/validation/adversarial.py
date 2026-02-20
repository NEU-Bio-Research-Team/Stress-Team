"""
Adversarial Subject Removal (Step 2, Test 3)
===============================================
Gradient Reversal Layer (GRL) for subject-invariant stress detection.

Architecture:
    Feature Extractor → [shared representation]
        ├── Stress Head       (minimize stress classification loss)
        └── Subject Head + GRL (maximize subject confusion)

If performance holds after GRL:
    → model truly learns stress, not subject identity.

Requires PyTorch. If not installed, falls back to a sklearn-based
domain adaptation approximation (CORAL-style).

Reference:
    Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation"
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import balanced_accuracy_score, f1_score

# Check PyTorch availability
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
# PyTorch GRL Implementation
# ═══════════════════════════════════════════════════════════════════

if _HAS_TORCH:
    class GradientReversalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output, None

    class GradientReversalLayer(nn.Module):
        def __init__(self, lambda_=1.0):
            super().__init__()
            self.lambda_ = lambda_

        def forward(self, x):
            return GradientReversalFunction.apply(x, self.lambda_)

    class AdversarialStressModel(nn.Module):
        """
        Two-head model:
          - Stress head: binary classification (stress/non-stress)
          - Subject head: multi-class classification (subject identity)
            with gradient reversal → forces feature extractor to be subject-invariant
        """
        def __init__(self, input_dim, n_subjects, hidden_dim=64, grl_lambda=1.0):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
            )
            self.stress_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            self.subject_head = nn.Sequential(
                GradientReversalLayer(grl_lambda),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_subjects),
            )

        def forward(self, x):
            features = self.feature_extractor(x)
            stress_logits = self.stress_head(features)
            subject_logits = self.subject_head(features)
            return stress_logits, subject_logits


def _train_adversarial_pytorch(
    X_train, y_train, subj_train,
    X_test, y_test,
    n_subjects, input_dim,
    grl_lambda=1.0, epochs=100, lr=1e-3, batch_size=256,
    device="cpu",
):
    """Train adversarial model with PyTorch and return test metrics."""
    le = LabelEncoder()
    subj_encoded = le.fit_transform(subj_train)

    # Tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    s_tr = torch.LongTensor(subj_encoded).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_tr, y_tr, s_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = AdversarialStressModel(input_dim, n_subjects, grl_lambda=grl_lambda).to(device)
    stress_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])
    ).to(device)
    subject_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y, batch_s in loader:
            optimizer.zero_grad()
            stress_logits, subject_logits = model(batch_X)
            loss_stress = stress_criterion(stress_logits.squeeze(), batch_y)
            loss_subject = subject_criterion(subject_logits, batch_s)
            loss = loss_stress + 0.5 * loss_subject
            loss.backward()
            optimizer.step()

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        stress_logits, _ = model(X_te)
        y_prob = torch.sigmoid(stress_logits.squeeze()).cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)

    return y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════
# Sklearn Fallback (without GRL, uses domain-weighted logistic)
# ═══════════════════════════════════════════════════════════════════

def _train_adversarial_sklearn_fallback(X_train, y_train, subj_train, X_test, y_test):
    """
    Fallback: Subject-balanced training (not true GRL, but directionally correct).
    Balances samples so each subject contributes equally → reduces subject bias.
    """
    from sklearn.linear_model import LogisticRegression

    unique_s = np.unique(subj_train)
    min_per_subj = min(np.sum(subj_train == s) for s in unique_s)

    # Subsample to balance subjects
    balanced_idx = []
    rng = np.random.RandomState(42)
    for s in unique_s:
        s_idx = np.where(subj_train == s)[0]
        chosen = rng.choice(s_idx, size=min_per_subj, replace=False)
        balanced_idx.extend(chosen)

    X_balanced = X_train[balanced_idx]
    y_balanced = y_train[balanced_idx]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X_balanced, y_balanced)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════
# Main LOSOCV with Adversarial Training
# ═══════════════════════════════════════════════════════════════════

def adversarial_subject_removal(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    grl_lambda: float = 1.0,
    epochs: int = 100,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
    dataset_name: str = "unknown",
) -> Dict[str, Any]:
    """
    LOSOCV evaluation with adversarial (GRL) subject removal.

    Compares:
        - Standard model (no GRL) → may encode subject
        - Adversarial model (with GRL) → forced subject-invariant

    If adversarial performance ≈ standard:
        → model truly learns stress (GOOD)
    If adversarial performance << standard:
        → standard model was partially learning subject identity (BAD)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  Adversarial Subject Removal (GRL)")
        if not _HAS_TORCH:
            print("  [WARN] PyTorch not installed. Using sklearn fallback.")
        print("=" * 60)

    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    scaler_base = RobustScaler()

    results_standard = []
    results_adversarial = []

    for i, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
        y_train, y_test = y[train_mask], y[test_mask]
        subj_train = subjects[train_mask]

        # Scale
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        input_dim = X_train.shape[1]

        # ── Standard model (no adversarial) ──
        from sklearn.linear_model import LogisticRegression
        std_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        std_clf.fit(X_train, y_train)
        std_pred = std_clf.predict(X_test)
        std_bal = balanced_accuracy_score(y_test, std_pred)
        std_f1 = f1_score(y_test, std_pred, zero_division=0)

        # ── Adversarial model ──
        if _HAS_TORCH:
            adv_pred, _ = _train_adversarial_pytorch(
                X_train, y_train, subj_train, X_test, y_test,
                n_subjects, input_dim,
                grl_lambda=grl_lambda, epochs=epochs,
            )
        else:
            adv_pred, _ = _train_adversarial_sklearn_fallback(
                X_train, y_train, subj_train, X_test, y_test,
            )

        adv_bal = balanced_accuracy_score(y_test, adv_pred)
        adv_f1 = f1_score(y_test, adv_pred, zero_division=0)

        results_standard.append({"subject": str(test_subj), "bal_acc": std_bal, "f1": std_f1})
        results_adversarial.append({"subject": str(test_subj), "bal_acc": adv_bal, "f1": adv_f1})

        if verbose:
            print(
                f"  Fold {i+1:2d}/{len(unique_subjects)} [{test_subj:>4s}] "
                f"std={std_bal:.3f}  adv(GRL)={adv_bal:.3f}  "
                f"delta={adv_bal - std_bal:+.3f}"
            )

    # Aggregate
    std_mean = np.mean([r["bal_acc"] for r in results_standard])
    adv_mean = np.mean([r["bal_acc"] for r in results_adversarial])
    delta = adv_mean - std_mean

    result = {
        "test_name": "adversarial_subject_removal",
        "backend": "pytorch" if _HAS_TORCH else "sklearn_fallback",
        "grl_lambda": grl_lambda,
        "standard_bal_acc_mean": round(float(std_mean), 4),
        "adversarial_bal_acc_mean": round(float(adv_mean), 4),
        "delta": round(float(delta), 4),
        "per_subject_standard": results_standard,
        "per_subject_adversarial": results_adversarial,
    }

    if abs(delta) < 0.02:
        result["verdict"] = "ROBUST"
        result["interpretation"] = (
            f"Adversarial model performance ({adv_mean:.3f}) ~= standard ({std_mean:.3f}), "
            f"delta={delta:+.3f}. Model genuinely learns stress, not subject identity."
        )
    elif delta < -0.05:
        result["verdict"] = "SUBJECT_DEPENDENT"
        result["interpretation"] = (
            f"Adversarial performance dropped by {abs(delta):.3f}. "
            "Standard model partially relies on subject identity. "
            "Consider stronger domain adaptation."
        )
    else:
        result["verdict"] = "IMPROVED"
        result["interpretation"] = (
            f"Adversarial model IMPROVED by {delta:+.3f}. "
            "GRL helped - subject confounds were hurting generalization."
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"adversarial_results_{dataset_name}.json"
        with open(output_dir / fname, "w") as f:
            json.dump(result, f, indent=2, default=str)

    if verbose:
        print(f"\n  Standard mean:     {std_mean:.4f}")
        print(f"  Adversarial mean:  {adv_mean:.4f}")
        print(f"  Delta:             {delta:+.4f}")
        print(f"  Verdict:           {result['verdict']}")
        # Use ASCII-safe output for Windows cp1252 compatibility
        interp = result['interpretation']
        try:
            print(f"  {interp}")
        except UnicodeEncodeError:
            print(f"  {interp.encode('ascii', 'replace').decode('ascii')}")

    return result
