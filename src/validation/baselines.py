"""
Baseline Models for Learnability Test (Step 1)
================================================
Three baseline models to prove signal existence before deep learning:

  Model A — Logistic Regression (linear signal detection)
  Model B — Random Forest (nonlinear, feature importance)
  Model C — MLP Classifier (shallow neural network)
  Model D — Tiny 1D-CNN on raw signals (optional, requires PyTorch)

Decision rule:
  - All 3 ≈ random → signal does not exist → deep model pointless
  - LogReg >70%  → strong signal → deep model can reach >85%
  - RF > LogReg  → nonlinear signal exists

All models evaluated via LOSOCV with in-fold scaling.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.validation.losocv import losocv_evaluate, learning_curve_losocv
from src.validation.scaling import get_scaler_factory
from src.validation.effect_size import compute_effect_sizes


def run_logistic_baseline(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Model A: Logistic Regression with balanced class weights."""
    if verbose:
        print("\n" + "=" * 60)
        print("  Model A — Logistic Regression (LOSOCV)")
        print("=" * 60)

    scaler = get_scaler_factory(dataset, feature_cols)

    def model_factory():
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )

    result = losocv_evaluate(model_factory, X, y, subjects,
                              scaler_factory=scaler, verbose=verbose)
    result["model_name"] = "LogisticRegression"
    result["model_type"] = "linear"
    return result


def run_rf_baseline(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Model B: Random Forest with balanced class weights + feature importance."""
    if verbose:
        print("\n" + "=" * 60)
        print("  Model B — Random Forest (LOSOCV)")
        print("=" * 60)

    scaler = get_scaler_factory(dataset, feature_cols)

    # Collect feature importances across folds
    fold_importances = []

    class RFWithImportanceCapture(RandomForestClassifier):
        """Wrapper to capture feature importances per fold."""
        def fit(self, X_train, y_train, **kwargs):
            super().fit(X_train, y_train, **kwargs)
            fold_importances.append(self.feature_importances_.copy())
            return self

    def model_factory():
        return RFWithImportanceCapture(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    result = losocv_evaluate(model_factory, X, y, subjects,
                              scaler_factory=scaler, verbose=verbose)
    result["model_name"] = "RandomForest"
    result["model_type"] = "nonlinear_ensemble"

    # Feature importance stability
    if fold_importances and feature_cols:
        imp_matrix = np.array(fold_importances)  # (n_folds, n_features)
        result["feature_importance"] = {
            "mean": {col: float(imp_matrix[:, i].mean())
                     for i, col in enumerate(feature_cols)},
            "std": {col: float(imp_matrix[:, i].std())
                    for i, col in enumerate(feature_cols)},
            "stability_cv": {col: float(imp_matrix[:, i].std() / (imp_matrix[:, i].mean() + 1e-10))
                             for i, col in enumerate(feature_cols)},
        }
        # Rank stability: does the top feature stay the same across folds?
        ranks = np.argsort(-imp_matrix, axis=1)  # (n_folds, n_features)
        top_feature_per_fold = [feature_cols[r[0]] for r in ranks]
        result["feature_importance"]["top_feature_per_fold"] = top_feature_per_fold
        result["feature_importance"]["top_feature_agreement"] = (
            max(top_feature_per_fold.count(f) for f in set(top_feature_per_fold))
            / len(top_feature_per_fold)
        )

    return result


def run_mlp_baseline(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Model C: MLP Classifier (tiny neural network on features)."""
    if verbose:
        print("\n" + "=" * 60)
        print("  Model C — MLP Classifier (LOSOCV)")
        print("=" * 60)

    scaler = get_scaler_factory(dataset, feature_cols)

    def model_factory():
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )

    result = losocv_evaluate(model_factory, X, y, subjects,
                              scaler_factory=scaler, verbose=verbose)
    result["model_name"] = "MLPClassifier"
    result["model_type"] = "neural_network"
    return result


def run_all_baselines(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all 3 baseline models and compile results.

    Returns summary dict with comparison and decision recommendation.
    """
    start = time.time()

    # ── Effect sizes first (no model needed) ──
    if verbose:
        print("\n" + "=" * 60)
        print("  Pre-model: Feature Effect Sizes (Cohen's d)")
        print("=" * 60)
    effect_sizes = compute_effect_sizes(X, y, feature_cols)
    if verbose and feature_cols:
        for col, d in sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True):
            strength = "LARGE" if abs(d) >= 0.8 else "MEDIUM" if abs(d) >= 0.5 else "SMALL" if abs(d) >= 0.2 else "NEGLIGIBLE"
            print(f"    {col:>15s}: d = {d:+.3f}  ({strength})")

    # ── Run baselines ──
    results = {}
    results["logistic"] = run_logistic_baseline(X, y, subjects, dataset, feature_cols, verbose)
    results["rf"] = run_rf_baseline(X, y, subjects, dataset, feature_cols, verbose)
    results["mlp"] = run_mlp_baseline(X, y, subjects, dataset, feature_cols, verbose)

    # ── Learning curve (using best baseline) ──
    best_model = max(results, key=lambda k: results[k]["aggregate"].get("balanced_accuracy_mean", 0))
    if verbose:
        print("\n" + "=" * 60)
        print(f"  Learning Curve (using {results[best_model]['model_name']})")
        print("=" * 60)

    best_factory_map = {
        "logistic": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "rf": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
    }
    scaler = get_scaler_factory(dataset, feature_cols)
    lc = learning_curve_losocv(
        best_factory_map[best_model], X, y, subjects,
        scaler_factory=scaler, verbose=verbose,
    )

    # ── Compile summary ──
    elapsed = time.time() - start
    summary = {
        "dataset": dataset,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "n_subjects": len(np.unique(subjects)),
        "stress_ratio": float(y.mean()),
        "effect_sizes": effect_sizes,
        "baselines": {
            name: {
                "model": r["model_name"],
                "type": r["model_type"],
                **r["aggregate"],
            }
            for name, r in results.items()
        },
        "learning_curve": lc,
        "best_baseline": best_model,
        "elapsed_sec": round(elapsed, 2),
    }

    # ── Decision ──
    best_bal_acc = results[best_model]["aggregate"].get("balanced_accuracy_mean", 0)
    if best_bal_acc >= 0.70:
        summary["decision"] = "STRONG_SIGNAL"
        summary["recommendation"] = (
            f"Best baseline ({results[best_model]['model_name']}) achieves "
            f"balanced_accuracy={best_bal_acc:.3f} (>0.70). "
            "Signal is strong. Deep model can likely reach >85%. Proceed."
        )
    elif best_bal_acc >= 0.55:
        summary["decision"] = "WEAK_SIGNAL"
        summary["recommendation"] = (
            f"Best baseline achieves balanced_accuracy={best_bal_acc:.3f}. "
            "Signal exists but is weak. Deep model may help, but investigate "
            "feature engineering first."
        )
    else:
        summary["decision"] = "NO_SIGNAL"
        summary["recommendation"] = (
            f"Best baseline achieves balanced_accuracy={best_bal_acc:.3f} (≈random). "
            "Signal does not exist in current features. Deep model will be meaningless. "
            "Re-examine feature extraction or stress proxy definition."
        )

    # ── Save ──
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Full details (per-subject)
        detail_path = output_dir / f"baseline_results_{dataset}.json"
        full_output = {
            "summary": summary,
            "logistic_detail": results["logistic"],
            "rf_detail": results["rf"],
            "mlp_detail": results["mlp"],
        }
        with open(detail_path, "w") as f:
            json.dump(full_output, f, indent=2, default=str)
        if verbose:
            print(f"\n  Results saved to {detail_path}")

    # ── Print decision ──
    if verbose:
        print("\n" + "=" * 60)
        print(f"  DECISION: {summary['decision']}")
        print(f"  {summary['recommendation']}")
        print("=" * 60)

    return summary
