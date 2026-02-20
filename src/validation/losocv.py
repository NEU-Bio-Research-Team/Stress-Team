"""
Leave-One-Subject-Out Cross-Validation (LOSOCV)
==================================================
Core evaluation framework with:
  - In-fold scaling (fit on train, transform test)
  - Class-weight handling
  - Per-subject + aggregate metrics
  - Effect size computation
  - Learning curves
  - Calibration analysis

This is the ONLY correct evaluation for physiological signal classification.
Subject-dependent evaluation inflates results and will be rejected by reviewers.
"""

import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score,
)
from typing import Callable, Dict, Optional, List, Any


def losocv_evaluate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    scaler_factory: Optional[Callable] = None,
    class_weight: Optional[Dict[int, float]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Leave-One-Subject-Out Cross-Validation with proper in-fold scaling.

    Args:
        model_factory: callable returning a FRESH model instance each fold
        X: (N, D) feature matrix
        y: (N,) binary labels
        subjects: (N,) subject IDs
        scaler_factory: callable returning a fresh scaler; fitted on train only
        class_weight: {0: w0, 1: w1} for sample weighting
        verbose: print per-fold progress

    Returns:
        dict with per_subject results and aggregate metrics
    """
    unique_subjects = np.unique(subjects)
    results = []
    start_time = time.time()

    for i, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
        y_train, y_test = y[train_mask], y[test_mask]

        # ── In-fold scaling ──
        if scaler_factory:
            scaler = scaler_factory()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # ── Handle NaN/Inf after scaling ──
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Create fresh model ──
        model = model_factory()

        # ── Fit with optional sample weights ──
        sample_weight = None
        if class_weight:
            sample_weight = np.array([class_weight.get(int(yi), 1.0) for yi in y_train])

        try:
            if sample_weight is not None:
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Fold {test_subj}: fit failed — {e}")
            continue

        # ── Predict ──
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        # ── Metrics ──
        fold_result = {
            "subject": str(test_subj),
            "n_test": len(y_test),
            "n_stress": int(y_test.sum()),
            "stress_ratio": float(y_test.mean()),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }

        if y_prob is not None:
            try:
                fold_result["auc_roc"] = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                fold_result["auc_roc"] = float("nan")
            fold_result["brier"] = float(brier_score_loss(y_test, y_prob))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fold_result["tn"] = int(cm[0, 0])
        fold_result["fp"] = int(cm[0, 1])
        fold_result["fn"] = int(cm[1, 0])
        fold_result["tp"] = int(cm[1, 1])

        results.append(fold_result)

        if verbose:
            auc_str = f"AUC={fold_result.get('auc_roc', 'N/A'):.3f}" if 'auc_roc' in fold_result else ""
            print(
                f"  Fold {i+1:2d}/{len(unique_subjects)} [{test_subj:>4s}] "
                f"acc={fold_result['accuracy']:.3f}  "
                f"bal_acc={fold_result['balanced_accuracy']:.3f}  "
                f"F1={fold_result['f1']:.3f}  "
                f"{auc_str}"
            )

    elapsed = time.time() - start_time

    # ── Aggregate ──
    aggregate = {}
    metric_keys = ["accuracy", "balanced_accuracy", "f1", "precision", "recall",
                   "auc_roc", "brier"]
    for metric in metric_keys:
        vals = [r[metric] for r in results
                if metric in r and not np.isnan(r.get(metric, float("nan")))]
        if vals:
            aggregate[f"{metric}_mean"] = float(np.mean(vals))
            aggregate[f"{metric}_std"] = float(np.std(vals))
            aggregate[f"{metric}_min"] = float(np.min(vals))
            aggregate[f"{metric}_max"] = float(np.max(vals))
            aggregate[f"{metric}_median"] = float(np.median(vals))

    return {
        "per_subject": results,
        "aggregate": aggregate,
        "n_subjects": len(unique_subjects),
        "n_subjects_evaluated": len(results),
        "n_samples": len(y),
        "n_features": X.shape[1],
        "stress_ratio": float(y.mean()),
        "elapsed_sec": round(elapsed, 2),
    }


def learning_curve_losocv(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    scaler_factory: Optional[Callable] = None,
    train_sizes: Optional[List[int]] = None,
    n_repeats: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Learning curve: how does performance scale with number of training subjects?

    For each train_size k, randomly sample k subjects for training,
    evaluate on remaining subjects, repeat n_repeats times.

    Args:
        train_sizes: list of number of training subjects to test
                     e.g. [3, 5, 8, 10, 14] for WESAD

    Returns:
        dict with train_sizes and corresponding metrics
    """
    unique_subjects = np.unique(subjects)
    n_subj = len(unique_subjects)

    if train_sizes is None:
        train_sizes = sorted(set([
            max(2, n_subj // 5),
            max(3, n_subj // 3),
            n_subj // 2,
            int(n_subj * 0.7),
            n_subj - 1,
        ]))

    rng = np.random.RandomState(42)
    curve_results = []

    for k in train_sizes:
        if k >= n_subj:
            continue

        repeat_metrics = []
        for rep in range(n_repeats):
            # Random split: k training subjects, rest for test
            perm = rng.permutation(unique_subjects)
            train_subjs = set(perm[:k])

            train_mask = np.array([s in train_subjs for s in subjects])
            test_mask = ~train_mask

            X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
            y_train, y_test = y[train_mask], y[test_mask]

            if scaler_factory:
                scaler = scaler_factory()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            model = model_factory()
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                repeat_metrics.append({"balanced_accuracy": acc, "f1": f1})
            except Exception:
                continue

        if repeat_metrics:
            curve_results.append({
                "n_train_subjects": k,
                "balanced_accuracy_mean": np.mean([m["balanced_accuracy"] for m in repeat_metrics]),
                "balanced_accuracy_std": np.std([m["balanced_accuracy"] for m in repeat_metrics]),
                "f1_mean": np.mean([m["f1"] for m in repeat_metrics]),
                "f1_std": np.std([m["f1"] for m in repeat_metrics]),
                "n_repeats": len(repeat_metrics),
            })

        if verbose:
            r = curve_results[-1] if curve_results else None
            if r:
                print(
                    f"  k={k:2d} subjects: "
                    f"bal_acc={r['balanced_accuracy_mean']:.3f}±{r['balanced_accuracy_std']:.3f}  "
                    f"F1={r['f1_mean']:.3f}±{r['f1_std']:.3f}"
                )

    return {"curve": curve_results, "train_sizes": train_sizes}
