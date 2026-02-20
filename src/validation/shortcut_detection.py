"""
Shortcut Detection Battery (Step 2)
======================================
Three tests to prove model learns stress, not subject identity.

Test 1 — Subject Classifier Probe:
    Train classifier: features → subject_id.
    If accuracy high → features encode subject identity.

Test 2 — Permutation Test:
    Shuffle labels → train model → expect accuracy ≈ chance.
    If not → data leakage.

Test 3 — Feature Importance Stability:
    Compare feature importance rankings across LOSOCV folds.
    If top features change every fold → unreliable signal.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.validation.scaling import get_scaler_factory


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Subject Classifier Probe
# ═══════════════════════════════════════════════════════════════════

def subject_classifier_probe(
    X: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    n_folds: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train classifier to predict subject_id from features.

    If accuracy >> chance (1/n_subjects):
        → features encode subject identity
        → model might learn subject, not stress

    Interpretation:
        chance = 1/n_subjects
        If probe_acc < 2 × chance → low subject encoding (GOOD)
        If probe_acc > 5 × chance → heavy subject encoding (BAD, need domain adaptation)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  Test 1 — Subject Classifier Probe")
        print("=" * 60)

    le = LabelEncoder()
    y_subj = le.fit_transform(subjects)
    n_subj = len(le.classes_)
    chance_level = 1.0 / n_subj

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified K-fold (not LOSOCV — we want to test within-distribution)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y_subj)):
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled[train_idx], y_subj[train_idx])
        y_pred = clf.predict(X_scaled[test_idx])
        acc = accuracy_score(y_subj[test_idx], y_pred)
        fold_accs.append(acc)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}: subject_acc = {acc:.3f}")

    mean_acc = np.mean(fold_accs)
    encoding_ratio = mean_acc / chance_level

    result = {
        "test_name": "subject_classifier_probe",
        "chance_level": round(chance_level, 4),
        "n_subjects": n_subj,
        "probe_accuracy_mean": round(float(mean_acc), 4),
        "probe_accuracy_std": round(float(np.std(fold_accs)), 4),
        "encoding_ratio": round(float(encoding_ratio), 2),
        "fold_accuracies": [round(a, 4) for a in fold_accs],
    }

    # Decision
    if encoding_ratio > 5:
        result["verdict"] = "HIGH_SUBJECT_ENCODING"
        result["interpretation"] = (
            f"Probe accuracy {mean_acc:.1%} is {encoding_ratio:.1f}× chance ({chance_level:.1%}). "
            "Features STRONGLY encode subject identity. "
            "Model may learn subject, not stress. "
            "MUST use adversarial GRL or subject normalization."
        )
    elif encoding_ratio > 2:
        result["verdict"] = "MODERATE_SUBJECT_ENCODING"
        result["interpretation"] = (
            f"Probe accuracy {mean_acc:.1%} is {encoding_ratio:.1f}× chance. "
            "Features partially encode subject identity. "
            "Adversarial training recommended but not critical."
        )
    else:
        result["verdict"] = "LOW_SUBJECT_ENCODING"
        result["interpretation"] = (
            f"Probe accuracy {mean_acc:.1%} is only {encoding_ratio:.1f}× chance. "
            "Features do NOT strongly encode subject identity. Good."
        )

    if verbose:
        print(f"\n  Chance level:    {chance_level:.3f}")
        print(f"  Probe accuracy:  {mean_acc:.3f} ({encoding_ratio:.1f}× chance)")
        print(f"  Verdict:         {result['verdict']}")
        print(f"  {result['interpretation']}")

    return result


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Permutation Test
# ═══════════════════════════════════════════════════════════════════

def permutation_test(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    n_permutations: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Permutation test: shuffle labels → expect accuracy ≈ chance.

    1. Compute true LOSOCV performance (or use pre-computed).
    2. For each permutation: shuffle y → compute LOSOCV → record score.
    3. p-value = fraction of permuted scores ≥ true score.

    If p-value > 0.05 → model performance is NOT significant.
    If p-value < 0.01 → strong evidence that model finds real signal.
    """
    from src.validation.losocv import losocv_evaluate

    if verbose:
        print("\n" + "=" * 60)
        print("  Test 2 — Permutation Test")
        print(f"  ({n_permutations} permutations)")
        print("=" * 60)

    scaler = get_scaler_factory(dataset, feature_cols)

    # True performance
    if verbose:
        print("\n  Computing true LOSOCV performance...")
    true_result = losocv_evaluate(model_factory, X, y, subjects,
                                   scaler_factory=scaler, verbose=False)
    true_score = true_result["aggregate"].get("balanced_accuracy_mean", 0)
    if verbose:
        print(f"  True balanced_accuracy = {true_score:.4f}")

    # Permuted performances
    rng = np.random.RandomState(42)
    perm_scores = []

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_result = losocv_evaluate(model_factory, X, y_perm, subjects,
                                       scaler_factory=scaler, verbose=False)
        perm_score = perm_result["aggregate"].get("balanced_accuracy_mean", 0)
        perm_scores.append(perm_score)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1:3d}/{n_permutations}: "
                  f"score = {perm_score:.4f}  "
                  f"(running p = {np.mean(np.array(perm_scores) >= true_score):.4f})")

    perm_scores = np.array(perm_scores)
    p_value = float(np.mean(perm_scores >= true_score))

    result = {
        "test_name": "permutation_test",
        "true_balanced_accuracy": round(true_score, 4),
        "permuted_mean": round(float(perm_scores.mean()), 4),
        "permuted_std": round(float(perm_scores.std()), 4),
        "permuted_max": round(float(perm_scores.max()), 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_permutations,
    }

    if p_value < 0.01:
        result["verdict"] = "SIGNIFICANT"
        result["interpretation"] = (
            f"p = {p_value:.4f} < 0.01. Model performance is highly significant. "
            "The learned signal is NOT due to chance or label noise."
        )
    elif p_value < 0.05:
        result["verdict"] = "MARGINALLY_SIGNIFICANT"
        result["interpretation"] = (
            f"p = {p_value:.4f} < 0.05. Model performance is marginally significant. "
            "Signal exists but is weak."
        )
    else:
        result["verdict"] = "NOT_SIGNIFICANT"
        result["interpretation"] = (
            f"p = {p_value:.4f} ≥ 0.05. Model performance is NOT significant. "
            "WARNING: model may be learning noise or label artifacts."
        )

    if verbose:
        print(f"\n  True score:      {true_score:.4f}")
        print(f"  Permuted mean:   {perm_scores.mean():.4f} ± {perm_scores.std():.4f}")
        print(f"  p-value:         {p_value:.4f}")
        print(f"  Verdict:         {result['verdict']}")

    return result


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Feature Importance Stability
# ═══════════════════════════════════════════════════════════════════

def feature_importance_stability(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_cols: List[str],
    dataset: str = "wesad",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Test if feature importance rankings are stable across LOSOCV folds.

    Trains Random Forest per fold, collects feature importances,
    and computes rank correlation (Kendall's tau) between folds.

    Good: top features consistently ranked high across folds.
    Bad: top feature changes every fold → unreliable signal.
    """
    from scipy.stats import kendalltau

    if verbose:
        print("\n" + "=" * 60)
        print("  Test 3 — Feature Importance Stability")
        print("=" * 60)

    scaler_factory = get_scaler_factory(dataset, feature_cols)
    unique_subjects = np.unique(subjects)
    fold_importances = []

    for test_subj in unique_subjects:
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        X_train = X[train_mask].copy()
        y_train = y[train_mask]

        scaler = scaler_factory()
        X_train = scaler.fit_transform(X_train)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        fold_importances.append(rf.feature_importances_)

    imp_matrix = np.array(fold_importances)

    # Pairwise Kendall's tau on importance rankings
    n_folds = len(fold_importances)
    taus = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            tau, _ = kendalltau(imp_matrix[i], imp_matrix[j])
            taus.append(tau)

    mean_tau = float(np.mean(taus))

    # Top feature agreement
    ranks = np.argsort(-imp_matrix, axis=1)
    top_features = [feature_cols[r[0]] for r in ranks]
    top3_features = [[feature_cols[r[k]] for k in range(min(3, len(feature_cols)))] for r in ranks]

    from collections import Counter
    top_counter = Counter(top_features)
    most_common_top = top_counter.most_common(1)[0]

    result = {
        "test_name": "feature_importance_stability",
        "kendall_tau_mean": round(mean_tau, 4),
        "kendall_tau_std": round(float(np.std(taus)), 4),
        "top_feature_agreement": round(most_common_top[1] / n_folds, 4),
        "most_common_top_feature": most_common_top[0],
        "top_feature_distribution": dict(top_counter),
        "mean_importance": {col: round(float(imp_matrix[:, i].mean()), 4)
                           for i, col in enumerate(feature_cols)},
        "std_importance": {col: round(float(imp_matrix[:, i].std()), 4)
                          for i, col in enumerate(feature_cols)},
    }

    if mean_tau > 0.7:
        result["verdict"] = "STABLE"
        result["interpretation"] = (
            f"Mean Kendall's τ = {mean_tau:.3f} > 0.7. "
            "Feature importance rankings are stable across folds. "
            "The model is learning consistent patterns."
        )
    elif mean_tau > 0.4:
        result["verdict"] = "MODERATE"
        result["interpretation"] = (
            f"Mean Kendall's τ = {mean_tau:.3f}. "
            "Some instability in feature rankings. "
            "Top features are generally consistent but lower-ranked features vary."
        )
    else:
        result["verdict"] = "UNSTABLE"
        result["interpretation"] = (
            f"Mean Kendall's τ = {mean_tau:.3f} < 0.4. "
            "Feature importance is UNSTABLE across folds. "
            "Model may be learning noise rather than consistent patterns."
        )

    if verbose:
        print(f"  Mean Kendall's τ: {mean_tau:.4f}")
        print(f"  Top feature agreement: {most_common_top[1]}/{n_folds} folds → {most_common_top[0]}")
        print(f"  Verdict: {result['verdict']}")

        print("\n  Feature importance (mean ± std):")
        sorted_feats = sorted(
            enumerate(feature_cols),
            key=lambda x: imp_matrix[:, x[0]].mean(),
            reverse=True,
        )
        for idx, col in sorted_feats:
            m = imp_matrix[:, idx].mean()
            s = imp_matrix[:, idx].std()
            print(f"    {col:>15s}: {m:.4f} ± {s:.4f}")

    return result


def run_all_shortcut_tests(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    dataset: str = "wesad",
    feature_cols: Optional[List[str]] = None,
    n_permutations: int = 100,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all 3 shortcut detection tests and save results."""
    start = time.time()

    results = {}

    # Test 1: Subject classifier probe
    results["subject_probe"] = subject_classifier_probe(
        X, subjects, dataset, feature_cols, verbose=verbose,
    )

    # Test 2: Permutation test (using logistic regression for speed)
    def model_factory():
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    results["permutation_test"] = permutation_test(
        model_factory, X, y, subjects, dataset, feature_cols,
        n_permutations=n_permutations, verbose=verbose,
    )

    # Test 3: Feature importance stability
    if feature_cols:
        results["feature_stability"] = feature_importance_stability(
            X, y, subjects, feature_cols, dataset, verbose=verbose,
        )

    elapsed = time.time() - start
    results["elapsed_sec"] = round(elapsed, 2)

    # Overall assessment
    verdicts = [r.get("verdict", "N/A") for r in results.values() if isinstance(r, dict)]
    if "HIGH_SUBJECT_ENCODING" in verdicts or "NOT_SIGNIFICANT" in verdicts:
        results["overall_verdict"] = "SHORTCUT_DETECTED"
        results["overall_interpretation"] = (
            "WARNING: Shortcut learning indicators found. "
            "Model may not be learning genuine stress signal. "
            "Adversarial training (GRL) is REQUIRED."
        )
    elif "MODERATE_SUBJECT_ENCODING" in verdicts:
        results["overall_verdict"] = "CAUTION"
        results["overall_interpretation"] = (
            "Some indicators of shortcut learning. "
            "Adversarial training recommended."
        )
    else:
        results["overall_verdict"] = "CLEAN"
        results["overall_interpretation"] = (
            "No strong indicators of shortcut learning. "
            "Model appears to learn genuine stress signal."
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"shortcut_results_{dataset}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    if verbose:
        print("\n" + "=" * 60)
        print(f"  OVERALL: {results['overall_verdict']}")
        print(f"  {results['overall_interpretation']}")
        print(f"  (elapsed: {elapsed:.1f}s)")
        print("=" * 60)

    return results
