"""
Script 18 – DREAMER Post-Recovery Validation
===============================================
Run ONLY after script 16 completes and shows RECOVERED or PARTIAL_RECOVERY.

Validates that the recovered DREAMER signal is genuine:
  1. Re-run adversarial GRL with PyTorch on z-normed + best target
  2. Feature ablation (drop-one) on best configuration
  3. Learning curve (performance vs # training subjects)

If adversarial is ROBUST + learning curve saturates → GENUINE SIGNAL.
If adversarial drops or curve doesn't improve → ARTIFACT of normalization.

Usage:
    python scripts/phase2_validation/18_dreamer_post_recovery_validation.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config.settings import VALIDATION_DIR
from src.validation.adversarial import adversarial_subject_removal
from src.validation.losocv import losocv_evaluate, learning_curve_losocv
from src.validation.scaling import get_scaler_factory


def load_dreamer_with_options(
    normalize_within_subject=False,
    target="stress",
    arousal_threshold=3,
    valence_threshold=3,
):
    """Load DREAMER features with options (same as script 16)."""
    from config.settings import PROCESSED_DIR
    out_dir = PROCESSED_DIR / "dreamer"
    npz_files = sorted(out_dir.glob("S*_preprocessed.npz"))

    all_X, all_y, all_subjects = [], [], []

    for npz_file in npz_files:
        subj = npz_file.stem.replace("_preprocessed", "")
        data = np.load(npz_file)
        X_subj = data["de_features"].astype(np.float64)

        if normalize_within_subject:
            mu = X_subj.mean(axis=0, keepdims=True)
            sigma = X_subj.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_subj = (X_subj - mu) / sigma

        if target == "stress":
            y_subj = data["stress_labels"].astype(np.int32)
        elif target == "arousal":
            y_subj = (data["arousal"] >= arousal_threshold).astype(np.int32)
        elif target == "valence":
            y_subj = (data["valence"] <= valence_threshold).astype(np.int32)
        else:
            raise ValueError(f"Unknown target: {target}")

        n = len(y_subj)
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subjects.extend([subj] * n)

    feature_cols = [f"f{i}" for i in range(70)]
    return (
        np.vstack(all_X),
        np.concatenate(all_y),
        np.array(all_subjects),
        feature_cols,
    )


def get_best_config_from_recovery():
    """Read script 16 results and determine best configuration."""
    recovery_path = VALIDATION_DIR / "dreamer_recovery_results.json"
    if not recovery_path.exists():
        print("  [ERROR] dreamer_recovery_results.json not found!")
        print("  Run script 16 first.")
        return None

    with open(recovery_path) as f:
        results = json.load(f)

    summary = results.get("_summary", {})
    decision = summary.get("decision", "UNKNOWN")
    best_key = summary.get("best_experiment", "")
    best_acc = summary.get("best_bal_acc", 0)

    print(f"  Recovery decision: {decision}")
    print(f"  Best experiment: {best_key} (bal_acc={best_acc})")

    if decision == "NO_RECOVERY":
        print("  [WARN] DREAMER did not recover. Post-recovery validation may not be meaningful.")
        print("         Running anyway for completeness...")

    # Parse best config from key
    best_result = results.get(best_key, {})
    config = {
        "normalize": "zscore" in best_result.get("normalization", ""),
        "target": best_result.get("target", "stress").replace("_binary", ""),
        "best_acc": best_acc,
        "decision": decision,
    }
    return config


def run_feature_ablation(X, y, subjects, feature_cols, base_config):
    """Drop-one feature ablation to identify critical features."""
    print("\n  Running feature ablation (drop-one)...")

    # Get baseline with all features
    model_factory = lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42)
    scaler = get_scaler_factory("dreamer", feature_cols)

    base_result = losocv_evaluate(
        model_factory=model_factory,
        X=X, y=y, subjects=subjects,
        scaler_factory=scaler,
        verbose=False,
    )
    base_acc = base_result["aggregate"]["balanced_accuracy_mean"]
    print(f"  Baseline (all 70 features): bal_acc = {base_acc:.4f}")

    # Test dropping each band (5 bands × 14 channels = 70 features)
    # Group by band for efficiency: delta(0-13), theta(14-27), alpha(28-41),
    # beta(42-55), gamma(56-69)
    bands = {
        "delta": list(range(0, 14)),
        "theta": list(range(14, 28)),
        "alpha": list(range(28, 42)),
        "beta": list(range(42, 56)),
        "gamma": list(range(56, 70)),
    }

    ablation_results = {}
    for band_name, indices in bands.items():
        keep_idx = [i for i in range(70) if i not in indices]
        X_drop = X[:, keep_idx]
        drop_cols = [f"f{i}" for i in keep_idx]

        r = losocv_evaluate(
            model_factory=model_factory,
            X=X_drop, y=y, subjects=subjects,
            scaler_factory=get_scaler_factory("dreamer", drop_cols),
            verbose=False,
        )
        drop_acc = r["aggregate"]["balanced_accuracy_mean"]
        delta = drop_acc - base_acc
        impact = "CRITICAL" if delta < -0.03 else "MODERATE" if delta < -0.01 else "NEGLIGIBLE" if abs(delta) < 0.005 else "HELPS_TO_DROP"

        ablation_results[band_name] = {
            "dropped_features": len(indices),
            "remaining_features": len(keep_idx),
            "bal_acc": round(drop_acc, 4),
            "delta": round(delta, 4),
            "impact": impact,
        }
        print(f"    Drop {band_name:>6s} ({len(indices)} features): "
              f"bal_acc={drop_acc:.4f}  delta={delta:+.4f}  [{impact}]")

    return {"baseline_acc": base_acc, "band_ablation": ablation_results}


def main():
    print("=" * 70)
    print("#  DREAMER POST-RECOVERY VALIDATION")
    print("#  Adversarial GRL + Ablation + Learning Curve")
    print("=" * 70)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Step 0: Read recovery results ──
    config = get_best_config_from_recovery()
    if config is None:
        return

    all_results["recovery_config"] = config

    # ── Load best configuration ──
    print(f"\n  Loading DREAMER (z-norm={config['normalize']}, target={config['target']})...")
    X, y, subjects, fcols = load_dreamer_with_options(
        normalize_within_subject=config["normalize"],
        target=config["target"],
    )
    print(f"  Samples: {len(y):,} | Positive: {y.sum():,} ({y.mean()*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    #  TEST 1: Adversarial GRL (PyTorch)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 1: Adversarial Subject Removal (PyTorch GRL)")
    print("=" * 70)

    adv_result = adversarial_subject_removal(
        X, y, subjects,
        grl_lambda=1.0,
        epochs=100,
        verbose=True,
        output_dir=VALIDATION_DIR,
        dataset_name="dreamer_recovered",
    )
    all_results["adversarial_grl"] = adv_result

    # ══════════════════════════════════════════════════════════════════
    #  TEST 2: Feature Ablation (by frequency band)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 2: Feature Ablation (drop-one band)")
    print("=" * 70)

    ablation = run_feature_ablation(X, y, subjects, fcols, config)
    all_results["feature_ablation"] = ablation

    # ══════════════════════════════════════════════════════════════════
    #  TEST 3: Learning Curve
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 3: Learning Curve (performance vs # training subjects)")
    print("=" * 70)

    model_factory = lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42)
    scaler = get_scaler_factory("dreamer", fcols)

    lc_result = learning_curve_losocv(
        model_factory=model_factory,
        X=X, y=y, subjects=subjects,
        scaler_factory=scaler,
        train_sizes=[3, 5, 10, 15, 20],
        n_repeats=5,
        verbose=True,
    )
    all_results["learning_curve"] = lc_result

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DREAMER POST-RECOVERY VALIDATION SUMMARY")
    print("=" * 70)

    # Adversarial verdict
    adv_verdict = adv_result.get("verdict", "UNKNOWN")
    adv_delta = adv_result.get("delta", 0)
    print(f"\n  Adversarial GRL: {adv_verdict} (delta={adv_delta:+.4f})")

    # Ablation - which bands matter?
    print(f"\n  Feature Ablation:")
    for band, res in ablation["band_ablation"].items():
        print(f"    {band:>6s}: delta={res['delta']:+.4f} [{res['impact']}]")

    # Learning curve - does it saturate?
    if "curve" in lc_result:
        print(f"\n  Learning Curve:")
        for point in lc_result["curve"]:
            k = point.get("n_train_subjects", "?")
            acc = point.get("balanced_accuracy_mean", 0)
            std = point.get("balanced_accuracy_std", 0)
            print(f"    k={k:>3}: bal_acc={acc:.4f} +/- {std:.4f}")

    # Overall verdict
    genuine_checks = 0
    total_checks = 3

    if adv_verdict == "ROBUST":
        genuine_checks += 1
    elif adv_verdict == "IMPROVED":
        genuine_checks += 1

    # Check if any band is CRITICAL
    has_critical_band = any(
        r["impact"] == "CRITICAL"
        for r in ablation["band_ablation"].values()
    )
    if has_critical_band:
        genuine_checks += 1  # Signal is localized, not noise

    # Check learning curve trend
    if "curve" in lc_result and len(lc_result["curve"]) >= 2:
        first_acc = lc_result["curve"][0].get("balanced_accuracy_mean", 0)
        last_acc = lc_result["curve"][-1].get("balanced_accuracy_mean", 0)
        if last_acc > first_acc + 0.01:
            genuine_checks += 1  # More data → better performance

    if genuine_checks >= 2:
        overall = "GENUINE_SIGNAL"
        msg = (f"Recovered signal passes {genuine_checks}/{total_checks} checks. "
               "DREAMER is viable for deep model.")
    elif genuine_checks == 1:
        overall = "WEAK_SIGNAL"
        msg = (f"Only {genuine_checks}/{total_checks} checks pass. "
               "DREAMER signal is marginal. Use as secondary dataset.")
    else:
        overall = "NO_GENUINE_SIGNAL"
        msg = (f"0/{total_checks} checks pass. "
               "Recovery was superficial. Accept DREAMER as negative control.")

    all_results["_summary"] = {
        "overall_verdict": overall,
        "genuine_checks_passed": genuine_checks,
        "total_checks": total_checks,
        "message": msg,
        "adversarial_verdict": adv_verdict,
        "has_critical_band": has_critical_band,
    }

    print(f"\n  OVERALL: {overall} ({genuine_checks}/{total_checks} checks)")
    print(f"  {msg}")

    # Save
    out_path = VALIDATION_DIR / "dreamer_post_recovery_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
