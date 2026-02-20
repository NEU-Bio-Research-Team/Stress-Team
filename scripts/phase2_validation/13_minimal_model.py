"""
Script 13 – Minimal Publishable Model (MPM)
==============================================
Test the smallest valid model to determine publishability floor.

Strategy (per advisor):
  - ECG branch only, RR features only (4 features)
  - If this minimal model reaches target → paper is publishable
  - If it fails → larger model will also fail

Also tests:
  - ECG-only (4 features) vs Full (7 features) comparison
  - Per-feature ablation (drop one feature at a time)

Usage:
    python scripts/phase2_validation/13_minimal_model.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from pathlib import Path
from config.settings import PROJECT_ROOT
from src.validation.data_loader import load_wesad_features, describe_dataset
from src.validation.baselines import run_all_baselines
from src.validation.losocv import losocv_evaluate
from src.validation.scaling import get_scaler_factory
from sklearn.linear_model import LogisticRegression


def ablation_study(X, y, subjects, feature_cols, dataset="wesad"):
    """
    Drop one feature at a time and measure impact.
    Large drop → feature is critical.
    No change → feature is redundant.
    """
    print("\n" + "=" * 60)
    print("  Feature Ablation Study (drop-one)")
    print("=" * 60)

    scaler = get_scaler_factory(dataset, feature_cols)

    def model_factory():
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)

    # Full model baseline
    full_result = losocv_evaluate(model_factory, X, y, subjects,
                                   scaler_factory=scaler, verbose=False)
    full_bal = full_result["aggregate"].get("balanced_accuracy_mean", 0)
    full_f1 = full_result["aggregate"].get("f1_mean", 0)

    print(f"\n  Full model ({len(feature_cols)} features): "
          f"bal_acc={full_bal:.3f}  F1={full_f1:.3f}")
    print(f"\n  {'Dropped Feature':<20s} {'Bal Acc':>10s} {'Δ Bal Acc':>10s} {'F1':>10s} {'Impact':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    ablation_results = []
    for i, col in enumerate(feature_cols):
        # Drop feature i
        X_ablated = np.delete(X, i, axis=1)
        remaining_cols = [c for j, c in enumerate(feature_cols) if j != i]

        # Need to rebuild scaler for reduced features
        reduced_scaler = get_scaler_factory(dataset, remaining_cols)

        result = losocv_evaluate(
            model_factory, X_ablated, y, subjects,
            scaler_factory=reduced_scaler, verbose=False,
        )
        abl_bal = result["aggregate"].get("balanced_accuracy_mean", 0)
        abl_f1 = result["aggregate"].get("f1_mean", 0)
        delta = abl_bal - full_bal

        impact = "CRITICAL" if delta < -0.03 else "MODERATE" if delta < -0.01 else "NEGLIGIBLE"
        ablation_results.append({
            "dropped": col, "bal_acc": abl_bal, "delta": delta, "f1": abl_f1, "impact": impact,
        })

        print(f"  {col:<20s} {abl_bal:>10.3f} {delta:>+10.3f} {abl_f1:>10.3f} {impact:>10s}")

    return ablation_results


def main():
    output_dir = PROJECT_ROOT / "reports" / "validation"

    print("\n" + "#" * 70)
    print("#  MINIMAL PUBLISHABLE MODEL — WESAD")
    print("#" * 70)

    # ── Test 1: ECG-only (4 features) → Minimal model ──
    print("\n" + "=" * 60)
    print("  Test 1: ECG-Only Model (4 HRV features)")
    print("=" * 60)
    ecg_features = ["hr_mean", "hr_std", "rmssd", "sdnn"]
    X_ecg, y_ecg, subj_ecg, cols_ecg = load_wesad_features(feature_subset=ecg_features)
    describe_dataset(X_ecg, y_ecg, subj_ecg, name="WESAD ECG-Only")

    ecg_summary = run_all_baselines(
        X_ecg, y_ecg, subj_ecg,
        dataset="wesad",
        feature_cols=cols_ecg,
        output_dir=output_dir,
        verbose=True,
    )

    # ── Test 2: Full model (7 features) ──
    print("\n" + "=" * 60)
    print("  Test 2: Full Model (7 features)")
    print("=" * 60)
    X_full, y_full, subj_full, cols_full = load_wesad_features()
    describe_dataset(X_full, y_full, subj_full, name="WESAD Full")

    full_summary = run_all_baselines(
        X_full, y_full, subj_full,
        dataset="wesad",
        feature_cols=cols_full,
        output_dir=output_dir,
        verbose=True,
    )

    # ── Test 3: Feature ablation ──
    ablation = ablation_study(X_full, y_full, subj_full, cols_full, "wesad")

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("  MINIMAL vs FULL MODEL COMPARISON")
    print("=" * 70)

    ecg_best = max(
        ecg_summary["baselines"].values(),
        key=lambda b: b.get("balanced_accuracy_mean", 0),
    )
    full_best = max(
        full_summary["baselines"].values(),
        key=lambda b: b.get("balanced_accuracy_mean", 0),
    )

    ecg_bal = ecg_best.get("balanced_accuracy_mean", 0)
    full_bal = full_best.get("balanced_accuracy_mean", 0)

    print(f"  ECG-only (4 feat): {ecg_bal:.3f} ({ecg_summary['decision']})")
    print(f"  Full     (7 feat): {full_bal:.3f} ({full_summary['decision']})")
    print(f"  Improvement:       {full_bal - ecg_bal:+.3f}")

    if ecg_bal >= 0.70:
        print("\n  ✅ PUBLISHABLE: ECG-only model already reaches target.")
        print("     Paper can use minimal architecture. EDA features are bonus.")
    elif full_bal >= 0.70:
        print("\n  🔧 EDA NEEDED: ECG-only is insufficient but full model works.")
        print("     Include EDA features in final architecture.")
    else:
        print("\n  ⚠️ INSUFFICIENT: Neither model reaches target.")
        print("     Consider: more features, different stress proxy, or deep model on raw signals.")

    # ── Save ablation ──
    import json
    ablation_path = output_dir / "ablation_results_wesad.json"
    with open(ablation_path, "w") as f:
        json.dump({
            "ecg_only_decision": ecg_summary["decision"],
            "full_decision": full_summary["decision"],
            "ecg_best_bal_acc": ecg_bal,
            "full_best_bal_acc": full_bal,
            "ablation": ablation,
        }, f, indent=2)
    print(f"\n  Ablation saved to {ablation_path}")


if __name__ == "__main__":
    main()
