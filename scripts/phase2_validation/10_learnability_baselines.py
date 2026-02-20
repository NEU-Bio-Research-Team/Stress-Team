"""
Script 10 – Stress Learnability Test (CRITICAL GATE)
======================================================
Before training any deep model, prove that signal actually exists.

Runs 3 baseline models with LOSOCV:
  A — Logistic Regression (linear signal detection)
  B — Random Forest (nonlinear + feature importance)
  C — MLP Classifier (shallow neural network)

Decision rule:
  - All 3 ≈ random → signal does not exist → deep model meaningless
  - LogReg >70% balanced_accuracy → strong signal → deep model can reach >85%

Also computes:
  - Cohen's d effect sizes per feature
  - Learning curves (performance vs n_training_subjects)
  - Feature importance stability

Usage:
    python scripts/phase2_validation/10_learnability_baselines.py
    python scripts/phase2_validation/10_learnability_baselines.py --dataset dreamer
    python scripts/phase2_validation/10_learnability_baselines.py --dataset wesad --ecg-only
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
from pathlib import Path
from config.settings import PROJECT_ROOT
from src.validation.data_loader import (
    load_wesad_features, load_dreamer_features, describe_dataset,
)
from src.validation.baselines import run_all_baselines


def main():
    parser = argparse.ArgumentParser(description="Learnability Baseline Test (LOSOCV)")
    parser.add_argument("--dataset", choices=["wesad", "dreamer", "both"],
                        default="both", help="Dataset to test")
    parser.add_argument("--ecg-only", action="store_true",
                        help="WESAD: test with ECG/HRV features only (4 features)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "reports" / "validation"

    datasets_to_run = []
    if args.dataset in ("wesad", "both"):
        datasets_to_run.append("wesad")
    if args.dataset in ("dreamer", "both"):
        datasets_to_run.append("dreamer")

    for ds in datasets_to_run:
        print("\n" + "#" * 70)
        print(f"#  LEARNABILITY TEST — {ds.upper()}")
        print("#" * 70)

        if ds == "wesad":
            if args.ecg_only:
                # Minimal Publishable Model: ECG/HRV features only
                ecg_features = ["hr_mean", "hr_std", "rmssd", "sdnn"]
                X, y, subjects, feature_cols = load_wesad_features(feature_subset=ecg_features)
                print("  [MODE] ECG-only (Minimal Publishable Model)")
            else:
                X, y, subjects, feature_cols = load_wesad_features()
        elif ds == "dreamer":
            X, y, subjects, feature_cols = load_dreamer_features()

        describe_dataset(X, y, subjects, name=f"{ds.upper()} Features")

        summary = run_all_baselines(
            X, y, subjects,
            dataset=ds,
            feature_cols=feature_cols,
            output_dir=output_dir,
            verbose=True,
        )

        # Print final comparison table
        print("\n" + "=" * 70)
        print(f"  FINAL COMPARISON — {ds.upper()}")
        print("=" * 70)
        print(f"  {'Model':<20s} {'Bal Acc':>10s} {'F1':>10s} {'AUC':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for name, b in summary["baselines"].items():
            bal = b.get("balanced_accuracy_mean", 0)
            f1 = b.get("f1_mean", 0)
            auc = b.get("auc_roc_mean", 0)
            print(f"  {name:<20s} {bal:>10.3f} {f1:>10.3f} {auc:>10.3f}")
        print(f"\n  Decision: {summary['decision']}")
        print(f"  {summary['recommendation']}")
        print()


if __name__ == "__main__":
    main()
