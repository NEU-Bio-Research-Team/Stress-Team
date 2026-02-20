"""
Script 14 – DREAMER ICA Resolution
=====================================
4 subjects flagged for EEG artifacts: S10 (FC6), S17 (AF4), S21 (AF4), S23 (F4).

Per advisor: Do NOT run full ICA on all subjects.
Instead, compare performance WITH vs WITHOUT flagged subjects.

Decision rule:
  - If performance difference < 1% → skip ICA entirely (save 2 weeks compute)
  - If difference > 3% → ICA needed for those subjects

Usage:
    python scripts/phase2_validation/14_dreamer_ica_check.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import numpy as np
from pathlib import Path
from config.settings import PROJECT_ROOT
from src.validation.data_loader import load_dreamer_features, describe_dataset
from src.validation.losocv import losocv_evaluate
from src.validation.scaling import get_scaler_factory
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


FLAGGED_SUBJECTS = ["S10", "S17", "S21", "S23"]


def main():
    output_dir = PROJECT_ROOT / "reports" / "validation"

    print("\n" + "#" * 70)
    print("#  DREAMER ICA RESOLUTION CHECK")
    print(f"#  Flagged subjects: {', '.join(FLAGGED_SUBJECTS)}")
    print("#" * 70)

    # ── Load ALL 23 subjects ──
    X_all, y_all, subj_all, cols = load_dreamer_features()
    describe_dataset(X_all, y_all, subj_all, name="DREAMER All 23 Subjects")

    # ── Load WITHOUT flagged subjects ──
    X_clean, y_clean, subj_clean, _ = load_dreamer_features(exclude_subjects=FLAGGED_SUBJECTS)
    describe_dataset(X_clean, y_clean, subj_clean,
                     name=f"DREAMER Without {len(FLAGGED_SUBJECTS)} Flagged Subjects")

    scaler = get_scaler_factory("dreamer")

    results = {}

    for model_name, factory in [
        ("LogisticRegression", lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42)),
        ("RandomForest", lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]:
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # All subjects
        print("\n  → All 23 subjects:")
        res_all = losocv_evaluate(factory, X_all, y_all, subj_all,
                                   scaler_factory=scaler, verbose=True)

        # Without flagged
        print(f"\n  → Without flagged ({len(FLAGGED_SUBJECTS)} excluded):")
        res_clean = losocv_evaluate(factory, X_clean, y_clean, subj_clean,
                                     scaler_factory=scaler, verbose=True)

        all_bal = res_all["aggregate"].get("balanced_accuracy_mean", 0)
        clean_bal = res_clean["aggregate"].get("balanced_accuracy_mean", 0)
        delta = clean_bal - all_bal

        # Performance on flagged subjects specifically
        flagged_results = [r for r in res_all["per_subject"]
                          if r["subject"] in FLAGGED_SUBJECTS]
        flagged_bal = np.mean([r["balanced_accuracy"] for r in flagged_results]) if flagged_results else 0
        non_flagged_results = [r for r in res_all["per_subject"]
                              if r["subject"] not in FLAGGED_SUBJECTS]
        non_flagged_bal = np.mean([r["balanced_accuracy"] for r in non_flagged_results]) if non_flagged_results else 0

        results[model_name] = {
            "all_subjects_bal_acc": round(all_bal, 4),
            "clean_subjects_bal_acc": round(clean_bal, 4),
            "delta": round(delta, 4),
            "flagged_subjects_bal_acc": round(flagged_bal, 4),
            "non_flagged_subjects_bal_acc": round(non_flagged_bal, 4),
        }

        print(f"\n  All 23:       {all_bal:.4f}")
        print(f"  Without 4:    {clean_bal:.4f}")
        print(f"  Delta:        {delta:+.4f}")
        print(f"  Flagged avg:  {flagged_bal:.4f}")
        print(f"  Non-flagged:  {non_flagged_bal:.4f}")

    # ── Decision ──
    avg_delta = np.mean([abs(r["delta"]) for r in results.values()])

    print("\n" + "=" * 70)
    print("  ICA DECISION")
    print("=" * 70)

    if avg_delta < 0.01:
        decision = "SKIP_ICA"
        reason = (
            f"Average performance difference = {avg_delta:.4f} (<1%). "
            "ICA is NOT needed. Flagged subjects do not meaningfully affect results. "
            "This saves approximately 2 weeks of compute."
        )
    elif avg_delta < 0.03:
        decision = "ICA_OPTIONAL"
        reason = (
            f"Average performance difference = {avg_delta:.4f} (1-3%). "
            "ICA would marginally improve results. "
            "Consider applying ICA only to flagged subjects if time permits."
        )
    else:
        decision = "ICA_REQUIRED"
        reason = (
            f"Average performance difference = {avg_delta:.4f} (>3%). "
            "Flagged subjects significantly affect model performance. "
            "Apply ICA to S10, S17, S21, S23 before continuing."
        )

    print(f"  Decision: {decision}")
    print(f"  {reason}")

    # Save
    output = {
        "flagged_subjects": FLAGGED_SUBJECTS,
        "model_results": results,
        "avg_delta": round(avg_delta, 4),
        "decision": decision,
        "reason": reason,
    }
    output_path = output_dir / "dreamer_ica_check.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
