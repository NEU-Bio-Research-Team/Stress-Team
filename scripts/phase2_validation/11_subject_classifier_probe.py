"""
Script 11 – Subject Classifier Probe + Feature Stability
==========================================================
Test if features encode subject identity rather than stress.

Test 1 — Subject Classifier Probe:
    Train RF: features → subject_id
    If accuracy >> 1/n_subjects → features encode subject identity

Test 2 — Permutation Test:
    Shuffle labels → train → expect accuracy ≈ chance
    If not → data leakage

Test 3 — Feature Importance Stability:
    Compare RF feature importance rankings across LOSOCV folds
    If top features change → unreliable signal

Usage:
    python scripts/phase2_validation/11_subject_classifier_probe.py
    python scripts/phase2_validation/11_subject_classifier_probe.py --dataset dreamer
    python scripts/phase2_validation/11_subject_classifier_probe.py --n-perm 200
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
from pathlib import Path
from config.settings import PROJECT_ROOT
from src.validation.data_loader import (
    load_wesad_features, load_dreamer_features, describe_dataset,
)
from src.validation.shortcut_detection import run_all_shortcut_tests


def main():
    parser = argparse.ArgumentParser(description="Shortcut Detection Battery")
    parser.add_argument("--dataset", choices=["wesad", "dreamer", "both"],
                        default="both", help="Dataset to test")
    parser.add_argument("--n-perm", type=int, default=100,
                        help="Number of permutations for permutation test")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "reports" / "validation"

    datasets_to_run = []
    if args.dataset in ("wesad", "both"):
        datasets_to_run.append("wesad")
    if args.dataset in ("dreamer", "both"):
        datasets_to_run.append("dreamer")

    for ds in datasets_to_run:
        print("\n" + "#" * 70)
        print(f"#  SHORTCUT DETECTION — {ds.upper()}")
        print("#" * 70)

        if ds == "wesad":
            X, y, subjects, feature_cols = load_wesad_features()
        elif ds == "dreamer":
            X, y, subjects, feature_cols = load_dreamer_features()

        describe_dataset(X, y, subjects, name=f"{ds.upper()} Features")

        results = run_all_shortcut_tests(
            X, y, subjects,
            dataset=ds,
            feature_cols=feature_cols,
            n_permutations=args.n_perm,
            output_dir=output_dir,
            verbose=True,
        )


if __name__ == "__main__":
    main()
