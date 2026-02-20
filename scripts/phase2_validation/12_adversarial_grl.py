"""
Script 12 – Adversarial Subject Removal (GRL)
================================================
Train model with Gradient Reversal Layer:
  - minimize stress classification loss
  - maximize subject confusion (via gradient reversal)

If performance holds → model truly learns stress, not subject identity.

Requires PyTorch for full GRL. Falls back to sklearn balanced training
if PyTorch is not installed.

Usage:
    python scripts/phase2_validation/12_adversarial_grl.py
    python scripts/phase2_validation/12_adversarial_grl.py --dataset dreamer
    python scripts/phase2_validation/12_adversarial_grl.py --lambda 0.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
from pathlib import Path
from config.settings import PROJECT_ROOT
from src.validation.data_loader import (
    load_wesad_features, load_dreamer_features, describe_dataset,
)
from src.validation.adversarial import adversarial_subject_removal


def main():
    parser = argparse.ArgumentParser(description="Adversarial GRL Subject Removal")
    parser.add_argument("--dataset", choices=["wesad", "dreamer", "both"],
                        default="both", help="Dataset to test")
    parser.add_argument("--lambda", dest="grl_lambda", type=float, default=1.0,
                        help="GRL lambda (gradient reversal strength)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs for PyTorch model")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "reports" / "validation"

    datasets_to_run = []
    if args.dataset in ("wesad", "both"):
        datasets_to_run.append("wesad")
    if args.dataset in ("dreamer", "both"):
        datasets_to_run.append("dreamer")

    for ds in datasets_to_run:
        print("\n" + "#" * 70)
        print(f"#  ADVERSARIAL SUBJECT REMOVAL — {ds.upper()}")
        print("#" * 70)

        if ds == "wesad":
            X, y, subjects, feature_cols = load_wesad_features()
        elif ds == "dreamer":
            X, y, subjects, feature_cols = load_dreamer_features()

        describe_dataset(X, y, subjects, name=f"{ds.upper()} Features")

        result = adversarial_subject_removal(
            X, y, subjects,
            grl_lambda=args.grl_lambda,
            epochs=args.epochs,
            verbose=True,
            output_dir=output_dir,
            dataset_name=ds,
        )


if __name__ == "__main__":
    main()
