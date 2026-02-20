"""
Script 04 – Preprocess WESAD
==============================
Run full WESAD preprocessing pipeline:
  ECG bandpass → R-peak detection → EDA filtering → windowing → features

Usage:
    python scripts/04_preprocess_wesad.py
    python scripts/04_preprocess_wesad.py --subjects S2 S3 S4
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import WESAD_EXPECTED_SUBJECTS, ensure_dirs
from src.preprocessing.wesad_preprocess import preprocess_all


def main():
    parser = argparse.ArgumentParser(description="Preprocess WESAD data")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs (e.g., S2 S3). Default: all")
    args = parser.parse_args()

    ensure_dirs()
    subjects = args.subjects or WESAD_EXPECTED_SUBJECTS
    print(f"Preprocessing WESAD: {len(subjects)} subjects")
    preprocess_all(subject_ids=subjects)
    print("\nDone. Output in data/processed/wesad/")


if __name__ == "__main__":
    main()
