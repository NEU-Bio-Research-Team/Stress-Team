"""
Script 05 – Preprocess DREAMER
================================
Run full DREAMER preprocessing pipeline:
  EEG bandpass → notch → baseline subtraction → DE features → stress proxy

Usage:
    python scripts/05_preprocess_dreamer.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import ensure_dirs
from src.preprocessing.dreamer_preprocess import preprocess_all


def main():
    ensure_dirs()
    print("Preprocessing DREAMER: all subjects × all trials")
    preprocess_all()
    print("\nDone. Output in data/processed/dreamer/")


if __name__ == "__main__":
    main()
