"""
Script 15 – Generate Model Validity Report
=============================================
Compile all Phase 2 validation results into a single Markdown report.

Reads from:
    reports/validation/baseline_results_*.json
    reports/validation/shortcut_results_*.json
    reports/validation/adversarial_results.json

Generates:
    reports/validation/model_validity_report_wesad.md
    reports/validation/model_validity_report_dreamer.md

This report pre-answers the 5 questions reviewers WILL ask:
    1. Does signal exist? (learnability)
    2. Is model learning stress or subject? (shortcut detection)
    3. Does it generalize? (cross-subject)
    4. Are features reliable? (importance stability)
    5. Where does it fail? (failure cases)

Usage:
    python scripts/phase2_validation/15_generate_validity_report.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from pathlib import Path
from config.settings import PROJECT_ROOT
from src.validation.report_generator import generate_validity_report


def load_json_safe(path):
    """Load JSON file, return None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    val_dir = PROJECT_ROOT / "reports" / "validation"

    for dataset in ["wesad", "dreamer"]:
        print(f"\n{'='*60}")
        print(f"  Generating Validity Report — {dataset.upper()}")
        print(f"{'='*60}")

        # Load results
        baseline_full = load_json_safe(val_dir / f"baseline_results_{dataset}.json")
        baseline_results = baseline_full.get("summary") if baseline_full else None

        shortcut_results = load_json_safe(val_dir / f"shortcut_results_{dataset}.json")
        # Try per-dataset file first, fall back to old shared file
        adversarial_results = load_json_safe(val_dir / f"adversarial_results_{dataset}.json")
        if adversarial_results is None:
            adversarial_results = load_json_safe(val_dir / "adversarial_results.json")

        # Generate
        report = generate_validity_report(
            baseline_results=baseline_results,
            shortcut_results=shortcut_results,
            adversarial_results=adversarial_results,
            dataset=dataset,
            output_dir=val_dir,
        )

        # Print summary
        if baseline_results:
            print(f"  Learnability:  {baseline_results.get('decision', 'N/A')}")
        if shortcut_results:
            print(f"  Shortcut:      {shortcut_results.get('overall_verdict', 'N/A')}")
        if adversarial_results:
            print(f"  Adversarial:   {adversarial_results.get('verdict', 'N/A')}")

        print(f"  Report saved to: {val_dir / f'model_validity_report_{dataset}.md'}")


if __name__ == "__main__":
    main()
