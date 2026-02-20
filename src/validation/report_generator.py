"""
Model Validity Report Generator
==================================
Creates a comprehensive Markdown report covering all validation tests,
structured to pre-answer the 5 questions reviewers WILL ask.

Sections:
  1. Learnability Test (baselines + effect sizes)
  2. Shortcut Detection (subject probe + permutation + stability)
  3. Cross-Subject Generalization (LOSOCV per-subject breakdown)
  4. Feature Importance Stability (rank consistency across folds)
  5. Failure Cases (worst-performing subjects & analysis)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def generate_validity_report(
    baseline_results: Optional[Dict[str, Any]] = None,
    shortcut_results: Optional[Dict[str, Any]] = None,
    adversarial_results: Optional[Dict[str, Any]] = None,
    dataset: str = "wesad",
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate a comprehensive Model Validity Report in Markdown.

    This report is designed to pre-answer reviewer questions:
      Q1: Does the signal actually exist?
      Q2: Is the model learning stress or subject identity?
      Q3: Does the model generalize across subjects?
      Q4: Are the features reliable?
      Q5: Where does the model fail?
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append(f"# Model Validity Report — {dataset.upper()}")
    lines.append(f"\n> Generated: {timestamp}")
    lines.append(f"> Dataset: {dataset.upper()}")
    lines.append("")

    # ─────────────────────── Section 1: Learnability ───────────────────────
    lines.append("## Section 1 — Learnability Test")
    lines.append("")
    lines.append("**Question**: Does a learnable signal exist in the data?")
    lines.append("")

    if baseline_results:
        # Effect sizes
        effect_sizes = baseline_results.get("effect_sizes", {})
        if effect_sizes:
            lines.append("### 1.1 Feature Effect Sizes (Cohen's d)")
            lines.append("")
            lines.append("| Feature | Cohen's d | Strength |")
            lines.append("|---------|-----------|----------|")
            for feat, d in sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True):
                strength = (
                    "LARGE" if abs(d) >= 0.8 else
                    "MEDIUM" if abs(d) >= 0.5 else
                    "SMALL" if abs(d) >= 0.2 else
                    "NEGLIGIBLE"
                )
                lines.append(f"| {feat} | {d:+.4f} | {strength} |")
            lines.append("")

        # Baseline comparison
        baselines = baseline_results.get("baselines", {})
        if baselines:
            lines.append("### 1.2 Baseline Model Comparison (LOSOCV)")
            lines.append("")
            lines.append("| Model | Balanced Acc | F1 | AUC-ROC | Type |")
            lines.append("|-------|--------------|----|---------|------|")
            for name, b in baselines.items():
                bal = b.get("balanced_accuracy_mean", "N/A")
                f1 = b.get("f1_mean", "N/A")
                auc = b.get("auc_roc_mean", "N/A")
                mtype = b.get("type", "N/A")
                bal_str = f"{bal:.3f}" if isinstance(bal, float) else bal
                f1_str = f"{f1:.3f}" if isinstance(f1, float) else f1
                auc_str = f"{auc:.3f}" if isinstance(auc, float) else auc
                lines.append(f"| {name} | {bal_str} | {f1_str} | {auc_str} | {mtype} |")
            lines.append("")

        # Decision
        decision = baseline_results.get("decision", "UNKNOWN")
        rec = baseline_results.get("recommendation", "")
        lines.append(f"### 1.3 Decision: **{decision}**")
        lines.append(f"\n{rec}")
        lines.append("")

        # Learning curve
        lc = baseline_results.get("learning_curve", {})
        if lc and lc.get("curve"):
            lines.append("### 1.4 Learning Curve")
            lines.append("")
            lines.append("| Train Subjects | Balanced Acc | F1 |")
            lines.append("|----------------|--------------|-----|")
            for point in lc["curve"]:
                lines.append(
                    f"| {point['n_train_subjects']} | "
                    f"{point['balanced_accuracy_mean']:.3f} ± {point['balanced_accuracy_std']:.3f} | "
                    f"{point['f1_mean']:.3f} ± {point['f1_std']:.3f} |"
                )
            lines.append("")
    else:
        lines.append("*Not yet computed. Run `10_learnability_baselines.py` first.*")
        lines.append("")

    # ─────────────────────── Section 2: Shortcut Detection ───────────────────────
    lines.append("## Section 2 — Shortcut Detection")
    lines.append("")
    lines.append("**Question**: Is the model learning stress or subject identity?")
    lines.append("")

    if shortcut_results:
        # Subject probe
        sp = shortcut_results.get("subject_probe", {})
        if sp:
            lines.append("### 2.1 Subject Classifier Probe")
            lines.append("")
            lines.append(f"- Chance level: {sp.get('chance_level', 'N/A')}")
            lines.append(f"- Probe accuracy: {sp.get('probe_accuracy_mean', 'N/A')}")
            lines.append(f"- Encoding ratio: {sp.get('encoding_ratio', 'N/A')}×")
            lines.append(f"- **Verdict**: {sp.get('verdict', 'N/A')}")
            lines.append(f"- {sp.get('interpretation', '')}")
            lines.append("")

        # Permutation test
        pt = shortcut_results.get("permutation_test", {})
        if pt:
            lines.append("### 2.2 Permutation Test")
            lines.append("")
            lines.append(f"- True balanced accuracy: {pt.get('true_balanced_accuracy', 'N/A')}")
            lines.append(f"- Permuted mean: {pt.get('permuted_mean', 'N/A')} ± {pt.get('permuted_std', 'N/A')}")
            lines.append(f"- **p-value**: {pt.get('p_value', 'N/A')}")
            lines.append(f"- **Verdict**: {pt.get('verdict', 'N/A')}")
            lines.append(f"- {pt.get('interpretation', '')}")
            lines.append("")

        # Feature stability
        fs = shortcut_results.get("feature_stability", {})
        if fs:
            lines.append("### 2.3 Feature Importance Stability")
            lines.append("")
            lines.append(f"- Mean Kendall's τ: {fs.get('kendall_tau_mean', 'N/A')}")
            lines.append(f"- Top feature agreement: {fs.get('top_feature_agreement', 'N/A')}")
            lines.append(f"- Most common top feature: {fs.get('most_common_top_feature', 'N/A')}")
            lines.append(f"- **Verdict**: {fs.get('verdict', 'N/A')}")
            lines.append(f"- {fs.get('interpretation', '')}")
            lines.append("")

        overall = shortcut_results.get("overall_verdict", "UNKNOWN")
        overall_interp = shortcut_results.get("overall_interpretation", "")
        lines.append(f"### 2.4 Overall Shortcut Assessment: **{overall}**")
        lines.append(f"\n{overall_interp}")
        lines.append("")
    else:
        lines.append("*Not yet computed. Run `11_subject_classifier_probe.py` and `12_permutation_test.py` first.*")
        lines.append("")

    # ─────────────────────── Section 3: Cross-Subject ───────────────────────
    lines.append("## Section 3 — Cross-Subject Generalization")
    lines.append("")
    lines.append("**Question**: Does the model generalize to unseen subjects?")
    lines.append("")

    if baseline_results and baseline_results.get("baselines"):
        best = baseline_results.get("best_baseline", "logistic")
        best_detail_key = f"{best}_detail"
        # Try to find per-subject results from the full output
        lines.append("### 3.1 Per-Subject Performance (Best Baseline)")
        lines.append("")

        baselines = baseline_results.get("baselines", {})
        best_b = baselines.get(best, {})
        if best_b:
            lines.append(f"Model: **{best_b.get('model', best)}**")
            bal_mean = best_b.get("balanced_accuracy_mean", 0)
            bal_std = best_b.get("balanced_accuracy_std", 0)
            bal_min = best_b.get("balanced_accuracy_min", 0)
            bal_max = best_b.get("balanced_accuracy_max", 0)
            lines.append(f"- Mean: {bal_mean:.3f} ± {bal_std:.3f}")
            lines.append(f"- Range: [{bal_min:.3f}, {bal_max:.3f}]")
            lines.append(f"- Spread: {bal_max - bal_min:.3f}")
            lines.append("")

            if bal_max - bal_min > 0.3:
                lines.append(
                    "⚠️ **High inter-subject variability** (range > 0.3). "
                    "Some subjects are much harder to classify. "
                    "Consider subject-specific fine-tuning or domain adaptation."
                )
            else:
                lines.append(
                    "✅ Inter-subject variability is within acceptable range."
                )
            lines.append("")
    else:
        lines.append("*Not yet computed.*")
        lines.append("")

    # ─────────────────────── Section 4: Adversarial ───────────────────────
    lines.append("## Section 4 — Adversarial Subject Removal")
    lines.append("")
    lines.append("**Question**: Does removing subject information hurt performance?")
    lines.append("")

    if adversarial_results:
        lines.append(f"- Backend: {adversarial_results.get('backend', 'N/A')}")
        lines.append(f"- Standard balanced acc: {adversarial_results.get('standard_bal_acc_mean', 'N/A')}")
        lines.append(f"- Adversarial balanced acc: {adversarial_results.get('adversarial_bal_acc_mean', 'N/A')}")
        lines.append(f"- Delta: {adversarial_results.get('delta', 'N/A')}")
        lines.append(f"- **Verdict**: {adversarial_results.get('verdict', 'N/A')}")
        lines.append(f"- {adversarial_results.get('interpretation', '')}")
        lines.append("")
    else:
        lines.append("*Not yet computed. Run `13_adversarial_grl.py` first.*")
        lines.append("")

    # ─────────────────────── Section 5: Failure Cases ───────────────────────
    lines.append("## Section 5 — Failure Cases")
    lines.append("")
    lines.append("**Question**: Where does the model fail, and why?")
    lines.append("")

    if baseline_results and baseline_results.get("baselines"):
        lines.append("### 5.1 Failure Analysis Guidelines")
        lines.append("")
        lines.append("After running all validation scripts, examine:")
        lines.append("1. **Worst-performing subjects**: Which subjects have bal_acc < 0.5?")
        lines.append("2. **Stress ratio correlation**: Do low-stress-ratio subjects fail more?")
        lines.append("3. **Feature distribution**: Are failing subjects' features significantly different?")
        lines.append("4. **Artifact correlation**: Do subjects with flagged artifacts perform worse?")
        lines.append("")
        lines.append("See per-subject detail in `reports/validation/baseline_results_*.json`")
        lines.append("")
    else:
        lines.append("*Populate after running baseline experiments.*")
        lines.append("")

    # ─────────────────────── Summary ───────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Test | Verdict | Action |")
    lines.append("|------|---------|--------|")

    if baseline_results:
        decision = baseline_results.get("decision", "N/A")
        lines.append(f"| Learnability | {decision} | {'Proceed to deep model' if decision == 'STRONG_SIGNAL' else 'Investigate'} |")
    else:
        lines.append("| Learnability | Not run | Run script 10 |")

    if shortcut_results:
        overall = shortcut_results.get("overall_verdict", "N/A")
        lines.append(f"| Shortcut Detection | {overall} | {'Clean' if overall == 'CLEAN' else 'Use GRL'} |")
    else:
        lines.append("| Shortcut Detection | Not run | Run scripts 11-12 |")

    if adversarial_results:
        verdict = adversarial_results.get("verdict", "N/A")
        lines.append(f"| Adversarial (GRL) | {verdict} | {'Model is genuine' if verdict == 'ROBUST' else 'Needs work'} |")
    else:
        lines.append("| Adversarial (GRL) | Not run | Run script 13 |")

    lines.append("")

    report = "\n".join(lines)

    # Save
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"model_validity_report_{dataset}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  Report saved to {report_path}")

    return report
