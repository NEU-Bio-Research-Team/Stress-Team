# Model Validity Report — WESAD

> Generated: 2026-02-20 12:33:20
> Dataset: WESAD

## Section 1 — Learnability Test

**Question**: Does a learnable signal exist in the data?

### 1.1 Feature Effect Sizes (Cohen's d)

| Feature | Cohen's d | Strength |
|---------|-----------|----------|
| hr_mean | +1.5541 | LARGE |
| eda_std | +0.4009 | SMALL |
| eda_mean | +0.3325 | SMALL |
| rmssd | -0.2890 | SMALL |
| hr_std | +0.2316 | SMALL |
| sdnn | -0.2018 | SMALL |
| eda_slope | +0.0859 | NEGLIGIBLE |

### 1.2 Baseline Model Comparison (LOSOCV)

| Model | Balanced Acc | F1 | AUC-ROC | Type |
|-------|--------------|----|---------|------|
| logistic | 0.763 | 0.444 | 0.892 | linear |
| rf | 0.704 | 0.405 | 0.873 | nonlinear_ensemble |
| mlp | 0.650 | 0.319 | 0.891 | neural_network |

### 1.3 Decision: **STRONG_SIGNAL**

Best baseline (LogisticRegression) achieves balanced_accuracy=0.763 (>0.70). Signal is strong. Deep model can likely reach >85%. Proceed.

### 1.4 Learning Curve

| Train Subjects | Balanced Acc | F1 |
|----------------|--------------|-----|
| 3 | 0.772 ± 0.025 | 0.444 ± 0.043 |
| 5 | 0.761 ± 0.031 | 0.458 ± 0.028 |
| 7 | 0.780 ± 0.028 | 0.475 ± 0.014 |
| 10 | 0.787 ± 0.019 | 0.454 ± 0.031 |
| 14 | 0.672 ± 0.095 | 0.317 ± 0.096 |

## Section 2 — Shortcut Detection

**Question**: Is the model learning stress or subject identity?

### 2.1 Subject Classifier Probe

- Chance level: 0.0667
- Probe accuracy: 0.7727
- Encoding ratio: 11.59×
- **Verdict**: HIGH_SUBJECT_ENCODING
- Probe accuracy 77.3% is 11.6× chance (6.7%). Features STRONGLY encode subject identity. Model may learn subject, not stress. MUST use adversarial GRL or subject normalization.

### 2.2 Permutation Test

- True balanced accuracy: 0.7629
- Permuted mean: 0.4997 ± 0.0057
- **p-value**: 0.0
- **Verdict**: SIGNIFICANT
- p = 0.0000 < 0.01. Model performance is highly significant. The learned signal is NOT due to chance or label noise.

### 2.3 Feature Importance Stability

- Mean Kendall's τ: 0.9873
- Top feature agreement: 1.0
- Most common top feature: hr_mean
- **Verdict**: STABLE
- Mean Kendall's τ = 0.987 > 0.7. Feature importance rankings are stable across folds. The model is learning consistent patterns.

### 2.4 Overall Shortcut Assessment: **SHORTCUT_DETECTED**

WARNING: Shortcut learning indicators found. Model may not be learning genuine stress signal. Adversarial training (GRL) is REQUIRED.

## Section 3 — Cross-Subject Generalization

**Question**: Does the model generalize to unseen subjects?

### 3.1 Per-Subject Performance (Best Baseline)

Model: **LogisticRegression**
- Mean: 0.763 ± 0.115
- Range: [0.506, 0.879]
- Spread: 0.373

⚠️ **High inter-subject variability** (range > 0.3). Some subjects are much harder to classify. Consider subject-specific fine-tuning or domain adaptation.

## Section 4 — Adversarial Subject Removal

**Question**: Does removing subject information hurt performance?

- Backend: sklearn_fallback
- Standard balanced acc: 0.7498
- Adversarial balanced acc: 0.7517
- Delta: 0.0019
- **Verdict**: ROBUST
- Adversarial model performance (0.752) ≈ standard (0.750), Δ=+0.002. Model genuinely learns stress, not subject identity.

## Section 5 — Failure Cases

**Question**: Where does the model fail, and why?

### 5.1 Failure Analysis Guidelines

After running all validation scripts, examine:
1. **Worst-performing subjects**: Which subjects have bal_acc < 0.5?
2. **Stress ratio correlation**: Do low-stress-ratio subjects fail more?
3. **Feature distribution**: Are failing subjects' features significantly different?
4. **Artifact correlation**: Do subjects with flagged artifacts perform worse?

See per-subject detail in `reports/validation/baseline_results_*.json`

---

## Summary Table

| Test | Verdict | Action |
|------|---------|--------|
| Learnability | STRONG_SIGNAL | Proceed to deep model |
| Shortcut Detection | SHORTCUT_DETECTED | Use GRL |
| Adversarial (GRL) | ROBUST | Model is genuine |
