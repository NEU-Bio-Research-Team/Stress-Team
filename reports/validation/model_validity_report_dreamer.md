# Model Validity Report — DREAMER

> Generated: 2026-02-20 12:33:20
> Dataset: DREAMER

## Section 1 — Learnability Test

**Question**: Does a learnable signal exist in the data?

### 1.1 Feature Effect Sizes (Cohen's d)

| Feature | Cohen's d | Strength |
|---------|-----------|----------|
| f4 | +0.1800 | NEGLIGIBLE |
| f2 | +0.1555 | NEGLIGIBLE |
| f1 | +0.1491 | NEGLIGIBLE |
| f49 | -0.1393 | NEGLIGIBLE |
| f44 | -0.1334 | NEGLIGIBLE |
| f35 | -0.1303 | NEGLIGIBLE |
| f0 | +0.1264 | NEGLIGIBLE |
| f69 | +0.1255 | NEGLIGIBLE |
| f43 | -0.1229 | NEGLIGIBLE |
| f41 | -0.1215 | NEGLIGIBLE |
| f48 | -0.1191 | NEGLIGIBLE |
| f3 | +0.1113 | NEGLIGIBLE |
| f36 | -0.1112 | NEGLIGIBLE |
| f30 | -0.1106 | NEGLIGIBLE |
| f38 | -0.1018 | NEGLIGIBLE |
| f50 | -0.1005 | NEGLIGIBLE |
| f57 | +0.0982 | NEGLIGIBLE |
| f45 | -0.0900 | NEGLIGIBLE |
| f31 | -0.0888 | NEGLIGIBLE |
| f20 | -0.0877 | NEGLIGIBLE |
| f23 | -0.0832 | NEGLIGIBLE |
| f33 | -0.0825 | NEGLIGIBLE |
| f46 | -0.0796 | NEGLIGIBLE |
| f21 | -0.0785 | NEGLIGIBLE |
| f7 | +0.0740 | NEGLIGIBLE |
| f28 | -0.0682 | NEGLIGIBLE |
| f59 | +0.0669 | NEGLIGIBLE |
| f39 | -0.0642 | NEGLIGIBLE |
| f40 | -0.0598 | NEGLIGIBLE |
| f24 | -0.0596 | NEGLIGIBLE |
| f9 | +0.0594 | NEGLIGIBLE |
| f11 | -0.0565 | NEGLIGIBLE |
| f58 | +0.0551 | NEGLIGIBLE |
| f63 | -0.0546 | NEGLIGIBLE |
| f29 | -0.0511 | NEGLIGIBLE |
| f66 | +0.0510 | NEGLIGIBLE |
| f55 | +0.0498 | NEGLIGIBLE |
| f56 | +0.0497 | NEGLIGIBLE |
| f26 | -0.0495 | NEGLIGIBLE |
| f32 | -0.0484 | NEGLIGIBLE |
| f68 | +0.0468 | NEGLIGIBLE |
| f5 | +0.0464 | NEGLIGIBLE |
| f10 | -0.0445 | NEGLIGIBLE |
| f34 | -0.0441 | NEGLIGIBLE |
| f53 | -0.0440 | NEGLIGIBLE |
| f22 | -0.0417 | NEGLIGIBLE |
| f17 | +0.0413 | NEGLIGIBLE |
| f67 | +0.0407 | NEGLIGIBLE |
| f27 | -0.0397 | NEGLIGIBLE |
| f25 | -0.0390 | NEGLIGIBLE |
| f6 | +0.0381 | NEGLIGIBLE |
| f47 | -0.0378 | NEGLIGIBLE |
| f51 | -0.0352 | NEGLIGIBLE |
| f42 | -0.0346 | NEGLIGIBLE |
| f64 | -0.0286 | NEGLIGIBLE |
| f19 | +0.0275 | NEGLIGIBLE |
| f16 | +0.0185 | NEGLIGIBLE |
| f65 | +0.0168 | NEGLIGIBLE |
| f60 | +0.0158 | NEGLIGIBLE |
| f37 | -0.0154 | NEGLIGIBLE |
| f8 | +0.0135 | NEGLIGIBLE |
| f61 | +0.0133 | NEGLIGIBLE |
| f62 | +0.0124 | NEGLIGIBLE |
| f12 | -0.0105 | NEGLIGIBLE |
| f52 | +0.0099 | NEGLIGIBLE |
| f15 | +0.0093 | NEGLIGIBLE |
| f14 | +0.0084 | NEGLIGIBLE |
| f54 | +0.0063 | NEGLIGIBLE |
| f18 | -0.0062 | NEGLIGIBLE |
| f13 | -0.0033 | NEGLIGIBLE |

### 1.2 Baseline Model Comparison (LOSOCV)

| Model | Balanced Acc | F1 | AUC-ROC | Type |
|-------|--------------|----|---------|------|
| logistic | 0.541 | 0.486 | 0.560 | linear |
| rf | 0.538 | 0.489 | 0.552 | nonlinear_ensemble |
| mlp | 0.515 | 0.433 | 0.531 | neural_network |

### 1.3 Decision: **NO_SIGNAL**

Best baseline achieves balanced_accuracy=0.541 (≈random). Signal does not exist in current features. Deep model will be meaningless. Re-examine feature extraction or stress proxy definition.

### 1.4 Learning Curve

| Train Subjects | Balanced Acc | F1 |
|----------------|--------------|-----|
| 4 | 0.521 ± 0.027 | 0.492 ± 0.035 |
| 7 | 0.531 ± 0.026 | 0.441 ± 0.123 |
| 11 | 0.531 ± 0.010 | 0.507 ± 0.018 |
| 16 | 0.552 ± 0.017 | 0.518 ± 0.029 |
| 22 | 0.491 ± 0.053 | 0.362 ± 0.153 |

## Section 2 — Shortcut Detection

**Question**: Is the model learning stress or subject identity?

### 2.1 Subject Classifier Probe

- Chance level: 0.0435
- Probe accuracy: 0.9258
- Encoding ratio: 21.29×
- **Verdict**: HIGH_SUBJECT_ENCODING
- Probe accuracy 92.6% is 21.3× chance (4.3%). Features STRONGLY encode subject identity. Model may learn subject, not stress. MUST use adversarial GRL or subject normalization.

### 2.2 Permutation Test

- True balanced accuracy: 0.5411
- Permuted mean: 0.4999 ± 0.0018
- **p-value**: 0.0
- **Verdict**: SIGNIFICANT
- p = 0.0000 < 0.01. Model performance is highly significant. The learned signal is NOT due to chance or label noise.

### 2.3 Feature Importance Stability

- Mean Kendall's τ: 0.8455
- Top feature agreement: 0.913
- Most common top feature: f58
- **Verdict**: STABLE
- Mean Kendall's τ = 0.845 > 0.7. Feature importance rankings are stable across folds. The model is learning consistent patterns.

### 2.4 Overall Shortcut Assessment: **SHORTCUT_DETECTED**

WARNING: Shortcut learning indicators found. Model may not be learning genuine stress signal. Adversarial training (GRL) is REQUIRED.

## Section 3 — Cross-Subject Generalization

**Question**: Does the model generalize to unseen subjects?

### 3.1 Per-Subject Performance (Best Baseline)

Model: **LogisticRegression**
- Mean: 0.541 ± 0.053
- Range: [0.391, 0.636]
- Spread: 0.245

✅ Inter-subject variability is within acceptable range.

## Section 4 — Adversarial Subject Removal

**Question**: Does removing subject information hurt performance?

- Backend: pytorch
- Standard balanced acc: 0.5415
- Adversarial balanced acc: 0.5376
- Delta: -0.004
- **Verdict**: ROBUST
- Adversarial model performance (0.538) ≈ standard (0.542), Δ=-0.004. Model genuinely learns stress, not subject identity.

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
| Learnability | NO_SIGNAL | Investigate |
| Shortcut Detection | SHORTCUT_DETECTED | Use GRL |
| Adversarial (GRL) | ROBUST | Model is genuine |
