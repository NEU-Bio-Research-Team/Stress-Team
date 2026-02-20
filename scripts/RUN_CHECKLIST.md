# Phase 3 Run Checklist

> **Environment**: `stress` conda env (has PyTorch 2.5.1+cu121)
> **Python**: `C:\Users\LENOVO\anaconda3\envs\stress\python.exe`
> **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU

---

## Pre-flight Check

```powershell
# Activate environment
conda activate stress

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected: `PyTorch 2.5.1+cu121, CUDA=True, GPU=NVIDIA GeForce RTX 3050 Laptop GPU`

---

## Execution Order (STRICT — each step depends on the previous)

### Step 1: DREAMER Recovery (Script 16)
```powershell
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/16_dreamer_recovery.py
```
- **Purpose**: Test advisor's recovery strategy (within-subject z-norm + target redefinition)
- **Experiments**: 5 configurations (no-norm/z-norm × stress/arousal/valence)
- **Est. time**: 15-30 min (5 experiments × subject probe + LogReg + RF each)
- **Output**: `reports/validation/dreamer_recovery_results.json`
- **Key to check**: Look at the SUMMARY table and `_summary.decision`:
  - `RECOVERED` (bal_acc ≥ 0.65) → proceed to Step 3
  - `PARTIAL_RECOVERY` (0.58-0.65) → proceed to Step 3 cautiously
  - `NO_RECOVERY` (< 0.58) → skip Step 3, accept as negative control

### Step 2: WESAD Adversarial Re-run with PyTorch GRL (Script 12)
```powershell
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/12_adversarial_grl.py --dataset wesad --epochs 100
```
- **Purpose**: Re-validate WESAD adversarial with real PyTorch GRL (previously used sklearn fallback)
- **Est. time**: 5-10 min (15 folds × 100 epochs neural net training)
- **Output**: `reports/validation/adversarial_results_wesad.json`
- **Key to check**: `backend` should say `"pytorch"` (not `"sklearn_fallback"`), verdict should still be `ROBUST`
- **Why this matters**: Original run used sklearn approximation. This uses actual gradient reversal — stronger validation.

### Step 3: DREAMER Post-Recovery Validation (Script 18)
```powershell
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/18_dreamer_post_recovery_validation.py
```
- **Purpose**: Validate that recovered DREAMER signal is genuine (adversarial + ablation + learning curve)
- **Depends on**: Script 16 must complete first (reads `dreamer_recovery_results.json`)
- **Est. time**: 30-60 min (adversarial GRL 23 folds + 5 band ablations + learning curve)
- **Output**: `reports/validation/dreamer_post_recovery_validation.json`
- **Key to check**: `_summary.overall_verdict`:
  - `GENUINE_SIGNAL` → DREAMER is rescued, proceed to deep model
  - `WEAK_SIGNAL` → marginal, use as secondary dataset
  - `NO_GENUINE_SIGNAL` → recovery was superficial, keep as negative control

### Step 4: WESAD Deep Model (Script 17)
```powershell
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/17_wesad_deep_model.py
```
- **Purpose**: Beat LogReg baseline (0.763) with 1D-CNN on raw ECG
- **Models**: TinyCNN1D (ECG-only) + HybridCNN (ECG + handcrafted features)
- **Est. time**: 30-60 min (2 models × 15 folds × 50 epochs each, GPU-accelerated)
- **Output**: `reports/validation/deep_model_results_wesad.json`
- **Key to check**: `_comparison.verdict`:
  - `DEEP_MODEL_WINS` → CNN > 0.783 (baseline+0.02). Great for paper.
  - `COMPARABLE` → CNN ≈ 0.763. Signal is in HRV statistics, not morphology. Still publishable.
  - `BASELINE_BETTER` → CNN < 0.743. Raw ECG doesn't add value. Focus on handcrafted features.

---

## Quick-Run All (copy-paste block)

```powershell
conda activate stress

# Step 1: DREAMER Recovery (~15-30 min)
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/16_dreamer_recovery.py 2>&1 | Tee-Object -FilePath scripts/phase2_validation/logs/16_output.log

# Step 2: WESAD Adversarial re-run (~5-10 min)
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/12_adversarial_grl.py --dataset wesad --epochs 100 2>&1 | Tee-Object -FilePath scripts/phase2_validation/logs/12_rerun_output.log

# Step 3: DREAMER Post-Recovery (~30-60 min, requires Step 1)
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/18_dreamer_post_recovery_validation.py 2>&1 | Tee-Object -FilePath scripts/phase2_validation/logs/18_output.log

# Step 4: WESAD Deep Model (~30-60 min, GPU)
& "C:\Users\LENOVO\anaconda3\envs\stress\python.exe" -u scripts/phase2_validation/17_wesad_deep_model.py 2>&1 | Tee-Object -FilePath scripts/phase2_validation/logs/17_output.log
```

---

## Output Files to Bring Back

After all scripts complete, share these with me for analysis:

| File | Script | What I'll analyze |
|------|--------|-------------------|
| `reports/validation/dreamer_recovery_results.json` | 16 | Which config rescued DREAMER? |
| `reports/validation/adversarial_results_wesad.json` | 12 | PyTorch GRL confirms ROBUST? |
| `reports/validation/dreamer_post_recovery_validation.json` | 18 | Is recovered signal genuine? |
| `reports/validation/deep_model_results_wesad.json` | 17 | Does CNN beat LogReg? |

Or just copy-paste the terminal output — it includes all key metrics.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | Ensure `stress` env is active: `conda activate stress` |
| `UnicodeEncodeError` | Scripts have UTF-8 reconfigure built-in. If still fails: `$env:PYTHONIOENCODING="utf-8"` |
| `CUDA out of memory` (script 17) | Reduce batch_size in script 17: find `batch_size=128` and change to `64` |
| Script 18 says "recovery results not found" | Run script 16 first |
| Slow subject probe (script 16) | Normal — subsample optimization limits to 10K samples, should take ~2min per experiment |

---

## Decision Tree After Results

```
Script 16 result?
├── RECOVERED (≥0.65)
│   └── Run Script 18 → GENUINE_SIGNAL?
│       ├── YES → DREAMER is viable! Build deep model for DREAMER too
│       └── NO → Accept as negative control  
├── PARTIAL_RECOVERY (0.58-0.65)
│   └── Run Script 18 anyway → provides data for paper discussion
└── NO_RECOVERY (<0.58)
    └── Skip Script 18. DREAMER stays as negative control (this is still good for paper)

Script 12 (re-run) result?
├── ROBUST → Strongest possible validation (real GRL confirms signal)
└── SUBJECT_DEPENDENT → Unexpected! Means sklearn fallback missed something
    └── Investigate: which subjects degrade most?

Script 17 result?
├── DEEP_MODEL_WINS → Paper contribution: raw ECG morphology contains stress signal
├── COMPARABLE → Paper conclusion: stress signal is in HRV statistics, not waveform
└── BASELINE_BETTER → Use LogReg as final model, mention CNN negative result
```
