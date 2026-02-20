# Scripts — Execution Guide

> **Bio Stage (Stage 1): CLOSED** — All 25 scripts complete  
> See `reports/BIO_STAGE_CLOSING.md` for full analysis

## Directory Structure

```
scripts/
├── README.md                        ← This file
├── RUN_CHECKLIST.md                 ← Execution checklist
│
├── phase1_data_engineering/         ─┐ Phase 1: Data Engineering (10 scripts)
│   ├── 00_fetch_tardis.py            │ Audit → Preprocess → Features → Alignment → Stylized Facts
│   ├── 01_audit_wesad.py             │ Status: ✅ COMPLETE (10/10)
│   ├── 02_audit_dreamer.py           │
│   ├── 03_audit_tardis.py            │
│   ├── 04_preprocess_wesad.py        │
│   ├── 05_preprocess_dreamer.py      │
│   ├── 06_preprocess_tardis.py       │
│   ├── 07_extract_features.py        │
│   ├── 08_alignment_check.py         │
│   └── 09_stylized_facts.py         ─┘
│
├── phase2_validation/               ─┐ Phase 2: Scientific Validation (6 scripts)
│   ├── 10_learnability_baselines.py   │ Baselines → Shortcuts → GRL → Minimal → ICA → Report
│   ├── 11_subject_classifier_probe.py │ Status: ✅ COMPLETE (6/6)
│   ├── 12_adversarial_grl.py         │
│   ├── 13_minimal_model.py           │
│   ├── 14_dreamer_ica_check.py       │
│   └── 15_generate_validity_report.py─┘
│
├── phase3_deep_models/              ─┐ Phase 3: Deep Model Exploration (3 scripts)
│   ├── 16_dreamer_recovery.py        │ DREAMER Recovery → WESAD CNN → Post-Recovery Validation
│   ├── 17_wesad_deep_model.py        │ Status: ✅ COMPLETE (3/3)
│   └── 18_dreamer_post_recovery.py  ─┘
│
├── phase3_improvements/             ─┐ Phase 3+: Advisor Hypotheses (4 scripts)
│   ├── 19_cnn_threshold_optimization │ Threshold → Connectivity → R-R Model → Label Ceiling
│   ├── 20_dreamer_connectivity.py    │ Status: ✅ COMPLETE (4/4)
│   ├── 21_rr_interval_model.py       │
│   └── 22_dreamer_label_noise.py    ─┘
│
├── phase4_representation/           ─┐ Phase 4: Stochastic Law Discovery (3 scripts)
│   ├── 23_representation_transfer.py  │ Transfer → Process ID → Final Validation
│   ├── 24_stress_process_id.py       │ Status: ✅ COMPLETE (3/3, θ artifact found)
│   └── 25_final_validation.py       ─┘
│
└── stage2_economics/                ─┐ Stage 2: Market Simulator & ABM
    └── README.md                     │ Status: NOT STARTED
                                     ─┘
```

## Execution Order (STRICT — do not reorder)

### Phase 1 — Data Engineering (COMPLETED)
```bash
python scripts/phase1_data_engineering/00_fetch_tardis.py
python scripts/phase1_data_engineering/01_audit_wesad.py
python scripts/phase1_data_engineering/02_audit_dreamer.py
python scripts/phase1_data_engineering/03_audit_tardis.py
python scripts/phase1_data_engineering/04_preprocess_wesad.py
python scripts/phase1_data_engineering/05_preprocess_dreamer.py
python scripts/phase1_data_engineering/06_preprocess_tardis.py
python scripts/phase1_data_engineering/07_extract_features.py
python scripts/phase1_data_engineering/08_alignment_check.py
python scripts/phase1_data_engineering/09_stylized_facts.py
```

### Phase 2 — Scientific Validation (COMPLETED)
```bash
python scripts/phase2_validation/10_learnability_baselines.py
python scripts/phase2_validation/11_subject_classifier_probe.py
python scripts/phase2_validation/12_adversarial_grl.py
python scripts/phase2_validation/13_minimal_model.py
python scripts/phase2_validation/14_dreamer_ica_check.py
python scripts/phase2_validation/15_generate_validity_report.py
```

### Phase 3 — Deep Model Exploration (COMPLETED)
```bash
python scripts/phase3_deep_models/16_dreamer_recovery.py
python scripts/phase3_deep_models/17_wesad_deep_model.py
python scripts/phase3_deep_models/18_dreamer_post_recovery_validation.py
```

### Phase 3+ — Advisor Hypotheses (COMPLETED)
```bash
python scripts/phase3_improvements/19_cnn_threshold_optimization.py
python scripts/phase3_improvements/20_dreamer_connectivity.py
python scripts/phase3_improvements/21_rr_interval_model.py
python scripts/phase3_improvements/22_dreamer_label_noise_ceiling.py
```

### Phase 4 — Stochastic Law Discovery (COMPLETED)
```bash
python scripts/phase4_representation/23_representation_transfer.py
python scripts/phase4_representation/24_stress_process_identification.py
python scripts/phase4_representation/25_final_validation.py
```

## Output

All results are saved to `reports/`:
- `reports/audit/` — Dataset audit reports
- `reports/alignment/` — Cross-dataset alignment
- `reports/validation/` — 21 JSON + 2 MD validation reports
- `reports/BIO_STAGE_CLOSING.md` — Comprehensive Bio Stage closing report
- `reports/PROGRESS.md` — Detailed progress log
