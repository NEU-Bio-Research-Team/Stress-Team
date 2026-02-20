# Algorithmic Panic — Workspace

> **"Endogenous Stress as State Variable in Financial Markets"**  
> Nghiên cứu mối quan hệ nhân quả giữa stress sinh lý và hành vi thị trường tài chính thông qua hệ thống ghép nối sinh-kỹ thuật (bio-technical coupled system).

---

## Mục lục

1. [Tổng quan nghiên cứu](#1-tổng-quan-nghiên-cứu)
2. [Trạng thái hiện tại](#2-trạng-thái-hiện-tại)
3. [Thiết lập môi trường (Setup)](#3-thiết-lập-môi-trường-setup)
4. [Cấu trúc workspace](#4-cấu-trúc-workspace)
5. [Hướng dẫn tái tạo kết quả (Reproduce)](#5-hướng-dẫn-tái-tạo-kết-quả-reproduce)
6. [Hệ thống tài liệu](#6-hệ-thống-tài-liệu)
7. [Kết quả chính của Bio Stage](#7-kết-quả-chính-của-bio-stage)

---

## 1. Tổng quan nghiên cứu

Nghiên cứu gồm 7 stages, được thiết kế trong tài liệu gốc:  
[`Algorithmic Panic  Full Analysis & Dataset Audit.md`](Algorithmic%20Panic%20%20Full%20Analysis%20%26%20Dataset%20Audit.md)

| Stage | Tên | Trạng thái |
|-------|-----|------------|
| 0 | Causal Model Construction (DAG) | Lý thuyết |
| **1** | **Stress Inference Engine (Bio Stage)** | **✅ CLOSED — 25 scripts, 5 phases** |
| 2 | Market Simulator (ABM) | Data curation xong, ABM chưa bắt đầu |
| 3 | Bio → Behavior Coupling | Chưa triển khai |
| 4 | Feedback Dynamical System | Chưa triển khai |
| 5 | Evidence Engine | Chưa triển khai |
| 6 | Policy Analysis | Chưa triển khai |

**Stage 1 (Bio Stage)** đã hoàn tất toàn bộ: data engineering, scientific validation, deep model exploration, advisor hypotheses, và stochastic law discovery. Kết quả chi tiết nằm trong [`reports/BIO_STAGE_CLOSING.md`](reports/BIO_STAGE_CLOSING.md).

---

## 2. Trạng thái hiện tại

Bio Stage đã trả lời được các câu hỏi cốt lõi:

| Câu hỏi | Kết quả | Bằng chứng |
|---------|---------|------------|
| Stress có phát hiện được từ ECG? | ✅ Có, bal_acc=0.763, hr_mean d=1.55 | Script 10 |
| Tín hiệu có phải là confound? | ✅ Không — GRL Δ=+0.014 (ROBUST) | Script 12 |
| Deep learning có vượt LogReg? | ❌ Không — LogReg 0.763 > CNN 0.686 | Script 17 |
| σ(t) tuân theo quy luật gì? | OU mean-reversion (15/15 subjects) | Script 24 |
| θ là hằng số sinh lý? | ❌ Không — artifact do window size | Script 25 |

→ Workspace sẵn sàng cho Stage 2 (ABM). Xem chi tiết tại [`reports/BIO_STAGE_CLOSING.md § IV`](reports/BIO_STAGE_CLOSING.md).

---

## 3. Thiết lập môi trường (Setup)

### 3.1 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|-----------|---------|
| **Python** | 3.10+ (tested: 3.10 trên Anaconda) |
| **GPU** | NVIDIA GPU + CUDA 12.1 (cho scripts 12, 16–25 dùng PyTorch) |
| **RAM** | ≥ 8 GB |
| **Disk** | ~15 GB (datasets + processed output) |

### 3.2 Tạo conda environment

```powershell
# Tạo environment
conda create -n stress python=3.10 -y
conda activate stress

# Cài dependencies cơ bản
pip install -r requirements.txt

# Cài PyTorch với CUDA 12.1 (BẮT BUỘC cho Phase 2–4)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Xác nhận PyTorch + GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
```

> **Lưu ý Windows:** Nếu gặp `UnicodeEncodeError` khi chạy scripts, set biến môi trường:
> ```powershell
> $env:PYTHONIOENCODING = "utf-8"
> ```

### 3.3 Chuẩn bị datasets (BẮT BUỘC)

Ba datasets cần được đặt đúng vị trí trong `data/raw/` **trước khi chạy bất kỳ script nào**:

#### Dataset 1: WESAD

- **Nguồn:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) hoặc [Zenodo](https://zenodo.org/records/7610964)
- **Tải about:** File `WESAD.zip` (~2.5 GB)
- **Giải nén vào:** `data/raw/wesad/` sao cho đường dẫn cuối cùng là:

```
data/raw/wesad/
└── WESAD_extracted/
    └── WESAD/
        ├── S2/
        │   ├── S2.pkl              ← File chính (chest + wrist)
        │   └── S2_respiban.txt     ← Fallback nếu .pkl lỗi
        ├── S3/
        ├── S4/
        ├── ...
        └── S17/
```

> **Quan trọng:** Pipeline đọc từ `data/raw/wesad/WESAD_extracted/WESAD/SXX/SXX.pkl`. Nếu đường dẫn khác, sửa `WESAD_RAW_DIR` trong `config/settings.py`.

#### Dataset 2: DREAMER

- **Nguồn:** [DREAMER Dataset](https://zenodo.org/records/546113) (cần request access)
- **Tải về:** File `DREAMER.mat` (~1.2 GB)
- **Đặt tại:**

```
data/raw/dreamer/
└── DREAMER.mat
```

#### Dataset 3: Tardis BTC (tự động tải)

Dữ liệu BTC futures được tải tự động bởi Script 00 từ Binance Vision (miễn phí, không cần API key):

```
data/raw/tardis/                   ← Script 00 tự tạo và populate
```

> **Tùy chọn:** Nếu muốn dùng Tardis.dev API (mất phí, dữ liệu đầy đủ hơn):
> ```powershell
> $env:TARDIS_API_KEY = "your_key_here"
> python scripts/phase1_data_engineering/00_fetch_tardis.py --source tardis
> ```

### 3.4 Kiểm tra setup

Sau khi chuẩn bị xong, verify:

```powershell
conda activate stress

# Kiểm tra Python + PyTorch + GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}')"

# Kiểm tra datasets đã có
python -c "from config.settings import WESAD_RAW_DIR; print('WESAD:', WESAD_RAW_DIR.exists())"
python -c "from config.settings import RAW_DIR; print('DREAMER:', (RAW_DIR / 'dreamer' / 'DREAMER.mat').exists())"

# Kiểm tra dependencies
python -c "import numpy, pandas, scipy, sklearn, matplotlib; print('All core deps OK')"
```

Kết quả mong đợi:
```
PyTorch 2.5.1+cu121, CUDA=True, GPU=NVIDIA GeForce RTX 3050 Laptop GPU
WESAD: True
DREAMER: True
All core deps OK
```

---

## 4. Cấu trúc workspace

```
NCKH/
├── README.md                                              ← File này
├── Algorithmic Panic  Full Analysis & Dataset Audit.md    ← Tài liệu nghiên cứu gốc (7 stages)
├── requirements.txt                                       ← Python dependencies
│
├── config/
│   └── settings.py              # Cấu hình tập trung (paths, constants, thresholds)
│
├── src/                         # Source package
│   ├── data/                    #   Data loaders (WESAD .pkl, DREAMER .mat, Tardis API)
│   ├── audit/                   #   Dataset audit checks (W1-W12, D1-D12, T1-T15, CA1-CA6)
│   ├── preprocessing/           #   Signal processing (ECG, EDA, EEG, market bars)
│   ├── features/                #   Feature extraction (HRV, EDA, DE, market)
│   ├── validation/              #   ML validation (LOSOCV, GRL, shortcuts, scaling)
│   ├── analysis/                #   Stylized facts analysis
│   └── utils/                   #   IO helpers, plotting
│
├── scripts/                     # 25 runner scripts, chia theo phase
│   ├── phase1_data_engineering/ #   Scripts 00-09: audit, preprocess, features, alignment
│   ├── phase2_validation/       #   Scripts 10-15: baselines, shortcuts, GRL, minimal model
│   ├── phase3_deep_models/      #   Scripts 16-18: DREAMER recovery, WESAD CNN, post-validation
│   ├── phase3_improvements/     #   Scripts 19-22: threshold, connectivity, R-R, label ceiling
│   ├── phase4_representation/   #   Scripts 23-25: transfer, OU process, final validation
│   ├── stage2_economics/        #   (chưa triển khai — placeholder cho ABM)
│   ├── README.md                #   ← Cấu trúc scripts + thứ tự chạy chi tiết
│   └── RUN_CHECKLIST.md         #   ← Checklist chạy Phase 3+ (có troubleshooting)
│
├── data/
│   ├── raw/                     # ⚠ CẦN TỰ CHUẨN BỊ (xem §3.3)
│   │   ├── wesad/               #   WESAD_extracted/WESAD/S2..S17/*.pkl
│   │   ├── dreamer/             #   DREAMER.mat
│   │   └── tardis/              #   Tự tải bởi Script 00
│   ├── interim/                 # Dữ liệu trung gian
│   └── processed/               # Output đã xử lý
│       ├── wesad/               #   15 × .npz (17,367 windows)
│       ├── dreamer/             #   23 × .npz (85,744 windows)
│       └── tardis/              #   Daily parquet + flash_crashes.csv + stylized_facts.json
│
├── reports/                     # Kết quả phân tích
│   ├── BIO_STAGE_CLOSING.md     #   ← BÁO CÁO TỔNG KẾT BIO STAGE
│   ├── PROGRESS.md              #   ← Log tiến trình chi tiết (1600+ dòng)
│   ├── audit/                   #   Audit reports (CSV)
│   ├── alignment/               #   Cross-dataset alignment (JSON)
│   └── validation/              #   21 JSON + 2 MD validation reports
│
└── notebooks/                   # (trống — dành cho exploration)
```

---

## 5. Hướng dẫn tái tạo kết quả (Reproduce)

### 5.1 Phase 1 — Data Engineering (Scripts 00–09)

Audit datasets → preprocess signals → extract features → validate alignment + stylized facts.

```powershell
conda activate stress

# Tải BTC data (Binance Vision, miễn phí)
python scripts/phase1_data_engineering/00_fetch_tardis.py

# Audit cả 3 datasets
python scripts/phase1_data_engineering/01_audit_wesad.py
python scripts/phase1_data_engineering/02_audit_dreamer.py
python scripts/phase1_data_engineering/03_audit_tardis.py

# Preprocess
python scripts/phase1_data_engineering/04_preprocess_wesad.py
python scripts/phase1_data_engineering/05_preprocess_dreamer.py
python scripts/phase1_data_engineering/06_preprocess_tardis.py

# Features, alignment, stylized facts
python scripts/phase1_data_engineering/07_extract_features.py
python scripts/phase1_data_engineering/08_alignment_check.py
python scripts/phase1_data_engineering/09_stylized_facts.py
```

**Output:** `data/processed/` (3 datasets) + `reports/audit/` + `reports/alignment/`

### 5.2 Phase 2 — Scientific Validation (Scripts 10–15)

Prove signal → detect shortcuts → adversarial test → minimal model → validity reports.

```powershell
python scripts/phase2_validation/10_learnability_baselines.py
python scripts/phase2_validation/11_subject_classifier_probe.py
python scripts/phase2_validation/12_adversarial_grl.py          # Cần PyTorch + GPU
python scripts/phase2_validation/13_minimal_model.py
python scripts/phase2_validation/14_dreamer_ica_check.py
python scripts/phase2_validation/15_generate_validity_report.py
```

**Output:** `reports/validation/baseline_results_*.json`, `adversarial_results_*.json`, `model_validity_report_*.md`

### 5.3 Phase 3 — Deep Model Exploration (Scripts 16–18)

DREAMER recovery (z-norm + valence) → WESAD raw ECG CNN → post-recovery validation.

```powershell
python scripts/phase3_deep_models/16_dreamer_recovery.py           # ~15-30 min
python scripts/phase3_deep_models/17_wesad_deep_model.py           # ~30-60 min, GPU
python scripts/phase3_deep_models/18_dreamer_post_recovery_validation.py  # ~30-60 min
```

**Output:** `reports/validation/dreamer_recovery_results.json`, `deep_model_results_wesad.json`

> **Troubleshooting Phase 3+:** Xem [`scripts/RUN_CHECKLIST.md`](scripts/RUN_CHECKLIST.md) — có hướng dẫn xử lý CUDA OOM, decision tree cho kết quả, và checklist từng bước.

### 5.4 Phase 3+ — Advisor Hypotheses (Scripts 19–22)

Threshold optimization → DREAMER connectivity → R-R interval model → label noise ceiling.

```powershell
python scripts/phase3_improvements/19_cnn_threshold_optimization.py   # ~20 min, GPU
python scripts/phase3_improvements/20_dreamer_connectivity.py          # ~15 min
python scripts/phase3_improvements/21_rr_interval_model.py             # ~30 min, GPU
python scripts/phase3_improvements/22_dreamer_label_noise_ceiling.py   # ~5 min
```

**Output:** `reports/validation/threshold_optimization_results.json`, `rr_interval_results.json`, `dreamer_label_noise_ceiling.json`

### 5.5 Phase 4 — Stochastic Law Discovery (Scripts 23–25)

Cross-dataset representation transfer → OU process identification → 4 robustness tests.

```powershell
python scripts/phase4_representation/23_representation_transfer.py       # ~20 min, GPU
python scripts/phase4_representation/24_stress_process_identification.py # ~5 min
python scripts/phase4_representation/25_final_validation.py              # ~10 min
```

**Output:** `reports/validation/stress_process_identification.json`, `final_validation.json`

> **Thứ tự chạy là TUYỆT ĐỐI — mỗi phase phụ thuộc vào output của phase trước.** Cấu trúc scripts chi tiết hơn nằm trong [`scripts/README.md`](scripts/README.md).

---

## 6. Hệ thống tài liệu

Workspace có 4 tầng tài liệu, từ tổng quan đến chi tiết:

| Tài liệu | Đặc điểm | Đọc khi nào |
|-----------|----------|-------------|
| **README.md** (file này) | Setup, cấu trúc, hướng dẫn reproduce | Mới clone repo, muốn chạy lại |
| [**reports/BIO_STAGE_CLOSING.md**](reports/BIO_STAGE_CLOSING.md) | Báo cáo tổng kết Bio Stage: 3 datasets, 5 phases (25 scripts), 10 confirmed claims, 8 falsified claims, 8 emergent insights, ABM specification, toàn bộ kết quả số | Muốn hiểu **Bio Stage đã chứng minh được gì** và **Stage 2 cần gì** |
| [**scripts/README.md**](scripts/README.md) | Cấu trúc thư mục scripts, thứ tự thực thi, output locations | Muốn biết **chạy script nào, ở đâu** |
| [**scripts/RUN_CHECKLIST.md**](scripts/RUN_CHECKLIST.md) | Checklist pre-flight, decision tree sau khi chạy, troubleshooting (CUDA OOM, Unicode, etc.) | Đang **chạy Phase 3+** và gặp lỗi hoặc cần interpret kết quả |
| [**reports/PROGRESS.md**](reports/PROGRESS.md) | Log chi tiết 1600+ dòng: mỗi script làm gì, kết quả cụ thể, bugs đã fix | Muốn đọc **diary của quá trình phát triển** |
| [**reports/validation/model_validity_report_wesad.md**](reports/validation/model_validity_report_wesad.md) | Paper-ready validation report cho WESAD (5 sections) | Viết paper, cần **bảng/số liệu cho validation section** |
| [**reports/validation/model_validity_report_dreamer.md**](reports/validation/model_validity_report_dreamer.md) | Paper-ready validation report cho DREAMER (negative control) | Viết paper, cần **negative control evidence** |
| [**Algorithmic Panic Full Analysis & Dataset Audit.md**](Algorithmic%20Panic%20%20Full%20Analysis%20%26%20Dataset%20Audit.md) | Tài liệu nghiên cứu gốc: 7 stages, audit checklists, falsification suite | Muốn hiểu **toàn bộ research design** từ đầu |
| [**scripts/stage2_economics/README.md**](scripts/stage2_economics/README.md) | Inputs từ Bio Stage cho ABM, constraints, planned scripts | Bắt đầu **Stage 2** |

### Thứ tự đọc đề xuất cho người mới

```
1. README.md (file này)                    → Setup + tổng quan
2. Algorithmic Panic Full Analysis.md      → Hiểu research design
3. reports/BIO_STAGE_CLOSING.md            → Kết quả + insights + handoff
4. scripts/README.md                       → Cấu trúc scripts nếu muốn reproduce
```

---

## 7. Kết quả chính của Bio Stage

> Chi tiết đầy đủ: [`reports/BIO_STAGE_CLOSING.md`](reports/BIO_STAGE_CLOSING.md)

### Confirmed

- **Stress detection từ ECG-HRV là khả thi:** bal_acc = 0.763 (LogReg, 7 features), AUC = 0.913 (RRCNN1D, R-R intervals)
- **Tín hiệu là sinh lý thực sự**, không phải confound subject identity (GRL adversarial Δ = +0.014)
- **σ(t) tuân theo OU mean-reversion:** 15/15 subjects, tất cả window sizes, bias-corrected
- **Standard OU đủ**, không cần fractional extension (ΔBIC = -377)
- **BTC futures có đủ 5 Cont stylized facts** để làm calibration target cho ABM

### Falsified

- Deep learning KHÔNG vượt LogReg cho stress detection (tín hiệu quá đơn giản — 1 feature dominant)
- θ = 0.074 KHÔNG phải hằng số sinh lý (window size artifact, slope = 0.979)
- EEG connectivity KHÔNG transfer cross-subject với 14-channel consumer EEG
- DREAMER accuracy = 0.600 = label noise ceiling (không thể cải thiện bằng model/features)

### Handoff cho Stage 2

ABM specification đề xuất:

$$d\sigma_i = \theta_i(\mu_i - \sigma_i)dt + \eta_i dW_i$$

Với $\theta_i$ phải calibrate tại ABM time-step (không fix tại 0.074), ~13% agents là extreme responders. Xem [`reports/BIO_STAGE_CLOSING.md § IV.3`](reports/BIO_STAGE_CLOSING.md) cho specification đầy đủ.

---

*Cập nhật lần cuối: 2026-02-20*
