# Algorithmic Panic — Bio Stage Visualization Guide

> **Date**: 2026-03-05  
> **Task**: Generate all publication-quality plots for the Bio Stage using R/RStudio

---

## Overview

Dự án này tạo **18 plots** (PDF + PNG) bao gồm:
- **4 plots Signal Pipeline**: Hiển thị signal trước/sau preprocessing cho WESAD và DREAMER
- **14 plots Contribution**: Visualize tất cả các contribution, falsified hypotheses, và emergent findings

---

## Cấu Trúc Thư Mục

```
r_visualization/
├── AlgorithmicPanic_Viz.Rproj    ← Mở file này trong RStudio
├── run_all.R                      ← File duy nhất cần chạy
├── 00_install_packages.R          ← Cài đặt packages (tự động)
├── 01_export_data_from_python.py  ← Export .npz → .csv (tự động)
├── 02_signal_pipeline_plots.R     ← Plot signal pipeline
├── 03_contribution_plots.R        ← Plot contributions
├── data_export/                   ← CSV files exported từ Python
│   └── (auto-generated)
└── output/                        ← Tất cả plots output
    ├── *.pdf                      ← Vector format (cho paper)
    └── *.png                      ← Raster format (cho slides)
```

---

## Hướng Dẫn Chạy (3 Bước)

### Bước 1: Chuẩn Bị

1. Mở **RStudio**
2. Mở project: `File → Open Project → AlgorithmicPanic_Viz.Rproj`
3. Đảm bảo Python có trong PATH (để export data từ `.npz`)

### Bước 2: Chạy

Mở file `run_all.R` và nhấn **Ctrl+Shift+S** (Source) hoặc:

```r
source("run_all.R")
```

Script sẽ tự động:
1. Cài đặt tất cả packages cần thiết
2. Chạy Python export (nếu data chưa có)
3. Generate tất cả signal pipeline plots
4. Generate tất cả contribution plots

### Bước 3: Xem Kết Quả

Tất cả plots nằm trong thư mục `output/`. Mỗi plot có 2 format:
- **PDF**: Dùng cho paper/publication (vector, scale tốt)
- **PNG**: Dùng cho slides/presentations (300 DPI)

---

## Danh Sách Plots

### Signal Pipeline (Task 1) — Data trước/sau preprocessing

| # | File | Nội Dung | Dataset |
|---|------|----------|---------|
| 01 | `01_wesad_ecg_pipeline` | Raw ECG → Bandpass Filter → R-peaks → HRV Features | WESAD |
| 02 | `02_wesad_eda_pipeline` | Raw EDA → Lowpass Filter → EDA Features | WESAD |
| 03 | `03_dreamer_eeg_pipeline` | Raw EEG → Bandpass+Notch → Baseline Subtraction | DREAMER |
| 04 | `04_dreamer_de_heatmap` | Differential Entropy per channel × frequency band | DREAMER |

### Contribution Plots (Task 2) — Chứng minh các contribution

| # | File | Contribution | Nội Dung |
|---|------|-------------|----------|
| 05 | `05_C1_stress_detectability` | **C1**: Stress is detectable | Ridge plot distributions + Cohen's d effect sizes |
| 06 | `06_C2_adversarial_robustness` | **C2**: Signal is physiological | GRL performance delta + Subject probe accuracy |
| 07 | `07_C3_model_comparison` | **C3**: Cardiac timing dominance | Bar chart: LogReg(0.763) > CNN(0.686) |
| 08 | `08_C4_ablation_hierarchy` | **C4**: HRV feature hierarchy | Drop-one-out ablation: hr_mean carries 80% info |
| 09 | `09_C5C6_ou_process` | **C5+C6**: OU process discovery | Simulated OU paths + θ window-size dependency |
| 10 | `10_C7C8_negative_control` | **C7+C8**: DREAMER as negative control | WESAD(0.763) vs DREAMER(0.541) vs ceiling(0.600) |
| 11 | `11_C9_stylized_facts` | **C9**: BTC market validation | 5/5 stylized facts summary card |
| 12 | `12_C10_representation_transfer` | **C10**: Domain-specific representations | CKA ≈ 0 + transfer accuracy at chance |
| 13 | `13_F1_deep_vs_baseline` | **F1** (Falsified): DL < baseline | All model strategies vs LogReg baseline |
| 14 | `14_subject_variability` | **E1+E2**: Individual differences | Lollipop chart per-subject balanced accuracy |
| 15 | `15_dreamer_recovery` | Recovery strategies | Z-norm strategies + noise ceiling |
| 16 | `16_wesad_correlation_matrix` | Feature relationships | Pearson correlation heatmap (7×7) |
| 17 | `17_results_dashboard` | **Summary**: All results | 4-panel dashboard of key metrics |
| 18 | `18_data_audit_summary` | Data quality | Clean rates + sample counts per dataset |

---

## Tóm Tắt Contributions

### 10 Confirmed Contributions (C1–C10)

| # | Contribution | Bằng Chứng | Plot |
|---|---|---|---|
| C1 | Stress detectable từ cardiac HRV | Bal. Acc = 0.763, Cohen's d = 1.554 | #05 |
| C2 | Signal là physiological, không phải subject identity | GRL Δ = +0.014, subject probe giảm | #06 |
| C3 | Cardiac TIMING dominance | R-R CNN AUC 0.913 >> Raw ECG CNN 0.828 | #07 |
| C4 | hr_mean chứa >80% information | Ablation drop = −0.184 | #08 |
| C5 | Stress follows Ornstein-Uhlenbeck | 15/15 subjects, bias-corrected slope ≈ 1.0 | #09 |
| C6 | Standard OU sufficient (không cần fractional) | ΔBIC = −377 | #09 |
| C7 | DREAMER validates pipeline limits | WESAD 0.763 vs DREAMER 0.541 cùng pipeline | #10 |
| C8 | DREAMER at label noise ceiling | Achieved 0.600 = ceiling 0.600 | #10 |
| C9 | BTC exhibits 5 stylized facts | Kurtosis 330.8, ACF 0.40, Hurst ≈ 0.5 | #11 |
| C10 | Representation domain-specific | CKA ≈ 0, separability 22.7× | #12 |

### 6 Falsified Hypotheses (F1–F6)

| # | Hypothesis | Reality | Plot |
|---|---|---|---|
| F1 | Deep learning > handcrafted | LogReg 0.763 > CNN 0.686 | #13 |
| F2 | EEG connectivity rescues DREAMER | Connectivity 0.506 ≈ chance | #10 |
| F3 | θ là physiological constant | θ phụ thuộc window size | #09 |
| F4 | CNN threshold transfers | Oracle 0.776, LOO 0.707 | #13 |
| F5 | Arousal > Valence for stress | Valence 0.600 >> Arousal 0.505 | #15 |
| F6 | Longer windows improve HRV | 30-beat 0.745 ≈ 5s 0.763 | #07 |

---

## Signal Shapes: Trước vs Sau Preprocessing

### WESAD (Chest ECG + EDA, 700 Hz)

| Giai Đoạn | Signal | Đặc Điểm |
|-----------|--------|-----------|
| **Raw** | ECG 700 Hz | Baseline wander 0-2 mV, QRS 1-5 mV, nhiễu 50 Hz, motion artifacts |
| **Bandpass (0.5-40 Hz)** | ECG filtered | Baseline corrected, QRS isolated, high-freq noise removed |
| **Pan-Tompkins** | R-peak indices | Detected peaks tại mỗi heartbeat, min distance 250 ms |
| **R-R extraction** | Intervals (ms) | Series of 250-2000 ms intervals, outliers rejected |
| **HRV features** | 4 values/window | hr_mean, hr_std, rmssd, sdnn per 5s window |
| **EDA raw** | 700 Hz | High-frequency noise, motion artifacts |
| **EDA filtered (5 Hz LP)** | Smooth SCL | Skin conductance level, slow trends |
| **EDA features** | 3 values/window | eda_mean, eda_std, eda_slope per 5s window |

### DREAMER (14-channel EEG, 128 Hz)

| Giai Đoạn | Signal | Đặc Điểm |
|-----------|--------|-----------|
| **Raw** | 14ch × 128 Hz | ±100 µV, 50 Hz mains, eye blinks ~150 µV, DC drift |
| **Bandpass (0.1-40 Hz)** | Filtered EEG | DC drift removed, high-freq artifacts eliminated |
| **Notch (48-52 Hz)** | Clean EEG | Powerline 50 Hz interference removed |
| **Baseline subtraction** | Corrected EEG | Subject-specific offset eliminated (61s baseline) |
| **Differential Entropy** | 70 features | 14 channels × 5 bands (delta/theta/alpha/beta/gamma) |

---

## Requirements

### R Packages (tự động cài trong `00_install_packages.R`)

```
tidyverse, jsonlite, data.table, patchwork, scales, ggridges, ggrepel,
RColorBrewer, viridis, ggbeeswarm, corrplot, pheatmap, signal, seewave,
effectsize, broom, Cairo, svglite
```

### Python (cho bước export data)

```
numpy, pandas, scipy
```

### System

- **RStudio** ≥ 2023.06 (recommended)
- **R** ≥ 4.2.0
- **Python** ≥ 3.8 (accessible from PATH)

---

## Troubleshooting

| Vấn Đề | Giải Pháp |
|---------|-----------|
| "Python not found" | Đảm bảo Python có trong PATH, hoặc chạy `01_export_data_from_python.py` thủ công |
| "No raw WESAD data" | Cần `data/raw/wesad/WESAD_extracted/` (giải nén WESAD.zip trước) |
| "No DREAMER.mat" | Cần `data/raw/dreamer/DREAMER.mat` |
| Package install fails | Chạy `install.packages("tên_package")` thủ công trong R console |
| Signal plots empty | Chạy Python export trước: `python r_visualization/01_export_data_from_python.py` |
| Contribution plots work nhưng signal plots không | Signal plots cần raw data; contribution plots chỉ cần CSV + JSON |

> **Lưu ý**: Nếu raw data chưa extract, bạn vẫn có thể chạy contribution plots (05–18). Chỉ signal pipeline plots (01–04) cần raw data.

---

## Ghi Chú Bổ Sung

- Tất cả plots sử dụng **theme_minimal** với publication-quality formatting
- Mỗi plot có **title, subtitle, axis labels** đầy đủ
- Color scheme nhất quán: **Blue = Non-stress**, **Red = Stress**, **Green = Good/Pass**, **Orange = Warning**
- PDF output dùng vector graphics (zoom không bị vỡ)
- PNG output ở 300 DPI (đủ cho print)
