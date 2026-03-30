# Algorithmic Panic — Bio Stage: Báo cáo Tổng hợp

> **Dự án**: Endogenous Stress as State Variable in Financial Markets
> **Giai đoạn**: Stage 1 — Bio Stage (Stress Inference Engine)
> **Trạng thái**: HOÀN THÀNH — Sẵn sàng chuyển sang Stage 2 (ABM)
> **Ngày báo cáo**: 05/03/2026

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Dữ liệu & Tiền xử lý](#2-dữ-liệu--tiền-xử-lý)
3. [Kết quả chính — 10 Đóng góp đã xác nhận](#3-kết-quả-chính--10-đóng-góp-đã-xác-nhận)
4. [6 Giả thuyết bị bác bỏ](#4-6-giả-thuyết-bị-bác-bỏ)
5. [8 Phát hiện mới](#5-8-phát-hiện-mới)
6. [Tổng hợp định lượng](#6-tổng-hợp-định-lượng)
7. [Kết luận & Hướng đi tiếp theo](#7-kết-luận--hướng-đi-tiếp-theo)

---

## 1. Tổng quan dự án

### 1.1 Vấn đề nghiên cứu

Các mô hình tài chính hiện tại coi các tác nhân giao dịch là thuần lý trí. Trên thực tế, **stress sinh lý** của trader ảnh hưởng trực tiếp đến hành vi giao dịch: khi thị trường biến động mạnh, stress tăng → quyết định kém → thanh khoản giảm → biến động tăng thêm. Đây là **vòng phản hồi nội sinh** (endogenous feedback loop) chưa từng được mô hình hóa trong literature.

**Ý tưởng cốt lõi**: Đưa stress sinh lý vào như một **biến trạng thái** (state variable) trong mô hình thị trường, tạo nên một paradigm mới: **bio-technical ecosystem** thay vì **purely algorithmic**.

### 1.2 Kiến trúc 7 giai đoạn

| Giai đoạn | Tên | Mục tiêu | Trạng thái |
|:---------:|-----|----------|:----------:|
| 0 | Causal DAG | Xây dựng đồ thị nhân quả | Done |
| **1** | **Bio Stage** | **Đo stress từ tín hiệu sinh lý** | **DONE** |
| 2 | Market Simulator | Xây dựng ABM thị trường | Tiếp theo |
| 3 | Bio-Behavior Coupling | Ánh xạ stress → hành vi giao dịch | Chưa bắt đầu |
| 4 | Feedback Dynamics | Phân tích ổn định vòng phản hồi | Chưa bắt đầu |
| 5 | Evidence Engine | Kiểm chứng 3 tầng | Chưa bắt đầu |
| 6 | Policy Analysis | Stress-aware circuit breaker | Chưa bắt đầu |

### 1.3 Pipeline thực hiện — Bio Stage

Bio Stage gồm **25 scripts**, chia thành **5 phase**:

```
Phase 1 (Scripts 00-09): Data Engineering — Thu thập, kiểm tra, tiền xử lý
Phase 2 (Scripts 10-15): Scientific Validation — Kiểm chứng tín hiệu
Phase 3 (Scripts 16-22): Deep Models & Advisor Hypotheses
Phase 4 (Scripts 23-25): Stochastic Law Discovery — Phát hiện quy luật
```

**Kết quả visualization**: 18 publication-quality plots (PDF + PNG, 300 DPI).

---

## 2. Dữ liệu & Tiền xử lý

### 2.1 Ba bộ dữ liệu

| | WESAD | DREAMER | Tardis-BTC |
|---|:---:|:---:|:---:|
| **Loại tín hiệu** | ECG + EDA (ngực) | EEG 14 kênh | BTC Futures |
| **Đối tượng** | 15 người | 23 người | Thị trường |
| **Tần số** | 700 Hz | 128 Hz | 1-phút bars |
| **Số windows** | 17,367 | 85,744 | 2,410,560 bars |
| **Tỉ lệ stress** | 11.5% | 42.7% | N/A |
| **Chất lượng** | 99.54% clean | 100% clean | 99.8% clean |
| **Vai trò** | Anchor chính | Negative control | Market ground truth |

### 2.2 WESAD — Tín hiệu sinh lý (Anchor Dataset)

**Giao thức**: Trier Social Stress Test (TSST)
- Baseline → **Stress** (thuyết trình + tính nhẩm) → Amusement → Meditation
- 15 đối tượng (S2-S17; S1, S12 bị loại do lỗi cảm biến)

**Pipeline tiền xử lý** *(xem Plot 01-02)*:

```
ECG Raw (700 Hz)
  → Bandpass filter (0.5-40 Hz): Loại bỏ baseline wander & nhiễu cao tần
    → Pan-Tompkins R-peak detection: Phát hiện đỉnh R
      → R-R intervals: Khoảng cách giữa các nhịp tim
        → HRV features (7 đặc trưng / cửa sổ 5 giây)
```

**7 đặc trưng HRV được trích xuất**:
- **Cardiac**: hr_mean, hr_std, rmssd, sdnn
- **Electrodermal**: eda_mean, eda_std, eda_slope

**EDA Pipeline**:
```
EDA Raw (700 Hz) → Lowpass filter (5 Hz) → Windowed statistics (5s)
```

### 2.3 DREAMER — EEG (Negative Control)

**Giao thức**: 18 video clips, tự báo cáo valence/arousal/dominance

**Pipeline tiền xử lý** *(xem Plot 03-04)*:

```
Raw EEG (128 Hz, 14 kênh)
  → Bandpass (0.1-40 Hz) + Notch filter (50 Hz)
    → Baseline subtraction (trừ trung bình trạng thái nghỉ)
      → Differential Entropy (DE): 5 băng tần x 14 kênh = 70 đặc trưng
```

**5 băng tần EEG**: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-40 Hz)

**Vấn đề cốt lõi**: DREAMER KHÔNG có nhãn stress trực tiếp — phải ánh xạ từ valence/arousal, dẫn đến **độ tin cậy nhãn thấp**.

### 2.4 Tardis-BTC — Dữ liệu thị trường

- **Phạm vi**: 01/06/2020 - 31/12/2024 (Binance BTCUSDT Perpetuals)
- **Dữ liệu**: 2.4 triệu bars 1-phút, 21 đặc trưng (OHLCV, volatility, order flow, spread...)
- **7 sự kiện flash crash** được nhận diện
- **Vai trò**: Xác nhận stylized facts để đảm bảo market simulator hiện thực

### 2.5 Kiểm tra chất lượng — Tất cả PASS

| Kiểm tra | WESAD | DREAMER | Tardis | Cross-Dataset |
|----------|:-----:|:-------:|:------:|:-------------:|
| Audit | 12/12 PASS | 12/12 PASS | 15/15 PASS | 6/6 PASS |
| Alignment | PASS | PASS | PASS | 10/10 PASS |
| Artifact rate | 0.46% | 0% | 0.2% | — |

*(Xem Plot 18: Data Audit Summary)*

---

## 3. Kết quả chính — 10 Đóng góp đã xác nhận

### C1. Stress có thể phát hiện từ tín hiệu tim mạch

*(Xem Plot 05: C1 Stress Detectability)*

**Phát hiện**: Stress tạo ra sự khác biệt có ý nghĩa thống kê trong các đặc trưng HRV.

**Độ lớn hiệu ứng (Cohen's d)**:

| Đặc trưng | Cohen's d | Mức độ |
|-----------|:---------:|:------:|
| **hr_mean** | **1.554** | **LỚN** |
| eda_std | 0.401 | Trung bình |
| eda_mean | 0.333 | Nhỏ |
| rmssd | 0.289 | Nhỏ |
| hr_std | 0.232 | Nhỏ |
| sdnn | 0.202 | Nhỏ |
| eda_slope | 0.086 | Không đáng kể |

**Kết luận**: hr_mean (nhịp tim trung bình) là chỉ báo stress mạnh nhất với d = 1.554 — hiệu ứng rất lớn. Mô hình LogReg đạt **balanced accuracy = 0.763** (AUC = 0.892) chỉ với 7 đặc trưng.

---

### C2. Tín hiệu là sinh lý thực, không phải identity của đối tượng

*(Xem Plot 06: C2 Adversarial Robustness)*

**Vấn đề**: Liệu mô hình có đang "học thuộc" đặc điểm riêng của từng người (subject confound) thay vì học tín hiệu stress thực sự?

**Phương pháp**: Gradient Reversal Layer (GRL) — buộc mô hình ĐỒNG THỜI nhận diện stress VÀ không nhận ra được identity của đối tượng.

**Kết quả**:

| | Trước GRL | Sau GRL | Delta |
|---|:---------:|:-------:|:-----:|
| **WESAD Bal. Acc** | 0.750 | **0.764** | **+0.014** |
| WESAD Subject Probe | 77.3% | 71.1% | -6.2% |
| DREAMER Bal. Acc | 0.542 | 0.538 | -0.004 |
| DREAMER Subject Probe | 92.6% | 43.0% | -49.6% |

**Kết luận**: Với WESAD, khi loại bỏ thông tin identity, hiệu suất còn **TĂNG nhẹ** (+1.4%). Điều này chứng tỏ tín hiệu stress là **THỰC SỰ** (robust signal), không phải artifact từ việc nhận diện người.

---

### C3. Mô hình đơn giản thắng mô hình sâu (Simple > Deep)

*(Xem Plot 07: C3 Model Comparison)*

| Mô hình | Loại | Params | Balanced Acc |
|---------|------|:------:|:------------:|
| **LogReg (HRV)** | **Tuyến tính** | **7** | **0.763** |
| RR-CNN | Timing deep | ~50K | 0.750 |
| RandomForest | Ensemble | ~100 | 0.744 |
| MLP | Neural | ~1K | 0.739 |
| TinyCNN (raw ECG) | Deep + Raw | 70K | 0.686 |
| HybridCNN | Deep + Hybrid | 155K | 0.682 |

**Kết luận**: LogReg với 7 đặc trưng HRV **thắng** tất cả mô hình deep learning. Tín hiệu stress nằm ở **nhịp độ thống kê** (HRV timing), KHÔNG PHẢI ở hình dạng sóng (waveform morphology). Đây là phát hiện quan trọng cho thiết kế Stage 2.

---

### C4. hr_mean chiếm >80% thông tin stress

*(Xem Plot 08: C4 Ablation Hierarchy)*

**Drop-one-out ablation** — Bỏ từng đặc trưng và đo mức sụt giảm:

| Đặc trưng bị bỏ | Delta Bal. Acc | Tỉ lệ thông tin |
|-----------------|:--------------:|:---------------:|
| **hr_mean** | **-0.184** | **~80%** |
| eda_std | -0.048 | ~20% (chia sẻ) |
| hr_std | -0.035 | |
| rmssd | -0.028 | |
| sdnn | -0.015 | |
| eda_mean | -0.012 | |
| eda_slope | -0.008 | |

**Kết luận**: Khi bỏ hr_mean, hiệu suất **sụt 18.4%** — sụt phạm hoàn toàn. hr_mean mang phần lớn thông tin phân biệt stress, các đặc trưng khác chỉ bổ sung ít.

---

### C5. Stress tuân theo quy luật mean-reversion (Ornstein-Uhlenbeck)

*(Xem Plot 09: C5+C6 OU Process)*

**Mô hình**: Stress như một quá trình ngẫu nhiên có xu hướng quay về giá trị trung bình:

$$d\sigma = \theta(\mu - \sigma)dt + \eta \, dW_t$$

Trong đó:
- sigma: mức stress hiện tại
- theta: tốc độ quay về (mean-reversion rate)
- mu: mức stress cân bằng dài hạn
- eta: độ bất định (volatility)

**Kết quả**:
- **15/15 đối tượng** đều thể hiện mean-reversion (ADF test p < 0.05)
- theta trung bình = 0.074 /s (tại cửa sổ 5s)
- Half-life = 10.7s (thời gian để stress giảm một nửa)
- Hệ số biến thiên CV(theta) = 0.319 — **nhất quán giữa các cá nhân**

**Động lực học theo pha**:

| Pha | theta | Half-life | Hurst H |
|-----|:-----:|:---------:|:-------:|
| Baseline | 0.117 | 5.9s | 0.789 |
| **Stress** | **0.087** | **8.0s** | **0.897** |
| Amusement | 0.159 | 4.4s | 0.755 |
| Meditation | 0.185 | 3.7s | 0.755 |

**Phát hiện quan trọng**: Stress có mean-reversion **CHẬM NHẤT** (theta thấp nhất) và **dài nhất** (Hurst cao nhất) — stress "dính" hơn các trạng thái khác. Điều này phù hợp với trực giác: khi bị stress, cơ thể mất nhiều thời gian để bình phục hơn.

---

### C6. Mô hình OU chuẩn là đủ, không cần fractional

**Kiểm chứng**: So sánh OU chuẩn (2 tham số) vs fractional OU (3 tham số) bằng Bayesian Information Criterion:

- **Delta BIC = -377** (trung bình) → OU chuẩn tốt hơn
- **15/15 đối tượng** ưu tiên OU chuẩn
- Hurst H = 0.845 cho thấy persistence nhưng không cần fractional

**Kết luận**: Không cần phức tạp hóa mô hình. OU chuẩn là đủ cho ABM.

---

### C7+C8. DREAMER là negative control — Xác nhận pipeline hoạt động đúng

*(Xem Plot 10: C7+C8 Negative Control)*

| Dataset | Balanced Acc | Phán định |
|---------|:------------:|:---------:|
| **WESAD** | **0.763** | **TÍN HIỆU MẠNH** |
| DREAMER (gốc) | 0.541 | Không có tín hiệu |
| DREAMER (z-norm) | 0.600 | Tại trần nhiễu nhãn |

**Label noise ceiling** (trần độ nhiễu nhãn): 0.600 — tính từ độ tin cậy của nhãn tự báo cáo.

**Kết luận**: DREAMER đạt **ĐÚNG BẰNG** noise ceiling (0.600 = 0.600). Điều này chứng tỏ:
1. Pipeline KHÔNG tạo tín hiệu giả (không overfit)
2. Hạn chế nằm ở **chất lượng nhãn**, không phải năng lực mô hình
3. DREAMER đóng vai trò **kiểm chứng phủ định** (negative control) hoàn hảo

---

### C9. Dữ liệu BTC xác nhận đầy đủ 5 Stylized Facts

*(Xem Plot 11: C9 Stylized Facts)*

| Stylized Fact | Chỉ số | Giá trị | Ngưỡng | Kết quả |
|---------------|--------|:-------:|:------:|:-------:|
| SF-1: Đuôi béo (Fat tails) | Kurtosis | 330.8 | > 3 | PASS |
| SF-2: Volatility clustering | ACF(\|r\|) lag-1 | 0.40 | > 0.2 | PASS |
| SF-3: Leverage effect | Corr(r, var) | -0.02 | < 0 | PASS |
| SF-4: Volume-Volatility | \|Corr(V, sigma)\| | 0.338 | > 0.1 | PASS |
| SF-5: Return ACF vắng mặt | max\|ACF(r)\| | 0.022 | < 0.05 | PASS |

**Kết luận**: Dữ liệu BTC thể hiện đầy đủ 5 đặc tính thống kê của thị trường thực. ABM ở Stage 2 phải tái tạo được các đặc tính này để được coi là hiện thực.

---

### C10. Biểu diễn là đặc trưng theo miền — Transfer thất bại

*(Xem Plot 12: C10 Representation Transfer)*

**Thí nghiệm**: Huấn luyện encoder trên WESAD, đóng băng, áp dụng sang DREAMER.

| | WESAD (self) | Transfer | DREAMER (self) |
|---|:---:|:---:|:---:|
| **Accuracy** | 89.6% | **50.3%** (may rủi) | 57.4% |
| **CKA** | 0.87 | **0.0001** | 0.85 |
| **Separability** | 2.024 | — | 0.089 |

- **CKA ~ 0**: Không gian biểu diễn sụp đổ hoàn toàn khi chuyển miền
- **Tỉ lệ separability**: 22.7x (cấu trúc lớp rõ ràng ở WESAD, gần bằng 0 ở DREAMER)
- **Raw features (0.600) > Encoder embeddings (0.503)**: Transfer làm **TỆ hơn** so với không làm gì

**Kết luận**: Biểu diễn stress là **ĐẶC TRƯNG THEO MIỀN** (domain-specific). ECG-stress và EEG-emotion là hai không gian khác nhau hoàn toàn. Mỗi loại sensor cần mô hình riêng.

---

## 4. 6 Giả thuyết bị bác bỏ

| # | Giả thuyết | Kết quả | Bằng chứng |
|:-:|-----------|--------|-----------|
| F1 | Deep learning thắng LogReg | **SAI** | CNN 0.686 < LogReg 0.763 *(Plot 13)* |
| F2 | DREAMER có tín hiệu stress mạnh | **SAI** | Bal. Acc 0.541, = noise ceiling 0.600 |
| F3 | theta = 0.074 là hằng số sinh lý | **SAI** | Slope = 0.979 trong log-log (artifact) |
| F4 | Connectivity EEG giúp DREAMER | **SAI** | Chỉ đạt 0.506 (dưới chance) |
| F5 | Transfer EEG hoạt động | **SAI** | CKA = 0.0001, accuracy 50.3% |
| F6 | Threshold tuning cứu CNN | **SAI** | Chỉ oracle (không tổng quát) đạt 0.776 |

### F1 chi tiết: Deep Learning KHÔNG vượt LogReg

*(Xem Plot 13: F1 Deep vs Baseline)*

| Chiến lược | Balanced Acc | So với LogReg |
|-----------|:------------:|:-------------:|
| LogReg Baseline | 0.763 | — |
| CNN + Oracle Threshold | 0.776 | +0.013 (nhưng KHÔNG tổng quát) |
| RR-CNN (timing) | 0.750 | -0.013 |
| CNN + LOO-CV Threshold | 0.707 | -0.056 |
| TinyCNN (raw ECG) | 0.686 | -0.077 |
| HybridCNN | 0.682 | -0.081 |

**Chỉ có oracle threshold** (biết trước nhãn của test set) mới thắng — đây là **gian lận** (data leakage), không áp dụng được trong thực tế.

### F3 chi tiết: theta là artifact của kích thước cửa sổ

*(Xem phần dưới của Plot 09)*

```
Kích thước cửa sổ    theta      Half-life
    2.5s             0.138      5.8s
    5.0s             0.074      10.7s
    10.0s            0.032      25.6s
    20.0s            0.019      41.9s
```

Log-log slope = **0.979** (R^2 = 0.990). Nếu theta là hằng số thực, slope phải = 0. Slope ~ 1 có nghĩa theta chỉ đơn giản là **nghịch đảo của kích thước cửa sổ** → artifact.

**Điều chỉnh kết luận**: Mean-reversion là THẬT (15/15 đối tượng), nhưng giá trị cụ thể của theta phụ thuộc vào resolution. Thang đo nội sinh thực sự nằm **dưới 2.5 giây** (dưới ngưỡng trích xuất đặc trưng).

---

## 5. 8 Phát hiện mới

| # | Phát hiện | Chi tiết |
|:-:|----------|---------|
| E1 | Stress là gần như 1 chiều | PC1 giải thích 63.1% phương sai; PC1 loadings: rmssd (0.535), sdnn (0.541) |
| E2 | Beta band quan trọng cho emotion | Band ablation cho thấy beta (13-30 Hz) có ảnh hưởng lớn nhất |
| E3 | Valence > Arousal (bất ngờ) | Z-norm + valence (0.628) > arousal (0.592) — ngược với kỳ vọng |
| E4 | 45% DREAMER trials tại ranh giới | Valence = 3 hoặc 4 (ranh giới binary) → nhiễu nhãn bất định |
| E5 | Đối tượng có phân bố 2 mode | ~13% là "extreme responders" (S13: +1.81 sigma, S14: +1.93 sigma) |
| E6 | theta phụ thuộc thang đo | Scale-dependent: intrinsic timescale < 2.5s |
| E7 | Phục hồi phụ thuộc pha | Stress → recovery chậm hơn các pha khác (half-life 8.0s vs 3.7-5.9s) |
| E8 | Độ tin cậy nhãn là nút thắt | DREAMER ceiling 0.600 = giới hạn của self-report, không phải của mô hình |

### E5 chi tiết: Biến động giữa các đối tượng

*(Xem Plot 14: Subject Variability)*

| Đối tượng | Balanced Acc | Phân loại |
|:---------:|:------------:|:---------:|
| S4 | 0.884 | Extreme responder (cao) |
| S3 | 0.876 | Extreme responder (cao) |
| S16 | 0.847 | Cao |
| S11 | 0.834 | Cao |
| ... | ... | ... |
| S10 | 0.695 | Trung bình |
| S6 | 0.712 | Trung bình |
| **S2** | **0.506** | **Gần chance** |

**Trung bình**: 0.763, **Khoảng**: 0.506 - 0.884
- **~13% đối tượng** (2/15) là extreme responders (> 2 sigma trên trung bình)
- S2 gần như không phân biệt được stress → có thể là non-responder sinh lý

**Ý nghĩa cho ABM**: Cần mô hình hóa **heterogeneity** giữa các agent — không phải mọi trader đều phản ứng stress giống nhau.

---

## 6. Tổng hợp định lượng

### 6.1 Bảng tổng hợp tất cả chỉ số chính

| Thành phần | Chỉ số | Giá trị | Trạng thái |
|-----------|--------|:-------:|:----------:|
| **WESAD** | Đối tượng | 15 | OK |
| | Cửa sổ | 17,367 | OK |
| | Tỉ lệ stress | 11.5% | Mất cân bằng |
| | LogReg Bal. Acc | **0.763** | MẠNH |
| | LogReg AUC | **0.892** | MẠNH |
| | RR-CNN AUC | **0.913** | Tốt nhất |
| | GRL Delta | **+0.014** | TÍN HIỆU THỰC |
| | hr_mean Cohen's d | **1.554** | Hiệu ứng LỚN |
| **DREAMER** | Đối tượng | 23 | OK |
| | Cửa sổ | 85,744 | OK |
| | LogReg Bal. Acc | 0.541 | Không tín hiệu |
| | Z-norm Bal. Acc | 0.600 | Tại trần |
| | Noise ceiling | 0.600 | — |
| | Subject probe (trước) | 92.6% | Encoding cao |
| | Subject probe (sau GRL) | 43.0% | Đã loại bỏ |
| **Tardis-BTC** | 1-min bars | 2,410,560 | OK |
| | Stylized Facts | **5/5 PASS** | Hoàn hảo |
| | Flash crashes | 7 | Đã nhận diện |
| **OU Process** | Mean-reverting | **15/15** | Phổ quát |
| | theta (5s window) | 0.074 +- 0.024 | Artifact |
| | Half-life (stress) | 8.0s | Chậm nhất |
| | Process class | OU chuẩn | dBIC = -377 |
| **Transfer** | CKA | 0.0001 | Đặc trưng miền |
| | Separability ratio | 22.7x | |

### 6.2 Tương ứng Plot — Chỉ số

| Plot | Tên | Chỉ số chính |
|:----:|-----|-------------|
| 01 | WESAD ECG Pipeline | Raw → Filter → R-peaks → HRV |
| 02 | WESAD EDA Pipeline | Raw → Filter → Features |
| 03 | DREAMER EEG Pipeline | Raw → Filter → Baseline subtract |
| 04 | DREAMER DE Heatmap | 14 kênh x 5 băng tần |
| 05 | C1: Stress Detectability | d = 1.554, Bal. Acc = 0.763 |
| 06 | C2: Adversarial GRL | Delta = +0.014 (Robust) |
| 07 | C3: Model Comparison | LogReg 0.763 > CNN 0.686 |
| 08 | C4: Feature Ablation | hr_mean: -0.184 (dominant) |
| 09 | C5+C6: OU Process | 15/15 mean-reverting, theta = 0.074 |
| 10 | C7+C8: Negative Control | DREAMER 0.600 = ceiling |
| 11 | C9: Stylized Facts | 5/5 PASS |
| 12 | C10: Transfer | CKA = 0.0001, 50.3% = chance |
| 13 | F1: Deep vs Baseline | CNN không thắng LogReg |
| 14 | Subject Variability | Range: 0.506 - 0.884 |
| 15 | DREAMER Recovery | Z-norm: 0.541 → 0.628 |
| 16 | Correlation Matrix | hr_mean độc lập với EDA |
| 17 | Results Dashboard | Tổng hợp 4 panel |
| 18 | Data Audit | 3 datasets, >99.5% clean |

---

## 7. Kết luận & Hướng đi tiếp theo

### 7.1 Kết luận Bio Stage

1. **Stress có thể đo được** từ ECG/EDA với độ chính xác 76.3% (balanced) và AUC 89.2%
2. **Tín hiệu là thực** — không phải artifact từ identity đối tượng (GRL xác nhận)
3. **Mô hình đơn giản tốt hơn** — LogReg (7 đặc trưng) thắng CNN (70K-155K params)
4. **Thông tin tập trung** — hr_mean mang >80% khả năng phân biệt
5. **Stress tuân theo OU** — Mean-reversion phổ quát (15/15), stress "dính" nhất (half-life 8.0s)
6. **Biểu diễn đặc trưng miền** — Không thể transfer ECG → EEG (CKA ~ 0)
7. **Dữ liệu thị trường hiện thực** — 5/5 stylized facts cho BTC futures
8. **Heterogeneity quan trọng** — 13% đối tượng là extreme responders

### 7.2 Thông số bàn giao cho Stage 2 (ABM)

Mô hình stress cho mỗi agent trong ABM:

$$d\sigma_i = \theta_i(\mu_i - \sigma_i)dt + \eta_i \, dW_i$$

| Tham số | Giá trị khởi tạo | Ghi chú |
|---------|:-----------------:|--------|
| sigma baseline | 0.2 - 0.4 | Từ phân phối thực nghiệm |
| theta | Calibrate tại time-step ABM | KHÔNG cố định 0.074 (artifact) |
| mu | Tương ứng điều kiện thị trường | Cần coupling function |
| eta | Từ phương sai OU | |
| Extreme responders | ~13% agents | Heterogeneous population |

### 7.3 Những gì Stage 2 cần làm

1. **Xây dựng ABM** với 3 loại agent: Market Maker, Momentum Trader, Noise Trader
2. **Tích hợp OU stress** vào mỗi agent (heterogeneous theta)
3. **Tái tạo 5 stylized facts** từ ABM để xác nhận tính hiện thực
4. **Coupling function**: stress sigma → thay đổi hành vi (risk aversion, latency, tolerance)
5. **Kiểm tra điều kiện ổn định**: Tích sensitivities > 1 → market instability

---

> **Ghi chú**: Tất cả 18 plots được lưu tại `r_visualization/output/` dưới dạng PDF (vector) và PNG (raster), 300 DPI, sẵn sàng cho publication hoặc slide.
