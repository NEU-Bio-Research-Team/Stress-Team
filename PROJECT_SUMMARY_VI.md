# Stress-Team Project Summary — Tổng Quan Nghiên Cứu

> **Tên dự án:** Algorithmic Panic: Endogenous Stress as State Variable in Financial Markets  
> **Chủ đề:** Mối quan hệ nhân quả giữa stress sinh lý và hành vi thị trường tài chính  
> **Trạng thái:** Stage 1 (Bio Stage) ✅ HOÀN THÀNH | Sẵn sàng cho Stage 2  
> **Thời gian cập nhật:** 28/06/2025

---

## I. TỔNG QUAN DỰ ÁN

### 1.1 Câu Hỏi Nghiên Cứu Chính

**"Liệu stress sinh lý của các trader có ảnh hưởng đến hành vi thị trường tài chính không?"**

Dự án giả định rằng thị trường tài chính không phải là hệ thống puro lý thuyết mà là một **hệ thống ghép nối sinh-kỹ thuật** (bio-technical coupled system):

```
Volatility ↑ → Stress sinh lý ↑ → Tính thanh khoản ↓ → Volatility ↑ (vòng lặp hồi đáp)
```

Điều này tương phản với các mô hình ABM (Agent-Based Model) hiện tại, chỉ coi stress như một cú sốc ngoại sinh (exogenous shocks) từ tin tức bên ngoài.

### 1.2 Tính Đột Phá (Novelty)

| Khía cạnh | Mức độ | Chi tiết |
|-----------|--------|---------|
| **Conceptual** | ⭐⭐⭐⭐⭐ | Stress nội sinh trong ABM chưa ai làm trong literature |
| **Methodological** | ⭐⭐⭐⭐ | Hybrid Deep Learning + ABM coupling là mới lạ |
| **Policy** | ⭐⭐⭐⭐ | Circuit breaker tư duy stress = công cụ chính sách chưa được đề xuất |
| **Tổng thể** | **8/10** | Phụ thuộc vào validation nhân quả (causal) ở các giai đoạn sau |

---

## II. KIẾN TRÚC NGHIÊN CỨU (7 STAGES)

Dự án được thiết kế theo 7 stages, mỗi stage giải quyết một khía cạnh của vấn đề:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 0: Causal Model Construction (DAG Lý Thuyết)          │ ← Nền tảng
├─────────────────────────────────────────────────────────────┤
│ Stage 1: Stress Inference Engine — BIO STAGE ✅             │ ← ĐANG LÀM
│         (Phát hiện stress từ tín hiệu sinh lý)              │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Market Simulator — ABM                             │ ← Chuẩn bị dữ liệu
│         (Mô phỏng hành vi thị trường)                       │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: Bio → Behavior Coupling                            │ ← Chưa triển khai
│         (Nối stress → quyết định giao dịch)                 │
├─────────────────────────────────────────────────────────────┤
│ Stage 4: Feedback Dynamical System                          │ ← Chưa triển khai
│         (Phân tích hệ thống hồi đáp)                        │
├─────────────────────────────────────────────────────────────┤
│ Stage 5: Evidence Engine                                    │ ← Chưa triển khai
│         (Kiểm chứng bằng chứng thị trường)                  │
├─────────────────────────────────────────────────────────────┤
│ Stage 6: Policy Analysis                                    │ ← Chưa triển khai
│         (Phân tích tác động của chính sách)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## III. STAGE 1 — BIO STAGE (STRESS INFERENCE ENGINE)

### 3.1 Mục Tiêu Stage 1

Xây dựng một enine suy luận stress sinh lý từ các tín hiệu sinh lý (ECG, EDA, EEG):

- **Input:** Tín hiệu sinh lý thô (raw) từ các cảm biến
- **Output:** Dự báo độ stress liên tục σ(t) và độ tin cậy (confidence intervals)
- **Yêu cầu:** Balanced Accuracy > 85%, ECE (Expected Calibration Error) < 0.05

### 3.2 Ba Bộ Dữ Liệu Chính

#### 🏥 Dataset 1: WESAD (Wearable Stress and Affect Detection)

| Thuộc tính | Chi tiết |
|------------|----------|
| **Chủ đề** | 15 người (nam/nữ hỗn hợp) |
| **Ghi cảm biến** | RespiBAN trên ngực (ECG, EDA, EMG, RESP, TEMP) + Empatica E4 trên cổ tay |
| **Tần số lấy mẫu** | 700 Hz (ngực), 4-64 Hz (cổ tay) |
| **Bằng chứng stress** | TSST protocol: Baseline → Stress (phát biểu công cộng + bài toán tính nhẩm) → Hài hước → Thiền |
| **Số lượng dữ liệu** | 17,367 cửa sổ 5 giây |
| **Class imbalance** | Stress ~10.9%, Non-stress ~89.1% |
| **Kết quả tìm thấy** | Heart rate mean (hr_mean) có Cohen's d = **1.55** (RẤT LỚN) — loại signal không mơ hồ |

**Phát hiện chính:** Stress dễ dàng phát hiện từ timing của nhịp tim (R-R interval), không phải từ hình dạng sóng ECG. LogReg trên 5-10 tín hiệu HRV đạt **0.763 balanced accuracy**, tương đương với các mô hình CNN phức tạp.

#### 🎬 Dataset 2: DREAMER (Emotion Recognition through EEG & ECG)

| Thuộc tính | Chi tiết |
|------------|----------|
| **Chủ đề** | 23 người |
| **Ghi cảm biến** | 14 kênh EEG (128 Hz) + ECG + cảm biến Shimmer |
| **Bằng chứng stress** | Phim cảm xúc (18 video: buồn, vui, trung lập) |
| **Nhãn dữ liệu** | Tự báo cáo: Valence (tiêu cực-tích cực), Arousal (yên tĩnh-kích động), Dominance |
| **Số lượng dữ liệu** | 85,744 cửa sổ 2 giây |
| **Kết quả tìm thấy** | Balanced accuracy chỉ **0.600** (thấp hơn nhiều so với WESAD) |

**Phát hiện chính:** Đây KHÔNG phải là thất bại của mô hình. Thay vào đó, nó **chứng minh** rằng phát hiện cảm xúc từ EEG tiêu dùng (consumer-grade) là vốn có kém hiệu quả. Label của DREAMER đã bị nhiễu (noise ceiling = 0.600), và mô hình ta đã đạt được ngưỡng này. Điều này là một **negative result có giá trị** cho lĩnh vực nghệ tận cảm xúc.

**Vai trò trong dự án:** DREAMER hoạt động như một **tập dữ liệu xác thực epistemic** — nó chứng minh pipeline của chúng ta có thể phát hiện các giới hạn của tín hiệu, không phải chỉ học mà không kiểm soát.

#### 💹 Dataset 3: Tardis-Binance BTC Futures

| Thuộc tính | Chi tiết |
|------------|----------|
| **Nguồn** | Tardis.dev API (dữ liệu Binance) |
| **Công cụ** | BTCUSDT Perpetual Futures |
| **Khoảng thời gian** | 01/06/2020 - 31/12/2024 |
| **Dữ liệu** | 2,410,560 thanh 1 phút (OHLCV) + microstructure |
| **Tính năng** | 21 tính năng (volatility, volume, order flow, spread, ...) |

**Vai trò:** Calibration thị trường — tham chiếu stylized facts (sự kiện lặp lại trong dữ liệu thực) để xác thực mô hình ABM sau này phù hợp với thực tế thị trường.

---

## IV. PIPELINE 5 PHASES — 25 SCRIPTS

Stage 1 được thực hiện qua 5 phases với 25 scripts nhỏ, mỗi script thực hiện một nhiệm vụ cụ thể:

### Phase 1: Data Engineering (Scripts 00-09) ✅

**Mục đích:** Kiểm tra, xử lý, trích xuất tính năng từ dữ liệu thô

| Script | Tên | Kết quả |
|--------|-----|---------|
| 00 | Tải dữ liệu Tardis | ✅ 2.4M thanh BTC |
| 01 | Kiểm tra WESAD | ✅ 15 chủ đề, 17,367 cửa sổ |
| 02 | Kiểm tra DREAMER | ✅ 23 chủ đề, 85,744 cửa sổ |
| 03 | Kiểm tra Tardis | ✅ Stylized facts xác thực |
| 04-06 | Xử lý trước (preprocess) | ✅ ECG filter, R-peak detection, EDA normalization |
| 07 | Trích xuất tính năng | ✅ HRV (hr_mean, rmssd, sdnn), EDA, DE (Differential Entropy) |
| 08 | Kiểm tra căn chỉnh | ✅ Đồng bộ hóa cross-dataset |
| 09 | Stylized facts | ✅ Xác thực 5/5 sự kiện lặp lại thị trường |

**Phát hiện:** Tất cả 10/10 scripts hoàn thành thành công. Dữ liệu sạch, aligned, sẵn sàng cho validation.

### Phase 2: Scientific Validation (Scripts 10-15) ✅

**Mục đích:** Kiểm tra độ chính xác khoa học — liệu tín hiệu có thực sự xuất phát từ sinh lý hay chỉ là identity của chủ đề?

| Script | Tên | Kết quả |
|--------|-----|---------|
| 10 | Baseline model (LogReg) | ✅ **0.763 balanced acc** trên WESAD |
| 11 | Probe: Subject classifier | ✅ 92.6% → 11.5% sau z-norm (chứng minh confound được loại) |
| 12 | Adversarial GRL | ✅ **+0.014 delta** (ROBUST — tín hiệu thực sự là sinh lý) |
| 13 | Minimal model | ✅ 5 tính năng HRV đủ (không cần EDA, EEG complexity) |
| 14 | ICA check DREAMER | ✅ Không có thành phần ẩn trừ noise (không phải có tín hiệu không thấy) |
| 15 | Validity report | ✅ Sinh báo cáo khoa học |

**Phát hiện chính:** Tín hiệu là THỰC (không phải artifact). GRL (Gradient Reversal Layer) thử loại bỏ thông tin chủ đề và khoảng cách giảm rất ít (+0.014), chứng minh tín hiệu là sinh lý thực thụ.

### Phase 3: Deep Model Exploration (Scripts 16-18) ✅

**Mục đích:** Kiểm tra liệu Deep Learning có vượt trội hơn LogReg đơn giản không?

| Script | Tên | Kết quả |
|--------|-----|---------|
| 16 | DREAMER recovery | ✅ Phát hiện label noise, đạt ceiling 0.600 |
| 17 | WESAD CNN | ✅ CNN 1D: **0.686** bal_acc (THẤP HƠN LogReg 0.763!) |
| 18 | Post-recovery validation | ✅ CNN R-R: 0.750 bal_acc, **AUC 0.913** (tốt!) |

**Phát hiện chính:** Deep Learning **KHÔNG** cải thiện được LogReg trên WESAD. Lý do: tín hiệu quá đơn giản (chỉ là nhịp tim). Dùng LogReg tiết kiệm chi phí tính toán!

### Phase 3+: Advisor Hypotheses (Scripts 19-22) ✅

**Mục đích:** Kiểm tra các giả thuyết bổ sung từ cố vấn

| Script | Tên | Kết quả |
|--------|-----|---------|
| 19 | CNN threshold optimization | ✅ Tối ưu NGS threshold = 0.41 |
| 20 | DREAMER connectivity | ✅ PLV + Coherence không cải thiện → signal hạn chế |
| 21 | R-R interval model | ✅ 30-beat window (1s) không tốt hơn 5s window |
| 22 | DREAMER label ceiling | ✅ **0.600 = pessimistic ceiling** (label noise, không mô hình) |

**Phát hiện chính:** Không có "silver bullet". Tối ưu hóa threshold, weighted regularization, multi-task learning đều không cải thiện. DREAMER's 0.600 là hard limit từ chính bộ dữ liệu.

### Phase 4: Stochastic Law Discovery (Scripts 23-25) ✅

**Mục đích:** Phát hiện quy luật stochastic của stress động học — σ(t) tuân theo quá trình nào?

| Script | Tên | Kết quả |
|--------|-----|---------|
| 23 | Representation transfer | ✅ CKA ≈ 0 — representation KHÔNG transfer |
| 24 | Stress process ID | ✅ **OU mean-reversion model** trên 15/15 subjects |
| 25 | Final validation | ✅ θ = 0.074, half-life = 10.7s, **θ là artifact của window size** |

**Phát hiện chính:** Stress tuân theo quá trình **Ornstein-Uhlenbeck (OU) mean-reverting**:

$$d\sigma = -\lambda(\sigma - \bar{\sigma})dt + \eta dW$$

- **λ = 0.074** (mean reversion rate)
- **Half-life = 10.7 giây** (thời gian recover về baseline)
- **CV = 0.32** (biến thiên inter-subject < 20%)

Điều này **KHÔNG** là một hằng số sinh lý thực (như bằng chứng gợi ý), mà là **artifact của cách chọn window size 5 giây**. Các process ẩn lặn ở beat-to-beat timescale (ms → 1s), nhanh hơn cửa sổ 5s của chúng ta. Nhưng ở 5s timescale, mean-reversion là mô hình tốt nhất.

---

## V. CÂU HỎI CHÍNH & CÂU TRẢ LỜI

| Câu Hỏi | Câu Trả Lời | Bằng Chứng |
|--------|----------|-----------|
| **Liệu stress có phát hiện được từ ECG không?** | ✅ **CÓ**, dễ dàng | bal_acc=0.763; hr_mean d=1.55 (rất lớn) |
| **Liệu tín hiệu có phải là artifact của subject identity không?** | ❌ **KHÔNG** — tín hiệu là sinh lý | GRL Δ=+0.014 (ROBUST); z-norm giảm probe từ 92.6% → 11.5% |
| **Liệu Deep Learning có tốt hơn LogReg không?** | ❌ **KHÔNG** — LogReg tốt hơn | LogReg 0.763 > CNN 0.686 |
| **Liệu DREAMER là tập dữ liệu tốt không?** | ⚠️ **KHÔNG**, nhưng hữu ích | 0.600 = label ceiling; chứa 45% trường hợp ambiguous; tín hiệu bị domain imbalance |
| **Liệu representation có transfer được không?** | ❌ **KHÔNG** — domain-specific | CKA ≈ 0; 22.7× separability khác biệt |
| **Liệu σ(t) tuân theo quy luật gì?** | ✅ **OU mean-reversion** | 15/15 subjects fit; θ=0.074, half-life=10.7s |
| **Liệu θ có phải hằng số sinh lý không?** | ❌ **KHÔNG** — window artifact | θ phụ thuộc resolution (5s window); không lỗi tuân theo timescale nhanh hơn |

---

## VI. LỖI CHÍNH & BÀI HỌC

### 6.1 Các Điều Phát Hiện Ra

| Khám Phá | Tác Động | Ứng Dụng |
|----------|----------|---------|
| **HR timing > HR magnitude** | Heart rate TIMING (R-R intervals) lỡ quan trọng hơn Average HR | ABM agents cần mô hình beat-to-beat autonomic dynamics |
| **Labeling is the bottleneck** | Với DREAMER, mô hình không phải là vấn đề — label quality là | Để phát hiện cảm xúc cross-subject, cần dữ liệu hoặc nhãn tốt hơn |
| **Domain gap > algorithm gap** | Giữa WESAD (lab) và DREAMER (film), domain khác biệt > bất kỳ tuning nào | Coupling layer (Stage 3) cần domain adaptation |
| **5s window hides fast dynamics** | Quá trình ẩn lặn nhanh hơn window size | Khi xây dựng ABM coupling, sử dụng beat-to-beat (ms), không phải windowed features |
| **OU model đủ** | Không cần Brownian motion phức tạp | Stress dynamics có thể được mô phỏng với OU, tiết kiệm chi phí tính toán trong ABM |

### 6.2 Cảnh Báo (Caveats)

1. **WESAD là acute stress, không phải financial stress** — Phát biểu công cộng ≠ loss aversion + time pressure. Coupling layer cần external grounding.

2. **Wearable sensor limitation** — Empatica E4 là low-cost, có noise cao. Signal quality sẽ khác khi sử dụng sensor y tế hoặc implant.

3. **Cross-subject generalization yếu** — DREAMER chứng minh: inter-individual khác biệt là lớn. ABM agents sẽ có heterogeneous stress response curves.

4. **Causal validity chưa được chứng minh** — Stage 1 chỉ chứng minh **association** (stress ↔ HR), không phải **causation** (stress → HR). Coupling layer + feedback dynamics cần để chứng minh causal chain.

---

## VII. GÌ SẼ LÀ BƯỚC TIẾP THEO (STAGE 2 onwards)

### 7.1 Stage 2: Market Simulator (ABM)

**Mục tiêu:** Xây dựng mô hình market mô phỏng thị trường BTC hành động như:

- 3 loại agent: Market Maker (cung cấp thanh khoản), Momentum (trend-following), Noise (random)
- State vector: spread, depth, volatility, order flow, midprice
- Validate: 5 stylized facts BTC (fat tails, volatility clustering, volume-volatility correlation, ...)

**Trạng thái:** Data curation hoàn thành (Tardis 2.4M bars). ABM pattern building chưa bắt đầu.

### 7.2 Stage 3: Bio → Behavior Coupling

**Mục tiêu:** Thiết lập ánh xạ stress → agent parameters:

$$ \theta = g(\sigma; \text{functional form}) $$

Ví dụ:
- $\gamma(t) = \gamma_0 \cdot (1 + \alpha \sigma(t))$ (risk aversion increases with stress)
- $\tau(t) = \tau_0 / (1 + \beta \sigma(t))$ (reaction time decreases — traders hurry)
- $I_{thr}(t) = I_{0} \cdot \exp(\gamma \sigma(t))$ (inattention increases)

**Dữ liệu từ Stage 1:** Hình dạng của σ(t) — mean-reverting OU process.

### 7.3 Stages 4-6: Closed-loop Validation

- **Stage 4:** Analyze stability — khi nào hệ thống bifurcate → crash?
- **Stage 5:** Evidence layer — kiểm tra kết quả ABM khớp với thị trường thực?
- **Stage 6:** Policy — nếu chúng ta can thiệp (circuit breaker, stress-aware regulation), sẽ xảy ra gì?

---

## VIII. CẤU TRÚC THƯ MỤC

```
Stress-Team/
├── README.md                          # Hướng dẫn setup
├── requirements.txt                   # Python dependencies
├── Algorithmic Panic...md             # Full research spec (7 stages)
│
├── config/
│   └── settings.py                    # Centralized config (paths, constants, thresholds)
│
├── src/                               # Source code package
│   ├── data/                          # Data loaders (WESAD, DREAMER, Tardis)
│   ├── audit/                         # Dataset validation checks
│   ├── preprocessing/                 # Signal processing
│   ├── features/                      # Feature extraction (HRV, EDA, DE, market)
│   ├── validation/                    # ML validation (LOSOCV, GRL, adversarial)
│   ├── analysis/                      # Stylized facts, stochastic fitting
│   └── utils/                         # IO helpers, plotting
│
├── scripts/                           # 25 runner scripts (5 phases)
│   ├── phase1_data_engineering/       # 00-09: Audit, preprocess, align
│   ├── phase2_validation/             # 10-15: Baselines, adversarial, validity
│   ├── phase3_deep_models/            # 16-18: Deep learning exploration
│   ├── phase3_improvements/           # 19-22: Advisor hypotheses
│   ├── phase4_representation/         # 23-25: Transfer, process ID, final validation
│   ├── stage2_economics/              # Placeholder for ABM
│   ├── README.md                      # Script guide
│   └── RUN_CHECKLIST.md               # Execution checklist
│
├── data/
│   ├── raw/                           # ⚠ Cần chuẩn bị thủ công
│   │   ├── wesad/WESAD_extracted/WESAD/S2..S17/
│   │   ├── dreamer/DREAMER.mat
│   │   └── tardis/                    # Tự tải bởi Script 00
│   └── processed/                     # Output từ scripts
│
└── reports/
    ├── PROGRESS.md                    # Chi tiết progress
    ├── BIO_STAGE_CLOSING.md           # Stage 1 final report
    └── validation/
        ├── model_validity_report_*.md # Báo cáo validity
        ├── stress_process_identification.json
        └── threshold_optimization_results.json
```

---

## IX. KỈ LỆNH CHẠY (Nếu Muốn Tái Tạo)

### Thiết lập môi trường

```powershell
# 1. Tạo conda environment
conda create -n stress python=3.10 -y
conda activate stress

# 2. Cài dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Chuẩn bị datasets (BẮTBUỘC)
# - Tải WESAD.zip từ UCI → giải nén vào data/raw/wesad/
# - Tải DREAMER.mat từ Zenodo → đặt vào data/raw/dreamer/
# - Script 00 tự tải Tardis
```

### Chạy các phases (tuần tự)

```powershell
# Phase 1: Data Engineering
for ($i = 0; $i -le 9; $i++) { python scripts/phase1_data_engineering/$("{0:02d}" -f $i)_*.py }

# Phase 2: Validation
for ($i = 10; $i -le 15; $i++) { python scripts/phase2_validation/$("{0:02d}" -f $i)_*.py }

# Phase 3: Deep models
for ($i = 16; $i -le 18; $i++) { python scripts/phase3_deep_models/$("{0:02d}" -f $i)_*.py }

# Phase 3+: Advisor
for ($i = 19; $i -le 22; $i++) { python scripts/phase3_improvements/$("{0:02d}" -f $i)_*.py }

# Phase 4: Stochastic
for ($i = 23; $i -le 25; $i++) { python scripts/phase4_representation/$("{0:02d}" -f $i)_*.py }
```

---

## X. KỀT LUẬN

**Stress-Team** là một dự án tham vọng nhằm chứng minh rằng thị trường tài chính là **hệ thống sinh-kỹ thuật ghép nối**, nơi stress sinh lý của các trader ảnh hưởng trực tiếp đến volatility thị trường.

**Giai đoạn 1 (Bio Stage)** đã hoàn thành:
- ✅ Xây dựng inference engine phát hiện stress từ ECG (acc=76.3%)
- ✅ Chứng minh tín hiệu là sinh lý (GRL robust)
- ✅ Phát hiện quy luật stochastic (OU mean-reversion)
- ✅ Hạn định ceiling của tập dữ liệu khác (DREAMER 60%)

**Bước tiếp theo** là xây dựng thị trường mô phỏng (Stage 2) và nối ghép sinh lý → hành vi giao dịch (Stage 3), rồi kiểm chứng vòng lặp hồi đáp.

Nếu thành công, dự án sẽ mở ra one **paradigm mới trong fintech**: *biometric-augmented market models*.

---

**Cập nhật lần cuối:** 28/06/2025  
**Trạng thái:** Stage 1 CLOSED → Sẵn sàng Stage 2  
**Liên hệ:** [Advisor] cho chi tiết.
