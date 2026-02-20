# Algorithmic Panic: Phân Tích Toàn Diện Research Idea & Dataset Audit Protocol

## I. Tổng Quan Kiến Trúc Nghiên Cứu

Framework nghiên cứu trong bản FINAL-RESEARCH-APPROACH của team bao gồm 7 stages (Stage 0 → Stage 6) với các nguyên tắc kiến trúc: Modular Validity, Coupled Emergence, Causal Transparency, Multi-Scale Consistency, và Falsifiability. Dưới đây là phân tích chi tiết từng khía cạnh, từ tính đột phá đến vấn đề kỹ thuật cụ thể.[^1]

***

## II. Đánh Giá Tính Đột Phá & Novelty

### 2.1 Core Novelty: Endogenous Stress as State Variable

Điểm mạnh lớn nhất của đề tài nằm ở việc **mô hình hóa stress sinh lý như biến nội sinh** (endogenous variable) trong feedback loop thị trường. Các nghiên cứu ABM hiện tại — kể cả các mô hình hàng đầu từ OFR (Office of Financial Research) — đều coi stress/panic là cú sốc ngoại sinh (exogenous shocks) từ tin tức hoặc sự kiện bên ngoài. Bạn đề xuất vòng lặp `volatility ↑ → stress ↑ → liquidity ↓ → volatility ↑` — đây là contribution chưa từng có trong literature.[^2][^3]

### 2.2 Hybrid Architecture DL + ABM

Việc kết hợp Deep Learning (để suy luận latent stress) với Agent-Based Modeling (để mô phỏng thị trường) trong cùng một hệ thống thống nhất là **methodologically novel**. Hiện tại, các nghiên cứu hybrid ML+ABM chủ yếu dùng ML để tối ưu tham số ABM hoặc phân tích output, chưa ai tạo ra "coupling layer" hai chiều như Stage 3 trong framework của bạn.[^4][^5]

### 2.3 Bio-Technical Ecosystem Vision

Claim khoa học "Financial markets are not purely algorithmic systems but coupled bio-technical ecosystems"  nếu được chứng minh sẽ mở ra một paradigm mới. Xu hướng hiện tại trong fintech đang move towards biometric integration (stress monitoring trong trading apps, wearable sensors cho traders), nên đề tài align rất tốt với hướng phát triển ngành.[^6][^1]

### 2.4 Đánh Giá Mức Độ Breakthrough

| Tiêu chí | Đánh giá | Ghi chú |
|-----------|----------|---------|
| **Conceptual novelty** | ⭐⭐⭐⭐⭐ | Endogenous bio-stress trong ABM chưa ai làm |
| **Methodological novelty** | ⭐⭐⭐⭐ | Hybrid DL+ABM coupling là mới |
| **Data novelty** | ⭐⭐⭐ | Datasets công khai, không có dữ liệu độc quyền |
| **Policy novelty** | ⭐⭐⭐⭐ | Stress-augmented circuit breaker chưa được đề xuất |
| **Overall** | **8/10** | Conditional: phụ thuộc causal validation |

***

## III. Phân Tích Chi Tiết Từng Stage

### 3.1 Stage 0: Causal Model Construction

**Đánh giá: Critical Foundation — Rất Tốt Khi Đã Được Thêm**

Việc thêm Stage 0 với DAG causal graph, assumption list, confounder analysis, identification strategy là **quyết định đúng đắn nhất** trong framework. Đây chính xác là best practice từ causal inference hiện đại.[^7][^8][^1]

**Điểm cần bổ sung:**
- **Estimand chính** cần được viết rõ dưới dạng: \(ATE = P(\text{crash} | \text{endogenous stress}) - P(\text{crash} | \text{baseline})\) — bạn đã có trong PDF nhưng cần elaborate chi tiết assumptions behind estimand này
- **Sensitivity analysis**: Cần plan sẵn phương pháp (e.g., E-value, Rosenbaum bounds) để test robustness khi unmeasured confounders tồn tại
- **DAG cụ thể**: Xác định rõ arrows giữa: External news → Stress; Market volatility → Stress; Stress → Risk aversion; Risk aversion → Trading behavior; Trading behavior → Market dynamics; Market dynamics → Market volatility → Stress (feedback)

### 3.2 Stage 1: Stress Inference Engine

**Đánh giá: Technically Sound — Cần Chi Tiết Về Data Pipeline**

Architecture đề xuất (EEG→CNN+Spectral, ECG→RNN, EDA→Transformer, Fusion→Bayesian head) là solid. State-of-the-art methods đã đạt 89-98% accuracy trên WESAD và DREAMER.[^9][^10][^1]

**Acceptance criteria `Accuracy >85%, ECE < 0.05`** là hợp lý. Tuy nhiên:[^1]

- **Subject-independent generalization** là thách thức lớn nhất. Nghiên cứu gần đây cho thấy cross-subject accuracy trên DREAMER chỉ đạt 64-68% dưới LOSOCV (Leave-One-Subject-Out Cross-Validation), thấp hơn nhiều so với within-subject.[^11][^12]
- **Bayesian head** cho uncertainty quantification là excellent choice — đây là điểm khác biệt so với hầu hết các paper chỉ output point estimates.
- **Calibration requirement (ECE < 0.05)** rất quan trọng cho downstream coupling — nếu stress predictions poorly calibrated, coupling layer sẽ propagate errors.

### 3.3 Stage 2: Market Simulator

**Đánh giá: Well-Structured — Cần Careful Calibration**

Thiết kế với 3 loại agent (Market Maker, Momentum, Noise) và state vector \(S_t = (\text{spread, depth, volatility, orderflow, midprice})\) là minimal viable.[^1]

**Stylized facts validation requirements** hợp lý. Nghiên cứu thực nghiệm đã xác nhận BTC futures thể hiện đầy đủ các stylized facts: fat tails (excess kurtosis 5-15), volatility clustering, và volume-volatility correlation. Tuy nhiên, recent analysis cho thấy chỉ 8/11 Cont's stylized facts remain fully robust ở intraday timeframe  — nên tập trung validate 5 facts core: (1) fat tails, (2) volatility clustering, (3) leverage effect, (4) volume-volatility correlation, (5) absence of return autocorrelation.[^13][^14][^15][^16]

**Rủi ro chính**: ABM thường chỉ reproduce stylized facts trong specific parameter regimes chứ không phải as asymptotic behavior. Cần systematic parameter sweep (Latin Hypercube Sampling) và report sensitivity.[^17][^18]

### 3.4 Stage 3: Bio → Behavior Coupling (CORE NOVELTY LAYER)

**Đánh giá: Highest Risk — Cần Empirical Grounding**

Đây là **make-or-break stage** của toàn bộ paper. Mapping \(\theta = g(\sigma)\) với 4 functional forms (linear, exp, sigmoid, neural) + Bayesian Model Averaging cho selection là methodologically correct.[^1]

**Critical issue**: Parameters affected (risk aversion γ, latency τ, tolerance \(I_{thr}\)) cần empirical bounds từ literature:

- **Risk aversion γ**: Nghiên cứu cho thấy stress có thể **giảm** risk aversion (do "narrowed focus" dưới cognitive load cao)  HOẶC **tăng** risk aversion (trong panic selling scenarios). Hướng ảnh hưởng phụ thuộc vào loại stress (acute vs. chronic) và context. **Recommendation**: Test cả hai directions trong simulation.[^19][^20][^21]
- **Autonomic balance → stress prediction**: Nghiên cứu gần nhất  xác nhận mối quan hệ mạnh giữa HRV-based autonomic markers và stress levels, với SHAP analysis cho thấy low autonomic balance → high stress prediction. Đây là foundation tốt cho coupling layer.[^22]
- **Temporal dynamics**: 83.5% of high-stress events được preceded by "state-transition" pattern — abrupt physiological changes chứ không phải steady-state. Coupling layer cần capture temporal transitions, không chỉ static mapping.[^22]

### 3.5 Stage 4: Feedback Dynamical System

**Đánh giá: Theoretically Strong — Cần Stability Analysis**

Stochastic stress bridge \(d\tilde{\sigma} = -\lambda(\tilde{\sigma} - \sigma)dt + \eta dW\) và instability condition "product sensitivities > 1" là mathematically rigorous.[^1]

**Cần bổ sung:**
- **Stability analysis**: Xác định vùng parameter space nào dẫn đến stable equilibrium vs. unstable runaway. Dùng Lyapunov exponents hoặc linear stability analysis.
- **Time-scale separation**: Bio signals (ms-seconds) vs. trading decisions (seconds-minutes) vs. market dynamics (minutes-hours). Cần explicit temporal aggregation rules.[^23][^24]
- **Bifurcation analysis**: Tìm tipping point \(\sigma_c\) qua bifurcation diagrams — tại giá trị nào stress level system transition từ stable → unstable.

### 3.6 Stage 5: Evidence Engine

**Đánh giá: Comprehensive — Layer Structure Excellent**

Three-layer evidence structure (Market Statistics, Mechanism Discovery, Comparative Validation) rất phù hợp cho journal paper.[^1]

**Comparative baselines** (ARIMA, classical ABM, no-stress model) là minimum viable. **Recommendation**: Thêm baseline "exogenous stress model" — inject random stress shocks thay vì endogenous — để demonstrate added value của endogenous mechanism cụ thể.

### 3.7 Stage 6: Policy Analysis

**Đánh giá: High Impact Potential**

Stress-aware circuit breaker với trigger `if PanicIndex > threshold → halt`  align với recent research showing welfare-optimized circuit breakers nên forward-looking và adapt to liquidity conditions. Tuy nhiên, "magnet effect" — nơi traders rush to trade before expected CB activation — vẫn controversial. Cần simulate magnet effect scenarios.[^25][^26][^1]

***

## IV. Model Ladder & Falsification Suite

### 4.1 Model Ladder (M0 → M3)

Model ladder design trong PDF (M0: no stress, M1: linear stress, M2: latent stress, M3: full system) với rule "giữ complexity chỉ khi statistically justified" là excellent experimental design. Đây chính là cách đúng để demonstrate each component's contribution.[^1]

**Recommendation**: Dùng likelihood ratio tests hoặc Bayesian Information Criterion (BIC) để formally compare M0 vs M1 vs M2 vs M3.

### 4.2 Falsification Suite

Falsification conditions (stress irrelevant agents → no crash; infinite liquidity → stable; tiny shock → no crash)  rất quan trọng cho credibility. Nếu model crash ở mọi case → invalid — đây là excellent scientific hygiene.[^1]

**Thêm falsification case**: "Random stress assignment" — nếu random stress (không correlated với market events) cũng gây crash tương tự, mechanism không thuyết phục.

***

## V. CHI TIẾT DATASET & AUDIT PROTOCOL

### 5.1 WESAD Dataset — Full Technical Profile

#### Cấu trúc dữ liệu

| Thuộc tính | Chi tiết |
|------------|----------|
| **Subjects** | 15 (ban đầu 17, S1 và S12 bị loại do sensor malfunction) [^27] |
| **Chest device** | RespiBAN: ECG, EDA, EMG, RESP, TEMP, 3-axis ACC — tất cả 700 Hz [^28] |
| **Wrist device** | Empatica E4: BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz), ACC (32 Hz) [^27] |
| **Conditions** | Baseline, Stress (TSST), Amusement (funny clips), Meditation [^29][^30] |
| **Labels** | Protocol-based ground truth + self-report questionnaires [^29] |
| **Total instances** | ~63 triệu data points [^28] |
| **Sync method** | Double-tapping gesture → ACC pattern matching giữa chest và wrist [^27] |
| **Storage** | Local storage, no wireless transmission → no packet loss [^31] |

#### Vấn đề bắt buộc phải check trước khi dùng

**1. Subject Exclusion & Missing Data**
- S1 và S12 đã bị loại bỏ do sensor malfunction. Kiểm tra rằng data folders cho S1, S12 thực sự không tồn tại.[^27]
- Một số nghiên cứu khác báo cáo S2 cũng bị loại  — **cần verify lại** xem bạn nhận được bao nhiêu subject folders khi download.[^32]
- **Check script**:
```python
import os
subjects = [f for f in os.listdir('WESAD/') if f.startswith('S')]
print(f"Available subjects: {sorted(subjects)}")
print(f"Total: {len(subjects)}")
# Expected: 15 subjects (S2-S11, S13-S17), missing S1, S12
```

**2. Sampling Rate Mismatch (CRITICAL)**
- Chest device: 700 Hz cho tất cả modalities
- Wrist device: 4-64 Hz tùy modality
- **Bắt buộc phải resample** trước khi combine. Best practice: Downsample chest signals hoặc upsample wrist signals lên common rate.[^33]
- **Recommendation cho Stage 1**: Dùng chest ECG (700 Hz) là primary cho cardiac features, wrist EDA (4 Hz) cần upsample nếu muốn fuse với ECG.

**3. Class Imbalance (SEVERE)**
- Chỉ **~11% data là stress**. Baseline chiếm đa số (~60%), amusement (~25%), meditation (~4%).[^34]
- **Bắt buộc phải xử lý** bằng: SMOTE, weighted loss function, hoặc time-window undersampling.[^31]
- **Check script**:
```python
import pickle
import numpy as np
with open('WESAD/S2/S2.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
labels = data['label']
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Label {u}: {c} samples ({c/len(labels)*100:.1f}%)")
# Labels: 0=undefined, 1=baseline, 2=stress, 3=amusement, 4=meditation
```

**4. RR-Interval Missing Data**
- Features liên quan RR-intervals có **>85% missing data** trong physical exertion periods.[^33]
- Heart rate variability (HRV) metrics (rr_mean, rr_std) bị ảnh hưởng nặng nhất.
- **Recommendation**: Nếu dùng HRV features, restrict analysis vào sedentary periods hoặc dùng ECG raw signal để extract R-peaks trực tiếp.

**5. Motion Artifacts**
- Significant degradation khi subjects di chuyển.[^35][^31]
- **Preprocessing pipeline bắt buộc**:
  - ECG: Bandpass filter 0.5-40 Hz, detect R-peaks (Pan-Tompkins algorithm)
  - EDA: Chebyshev II hoặc Butterworth low-pass filter; artifact removal via extended Kalman filter hoặc particle filter[^35]
  - IBI (Inter-Beat Interval): Reject IBIs outside 250-2000ms (physiological range)[^35]
  - ACC-based artifact detection: Use accelerometer to flag motion-contaminated segments[^31]

**6. Device Synchronization**
- RespiBAN và Empatica E4 cần manual synchronization qua double-tapping gesture pattern trong ACC signal.[^27]
- File `SX.pkl` đã chứa synchronized data — **dùng file này** thay vì sync thủ công.
- **Verify**: Check rằng labels align đúng với physiological data bằng cách visualize stress onset periods.

**7. Ecological Validity Gap**
- Stress induced bằng TSST (Trier Social Stress Test) = public speaking + arithmetic task.[^29]
- **KHÔNG phải trading stress**. Cần frame rõ trong paper rằng đây là "acute psychological stress proxy" chứ không phải "financial decision-making stress".

***

### 5.2 DREAMER Dataset — Full Technical Profile

#### Cấu trúc dữ liệu

| Thuộc tính | Chi tiết |
|------------|----------|
| **Subjects** | 23 (9 nữ, 14 nam) [^36] |
| **EEG device** | Emotiv EPOC: 14 channels, 128 Hz [^37][^38] |
| **ECG device** | Shimmer2 sensor [^37] |
| **Stimuli** | 18 film clips (emotion induction) [^11] |
| **Labels** | Self-report: Valence, Arousal, Dominance (1-5 scale) [^39] |
| **Baseline** | 61 seconds per trial [^40] |
| **Data format** | MATLAB .mat file [^41] |
| **Channels** | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 [^38] |

#### Vấn đề bắt buộc phải check trước khi dùng

**1. Consumer-Grade Signal Quality (CRITICAL)**
- Emotiv EPOC là consumer-grade device với **14 channels** so với 32-256 channels của research-grade systems.[^38]
- **Trong controlled lab conditions**: Signal quality comparable với research-grade (nearly identical event-related potentials).[^38]
- **Trong uncontrolled environments**: Significant degradation từ motion artifacts, electrode impedance variations, electromagnetic interference.[^38]
- **Implication cho Stage 1**: Model trained trên DREAMER có thể không generalize tốt sang real-world sensors. Cần discuss limitation này.

**2. No Explicit Stress Labels (CRITICAL)**
- DREAMER labels là **Valence, Arousal, Dominance** — KHÔNG phải stress trực tiếp.[^37][^39]
- **Mapping required**: Stress thường tương ứng với **Low Valence + High Arousal**. Cần define stress proxy:[^12][^42]
  - `stress_proxy = (arousal >= threshold_high) AND (valence <= threshold_low)`
  - Common thresholds: Binarize tại midpoint (3 trên scale 1-5)[^12][^38]
- **Validation**: So sánh stress proxy này với WESAD ground truth labels để check consistency.

**3. EEG Artifact Contamination**
- Cần ICA (Independent Component Analysis) cho artifact removal:[^43][^44]
  - Decompose thành 13 independent components (recommended cho 14-channel Emotiv)[^43]
  - Classify components bằng ICLabel
  - Remove eye blink (EOG), muscle (EMG) artifacts
  - Reconstruct cleaned signal
- **Caution**: ICA có thể remove brain signal components cùng artifacts. Dùng EEG-X approach hoặc CLEnet cho controlled artifact removal.[^44][^43]
- **Bandpass filter**: 0.1-40 Hz mandatory; notch filter 48-52 Hz cho powerline interference.[^44]

**4. Label Distribution & Imbalance**
- Self-report labels trên scale 1-5 — distribution không uniform.
- **Check script**:
```python
import scipy.io
data = scipy.io.loadmat('DREAMER.mat')
# Check label distributions for valence, arousal, dominance
for dim in ['valence', 'arousal', 'dominance']:
    all_labels = []  # extract from data structure
    print(f"{dim}: mean={np.mean(all_labels):.2f}, std={np.std(all_labels):.2f}")
    print(f"Distribution: {np.bincount(all_labels)}")
```
- Binary classification: Split tại threshold 3 → check if balanced.[^38]

**5. Cross-Subject Generalization (SEVERE)**
- **Within-dataset LOSOCV accuracy chỉ 64-68%** cho DREAMER  — thấp hơn nhiều so với within-subject 85-90%+.[^42][^11][^12]
- Cross-dataset transfer (DREAMER→SEED-VII): Deep learning F1 = **0.007** (gần zero!), traditional ML + domain adaptation F1 = 0.619.[^45][^38]
- **Implication**: Bạn đặt acceptance criteria "subject-independent generalization"  — cần realistic expectations. Có thể cần fine-tuning per subject hoặc domain adaptation techniques.[^1]

**6. Differential Entropy Feature Extraction**
- Standard preprocessing: Chia EEG thành 4-5 frequency sub-bands (delta, theta, alpha, beta, gamma), compute differential entropy per band.[^40][^36]
- **Baseline removal**: Subtract baseline (61s) features từ stimulus features.[^40]
- **Window size**: 1-2 seconds windows, 128-256 data points per window.[^41][^43]

***

### 5.3 Tardis-Binance BTC Futures — Full Technical Profile

#### Cấu trúc dữ liệu

| Thuộc tính | Chi tiết |
|------------|----------|
| **Available since** | 2019-11-17 [^46] |
| **Data types** | trades, incremental_book_L2, quotes, book_snapshot_25, derivative_ticker, liquidations [^46] |
| **Order book depth** | Real-time updates (depth@0ms since 2020-01-07) [^46] |
| **Snapshots** | Top 1000 levels via REST API (generated, not native WS) [^46] |
| **Integrity** | Sequence number validation (pu, u fields) [^46] |
| **Infrastructure** | Tokyo DC since 2020-05-14, London before [^46] |
| **Update speed** | depth@100ms before 2020-01-07, depth@0ms after [^46] |

#### Vấn đề bắt buộc phải check trước khi dùng

**1. Pre-2020-05-14 Data Quality (CRITICAL)**
- Tardis official documentation confirms: **"Data collection before 2020-05-14 suffered some issues (missing data, latency spikes) during market volatility periods"**.[^46]
- **Recommendation**: Chỉ dùng data **từ 2020-05-14 trở đi** cho research. Data trước đó unreliable cho microstructure analysis.

**2. Daily Data Gaps (300-3000ms)**
- Mỗi 24h, có gap **300-3000ms** khi WebSocket re-subscribes.[^47]
- Với HFT simulation, gap này có thể ảnh hưởng orderbook state. **Cần detect và interpolate** hoặc skip affected periods.
- **Check**: Tìm `is_snapshot=true` rows không phải đầu ngày → indicates connection restart.

**3. May 19, 2021 Flash Crash — DATA INTEGRITY CRISIS**
- Bitcoin crashed 30% trong ngày này.
- Binance **halted trading cho retail clients** và **stopped providing transaction data** trong ~40 phút (13:00-15:00 UTC).[^48][^49]
- **Tardis data chứa gap** trong khoảng này.[^49]
- Binance later **back-filled** missing transactions — nhưng nghiên cứu từ IWH Halle cho thấy back-filled data **không conform Benford's Law**, indicating potential data manipulation.[^48]
- **Bắt buộc**: 
  - Flag toàn bộ ngày 2021-05-19 trong analysis
  - Nếu dùng làm "crash event" để validate model, cần cross-reference với exchanges khác (FTX archive, Bybit) 
  - Best practice: **Exclude** ngày này hoặc handle riêng

**4. Orderbook Reconstruction Protocol**
- Binance KHÔNG provide native WebSocket snapshots — Tardis generates từ REST API.[^50][^46]
- Reconstruction flow:
  1. Start từ `is_snapshot=true` row (đầu ngày hoặc sau connection restart)
  2. Apply incremental updates sequentially
  3. Khi `amount = 0` → remove price level
  4. Khi `is_snapshot=true` appears mid-day → **RESET** local orderbook state[^50]
- **Validation**: So sánh reconstructed midprice với trades data — should match within spread.
- **Check script concept**:
```python
# Validate orderbook integrity
# 1. Check sequence numbers are monotonically increasing
# 2. Check no sequence gaps (would indicate missed messages)
# 3. Verify best_bid < best_ask at all times
# 4. Cross-validate midprice with trade prices
```

**5. Hidden/Iceberg Orders**
- L2 data chỉ show visible orders. Hidden orders và iceberg orders (chỉ show partial size) KHÔNG hiển thị.[^51]
- **Implication cho ABM**: True liquidity luôn > observed liquidity. Simulator cần account for "latent liquidity" beyond visible orderbook.

**6. Liquidation Data Availability**
- Liquidation data chỉ available since **2021-09-01**.[^46]
- Open interest data since **2020-05-13**.[^46]
- Nếu muốn study leverage-induced cascading liquidations, cần data từ 2021-09+ trở đi.

**7. BTC Stylized Facts Verification**
- BTC futures đã được confirm exhibit: fat tails (inverse cubic law tail exponents), volatility clustering, aggregational Gaussianity.[^14][^15][^13]
- **Đặc biệt**: BTC KHÔNG exhibit inverse volatility-asymmetry (leverage effect), khác với stocks. Cần adjust ABM expectations accordingly.[^13]
- **Absence of leverage effect** trong crypto challenge assumption rằng "losses → higher volatility" — cần explicit discussion.

**8. Tardis API Client Issues**
- API không provide 15 phút gần nhất của historical data.[^52]
- Live streaming 1-second interval **loses data points randomly**.[^52]
- Historical download luôn starts từ 00:00 UTC — downloads unwanted extra data.[^52]
- **Recommendation**: Dùng tardis-machine local server cho bulk downloads thay vì HTTP API.

***

## VI. Chiến Lược Khai Thác Dataset Cho Từng Stage

### 6.1 Stage 1: WESAD + DREAMER → Stress Inference Engine

#### Pipeline đề xuất:

**Phase A: WESAD — Primary Stress Model**
1. Load `SX.pkl` files (pre-synchronized)[^27]
2. Extract ECG (700 Hz) + EDA (700 Hz from chest) 
3. Preprocessing: bandpass filter, artifact rejection, normalization
4. Window: 5-second non-overlapping  hoặc 10-second overlapping[^53]
5. Labels: Binary (stress vs. non-stress) hoặc 4-class
6. Architecture: ECG→RNN branch + EDA→Transformer branch + Bayesian fusion head[^1]
7. Evaluation: LOSOCV (leave-one-subject-out) — expect ~80-93% binary accuracy[^30]

**Phase B: DREAMER — Cross-Modal Transfer & Augmentation**
1. Extract EEG (14 channels, 128 Hz) + ECG 
2. EEG preprocessing: bandpass 0.1-40Hz, ICA artifact removal (13 components)[^43]
3. Map Valence/Arousal → Stress proxy (low V + high A)
4. Architecture: EEG→CNN+Spectral branch[^1]
5. **Transfer learning**: Pre-train ECG branch trên WESAD → fine-tune trên DREAMER ECG → add EEG branch
6. Combined model outputs: `σ(t)` (mean stress) + uncertainty interval

**Phase C: Merging Strategy**
- Recent work  đã demonstrate viable cross-domain framework combining WESAD + DREAMER:[^10]
  - 1D-CNN trained trên WESAD → fine-tune trên DREAMER via transfer learning
  - Achieved 98% trên WESAD stress classification, 87.59% trên DREAMER emotion[^10]
- **Key insight**: ECG là common modality giữa hai datasets — dùng ECG branch làm "bridge" cho transfer learning.

### 6.2 Stage 2: Tardis-Binance BTC → Market Simulator

#### Data Curation Pipeline:

1. **Time range selection**: 2020-06-01 to 2024-12-31 (post-infrastructure fix)
2. **Primary data**: `incremental_book_L2` + `trades` + `liquidations` (from 2021-09)
3. **Orderbook reconstruction**: Follow Tardis protocol, validate sequence numbers
4. **Feature extraction** cho ABM calibration:
   - Spread: best_ask - best_bid
   - Depth: total volume within X% of midprice
   - Volatility: realized volatility at various frequencies (1min, 5min, 1hr)
   - Order flow: signed volume (buy - sell)
   - Midprice: (best_bid + best_ask) / 2

5. **Stylized facts extraction** (validation targets cho ABM):
   - Return distribution: fit power-law tails, compute excess kurtosis
   - Autocorrelation function of returns (should be ~0 at lag > 1)
   - Autocorrelation of |returns| and returns² (should decay slowly — volatility clustering)
   - Volume-volatility correlation
   - Spread distribution

6. **Event identification** cho scenario testing:
   - Flash crashes (>5% drop in <1 hour)
   - Liquidation cascades (spikes in `forceOrder` data)
   - Volatility regime changes

***

## VII. Comprehensive Dataset Audit Checklist

Đây là checklist **bắt buộc phải chạy** trước khi train bất kỳ model nào, dựa trên PDF framework  và enriched với findings từ literature:[^1]

### 7.1 WESAD Audit Checklist

| # | Check Item | Method | Pass Condition | Priority |
|---|-----------|--------|----------------|----------|
| W1 | Subject count verification | Count folders | 15 subjects (S2-S17, minus S12) | 🔴 Critical |
| W2 | Sampling rate consistency | Read pkl headers | Chest=700Hz, Wrist varies | 🔴 Critical |
| W3 | Label distribution | Count per class | Document exact % per class | 🔴 Critical |
| W4 | Class imbalance ratio | stress/total | If <15%, implement balancing | 🔴 Critical |
| W5 | Missing data per channel | Count NaN/None | <5% per channel | 🟡 High |
| W6 | RR-interval completeness | Check HRV features | Flag >20% missing periods | 🟡 High |
| W7 | Device synchronization | Correlate ACC patterns | Chest-wrist sync offset <100ms | 🔴 Critical |
| W8 | ECG signal quality | SNR estimation | SNR > 10dB per subject | 🟡 High |
| W9 | EDA artifact rate | Motion-correlated noise | Flag ACC-EDA correlated segments | 🟡 High |
| W10 | Label reliability | Compare protocol vs self-report | Cohen's κ > 0.6 | 🟢 Medium |
| W11 | Stress distribution shape | Histogram of stress segments | Mean 0.2-0.4, std 0.05-0.15 [^1] | 🟡 High |
| W12 | Subject demographic balance | Check SX_readme files | Report age, gender distribution | 🟢 Medium |

### 7.2 DREAMER Audit Checklist

| # | Check Item | Method | Pass Condition | Priority |
|---|-----------|--------|----------------|----------|
| D1 | Subject count | Load .mat file | 23 subjects | 🔴 Critical |
| D2 | Channel count & order | Verify channel names | 14 channels matching 10-20 system | 🔴 Critical |
| D3 | Sampling rate | Check data dimensions | 128 Hz (128 points/second) | 🔴 Critical |
| D4 | Label distribution (V/A/D) | Histogram per dimension | Document skewness, identify imbalance | 🔴 Critical |
| D5 | Stress proxy definition | V-A mapping | Explicitly define and justify thresholds | 🔴 Critical |
| D6 | EEG artifact detection | Compute variance per channel | Flag channels >3 std from mean | 🟡 High |
| D7 | ICA component quality | ICLabel classification | >70% brain components retained | 🟡 High |
| D8 | Baseline signal integrity | Check 61s baseline per trial | No missing/corrupted baselines | 🟡 High |
| D9 | Cross-subject variance | Compute inter-subject variability | Report range for normalization | 🟡 High |
| D10 | ECG signal availability | Check ECG channels exist | All 23 subjects have ECG | 🔴 Critical |
| D11 | Trial completeness | Count trials per subject | 18 trials × 23 subjects = 414 | 🟢 Medium |
| D12 | Frequency band power | PSD per band (δ,θ,α,β,γ) | Realistic PSD shape per subject | 🟢 Medium |

### 7.3 Tardis-Binance BTC Audit Checklist

| # | Check Item | Method | Pass Condition | Priority |
|---|-----------|--------|----------------|----------|
| T1 | Date range coverage | Check first/last timestamps | Continuous from 2020-05-14+ | 🔴 Critical |
| T2 | Pre-2020-05-14 exclusion | Filter by date | No data before infrastructure fix | 🔴 Critical |
| T3 | Timestamp ordering | `df['timestamp'].is_monotonic_increasing` | True for all files | 🔴 Critical |
| T4 | Sequence number gaps | Check `u` field continuity | No gaps (indicates missed messages) | 🔴 Critical |
| T5 | May 19, 2021 data check | Inspect 2021-05-19 | Flag/exclude 13:00-15:00 UTC gap | 🔴 Critical |
| T6 | Orderbook validity | best_bid < best_ask | True 100% of time after reconstruction | 🔴 Critical |
| T7 | Snapshot completeness | Count `is_snapshot=true` | ≥1 per day + after each restart | 🟡 High |
| T8 | Price outliers | Z-score of midprice returns | Flag |z| > 10 for manual inspection | 🟡 High |
| T9 | Missing ticks detection | Expected vs actual message count | <1% missing per day | 🟡 High |
| T10 | Daily reconnection gaps | Detect 300-3000ms gaps | Document and interpolate | 🟡 High |
| T11 | Trade-orderbook consistency | Cross-validate trade price vs spread | Trades should occur within spread | 🟡 High |
| T12 | Stylized facts validation | Compute kurtosis, ACF | Fat tails (kurt>3), vol clustering | 🔴 Critical |
| T13 | Liquidation data availability | Check `forceOrder` channel | Available from 2021-09-01 | 🟢 Medium |
| T14 | Open interest availability | Check `openInterest` channel | Available from 2020-05-13 | 🟢 Medium |
| T15 | Volume distribution | Intraday volume pattern | Check for realistic U-shape pattern | 🟢 Medium |

***

## VIII. Cross-Dataset Alignment Check

Vì WESAD/DREAMER dùng cho Stage 1 và Tardis dùng cho Stage 2, cần đảm bảo **compatibility** giữa outputs:

| Aspect | WESAD/DREAMER Output | Tardis/ABM Input | Alignment Required |
|--------|---------------------|-----------------|-------------------|
| **Time resolution** | Stress σ(t) per 1-10s window | Agent decisions per tick/second | Temporal aggregation rule |
| **Value range** | σ ∈ [^54] (calibrated probability) | θ = g(σ) → agent parameters | Coupling function g() |
| **Uncertainty** | Bayesian uncertainty interval | Stochastic noise in behavior | η parameter calibration |
| **Distribution** | Mean 0.2-0.4, std 0.05-0.15 [^1] | Must produce realistic market dynamics | Sensitivity analysis |

**Critical alignment**: Stress distribution từ WESAD/DREAMER phải có `mean ≈ 0.2-0.4, std ≈ 0.05-0.15`  SAU KHI model inference. Nếu distribution không match → coupling layer assumptions bị violated → recalibrate.[^1]

***

## IX. Implementation Priority & Risk Mitigation

### High Priority — Do First
1. **Run all audit checklists** trước khi bắt đầu bất kỳ modeling nào
2. **Define stress proxy cho DREAMER** và validate against WESAD stress labels
3. **Establish Tardis data pipeline** — reconstruct orderbook, verify stylized facts
4. **Build M0 (no-stress ABM)** — prove simulator works trước khi add complexity

### Medium Priority — Phase 2
5. **Build Stage 1 stress model** trên WESAD first (cleaner labels), then transfer to DREAMER
6. **Implement coupling layer** — start linear, test sensitivity
7. **Run Model Ladder** (M0 → M1 → M2 → M3) với formal statistical comparison

### Lower Priority — Phase 3
8. **Feedback loop implementation** (Stage 4)
9. **Tipping point detection** (Stage 5, Layer B)
10. **Policy analysis** (Stage 6)

### Critical Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| WESAD only 15 subjects | Augment with DREAMER ECG; use data augmentation; report confidence intervals |
| DREAMER no stress labels | Validate V-A→stress mapping against WESAD; sensitivity to threshold choices |
| Tardis data gaps | Exclude pre-2020-05-14; flag known incidents; use multiple crash events for validation |
| Coupling layer no empirical foundation | Start simple (linear), justify with literature bounds, extensive sensitivity analysis |
| Cross-subject generalization poor (~64%) | Fine-tune per subject cluster; domain adaptation; report subject-level results |
| Crypto ≠ traditional markets | Discuss limitation explicitly; note BTC lacks leverage effect unlike stocks [^13] |

***

## X. Verdict Tổng Thể

Framework FINAL-RESEARCH-APPROACH đã được thiết kế rất kỹ lưỡng với scientific rigor cao (falsification suite, model ladder, cross-module validation protocol). Tuy nhiên, **thành bại phụ thuộc vào 3 yếu tố quyết định**:

1. **Dataset audit quality**: Nếu data bị contaminated (artifacts, missing, imbalanced) mà không detect → toàn bộ downstream results unreliable. Chạy audit checklist là **non-negotiable first step**.

2. **Coupling layer grounding**: Stage 3 là core novelty nhưng cũng là highest risk. Bắt đầu simple (linear), provide multiple functional forms, và transparent sensitivity analysis.

3. **Honest limitations**: Frame contributions là "proof-of-concept framework demonstrating feasibility" chứ không phải "validated causal model". Lab stress ≠ trading stress; crypto ≠ traditional markets. Transparency builds credibility.

Nếu execute đúng, đây có potential trở thành **foundational work** mở ra research direction "Bio-Finance Coupled Systems". Kiến trúc 7-stage modular cho phép publish incrementally (Phase 1 = conference paper, Phase 2-3 = journal) và extend trong tương lai.

---

## References

1. [FINAL-RESEARCH-APPROACH.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/164711194/84355618-eebb-4c76-a5e3-b5084e14cc8f/FINAL-RESEARCH-APPROACH.pdf?AWSAccessKeyId=ASIA2F3EMEYEVBAYSHZX&Signature=onyUyPkaOt4qUrKMD3votQdbzGQ%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEKz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDRNGT6FBdhbQ4tv0ClXGC3rr1F9fZb8hX%2BWAxILhKD3AIgafrrO4um7l2Rwd62gOJ6syIO1q7wLLV7ACKOIq8pxPgq8wQIdRABGgw2OTk3NTMzMDk3MDUiDJRFnDcNLrxbQK4DpyrQBPB1uG0MRcBmOIu66vvyKbA4SirIskb%2B5j9z4xHTMS0hm8kXb7lbXlsARpAmip30QXMzLp0lGB77EP3Z%2B8flIc4tcqm3D3Z0w%2BzfYlU1RldRt1ywc5zABGymFPmgfiw74J4esEvOXedgxB9iB8PHZmCiqBxg59myGnnkcSuaTKqrHvySo32HSqTc7musnV83onMXmb0xHMq1Pi1Wrud7vvqFaSkjmvjYMFQEpfNbQgY6yUXMp2jXUyD2%2Fo8oSxjMaL3hW5%2FMGc7Ghjol%2BuyBtc1M99cr%2FLkWfbTa35lg7Pb%2Bc78jDwbA7gjLfA5QddfS7x3aKD9z6mzQfXMWaYSAWK0GmuNC%2FqZ0KoRCjMaOuRAg%2Fiidu0jKVYPRPaXuxFdy0UyO2IzwAyx%2F4w8yWf35AOb6zSuohzS8fkF2LSqiMz9oG3I%2BWvogoNHanbVEJfdn5bbeZYwHfurbyzEuv6zgpDAwth7Js1uNmnrTRX47O8vhpjVRRe4m%2FgqnRoQ31p1FcMTPh4UEVKDi9B15%2FN51e9ml1san%2BkZ3drq3jx%2FUugr%2BEgRrvv2mGfkM%2BPkpel4ieuPebvlEz8p55eOIKGIOeQRgsC7ISgX60vIzvKbV4OuYHXNqBu9FEADUAd8DtMSdd%2Bqg93PC5A%2Bq0d3sRNlTP6Kv5BTdsuVF8t45pVbBKtDkbBlhWhkCF%2FZrjk3xRXTbpXm%2BwDiEM%2FP0P9HnR0%2F1BIaDCEoUW79Tv7jZKcgj8ZSmXGLz%2FGFTNm9pyhyEfh%2FIx9KxtokzC%2FODip%2BZMEZ%2FxQQw1ZXazAY6mAHtvY2JoqlJv26gOPlv8imTm4SPV28K69gxATmN5zuVXCAuJGxE9QuPLaaR8ULmvC9PQjTJxRpzeydZ8%2Fof4owBW3%2BAlkkgBAXwJ7JNOiWlwNrMlMy5c05R59Zdm3VtF05P1UbJ2rdAB49BmzrhKv2PgWxElRD16fcqSThpEOEPpaUcDQo1aoJyGqcw4K3bu%2FHc5eAAMi5GLA%3D%3D&Expires=1771478777) - Nguyên tắc kiến trúc cuối cùng
Principle
Meaning
Modular Validity
mỗi stage independently testable
C...

2. [[PDF] An Agent-based Model for Financial Vulnerability](https://www.financialresearch.gov/working-papers/files/OFRwp2014-05_BookstaberPaddrikTivnan_Agent-basedModelforFinancialVulnerability_revised.pdf) - This paper develops the structure for an agent-based model to provide a system-wide view of the tran...

3. [[PDF] Agent-based model of system-wide implications of funding risk](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2121.en.pdf) - We build a model that focuses on systemic aspects of liquidity and its links with solvency condition...

4. [Using Agent-Based Modelling and Reinforcement Learning to Study ...](https://www.jasss.org/28/1/1.html) - To study hybrid threats, we present a novel agent-based model in which, for the first time, agents u...

5. [[PDF] Combining Machine Learning and Agent-Based Modeling to Study ...](https://arxiv.org/pdf/2206.01092.pdf) - As an example of the applicability of this type of ML in ABMs, one study developed a 3D hybrid agent...

6. [Neuromorphic energy economics: toward biologically inspired and ...](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1597038/full) - By merging biological inspiration with cutting-edge technology, future markets could achieve unprece...

7. [Step-by-step causal analysis of EHRs to ground decision ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC11790099/) - by M Doutreligne · 2025 · Cited by 8 — Causal inference enables machine learning methods to estimate...

8. [P. Ding (2024). A First Course in Causal Inference. Boca ...](https://www.cambridge.org/core/journals/psychometrika/article/p-ding-2024-a-first-course-in-causal-inference-boca-raton-fl-crc-press/9569D5AEFCB9859074C20E035EE3472B) - by J Rickles · Cited by 1 — At its core, causal inference is concerned with how we know when an obse...

9. [Cross-Modality Investigation on WESAD Stress Classification](https://arxiv.org/html/2502.18733v1) - Research has focused on optimizing model architectures, feature engineering, and dataset preprocessi...

10. [A cross-domain framework for emotion and stress detection ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12685819/) - by A Almadhor · 2025 — The DREAMER dataset includes EEG signals and is designed for analyzing valenc...

11. [Cross-subject EEG Emotion Classification on Datasets with Limited Channels](https://www.ewadirect.com/proceedings/tns/article/view/24062) - To solve the challenge of achieving strong and accurate EEG datasets, while solving the domain shift...

12. [A multi-task EEG emotion recognition method based on ... - Nature](https://www.nature.com/articles/s41598-025-34211-z) - Following common practice, valence/arousal/dominance (V–A–D) labels were binarized using a threshold...

13. [Return and volatility properties: Stylized facts from the universe of cryptocurrencies and NFTs](https://khu.elsevierpure.com/en/publications/return-and-volatility-properties-stylized-facts-from-the-universe)

14. [5.2 Volatility Clustering](https://arxiv.org/html/2402.11930v2)

15. [Relevant stylized facts about bitcoin: Fluctuations, first ...](https://www.sciencedirect.com/science/article/abs/pii/S0378437120300133)

16. [Revisiting Cont's Stylized Facts for Modern Stock Markets †](https://arxiv.org/html/2311.07738v2) - These high-level characterizations are referred to as stylized facts, which are then used to inform ...

17. [Agent-Based Modelling for Financial Markets](https://openaccess.city.ac.uk/id/eprint/1744/1/iori_porter_2012.pdf) - by G Iori · 2012 · Cited by 64 — The typical approach taken for ABM, as with most of the work survey...

18. [Finance and Market Concentration Using Agent-Based Modeling](https://www.jasss.org/28/3/5.html) - Using agent-based modeling (ABM), we conduct qualitative and quantitative analyses to examine the im...

19. [Financial Decision Making Under Stress](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=1218&context=cgu_etd) - by NV Bejanyan · 2021 · Cited by 1 — The Cold Pressor Test (CPT) was used to induce a safe level of ...

20. [An Evaluation of the Consistency of Financial Risk-Aversion ...](https://fpperformancelab.org/wp-content/uploads/An-Evaluation-of-the-Consistency-of-Financial-Risk-aversion-Estimates-1.pdf) - by EJ Kwak · 2022 · Cited by 1 — This means that a financial decision-maker's risk-aversion score sh...

21. [Assessing Risk Aversion From the Investor's Point of View - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6614341/) - by A Díaz · 2019 · Cited by 38 — This paper contributes to filling the gap that exists in the litera...

22. [Implementation of a Stress Biomarker and Development of a ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12729542/) - The purpose of this study was to develop a model capable of predicting stress levels and interpretin...

23. [Dynamical systems with multiple long-delayed feedbacks](https://link.aps.org/doi/10.1103/PhysRevE.92.042903) - by S Yanchuk · 2015 · Cited by 30 — Dynamical systems with multiple, hierarchically long-delayed fee...

24. [Multi-Scale Simulation of Complex Systems](https://fi.ee.tsinghua.edu.cn/public/publications/d7269c60-2ea9-11ef-9fa8-0242ac120006.pdf) - by H WANG · 2024 · Cited by 42 — To provide a comprehensive understanding of interdisciplinary work ...

25. [Circuit breakers and market runs | Review of Finance](https://academic.oup.com/rof/article/28/6/1953/7749880) - by D Bongaerts · 2024 · Cited by 4 — We present a model that shows that adequately calibrated circui...

26. [Market Microstructure Evidence of China's Market-Wide Circuit ...](https://waf-e.dubuplus.com/apjfs.dubuplus.com/anonymous/O18C3WH/DubuDisk/public/cafm/2019/2019-11-4.pdf) - by X Wang · Cited by 1 — In summary, proponents of circuit breakers argue that the mechanism is able...

27. [WESAD Dataset Readme: Wearable Stress & Affect ...](https://www.studocu.com/in/document/masters-union-school-of-business/computer-science-sl/wesad-readme-it-is-very-helpful-for-stupids/116820688) - Share free summaries, lecture notes, exam prep and more!!

28. [WESAD (Wearable Stress and Affect Detection)](https://archive.ics.uci.edu/ml/datasets/WESAD+(Wearable+Stress+and+Affect+Detection)) - WESAD is a publicly available dataset for wearable stress and affect detection. This multimodal data...

29. [Introducing WESAD, a Multimodal Dataset for Wearable Stress ...](https://ai.updf.com/paper-detail/introducing-wesad-a-multimodal-dataset-for-wearable-stress-and-affect-schmidt-reiss-f7d4957127bb35b0d3cb1042a676ea60e259463d) - This work introduces WESAD, a new publicly available dataset for wearable stress and affect detectio...

30. [Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection](https://dl.acm.org/doi/10.1145/3242969.3242985) - Affect recognition aims to detect a person's affective state based on observables, with the goal to ...

31. [Stress and Emotion Open Access Data: A Review on Datasets ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12290141/) - Preprocessing techniques to detect and remove motion artifacts ... This issue can result in incomple...

32. [DATASET WESAD - Wearable Stress and Affect Detection](https://cicero.engcomp.uema.br/wp-content/uploads/sites/2/2024/09/WESAD.pdf)

33. [Stress Detection from Multimodal Wearable Sensor Data - arXiv](https://arxiv.org/html/2508.10468v1) - This dataset comprises physiological responses of 35 subjects ... The WESAD dataset was introduced b...

34. [An Advanced Stress Detection Approach based on ...](https://thesai.org/Downloads/Volume12No7/Paper_45-An_Advanced_Stress_Detection_Approach.pdf) - Exploratory data analysis in this research was performed by using subjects S2 to S10 from the WESAD ...

35. [[PDF] FLIRT - UbiWell Lab](https://ubiwell.io/public/papers/foll-flirt.pdf) - Preprocessing the data with FLIRT ensures that unintended noise and artifacts are appropriately filt...

36. [CDBA: a novel multi-branch feature fusion model for EEG ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC10399240/) - EEG-based emotion recognition through artificial intelligence is one of the major areas of biomedica...

37. [DREAMER: A Database for Emotion Recognition through ...](https://zenodo.org/records/546113) - We present DREAMER, a multi-modal database consisting of electroencephalogram (EEG) and electrocardi...

38. [Traditional Machine Learning Outperforms EEGNet for ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12693886/) - by CRP Ocaranza · 2025 — The DREAMER dataset is specially valuable for consumer-grade BCI research a...

39. [DREAMER: A Database for Emotion Recognition Through EEG and ...](https://pubmed.ncbi.nlm.nih.gov/28368836/) - In this paper, we present DREAMER, a multimodal database consisting of electroencephalogram (EEG) an...

40. [Train a Tsception Model on the DREAMER Dataset¶](https://torcheeg.readthedocs.io/en/latest/auto_examples/examples_dreamer_tsception.html)

41. [DREAMERDataset¶](https://torcheeg.readthedocs.io/en/v1.0.10/generated/torcheeg.datasets.DREAMERDataset.html)

42. [Emotion Recognition Model of EEG Signals Based on Double ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11674476/) - On the other hand, the DREAMER dataset employs a two-dimensional emotional space model for self-asse...

43. [EEG-X: Device-Agnostic and Noise-Robust Foundation ...](https://arxiv.org/html/2511.08861v1) - EEG-X surpasses both by using artifact removal and the DiCT-enhanced reconstruction loss instead of ...

44. [A novel EEG artifact removal algorithm based on an advanced ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12134218/) - EEG is widely applied in emotion recognition, brain disease detection, and other fields due to its h...

45. [Traditional Machine Learning Outperforms EEGNet for Consumer-Grade EEG Emotion Recognition: A Comprehensive Evaluation with Cross-Dataset Validation - PubMed](https://pubmed.ncbi.nlm.nih.gov/41374637/) - These findings challenge the assumption that architectural complexity universally improves biosignal...

46. [Binance USDT Futures - Tardis.dev Documentation](https://docs.tardis.dev/historical-data-details/binance-futures) - Binance USDT Margined Futures historical market data details - instruments, data coverage and data c...

47. [Historical Data Details](https://docs.tardis.dev/historical-data-details) - It also means there is a tiny gap in historical data (around 300-3000ms , depending on the exchange)...

48. [Bitcoin Flash Crash on May 19, 2021: What Did Really ...](https://www.iwh-halle.de/publikationen/detail/bitcoin-flash-crash-on-may-19-2021-what-did-really-happen-on-binance) - Bitcoin plunged by 30% on May 19, 2021. We examine the outage the largest crypto exchange Binance ex...

49. [Bitcoin Flash Crash on May 19, 2021](https://www.paris-december.eu/sites/default/files/papers/2022/Baumgartner_2022_2.pdf)

50. [Data](https://docs.tardis.dev/faq/data) - Cryptocurrency exchanges' real-time APIs vary a lot, but for L2 order book data they all tend to fol...

51. [Local Order Book Tutorial Part 2: Snapshot From Rest Depth](https://www.binance.com/en/academy/articles/local-order-book-tutorial-part-2-snapshot-from-rest-depth) - Learn how to retrieve and use order book snapshots with Depth streams to track market liquidity and ...

52. [GitHub - 0liu/tardis-client: Tardis.dev cryptocurrency data client.](https://github.com/0liu/tardis-client) - Tardis.dev cryptocurrency data client. Contribute to 0liu/tardis-client development by creating an a...

53. [Emotion recognition with multiple physiological parameters ...](https://www.nature.com/articles/s41598-025-96616-0) - by Y Liao · 2025 · Cited by 7 — The WESAD dataset produces 8,973 three-channel physiological signal ...

54. [Investment Strategy 2024-2028 July 18, 2024 - BIO invests](https://www.bio-invest.be/files/BIO-invest/About-BIO/Governance/INVESTMENT-STRATEGY-2024-2028-Final.pdf) - Impact Finance Belgium association has further connected and promoted the ecosystem of impact. BIO, ...

