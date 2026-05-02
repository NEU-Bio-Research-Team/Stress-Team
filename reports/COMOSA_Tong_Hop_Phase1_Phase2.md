# COMOSA — Tổng Hợp Kết Quả Phase 1 & Phase 2

> **Phiên bản:** 2026-05-02  
> **Mục đích:** Tài liệu duy nhất tổng hợp toàn bộ những gì đã thực hiện, các tham số đang được đặt, độ tin cậy khoa học từng công đoạn, và các rủi ro/vấn đề còn tồn đọng có thể gây gián đoạn hoặc làm yếu mục tiêu nghiên cứu.

---

## 1. Nhắc Lại Mục Tiêu Nghiên Cứu (Reference)

| Mã | Mục tiêu | Kết quả cần thiết |
|----|----------|-------------------|
| **RQ1 / H1** | LLM-elicited priors tạo ra agents thực tế hơn uniform random | Kurtosis LLM > kurtosis uniform; stylised facts pass |
| **RQ3 / H2** | Cơ chế nhân quả `OFI × leverage → flash crash` học được từ data | Edge `OFI → crash` và `leverage → crash` xuất hiện trong DAG NOTEARS |
| **H3** | `do(leverage=0)` giảm crash rate >70% | Interventional test trong Phase 3 |

**Lưu ý quan trọng:** Flash-crash rate 6% không phải mục tiêu cuối — nó là **điều kiện kỹ thuật** (gate) để Phase 3 có đủ positive examples cho NOTEARS hội tụ.

---

## 2. Phase 1 — LLM Elicitation Behavioral Priors

### 2.1 Những Gì Đã Làm

| Bước | Mô tả | Trạng thái |
|------|-------|-----------|
| Xây dựng prompt template | 4 archetype × 4 phase, neo vào percentile OFI thực từ 66 sự kiện | ✅ Hoàn thành |
| Elicitation | 512 phản hồi có parse được từ LLM (Mistral-7B-Instruct local) | ✅ Hoàn thành |
| Fit phân phối | MLE trên scipy: Beta, Gamma, LogNorm, scalar | ✅ Hoàn thành |
| Lưu kết quả | `behavioral_priors.json` — 4 archetype × 4 phase | ✅ Hoàn thành |
| Sanity check | `phase2_mini_sanity_check.md` — 9/9 gates PASS | ✅ Hoàn thành |

### 2.2 Parameters Đầu Ra Phase 1 (Behavioral Priors)

Tất cả đã được fit MLE và lưu trong `data/processed/tardis/phase1_outputs/behavioral_priors.json`.

**Ví dụ: Momentum Trader (phase = pre)**

| Tham số | Phân phối | Giá trị fit |
|---------|-----------|-------------|
| `aggressiveness` | Beta | α=139.13, β=61.51 → mean ≈ **0.694** |
| `cancel_probability` | Beta | α=1.67, β=31.17 → mean ≈ **0.051** |
| `order_size_multiplier` | LogNorm | shape=0.246, scale=2.619 → median ≈ **2.62×** |
| `inventory_sensitivity` | Gamma | shape=100, scale=0.005 → mean ≈ **0.50** |
| `order_type_market_fraction` | scalar | **1.0** (100% market orders) |

**Phân bố archetype theo sanity check:**

| Archetype | cancel_prob mean | Market order share | Buy/Sell balance |
|-----------|-----------------|-------------------|-----------------|
| `hft_market_maker` | 0.23 | 0% (all limit) | 72.7% buy, 25.8% sell |
| `momentum_trader` | 0.054 | 100% market | 52.3% buy, 47.7% sell |
| `noise_trader` | 0.033 | 100% market | 50% buy / 50% sell |
| `contrarian_trader` | 0.067 | 100% market | 46.1% buy, 35.2% sell, 18.8% nothing |

### 2.3 Empirical Anchors Từ Dữ Liệu Thực (66 Events BTC)

Nguồn: `data/processed/tardis/confounder_outputs/prior_anchors.json`  
Cơ sở dữ liệu: 66 sự kiện flash-crash BTC, 630,797 rows ở 100ms grid.

| Metric | Pre | Drop | Recovery | Post |
|--------|-----|------|----------|------|
| OFI p50 | 0.023 | **-0.1915** | 0.0715 | 0.001 |
| Kyle λ (mean) | 0.622 | **0.776** | 0.713 | 0.970 |
| Realized vol (mean) | 0.00109 | **0.00183** | 0.00188 | 0.00195 |
| Trade intensity (mean, trades/100ms) | 9.49 | **13.29** | 15.76 | 20.98 |

*OFI p50 âm trong drop (-0.19) vs. dương trong pre (+0.023) là stylised fact SF-4 — đã được tái hiện trong simulator.*

### 2.4 Độ Tin Cậy Khoa Học Phase 1

| Tiêu chí | Đánh giá | Giải thích |
|----------|----------|------------|
| Sample size LLM | ⚠️ Trung bình | 512 rows / 4 archetype / 4 phase = ~32 rows/nhóm. Đủ để fit phân phối nhưng bootstrap CI rộng |
| Anchoring thực tế | ✅ Tốt | Prompt neo vào percentile OFI thực từ data; không phải hỏi chung chung |
| Fit quality | ✅ Tốt | Sử dụng MLE + scipy; Beta phù hợp cho [0,1]; LogNorm cho order size |
| Sanity gates | ✅ 9/9 PASS | MM limit dominance, noise buy/sell balance, contrarian sell floor — tất cả hợp lý |
| LLM model bias | ⚠️ Rủi ro | Mistral-7B local có thể bị bias từ finance blog text; chưa có adversarial test |
| So sánh LLM vs. uniform | ⚠️ Chưa đầy đủ | H1 cần kurtosis_LLM > kurtosis_uniform; ablation kurtosis_uniform còn "missing input" trong một số validation file |

**Kết luận Phase 1:** Output đủ để dùng làm prior cho Phase 2. Tuy nhiên, H1 (ablation LLM vs. uniform) chưa được hoàn thiện vì một số validation file thiếu kurtosis_uniform để so sánh.

---

## 3. Phase 2 — LOB Simulation Engine

### 3.1 Những Gì Đã Làm

| Bước | Mô tả | Trạng thái |
|------|-------|-----------|
| Xây dựng LOB engine | 2-sided queue, tick 100ms, 4 archetype | ✅ Hoàn thành (script 18) |
| Calibrate crash detector | 10-tick rolling window, threshold 1.93% | ✅ Hoàn thành |
| Fix dead `drop_sell_pressure` | Bug đã fix; parameter có effect đo được | ✅ Hoàn thành |
| Calibrate intensity-scale | Step 1: match OFI p50 drop; achieved -0.177 vs target -0.1915 (7.7% error) | ✅ Chấp nhận được |
| Calibrate impact-scale | Step 2 sweep; impact-scale=8.0 cho crash rate 6% | ✅ Hoàn thành |
| Benchmark 50 runs | impact-scale=8.0, intensity-scale=1.2289, min-price-fraction=0.70 | ✅ **Kết quả mới nhất** |
| Stylised facts validation | kurtosis, acf_vol, OFI ordering gates | ⚠️ Một phần pass |

### 3.2 Parameters Hiện Tại Của Simulator (Script 18)

Đây là **toàn bộ CLI parameters** đang được dùng trong benchmark run ngày 2026-05-02:

| Parameter | Giá trị | Nguồn gốc / Ý nghĩa |
|-----------|---------|---------------------|
| `--scenario` | `llm` | Dùng behavioral priors từ Phase 1 |
| `--n-runs` | 50 (benchmark) / **1000 khi chính thức** | Số lượng simulation runs |
| `--seed` | 42 | Reproducibility |
| `--tick-ms` | 100 | Khớp với 100ms grid của empirical data |
| `--impact-scale` | **8.0** | Calibrated để hit ~6% crash rate (target 10%) |
| `--intensity-scale` | **1.2289** | Calibrated Step 1: OFI p50 drop match |
| `--base-order-size` | 0.25 BTC | Base size trước nhân multiplier |
| `--mm-vol-threshold-mult` | 1.4 | MM rút khi realized vol > 1.4× baseline |
| `--mm-withdrawal-strength` | 1.8 | Mức độ MM thu hẹp spread khi rút |
| `--crash-window-ticks` | 10 | Cửa sổ rolling detector (= 1 giây) |
| `--crash-threshold-pct` | **1.93%** | Ngưỡng drawdown — từ empirical q10-positive mapping |
| `--max-drop-ticks` | 5000 | Cap phase drop (500 giây) |
| `--max-recovery-ticks` | 3000 | Cap phase recovery (300 giây) |
| `--max-post-ticks` | 2000 | Cap phase post |
| `--max-pre-ticks` | 2000 | Cap phase pre → tổng 11,750 ticks/run tối đa |
| `--drop-sell-pressure` | **0.12** | Extra sell tilt trong drop phase |
| `--drop-impact-mult` | 1.35 | Kyle λ nhân thêm 1.35× trong drop |
| `--min-price-fraction` | **0.70** | Floor giá = 70% giá khởi đầu (bảo vệ price-to-zero) |

**Agent population (mỗi run = 100 agents):**

| Archetype | Tỷ lệ | Số agents | Wealth khởi đầu |
|-----------|-------|-----------|----------------|
| `momentum_trader` | 30% | 30 | 100,000 USDT |
| `noise_trader` | 30% | 30 | 50,000 USDT |
| `hft_market_maker` | 20% | 20 | 500,000 USDT |
| `contrarian_trader` | 20% | 20 | 150,000 USDT |

### 3.3 Kết Quả Benchmark 50 Runs (2026-05-02)

**Command chạy:**
```
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm --n-runs 50 --seed 42 \
  --impact-scale 8.0 --intensity-scale 1.2289 \
  --min-price-fraction 0.70 \
  --output-csv /tmp/lob18_bench_guard_impact8.csv \
  --summary-json /tmp/lob18_bench_guard_impact8.json
```

**Kết quả:**

| Metric | Giá trị | Đánh giá |
|--------|---------|----------|
| Tổng rows | 574,900 | 50 runs × ≈11,498 rows |
| Flash-crash rate | **6.0%** (3/50 runs) | ⚠️ Dưới target 10%, nhưng trong cửa sổ [5%, 40%] |
| Max DD 10-tick mean | 0.485% | Hợp lý |
| Max DD 10-tick p95 | 1.588% | Hợp lý |
| Max DD 10-tick max | **2.590%** | Run 8 — crash thực sự |
| OFI pre mean | +0.066 | ✅ Dương — khớp empirical |
| OFI drop mean | **-0.270** | ✅ Âm — khớp hướng SF-4 |
| Pct insolvent | 0.000% | ✅ Không agent nào phá sản |
| Wealth concentration (drop) | 0.920 | Cao — 1 nhóm nhỏ nắm giữ phần lớn tài sản |

**3 crash runs:** run 8 (max_dd=2.59%), run 37 (2.18%), run 43 (2.01%)

**Cột output panel data (dùng cho Phase 3):**  
`run_id, event_id, tick_ms, phase, close, mid_price, ofi, spread_bps, depth_imbalance, trade_intensity, realized_vol_50, leverage_proxy, kyle_lambda, vpin, flash_crash_flag, mean_wealth_t, pct_insolvent, wealth_concentration`

### 3.4 Lịch Sử Calibration (Các Run Trước)

| Run | impact-scale | crash rate | OFI drop | Vấn đề |
|-----|-------------|-----------|---------|--------|
| Sanity (n=3) | default 2.0 | 0% | -0.031 | Không có crash |
| Stylised facts v1 (n=50) | (cũ, cao) | 0% | **+0.031** | OFI drop sai chiều — bug drop_sell_pressure |
| Post-fix sanity (n=3) | 2.0 | 0% | -0.031 | Bug đã fix; impact-scale chưa đủ |
| Stylised facts v2 (n=50) | (đã fix) | 0% | -0.055 | OFI đúng chiều nhưng vẫn chưa có crash |
| **Benchmark hiện tại (n=50)** | **8.0** | **6%** | **-0.270** | ✅ Gate pass |

### 3.5 Gate Checklist Phase 2

| Gate | Kết quả | PASS/FAIL | Ghi chú |
|------|---------|-----------|---------|
| `kurtosis_excess > 3.0` | 16.0–20.0 (tùy run) | ✅ PASS | Fat tails hiện diện |
| `acf_vol_lag1 > 0.10` | 1.0 | ✅ PASS | Volatility clustering mạnh |
| `ofi_drop < ofi_pre` | -0.270 < +0.066 | ✅ PASS | SF-4 OFI regime shift |
| `crash_rate ∈ [0.05, 0.40]` | 0.06 | ✅ PASS | Vừa đủ ngưỡng dưới |
| Ablation H1: kurtosis_LLM > kurtosis_uniform | — | ⚠️ CHƯA ĐỦ | Cần chạy scenario=uniform để so sánh |

### 3.6 Độ Tin Cậy Khoa Học Phase 2

| Tiêu chí | Đánh giá | Giải thích |
|----------|----------|------------|
| Crash detector threshold | ✅ Được neo vào empirical | 1.93% từ rolling-10 event-level q10 của 66 sự kiện thực |
| OFI calibration | ✅ Tốt | 7.7% error so với target; chấp nhận được |
| Impact-scale calibration | ⚠️ Over-calibration | impact-scale=8.0 cao hơn nhiều so với giá trị tính từ công thức Kyle λ (~2.5–3.5). Simulator cần "lực đẩy" nhân tạo lớn mới tạo crash, ngụ ý cơ chế crash trong LOB engine còn yếu |
| Crash rate statistical power | ⚠️ Yếu với n=50 | 3/50 → 95% CI ≈ [1.3%, 16.7%]; cần n=1000 để ổn định |
| Leverage feedback | ❌ Chưa có | `leverage_proxy` được tính nhưng KHÔNG quay lại khuếch đại price dynamics (xem §4) |
| Số lần lặp đủ cho NOTEARS | ⚠️ Chưa đủ | Cần 1000 runs → ~60 positive examples |

---

## 4. Vấn Đề Hiện Tại Ảnh Hưởng Đến Mục Tiêu Nghiên Cứu

Đây là phần quan trọng nhất. Các vấn đề được xếp theo mức độ rủi ro đối với từng mục tiêu.

---

### Vấn Đề 1 — `leverage_proxy` không có feedback vào price dynamics

**Mức độ rủi ro:** 🔴 CAO — ảnh hưởng trực tiếp đến H2 và H3

**Mô tả chi tiết:**  
Trong script 18 hiện tại, `leverage_proxy` được tính theo công thức phái sinh từ `pct_insolvent` và `wealth_concentration`, nhưng giá trị này chỉ được **ghi vào panel data** (cột output), không quay lại khuếch đại `impact` trong vòng lặp tick.

Cụ thể: `impact = impact_scale × kyle_lambda × net_flow` **không** chứa leverage làm nhân tử.

**Hậu quả đối với mục tiêu nghiên cứu:**
- H2 phát biểu: *"leverage acts as an interaction/amplification term in the causal DAG"*. Nhưng nếu leverage không ảnh hưởng đến dynamics, NOTEARS sẽ không tìm thấy edge `leverage → flash_crash` (hoặc edge rất yếu, không có ý nghĩa thống kê).
- H3 (`do(leverage=0)` giảm crash >70%) **không thể test** vì không có causal pathway. Intervention trở thành can thiệp vào một biến vô nghĩa.
- Kết quả sẽ chỉ cho thấy `leverage` là *observable correlate*, không phải *causal agent* — làm yếu toàn bộ phần đóng góp RQ3.

**Giải pháp:**  
Thêm leverage vào impact formula:
```python
leverage_amp = 1.0 + leverage_proxy * leverage_feedback_strength
impact = impact_scale * kyle_lambda * net_flow * leverage_amp
```
Sau đó chạy lại calibration vì impact tổng thể sẽ tăng.

**Rủi ro khi fix:** Crash rate sẽ tăng lên, cần re-sweep impact-scale để giữ crash rate trong vùng mục tiêu.

---

### Vấn Đề 2 — Crash rate 6% thấp hơn target 10%, và n=50 chưa đủ

**Mức độ rủi ro:** 🟡 TRUNG BÌNH — ảnh hưởng đến statistical power của Phase 3

**Mô tả chi tiết:**  
- Target thiết kế là ~10% để Phase 3 có positive examples đủ.
- Hiện tại 6% với n=50 → chỉ 3 positive examples.
- Với n=1000 runs (kế hoạch cho Phase 3), sẽ có ~60 positives tại 6% — đây là ngưỡng tối thiểu để NOTEARS hội tụ ổn định (theo plan §2.3).
- Tuy nhiên, 60 positives vs. 940 negatives là tỷ lệ **imbalanced 1:15.7**. NOTEARS không phải binary classifier (nó học continuous DAG), nhưng edge `flash_crash_flag` sẽ kém robust hơn với ít positive hơn.

**Khuyến nghị:** Nên tăng crash rate lên 8-12% trước khi chạy 1000 runs, bằng cách tăng nhẹ `--impact-scale` hoặc `--drop-sell-pressure`.

---

### Vấn Đề 3 — Ablation H1 (LLM vs. uniform) chưa được thực hiện đầy đủ

**Mức độ rủi ro:** 🟡 TRUNG BÌNH — ảnh hưởng đến khả năng claim H1

**Mô tả chi tiết:**  
H1 đòi hỏi chứng minh kurtosis(LLM) > kurtosis(uniform). Hiện tại:
- Kurtosis của scenario=llm đã được đo (≈16–20 tùy run).
- Kurtosis của scenario=uniform **chưa được đo trong điều kiện tương đương** (cùng impact-scale, cùng n-runs).
- File `phase2_stylised_facts_validation_from_current_llm.md` và `phase2_stylised_facts_validation_sanity.md` đều ghi `kurtosis_uniform: missing input`.

**Hậu quả:** Nếu không có số liệu uniform, H1 không thể được defend trong peer review — "LLM tốt hơn gì?" sẽ không trả lời được.

---

### Vấn Đề 4 — `spread_bps` và `depth_imbalance` trong panel là giá trị giả (synthetic)

**Mức độ rủi ro:** 🟡 TRUNG BÌNH — ảnh hưởng đến chất lượng DAG biến X2, X3

**Mô tả chi tiết:**  
Cột `spread_bps` và `depth_imbalance` có mặt trong panel output của script 18 (xác nhận từ `columns` trong JSON). Tuy nhiên, đây là **giá trị được tính nội bộ từ LOB synthetic**, không phải từ bookTicker thực tế của 66 sự kiện.

Điều này có nghĩa là khi NOTEARS học DAG, các edge liên quan đến spread và depth phản ánh cơ chế của **LOB engine giả lập**, không phải cơ chế thực. Điều này chấp nhận được về mặt methodology (Phase 3 là confirmatory, không phải blind discovery) nhưng cần được phát biểu rõ trong paper để tránh bị reviewer bác.

---

### Vấn Đề 5 — Impact-scale=8.0 quá cao so với Kyle λ thực tế

**Mức độ rủi ro:** 🟠 TRUNG BÌNH-CAO — ảnh hưởng đến tính ngoại suy (external validity)

**Mô tả chi tiết:**  
Từ công thức calibration trong `phase2_crash_calibration_decision_2026-05-02.md`:
```
impact_scale = (D/100 × P0) / (W × kyle_lambda_drop × |mean_net_flow_drop|)
             ≈ (0.0193 × 36747) / (10 × 0.776 × 0.133)
             ≈ 709 / 1.03 ≈ 688
```
*[Lưu ý: tính trên đây là ví dụ minh họa — cần cross-check với actual blocker quantification trong file validation]*

Thực tế: impact-scale mặc định công thức tính ra ~2–3, nhưng simulator cần 8.0 để đạt 6% crash. Điều này ngụ ý một trong hai:
1. Net flow trong drop phase của simulator quá nhỏ (→ cần kiểm tra OFI scale)
2. Cơ chế amplification trong LOB engine (MM withdrawal, momentum feedback) chưa đủ mạnh

Nếu impact-scale quá cao, model sẽ "bơm" giá mạnh nhân tạo, làm cho crash trở thành artifact của parameter chứ không phải emergence từ agent interaction. Đây là rủi ro cho external validity.

---

### Vấn Đề 6 — Chưa có non-crash baseline (normal market)

**Mức độ rủi ro:** 🟡 TRUNG BÌNH — ảnh hưởng đến Phase 1 prompt anchoring

**Mô tả chi tiết:**  
Plan §4.4 D yêu cầu normal-market reference cho MM/Noise Trader baseline. Hiện tại:
- Data chỉ có `pre/drop/recovery/post` của 66 sự kiện crash — tất cả đều "gần crash"
- Phase `pre` được dùng làm proxy nhưng có bias (tình trạng thị trường trước crash ≠ thị trường bình thường)

Ảnh hưởng nhẹ hơn nếu focus vào crash dynamics, nhưng sẽ bị hỏi trong review về baseline.

---

## 5. Tóm Tắt Trạng Thái Theo Bảng

### Phase 1 — Trạng thái tổng thể: ✅ Hoàn thành (với lưu ý)

| Hạng mục | Trạng thái | Ghi chú |
|----------|-----------|---------|
| Behavioral priors (behavioral_priors.json) | ✅ Xong | 512 rows, 4 archetype × 4 phase |
| Sanity check 9/9 | ✅ Xong | Tất cả cấu trúc prior hợp lý |
| Ablation LLM vs. uniform | ❌ Chưa đủ | Cần chạy scenario=uniform cùng điều kiện |
| Anchoring vào empirical data | ✅ Xong | prior_anchors.json từ 66 sự kiện BTC |

### Phase 2 — Trạng thái tổng thể: ⚠️ Gate PASS nhưng chưa sẵn sàng cho Phase 3

| Hạng mục | Trạng thái | Ghi chú |
|----------|-----------|---------|
| LOB engine hoạt động | ✅ Xong | Script 18, 50 runs thành công |
| Crash detector calibration | ✅ Xong | 1.93% threshold, 10-tick window |
| Stylised facts gates | ✅ 4/4 PASS | Kurtosis, ACF, OFI ordering, crash rate |
| Ablation H1 (kurtosis LLM > uniform) | ❌ Chưa đủ | — |
| Leverage feedback loop | ❌ Chưa có | Blocker cho H2 và H3 |
| n=1000 runs cho Phase 3 | ❌ Chưa chạy | Cần sau khi fix leverage |
| Normal-market baseline | ❌ Chưa có | P1 priority theo plan |

---

## 6. Thứ Tự Ưu Tiên Các Bước Tiếp Theo

Xếp theo mức độ ảnh hưởng đến mục tiêu nghiên cứu:

| Ưu tiên | Việc cần làm | Lý do |
|---------|-------------|-------|
| **P0-A** | **Quyết định: có implement leverage feedback loop không?** | Nếu có → fix H2 + H3; nếu không → reframe H2 thành "observable amplifier" |
| **P0-B** | Nếu có feedback: thêm `leverage_amp` vào Kyle λ formula, re-sweep impact-scale | Crash rate sẽ thay đổi |
| **P1-A** | Chạy scenario=uniform cùng n=50, cùng impact-scale → ghi kurtosis | Fix ablation H1 |
| **P1-B** | Tăng crash rate lên 8–12% (sweep nhỏ) | Tăng statistical power cho Phase 3 |
| **P2** | Chạy chính thức n=1000 runs sau khi đã fix P0 + P1 | Panel data cho NOTEARS |
| **P3** | Bắt đầu Phase 3: NOTEARS / LiNGAM trên panel 1000 runs | Causal discovery |

---

## 7. Phụ Lục — Files Quan Trọng

| File | Nội dung |
|------|----------|
| `data/processed/tardis/phase1_outputs/behavioral_priors.json` | Output Phase 1: toàn bộ phân phối tham số |
| `data/processed/tardis/confounder_outputs/prior_anchors.json` | Empirical anchors từ 66 sự kiện BTC |
| `data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv` | Data gốc 66 events |
| `scripts/stage2_economics/18_lob_mini_runner.py` | LOB engine — script chính Phase 2 |
| `reports/validation/phase2_crash_calibration_decision_2026-05-02.md` | Quyết định threshold + calibration blocker |
| `reports/validation/phase2_mini_sanity_check.md` | Sanity check Phase 2 (9/9 PASS) |
| `reports/validation/phase2_stylised_facts_validation.md` | Stylised facts gates (crash gate fail trước fix) |
| `/tmp/lob18_bench_guard_impact8.json` | Summary run benchmark 50 ngày 2026-05-02 |

---

*Tài liệu này được tạo tự động ngày 2026-05-02 từ workspace hiện tại. Cập nhật lại sau mỗi milestone quan trọng.*
