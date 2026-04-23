# BẢN GIẢI TRÌNH CƠ CHẾ NHÂN QUẢ  
## Tác nhân mua đáy / Phản xu hướng (Bottom-Fisher / Antigravity Agent)

---

## 1. Tổng quan về vai trò của Bottom-Fisher

Trong hệ sinh thái vi cấu trúc thị trường, **Bottom-Fisher** là tác nhân **duy nhất** cung cấp lực mua phản xu hướng (contrarian buy-side force) trong giai đoạn Flash Crash.

Khác với cả ba nhóm agent khác (MM rút thanh khoản, Momentum bán tháo, Noise khuếch đại), Bottom-Fisher nhận dạng:
- Giá đã **lệch quá xa** so với giá trị trung bình → cơ hội mean-reversion

Nền tảng lý thuyết dựa trên ba trụ cột:

> **DeBondt & Thaler (1985)**: Cổ phiếu giảm mạnh có xu hướng hồi phục do overreaction reversal  
> **Coates et al. (2014, PNAS)**: Dưới stress mãn tính, ~44% trader giảm risk-taking, nhưng Bottom-Fisher là nhóm **exception** — conviction-based override  
> **SciTePress ABM (2016)**: Recovery sau flash crash phụ thuộc vào "big fishes provide liquidity over first 200 ticks"

Vai trò trong DAG:

> **Antigravity** — Cung cấp negative feedback chống lại positive feedback loop (MM withdraw + MoM cascade). Nếu không có Bottom-Fisher, `gain_feedback > λ_OU` → hệ thống **bifurcate** → giá sập vô tận.

---

## 2. Bảng chú giải tín hiệu (Color Legend & Visual Semantics)

### 🔴 Biến can thiệp (Treatment - X)
- Stress sinh lý ($\sigma$)  
→ Nhưng tác động **khác biệt** so với 3 agent kia

---

### 🟢 Biến kết quả (Outcome - Y)
- Khối lượng lệnh mua phản xu hướng  
- Giá phục hồi sau crash

---

### ⚪ Biến nhiễu (Confounders - Z)
- $W_{t-1}, I_{t-1}, Vol_{t-1}$  
→ Cần control để tránh bias

---

### 🟡 Biến ngoại sinh / dị chất
- $\xi_i$, $\eta_L$  
→ Tạo **Heterogeneity** trong mức conviction

---

### 🔵 Biến trung gian (Mediators - M)
- $\delta_{entry}(\sigma)$: Ngưỡng entry tightened by stress
- $f_{stress}(\sigma)$: Stress dampening function cho position size
- $T_{max}(\sigma)$: Patience extends with stress

---

### 🟣 Market State & Memory
- $P_t$, $LOB_t$, $MA_N$  
- $\text{deviation}_t = (P_t - MA_N) / MA_N$

---

### 🩵 Tính toán nội tại
- RegimeClassifier output, ConvictionSizer, DislocDetect

---

### ⚫ Gates
- Solvency Gate (Margin Call): $W \le \frac{|I| \cdot \bar{P}}{L}$
- Panic Freeze Gate: $\sigma > \sigma_{panic} = 0.80$

---

## 3. Giai đoạn t-1: Nền tảng conviction

### Dị chất tác nhân ($\xi_i$)
- Mỗi Bottom-Fisher có mức conviction khác nhau  
- Conviction cao → thresholds thấp hơn, entry sớm hơn
- Conviction thấp → chờ lâu hơn, entry muộn hơn

### Mark-to-Market ($\pi_{t-1}$)
- Từ:
  - $I_{t-1}$ (vị thế mở hiện tại)
  - $P_t$ (giá thị trường)

### Stress sinh lý ($\sigma_{t-1}$)
$$
\sigma_{t-1} = f\left(\frac{\pi_{t-1}}{W_{t-1}}, \xi_i\right)
$$

**Đặc biệt**: Bottom-Fisher có baseline stress **thấp hơn** các agent khác:
- WESAD archetype: LOW arousal, MODERATE valence
- OU params: $\kappa = 0.08$, $\mu = 0.30$, $\sigma_{noise} = 0.04$

### Solvency Gate (DAG v2 — Margin Call)
$$
W_{t-1} \le \frac{|I_{t-1}| \cdot \bar{P}}{L}
$$

→ Agent bị **thanh lý bắt buộc** (không phải đợi W = 0)

---

## 4. Giai đoạn t: Cơ chế phản xu hướng

### Regime Classification (Bước đầu tiên — UNIQUE cho Bottom-Fisher)

Bottom-Fisher **KHÔNG** entry mù quáng. Trước hết phân loại regime:

$$
\text{Regime} = \begin{cases}
\text{FLASH\_CRASH} & \text{if spread\_spike AND depth\_collapse AND fast\_move} \\
\text{REGIME\_CHANGE} & \text{if gradual\_spread AND one\_sided\_flow} \\
\text{NORMAL} & \text{otherwise}
\end{cases}
$$

→ Chỉ activate khi **FLASH_CRASH** (dislocation tạm thời, không phải bear market thật)  
→ Tránh mua vào hố sâu trong regime change thực sự

### Phát hiện Price Dislocation

$$
\text{deviation}_t = \frac{P_t - MA_N}{MA_N}
$$

Entry condition:
$$
\text{deviation}_t < -\delta_{entry}(\sigma)
$$

Với:
$$
\delta_{entry}(\sigma) = \delta_{base} \cdot (1 + \alpha_{stress} \cdot \sigma)
$$

| Parameter | Value | Basis |
|-----------|-------|-------|
| $\delta_{base}$ | 0.015 (1.5%) | Jegadeesh (1990): intraday reversal threshold |
| $\alpha_{stress}$ | 0.8 | Stress tightens threshold → needs deeper crash |
| $N$ (lookback) | 20 ticks | Calibrated to flash crash duration (~200 ticks / 10 levels) |

### Stress Effect: Threshold Tightens (KHÁC với 3 agent kia)

Dưới stress, Bottom-Fisher **KHÔNG bỏ chạy** — thay vào đó:

1. **Entry threshold TIGHTENS**: $\delta_{entry} \uparrow$ → cần crash sâu hơn mới entry
2. **Position size SHRINKS**: $\text{size}(\sigma) = \text{size}_{base} \cdot \max(0.25, 1 - 0.5\sigma)$
3. **Patience EXTENDS**: $T_{max}(\sigma) = T_{base} \cdot (1 + 0.3\sigma)$

→ Bottom-Fisher trở nên **selective hơn** nhưng vẫn tham gia thị trường

**Exception**: Khi $\sigma > \sigma_{panic} = 0.80$:
- **Full trading freeze** — conviction override bị vượt quá
- Mô phỏng physiological shutdown ở cortisol cực cao
- Tạo dynamic thực tế: flash crash BÓ THỂ kéo dài nếu stress extreme

### Position Sizing — Scaling-in (3 Tranches)

| Tranche | % of Position | Entry Level |
|---------|--------------|-------------|
| 1 | 33% | At $\delta_{entry}$ |
| 2 | 33% | At $\delta_{entry} + 0.5\%$ |
| 3 | 34% | At $\delta_{entry} + 1.0\%$ (max dislocation) |

Basis: DeBondt-Thaler (1985) — value investors scale in, don't enter all-at-once  
Ref: Damodaran, Investment Fables Ch.8

---

## 5. Exit Conditions

### R3.1 — Mean Reversion Target
$$
\text{EXIT if } P_t \ge MA_N \times (1 - \delta_{exit})
$$
$\delta_{exit} = 0.003$ (0.3% — partial reversion acceptable)

### R3.2 — Stop-Loss
$$
\text{EXIT if unrealized PnL} < -0.025 \times \text{entry\_price} \times \text{size}
$$

### R3.3 — Time-Based Exit
$$
\text{EXIT if held} > T_{max} = 200 \text{ ticks}
$$
Basis: SciTePress (2016) — "liquidity returns within 200 best limit updates"

### R3.4 — Margin Call (DAG v2)
$$
\text{FORCED EXIT if } W \le \frac{|I| \cdot \bar{P}}{L}
$$

---

## 6. OU Stress Process (Bottom-Fisher Profile)

$$
d\sigma = \kappa(\mu - \sigma)dt + \sigma_{noise} \cdot dW
$$

| Parameter | Value | Basis |
|-----------|-------|-------|
| $\kappa$ | 0.08 | Slower mean-reversion than MM (0.05-0.07) — more stable |
| $\mu$ | 0.30 | Moderate baseline (WESAD low-arousal cluster centroid) |
| $\sigma_{noise}$ | 0.04 | Low noise — stable physiology |
| $\sigma_{panic}$ | 0.80 | Above this: full freeze |

Half-life: $t_{1/2} = \ln(2)/\kappa \approx 8.7\text{s}$

---

## 7. Kết luận

Bottom-Fisher **không phải contrarian đơn thuần**, mà là:

> **Cơ chế ổn định hệ thống (System Stabilizer)**

Flash Crash recovery phụ thuộc vào:
- Bottom-Fisher hoạt động (σ < σ_panic)  
- Đủ số lượng agent (N_BF > N_MM / 3)  
- Regime được phân loại đúng (flash crash, không phải regime change)

Nếu thiếu Bottom-Fisher:
- Feedback loop positive **không có đối trọng**
- $\text{gain}_{feedback} > \lambda_{OU}$ → bifurcation
- Giá **sập vô tận** → simulation invalid
