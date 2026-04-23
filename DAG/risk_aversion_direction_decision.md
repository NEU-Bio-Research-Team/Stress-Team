# Risk Aversion Direction Under Stress — Design Decision

> **Status**: ĐÃ QUYẾT ĐỊNH — Chạy cả 2 scenarios (Dual Scenario Protocol)  
> **Ngày quyết định**: 2026-04-12  
> **Phạm vi ảnh hưởng**: Tất cả 4 agent DAGs, simulation runner, counterfactual analysis

---

## 1. Vấn đề

Có mâu thuẫn giữa DAG specification hiện tại và literature review mới nhất:

### DAG Specification Hiện Tại (Layer 2.3)
- Stress ↑ → Risk aversion ↑ → Spread ↑, Depth ↓  
- Tức là: **σ↑ → γ↑** (traders sợ hãi hơn, thu hẹp hoạt động)

### Literature Contrary Finding
- **Kandasamy et al. (2014, PNAS)**: Cortisol injections **giảm** risk-taking 44% trong tài chính thí nghiệm  
- **NHƯNG**: Nghiên cứu trên nhóm **general population**, không phải professional traders  
- **Lo & Repin (2002)**: Professional traders có **habituation** — arousal KHÔNG giảm risk-taking ở mức cùng bằng
- **Starcke & Brand (2012, Neuroscience Review)**: Acute stress → **impaired decision-making** → risk-SEEKING (under hot cognition)
- **Porcelli & Delgado (2009, Psychological Science)**: Acute stress → risk-seeking cho **gains**, risk-averse cho **losses** (Prospect Theory modulated by stress)

### Tổng hợp
Literature cho thấy **KHÔNG CÓ consensus**. Hướng tác động phụ thuộc vào:
1. **Loại stress**: Acute (vài phút) vs Chronic (vài giờ/ngày)
2. **Loại trader**: Professional (habituated) vs Retail (naive)
3. **Framing**: Gain domain vs Loss domain
4. **Emotion**: Fear/Sad → risk-averse vs Anger → risk-seeking

---

## 2. Quyết định: Dual Scenario Protocol

### Scenario A: "Fear Response" (σ↑ → γ↑)
- **Khi nào áp dụng**: Slowly building stress (chronic cortisol)
- **Cơ chế**: Kandasamy (2014) — cortisol reduces risk appetite
- **Agent behavior**: MM widens spread, MoM reduces position, Noise freezes
- **Kỳ vọng**: Flash crash do **rút thanh khoản** (liquidity vacuum)

### Scenario B: "Anger Response" (σ↑ → γ↓)  
- **Khi nào áp dụng**: Acute spike stress (adrenaline burst)
- **Cơ chế**: Starcke & Brand (2012) — hot cognition overrides rational assessment
- **Agent behavior**: MM tightens spread nhưng overcommits, MoM increases bet size, Noise panic-sells
- **Kỳ vọng**: Flash crash do **aggressive selling** (cascade selling)

### Implementation

```python
# config/ou_params_by_trader_type.json — mỗi agent type có 2 scenarios
{
    "market_maker": {
        "scenario_A": {
            "gamma_stress_coeff": 0.8,   # γ(σ) = γ_0 * (1 + 0.8σ)  → tăng risk aversion
            "description": "Fear response: stress increases risk aversion"
        },
        "scenario_B": {
            "gamma_stress_coeff": -0.4,  # γ(σ) = γ_0 * (1 - 0.4σ)  → giảm risk aversion
            "description": "Anger response: stress decreases risk aversion"
        }
    }
}
```

### Simulation Protocol
1. Run Scenario A 100 lần (random seeds) → collect distributions
2. Run Scenario B 100 lần (same seeds) → collect distributions
3. Compare:
   - Crash frequency (A vs B)
   - Crash severity (max drawdown)
   - Recovery time
   - Stylized facts compliance
4. **Report CẢ HAI** results — đây là contribution chính: "direction matters"

---

## 3. DAG Update Required

### Existing DAGs chỉ encode Scenario A:
- `MM_explaination.md` line 99: "$\Delta x_t \uparrow$" (spread widens under stress)
- `MT_explaination.md` line 144: "$\alpha_t \downarrow$" (aggressiveness decreases)
- `NT_explaination.md` line 140: "$\alpha_t \downarrow$" (same)

### Cần thêm annotation trong mỗi DAG:
> **Ghi chú DAG v2**: Mediator direction phụ thuộc scenario (A: Fear → γ↑, B: Anger → γ↓). Xem `risk_aversion_direction_decision.md` cho chi tiết.

**Quyết định**: KHÔNG thay đổi diagram structure (giữ nguyên arrows), chỉ thêm annotation text ở phần giải trình.

---

## 4. Literature References

| Citation | Key Finding | Scenario |
|----------|-------------|----------|
| Kandasamy et al. (2014, PNAS) | Cortisol ↑ → risk-taking ↓ 44% | A |
| Starcke & Brand (2012, NeuroReview) | Acute stress → risk-seeking (hot cognition) | B |
| Porcelli & Delgado (2009, PsychSci) | Stress + losses → risk-averse; stress + gains → risk-seeking | Mixed |
| Lo & Repin (2002, JCF) | Professional traders habituate to stress | Attenuated |
| Coates et al. (2014, PNAS) | Chronic cortisol → conservative strategies | A |
| Prospect Theory (1979) | Loss domain → risk-seeking (reflection effect) | B for losses |

---

## 5. Impact on Bottom-Fisher

Bottom-Fisher agent is **LESS affected** by direction choice because:
- BF activates **post-crash** (not during stress escalation phase)
- BF stress profile is LOW arousal (OU μ=0.30)
- BF stress effect is on **threshold** and **sizing**, not on **direction**

Both scenarios A and B keep BF as buy-side stabilizer — the main difference is **when** BF activates (earlier in Scenario A due to faster MM withdrawal, later in Scenario B due to later price dislocation).

---

## 6. Kết luận

> **"We don't know which direction is correct — and that's the point."**

Running both scenarios and comparing outputs IS the scientific contribution. If Scenario A and B produce similar crash dynamics (convergent behavior), it suggests the feedback loop structure matters more than individual risk-aversion direction. If they diverge, it identifies risk-aversion direction as a critical parameter requiring empirical calibration from real-trader physiological data.
