# Noise Trader — Agent Profile (Baseline / No Stress)

---

## Archetype Summary

The Noise Trader is a **low-frequency, uninformed liquidity taker**. It does not provide liquidity, nor does it extrapolate trends. 

Its core behavioral thesis: Market movements are fundamentally random to those without private information. It generates trading signals based on statistical noise rather than structural edge. By continuously submitting aggressive market orders based on pseudo-random price expectations, it acts as the baseline "fuel" of the market, providing the steady volume and liquidity premium (spread) that sustains Market Makers.

---

## Core Behavioral Traits

### Chronological Order Arrival (Poisson Process)

Unlike High-Frequency Traders (HFTs) that are event-driven, the Noise Trader operates on **clock time**. Its order submissions follow an independent Poisson process. 
- It acts at randomized intervals driven by a baseline arrival intensity ($\lambda_L$).
- It is structurally blind to micro-events (e.g., it does not wake up just because a flash crash starts).

### Volatility-Bounded Randomness (Equation 17)

The agent’s expected future return is mathematically structured. It is drawn from a standard normal distribution $N(0,1)$, scaled by the market's historical volatility (`realized_vol_50`), and critically adjusted by a leveling coefficient $\sqrt{2/\pi}$:

E_noise[r] = √(2/π) * N(0,1) * realized_vol_50
Expected_Price = mid_price * (1 + E_noise[r])

*Note: The $\sqrt{2/\pi}$ coefficient is strictly required to preserve the principle that traders believe historical volatility is the best proxy for future volatility; without it, the absolute expectation of noise would systematically overestimate actual volatility.*

### Demand Formulation (Price Gap vs. Bid/Ask)

The agent calculates its desired volume based on the gap between its expected price and the *executable* side of the order book (not just the mid-price).
- **If Buying ($E[P] > best\_ask$):** `gap = (Expected_Price - best_ask) / best_ask`
- **If Selling ($E[P] < best\_bid$):** `gap = (best_bid - Expected_Price) / best_bid`

raw_qty = aggressiveness_alpha * gap

### Directional Inventory Penalty (Mean-Reversion)

The agent is subject to a strict, directional inventory control mechanism ($\Omega_t$). Unlike trend followers, its penalty is linearly proportional to the absolute inventory size and current price, without wealth normalization:
Omega_t = v_g * |inventory_t| * mid_price_t

Crucially, this penalty is applied **asymmetrically** to force mean-reversion toward zero inventory:
- If holding a **Long** position: Buy volume is reduced (`raw_buy_qty - Omega_t`), while Sell volume is amplified (`raw_sell_qty + Omega_t`).
- If holding a **Short** position: Sell volume is reduced (`raw_sell_qty - Omega_t`), while Buy volume is amplified (`raw_buy_qty + Omega_t`).

### Order Type & Latency

The agent uses **market orders exclusively**, acting as an aggressive taker.
It operates with high latency ($\eta_L \gg \eta_H$). It submits orders blindly based on delayed states of the LOB, unaware of the microsecond adjustments HFTs are making.

---

## Market State Sensitivity (100ms Grid)

Unlike Momentum or Market Maker agents, the Noise Trader is defined by its *lack* of sensitivity to most microstructure signals:

| Signal | Agent Response |
|--------|---------------|
| `realized_vol_50` high | **Highly Sensitive:** Expands the magnitude of its random price expectations via Equation 17. |
| `ofi` (positive or negative) | **Ignored:** Does not read order flow imbalance. |
| `spread_bps` wide | **Ignored:** Prioritizes execution over cost; blindly accepts market price. |
| `touch_depth` | **Ignored:** Does not check liquidity before firing market orders. |

---

## Risk Controls

The agent enforces cascading risk layers:

1. **Inventory Gating** — The directional $\Omega_t$ continuously dampens trades that increase exposure and amplifies trades that reduce it.
2. **Solvency Gating (Margin Call)** — A strict leverage-adjusted maintenance margin rule forces default before total wealth hits zero:
if W_{k,t} ≤ (|inventory_t| * mid_price_t) / Leverage_Factor → Margin Call → Default

---

## Role in Flash Crash Dynamics

The Noise Trader is the **random spark** in the simulation.
1. During normal regimes, its Poisson arrival rate generates safe, mean-reverting volume.
2. It does not actively herd or panic during a crash.
3. However, if a Noise Trader randomly draws a massive sell expectation (large negative $N(0,1)$ coupled with high `realized_vol_50`), it may fire a massive market sell order.
4. If this coincides with thin `touch_depth`, it triggers a localized price drop, which Momentum Traders then extrapolate to ignite algorithmic panic.

---

## Parameters at a Glance

| Parameter | Source / Nature | Default / Range |
|-----------|-----------------|-----------------|
| `arrival_rate_lambda` ($\lambda_L$) | Config (Poisson Process) | e.g., 0.1 orders/sec |
| `aggressiveness_alpha` ($\alpha$) | Phase 1 LLM Elicitation | Fixed scaling multiplier |
| `inventory_sensitivity_vg` ($v_g$) | Phase 1 LLM Elicitation | Gamma(shape, scale) |
| `leverage_factor` ($L$) | Config | Defines Margin buffer |
| `latency_penalty` ($\eta_L$) | Microstructure Setup | $\eta_L \gg \eta_H$ |

---

## Academic Grounding

| Trait | Reference |
|-------|-----------|
| Expected Return Formula with $\sqrt{2/\pi}$ | Witte (2012), Equation 17 |
| Demand formulation against Bid/Ask | Witte (2012), Equations 19 & 20 |
| Market Order execution & Baseline Demand | Leal et al. (2014) |
| Poisson Process Arrival (Clock-time) | Cartlidge & Shi (2023) |
| Margin-based Solvency constraint | Bookstaber et al. (2014) |