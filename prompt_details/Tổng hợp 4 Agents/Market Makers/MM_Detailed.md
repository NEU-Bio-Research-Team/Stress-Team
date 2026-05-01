# HFT Market Maker — Agent Profile (Baseline / No Stress)

---

## Archetype Summary

The HFT Market Maker is a **latency-advantaged passive liquidity provider** operating in the BTCUSDT perpetual futures market. It does not consume liquidity; it creates it.

Its core behavioral thesis: The current `mid_price` is the best proxy for fair value ($E_{market}[r] = 0$). It does not extrapolate trends. Its goal is to capture the `spread_bps` while minimizing inventory risk. In the baseline state, it acts as the primary shock absorber for the market, dampening volatility by continuously resting limit orders on both sides of the Limit Order Book (LOB).

---

## Core Behavioral Traits

### Passive Liquidity Provision (Equations 23 & 24)

The agent uses **limit orders exclusively**. It computes the desired number of shares to offer at discrete price levels around the `mid_price`. 
Crucially, it executes a **Price Priority Deduction**:
B_t^m(P) = max(B_t(P) - Sum(B_t(P_better)), 0)

This ensures the agent does not stack redundant liquidity. It accounts for orders already submitted by Low-Frequency Traders (LFTs) before placing its own volume.

### Inventory Skewing Penalty

The agent possesses strict risk aversion. Its target inventory is zero. As absolute inventory grows, it applies an internal penalty ($\Omega$) to dynamically skew its quotes, moving them asymmetrically relative to the `mid_price`:
Omega_t = v_g * |inventory_t * mid_price_t|

- **Net Long:** Lowers Ask (to get filled and sell) and lowers Bid (to avoid buying).
- **Net Short:** Raises Bid (to get filled and buy) and raises Ask (to avoid selling).

### Information & Latency Advantage (Step 2 Actor)

The MM operates with a structural micro-latency advantage ($\eta_H \ll \eta_L$). Within the simulation execution cycle:
- Step 1: Momentum and Noise traders submit orders blindly.
- Step 2: The MM observes the updated LOB state and then injects its liquidity.
This latency advantage allows it to survive against aggressive LFT order flow.

### Event-Driven Activation & Temporal Frequency

Unlike LFTs operating on continuous clock time, the MM is **event-driven**. Operating on the 100ms simulation grid, it wakes up and reprices its quotes only when triggered by:
- Price thresholds ($|\Delta P_t| \ge \delta_P$)
- Volume thresholds ($|\Delta V_t| \ge \delta_V$)
*Note: In the Baseline (No Stress) phase, its latent stress variable ($\sigma = 0$). Thus, its Hawkes process intensity operates solely at the exogenous baseline ($\mu_k$), exhibiting steady quoting without clustered cancellation cascades.*

---

## Market State Sensitivity (100ms Grid)

The MM monitors the feature set (extracted via Script 10 & 12) and responds as follows:

| Signal | Agent Response |
|--------|---------------|
| `realized_vol_50` high | Increases baseline risk; widens its quoting spread to avoid adverse selection. |
| `ofi` strongly directional | Skews quotes defensively away from the heavy flow side to prevent toxic fills. |
| `touch_depth` thick | Feels secure; willing to quote tighter spreads. |
| `depth_imbalance` skewed | Anticipates short-term pressure; adjusts Bid-Ask placement asymmetrically. |
| `amihud_illiq` high | Detects thin market conditions; reduces quoted volume sizes. |

---

## Risk Controls

The agent enforces cascading risk layers at every active micro-tick:

1. **Volume Cap Gating (25% Rule):** To reflect empirical LOB limits, the MM's order size at any level **must never exceed 25%** of the total volume resting on the opposite side of the LOB (`touch_depth` fraction constraint).
2. **Inventory Gating:** `Omega_t` continuously limits quoting on the side that would increase net position.
3. **Net Position Boundary:** A hard absolute limit on maximum allowable inventory.
4. **Solvency Gating (Margin Call):** Agent defaults and is permanently removed if wealth drops below leverage-adjusted maintenance margin:
if W_{k,t} ≤ (|inventory_t| * mid_price_t) / Leverage_Factor → Margin Call → Default

---

## Role in Flash Crash Dynamics

The MM is the primary **liquidity buffer** in the baseline simulation, preventing random noise from causing dislocations. 
However, its structural constraints make it the **catalyst for liquidity vacuums** during a crash:
1. Aggressive Momentum flow causes extreme negative `ofi` and high `realized_vol_50`.
2. MM incurs adverse selection and rapidly accumulating inventory.
3. Inventory constraints ($\Omega$) and Solvency limits force the MM to widen `spread_bps` and eventually withdraw quotes.
4. Withdrawal leads to `touch_depth` collapse, allowing small market orders to cause massive price drops (Flash Crash).

---

## Parameters at a Glance

| Parameter | Source / Nature | Default / Range |
|-----------|-----------------|-----------------|
| `micro_latency_advantage` ($\eta_H$) | Microstructure Setup | $\eta_H \ll \eta_L$ |
| `inventory_sensitivity_vg` | Extracted / Calibrated | > 0 |
| `max_order_frac_depth` | LOB empirical limit | 0.25 (25%) |
| `event_trigger_price_bps` ($\delta_P$) | Config | e.g., 2-5 bps |
| `leverage_factor` ($L$) | Config / Baseline | Empirical HFT scale |
| `baseline_intensity` ($\mu_k$) | Hawkes Process | $\mu_k > 0$ (Stress $\sigma = 0$) |

---

## Academic Grounding

| Trait | Reference |
|-------|-----------|
| Zero-mean expectation & Spread focus | Witte (2012) |
| Inventory skewing penalty | Avellaneda & Stoikov (2008) |
| Latency advantage & Step 2 execution | Leal, Napoletano, Roventini, & Fagiolo (2014) |
| Event-driven activation (tick-level) | Wellman & Wah (2017) |
| Stress-modulated Hawkes process | Cartlidge & Shi (2023) |
| Margin-based solvency constraint | Bookstaber et al. (2014) |