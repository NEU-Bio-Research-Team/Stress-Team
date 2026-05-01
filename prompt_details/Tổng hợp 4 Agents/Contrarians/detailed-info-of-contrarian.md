# Contrarian Trader — Agent Profile

### Archetype Summary
The contrarian trader is a medium-frequency mean-reversion specialist and a liquidity-providing taker. Unlike momentum traders, it bets against recent price moves, seeking to profit from market overreactions.

Its core behavioral thesis: short-term price trends are often driven by irrational herding and tend to overshoot. When price deviates significantly from its long-term memory mean, the agent expects a reversal — not continuation. It acts as a "Market Stabilizer" by submitting market orders that oppose the prevailing trend, effectively providing counter-cyclical liquidity when the market reaches an extreme.


### Core Behavioral Traits
#### Mean Reversion (The Overshoot Hypothesis)
The agent compares the current price $P_t$ against a long-term moving average $\hat{P}_t$ computed over a substantial rolling window (ma_window_ticks, default 200 ticks = 20 seconds).
$trend\_signal = -sign(P_t - \hat{P}_t)$

- $P_t > \hat{P}_t$ by threshold → overbought → **sell bias** (fading the rally)
- $P_t < \hat{P}_t$ by threshold → oversold → **buy bias** (bottom-fishing)
- Deviation below **edge_epsilon_bps (10bps/0.10%)** threshold → **no trade**

The expected price target reflects the belief in a return to the mean:
$E_t[P] = \hat{P}_t$
The agent calculates the expected edge based on the distance to this mean, filtered by a microstructure-calibrated 10 bps rule for this 100 ms event dataset.

#### Aggressiveness (ESTAR Logic)
Unlike momentum traders, the contrarian’s aggressiveness is non-linear and follows an **Exponential Smooth Transition (ESTAR)** logic. The agent is passive during minor moves but becomes exponentially more aggressive as the price "overshoots" further from the mean.

- **Low deviation (near 0.10%):** Minimal participation (small probe positions).
- **Extreme deviation (0.60%+):** Maximum conviction; the agent leans heavily into the trade to capture the reversal apex.

Aggressiveness scales with the standardized price deviation:
$qty\_raw = Aggressiveness \times (1 - \exp(-\gamma \cdot (\Delta P)^2)) - \Omega_t$
#### Inventory Penalty
The agent is sensitive to inventory risk, especially when "averaging down" into a contrarian position. As absolute inventory grows, an internal penalty ($\Omega_t$) reduces the desired order size to prevent over-exposure and "catching a falling knife" indefinitely.

**Mathematical Logic:**
```text
Wealth_t = Cash_t + (Inventory_t * Mid_Price_t)
Omega_t = v_g * |Inventory_t * Mid_Price_t| / max(Wealth_t, epsilon)
```

**Behavioral Impact:**
- **Wealth-Normalized:** This formulation ensures unit consistency. The penalty scales with the position's notional value relative to the agent's total wealth.
- **Risk Aversion:** The parameter **v_g** (inventory sensitivity) acts as a brake; it forces the agent to reduce or cease buying as it approaches its maximum capacity, even if the price deviation signal remains strong.


#### Order Type
The agent uses **market orders** exclusively for execution certainty during short-lived reversal windows.
- **Buy signal (Oversold)** → market buy at ask_price
- **Sell signal (Overbought)** → market sell at bid_price

#### Frequency
The agent operates at a medium-frequency, allowing "overshoots" to deelop before stepping in.
- Observes market state every tick.
- Acts every **act_every_n_ticks** (default: 10 ticks = 1 second) to filter out high-frequency noise.


### Market State Sensitivity

| Signal | Agent Response |
| :--- | :--- |
| **OFI strongly negative** | Interprets as panic-selling overreaction; increases **buy conviction** |
| **OFI strongly positive** | Interprets as euphoria overreaction; increases **sell conviction** |
| **OFI near zero** | Market in equilibrium; **do_nothing** |
| **Spread wide** | Detects liquidity vacuum; steps in as "Provider of Last Resort" if edge is sufficient |
| **realized_vol_50 high** | Confirms extreme stress; triggers **ESTAR graduated response** |
| **touch_depth collapse** | High conviction signal for reversal entry; agent provides buffer |


### Risk Controls
The agent enforces four cascading risk layers:
1. **Signal gating** — No trade if $|P_t - \hat{P}_t| < 0.10\%$ (threshold-gated microstructure filter).
2. **Inventory gating** — $\Omega_t$ limits position size to avoid "catching a falling knife" indefinitely.
3. **Liquidity gating** — Order size capped by available depth to manage price impact.
4. **Solvency gating** — Agent defaults and is removed when $Wealth_t \le 0$ (*Bookstaber et al., 2014*).


### Role in Flash Crash Dynamics
The contrarian trader is the **primary stabilizer (Brake)** in the simulation.
- During a crash, while momentum traders amplify the drop, the contrarian reads the extreme negative OFI as an **overreaction**.
- They submit market buys → providing the necessary buy-side pressure to slow the descent.
- Their entry helps restore **touch_depth**, allowing the market to bottom out and eventually mean-revert.
- **Critical Risk:** If contrarians hit their **Solvency gating** during the drop, their forced liquidation adds to the sell pressure, potentially turning a correction into a terminal crash.


### Parameters at a Glance

| Parameter | Source | Nature | Default / Range |
| :--- | :--- | :--- | :--- |
| **Activation Threshold** | Dataset-calibrated | Fixed | 0.10% (10 bps) |
| **Aggressiveness** | *ESTAR Logic* | Non-linear | Scaled by deviation |
| **inventory_sensitivity_vg** | Phase 1 Elicitation | Gamma | Default 0.15 |
| **ma_window_ticks** | Config | Long-term | 200 (range 100–500) |
| **act_every_n_ticks** | Config | Discrete | 10 (range 5–50) |
| **max_inventory_notional** | Config | Fixed | 0.25 (wealth frac) |


### Academic Grounding

| Trait | Reference |
| :--- | :--- |
| Overreaction & Mean Reversion | *DeBondt & Thaler (1985)* |
| Non-linear Transition (ESTAR) | *Terasvirta (1994); Wan (2022)* |
| Contrarian Demand Function | *Liao et al. (2017)* |
| Threshold Filter & Whiplash Prevention | *Balsara et al. (2009)*, rescaled to 100 ms event windows |
| Inventory Risk & Solvency | *Bookstaber et al. (2014)* |