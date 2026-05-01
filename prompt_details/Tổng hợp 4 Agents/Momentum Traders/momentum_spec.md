# Momentum Trader — Agent Profile

---

## Archetype Summary

The momentum trader is a **low-frequency trend follower** and **aggressive liquidity taker**.
It does not provide liquidity to the market. It consumes it.

Its core behavioral thesis: short-term price trends persist. When price deviates from a
recent memory mean, the agent expects continuation in the same direction — not reversion.
It acts on that expectation by submitting market orders immediately, accepting the current
best bid or ask without negotiation.

---

## Core Behavioral Traits

### Trend Extrapolation

The agent compares the current price `P_t` against a moving average `P_hat_t` computed
over a short rolling window (`ma_window_ticks`, default 50 ticks = 5 seconds at 100ms resolution).
trend_signal = sign(P_t - P_hat_t)

text

- `P_t > P_hat_t` → bullish continuation expected → buy bias
- `P_t < P_hat_t` → bearish continuation expected → sell bias
- Deviation below `edge_epsilon_bps` threshold → no trade

The expected price target is capped to avoid unrealistic extrapolation:
p_t = min(|P_t - P_hat_t|, E_t[|r|])
E_t[P] = P_t + trend_signal × p_t

text

Where `E_t[|r|]` is approximated by `realized_vol_50` — the rolling 50-bin
realized volatility from the market state.

---

### Aggressiveness

The agent's willingness to act on a signal is captured by the `aggressiveness` parameter,
sampled from a **Beta distribution** fitted during Phase 1 elicitation.

- High aggressiveness (close to 1.0): the agent commits heavily to a strong signal
- Low aggressiveness: the agent participates only minimally, or not at all
- Expected distribution: right-skewed Beta (α > β), reflecting the agent's aggressive nature

Aggressiveness scales directly with the expected edge after spread:
qty_raw = aggressiveness × edge - Omega_t

text

---

### Inventory Penalty

The agent penalizes itself for holding large positions. The larger the existing inventory
relative to total wealth, the more `Omega_t` reduces the desired order size:
wealth_t = cash_t + inventory_t × mid_price_t
Omega_t = v_g × |inventory_t × mid_price_t| / max(wealth_t, ε)

text

This formulation is **wealth-normalized**, ensuring unit consistency across BTC notional
regimes. The parameter `v_g` (inventory sensitivity) is sampled from a **Gamma distribution**
at initialization.

---

### Order Type

The agent uses **market orders exclusively**. It never places passive limit orders.

- **Buy signal** → market buy at `ask_price`
- **Sell signal** → market sell at `bid_price`

This is consistent with the aggressive taker archetype: the agent prioritizes
execution certainty over price improvement. The `order_type_market_fraction`
from Phase 1 elicitation is expected to be > 0.80.

---

### Cancellation Behavior

The momentum trader **rarely cancels** submitted orders.

`cancel_prob` is sampled from a Beta distribution expected to be near 0
(distribution skewed heavily toward the lower end). This reflects the agent's
commitment to execution once a signal is taken.

---

### Frequency

The agent is **low-frequency** relative to the 100ms simulation grid.

- It observes market state every tick
- It acts every `act_every_n_ticks` ticks (default: 5 ticks = 500ms)
- This prevents excessive churn while maintaining responsiveness to trend shifts

---

## Market State Sensitivity

| Signal | Agent Response |
|--------|---------------|
| OFI strongly negative | Reinforces bearish bias; increases sell conviction |
| OFI strongly positive | Reinforces bullish bias; increases buy conviction |
| OFI near zero | No directional edge; lean toward do_nothing |
| Spread wide | Edge after spread shrinks; raises bar to trade |
| `realized_vol_50` high | Expected continuation larger, but size capped more conservatively |
| `order_flow_toxicity` high | Optional gating: agent may reduce size or skip |
| `leverage_proxy` high | Risk context; indirectly constrains via wealth/margin checks |
| `proxy_mode = True` | Size cap halved; agent uses conservative execution |

---

## Risk Controls

The agent enforces four cascading risk layers at every active tick:

1. **Signal gating** — No trade if `|P_t - P_hat_t| < epsilon` or edge is non-positive after spread
2. **Inventory gating** — `Omega_t` grows with position size, reducing desired order qty
3. **Liquidity gating** — Order size capped by `touch_depth` (or 50% of proxy depth)
4. **Solvency gating** — Agent defaults and is removed when `wealth_t ≤ 0`

An optional stricter **margin condition** can be activated:
if wealth_t ≤ margin_buffer × |inventory_t| × mid_price_t → margin call → default

text

---

## Shorting Behavior

If `shorting_allowed = True` (default):
- Bearish signal with zero inventory → agent submits market sell → enters short position

If `shorting_allowed = False`:
- Bearish signal with zero inventory → `do_nothing`
- Bearish signal with existing long → agent may reduce/exit long, but cannot go negative

---

## Role in Flash Crash Dynamics

The momentum trader is the primary **amplifier agent** in the simulation.

During a flash crash event:
1. OFI turns sharply negative as sell pressure dominates
2. Momentum traders read the bearish continuation signal
3. They submit market sells → OFI turns more negative → spread widens
4. Spread widening triggers market maker withdrawal
5. Reduced depth increases price impact → price drops further
6. Further price drop → stronger bearish signal → more momentum sells

This self-reinforcing feedback loop is an **emergent property** of the structural
interaction between momentum traders and market makers — not encoded individually
into any single agent.

This is the core causal chain that Phase 3 (NOTEARS/LiNGAM) attempts to recover:
OFI (negative) → spread widening → depth collapse → flash_crash
↑ │
└───────── momentum trader amplification ───────────┘

text

---

## Parameters at a Glance

| Parameter | Source | Distribution | Default / Range |
|-----------|--------|-------------|-----------------|
| `aggressiveness` | Phase 1 LLM elicitation | Beta(α, β) — right-skewed | mean ~0.75 |
| `cancel_prob` | Phase 1 LLM elicitation | Beta(α, β) — near 0 | mean ~0.05 |
| `inventory_sensitivity_vg` | Phase 1 LLM elicitation | Gamma(shape, scale) | default 0.10 |
| `order_size` | Script 11 MLE from real trades | Pareto(α) | α fallback 1.7 |
| `ma_window_ticks` | Config | Discrete | 50 (range 20–100) |
| `act_every_n_ticks` | Config | Discrete | 5 (range 1–20) |
| `edge_epsilon_bps` | Config | Fixed | 2 bps (range 1–5) |
| `max_order_notional_frac_depth` | Config | Fixed | 0.10 (0.05 in proxy mode) |
| `max_inventory_notional_frac_wealth` | Config | Fixed | 0.20 (range 0.10–0.35) |
| `shorting_allowed` | Config | Boolean | True |
| `legacy_stress_mode` | Config | Boolean | False (always default OFF) |

---

## Academic Grounding

| Trait | Reference |
|-------|-----------|
| Trend extrapolation with moving average | Witte (2012); Leal et al. (2014) |
| Momentum as cross-asset pattern | Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012) |
| Aggressive taker, market order execution | Consistent with HFT taker classification in Kirilenko et al. (2017) |
| Inventory penalty, wealth-based solvency | Brunnermeier & Pedersen (2009) |
| Heavy-tailed order size distribution | Clauset, Shalizi & Newman (2009); standard in LOB literature |
| Flash crash amplification via feedback | Leal et al. (2016); SEC/CFTC Flash Crash Report (2010) |