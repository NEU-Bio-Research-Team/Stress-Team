# Financial Market Bio-Technical System DAG Model

## 1. Big Picture Overview
This DAG models financial markets as a bio-technical system where traders' physiological stress acts as a latent state variable, driving crashes through feedback loops. Exogenous shocks like news create uncertainty and baseline stress, which elevate latent stress (Sigma_t). This triggers behavioral changes in market makers (MM) and momentum traders—increased risk aversion, slower reactions, inattention, and reduced participation—leading to degraded market microstructure (wider spreads, thinner depth). Order flow imbalances form returns and volatility, which feed back to amplify stress, creating endogenous instability. 

**The full causal chain:** Stress → Behavior (risk aversion, timing) → Market (liquidity withdrawal, order flow) → Output (returns, volatility) → Feedback (volatility → uncertainty → stress).

---

## 2. Layer-by-Layer Explanation

### 2.1 Confounders (Layer 0)
Confounders influence multiple downstream paths, creating correlated errors if ignored.
* **Theta_i (personal heterogeneity):** Captures individual differences in stress response, risk aversion (GammaMM, GammaMom), reaction time (Tau_t), and inattention (Ithr_t); *why: traders vary in resilience, some panic more.*
* **TimeOfDay (ToD):** Affects latent stress (Sigma_t) and past volatility (Vol_t-1); *why: diurnal patterns like end-of-day fatigue amplify stress and volatility.*
* **Leverage:** Boosts momentum trader risk aversion (GammaMom_t) and order sizes (OrderSize_t); *why: high leverage magnifies losses, heightening sensitivity.*
* **PortfolioLimit:** Constrains MM risk aversion (GammaMM_t) and quote depth (QuoteDepth_t); *why: inventory caps force conservative quoting.*

### 2.2 Exogenous & Stress (Layer 1)
Exogenous inputs initiate stress propagation.
* **News_t** and **Vol_t-1** form Uncertainty_t, raising perceived risk.
* **BaselineStress_i** directly feeds LatentStress_t (Sigma_t), representing chronic trader tension.
* **Baselines (BaseGammaMM, BaseGammaMom)** set default risk aversions, modulated by stress.
* *Note:* Uncertainty and latent stress emerge as the core endogenous drivers, mean-reverting but shock-amplifiable (OU process, half-life ~10s).

### 2.3 Stress → Behavior (Layer 2)
Stress physiologically alters trader psychology, per empirical proxies (ECG/EDA).
* Elevates **GammaMM_t** (MM risk aversion) and **GammaMom_t** (momentum aversion): fight-or-flight prioritizes preservation → wider quotes, smaller bets.
* Increases **Tau_t** (reaction time): panic speeds rushed trades, reducing deliberation.
* Raises **Ithr_t** (inattention threshold): overload ignores weak signals → myopic errors.
* Lowers **Part_t** (participation): withdrawal conserves energy amid threat.
* *Intuition:* Stress narrows focus (tunnel vision), amplifying conservatism in liquidity providers.

### 2.4 Inattention Mechanism (Layer 2b)
Traders filter signals via probabilistic check: `f(Sigma_t, Ithr_t)` decides attention.
* SignalInput → Check → Drop (no trade) or SigProc_t2 (trade).
* High stress raises threshold → more drops, fewer informed trades.
* *Why this matters in markets:* Prevents noise trading but during stress, filters out stabilizing orders, worsening imbalances.

### 2.5 Inventory & Agent Actions (Layer 3)
Agent types differ: MM stabilize, momentum amplify.
* Past **OrderFlow_t-1** builds Inventory_MM_t → feeds GammaMM_t (aversion rises with imbalance) and future inventory.
* Stressed MM: higher GammaMM_t → wider QuoteSpread_t, shallower QuoteDepth_t.
* Stressed momentum: higher GammaMom_t → larger OrderSize_t; delayed Tau_t → suboptimal OrderTiming_t.
* SigProc_t2 and Part_t drive OrderArrival_t and OrderFlow_t.
* *Result:* Inventory pressure forces MM conservatism, snowballing under stress.

### 2.6 Market Microstructure (Layer 4)
Agent actions shape liquidity dynamics.
* **QuoteDepth_t** and **QuoteSpread_t** form Spread_t and Depth_{t+1}; Depth_t reinforces.
* *Key novelty:* Spread_t ↑ → OrderArrival_t ↓ (wide spreads deter entries via costs).
* OrderArrival_t, OrderSize_t, OrderTiming_t aggregate to OrderFlow_t; sizes consume Depth_{t+1}.
* OrderFlow_t impacts Midprice_t.
* *Result:* Liquidity self-reinforces: stress widens spreads → fewer orders → thinner books → wider spreads.

### 2.7 Output (Layer 5)
Market outcomes emerge from microstructure.
* **Midprice_t** → **Returns_t** → **Volatility_t** (realized std of log returns).
* Price forms via order flow imbalances in auction clearing.
* Volatility clusters from herding, matching stylized facts.

### 2.8 Feedback Loop (Layer 6)
Closes the cycle for dynamics.
* Vol_t → Uncertainty_{t+1} → LatentStress_{t+1} (Sigma_t+1).
* Sigma_t+1 → GammaMM/GamMom_{t+1}, propagating forward.
* *System evolves:* shocks decay unless feedback > mean-reversion (instability if eigenvalue >1).

---

## 3. Key Mechanisms (IMPORTANT)
* **Stress amplification:** Uncertainty → stress → behavior → vol → more uncertainty; endogenous crashes from bio-feedback.
* **Liquidity withdrawal:** Stressed MM raise spreads/depth aversion → Spread ↑ → fewer arrivals → thinner markets.
* **Inventory pressure:** Past flow imbalances hike aversion, forcing position limits → conservative quoting.
* **Attention filtering:** High Ithr_t drops signals → uninformed flow, herding vol spikes.
* **Participation drop:** Stress curbs Part_t → order drought, amplifying imbalances.

---

## 4. Intuition Through Example
Bad news (e.g., BTC hack rumor) spikes `Uncertainty_t` → `Sigma_t` rises. MM aversion surges: `QuoteSpread_t` widens, `Depth_t` thins → high costs deter arrivals. Momentum traders herd on weak signals (inattention), imbalanced `OrderFlow_t` → `Returns_t` plunge, `Vol_t` explodes → feeds `Sigma_{t+1}`, creating a panic cascade.

---

## 5. Stability vs Crash
* **Stable** when feedback < decay (vol → stress gain <1): small shocks revert via OU process, noise traders buffer.
* **Unstable/crash:** strong positive loop (gain >1) → bifurcation, runaway (Lyapunov unstable). Small shocks contained; large ones (news + high baseline) cascade via amplification.

---

## 6. How to Use This Model
* **Research:** DAG for identification (backdoor criteria), ABM calibration, test couplings (M0-M3 ladder).
* **Trading:** Anticipate stress (vol + phys proxies) → fade liquidity withdrawals, avoid herding.
* **Simulation:** Agent-based (5-20 MM, 50-200 momentum, 100-500 noise); bifurcate for crash probs, policy test circuit breakers.