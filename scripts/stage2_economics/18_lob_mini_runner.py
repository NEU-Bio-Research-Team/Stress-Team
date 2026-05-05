"""Script 18 - LOB mini runner for Phase 2 simulation.

Simulates flash-crash dynamics on a 100ms grid using:
- Behavioral priors from Phase 1
- Empirical anchors from prior_anchors.json
- Event-level initialization from Flash_Crash_Events_Labeled.csv

Output schema (per 100ms tick):
    run_id, event_id, tick_ms, phase,
    close, mid_price, ofi, spread_bps,
    depth_imbalance, trade_intensity,
    realized_vol_50, leverage_proxy,
    kyle_lambda, vpin, flash_crash_flag,
    mean_wealth_t, pct_insolvent, wealth_concentration

Wealth tracking (Confounder):
    Each agent carries W_t (cash, USDT) + inventory (BTC).
    Order size is capped at max_wealth_fraction of mark-to-market wealth.
    Agents that reach W_total <= 0 become insolvent and stop trading.
    pct_insolvent drives MM-withdrawal amplification and is a causal
    confounder in DAG: wealth_concentration -> leverage_proxy -> flash_crash.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORS_JSON = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase1_outputs" / "behavioral_priors.json"
DEFAULT_ANCHORS_JSON = PROJECT_ROOT / "data" / "processed" / "tardis" / "confounder_outputs" / "prior_anchors.json"
DEFAULT_EVENTS_CSV = (
    PROJECT_ROOT / "data" / "processed" / "tardis" / "confounder_outputs" / "Flash_Crash_Events_Labeled.csv"
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_simulation_llm.csv"
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_summary_llm.json"

PHASES = ["pre", "drop", "recovery", "post"]
CALIBRATION_PHASES = ["pre", "normal_bull", "normal_bear"]
AGENT_TYPES = ["momentum_trader", "contrarian_trader", "hft_market_maker", "noise_trader"]
DEFAULT_MIX = {
    "momentum_trader": 0.30,
    "contrarian_trader": 0.20,
    "hft_market_maker": 0.20,
    "noise_trader": 0.30,
}


# Initial wealth per agent type (USDT notional).
AGENT_INIT_WEALTH: dict[str, float] = {
    "momentum_trader": 100_000.0,
    "contrarian_trader": 150_000.0,
    "hft_market_maker": 500_000.0,
    "noise_trader": 50_000.0,
}
# Max fraction of mark-to-market wealth deployed per single order.
MAX_WEALTH_FRACTION = 0.05


@dataclass
class AgentBehavior:
    aggressiveness: float
    cancel_probability: float
    order_size_multiplier: float
    inventory_sensitivity: float
    market_order_fraction: float


@dataclass
class AgentState:
    agent_type: str
    W_t: float          # cash balance (USDT)
    inventory: float    # BTC units (positive = long, negative = short)
    is_solvent: bool = True

    def mark_to_market(self, price: float) -> float:
        return self.W_t + self.inventory * price

    def update(self, signed_qty: float, trade_price: float) -> None:
        """signed_qty > 0 = buy (cash out), < 0 = sell (cash in)."""
        self.W_t -= signed_qty * trade_price
        self.inventory += signed_qty
        if self.mark_to_market(trade_price) <= 0:
            self.is_solvent = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mini LOB simulation for flash-crash events")
    parser.add_argument("--priors-json", type=Path, default=DEFAULT_PRIORS_JSON)
    parser.add_argument("--anchors-json", type=Path, default=DEFAULT_ANCHORS_JSON)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--scenario", choices=["llm", "uniform", "literature"], default="llm")
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tick-ms", type=int, default=100)
    parser.add_argument("--impact-scale", type=float, default=2.0,
                        help="Calibrated in Step-2 benchmark sweep to hit ~10%% crash-rate target")
    parser.add_argument("--intensity-scale", type=float, default=1.2289,
                        help="Calibrated via Step-1 OFI p50 match: target -0.1915, achieved -0.177 (7.7%% error)")
    parser.add_argument("--calibration-phase", choices=CALIBRATION_PHASES, default="pre",
                        help="Anchor phase used for Noise/MM calibration priors")
    parser.add_argument("--base-order-size", type=float, default=0.25)
    parser.add_argument("--mm-vol-threshold-mult", type=float, default=1.4)
    parser.add_argument("--mm-withdrawal-strength", type=float, default=1.8)
    parser.add_argument("--crash-window-ticks", type=int, default=10)
    parser.add_argument("--crash-threshold-pct", type=float, default=1.93)
    parser.add_argument("--max-drop-ticks", type=int, default=5000,
                        help="Upper cap for drop-phase ticks per run to avoid pathological long events")
    parser.add_argument("--max-recovery-ticks", type=int, default=3000,
                        help="Upper cap for recovery-phase ticks per run")
    parser.add_argument("--max-post-ticks", type=int, default=2000,
                        help="Upper cap for post-phase ticks per run")
    parser.add_argument("--max-pre-ticks", type=int, default=2000,
                        help="Upper cap for pre-phase ticks per run")
    parser.add_argument("--drop-sell-pressure", type=float, default=0.12,
                        help="Extra sell tilt in drop phase (probability shift from buy to sell)")
    parser.add_argument("--drop-impact-mult", type=float, default=1.35,
                        help="Impact multiplier applied only during drop phase")
    parser.add_argument("--min-price-fraction", type=float, default=0.70,
                        help="Per-run hard floor as a fraction of init_price to avoid price-to-zero collapse")
    parser.add_argument("--resilience-floor-fraction", type=float, default=0.85,
                        help="Reference depth level for drop-phase impact damping")
    parser.add_argument("--resilience-min-damp", type=float, default=0.20,
                        help="Minimum impact multiplier near deep-drop region (0-1)")
    parser.add_argument("--log-every-runs", type=int, default=1)
    parser.add_argument("--log-every-ticks", type=int, default=500)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_phase_metric(anchors: dict[str, Any], key: str, phase: str, fallback: float) -> float:
    obj = anchors.get(key, {})
    if phase not in obj:
        return fallback
    value = obj[phase]
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        v = float(value)
    except Exception:
        return fallback
    if not np.isfinite(v):
        return fallback
    return v


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def resolve_anchor_phase(sim_phase: str, calibration_phase: str) -> str:
    """Select which anchor phase to use for calibration-sensitive metrics.

    Only apply calibration_phase override for normal-baseline phases
    (normal_bull, normal_bear). Flash-crash phases (pre, drop, recovery,
    post) must use their own empirical anchors so that drop-phase dynamics
    (higher trade intensity, directional OFI) are preserved.
    """
    NORMAL_CALIBRATION = {"normal_bull", "normal_bear"}
    if calibration_phase in NORMAL_CALIBRATION:
        return calibration_phase
    return sim_phase


def wealth_order_size(
    agent: AgentState,
    price: float,
    behavior: "AgentBehavior",
    base_order_size: float,
    rng: np.random.Generator,
) -> float:
    """Wealth-normalized order size capped to MAX_WEALTH_FRACTION of W_total.

    Returns 0.0 for insolvent agents.
    """
    if not agent.is_solvent:
        return 0.0
    W_total = agent.mark_to_market(price)
    if W_total <= 0:
        return 0.0

    # Base size from prior, scaled by wealth
    wealth_cap_btc = (W_total * MAX_WEALTH_FRACTION) / max(price, 1e-9)
    base_size = base_order_size * behavior.order_size_multiplier * max(0.1, rng.lognormal(0.0, 0.35))

    # Inventory sensitivity penalty for momentum traders (wealth-normalized)
    if agent.agent_type == "momentum_trader" and behavior.inventory_sensitivity > 0:
        inv_notional = abs(agent.inventory) * price
        inv_ratio = inv_notional / max(W_total, 1e-9)
        penalty = max(0.0, 1.0 - behavior.inventory_sensitivity * inv_ratio)
        base_size *= penalty

    return float(min(base_size, wealth_cap_btc))


def init_agent_states(agent_mix: dict[str, float], population_n: int = 100) -> list[AgentState]:
    """Create a fixed population of AgentState objects for one simulation run."""
    states: list[AgentState] = []
    for agent_type, fraction in agent_mix.items():
        n = max(1, round(fraction * population_n))
        init_w = AGENT_INIT_WEALTH.get(agent_type, 100_000.0)
        for _ in range(n):
            states.append(AgentState(agent_type=agent_type, W_t=init_w, inventory=0.0))
    return states


def wealth_stats(agents: list[AgentState], price: float) -> tuple[float, float, float]:
    """Returns (mean_wealth_t, pct_insolvent, wealth_concentration).

    wealth_concentration = std(W_total) / mean(W_total) — Gini proxy.
    """
    totals = [a.mark_to_market(price) for a in agents]
    arr = np.asarray(totals, dtype=float)
    mean_w = float(np.mean(arr))
    pct_ins = float(np.mean(arr <= 0))
    std_w = float(np.std(arr, ddof=0))
    concentration = std_w / max(abs(mean_w), 1e-9)
    return mean_w, pct_ins, concentration


def parse_agent_mix(mix_arg: str | None) -> dict[str, float]:
    if not mix_arg:
        return DEFAULT_MIX.copy()

    pairs = [s.strip() for s in mix_arg.split(",") if s.strip()]
    parsed: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid mix token: {pair}")
        name, value = pair.split("=", 1)
        name = name.strip()
        if name not in AGENT_TYPES:
            raise ValueError(f"Unknown agent type in mix: {name}")
        parsed[name] = float(value)

    for agent in AGENT_TYPES:
        parsed.setdefault(agent, 0.0)

    total = sum(max(v, 0.0) for v in parsed.values())
    if total <= 0:
        raise ValueError("Agent mix total must be positive")

    return {k: max(v, 0.0) / total for k, v in parsed.items()}


def sample_prior_dist(dist_obj: dict[str, Any], rng: np.random.Generator) -> float:
    dist = dist_obj.get("dist")
    params = dist_obj.get("params", {})

    if dist == "beta":
        a = max(float(params.get("alpha", 1.0)), 1e-6)
        b = max(float(params.get("beta", 1.0)), 1e-6)
        return float(rng.beta(a, b))

    if dist == "gamma":
        shape = max(float(params.get("shape", 1.0)), 1e-6)
        scale = max(float(params.get("scale", 1.0)), 1e-6)
        return float(rng.gamma(shape, scale))

    if dist == "lognorm":
        shape = max(float(params.get("shape", 0.5)), 1e-6)
        loc = float(params.get("loc", 0.0))
        scale = max(float(params.get("scale", 1.0)), 1e-6)
        return float(np.exp(rng.normal(np.log(scale), shape)) + loc)

    if dist == "scalar":
        return float(params.get("value", 0.0))

    return 0.5


def sample_behavior(
    scenario: str,
    priors: dict[str, Any],
    agent_type: str,
    phase: str,
    rng: np.random.Generator,
) -> AgentBehavior:
    if scenario == "uniform":
        return AgentBehavior(
            aggressiveness=float(rng.uniform(0.0, 1.0)),
            cancel_probability=float(rng.uniform(0.0, 0.8)),
            order_size_multiplier=float(rng.uniform(0.5, 1.5)),
            inventory_sensitivity=float(rng.uniform(0.0, 1.0)),
            market_order_fraction=float(rng.uniform(0.3, 0.9)),
        )

    if scenario == "literature":
        # Stylized Kirilenko-like agent behavior constants for ablation.
        table: dict[str, dict[str, float]] = {
            "momentum_trader": {
                "aggressiveness": 0.72,
                "cancel_probability": 0.08,
                "order_size_multiplier": 1.20,
                "inventory_sensitivity": 0.25,
                "market_order_fraction": 0.85,
            },
            "contrarian_trader": {
                "aggressiveness": 0.55,
                "cancel_probability": 0.10,
                "order_size_multiplier": 1.00,
                "inventory_sensitivity": 0.35,
                "market_order_fraction": 0.70,
            },
            "hft_market_maker": {
                "aggressiveness": 0.40,
                "cancel_probability": 0.30,
                "order_size_multiplier": 0.75,
                "inventory_sensitivity": 0.50,
                "market_order_fraction": 0.10,
            },
            "noise_trader": {
                "aggressiveness": 0.45,
                "cancel_probability": 0.12,
                "order_size_multiplier": 0.95,
                "inventory_sensitivity": 0.15,
                "market_order_fraction": 0.90,
            },
        }
        values = table[agent_type]
        return AgentBehavior(**values)

    # LLM priors scenario.
    group = priors.get(agent_type, {}).get(phase, {})
    behavior_obj = group.get("behavioral_priors") or group.get("common_behavior") or {}
    common_obj = group.get("common_behavior", {})

    aggressiveness = sample_prior_dist(behavior_obj.get("aggressiveness", {"dist": "scalar", "params": {"value": 0.5}}), rng)
    cancel_probability = sample_prior_dist(
        behavior_obj.get("cancel_probability", {"dist": "scalar", "params": {"value": 0.1}}), rng
    )
    order_size_multiplier = sample_prior_dist(
        common_obj.get("order_size_multiplier", {"dist": "scalar", "params": {"value": 1.0}}), rng
    )
    inventory_sensitivity = sample_prior_dist(
        behavior_obj.get("inventory_sensitivity_vg", {"dist": "scalar", "params": {"value": 0.2}}), rng
    )
    market_order_fraction = sample_prior_dist(
        behavior_obj.get("order_type_market_fraction", {"dist": "scalar", "params": {"value": 0.8}}), rng
    )

    return AgentBehavior(
        aggressiveness=clip01(aggressiveness),
        cancel_probability=clip01(cancel_probability),
        order_size_multiplier=max(order_size_multiplier, 0.05),
        inventory_sensitivity=max(inventory_sensitivity, 0.0),
        market_order_fraction=clip01(market_order_fraction),
    )


def infer_side_probability(
    agent_type: str,
    phase: str,
    ofi_anchor_median: float,
    drop_sell_pressure: float,
) -> float:
    # Returns P(buy).
    trend_buy = 0.58 if ofi_anchor_median >= 0 else 0.42
    p_buy = 0.50

    if agent_type == "momentum_trader":
        if phase == "drop":
            p_buy = 0.25
        elif phase == "recovery":
            p_buy = 0.70
        elif phase == "post":
            p_buy = 0.55
        else:
            p_buy = trend_buy
    elif agent_type == "contrarian_trader":
        if phase == "drop":
            p_buy = 0.55
        elif phase in {"recovery", "post"}:
            p_buy = 0.35
        else:
            p_buy = 0.45
    elif agent_type == "hft_market_maker":
        p_buy = 0.50
    elif agent_type == "noise_trader":
        if phase == "drop":
            p_buy = 0.45
        else:
            p_buy = 0.50 + 0.05 * np.sign(ofi_anchor_median)

    # Apply a uniform sell-tilt in drop phase for all archetypes EXCEPT contrarian_trader.
    # Contrarian is exempt: it acts as a liquidity restorer / stabilizer (Kyle 1985,
    # Glosten-Milgrom 1985) and must retain its buying bias during drop to preserve
    # the recovery mechanism required by COMOSA §2.2 and Hypothesis H2.
    if phase == "drop" and agent_type != "contrarian_trader":
        p_buy = p_buy - drop_sell_pressure
    return clip01(p_buy)


def max_drawdown_pct_rolling(close_series: pd.Series, window_ticks: int) -> float:
    """Maximum rolling-window drawdown (%) using the same detector logic as flash_crash_flag."""
    if window_ticks <= 1:
        return 0.0
    prices = pd.to_numeric(close_series, errors="coerce").to_numpy(dtype=float)
    n = len(prices)
    if n < window_ticks:
        return 0.0

    best = 0.0
    for end_idx in range(window_ticks - 1, n):
        win = prices[end_idx - window_ticks + 1: end_idx + 1]
        first = float(win[0])
        if first <= 0 or not np.isfinite(first):
            continue
        min_price = float(np.nanmin(win))
        if not np.isfinite(min_price):
            continue
        drop_pct = (first - min_price) / first * 100.0
        if drop_pct > best:
            best = drop_pct
    return float(best)


def rolling_realized_vol(log_returns: list[float], window: int = 50) -> float:
    if not log_returns:
        return 0.0
    arr = np.asarray(log_returns[-window:], dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=0))


def rolling_vpin(flow_history: list[float], window: int = 50) -> float:
    if not flow_history:
        return 0.0
    arr = np.asarray(flow_history[-window:], dtype=float)
    numerator = float(np.sum(np.abs(arr)))
    denominator = float(np.sum(np.abs(arr)) + 1e-9)
    return numerator / denominator if denominator > 0 else 0.0


def compute_phase_lengths(event_row: pd.Series, args: argparse.Namespace) -> dict[str, int]:
    drop = int(max(pd.to_numeric(event_row.get("drop_duration_100ms"), errors="coerce") or 200, 20))
    drop = int(min(drop, max(int(args.max_drop_ticks), 20)))

    recovery_raw = pd.to_numeric(event_row.get("recovery_duration_100ms"), errors="coerce")
    recovery = int(max(recovery_raw if pd.notna(recovery_raw) else 120, 20))
    recovery = int(min(recovery, max(int(args.max_recovery_ticks), 20)))

    total_raw = pd.to_numeric(event_row.get("total_event_duration_100ms"), errors="coerce")
    if pd.notna(total_raw):
        post = int(max(float(total_raw) - drop - recovery, 20))
    else:
        post = int(max(drop * 0.25, 20))
    post = int(min(post, max(int(args.max_post_ticks), 20)))

    pre = int(max(drop * 0.35, 30))
    pre = int(min(pre, max(int(args.max_pre_ticks), 30)))
    return {"pre": pre, "drop": drop, "recovery": recovery, "post": post}


def assign_phase(tick_idx: int, phase_lengths: dict[str, int]) -> str:
    t = tick_idx
    for phase in PHASES:
        length = phase_lengths[phase]
        if t < length:
            return phase
        t -= length
    return "post"


def run_one_simulation(
    run_id: int,
    event_row: pd.Series,
    scenario: str,
    priors: dict[str, Any],
    anchors: dict[str, Any],
    agent_mix: dict[str, float],
    args: argparse.Namespace,
    rng: np.random.Generator,
    run_label: str,
    calibration_phase: str,
) -> pd.DataFrame:
    phase_lengths = compute_phase_lengths(event_row, args)
    total_ticks = int(sum(phase_lengths.values()))

    init_price = float(pd.to_numeric(event_row.get("tick_start_price"), errors="coerce") or 30_000.0)
    mid_price = init_price

    close_series: list[float] = []
    log_returns: list[float] = []
    flow_history: list[float] = []

    rows: list[dict[str, Any]] = []

    event_id = int(pd.to_numeric(event_row.get("event_id"), errors="coerce") or run_id)

    # ── Wealth: initialise agent population for this run ──────────────────
    agents = init_agent_states(agent_mix, population_n=100)

    for tick_idx in range(total_ticks):
        phase = assign_phase(tick_idx, phase_lengths)
        anchor_phase = resolve_anchor_phase(phase, calibration_phase)

        kyle_lambda = safe_phase_metric(anchors, "kyle_lambda_per_phase", anchor_phase, fallback=0.5)
        trade_intensity_anchor = safe_phase_metric(anchors, "trade_intensity_per_phase", anchor_phase, fallback=8.0)
        spread_anchor = safe_phase_metric(anchors, "spread_bps_per_phase", anchor_phase, fallback=2.0)
        realized_anchor = safe_phase_metric(anchors, "realized_vol_per_phase", anchor_phase, fallback=0.001)
        depth_anchor = safe_phase_metric(anchors, "depth_imbalance_per_phase", anchor_phase, fallback=0.0)
        vpin_anchor = safe_phase_metric(anchors, "vpin_per_phase", anchor_phase, fallback=0.25)
        ofi_p50 = float(anchors.get("ofi_percentiles_per_phase", {}).get(anchor_phase, {}).get("p50", 0.0) or 0.0)

        lam = max(min(trade_intensity_anchor * args.intensity_scale, 500.0), 0.05)
        arrivals = int(rng.poisson(lam=lam))

        net_flow = 0.0
        gross_flow = 0.0
        cancel_volume = 0.0

        vol_now = rolling_realized_vol(log_returns, window=50)
        mm_threshold = args.mm_vol_threshold_mult * max(realized_anchor, 1e-7)
        mm_stress = max(vol_now / max(mm_threshold, 1e-8) - 1.0, 0.0)

        for _ in range(arrivals):
            agent_type = rng.choice(AGENT_TYPES, p=[agent_mix[a] for a in AGENT_TYPES])
            behavior = sample_behavior(scenario, priors, agent_type, phase, rng)

            # Participation gate keeps the simulator behavior-sensitive.
            if rng.random() > behavior.aggressiveness:
                continue

            # ── Pick a solvent agent of the chosen type ───────────────────
            solvent_pool = [a for a in agents if a.agent_type == agent_type and a.is_solvent]
            if not solvent_pool:
                continue
            agent = solvent_pool[int(rng.integers(0, len(solvent_pool)))]

            side_buy_prob = infer_side_probability(
                agent_type=agent_type,
                phase=phase,
                ofi_anchor_median=ofi_p50,
                drop_sell_pressure=args.drop_sell_pressure,
            )
            side = 1.0 if rng.random() < side_buy_prob else -1.0

            # ── Wealth-normalised order size ──────────────────────────────
            order_size = wealth_order_size(agent, mid_price, behavior, args.base_order_size, rng)
            if order_size <= 0:
                continue

            cancel_prob = behavior.cancel_probability
            if agent_type == "hft_market_maker":
                cancel_prob = clip01(cancel_prob + mm_stress * 0.35)

            is_canceled = rng.random() < cancel_prob
            if is_canceled:
                cancel_volume += order_size
                continue

            gross_flow += order_size
            net_flow += side * order_size

            # ── Update agent's balance and solvency ───────────────────────
            agent.update(signed_qty=side * order_size, trade_price=mid_price)

        phase_impact_mult = args.drop_impact_mult if phase == "drop" else 1.0
        impact = args.impact_scale * phase_impact_mult * kyle_lambda * net_flow

        if phase == "drop":
            floor_ref = init_price * args.resilience_floor_fraction
            span = max(init_price - floor_ref, 1e-9)
            depth_ratio = (mid_price - floor_ref) / span
            depth_ratio = float(np.clip(depth_ratio, 0.0, 1.0))
            min_damp = float(np.clip(args.resilience_min_damp, 0.0, 1.0))
            resilience_damp = min_damp + (1.0 - min_damp) * depth_ratio
            impact *= resilience_damp

        if scenario == "literature":
            impact *= 1.05
        if scenario == "uniform":
            impact *= 0.85

        prev_price = max(mid_price, 1e-9)
        mid_price = max(prev_price + impact, 1e-6)
        # Keep runs inside flash-crash regime and avoid pathological collapse-to-zero.
        mid_price = max(mid_price, init_price * args.min_price_fraction)
        close_price = mid_price

        log_ret = float(np.log(close_price / prev_price)) if prev_price > 0 else 0.0
        log_returns.append(log_ret)
        close_series.append(close_price)
        flow_history.append(net_flow)

        realized_vol_50 = rolling_realized_vol(log_returns, window=50)
        vpin = 0.5 * vpin_anchor + 0.5 * rolling_vpin(flow_history, window=50)

        spread_bps = spread_anchor * (1.0 + args.mm_withdrawal_strength * mm_stress)
        spread_bps = max(spread_bps, 0.01)

        depth_imbalance = np.tanh(0.5 * depth_anchor + 0.04 * net_flow - 0.02 * mm_stress)
        trade_intensity = float(arrivals)

        leverage_proxy = 1.0 + abs(log_ret) / max(realized_anchor, 1e-8)
        leverage_proxy = float(np.clip(leverage_proxy, 0.0, 25.0))

        # ── Wealth metrics for this tick ──────────────────────────────────
        mean_wealth_t, pct_insolvent, wealth_concentration = wealth_stats(agents, mid_price)

        # Amplify MM-withdrawal if many agents are insolvent
        if pct_insolvent > 0.10:
            mm_stress = mm_stress * (1.0 + pct_insolvent)
            spread_bps = spread_anchor * (1.0 + args.mm_withdrawal_strength * mm_stress)
            spread_bps = max(spread_bps, 0.01)

        window = close_series[-args.crash_window_ticks :]
        flash_crash_flag = 0
        if len(window) == args.crash_window_ticks and window[0] > 0:
            drop_pct = (window[0] - min(window)) / window[0] * 100.0
            if drop_pct >= args.crash_threshold_pct:
                flash_crash_flag = 1

        rows.append(
            {
                "run_id": run_id,
                "event_id": event_id,
                "tick_ms": tick_idx * args.tick_ms,
                "phase": phase,
                "close": close_price,
                "mid_price": mid_price,
                "ofi": net_flow,
                "spread_bps": spread_bps,
                "depth_imbalance": float(depth_imbalance),
                "trade_intensity": trade_intensity,
                "realized_vol_50": realized_vol_50,
                "leverage_proxy": leverage_proxy,
                "kyle_lambda": kyle_lambda,
                "vpin": vpin,
                "flash_crash_flag": flash_crash_flag,
                # Wealth confounder columns
                "mean_wealth_t": float(mean_wealth_t),
                "pct_insolvent": float(pct_insolvent),
                "wealth_concentration": float(wealth_concentration),
            }
        )

        if args.log_every_ticks > 0 and ((tick_idx + 1) % args.log_every_ticks == 0 or (tick_idx + 1) == total_ticks):
            print(
                f"{run_label} tick {tick_idx + 1}/{total_ticks} "
                f"phase={phase} price={close_price:.2f} ofi={net_flow:.4f} insolvent={pct_insolvent:.3f}",
                flush=True,
            )

    return pd.DataFrame(rows)


def summarize_output(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    run_flags = df.groupby("run_id")["flash_crash_flag"].max()
    phase_ofi = df.groupby("phase")["ofi"].mean().to_dict()
    run_max_dd = df.groupby("run_id")["close"].apply(
        lambda s: max_drawdown_pct_rolling(s, args.crash_window_ticks)
    )

    mean_wealth_by_phase = df.groupby("phase")["mean_wealth_t"].mean().to_dict()
    pct_ins_peak = float(df["pct_insolvent"].max())
    wealth_conc_drop = float(df.loc[df["phase"] == "drop", "wealth_concentration"].mean()) if "drop" in df["phase"].values else float("nan")

    return {
        "scenario": args.scenario,
        "calibration_phase": args.calibration_phase,
        "n_runs": int(df["run_id"].nunique()),
        "n_rows": int(len(df)),
        "tick_ms": args.tick_ms,
        "crash_window_ticks": int(args.crash_window_ticks),
        "crash_threshold_pct": float(args.crash_threshold_pct),
        "min_price_fraction": float(args.min_price_fraction),
        "flash_crash_rate": float(run_flags.mean()) if len(run_flags) else 0.0,
        "run_max_drawdown_pct": {
            "mean": float(run_max_dd.mean()) if len(run_max_dd) else 0.0,
            "p50": float(run_max_dd.quantile(0.50)) if len(run_max_dd) else 0.0,
            "p90": float(run_max_dd.quantile(0.90)) if len(run_max_dd) else 0.0,
            "p95": float(run_max_dd.quantile(0.95)) if len(run_max_dd) else 0.0,
            "p99": float(run_max_dd.quantile(0.99)) if len(run_max_dd) else 0.0,
            "max": float(run_max_dd.max()) if len(run_max_dd) else 0.0,
        },
        "mean_ofi_pre": float(phase_ofi.get("pre", np.nan)),
        "mean_ofi_drop": float(phase_ofi.get("drop", np.nan)),
        "mean_wealth_by_phase": {k: float(v) for k, v in mean_wealth_by_phase.items()},
        "pct_insolvent_peak": pct_ins_peak,
        "wealth_concentration_drop": wealth_conc_drop,
        "columns": list(df.columns),
    }


def main() -> None:
    args = parse_args()

    # Keep optional override string out of parser defaults for backwards compatibility.
    if not hasattr(args, "agent_mix"):
        setattr(args, "agent_mix", None)

    for path in [args.priors_json, args.anchors_json, args.events_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    priors = load_json(args.priors_json)
    anchors = load_json(args.anchors_json)
    events = pd.read_csv(args.events_csv)

    rng = np.random.default_rng(args.seed)

    agent_mix = DEFAULT_MIX.copy()

    all_runs: list[pd.DataFrame] = []
    event_indices = np.arange(len(events))
    started = time.time()

    for run_id in range(args.n_runs):
        idx = int(rng.choice(event_indices))
        event_row = events.iloc[idx]
        event_id = int(pd.to_numeric(event_row.get("event_id"), errors="coerce") or idx)
        phase_lengths = compute_phase_lengths(event_row, args)
        total_ticks = int(sum(phase_lengths.values()))

        if args.log_every_runs > 0 and ((run_id + 1) % args.log_every_runs == 0 or run_id == 0):
            print(
                f"[run {run_id + 1}/{args.n_runs}] start event_id={event_id} "
                f"ticks={total_ticks} phases={phase_lengths}",
                flush=True,
            )

        run_started = time.time()
        run_df = run_one_simulation(
            run_id=run_id,
            event_row=event_row,
            scenario=args.scenario,
            priors=priors,
            anchors=anchors,
            agent_mix=agent_mix,
            args=args,
            rng=rng,
            run_label=f"[run {run_id + 1}/{args.n_runs}]",
            calibration_phase=args.calibration_phase,
        )
        all_runs.append(run_df)

        run_elapsed = time.time() - run_started
        elapsed = time.time() - started
        done = run_id + 1
        avg_per_run = elapsed / done
        remaining = max(args.n_runs - done, 0)
        eta_sec = avg_per_run * remaining
        run_max_dd = max_drawdown_pct_rolling(run_df["close"], args.crash_window_ticks)
        run_crash = int(run_df["flash_crash_flag"].max()) if len(run_df) else 0
        print(
            f"[run {done}/{args.n_runs}] done rows={len(run_df)} "
            f"max_dd_{args.crash_window_ticks}t={run_max_dd:.4f}% "
            f"crash={run_crash} run_time={run_elapsed:.1f}s elapsed={elapsed:.1f}s eta={eta_sec:.1f}s",
            flush=True,
        )

    out_df = pd.concat(all_runs, ignore_index=True)

    expected_cols = [
        "run_id",
        "event_id",
        "tick_ms",
        "phase",
        "close",
        "mid_price",
        "ofi",
        "spread_bps",
        "depth_imbalance",
        "trade_intensity",
        "realized_vol_50",
        "leverage_proxy",
        "kyle_lambda",
        "vpin",
        "flash_crash_flag",
        "mean_wealth_t",
        "pct_insolvent",
        "wealth_concentration",
    ]
    out_df = out_df[expected_cols].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    summary = summarize_output(out_df, args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Script 18: LOB mini runner")
    print("=" * 72)
    print(f"Scenario        : {args.scenario}")
    print(f"Calibration     : {args.calibration_phase}")
    print(f"Runs            : {summary['n_runs']}")
    print(f"Rows            : {summary['n_rows']}")
    print(f"Crash rate      : {summary['flash_crash_rate']:.4f}")
    print(
        "Run max DD (%) : "
        f"mean={summary['run_max_drawdown_pct']['mean']:.4f} "
        f"p95={summary['run_max_drawdown_pct']['p95']:.4f} "
        f"max={summary['run_max_drawdown_pct']['max']:.4f}"
    )
    print(f"OFI pre mean    : {summary['mean_ofi_pre']:.6f}")
    print(f"OFI drop mean   : {summary['mean_ofi_drop']:.6f}")
    print(f"Pct insolvent   : {summary['pct_insolvent_peak']:.4f}")
    print(f"Wealth conc drop: {summary['wealth_concentration_drop']:.4f}")
    print(f"Saved panel     : {args.output_csv}")
    print(f"Saved summary   : {args.summary_json}")


if __name__ == "__main__":
    main()
