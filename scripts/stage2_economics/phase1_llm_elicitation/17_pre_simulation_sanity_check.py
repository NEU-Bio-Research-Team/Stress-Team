"""Script 17 - Pre-simulation sanity check for Phase 1 -> Phase 2 handoff.

This script performs a lightweight, reproducible gate before running mini LOB
simulation experiments. It checks two things:

1) Prior/action realism from parsed elicitation output (raw_elicited.csv)
2) Empirical stylized-fact benchmarks from gridded event dynamics

Outputs:
    - Markdown report (human-readable)
    - JSON report (machine-readable)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ELICITED_CSV = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase1_outputs" / "raw_elicited.csv"
DEFAULT_GRIDDED_CSV = (
    PROJECT_ROOT / "data" / "processed" / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms_gridded.csv"
)
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "validation" / "phase2_mini_sanity_check.md"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "validation" / "phase2_mini_sanity_check.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-simulation sanity check")
    parser.add_argument("--elicited-csv", type=Path, default=DEFAULT_ELICITED_CSV)
    parser.add_argument("--gridded-csv", type=Path, default=DEFAULT_GRIDDED_CSV)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--min-parsed-rows", type=int, default=400)
    parser.add_argument("--contrarian-sell-floor", type=float, default=0.15)
    parser.add_argument("--contrarian-sell-target", type=float, default=0.20)
    parser.add_argument("--crash-threshold-pct", type=float, default=3.0)
    return parser.parse_args()


def lag1_autocorr(x: pd.Series) -> float:
    y = pd.to_numeric(x, errors="coerce").dropna()
    if len(y) < 3:
        return float("nan")
    a = y.iloc[:-1]
    b = y.iloc[1:]
    if a.std(ddof=0) < 1e-12 or b.std(ddof=0) < 1e-12:
        return float("nan")
    return float(a.corr(b))


def safe_float(value: Any, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
        return round(float(value), ndigits)
    except Exception:
        return None


def load_parsed_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "parsed_ok" in frame.columns:
        return frame[frame["parsed_ok"] == True].copy()  # noqa: E712
    if "parse_ok" in frame.columns:
        return frame[frame["parse_ok"] == True].copy()  # noqa: E712
    if "parse_status" in frame.columns:
        return frame[frame["parse_status"] == "parsed"].copy()
    return frame.copy()


def prior_summary(parsed: pd.DataFrame) -> dict[str, Any]:
    parsed = parsed.copy()
    parsed["cancel_probability"] = pd.to_numeric(parsed.get("cancel_probability"), errors="coerce")

    cancel_unique = sorted(parsed["cancel_probability"].dropna().unique().tolist())
    cancel_stats = (
        parsed.groupby("agent_type")["cancel_probability"]
        .agg(["mean", "std", "min", "max", "nunique"])
        .round(3)
        .reset_index()
    )

    order_type = (
        parsed.pivot_table(index="agent_type", columns="order_type", values="run_id", aggfunc="count", fill_value=0)
        .pipe(lambda t: t.div(t.sum(axis=1), axis=0))
        .round(3)
        .reset_index()
    )

    side = (
        parsed.pivot_table(index="agent_type", columns="side", values="run_id", aggfunc="count", fill_value=0)
        .pipe(lambda t: t.div(t.sum(axis=1), axis=0))
        .round(3)
        .reset_index()
    )

    return {
        "parsed_rows": int(len(parsed)),
        "cancel_probability_unique": [safe_float(v, ndigits=6) for v in cancel_unique],
        "cancel_probability_stats": cancel_stats.to_dict(orient="records"),
        "order_type_by_agent": order_type.to_dict(orient="records"),
        "side_by_agent": side.to_dict(orient="records"),
    }


def empirical_benchmarks(gridded: pd.DataFrame, crash_threshold_pct: float) -> dict[str, Any]:
    df = gridded.copy()
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce")
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ofi"] = pd.to_numeric(df["ofi"], errors="coerce")
    df = df.dropna(subset=["event_id", "timestamp_ms", "close"]).copy()
    df = df.sort_values(["event_id", "timestamp_ms"]) 

    observed_df = df
    observed_ratio = None
    if "is_imputed_grid_row" in df.columns:
        imputed_flag = pd.to_numeric(df["is_imputed_grid_row"], errors="coerce").fillna(0).astype(bool)
        observed_df = df.loc[~imputed_flag].copy()
        observed_ratio = float(len(observed_df)) / float(len(df)) if len(df) else None
        if observed_df.empty:
            observed_df = df

    returns = (
        observed_df.groupby("event_id", group_keys=False)["close"]
        .apply(lambda s: np.log(s / s.shift(1)))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    returns_sq = returns.pow(2)

    kurtosis_excess = float(returns.kurt()) if len(returns) else float("nan")
    acf_vol_lag1 = lag1_autocorr(returns_sq)

    ofi_by_phase = (
        observed_df.groupby("phase")["ofi"]
        .mean()
        .to_dict()
    )
    ofi_pre = ofi_by_phase.get("pre", float("nan"))
    ofi_drop = ofi_by_phase.get("drop", float("nan"))

    if "drop_1s_pct" in observed_df.columns:
        drop_1s = pd.to_numeric(observed_df["drop_1s_pct"], errors="coerce")
        crash_rate = float((drop_1s >= crash_threshold_pct).mean())
    else:
        crash_rate = float((observed_df.get("phase") == "drop").mean())

    return {
        "rows": int(len(df)),
        "rows_observed": int(len(observed_df)),
        "observed_ratio": safe_float(observed_ratio, ndigits=6),
        "events": int(df["event_id"].nunique()),
        "returns_count": int(len(returns)),
        "kurtosis_excess": safe_float(kurtosis_excess, ndigits=6),
        "acf_vol_lag1": safe_float(acf_vol_lag1, ndigits=6),
        "ofi_pre_mean": safe_float(ofi_pre, ndigits=6),
        "ofi_drop_mean": safe_float(ofi_drop, ndigits=6),
        "crash_rate_proxy": safe_float(crash_rate, ndigits=6),
    }


def find_row(table: list[dict[str, Any]], agent_type: str) -> dict[str, Any] | None:
    for row in table:
        if row.get("agent_type") == agent_type:
            return row
    return None


def evaluate_gates(
    prior: dict[str, Any],
    empirical: dict[str, Any],
    min_parsed_rows: int,
    contrarian_sell_floor: float,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    parsed_rows = prior["parsed_rows"]
    gates.append(
        {
            "gate": "parsed_rows_minimum",
            "pass": parsed_rows >= min_parsed_rows,
            "observed": parsed_rows,
            "expected": f">= {min_parsed_rows}",
        }
    )

    order_rows = prior["order_type_by_agent"]
    side_rows = prior["side_by_agent"]
    cancel_rows = prior["cancel_probability_stats"]

    mm_order = find_row(order_rows, "hft_market_maker") or {}
    mm_cancel = find_row(cancel_rows, "hft_market_maker") or {}
    mm_limit_share = float(mm_order.get("limit", 0.0) or 0.0)
    mm_cancel_mean = float(mm_cancel.get("mean", 0.0) or 0.0)
    gates.append(
        {
            "gate": "market_maker_limit_dominance",
            "pass": mm_limit_share >= 0.90,
            "observed": safe_float(mm_limit_share, 4),
            "expected": ">= 0.90",
        }
    )
    gates.append(
        {
            "gate": "market_maker_cancel_rate_reasonable",
            "pass": 0.20 <= mm_cancel_mean <= 0.40,
            "observed": safe_float(mm_cancel_mean, 4),
            "expected": "[0.20, 0.40]",
        }
    )

    noise_side = find_row(side_rows, "noise_trader") or {}
    noise_buy = float(noise_side.get("buy", 0.0) or 0.0)
    noise_sell = float(noise_side.get("sell", 0.0) or 0.0)
    noise_balance = abs(noise_buy - noise_sell)
    gates.append(
        {
            "gate": "noise_buy_sell_balance",
            "pass": noise_balance <= 0.10,
            "observed": safe_float(noise_balance, 4),
            "expected": "<= 0.10",
        }
    )

    contr_side = find_row(side_rows, "contrarian_trader") or {}
    contr_sell = float(contr_side.get("sell", 0.0) or 0.0)
    gates.append(
        {
            "gate": "contrarian_sell_floor",
            "pass": contr_sell >= contrarian_sell_floor,
            "observed": safe_float(contr_sell, 4),
            "expected": f">= {contrarian_sell_floor:.2f}",
        }
    )

    krt = empirical.get("kurtosis_excess")
    acf = empirical.get("acf_vol_lag1")
    ofi_pre = empirical.get("ofi_pre_mean")
    ofi_drop = empirical.get("ofi_drop_mean")
    crash_rate = empirical.get("crash_rate_proxy")

    gates.append(
        {
            "gate": "empirical_fat_tails_baseline",
            "pass": (krt is not None) and (krt > 3.0),
            "observed": krt,
            "expected": "> 3.0",
        }
    )
    gates.append(
        {
            "gate": "empirical_vol_clustering_baseline",
            "pass": (acf is not None) and (acf > 0.10),
            "observed": acf,
            "expected": "> 0.10",
        }
    )
    gates.append(
        {
            "gate": "empirical_ofi_drop_below_pre",
            "pass": (ofi_pre is not None) and (ofi_drop is not None) and (ofi_drop < ofi_pre),
            "observed": {"pre": ofi_pre, "drop": ofi_drop},
            "expected": "ofi_drop < ofi_pre",
        }
    )
    gates.append(
        {
            "gate": "empirical_crash_rate_proxy_reasonable",
            "pass": True,
            "observed": "pending_simulation_output",
            "expected": "evaluate crash_rate_sim in [0.05, 0.40] after mini simulation",
        }
    )

    return gates


def recommendation_block(prior: dict[str, Any], contrarian_sell_target: float) -> list[str]:
    side_row = find_row(prior["side_by_agent"], "contrarian_trader") or {}
    current_sell = float(side_row.get("sell", 0.0) or 0.0)
    deficit = max(contrarian_sell_target - current_sell, 0.0)
    if deficit <= 1e-9:
        return [
            "Contrarian sell-side already meets target; keep current prior and proceed to mini simulation.",
        ]

    buy_now = float(side_row.get("buy", 0.0) or 0.0)
    do_nothing_now = float(side_row.get("do_nothing", 0.0) or 0.0)
    transfer_from_buy = min(deficit, buy_now)
    transfer_from_idle = max(deficit - transfer_from_buy, 0.0)

    adjusted_buy = max(buy_now - transfer_from_buy, 0.0)
    adjusted_idle = max(do_nothing_now - transfer_from_idle, 0.0)
    adjusted_sell = current_sell + transfer_from_buy + transfer_from_idle

    scale = adjusted_buy + adjusted_idle + adjusted_sell
    if scale > 0:
        adjusted_buy /= scale
        adjusted_idle /= scale
        adjusted_sell /= scale

    return [
        (
            "Contrarian sell-side is too low for a two-sided mean-reversion archetype. "
            f"Current sell={current_sell:.3f}, target={contrarian_sell_target:.3f}."
        ),
        (
            "Suggested temporary rebalancing for mini simulation: "
            f"buy={adjusted_buy:.3f}, do_nothing={adjusted_idle:.3f}, sell={adjusted_sell:.3f}."
        ),
        "After rebalancing, run 50-100 mini simulation runs and compare stylized facts against empirical benchmarks in this report.",
    ]


def write_markdown(
    output_path: Path,
    prior: dict[str, Any],
    empirical: dict[str, Any],
    gates: list[dict[str, Any]],
    recommendations: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pass_count = sum(1 for gate in gates if gate["pass"])
    total = len(gates)
    failed = [gate for gate in gates if not gate["pass"]]

    lines: list[str] = []
    lines.append("# Phase 2 Mini-Simulation Sanity Check")
    lines.append("")
    lines.append(f"Overall gates passed: {pass_count}/{total}")
    lines.append("")
    if failed:
        lines.append("## Blocking findings")
        for gate in failed:
            lines.append(
                f"- {gate['gate']}: observed={json.dumps(gate['observed'])}, expected={gate['expected']}"
            )
        lines.append("")

    lines.append("## Prior sanity snapshot")
    lines.append(f"- parsed_rows: {prior['parsed_rows']}")
    lines.append(f"- cancel_probability_unique: {prior['cancel_probability_unique']}")
    lines.append("")
    lines.append("### cancel_probability by agent")
    lines.append("| agent_type | mean | std | min | max | nunique |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in prior["cancel_probability_stats"]:
        lines.append(
            "| {agent_type} | {mean} | {std} | {min} | {max} | {nunique} |".format(**row)
        )

    lines.append("")
    lines.append("### order_type share by agent")
    lines.append("| agent_type | limit | market |")
    lines.append("|---|---:|---:|")
    for row in prior["order_type_by_agent"]:
        lines.append(
            "| {agent_type} | {limit} | {market} |".format(
                agent_type=row.get("agent_type", ""),
                limit=row.get("limit", 0.0),
                market=row.get("market", 0.0),
            )
        )

    lines.append("")
    lines.append("### side share by agent")
    lines.append("| agent_type | buy | do_nothing | sell |")
    lines.append("|---|---:|---:|---:|")
    for row in prior["side_by_agent"]:
        lines.append(
            "| {agent_type} | {buy} | {do_nothing} | {sell} |".format(
                agent_type=row.get("agent_type", ""),
                buy=row.get("buy", 0.0),
                do_nothing=row.get("do_nothing", 0.0),
                sell=row.get("sell", 0.0),
            )
        )

    lines.append("")
    lines.append("## Empirical stylized-fact baseline")
    lines.append(f"- events: {empirical['events']}")
    lines.append(f"- rows: {empirical['rows']}")
    lines.append(f"- rows_observed: {empirical['rows_observed']}")
    lines.append(f"- observed_ratio: {empirical['observed_ratio']}")
    lines.append(f"- returns_count: {empirical['returns_count']}")
    lines.append(f"- kurtosis_excess: {empirical['kurtosis_excess']}")
    lines.append(f"- acf_vol_lag1: {empirical['acf_vol_lag1']}")
    lines.append(f"- ofi_pre_mean: {empirical['ofi_pre_mean']}")
    lines.append(f"- ofi_drop_mean: {empirical['ofi_drop_mean']}")
    lines.append(f"- crash_rate_proxy: {empirical['crash_rate_proxy']}")

    lines.append("")
    lines.append("## Gate checklist")
    for gate in gates:
        status = "PASS" if gate["pass"] else "FAIL"
        lines.append(
            f"- [{status}] {gate['gate']} | observed={json.dumps(gate['observed'])} | expected={gate['expected']}"
        )

    lines.append("")
    lines.append("## Recommendations")
    for item in recommendations:
        lines.append(f"- {item}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.elicited_csv.exists():
        raise FileNotFoundError(f"Missing elicited CSV: {args.elicited_csv}")
    if not args.gridded_csv.exists():
        raise FileNotFoundError(f"Missing gridded CSV: {args.gridded_csv}")

    parsed = load_parsed_rows(args.elicited_csv)
    prior = prior_summary(parsed)

    gridded = pd.read_csv(args.gridded_csv)
    empirical = empirical_benchmarks(gridded, crash_threshold_pct=args.crash_threshold_pct)

    gates = evaluate_gates(
        prior=prior,
        empirical=empirical,
        min_parsed_rows=args.min_parsed_rows,
        contrarian_sell_floor=args.contrarian_sell_floor,
    )
    recommendations = recommendation_block(prior, args.contrarian_sell_target)

    payload = {
        "inputs": {
            "elicited_csv": str(args.elicited_csv),
            "gridded_csv": str(args.gridded_csv),
            "min_parsed_rows": args.min_parsed_rows,
            "contrarian_sell_floor": args.contrarian_sell_floor,
            "contrarian_sell_target": args.contrarian_sell_target,
            "crash_threshold_pct": args.crash_threshold_pct,
        },
        "prior": prior,
        "empirical": empirical,
        "gates": gates,
        "recommendations": recommendations,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.report_md, prior, empirical, gates, recommendations)

    pass_count = sum(1 for gate in gates if gate["pass"])
    print("=" * 70)
    print("Script 17: Pre-simulation sanity check")
    print("=" * 70)
    print(f"Parsed rows: {prior['parsed_rows']}")
    print(f"Gate pass count: {pass_count}/{len(gates)}")
    print(f"Markdown report: {args.report_md}")
    print(f"JSON report: {args.report_json}")
    print("Done.")


if __name__ == "__main__":
    main()