"""Script 19 - Stylized-facts validation for Phase 2 simulation output.

Compares simulation output against empirical baseline gates and supports
ablation checks for H1:
    kurtosis_llm_prior > kurtosis_uniform
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_LLM = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_simulation_llm.csv"
DEFAULT_SIM_UNIFORM = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_simulation_uniform.csv"
DEFAULT_SIM_LITERATURE = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_simulation_literature.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "validation" / "phase2_stylised_facts_validation.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "validation" / "phase2_stylised_facts_validation.md"

EMPIRICAL_BASELINE = {
    "kurtosis_excess": 8581.0,
    "acf_vol_lag1": 1.0,
    "ofi_drop_mean": -2.70,
    "ofi_pre_mean": 0.26,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stylized facts from simulated LOB output")
    parser.add_argument("--sim-llm", type=Path, default=DEFAULT_SIM_LLM)
    parser.add_argument("--sim-uniform", type=Path, default=DEFAULT_SIM_UNIFORM)
    parser.add_argument("--sim-literature", type=Path, default=DEFAULT_SIM_LITERATURE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--crash-rate-min", type=float, default=0.05)
    parser.add_argument("--crash-rate-max", type=float, default=0.40)
    return parser.parse_args()


def lag1_autocorr(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return float("nan")
    a = s.iloc[:-1]
    b = s.iloc[1:]
    if a.std(ddof=0) < 1e-12 or b.std(ddof=0) < 1e-12:
        return float("nan")
    return float(a.corr(b))


def compute_metrics(sim_path: Path) -> dict[str, Any]:
    if not sim_path.exists():
        return {"exists": False}

    df = pd.read_csv(sim_path)
    required_cols = {"run_id", "phase", "close", "ofi", "flash_crash_flag"}
    if missing := (required_cols - set(df.columns)):
        raise ValueError(f"Missing columns in {sim_path}: {sorted(missing)}")

    df = df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ofi"] = pd.to_numeric(df["ofi"], errors="coerce")
    df["flash_crash_flag"] = pd.to_numeric(df["flash_crash_flag"], errors="coerce").fillna(0).astype(int)

    returns = df.groupby("run_id", group_keys=False)["close"].apply(lambda s: np.log(s / s.shift(1))).dropna()
    returns_sq = returns.pow(2)

    kurtosis_excess = float(returns.kurt()) if len(returns) else float("nan")
    acf_vol_lag1 = lag1_autocorr(returns_sq)

    ofi_phase = df.groupby("phase")["ofi"].mean().to_dict()
    ofi_pre = float(ofi_phase.get("pre", np.nan))
    ofi_drop = float(ofi_phase.get("drop", np.nan))

    run_crashes = df.groupby("run_id")["flash_crash_flag"].max()
    crash_rate_sim = float(run_crashes.mean()) if len(run_crashes) else float("nan")

    return {
        "exists": True,
        "path": str(sim_path),
        "rows": int(len(df)),
        "runs": int(df["run_id"].nunique()),
        "kurtosis_excess": kurtosis_excess,
        "acf_vol_lag1": acf_vol_lag1,
        "ofi_pre_mean": ofi_pre,
        "ofi_drop_mean": ofi_drop,
        "crash_rate_sim": crash_rate_sim,
    }


def evaluate_gates(llm: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []

    gates.append(
        {
            "gate": "kurtosis_excess",
            "pass": llm["kurtosis_excess"] > 3.0,
            "observed": llm["kurtosis_excess"],
            "target": "> 3.0",
            "empirical_baseline": EMPIRICAL_BASELINE["kurtosis_excess"],
        }
    )
    gates.append(
        {
            "gate": "acf_vol_lag1",
            "pass": llm["acf_vol_lag1"] > 0.10,
            "observed": llm["acf_vol_lag1"],
            "target": "> 0.10",
            "empirical_baseline": EMPIRICAL_BASELINE["acf_vol_lag1"],
        }
    )
    gates.append(
        {
            "gate": "ofi_drop_less_than_pre",
            "pass": llm["ofi_drop_mean"] < llm["ofi_pre_mean"],
            "observed": {
                "ofi_pre_mean": llm["ofi_pre_mean"],
                "ofi_drop_mean": llm["ofi_drop_mean"],
            },
            "target": "ofi_drop_mean < ofi_pre_mean",
            "empirical_baseline": {
                "ofi_pre_mean": EMPIRICAL_BASELINE["ofi_pre_mean"],
                "ofi_drop_mean": EMPIRICAL_BASELINE["ofi_drop_mean"],
            },
        }
    )
    gates.append(
        {
            "gate": "crash_rate_sim",
            "pass": args.crash_rate_min <= llm["crash_rate_sim"] <= args.crash_rate_max,
            "observed": llm["crash_rate_sim"],
            "target": f"in [{args.crash_rate_min:.2f}, {args.crash_rate_max:.2f}]",
            "empirical_baseline": "pending_real_world_mapping",
        }
    )

    return gates


def build_recommendations(gates: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    gate_lookup = {g["gate"]: g for g in gates}

    if not gate_lookup["crash_rate_sim"]["pass"]:
        obs = float(gate_lookup["crash_rate_sim"]["observed"])
        if obs > 0.40:
            recs.append("crash_rate_sim > 0.40: increase MM withdrawal threshold or reduce withdrawal strength")
        elif obs < 0.05:
            recs.append("crash_rate_sim < 0.05: increase sell pressure in drop phase or raise leverage amplification")

    if not gate_lookup["acf_vol_lag1"]["pass"]:
        recs.append("acf_vol_lag1 too low: increase kyle_lambda impact-scale or persistence in volatility process")

    if not gate_lookup["ofi_drop_less_than_pre"]["pass"]:
        recs.append("OFI phase ordering invalid: verify phase-conditioned side probabilities and Poisson rates")

    if not recs:
        recs.append("All stylized-fact gates pass for current thresholds")

    return recs


def to_markdown(report: dict[str, Any]) -> str:
    llm = report["metrics"]["llm"]
    uniform = report["metrics"]["uniform"]
    literature = report["metrics"]["literature"]

    lines: list[str] = []
    lines.append("# Phase 2 Stylised Facts Validation")
    lines.append("")
    lines.append("## LLM Metrics")
    lines.append("")
    lines.append(f"- rows: {llm.get('rows')}")
    lines.append(f"- runs: {llm.get('runs')}")
    lines.append(f"- kurtosis_excess: {llm.get('kurtosis_excess')}")
    lines.append(f"- acf_vol_lag1: {llm.get('acf_vol_lag1')}")
    lines.append(f"- ofi_pre_mean: {llm.get('ofi_pre_mean')}")
    lines.append(f"- ofi_drop_mean: {llm.get('ofi_drop_mean')}")
    lines.append(f"- crash_rate_sim: {llm.get('crash_rate_sim')}")
    lines.append("")

    lines.append("## Ablation")
    lines.append("")
    if uniform.get("exists"):
        lines.append(f"- kurtosis_uniform: {uniform.get('kurtosis_excess')}")
    else:
        lines.append("- kurtosis_uniform: missing input")

    if literature.get("exists"):
        lines.append(f"- kurtosis_literature: {literature.get('kurtosis_excess')}")
    else:
        lines.append("- kurtosis_literature: missing input")

    lines.append(f"- h1_pass_kurtosis_llm_gt_uniform: {report['ablation']['h1_pass_kurtosis_llm_gt_uniform']}")
    lines.append("")

    lines.append("## Gate Results")
    lines.append("")
    for gate in report["gates"]:
        lines.append(f"- {gate['gate']}: pass={gate['pass']} observed={gate['observed']} target={gate['target']}")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for rec in report["recommendations"]:
        lines.append(f"- {rec}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.sim_llm.exists():
        raise FileNotFoundError(f"LLM simulation output not found: {args.sim_llm}")

    llm_metrics = compute_metrics(args.sim_llm)
    uniform_metrics = compute_metrics(args.sim_uniform)
    literature_metrics = compute_metrics(args.sim_literature)

    gates = evaluate_gates(llm_metrics, args)

    h1_pass = None
    if uniform_metrics.get("exists"):
        h1_pass = bool(llm_metrics["kurtosis_excess"] > uniform_metrics["kurtosis_excess"])

    report = {
        "metadata": {
            "script": "19_stylised_facts_validation.py",
            "empirical_baseline": EMPIRICAL_BASELINE,
        },
        "metrics": {
            "llm": llm_metrics,
            "uniform": uniform_metrics,
            "literature": literature_metrics,
        },
        "ablation": {
            "h1_pass_kurtosis_llm_gt_uniform": h1_pass,
            "kurtosis_ratio_llm_over_uniform": (
                llm_metrics["kurtosis_excess"] / uniform_metrics["kurtosis_excess"]
                if uniform_metrics.get("exists") and uniform_metrics["kurtosis_excess"] not in (0, None)
                else None
            ),
        },
        "gates": gates,
        "all_gates_pass": bool(all(g["pass"] for g in gates)),
        "recommendations": build_recommendations(gates),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(report), encoding="utf-8")

    print("=" * 72)
    print("Script 19: Stylised facts validation")
    print("=" * 72)
    print(f"LLM panel        : {args.sim_llm}")
    print(f"Uniform panel    : {args.sim_uniform} ({'found' if uniform_metrics.get('exists') else 'missing'})")
    print(
        f"Literature panel : {args.sim_literature} "
        f"({'found' if literature_metrics.get('exists') else 'missing'})"
    )
    print(f"All gates pass   : {report['all_gates_pass']}")
    print(f"Saved JSON report: {args.report_json}")
    print(f"Saved MD report  : {args.report_md}")


if __name__ == "__main__":
    main()
