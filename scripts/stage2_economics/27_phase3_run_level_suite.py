"""Script 27 - Phase 3 run-level main experiments and ablation suite.

Uses the locked 500-run canonical LLM panel, aggregates to one row per run via
the Script 26 helpers, then runs:

1. Main run-level causal discovery
2. Main run-level intervention model
3. Compact ablation studies over pre-gap settings and feature presets
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT26_PATH = PROJECT_ROOT / "scripts" / "stage2_economics" / "26_run_level_causal_discovery.py"
DEFAULT_SIM_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tardis"
    / "phase2_outputs"
    / "phase2_canonical_tuned_legacy_500runs"
    / "lob_full_simulation_llm_tuned_legacy.csv"
)
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "validation" / "phase3_run_level_suite.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "validation" / "phase3_run_level_suite.md"
DEFAULT_SUMMARY_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tardis"
    / "phase2_outputs"
    / "phase2_canonical_tuned_legacy_500runs"
    / "phase3_run_level_ablation_summary.csv"
)

CAUSAL_NODE_PRESETS = {
    "crash_only": [
        "ofi_pre_mean",
        "spread_pre_mean",
        "depth_imb_pre_mean",
        "leverage_pre_max",
        "kyle_lambda_pre_mean",
        "mean_wealth_pre_mean",
        "pct_insolvent_pre_max",
        "wealth_concentration_pre_mean",
        "crashed",
    ],
    "crash_plus_severity": [
        "ofi_pre_mean",
        "spread_pre_mean",
        "depth_imb_pre_mean",
        "leverage_pre_max",
        "kyle_lambda_pre_mean",
        "mean_wealth_pre_mean",
        "pct_insolvent_pre_max",
        "wealth_concentration_pre_mean",
        "crashed",
        "crash_severity_pct",
    ],
}

INTERVENTION_FEATURE_PRESETS = {
    "market_core": [
        "ofi_pre_mean",
        "spread_pre_mean",
        "depth_imb_pre_mean",
        "leverage_pre_max",
        "kyle_lambda_pre_mean",
    ],
    "market_plus_wealth": [
        "ofi_pre_mean",
        "spread_pre_mean",
        "depth_imb_pre_mean",
        "leverage_pre_max",
        "kyle_lambda_pre_mean",
        "mean_wealth_pre_mean",
        "pct_insolvent_pre_max",
        "wealth_concentration_pre_mean",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 run-level suite")
    parser.add_argument("--sim-panel", type=Path, default=DEFAULT_SIM_PANEL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--min-pre-rows", type=int, default=25)
    parser.add_argument("--min-variance", type=float, default=1e-6)
    parser.add_argument("--weight-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_script26_module() -> Any:
    spec = importlib.util.spec_from_file_location("phase3_run_level_base", SCRIPT26_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Script 26 from {SCRIPT26_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_floor_touch_summary(sim_panel_path: Path, min_price_fraction: float = 0.60, floor_eps_mult: float = 1.001) -> dict[str, Any]:
    df = pd.read_csv(sim_panel_path, usecols=["run_id", "tick_ms", "phase", "close", "flash_crash_flag"])
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype(int)
    df["tick_ms"] = pd.to_numeric(df["tick_ms"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["flash_crash_flag"] = pd.to_numeric(df["flash_crash_flag"], errors="coerce").fillna(0.0).astype(int)
    df = df.dropna(subset=["run_id", "tick_ms", "phase", "close"]).copy()

    records: list[dict[str, int]] = []
    for run_id, sub in df.groupby("run_id", sort=False):
        sub = sub.sort_values("tick_ms")
        pre = sub[sub["phase"] == "pre"]
        drop = sub[sub["phase"] == "drop"]
        init_price = float(pre["close"].iloc[-1]) if len(pre) else float(sub["close"].iloc[0])
        floor_price = init_price * min_price_fraction
        floor_touched = int(len(drop[drop["close"] <= floor_price * floor_eps_mult]) > 0)
        crashed = int(sub["flash_crash_flag"].max() > 0)
        records.append({"run_id": int(run_id), "crashed": crashed, "floor_touched": floor_touched})

    work = pd.DataFrame(records)
    counts = {
        f"crashed={int(crashed)},floor_touched={int(floor_touched)}": int(count)
        for (crashed, floor_touched), count in work.groupby(["crashed", "floor_touched"]).size().items()
    }
    return {
        "floor_touch_rate": float(work["floor_touched"].mean()) if len(work) else 0.0,
        "counts": counts,
    }


def keep_columns_by_variance(df: pd.DataFrame, columns: list[str], min_variance: float) -> tuple[list[str], dict[str, float]]:
    kept: list[str] = []
    variances: dict[str, float] = {}
    for column in columns:
        variance = float(df[column].var(ddof=0))
        variances[column] = variance
        if variance > min_variance:
            kept.append(column)
    return kept, variances


def run_causal_experiment(
    module: Any,
    panel: pd.DataFrame,
    pre_gap_ticks: int,
    min_pre_rows: int,
    node_preset: str,
    min_variance: float,
    weight_threshold: float,
    seed: int,
) -> dict[str, Any]:
    run_panel, panel_summary = module.build_run_level_panel(panel, pre_gap_ticks=pre_gap_ticks, min_pre_rows=min_pre_rows)
    candidate_nodes = CAUSAL_NODE_PRESETS[node_preset]
    kept_nodes, variances = keep_columns_by_variance(run_panel, candidate_nodes, min_variance=min_variance)
    if len(kept_nodes) < 2:
        raise ValueError(f"Too few nodes remain after variance filter for causal preset {node_preset}")

    _, matrix = module.prepare_matrix(run_panel, kept_nodes)
    dagma_edges, dagma_status = module.run_notears(matrix, kept_nodes, weight_threshold=weight_threshold)
    lingam_edges, lingam_status = module.run_direct_lingam(
        matrix,
        kept_nodes,
        weight_threshold=weight_threshold,
        seed=seed,
    )
    dagma_summary = module.summarize_method(dagma_edges, method="dagma", top_edges=10)
    lingam_summary = module.summarize_method(dagma_edges + lingam_edges, method="direct_lingam", top_edges=10)

    return {
        "pre_gap_ticks": int(pre_gap_ticks),
        "node_preset": node_preset,
        "runs_total": panel_summary["runs_total"],
        "crash_runs_total": panel_summary["crash_runs_total"],
        "mean_pre_rows": panel_summary["mean_pre_rows"],
        "window_strategy_counts": panel_summary["window_strategy_counts"],
        "kept_nodes": kept_nodes,
        "dropped_nodes": [column for column in candidate_nodes if column not in kept_nodes],
        "variances": variances,
        "dagma_status": dagma_status,
        "direct_lingam_status": lingam_status,
        "dagma": dagma_summary,
        "direct_lingam": lingam_summary,
    }


def run_intervention_experiment(
    module: Any,
    panel: pd.DataFrame,
    pre_gap_ticks: int,
    min_pre_rows: int,
    feature_preset: str,
    min_variance: float,
    seed: int,
) -> dict[str, Any]:
    run_panel, panel_summary = module.build_run_level_panel(panel, pre_gap_ticks=pre_gap_ticks, min_pre_rows=min_pre_rows)
    candidate_features = INTERVENTION_FEATURE_PRESETS[feature_preset]
    kept_features, variances = keep_columns_by_variance(run_panel, candidate_features, min_variance=min_variance)
    if len(kept_features) < 1:
        raise ValueError(f"Too few features remain after variance filter for intervention preset {feature_preset}")

    X = run_panel[kept_features].copy()
    y = run_panel["crashed"].astype(int)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=5000, random_state=seed)),
        ]
    )
    clf.fit(X, y)
    p_obs = clf.predict_proba(X)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    observational_rate = float(np.mean(p_obs))
    observational_rate_raw = float(y.mean())
    auc_in_sample = float(roc_auc_score(y, p_obs))
    cv_auc_mean = float(cv_auc_scores.mean())
    cv_auc_std = float(cv_auc_scores.std())

    do_ofi_rate = None
    ofi_reduction_pct = None
    if "ofi_pre_mean" in kept_features:
        X_do_ofi = X.copy()
        X_do_ofi["ofi_pre_mean"] = 0.0
        p_do_ofi = clf.predict_proba(X_do_ofi)[:, 1]
        do_ofi_rate = float(np.mean(p_do_ofi))
        ofi_reduction_pct = float((observational_rate - do_ofi_rate) / max(observational_rate, 1e-9) * 100.0)

    do_leverage_rate = None
    leverage_reduction_pct = None
    severity_reduction_pct = None
    if "leverage_pre_max" in kept_features:
        X_do_lev = X.copy()
        X_do_lev["leverage_pre_max"] = 0.0
        p_do_lev = clf.predict_proba(X_do_lev)[:, 1]
        do_leverage_rate = float(np.mean(p_do_lev))
        leverage_reduction_pct = float((observational_rate - do_leverage_rate) / max(observational_rate, 1e-9) * 100.0)

        severity_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linear", LinearRegression()),
            ]
        )
        severity_model.fit(X, run_panel["crash_severity_pct"])
        severity_obs = float(np.mean(severity_model.predict(X)))
        severity_do_lev = float(np.mean(severity_model.predict(X_do_lev)))
        severity_reduction_pct = float((severity_obs - severity_do_lev) / max(abs(severity_obs), 1e-9) * 100.0)

    group_means = (
        run_panel.groupby("crashed")[kept_features + ["crash_severity_pct"]].mean().reset_index().to_dict(orient="records")
    )

    return {
        "pre_gap_ticks": int(pre_gap_ticks),
        "feature_preset": feature_preset,
        "runs_total": panel_summary["runs_total"],
        "crash_runs_total": panel_summary["crash_runs_total"],
        "mean_pre_rows": panel_summary["mean_pre_rows"],
        "window_strategy_counts": panel_summary["window_strategy_counts"],
        "kept_features": kept_features,
        "dropped_features": [column for column in candidate_features if column not in kept_features],
        "variances": variances,
        "observational_rate_raw": observational_rate_raw,
        "observational_rate_pred": observational_rate,
        "auc_in_sample": auc_in_sample,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "do_ofi_rate_pred": do_ofi_rate,
        "ofi_reduction_pct": ofi_reduction_pct,
        "do_leverage_rate_pred": do_leverage_rate,
        "leverage_reduction_pct": leverage_reduction_pct,
        "severity_reduction_pct": severity_reduction_pct,
        "group_means": group_means,
    }


def flatten_for_csv(report: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    main_causal = report["main"]["causal"]
    rows.append(
        {
            "experiment_type": "main_causal",
            "config": f"pre_gap={main_causal['pre_gap_ticks']},preset={main_causal['node_preset']}",
            "primary_method": "dagma",
            "primary_outcome_edges": " | ".join(main_causal["dagma"]["outcome_edges"]),
            "secondary_method": "direct_lingam",
            "secondary_outcome_edges": " | ".join(main_causal["direct_lingam"]["outcome_edges"]),
        }
    )

    main_intervention = report["main"]["intervention"]
    rows.append(
        {
            "experiment_type": "main_intervention",
            "config": f"pre_gap={main_intervention['pre_gap_ticks']},preset={main_intervention['feature_preset']}",
            "cv_auc_mean": main_intervention["cv_auc_mean"],
            "ofi_reduction_pct": main_intervention["ofi_reduction_pct"],
            "leverage_reduction_pct": main_intervention["leverage_reduction_pct"],
            "severity_reduction_pct": main_intervention["severity_reduction_pct"],
        }
    )

    for row in report["ablations"]["causal"]:
        rows.append(
            {
                "experiment_type": "causal_ablation",
                "config": f"pre_gap={row['pre_gap_ticks']},preset={row['node_preset']}",
                "primary_method": "dagma",
                "primary_outcome_edges": " | ".join(row["dagma"]["outcome_edges"]),
                "secondary_method": "direct_lingam",
                "secondary_outcome_edges": " | ".join(row["direct_lingam"]["outcome_edges"]),
            }
        )

    for row in report["ablations"]["intervention"]:
        rows.append(
            {
                "experiment_type": "intervention_ablation",
                "config": f"pre_gap={row['pre_gap_ticks']},preset={row['feature_preset']}",
                "cv_auc_mean": row["cv_auc_mean"],
                "ofi_reduction_pct": row["ofi_reduction_pct"],
                "leverage_reduction_pct": row["leverage_reduction_pct"],
                "severity_reduction_pct": row["severity_reduction_pct"],
            }
        )

    return pd.DataFrame(rows)


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 3 Run-Level Suite")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- sim_panel: {report['data']['sim_panel']}")
    lines.append(f"- floor_touch_rate_060: {report['data']['floor_touch']['floor_touch_rate']}")
    for key, value in sorted(report["data"]["floor_touch"]["counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Main Causal")
    lines.append("")
    main_causal = report["main"]["causal"]
    lines.append(f"- pre_gap_ticks: {main_causal['pre_gap_ticks']}")
    lines.append(f"- node_preset: {main_causal['node_preset']}")
    lines.append(f"- kept_nodes: {', '.join(main_causal['kept_nodes'])}")
    lines.append(f"- dagma_status: {main_causal['dagma_status']}")
    lines.append(f"- direct_lingam_status: {main_causal['direct_lingam_status']}")
    lines.append(f"- dagma_outcome_edges: {main_causal['dagma']['outcome_edges']}")
    lines.append(f"- direct_lingam_outcome_edges: {main_causal['direct_lingam']['outcome_edges']}")
    lines.append("")

    lines.append("## Main Intervention")
    lines.append("")
    main_intervention = report["main"]["intervention"]
    lines.append(f"- pre_gap_ticks: {main_intervention['pre_gap_ticks']}")
    lines.append(f"- feature_preset: {main_intervention['feature_preset']}")
    lines.append(f"- kept_features: {', '.join(main_intervention['kept_features'])}")
    lines.append(f"- observational_rate_raw: {main_intervention['observational_rate_raw']}")
    lines.append(f"- observational_rate_pred: {main_intervention['observational_rate_pred']}")
    lines.append(f"- auc_in_sample: {main_intervention['auc_in_sample']}")
    lines.append(f"- cv_auc_mean: {main_intervention['cv_auc_mean']}")
    lines.append(f"- cv_auc_std: {main_intervention['cv_auc_std']}")
    lines.append(f"- ofi_reduction_pct: {main_intervention['ofi_reduction_pct']}")
    lines.append(f"- leverage_reduction_pct: {main_intervention['leverage_reduction_pct']}")
    lines.append(f"- severity_reduction_pct: {main_intervention['severity_reduction_pct']}")
    lines.append("")
    lines.append("Group means by crashed:")
    for row in main_intervention["group_means"]:
        lines.append(f"- crashed={int(row['crashed'])}: {row}")
    lines.append("")

    lines.append("## Causal Ablations")
    lines.append("")
    for row in report["ablations"]["causal"]:
        lines.append(
            f"- pre_gap={row['pre_gap_ticks']}, preset={row['node_preset']}: "
            f"dagma_outcome_edges={row['dagma']['outcome_edges']}, "
            f"direct_lingam_outcome_edges={row['direct_lingam']['outcome_edges']}"
        )
    lines.append("")

    lines.append("## Intervention Ablations")
    lines.append("")
    for row in report["ablations"]["intervention"]:
        lines.append(
            f"- pre_gap={row['pre_gap_ticks']}, preset={row['feature_preset']}: "
            f"cv_auc_mean={row['cv_auc_mean']}, "
            f"ofi_reduction_pct={row['ofi_reduction_pct']}, "
            f"leverage_reduction_pct={row['leverage_reduction_pct']}, "
            f"severity_reduction_pct={row['severity_reduction_pct']}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.sim_panel.exists():
        raise FileNotFoundError(f"Simulation panel not found: {args.sim_panel}")

    module = load_script26_module()
    panel = module.clean_panel(args.sim_panel)
    floor_touch = compute_floor_touch_summary(args.sim_panel)

    main_causal = run_causal_experiment(
        module,
        panel,
        pre_gap_ticks=20,
        min_pre_rows=args.min_pre_rows,
        node_preset="crash_only",
        min_variance=args.min_variance,
        weight_threshold=args.weight_threshold,
        seed=args.seed,
    )
    main_intervention = run_intervention_experiment(
        module,
        panel,
        pre_gap_ticks=20,
        min_pre_rows=args.min_pre_rows,
        feature_preset="market_core",
        min_variance=args.min_variance,
        seed=args.seed,
    )

    causal_ablation_configs = [
        {"pre_gap_ticks": 0, "node_preset": "crash_only"},
        {"pre_gap_ticks": 20, "node_preset": "crash_only"},
        {"pre_gap_ticks": 50, "node_preset": "crash_only"},
        {"pre_gap_ticks": 20, "node_preset": "crash_plus_severity"},
    ]
    causal_ablations = [
        run_causal_experiment(
            module,
            panel,
            pre_gap_ticks=config["pre_gap_ticks"],
            min_pre_rows=args.min_pre_rows,
            node_preset=config["node_preset"],
            min_variance=args.min_variance,
            weight_threshold=args.weight_threshold,
            seed=args.seed,
        )
        for config in causal_ablation_configs
    ]

    intervention_ablations: list[dict[str, Any]] = []
    for pre_gap_ticks in [0, 20, 50]:
        for feature_preset in ["market_core", "market_plus_wealth"]:
            intervention_ablations.append(
                run_intervention_experiment(
                    module,
                    panel,
                    pre_gap_ticks=pre_gap_ticks,
                    min_pre_rows=args.min_pre_rows,
                    feature_preset=feature_preset,
                    min_variance=args.min_variance,
                    seed=args.seed,
                )
            )

    report = {
        "data": {
            "sim_panel": str(args.sim_panel),
            "floor_touch": floor_touch,
            "min_pre_rows": int(args.min_pre_rows),
            "min_variance": float(args.min_variance),
            "weight_threshold": float(args.weight_threshold),
        },
        "main": {
            "causal": main_causal,
            "intervention": main_intervention,
        },
        "ablations": {
            "causal": causal_ablations,
            "intervention": intervention_ablations,
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(report), encoding="utf-8")

    summary_df = flatten_for_csv(report)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_csv, index=False)

    print("=" * 72)
    print("Script 27: Phase 3 run-level suite")
    print("=" * 72)
    print(f"Simulation panel   : {args.sim_panel}")
    print(f"Main causal preset : pre_gap=20, crash_only")
    print(f"Main DAGMA edges   : {main_causal['dagma']['outcome_edges']}")
    print(f"Main intervention  : pre_gap=20, market_core")
    print(f"Main CV AUC        : {main_intervention['cv_auc_mean']:.4f}")
    print(f"Main do(OFI=0)     : {main_intervention['ofi_reduction_pct']:.2f}%")
    print(f"Main do(lev=0)     : {main_intervention['leverage_reduction_pct']:.2f}%")
    print(f"Saved JSON report  : {args.report_json}")
    print(f"Saved MD report    : {args.report_md}")
    print(f"Saved summary CSV  : {args.summary_csv}")


if __name__ == "__main__":
    main()