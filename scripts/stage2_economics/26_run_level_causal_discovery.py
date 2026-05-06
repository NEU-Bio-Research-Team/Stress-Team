"""Script 26 - Run-level causal discovery from the locked 500-run panel.

Aggregates the full tick-level simulation panel into one row per run using
pre-crash summary statistics, then runs cross-sectional causal discovery on the
aggregated panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tardis"
    / "phase2_outputs"
    / "phase2_canonical_tuned_legacy_500runs"
    / "lob_full_simulation_llm_tuned_legacy.csv"
)
DEFAULT_RUN_PANEL_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tardis"
    / "phase2_outputs"
    / "phase2_canonical_tuned_legacy_500runs"
    / "run_level_causal_panel_llm_tuned_legacy.csv"
)
DEFAULT_EDGES_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tardis"
    / "phase2_outputs"
    / "phase2_canonical_tuned_legacy_500runs"
    / "run_level_causal_edges_llm_tuned_legacy.csv"
)
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "validation" / "phase3_run_level_causal_discovery.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "validation" / "phase3_run_level_causal_discovery.md"

RAW_FEATURES = [
    "ofi",
    "spread_bps",
    "depth_imbalance",
    "leverage_proxy",
    "kyle_lambda",
    "vpin",
    "mean_wealth_t",
    "pct_insolvent",
    "wealth_concentration",
]

RUN_LEVEL_COLUMNS = [
    "ofi_pre_mean",
    "spread_pre_mean",
    "depth_imb_pre_mean",
    "leverage_pre_max",
    "kyle_lambda_pre_mean",
    "vpin_pre_max",
    "mean_wealth_pre_mean",
    "pct_insolvent_pre_max",
    "wealth_concentration_pre_mean",
    "crashed",
    "crash_severity_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run-level causal discovery on the locked 500-run panel")
    parser.add_argument("--sim-panel", type=Path, default=DEFAULT_SIM_PANEL)
    parser.add_argument("--run-panel-csv", type=Path, default=DEFAULT_RUN_PANEL_CSV)
    parser.add_argument("--edges-csv", type=Path, default=DEFAULT_EDGES_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--pre-gap-ticks", type=int, default=20)
    parser.add_argument("--min-pre-rows", type=int, default=25)
    parser.add_argument("--min-variance", type=float, default=1e-6)
    parser.add_argument("--weight-threshold", type=float, default=0.05)
    parser.add_argument("--top-edges", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_panel(path: Path) -> pd.DataFrame:
    required = ["run_id", "tick_ms", "close", "flash_crash_flag"] + RAW_FEATURES
    frame = pd.read_csv(path, usecols=required)

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["run_id", "tick_ms", "close", "flash_crash_flag"] + RAW_FEATURES).copy()
    frame["run_id"] = frame["run_id"].astype(int)
    frame["flash_crash_flag"] = frame["flash_crash_flag"].astype(int)
    return frame.sort_values(["run_id", "tick_ms"]).reset_index(drop=True)


def compute_crash_severity_pct(close: pd.Series) -> float:
    start_close = float(close.iloc[0])
    min_close = float(close.min())
    if start_close == 0.0:
        return 0.0
    return max(0.0, (start_close - min_close) / start_close * 100.0)


def select_pre_window(run_frame: pd.DataFrame, pre_gap_ticks: int, min_pre_rows: int) -> tuple[pd.DataFrame, str]:
    crash_rows = run_frame[run_frame["flash_crash_flag"] > 0]
    if len(crash_rows) == 0:
        return run_frame.copy(), "full_run_no_crash"

    crash_tick = int(crash_rows["run_tick_index"].iloc[0])
    pre_frame = run_frame[run_frame["run_tick_index"] < crash_tick - pre_gap_ticks].copy()
    if len(pre_frame) >= min_pre_rows:
        return pre_frame, "before_crash_minus_gap"

    pre_frame = run_frame[run_frame["run_tick_index"] < crash_tick].copy()
    if len(pre_frame) >= min_pre_rows:
        return pre_frame, "before_crash"

    return run_frame.copy(), "full_run_fallback"


def build_run_level_panel(df: pd.DataFrame, pre_gap_ticks: int, min_pre_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    window_strategies: dict[str, int] = {}

    for run_id, sub in df.groupby("run_id", sort=False):
        sub = sub.sort_values("tick_ms").copy()
        sub["run_tick_index"] = np.arange(len(sub), dtype=int)
        pre_frame, strategy = select_pre_window(sub, pre_gap_ticks=pre_gap_ticks, min_pre_rows=min_pre_rows)
        window_strategies[strategy] = window_strategies.get(strategy, 0) + 1

        crash_rows = sub[sub["flash_crash_flag"] > 0]
        crash_tick = int(crash_rows["run_tick_index"].iloc[0]) if len(crash_rows) else -1

        record = {
            "run_id": int(run_id),
            "run_rows": int(len(sub)),
            "pre_rows": int(len(pre_frame)),
            "crashed": int(len(crash_rows) > 0),
            "first_crash_tick": crash_tick,
            "window_strategy": strategy,
            "ofi_pre_mean": float(pre_frame["ofi"].mean()),
            "spread_pre_mean": float(pre_frame["spread_bps"].mean()),
            "depth_imb_pre_mean": float(pre_frame["depth_imbalance"].mean()),
            "leverage_pre_max": float(pre_frame["leverage_proxy"].max()),
            "kyle_lambda_pre_mean": float(pre_frame["kyle_lambda"].mean()),
            "vpin_pre_max": float(pre_frame["vpin"].max()),
            "mean_wealth_pre_mean": float(pre_frame["mean_wealth_t"].mean()),
            "pct_insolvent_pre_max": float(pre_frame["pct_insolvent"].max()),
            "wealth_concentration_pre_mean": float(pre_frame["wealth_concentration"].mean()),
            "crash_severity_pct": float(compute_crash_severity_pct(sub["close"])),
        }
        records.append(record)

    run_panel = pd.DataFrame(records).sort_values("run_id").reset_index(drop=True)
    summary = {
        "runs_total": int(run_panel["run_id"].nunique()),
        "crash_runs_total": int(run_panel["crashed"].sum()),
        "mean_pre_rows": float(run_panel["pre_rows"].mean()) if len(run_panel) else 0.0,
        "min_pre_rows": int(run_panel["pre_rows"].min()) if len(run_panel) else 0,
        "max_pre_rows": int(run_panel["pre_rows"].max()) if len(run_panel) else 0,
        "window_strategy_counts": window_strategies,
    }
    return run_panel, summary


def variance_filter(df: pd.DataFrame, min_variance: float) -> tuple[list[str], dict[str, float]]:
    variances: dict[str, float] = {}
    kept: list[str] = []

    for column in RUN_LEVEL_COLUMNS:
        variance = float(df[column].var(ddof=0))
        variances[column] = variance
        if variance > min_variance:
            kept.append(column)

    return kept, variances


def prepare_matrix(df: pd.DataFrame, node_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    work = df[["run_id"] + node_names].copy()
    scaler = StandardScaler()
    matrix = scaler.fit_transform(work[node_names].to_numpy(dtype=float))
    scaled = pd.concat(
        [
            work[["run_id"]].reset_index(drop=True),
            pd.DataFrame(matrix, columns=node_names),
        ],
        axis=1,
    )
    return scaled, matrix


def run_notears(matrix: np.ndarray, node_names: list[str], weight_threshold: float) -> tuple[list[dict[str, Any]], str]:
    try:
        from dagma.linear import DagmaLinear  # type: ignore
    except Exception as exc:
        return [], f"dagma_unavailable: {exc}"

    try:
        model = DagmaLinear(loss_type="l2")
        weights = model.fit(matrix, lambda1=0.02, w_threshold=0.0)
    except Exception as exc:
        return [], f"dagma_failed: {exc}"

    edges: list[dict[str, Any]] = []
    for child_i, child in enumerate(node_names):
        for parent_j, parent in enumerate(node_names):
            if child_i == parent_j:
                continue
            weight = float(weights[child_i, parent_j])
            if abs(weight) < weight_threshold:
                continue
            edges.append({"method": "dagma", "source": parent, "target": child, "weight": weight, "lag": 0})
    return edges, "ok"


def run_direct_lingam(matrix: np.ndarray, node_names: list[str], weight_threshold: float, seed: int) -> tuple[list[dict[str, Any]], str]:
    try:
        from lingam import DirectLiNGAM  # type: ignore
    except Exception as exc:
        return [], f"direct_lingam_unavailable: {exc}"

    try:
        model = DirectLiNGAM(random_state=seed)
        model.fit(matrix)
        adjacency = np.asarray(model.adjacency_matrix_, dtype=float)
    except Exception as exc:
        return [], f"direct_lingam_failed: {exc}"

    edges: list[dict[str, Any]] = []
    for child_i, child in enumerate(node_names):
        for parent_j, parent in enumerate(node_names):
            if child_i == parent_j:
                continue
            weight = float(adjacency[child_i, parent_j])
            if abs(weight) < weight_threshold:
                continue
            edges.append({"method": "direct_lingam", "source": parent, "target": child, "weight": weight, "lag": 0})
    return edges, "ok"


def strongest_edges_by_pair(edges: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    strongest: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        if edge["method"] != method:
            continue
        key = (str(edge["source"]), str(edge["target"]))
        current = strongest.get(key)
        if current is None or abs(float(edge["weight"])) > abs(float(current["weight"])):
            strongest[key] = edge
    return sorted(strongest.values(), key=lambda item: abs(float(item["weight"])), reverse=True)


def summarize_method(edges: list[dict[str, Any]], method: str, top_edges: int) -> dict[str, Any]:
    strongest = strongest_edges_by_pair(edges, method=method)
    top = [
        f"{edge['source']}->{edge['target']} ({float(edge['weight']):.3f})"
        for edge in strongest[:top_edges]
    ]
    outcome_edges = [
        f"{edge['source']}->{edge['target']} ({float(edge['weight']):.3f})"
        for edge in strongest
        if edge["target"] in {"crashed", "crash_severity_pct"}
    ][:top_edges]
    return {
        "method": method,
        "num_edges": len(strongest),
        "top_edges": top,
        "outcome_edges": outcome_edges,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 3 Run-Level Causal Discovery")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- sim_panel: {report['data']['sim_panel']}")
    lines.append(f"- run_panel_csv: {report['data']['run_panel_csv']}")
    lines.append(f"- runs_total: {report['data']['runs_total']}")
    lines.append(f"- crash_runs_total: {report['data']['crash_runs_total']}")
    lines.append(f"- pre_gap_ticks: {report['data']['pre_gap_ticks']}")
    lines.append(f"- mean_pre_rows: {report['data']['mean_pre_rows']}")
    lines.append(f"- min_pre_rows: {report['data']['min_pre_rows']}")
    lines.append(f"- max_pre_rows: {report['data']['max_pre_rows']}")
    lines.append(f"- crash_rate: {report['data']['crash_rate']:.3f}")
    lines.append(f"- kept_nodes: {', '.join(report['data']['kept_nodes'])}")
    lines.append(f"- dropped_nodes: {', '.join(report['data']['dropped_nodes']) or 'none'}")
    lines.append(f"- dagma_status: {report['data']['dagma_status']}")
    lines.append(f"- direct_lingam_status: {report['data']['direct_lingam_status']}")
    lines.append(f"- pcmci_status: {report['data']['pcmci_status']}")
    lines.append("")
    lines.append("## Variance")
    lines.append("")
    for column, variance in report["variance"].items():
        lines.append(f"- {column}: {variance:.6g}")
    lines.append("")
    lines.append("## Window Strategies")
    lines.append("")
    for strategy, count in sorted(report["data"]["window_strategy_counts"].items()):
        lines.append(f"- {strategy}: {count}")
    lines.append("")
    lines.append("## Method Summaries")
    lines.append("")
    for row in report["method_summaries"]:
        lines.append(f"- {row['method']}: edges={row['num_edges']}")
        if row["outcome_edges"]:
            lines.append(f"  outcome_edges={row['outcome_edges']}")
        if row["top_edges"]:
            lines.append(f"  top_edges={row['top_edges']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.sim_panel.exists():
        raise FileNotFoundError(f"Simulation panel not found: {args.sim_panel}")

    panel = clean_panel(args.sim_panel)
    run_panel, panel_summary = build_run_level_panel(
        panel,
        pre_gap_ticks=args.pre_gap_ticks,
        min_pre_rows=args.min_pre_rows,
    )

    kept_nodes, variances = variance_filter(run_panel, min_variance=args.min_variance)
    dropped_nodes = [column for column in RUN_LEVEL_COLUMNS if column not in kept_nodes]
    if len(kept_nodes) < 2:
        raise ValueError("Variance filter removed too many nodes for causal discovery")

    scaled_panel, matrix = prepare_matrix(run_panel, kept_nodes)
    dagma_edges, dagma_status = run_notears(matrix, kept_nodes, weight_threshold=args.weight_threshold)
    direct_lingam_edges, direct_lingam_status = run_direct_lingam(
        matrix,
        kept_nodes,
        weight_threshold=args.weight_threshold,
        seed=args.seed,
    )
    pcmci_status = "not_run: run-level panel is cross-sectional rather than an ordered time series"

    all_edges = dagma_edges + direct_lingam_edges
    edges_df = pd.DataFrame(all_edges)

    args.run_panel_csv.parent.mkdir(parents=True, exist_ok=True)
    run_panel.to_csv(args.run_panel_csv, index=False)

    args.edges_csv.parent.mkdir(parents=True, exist_ok=True)
    if edges_df.empty:
        pd.DataFrame(columns=["method", "source", "target", "weight", "lag"]).to_csv(args.edges_csv, index=False)
    else:
        edges_df.to_csv(args.edges_csv, index=False)

    method_summaries = []
    for method in ["dagma", "direct_lingam"]:
        method_summaries.append(summarize_method(all_edges, method=method, top_edges=args.top_edges))

    report = {
        "data": {
            "sim_panel": str(args.sim_panel),
            "run_panel_csv": str(args.run_panel_csv),
            "runs_total": panel_summary["runs_total"],
            "crash_runs_total": panel_summary["crash_runs_total"],
            "crash_rate": float(run_panel["crashed"].mean()),
            "pre_gap_ticks": int(args.pre_gap_ticks),
            "mean_pre_rows": panel_summary["mean_pre_rows"],
            "min_pre_rows": panel_summary["min_pre_rows"],
            "max_pre_rows": panel_summary["max_pre_rows"],
            "window_strategy_counts": panel_summary["window_strategy_counts"],
            "kept_nodes": kept_nodes,
            "dropped_nodes": dropped_nodes,
            "dagma_status": dagma_status,
            "direct_lingam_status": direct_lingam_status,
            "pcmci_status": pcmci_status,
            "weight_threshold": float(args.weight_threshold),
            "min_variance": float(args.min_variance),
        },
        "variance": variances,
        "method_summaries": method_summaries,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(report), encoding="utf-8")

    print("=" * 72)
    print("Script 26: Run-level causal discovery")
    print("=" * 72)
    print(f"Simulation panel    : {args.sim_panel}")
    print(f"Run-level panel     : {args.run_panel_csv}")
    print(f"Runs total          : {panel_summary['runs_total']}")
    print(f"Crash runs total    : {panel_summary['crash_runs_total']}")
    print(f"Kept nodes          : {kept_nodes}")
    print(f"Dropped nodes       : {dropped_nodes}")
    print(f"DAGMA status        : {dagma_status}")
    print(f"DirectLiNGAM status : {direct_lingam_status}")
    print(f"PCMCI status        : {pcmci_status}")
    print(f"Saved run panel     : {args.run_panel_csv}")
    print(f"Saved edges         : {args.edges_csv}")
    print(f"Saved JSON report   : {args.report_json}")
    print(f"Saved MD report     : {args.report_md}")


if __name__ == "__main__":
    main()