"""Script 20 - Causal discovery on simulation panel output.

Runs NOTEARS and LiNGAM when available, then compares discovered edges against
an expected theoretical mechanism graph.
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
DEFAULT_SIM_PANEL = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_simulation_llm.csv"
DEFAULT_EDGES_CSV = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "causal_discovery_edges.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "validation" / "phase2_causal_discovery.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "validation" / "phase2_causal_discovery.md"

NODES = [
    "ofi",
    "spread_bps",
    "depth_imbalance",
    "leverage_proxy",
    "kyle_lambda",
    "vpin",
    "flash_crash_flag",
    # Wealth confounder nodes (added Phase 2)
    "mean_wealth_t",
    "pct_insolvent",
    "wealth_concentration",
]

THEORETICAL_EDGES = {
    ("ofi", "spread_bps"),
    ("spread_bps", "depth_imbalance"),
    ("depth_imbalance", "flash_crash_flag"),
    ("ofi", "flash_crash_flag"),
    ("leverage_proxy", "flash_crash_flag"),
    # Wealth confounder edges (Bookstaber 2014 / Kirilenko 2017 extended)
    ("wealth_concentration", "leverage_proxy"),
    ("pct_insolvent", "spread_bps"),
    ("mean_wealth_t", "ofi"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal discovery on simulation panel")
    parser.add_argument("--sim-panel", type=Path, default=DEFAULT_SIM_PANEL)
    parser.add_argument("--edges-csv", type=Path, default=DEFAULT_EDGES_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--sample-rows", type=int, default=120000)
    parser.add_argument("--weight-threshold", type=float, default=0.05)
    parser.add_argument("--lambda1", type=float, default=0.1)
    return parser.parse_args()


def prepare_matrix(df: pd.DataFrame, sample_rows: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    work = df[NODES].copy()
    for col in NODES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) > sample_rows:
        work = work.sample(n=sample_rows, random_state=seed)

    scaler = StandardScaler()
    X = scaler.fit_transform(work.values)
    return work, X


def run_notears(X: np.ndarray, node_names: list[str], lambda1: float) -> tuple[list[dict[str, Any]], str]:
    try:
        from dagma.linear import DagmaLinear  # type: ignore
    except Exception as e:
        return [], f"notears_unavailable: {e}"

    model = DagmaLinear(loss_type="l2")
    W = model.fit(X, lambda1=lambda1)
    edges: list[dict[str, Any]] = []
    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if i == j:
                continue
            w = float(W[i, j])
            edges.append({"method": "notears", "source": src, "target": dst, "weight": w})
    return edges, "ok"


def run_lingam(X: np.ndarray, node_names: list[str]) -> tuple[list[dict[str, Any]], str]:
    try:
        from lingam import DirectLiNGAM  # type: ignore
    except Exception as e:
        return [], f"lingam_unavailable: {e}"

    model = DirectLiNGAM()
    model.fit(X)
    B = model.adjacency_matrix_

    edges: list[dict[str, Any]] = []
    for child_i, child in enumerate(node_names):
        for parent_j, parent in enumerate(node_names):
            if child_i == parent_j:
                continue
            w = float(B[child_i, parent_j])
            edges.append({"method": "lingam", "source": parent, "target": child, "weight": w})
    return edges, "ok"


def fallback_correlation_dag(df: pd.DataFrame, node_names: list[str]) -> list[dict[str, Any]]:
    # Fallback if NOTEARS/LiNGAM packages are missing.
    corr = df[node_names].corr().fillna(0.0)
    edges: list[dict[str, Any]] = []

    for i, src in enumerate(node_names):
        for j, dst in enumerate(node_names):
            if i == j:
                continue
            # Orient from earlier index to later index to guarantee acyclicity.
            if i >= j:
                continue
            w = float(corr.loc[src, dst])
            edges.append({"method": "corr_fallback", "source": src, "target": dst, "weight": w})
    return edges


def evaluate_method(edges: list[dict[str, Any]], method: str, threshold: float) -> dict[str, Any]:
    kept = {
        (e["source"], e["target"])
        for e in edges
        if e["method"] == method and abs(float(e["weight"])) >= threshold
    }

    tp = len(kept & THEORETICAL_EDGES)
    fp = len(kept - THEORETICAL_EDGES)
    fn = len(THEORETICAL_EDGES - kept)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "method": method,
        "num_edges_above_threshold": len(kept),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "matched_edges": sorted([f"{a}->{b}" for a, b in (kept & THEORETICAL_EDGES)]),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 Causal Discovery")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- rows_used: {report['data']['rows_used']}")
    lines.append(f"- nodes: {', '.join(report['data']['nodes'])}")
    lines.append(f"- notears_status: {report['data']['notears_status']}")
    lines.append(f"- lingam_status: {report['data']['lingam_status']}")
    lines.append("")

    lines.append("## Expected Edges")
    lines.append("")
    for src, dst in sorted(THEORETICAL_EDGES):
        lines.append(f"- {src} -> {dst}")
    lines.append("")

    lines.append("## Method Scores")
    lines.append("")
    for row in report["method_scores"]:
        lines.append(
            f"- {row['method']}: precision={row['precision']:.3f}, recall={row['recall']:.3f}, "
            f"edges={row['num_edges_above_threshold']}, matched={row['matched_edges']}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.sim_panel.exists():
        raise FileNotFoundError(f"Simulation panel not found: {args.sim_panel}")

    panel = pd.read_csv(args.sim_panel)
    missing = set(NODES) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns for causal discovery: {sorted(missing)}")

    prepared_df, X = prepare_matrix(panel, sample_rows=args.sample_rows)

    notears_edges, notears_status = run_notears(X, NODES, lambda1=args.lambda1)
    lingam_edges, lingam_status = run_lingam(X, NODES)

    all_edges = notears_edges + lingam_edges
    if not all_edges:
        all_edges = fallback_correlation_dag(prepared_df, NODES)

    edges_df = pd.DataFrame(all_edges)
    args.edges_csv.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(args.edges_csv, index=False)

    methods = sorted(set(edges_df["method"])) if not edges_df.empty else []
    method_scores = [evaluate_method(all_edges, m, args.weight_threshold) for m in methods]

    report = {
        "data": {
            "sim_panel": str(args.sim_panel),
            "rows_used": int(len(prepared_df)),
            "nodes": NODES,
            "notears_status": notears_status,
            "lingam_status": lingam_status,
            "weight_threshold": args.weight_threshold,
        },
        "method_scores": method_scores,
        "theoretical_edges": sorted([f"{a}->{b}" for a, b in THEORETICAL_EDGES]),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(report), encoding="utf-8")

    print("=" * 72)
    print("Script 20: Causal discovery")
    print("=" * 72)
    print(f"Panel             : {args.sim_panel}")
    print(f"Rows used         : {len(prepared_df)}")
    print(f"Methods           : {methods if methods else ['none']}")
    print(f"NOTEARS status    : {notears_status}")
    print(f"LiNGAM status     : {lingam_status}")
    print(f"Saved edges       : {args.edges_csv}")
    print(f"Saved JSON report : {args.report_json}")
    print(f"Saved MD report   : {args.report_md}")


if __name__ == "__main__":
    main()
