"""Script 20 - Regime-aware causal discovery on simulation panel output.

Runs NOTEARS/DAGMA and VAR-LiNGAM on a crash-proximate, floor-aware panel
slice, then compares discovered edges against an expected theoretical mechanism
graph.
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
    parser.add_argument("--phase-filter", nargs="+", default=["drop"])
    parser.add_argument("--crash-window-pre-ticks", type=int, default=50)
    parser.add_argument("--crash-window-post-ticks", type=int, default=20)
    parser.add_argument(
        "--floor-policy",
        choices=["none", "exclude_runs", "censor_rows"],
        default="censor_rows",
    )
    parser.add_argument("--min-price-fraction", type=float, default=0.60)
    parser.add_argument("--floor-eps-mult", type=float, default=1.001)
    parser.add_argument("--var-lag", type=int, default=1)
    parser.add_argument("--min-sequence-rows", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    required = set(NODES + ["run_id", "phase", "tick_ms", "close", "flash_crash_flag"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for causal discovery: {sorted(missing)}")

    work = df.copy()
    for col in NODES + ["tick_ms", "close", "flash_crash_flag"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["run_id"] = pd.to_numeric(work["run_id"], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["run_id", "phase", "tick_ms", "close"] + NODES).copy()
    work["run_id"] = work["run_id"].astype(int)
    work["flash_crash_flag"] = work["flash_crash_flag"].fillna(0.0).astype(int)
    return work.sort_values(["run_id", "tick_ms"]).reset_index(drop=True)


def annotate_runs(df: pd.DataFrame, min_price_fraction: float, floor_eps_mult: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for _, sub in df.groupby("run_id", sort=False):
        sub = sub.sort_values("tick_ms").copy()
        sub["run_tick_index"] = np.arange(len(sub), dtype=int)

        pre_rows = sub[sub["phase"] == "pre"]
        drop_rows = sub[sub["phase"] == "drop"]
        init_price = float(pre_rows["close"].iloc[-1]) if len(pre_rows) else float(sub["close"].iloc[0])
        floor_price = init_price * min_price_fraction

        first_floor_tick = np.nan
        last_drop_tick = np.nan
        if len(drop_rows):
            last_drop_tick = float(drop_rows["run_tick_index"].iloc[-1])
            floor_rows = drop_rows[drop_rows["close"] <= floor_price * floor_eps_mult]
            if len(floor_rows):
                first_floor_tick = float(floor_rows["run_tick_index"].iloc[0])

        first_crash_tick = np.nan
        crash_rows = sub[sub["flash_crash_flag"] > 0]
        if len(crash_rows):
            first_crash_tick = float(crash_rows["run_tick_index"].iloc[0])

        sub["floor_price"] = floor_price
        sub["first_floor_tick"] = first_floor_tick
        sub["first_crash_tick"] = first_crash_tick
        sub["last_drop_tick"] = last_drop_tick
        sub["floor_touched_run"] = int(pd.notna(first_floor_tick))
        sub["has_crash_run"] = int(pd.notna(first_crash_tick))
        if pd.notna(first_crash_tick):
            sub["crash_tick_offset"] = sub["run_tick_index"] - int(first_crash_tick)
        else:
            sub["crash_tick_offset"] = np.nan
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def apply_floor_policy(df: pd.DataFrame, floor_policy: str) -> pd.DataFrame:
    if floor_policy == "none":
        return df.copy()
    if floor_policy == "exclude_runs":
        return df[df["floor_touched_run"] == 0].copy()
    keep_mask = df["first_floor_tick"].isna() | (df["run_tick_index"] < df["first_floor_tick"])
    return df.loc[keep_mask].copy()


def build_discovery_panel(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotated = annotate_runs(df, min_price_fraction=args.min_price_fraction, floor_eps_mult=args.floor_eps_mult)
    total_runs = int(annotated["run_id"].nunique())
    crash_runs_total = int(annotated.groupby("run_id")["flash_crash_flag"].max().gt(0).sum())
    floor_touched_run_rate = float(annotated.groupby("run_id")["floor_touched_run"].max().mean()) if total_runs else 0.0

    work = apply_floor_policy(annotated, args.floor_policy)
    work = work.sort_values(["run_id", "run_tick_index"]).reset_index(drop=True)

    valid_crash_runs = set(work.loc[work["flash_crash_flag"] > 0, "run_id"].unique())
    work = work[work["run_id"].isin(valid_crash_runs)].copy()
    if args.phase_filter:
        work = work[work["phase"].isin(args.phase_filter)].copy()
    work = work[
        work["crash_tick_offset"].between(
            -args.crash_window_pre_ticks,
            args.crash_window_post_ticks,
            inclusive="both",
        )
    ].copy()
    work = work.sort_values(["run_id", "run_tick_index"]).reset_index(drop=True)

    runs_used = int(work["run_id"].nunique())
    floor_touched_used_rate = float(work.groupby("run_id")["floor_touched_run"].max().mean()) if runs_used else 0.0
    summary = {
        "rows_raw": int(len(annotated)),
        "rows_after_floor_policy": int(len(apply_floor_policy(annotated, args.floor_policy))),
        "rows_after_slice": int(len(work)),
        "runs_total": total_runs,
        "crash_runs_total": crash_runs_total,
        "runs_used": runs_used,
        "phase_filter": list(args.phase_filter),
        "crash_window_pre_ticks": int(args.crash_window_pre_ticks),
        "crash_window_post_ticks": int(args.crash_window_post_ticks),
        "floor_policy": args.floor_policy,
        "floor_touched_run_rate": floor_touched_run_rate,
        "floor_touched_used_rate": floor_touched_used_rate,
    }
    return work, summary


def limit_rows_preserving_runs(df: pd.DataFrame, sample_rows: int, seed: int) -> pd.DataFrame:
    if sample_rows <= 0 or len(df) <= sample_rows:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    run_ids = df["run_id"].drop_duplicates().to_numpy()
    selected_frames: list[pd.DataFrame] = []
    total_rows = 0

    for run_id in rng.permutation(run_ids):
        sub = df[df["run_id"] == run_id]
        selected_frames.append(sub)
        total_rows += len(sub)
        if total_rows >= sample_rows:
            break

    out = pd.concat(selected_frames, ignore_index=True)
    return out.sort_values(["run_id", "run_tick_index"]).reset_index(drop=True)


def prepare_matrix(df: pd.DataFrame, sample_rows: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    work = df.copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=NODES).copy()
    work = limit_rows_preserving_runs(work, sample_rows=sample_rows, seed=seed)

    scaler = StandardScaler()
    X = scaler.fit_transform(work[NODES].to_numpy())

    scaled = work.copy()
    scaled = scaled.astype({col: float for col in NODES})
    scaled.loc[:, NODES] = pd.DataFrame(X, columns=NODES, index=scaled.index)
    sequences = [sub[NODES].to_numpy(dtype=float) for _, sub in scaled.groupby("run_id", sort=False)]
    return scaled, X, sequences


def run_notears(X: np.ndarray, node_names: list[str], lambda1: float) -> tuple[list[dict[str, Any]], str]:
    try:
        from dagma.linear import DagmaLinear  # type: ignore
    except Exception as e:
        return [], f"notears_unavailable: {e}"

    model = DagmaLinear(loss_type="l2")
    W = model.fit(X, lambda1=lambda1, w_threshold=0.0)
    edges: list[dict[str, Any]] = []
    for child_i, child in enumerate(node_names):
        for parent_j, parent in enumerate(node_names):
            if child_i == parent_j:
                continue
            # DAGMA/NOTEARS return W[child, parent], so parent_j -> child_i.
            w = float(W[child_i, parent_j])
            edges.append({"method": "notears", "source": parent, "target": child, "weight": w, "lag": 0})
    return edges, "ok"


def run_var_lingam(
    sequences: list[np.ndarray],
    node_names: list[str],
    lags: int,
    min_sequence_rows: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from lingam import VARLiNGAM  # type: ignore
    except Exception as e:
        return [], f"var_lingam_unavailable: {e}"

    matrices: list[np.ndarray] = []
    failed = 0
    skipped_short = 0
    for sequence in sequences:
        if len(sequence) <= max(lags + 1, min_sequence_rows):
            skipped_short += 1
            continue
        try:
            model = VARLiNGAM(lags=lags, criterion=None, random_state=random_state)
            model.fit(sequence)
            matrices.append(np.asarray(model.adjacency_matrices_, dtype=float))
        except Exception:
            failed += 1

    if not matrices:
        return [], f"var_lingam_failed: no_valid_sequences (skipped_short={skipped_short}, failed={failed})"

    avg = np.mean(matrices, axis=0)

    edges: list[dict[str, Any]] = []
    for lag in range(avg.shape[0]):
        for child_i, child in enumerate(node_names):
            for parent_j, parent in enumerate(node_names):
                if child_i == parent_j:
                    continue
                w = float(avg[lag, child_i, parent_j])
                edges.append({"method": "varlingam", "source": parent, "target": child, "weight": w, "lag": lag})
    status = f"ok (sequences_used={len(matrices)}, skipped_short={skipped_short}, failed={failed})"
    return edges, status


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
            edges.append({"method": "corr_fallback", "source": src, "target": dst, "weight": w, "lag": 0})
    return edges


def strongest_edges_by_pair(
    edges: list[dict[str, Any]],
    method: str,
    threshold: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    strongest: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        if edge["method"] != method:
            continue
        weight = float(edge["weight"])
        if abs(weight) < threshold:
            continue
        key = (edge["source"], edge["target"])
        current = strongest.get(key)
        if current is None or abs(weight) > abs(float(current["weight"])):
            strongest[key] = {
                "weight": weight,
                "lag": int(edge.get("lag", 0)),
            }
    return strongest


def pair_shd(true_dirs: set[str], pred_dirs: set[str]) -> int:
    if true_dirs == pred_dirs:
        return 0
    if len(true_dirs) == 1 and len(pred_dirs) == 1 and true_dirs != pred_dirs:
        return 1
    return len(true_dirs - pred_dirs) + len(pred_dirs - true_dirs)


def structural_hamming_distance(pred_edges: set[tuple[str, str]], node_names: list[str]) -> int:
    shd = 0
    for idx, left in enumerate(node_names):
        for right in node_names[idx + 1 :]:
            true_dirs: set[str] = set()
            pred_dirs: set[str] = set()
            if (left, right) in THEORETICAL_EDGES:
                true_dirs.add("lr")
            if (right, left) in THEORETICAL_EDGES:
                true_dirs.add("rl")
            if (left, right) in pred_edges:
                pred_dirs.add("lr")
            if (right, left) in pred_edges:
                pred_dirs.add("rl")
            shd += pair_shd(true_dirs, pred_dirs)
    return shd


def evaluate_method(edges: list[dict[str, Any]], method: str, threshold: float) -> dict[str, Any]:
    strongest = strongest_edges_by_pair(edges, method=method, threshold=threshold)
    kept = set(strongest)

    tp = len(kept & THEORETICAL_EDGES)
    fp = len(kept - THEORETICAL_EDGES)
    fn = len(THEORETICAL_EDGES - kept)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    top_edges = []
    for (source, target), meta in sorted(
        strongest.items(),
        key=lambda item: abs(float(item[1]["weight"])),
        reverse=True,
    )[:10]:
        top_edges.append(f"{source}->{target}@lag{meta['lag']} ({meta['weight']:.3f})")

    return {
        "method": method,
        "num_edges_above_threshold": len(kept),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": structural_hamming_distance(kept, NODES),
        "matched_edges": sorted([f"{a}->{b}" for a, b in (kept & THEORETICAL_EDGES)]),
        "top_edges": top_edges,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 Causal Discovery")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- sim_panel: {report['data']['sim_panel']}")
    lines.append(f"- rows_raw: {report['data']['rows_raw']}")
    lines.append(f"- rows_after_floor_policy: {report['data']['rows_after_floor_policy']}")
    lines.append(f"- rows_after_slice: {report['data']['rows_after_slice']}")
    lines.append(f"- rows_used: {report['data']['rows_used']}")
    lines.append(f"- runs_total: {report['data']['runs_total']}")
    lines.append(f"- crash_runs_total: {report['data']['crash_runs_total']}")
    lines.append(f"- runs_used: {report['data']['runs_used']}")
    lines.append(f"- nodes: {', '.join(report['data']['nodes'])}")
    lines.append(f"- phase_filter: {', '.join(report['data']['phase_filter'])}")
    lines.append(f"- crash_window_pre_ticks: {report['data']['crash_window_pre_ticks']}")
    lines.append(f"- crash_window_post_ticks: {report['data']['crash_window_post_ticks']}")
    lines.append(f"- floor_policy: {report['data']['floor_policy']}")
    lines.append(f"- floor_touched_run_rate: {report['data']['floor_touched_run_rate']}")
    lines.append(f"- floor_touched_used_rate: {report['data']['floor_touched_used_rate']}")
    lines.append(f"- notears_status: {report['data']['notears_status']}")
    lines.append(f"- var_lingam_status: {report['data']['var_lingam_status']}")
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
            f"f1={row['f1']:.3f}, shd={row['shd']}, edges={row['num_edges_above_threshold']}, "
            f"matched={row['matched_edges']}"
        )
        if row["top_edges"]:
            lines.append(f"  strongest={row['top_edges']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.sim_panel.exists():
        raise FileNotFoundError(f"Simulation panel not found: {args.sim_panel}")

    panel = clean_panel(pd.read_csv(args.sim_panel))
    discovery_panel, slice_summary = build_discovery_panel(panel, args)
    if discovery_panel.empty:
        raise ValueError("No rows remain after applying crash-window and floor-policy filters")

    prepared_df, X, sequences = prepare_matrix(discovery_panel, sample_rows=args.sample_rows, seed=args.seed)

    notears_edges, notears_status = run_notears(X, NODES, lambda1=args.lambda1)
    var_lingam_edges, var_lingam_status = run_var_lingam(
        sequences,
        NODES,
        lags=args.var_lag,
        min_sequence_rows=args.min_sequence_rows,
        random_state=args.seed,
    )

    all_edges = notears_edges + var_lingam_edges
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
            **slice_summary,
            "rows_used": int(len(prepared_df)),
            "nodes": NODES,
            "notears_status": notears_status,
            "lingam_status": var_lingam_status,
            "var_lingam_status": var_lingam_status,
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
    print(f"Rows after slice  : {slice_summary['rows_after_slice']}")
    print(f"Rows used         : {len(prepared_df)}")
    print(f"Methods           : {methods if methods else ['none']}")
    print(f"NOTEARS status    : {notears_status}")
    print(f"VAR-LiNGAM status : {var_lingam_status}")
    print(f"Saved edges       : {args.edges_csv}")
    print(f"Saved JSON report : {args.report_json}")
    print(f"Saved MD report   : {args.report_md}")


if __name__ == "__main__":
    main()
