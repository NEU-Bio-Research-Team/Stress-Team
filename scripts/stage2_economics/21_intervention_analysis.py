"""Script 21 - Regime-aware intervention analysis for Phase 2.

Builds a run-level dataset from pre-crash drop windows, then estimates
counterfactual crash-rate changes under:
- do(OFI = 0)
- do(leverage_proxy = 0)

This avoids fitting a rare-event classifier on pooled tick-level rows where OFI
can become endogenous once the crash is already underway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_PANEL = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "lob_mini_simulation_llm.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "validation" / "phase2_intervention_analysis.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "validation" / "phase2_intervention_analysis.md"

FEATURES = ["ofi", "spread_bps", "depth_imbalance", "leverage_proxy", "kyle_lambda", "vpin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intervention analysis for simulated flash-crash panel")
    parser.add_argument("--sim-panel", type=Path, default=DEFAULT_SIM_PANEL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--window-pre-ticks", type=int, default=50)
    parser.add_argument("--min-window-rows", type=int, default=25)
    parser.add_argument(
        "--floor-policy",
        choices=["none", "exclude_runs", "censor_rows"],
        default="censor_rows",
    )
    parser.add_argument("--min-price-fraction", type=float, default=0.60)
    parser.add_argument("--floor-eps-mult", type=float, default=1.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURES + ["run_id", "phase", "tick_ms", "close", "flash_crash_flag"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in panel: {sorted(missing)}")

    work = df.copy()
    for col in FEATURES + ["tick_ms", "close", "flash_crash_flag"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["run_id"] = pd.to_numeric(work["run_id"], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["run_id", "phase", "tick_ms", "close"] + FEATURES).copy()
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
        if len(drop_rows):
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
        sub["floor_touched_run"] = int(pd.notna(first_floor_tick))
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def apply_floor_policy(df: pd.DataFrame, floor_policy: str) -> pd.DataFrame:
    if floor_policy == "none":
        return df.copy()
    if floor_policy == "exclude_runs":
        return df[df["floor_touched_run"] == 0].copy()
    keep_mask = df["first_floor_tick"].isna() | (df["run_tick_index"] < df["first_floor_tick"])
    return df.loc[keep_mask].copy()


def compute_run_drawdown_1s_pct(close: pd.Series, window_ticks: int = 10) -> float:
    s = pd.to_numeric(close, errors="coerce")
    base = s.shift(window_ticks - 1)
    rolling_min = s.rolling(window_ticks, min_periods=window_ticks).min()
    dd = (base - rolling_min) / base.replace(0, np.nan) * 100.0
    return float(dd.fillna(0.0).clip(lower=0.0).max())


def compute_max_drawdown_pct(close: pd.Series) -> float:
    s = pd.to_numeric(close, errors="coerce")
    running_max = s.cummax()
    dd = (running_max - s) / running_max.replace(0, np.nan) * 100.0
    return float(dd.fillna(0.0).max())


def build_run_level_dataset(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    annotated = annotate_runs(df, min_price_fraction=args.min_price_fraction, floor_eps_mult=args.floor_eps_mult)
    total_runs = int(annotated["run_id"].nunique())
    crash_runs_total = int(annotated.groupby("run_id")["flash_crash_flag"].max().gt(0).sum())
    floor_touched_run_rate = float(annotated.groupby("run_id")["floor_touched_run"].max().mean()) if total_runs else 0.0

    work = apply_floor_policy(annotated, args.floor_policy)
    records: list[dict[str, Any]] = []
    for run_id, sub in work.groupby("run_id", sort=False):
        sub = sub.sort_values("run_tick_index").copy()
        drop_rows = sub[sub["phase"] == "drop"]
        if len(drop_rows) == 0:
            continue

        crash_rows = sub[sub["flash_crash_flag"] > 0]
        has_crash = int(len(crash_rows) > 0)
        anchor_tick = int(crash_rows["run_tick_index"].iloc[0]) if has_crash else int(drop_rows["run_tick_index"].iloc[-1])
        window_end = anchor_tick - 1
        if window_end < int(drop_rows["run_tick_index"].iloc[0]):
            continue

        window_start = max(window_end - args.window_pre_ticks + 1, int(drop_rows["run_tick_index"].iloc[0]))
        window_df = drop_rows[
            (drop_rows["run_tick_index"] >= window_start) & (drop_rows["run_tick_index"] <= window_end)
        ].copy()
        if len(window_df) < args.min_window_rows:
            continue

        row: dict[str, Any] = {
            "run_id": int(run_id),
            "has_crash": has_crash,
            "window_rows": int(len(window_df)),
            "anchor_tick": anchor_tick,
            "floor_touched_run": int(sub["floor_touched_run"].iloc[0]),
            "max_drawdown_pct": compute_max_drawdown_pct(sub["close"]),
            "max_drawdown_1s_pct": compute_run_drawdown_1s_pct(sub["close"]),
        }
        for feature in FEATURES:
            row[f"{feature}_mean"] = float(window_df[feature].mean())
        row["ofi_x_leverage_mean"] = row["ofi_mean"] * row["leverage_proxy_mean"]
        records.append(row)

    runs_df = pd.DataFrame(records)
    used_runs = int(len(runs_df))
    summary = {
        "sim_panel": str(args.sim_panel),
        "runs_total": total_runs,
        "crash_runs_total": crash_runs_total,
        "runs_used": used_runs,
        "crash_runs_used": int(runs_df["has_crash"].sum()) if used_runs else 0,
        "window_pre_ticks": int(args.window_pre_ticks),
        "min_window_rows": int(args.min_window_rows),
        "floor_policy": args.floor_policy,
        "floor_touched_run_rate": floor_touched_run_rate,
        "floor_touched_used_rate": float(runs_df["floor_touched_run"].mean()) if used_runs else 0.0,
        "mean_window_rows": float(runs_df["window_rows"].mean()) if used_runs else 0.0,
    }
    return runs_df, summary

def run_interventions(work: pd.DataFrame, seed: int) -> dict[str, Any]:
    x_cols = [f"{feature}_mean" for feature in FEATURES] + ["ofi_x_leverage_mean"]

    X = work[x_cols].copy()
    y = work["has_crash"].astype(int)

    crash_rate_raw = float(y.mean())
    crash_rate_obs = crash_rate_raw
    auc = None

    if y.nunique() < 2:
        # Degenerate: all runs are in the same class.
        crash_rate_do_ofi0 = 0.0
        crash_rate_do_lev0 = 0.0
        clf = None
    else:
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        clf.fit(X, y)

        p_obs = clf.predict_proba(X)[:, 1]
        crash_rate_obs = float(np.mean(p_obs))
        auc = float(roc_auc_score(y, p_obs))

        X_do_ofi0 = X.copy()
        X_do_ofi0["ofi_mean"] = 0.0
        X_do_ofi0["ofi_x_leverage_mean"] = 0.0
        p_do_ofi0 = clf.predict_proba(X_do_ofi0)[:, 1]
        crash_rate_do_ofi0 = float(np.mean(p_do_ofi0))

        X_do_lev0 = X.copy()
        X_do_lev0["leverage_proxy_mean"] = 0.0
        X_do_lev0["ofi_x_leverage_mean"] = 0.0
        p_do_lev0 = clf.predict_proba(X_do_lev0)[:, 1]
        crash_rate_do_lev0 = float(np.mean(p_do_lev0))

    X_do_lev0 = X.copy()
    X_do_lev0["leverage_proxy_mean"] = 0.0
    X_do_lev0["ofi_x_leverage_mean"] = 0.0

    y_severity = work["max_drawdown_1s_pct"].clip(lower=0.0)
    if len(work) < 2:
        severity_obs = float(y_severity.mean())
        severity_do_lev0 = severity_obs
    else:
        reg = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linear", LinearRegression()),
            ]
        )
        reg.fit(X, y_severity)
        severity_obs = float(np.mean(np.clip(reg.predict(X), a_min=0.0, a_max=None)))
        severity_do_lev0 = float(np.mean(np.clip(reg.predict(X_do_lev0), a_min=0.0, a_max=None)))

    ofi_reduction_pct = (crash_rate_obs - crash_rate_do_ofi0) / max(crash_rate_obs, 1e-9) * 100.0
    lev_reduction_pct = (crash_rate_obs - crash_rate_do_lev0) / max(crash_rate_obs, 1e-9) * 100.0
    severity_reduction_pct = (severity_obs - severity_do_lev0) / max(abs(severity_obs), 1e-9) * 100.0

    return {
        "model": {
            "logistic_auc": auc,
            "n_runs": int(len(work)),
            "positive_rate_raw": crash_rate_raw,
        },
        "observational": {
            "crash_rate_raw": crash_rate_raw,
            "crash_rate_pred": crash_rate_obs,
            "severity_max_drawdown_1s_pct_pred": severity_obs,
            "severity_max_drawdown_1s_pct_raw": float(y_severity.mean()),
        },
        "do_ofi_0": {
            "crash_rate_pred": crash_rate_do_ofi0,
            "relative_reduction_pct": ofi_reduction_pct,
            "h3_target_30pct_pass": bool(ofi_reduction_pct >= 30.0),
        },
        "do_leverage_0": {
            "crash_rate_pred": crash_rate_do_lev0,
            "relative_reduction_pct": lev_reduction_pct,
            "severity_max_drawdown_1s_pct_pred": severity_do_lev0,
            "severity_reduction_pct": severity_reduction_pct,
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 Intervention Analysis")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- sim_panel: {report['data']['sim_panel']}")
    lines.append(f"- runs_total: {report['data']['runs_total']}")
    lines.append(f"- crash_runs_total: {report['data']['crash_runs_total']}")
    lines.append(f"- runs_used: {report['data']['runs_used']}")
    lines.append(f"- crash_runs_used: {report['data']['crash_runs_used']}")
    lines.append(f"- window_pre_ticks: {report['data']['window_pre_ticks']}")
    lines.append(f"- min_window_rows: {report['data']['min_window_rows']}")
    lines.append(f"- floor_policy: {report['data']['floor_policy']}")
    lines.append(f"- floor_touched_run_rate: {report['data']['floor_touched_run_rate']}")
    lines.append(f"- floor_touched_used_rate: {report['data']['floor_touched_used_rate']}")
    lines.append(f"- mean_window_rows: {report['data']['mean_window_rows']}")
    lines.append("")

    lines.append("## Model")
    lines.append("")
    lines.append(f"- n_runs: {report['model']['n_runs']}")
    lines.append(f"- logistic_auc: {report['model']['logistic_auc']}")
    lines.append(f"- positive_rate_raw: {report['model']['positive_rate_raw']}")
    lines.append("")

    lines.append("## Observational")
    lines.append("")
    lines.append(f"- crash_rate_raw: {report['observational']['crash_rate_raw']}")
    lines.append(f"- crash_rate_pred: {report['observational']['crash_rate_pred']}")
    lines.append(f"- severity_max_drawdown_1s_pct_pred: {report['observational']['severity_max_drawdown_1s_pct_pred']}")
    lines.append(f"- severity_max_drawdown_1s_pct_raw: {report['observational']['severity_max_drawdown_1s_pct_raw']}")
    lines.append("")

    lines.append("## do(OFI=0)")
    lines.append("")
    lines.append(f"- crash_rate_pred: {report['do_ofi_0']['crash_rate_pred']}")
    lines.append(f"- relative_reduction_pct: {report['do_ofi_0']['relative_reduction_pct']}")
    lines.append(f"- h3_target_30pct_pass: {report['do_ofi_0']['h3_target_30pct_pass']}")
    lines.append("")

    lines.append("## do(leverage=0)")
    lines.append("")
    lines.append(f"- crash_rate_pred: {report['do_leverage_0']['crash_rate_pred']}")
    lines.append(f"- relative_reduction_pct: {report['do_leverage_0']['relative_reduction_pct']}")
    lines.append(f"- severity_max_drawdown_1s_pct_pred: {report['do_leverage_0']['severity_max_drawdown_1s_pct_pred']}")
    lines.append(f"- severity_reduction_pct: {report['do_leverage_0']['severity_reduction_pct']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.sim_panel.exists():
        raise FileNotFoundError(f"Simulation panel not found: {args.sim_panel}")

    panel = clean_panel(pd.read_csv(args.sim_panel))
    work, data_summary = build_run_level_dataset(panel, args)
    if work.empty:
        raise ValueError("No run-level windows remain after floor-policy and pre-crash filtering")

    report = {"data": data_summary, **run_interventions(work, seed=args.seed)}

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(report), encoding="utf-8")

    print("=" * 72)
    print("Script 21: Intervention analysis")
    print("=" * 72)
    print(f"Panel             : {args.sim_panel}")
    print(f"Runs used         : {report['model']['n_runs']}")
    print(f"do(OFI=0) reduce  : {report['do_ofi_0']['relative_reduction_pct']:.2f}%")
    print(f"do(lev=0) reduce  : {report['do_leverage_0']['relative_reduction_pct']:.2f}%")
    print(f"H3 30% target pass: {report['do_ofi_0']['h3_target_30pct_pass']}")
    print(f"Saved JSON report : {args.report_json}")
    print(f"Saved MD report   : {args.report_md}")


if __name__ == "__main__":
    main()
