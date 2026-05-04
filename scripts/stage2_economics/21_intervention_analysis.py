"""Script 21 - Intervention analysis for Phase 2 (do-calculus inspired).

Estimates counterfactual crash-rate changes under:
- do(OFI = 0)
- do(leverage_proxy = 0)

Uses predictive structural approximations on top of simulation panel data.
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
    parser.add_argument("--sample-rows", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_drawdown_1s_pct(df: pd.DataFrame, window_ticks: int = 10) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for run_id, sub in df.groupby("run_id"):
        s = pd.to_numeric(sub["close"], errors="coerce")
        base = s.shift(window_ticks - 1)
        rolling_min = s.rolling(window_ticks, min_periods=window_ticks).min()
        dd = (base - rolling_min) / base.replace(0, np.nan) * 100.0
        out.loc[sub.index] = dd
    return out.fillna(0.0)


def prepare_data(df: pd.DataFrame, sample_rows: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    for col in FEATURES + ["flash_crash_flag", "close"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["flash_crash_flag", "close"]) 
    work["ofi_x_leverage"] = work["ofi"] * work["leverage_proxy"]
    work["drawdown_1s_pct"] = compute_drawdown_1s_pct(work)

    if len(work) > sample_rows:
        work = work.sample(n=sample_rows, random_state=seed)

    return work


def run_interventions(work: pd.DataFrame) -> dict[str, Any]:
    x_cols = FEATURES + ["ofi_x_leverage"]

    X = work[x_cols].copy()
    y = work["flash_crash_flag"].astype(int)

    crash_rate_obs = float(y.mean())
    auc = None

    if y.nunique() < 2:
        # Degenerate: all ticks in same class — logistic regression is undefined.
        # Report observed rates as zero; intervention counterfactuals are not estimable.
        crash_rate_do_ofi0 = 0.0
        crash_rate_do_lev0 = 0.0
        p_obs = np.zeros(len(y))
        p_do_ofi0 = np.zeros(len(y))
        p_do_lev0 = np.zeros(len(y))
        clf = None
    else:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X, y)

        p_obs = clf.predict_proba(X)[:, 1]
        crash_rate_obs = float(np.mean(p_obs))
        auc = float(roc_auc_score(y, p_obs))

        X_do_ofi0 = X.copy()
        X_do_ofi0["ofi"] = 0.0
        X_do_ofi0["ofi_x_leverage"] = 0.0
        p_do_ofi0 = clf.predict_proba(X_do_ofi0)[:, 1]
        crash_rate_do_ofi0 = float(np.mean(p_do_ofi0))

        X_do_lev0 = X.copy()
        X_do_lev0["leverage_proxy"] = 0.0
        X_do_lev0["ofi_x_leverage"] = 0.0
        p_do_lev0 = clf.predict_proba(X_do_lev0)[:, 1]
        crash_rate_do_lev0 = float(np.mean(p_do_lev0))

    X_do_lev0 = X.copy()
    X_do_lev0["leverage_proxy"] = 0.0
    X_do_lev0["ofi_x_leverage"] = 0.0

    y_severity = work["drawdown_1s_pct"].clip(lower=0.0)
    reg = LinearRegression()
    reg.fit(X, y_severity)

    severity_obs = float(np.mean(reg.predict(X)))
    severity_do_lev0 = float(np.mean(reg.predict(X_do_lev0)))

    ofi_reduction_pct = (crash_rate_obs - crash_rate_do_ofi0) / max(crash_rate_obs, 1e-9) * 100.0
    lev_reduction_pct = (crash_rate_obs - crash_rate_do_lev0) / max(crash_rate_obs, 1e-9) * 100.0
    severity_reduction_pct = (severity_obs - severity_do_lev0) / max(abs(severity_obs), 1e-9) * 100.0

    return {
        "model": {
            "logistic_auc": auc,
            "n_samples": int(len(work)),
            "positive_rate_raw": float(y.mean()),
        },
        "observational": {
            "crash_rate_pred": crash_rate_obs,
            "severity_drawdown_1s_pct_pred": severity_obs,
        },
        "do_ofi_0": {
            "crash_rate_pred": crash_rate_do_ofi0,
            "relative_reduction_pct": ofi_reduction_pct,
            "h3_target_30pct_pass": bool(ofi_reduction_pct >= 30.0),
        },
        "do_leverage_0": {
            "crash_rate_pred": crash_rate_do_lev0,
            "relative_reduction_pct": lev_reduction_pct,
            "severity_drawdown_1s_pct_pred": severity_do_lev0,
            "severity_reduction_pct": severity_reduction_pct,
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase 2 Intervention Analysis")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append(f"- n_samples: {report['model']['n_samples']}")
    lines.append(f"- logistic_auc: {report['model']['logistic_auc']}")
    lines.append(f"- positive_rate_raw: {report['model']['positive_rate_raw']}")
    lines.append("")

    lines.append("## Observational")
    lines.append("")
    lines.append(f"- crash_rate_pred: {report['observational']['crash_rate_pred']}")
    lines.append(f"- severity_drawdown_1s_pct_pred: {report['observational']['severity_drawdown_1s_pct_pred']}")
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
    lines.append(f"- severity_drawdown_1s_pct_pred: {report['do_leverage_0']['severity_drawdown_1s_pct_pred']}")
    lines.append(f"- severity_reduction_pct: {report['do_leverage_0']['severity_reduction_pct']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.sim_panel.exists():
        raise FileNotFoundError(f"Simulation panel not found: {args.sim_panel}")

    panel = pd.read_csv(args.sim_panel)
    required = set(FEATURES + ["run_id", "close", "flash_crash_flag"])
    if missing := (required - set(panel.columns)):
        raise ValueError(f"Missing columns in panel: {sorted(missing)}")

    work = prepare_data(panel, sample_rows=args.sample_rows, seed=args.seed)
    report = run_interventions(work)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(report), encoding="utf-8")

    print("=" * 72)
    print("Script 21: Intervention analysis")
    print("=" * 72)
    print(f"Panel             : {args.sim_panel}")
    print(f"Samples used      : {report['model']['n_samples']}")
    print(f"do(OFI=0) reduce  : {report['do_ofi_0']['relative_reduction_pct']:.2f}%")
    print(f"do(lev=0) reduce  : {report['do_leverage_0']['relative_reduction_pct']:.2f}%")
    print(f"H3 30% target pass: {report['do_ofi_0']['h3_target_30pct_pass']}")
    print(f"Saved JSON report : {args.report_json}")
    print(f"Saved MD report   : {args.report_md}")


if __name__ == "__main__":
    main()
