"""Build a floor-censored causal panel and Step 4 lock-test summary.

Inputs:
  raw shard CSV/JSON pairs from Script 18.

Outputs:
  - censored_panel.csv (all kept rows after floor censoring)
  - analysis/per_run_censored_metrics.csv
  - analysis/step4_lock_test_summary.json

The censoring rule is explicit and fixed:
  censor_tick = first_floor_tick - guard_window

Rows at or after censor_tick are removed from the causal panel.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tardis"
    / "phase2_outputs"
    / "phase2_censored_causal_candidate_cfgC_dim350_dsp020_v1_200runs_gw5t"
)
DEFAULT_INPUT_DIR = DEFAULT_RUN_ROOT / "raw"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "analysis"
DEFAULT_PANEL_FILE = DEFAULT_RUN_ROOT / "censored_panel.csv"
DEFAULT_FILE_PATTERN = "lob_pilot_*.csv"
DEFAULT_GUARD_WINDOW = 5
DEFAULT_CRASH_WINDOW = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build floor-censored panel for Step 4")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-file", type=Path, default=DEFAULT_PANEL_FILE)
    parser.add_argument("--file-pattern", type=str, default=DEFAULT_FILE_PATTERN)
    parser.add_argument("--cfg-label", type=str, default="cfgC_dim350_dsp020")
    parser.add_argument("--min-price-fraction", type=float, default=0.60)
    parser.add_argument("--floor-eps-mult", type=float, default=1.001)
    parser.add_argument("--guard-window", type=int, default=DEFAULT_GUARD_WINDOW)
    parser.add_argument("--crash-window", type=int, default=DEFAULT_CRASH_WINDOW)
    return parser.parse_args()


def safe_seed_from_name(name: str) -> int:
    marker = "_s"
    pos = name.rfind(marker)
    if pos < 0:
        return -1
    try:
        return int(name[pos + len(marker) :])
    except Exception:
        return -1


def safe_cfg_from_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) < 2:
        return name
    return "_".join(parts[:-1])


def compute_run_drawdown(price: pd.Series) -> pd.Series:
    running_max = price.cummax()
    return (running_max - price) / running_max * 100.0


def safe_float(value: float | int | np.floating | None) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def summarize_series(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "mean": float(clean.mean()),
        "p50": float(clean.quantile(0.50)),
        "p95": float(clean.quantile(0.95)),
    }


def analyze_run(
    run_df: pd.DataFrame,
    cfg: str,
    shard_name: str,
    seed: int,
    run_id: int,
    min_price_fraction: float,
    floor_eps_mult: float,
    guard_window: int,
    crash_window: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    run_df = run_df.sort_values("tick_ms").reset_index(drop=True).copy()
    run_df["tick_idx"] = np.arange(len(run_df))

    event_id = int(run_df["event_id"].iloc[0]) if "event_id" in run_df.columns else -1
    pre_rows = run_df[run_df["phase"] == "pre"]
    drop_rows = run_df[run_df["phase"] == "drop"].copy()

    if len(pre_rows) == 0 or len(drop_rows) == 0:
        empty = run_df.iloc[0:0].copy()
        metrics = {
            "cfg": cfg,
            "shard_name": shard_name,
            "seed": seed,
            "run_id": int(run_id),
            "event_id": event_id,
            "n_rows_raw": int(len(run_df)),
            "n_rows_censored": 0,
            "n_drop_rows_raw": int(len(drop_rows)),
            "n_drop_rows_censored": 0,
            "floor_price": float("nan"),
            "first_floor_tick": float("nan"),
            "censor_tick": float("nan"),
            "has_floor_raw": 0,
            "has_crash_raw": 0,
            "has_crash_censored": 0,
            "floor_exposure_run": 0,
            "frac_at_min_censored": float("nan"),
            "usable_fraction_total": 0.0,
            "usable_fraction_drop": float("nan"),
            "first_crash_tick_censored": float("nan"),
            "drawdown_at_first_crash": float("nan"),
            "ofi_first_crash": float("nan"),
            "ofi_crash_window_mean": float("nan"),
            "spread_crash_window_mean": float("nan"),
            "depth_crash_window_mean": float("nan"),
            "spread_pre_mean": float("nan"),
            "depth_pre_mean": float("nan"),
        }
        return metrics, empty

    init_price = float(pre_rows["close"].iloc[-1])
    floor_price = init_price * min_price_fraction

    drop_rows["at_floor"] = drop_rows["close"] <= floor_price * floor_eps_mult
    floor_ticks = drop_rows.loc[drop_rows["at_floor"], "tick_idx"]
    has_floor_raw = int(len(floor_ticks) > 0)
    first_floor_tick = float(floor_ticks.iloc[0]) if has_floor_raw else float("nan")

    crash_rows_raw = run_df[run_df["flash_crash_flag"] > 0]
    has_crash_raw = int(len(crash_rows_raw) > 0)

    if has_floor_raw:
        censor_tick = max(int(first_floor_tick) - guard_window, 0)
        censored_df = run_df[run_df["tick_idx"] < censor_tick].copy()
    else:
        censor_tick = int(run_df["tick_idx"].max()) + 1
        censored_df = run_df.copy()

    censored_drop = censored_df[censored_df["phase"] == "drop"].copy()
    censored_drop["at_floor"] = censored_drop["close"] <= floor_price * floor_eps_mult
    floor_exposure_run = int(censored_drop["at_floor"].any()) if len(censored_drop) else 0
    frac_at_min_censored = float(censored_drop["at_floor"].mean()) if len(censored_drop) else 0.0

    crash_rows_censored = censored_df[censored_df["flash_crash_flag"] > 0]
    has_crash_censored = int(len(crash_rows_censored) > 0)
    first_crash_tick_censored = (
        float(crash_rows_censored["tick_idx"].iloc[0]) if has_crash_censored else float("nan")
    )

    dd_series = compute_run_drawdown(run_df["close"].astype(float))
    if has_crash_censored:
        first_crash_idx = int(first_crash_tick_censored)
        drawdown_at_first_crash = float(dd_series.iloc[first_crash_idx])
        first_crash_row = censored_df.loc[censored_df["tick_idx"] == first_crash_idx].iloc[0]
        ofi_first_crash = float(first_crash_row["ofi"])
        window_start = max(first_crash_idx - crash_window + 1, 0)
        crash_window_df = censored_df[
            (censored_df["tick_idx"] >= window_start) & (censored_df["tick_idx"] <= first_crash_idx)
        ]
        ofi_crash_window_mean = float(crash_window_df["ofi"].mean())
        spread_crash_window_mean = float(crash_window_df["spread_bps"].mean())
        depth_crash_window_mean = float(crash_window_df["depth_imbalance"].mean())
    else:
        drawdown_at_first_crash = float("nan")
        ofi_first_crash = float("nan")
        ofi_crash_window_mean = float("nan")
        spread_crash_window_mean = float("nan")
        depth_crash_window_mean = float("nan")

    spread_pre_mean = float(pre_rows["spread_bps"].mean()) if len(pre_rows) else float("nan")
    depth_pre_mean = float(pre_rows["depth_imbalance"].mean()) if len(pre_rows) else float("nan")
    n_drop_rows_censored = int(len(censored_drop))
    usable_fraction_drop = (
        float(n_drop_rows_censored / len(drop_rows)) if len(drop_rows) else float("nan")
    )

    censored_df["cfg"] = cfg
    censored_df["shard_name"] = shard_name
    censored_df["seed"] = seed
    censored_df["floor_price"] = floor_price
    censored_df["first_floor_tick"] = first_floor_tick
    censored_df["censor_tick"] = censor_tick

    metrics = {
        "cfg": cfg,
        "shard_name": shard_name,
        "seed": seed,
        "run_id": int(run_id),
        "event_id": event_id,
        "n_rows_raw": int(len(run_df)),
        "n_rows_censored": int(len(censored_df)),
        "n_drop_rows_raw": int(len(drop_rows)),
        "n_drop_rows_censored": n_drop_rows_censored,
        "floor_price": floor_price,
        "first_floor_tick": safe_float(first_floor_tick),
        "censor_tick": safe_float(censor_tick),
        "has_floor_raw": has_floor_raw,
        "has_crash_raw": has_crash_raw,
        "has_crash_censored": has_crash_censored,
        "floor_exposure_run": floor_exposure_run,
        "frac_at_min_censored": frac_at_min_censored,
        "usable_fraction_total": float(len(censored_df) / len(run_df)) if len(run_df) else 0.0,
        "usable_fraction_drop": usable_fraction_drop,
        "first_crash_tick_censored": safe_float(first_crash_tick_censored),
        "drawdown_at_first_crash": safe_float(drawdown_at_first_crash),
        "ofi_first_crash": safe_float(ofi_first_crash),
        "ofi_crash_window_mean": safe_float(ofi_crash_window_mean),
        "spread_crash_window_mean": safe_float(spread_crash_window_mean),
        "depth_crash_window_mean": safe_float(depth_crash_window_mean),
        "spread_pre_mean": safe_float(spread_pre_mean),
        "depth_pre_mean": safe_float(depth_pre_mean),
    }
    return metrics, censored_df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.panel_file.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(glob.glob(str(args.input_dir / args.file_pattern)))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.input_dir} matching {args.file_pattern}")

    all_metrics: list[dict[str, float | int | str]] = []
    all_censored_frames: list[pd.DataFrame] = []
    shard_summaries: list[dict[str, float | int | str]] = []

    for csv_path in csv_paths:
        csv_file = Path(csv_path)
        shard_name = csv_file.stem.replace("lob_pilot_", "")
        cfg = safe_cfg_from_name(shard_name)
        seed = safe_seed_from_name(shard_name)

        df = pd.read_csv(csv_file)
        required = {"run_id", "event_id", "tick_ms", "phase", "close", "ofi", "spread_bps", "depth_imbalance", "flash_crash_flag"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_file.name}: {sorted(missing)}")

        shard_metrics: list[dict[str, float | int | str]] = []
        for run_id, run_df in df.groupby("run_id"):
            metrics, censored_df = analyze_run(
                run_df=run_df,
                cfg=cfg,
                shard_name=shard_name,
                seed=seed,
                run_id=int(run_id),
                min_price_fraction=args.min_price_fraction,
                floor_eps_mult=args.floor_eps_mult,
                guard_window=args.guard_window,
                crash_window=args.crash_window,
            )
            all_metrics.append(metrics)
            shard_metrics.append(metrics)
            all_censored_frames.append(censored_df)

        shard_df = pd.DataFrame(shard_metrics)
        shard_summaries.append(
            {
                "cfg": cfg,
                "shard_name": shard_name,
                "seed": seed,
                "n_runs": int(len(shard_df)),
                "crash_rate_censored": float(shard_df["has_crash_censored"].mean()) if len(shard_df) else float("nan"),
                "floor_exposure": float(shard_df["floor_exposure_run"].mean()) if len(shard_df) else float("nan"),
                "avg_frac_at_min_censored": float(shard_df["frac_at_min_censored"].mean()) if len(shard_df) else float("nan"),
            }
        )

    per_run_df = pd.DataFrame(all_metrics).sort_values(["seed", "run_id"]).reset_index(drop=True)
    censored_panel_df = pd.concat(all_censored_frames, ignore_index=True) if all_censored_frames else pd.DataFrame()
    shard_summary_df = pd.DataFrame(shard_summaries).sort_values(["seed"]).reset_index(drop=True)

    per_run_path = args.output_dir / "per_run_censored_metrics.csv"
    per_run_df.to_csv(per_run_path, index=False)

    shard_path = args.output_dir / "per_shard_censored_metrics.csv"
    shard_summary_df.to_csv(shard_path, index=False)

    censored_panel_df.to_csv(args.panel_file, index=False)

    crash_runs = per_run_df[per_run_df["has_crash_censored"] == 1].copy()
    summary = {
        "cfg_label": args.cfg_label,
        "input_dir": str(args.input_dir),
        "panel_file": str(args.panel_file),
        "guard_window": int(args.guard_window),
        "crash_window": int(args.crash_window),
        "min_price_fraction": float(args.min_price_fraction),
        "n_shards": int(len(csv_paths)),
        "n_runs": int(len(per_run_df)),
        "raw_rows_total": int(per_run_df["n_rows_raw"].sum()),
        "censored_rows_total": int(per_run_df["n_rows_censored"].sum()),
        "crash_rate_raw": float(per_run_df["has_crash_raw"].mean()) if len(per_run_df) else float("nan"),
        "crash_rate_censored": float(per_run_df["has_crash_censored"].mean()) if len(per_run_df) else float("nan"),
        "avg_frac_at_min_censored": float(per_run_df["frac_at_min_censored"].mean()) if len(per_run_df) else float("nan"),
        "floor_exposure": float(per_run_df["floor_exposure_run"].mean()) if len(per_run_df) else float("nan"),
        "usable_fraction_total_mean": float(per_run_df["usable_fraction_total"].mean()) if len(per_run_df) else float("nan"),
        "usable_fraction_drop_mean": float(per_run_df["usable_fraction_drop"].mean()) if len(per_run_df) else float("nan"),
        "drawdown_at_first_crash": summarize_series(crash_runs["drawdown_at_first_crash"]),
        "ofi_first_crash": summarize_series(crash_runs["ofi_first_crash"]),
        "ofi_crash_window": summarize_series(crash_runs["ofi_crash_window_mean"]),
        "spread_crash_window": summarize_series(crash_runs["spread_crash_window_mean"]),
        "spread_pre": summarize_series(crash_runs["spread_pre_mean"]),
        "depth_crash_window": summarize_series(crash_runs["depth_crash_window_mean"]),
        "depth_pre": summarize_series(crash_runs["depth_pre_mean"]),
        "per_run_metrics_csv": str(per_run_path),
        "per_shard_metrics_csv": str(shard_path),
    }

    summary_path = args.output_dir / "step4_lock_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== STEP 4 FLOOR-CENSORED PANEL COMPLETE ===")
    print(f"Input shards              : {len(csv_paths)}")
    print(f"Runs                      : {len(per_run_df)}")
    print(f"guard_window              : {args.guard_window}")
    print(f"Crash rate raw            : {summary['crash_rate_raw']:.4f}")
    print(f"Crash rate censored       : {summary['crash_rate_censored']:.4f}")
    print(f"avg_frac_at_min censored  : {summary['avg_frac_at_min_censored']:.4f}")
    print(f"floor_exposure censored   : {summary['floor_exposure']:.4f}")
    print(f"usable_fraction_total     : {summary['usable_fraction_total_mean']:.4f}")
    print(f"usable_fraction_drop      : {summary['usable_fraction_drop_mean']:.4f}")
    print(f"OFI crash-window mean     : {summary['ofi_crash_window']['mean']:.4f}")
    print(f"Spread crash-window mean  : {summary['spread_crash_window']['mean']:.4f}")
    print(f"Depth crash-window mean   : {summary['depth_crash_window']['mean']:.4f}")
    print(f"Saved censored panel      : {args.panel_file}")
    print(f"Saved per-run metrics     : {per_run_path}")
    print(f"Saved summary             : {summary_path}")


if __name__ == "__main__":
    main()