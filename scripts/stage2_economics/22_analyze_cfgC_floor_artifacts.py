"""Analyze floor-crash timing artifacts for cfgC 9-config screen outputs.

Inputs:
  data/processed/tardis/phase2_outputs/cfgC_screen/lob_pilot_cfgC_*.csv
  data/processed/tardis/phase2_outputs/cfgC_screen/lob_pilot_cfgC_*.json

Outputs:
  data/processed/tardis/phase2_outputs/cfgC_screen/analysis/per_run_floor_timing.csv
  data/processed/tardis/phase2_outputs/cfgC_screen/analysis/per_config_floor_timing.csv

Metrics required by teacher (per config):
  - crash_rate_total
  - pre_floor_crash_rate
  - post_floor_crash_rate
  - crash_near_floor_fraction
  - avg_frac_at_min
  - floor_hit_rate
  - usable_pre_floor_fraction

Run-level extraction (per run):
  - first_floor_tick
  - first_crash_tick
  - crash_before_floor
  - crash_after_floor
  - crash_near_floor
  - frac_at_min
  - max_drawdown_pct
  - drawdown_at_first_crash
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "cfgC_screen"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "analysis"


@dataclass
class RunMetrics:
    cfg: str
    shard_name: str
    seed: int
    run_id: int
    event_id: int
    n_rows: int
    n_drop_rows: int
    floor_price: float
    first_floor_tick: float
    first_crash_tick: float
    crash_before_floor: int
    crash_after_floor: int
    crash_near_floor: int
    has_crash: int
    has_floor: int
    frac_at_min: float
    usable_pre_floor_fraction: float
    max_drawdown_pct: float
    drawdown_at_first_crash: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze floor artifacts for cfgC outputs")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-price-fraction", type=float, default=0.60)
    parser.add_argument("--floor-eps-mult", type=float, default=1.001)
    parser.add_argument("--guard-window", type=int, default=100,
                        help="Crash near-floor window: +/- guard_window ticks around first_floor_tick")
    return parser.parse_args()


def safe_seed_from_name(name: str) -> int:
    # Expected shard name suffix: ..._s100
    marker = "_s"
    pos = name.rfind(marker)
    if pos < 0:
        return -1
    try:
        return int(name[pos + len(marker):])
    except Exception:
        return -1


def safe_cfg_from_name(name: str) -> str:
    # Input shard name: cfgC_dim350_dsp020_s100 -> cfgC_dim350_dsp020
    parts = name.split("_")
    if len(parts) < 2:
        return name
    return "_".join(parts[:-1])


def compute_run_drawdown(price: pd.Series) -> pd.Series:
    running_max = price.cummax()
    dd = (running_max - price) / running_max * 100.0
    return dd


def classify_run(
    run_df: pd.DataFrame,
    cfg: str,
    shard_name: str,
    seed: int,
    run_id: int,
    min_price_fraction: float,
    floor_eps_mult: float,
    guard_window: int,
) -> RunMetrics:
    run_df = run_df.sort_values("tick_ms").reset_index(drop=True)
    n_rows = len(run_df)
    event_id = int(run_df["event_id"].iloc[0]) if "event_id" in run_df.columns else -1

    # Tick index inside this run (0, 1, 2, ...)
    run_df["tick_idx"] = np.arange(n_rows)

    pre_rows = run_df[run_df["phase"] == "pre"]
    drop_rows = run_df[run_df["phase"] == "drop"].copy()

    if len(pre_rows) == 0 or len(drop_rows) == 0:
        return RunMetrics(
            cfg=cfg,
            shard_name=shard_name,
            seed=seed,
            run_id=int(run_id),
            event_id=event_id,
            n_rows=n_rows,
            n_drop_rows=len(drop_rows),
            floor_price=float("nan"),
            first_floor_tick=float("nan"),
            first_crash_tick=float("nan"),
            crash_before_floor=0,
            crash_after_floor=0,
            crash_near_floor=0,
            has_crash=0,
            has_floor=0,
            frac_at_min=float("nan"),
            usable_pre_floor_fraction=float("nan"),
            max_drawdown_pct=float("nan"),
            drawdown_at_first_crash=float("nan"),
        )

    init_price = float(pre_rows["close"].iloc[-1])
    floor_price = init_price * min_price_fraction

    # Floor events inside drop phase only
    drop_rows["at_floor"] = drop_rows["close"] <= floor_price * floor_eps_mult
    floor_ticks = drop_rows.loc[drop_rows["at_floor"], "tick_idx"]
    has_floor = int(len(floor_ticks) > 0)
    first_floor_tick = float(floor_ticks.iloc[0]) if has_floor else float("nan")

    # Crash events in full run panel (flash_crash_flag from detector)
    crash_rows = run_df[run_df["flash_crash_flag"] > 0]
    has_crash = int(len(crash_rows) > 0)
    first_crash_tick = float(crash_rows["tick_idx"].iloc[0]) if has_crash else float("nan")

    crash_before_floor = 0
    crash_after_floor = 0
    crash_near_floor = 0

    if has_crash:
        crash_ticks = crash_rows["tick_idx"].to_numpy()

        if has_floor:
            crash_before_floor = int(np.any(crash_ticks < (first_floor_tick - guard_window)))
            crash_after_floor = int(np.any(crash_ticks > (first_floor_tick + guard_window)))
            crash_near_floor = int(np.any(np.abs(crash_ticks - first_floor_tick) <= guard_window))
        else:
            # If no floor hit, crashes are usable pre-floor by definition.
            crash_before_floor = 1
            crash_after_floor = 0
            crash_near_floor = 0

    frac_at_min = float(drop_rows["at_floor"].mean())

    # Usable data fraction after censoring all drop rows from first floor onward.
    if has_floor:
        usable_pre_floor_fraction = float((drop_rows["tick_idx"] < first_floor_tick).mean())
    else:
        usable_pre_floor_fraction = 1.0

    # Drawdown metrics
    dd_series = compute_run_drawdown(run_df["close"].astype(float))
    max_drawdown_pct = float(dd_series.max())

    if has_crash:
        first_idx = int(first_crash_tick)
        drawdown_at_first_crash = float(dd_series.iloc[first_idx])
    else:
        drawdown_at_first_crash = float("nan")

    return RunMetrics(
        cfg=cfg,
        shard_name=shard_name,
        seed=seed,
        run_id=int(run_id),
        event_id=event_id,
        n_rows=n_rows,
        n_drop_rows=len(drop_rows),
        floor_price=floor_price,
        first_floor_tick=first_floor_tick,
        first_crash_tick=first_crash_tick,
        crash_before_floor=crash_before_floor,
        crash_after_floor=crash_after_floor,
        crash_near_floor=crash_near_floor,
        has_crash=has_crash,
        has_floor=has_floor,
        frac_at_min=frac_at_min,
        usable_pre_floor_fraction=usable_pre_floor_fraction,
        max_drawdown_pct=max_drawdown_pct,
        drawdown_at_first_crash=drawdown_at_first_crash,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(glob.glob(str(args.input_dir / "lob_pilot_cfgC_*.csv")))
    if not csv_paths:
        raise SystemExit(f"No cfgC CSV files found in {args.input_dir}")

    run_metrics: list[RunMetrics] = []
    shard_json_records: list[dict] = []

    for csv_path in csv_paths:
        csv_file = Path(csv_path)
        shard_name = csv_file.stem.replace("lob_pilot_", "")
        cfg = safe_cfg_from_name(shard_name)
        seed = safe_seed_from_name(shard_name)

        df = pd.read_csv(csv_file)
        required_cols = {"run_id", "phase", "close", "flash_crash_flag", "tick_ms"}
        if not required_cols.issubset(df.columns):
            missing = required_cols.difference(df.columns)
            raise ValueError(f"Missing columns in {csv_file.name}: {sorted(missing)}")

        # Per-run extraction
        for run_id, run_df in df.groupby("run_id"):
            m = classify_run(
                run_df=run_df,
                cfg=cfg,
                shard_name=shard_name,
                seed=seed,
                run_id=int(run_id),
                min_price_fraction=args.min_price_fraction,
                floor_eps_mult=args.floor_eps_mult,
                guard_window=args.guard_window,
            )
            run_metrics.append(m)

        # Bring in shard-level JSON summary for crash_rate_total and drawdown stats
        json_file = csv_file.with_suffix(".json")
        if json_file.exists():
            with json_file.open("r", encoding="utf-8") as f:
                js = json.load(f)
            dd_obj = js.get("run_max_drawdown_pct", {})
            shard_json_records.append(
                {
                    "cfg": cfg,
                    "shard_name": shard_name,
                    "seed": seed,
                    "flash_crash_rate": float(js.get("flash_crash_rate", np.nan)),
                    "run_max_drawdown_mean": float(dd_obj.get("mean", np.nan)) if isinstance(dd_obj, dict) else np.nan,
                    "run_max_drawdown_p95": float(dd_obj.get("p95", np.nan)) if isinstance(dd_obj, dict) else np.nan,
                    "run_max_drawdown_max": float(dd_obj.get("max", np.nan)) if isinstance(dd_obj, dict) else np.nan,
                }
            )

    run_df = pd.DataFrame([m.__dict__ for m in run_metrics])
    shard_df = pd.DataFrame(shard_json_records)

    per_run_path = args.output_dir / "per_run_floor_timing.csv"
    run_df.to_csv(per_run_path, index=False)

    # Config-level aggregation required by teacher
    cfg_agg = (
        run_df.groupby("cfg", as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            floor_hit_rate=("has_floor", "mean"),
            avg_frac_at_min=("frac_at_min", "mean"),
            pre_floor_crash_rate=("crash_before_floor", "mean"),
            post_floor_crash_rate=("crash_after_floor", "mean"),
            crash_near_floor_fraction=("crash_near_floor", "mean"),
            usable_pre_floor_fraction=("usable_pre_floor_fraction", "mean"),
            first_floor_tick_median=("first_floor_tick", "median"),
            first_crash_tick_median=("first_crash_tick", "median"),
            max_drawdown_pct_mean=("max_drawdown_pct", "mean"),
            max_drawdown_pct_p95=("max_drawdown_pct", lambda x: float(np.nanpercentile(x, 95))),
            drawdown_at_first_crash_mean=("drawdown_at_first_crash", "mean"),
        )
    )

    if len(shard_df) > 0:
        crash_from_json = (
            shard_df.groupby("cfg", as_index=False)
            .agg(
                crash_rate_total=("flash_crash_rate", "mean"),
                run_max_drawdown_json_mean=("run_max_drawdown_mean", "mean"),
                run_max_drawdown_json_p95=("run_max_drawdown_p95", "mean"),
                run_max_drawdown_json_max=("run_max_drawdown_max", "max"),
            )
        )
        cfg_agg = cfg_agg.merge(crash_from_json, on="cfg", how="left")
    else:
        cfg_agg["crash_rate_total"] = np.nan

    # Deterministic ordering by dim then dsp from cfg name
    cfg_agg = cfg_agg.sort_values("cfg").reset_index(drop=True)

    per_cfg_path = args.output_dir / "per_config_floor_timing.csv"
    cfg_agg.to_csv(per_cfg_path, index=False)

    print("=== CFGC FLOOR-TIMING ANALYSIS COMPLETE ===")
    print(f"Input CSV shards : {len(csv_paths)}")
    print(f"Run-level rows    : {len(run_df)}")
    print(f"Saved per-run     : {per_run_path}")
    print(f"Saved per-config  : {per_cfg_path}")
    print("\n=== PER-CONFIG SUMMARY (teacher metrics) ===")
    cols = [
        "cfg",
        "crash_rate_total",
        "pre_floor_crash_rate",
        "post_floor_crash_rate",
        "crash_near_floor_fraction",
        "avg_frac_at_min",
        "floor_hit_rate",
        "usable_pre_floor_fraction",
    ]
    print(cfg_agg[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
