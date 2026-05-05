"""Step 1-3: Label-window clearance & config ranking for cfgC floor-censored causal panel.

Inputs:
  data/processed/tardis/phase2_outputs/cfgC_screen/analysis/per_run_floor_timing.csv
  (requires: first_crash_tick, first_floor_tick, has_crash, crash_before_floor, etc.)

Outputs:
  data/processed/tardis/phase2_outputs/cfgC_screen/analysis/step3_config_ranking.csv
  (final ranking with clearance rates per guard_window)

Workflow:
  Step 1: Compute crash_floor_gap_ticks per run, check label-window clearance
  Step 2: Aggregate clearance rates per (config, guard_window)
  Step 3: Rank configs by composite score
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "tardis" / "phase2_outputs" / "cfgC_screen" / "analysis" / "per_run_floor_timing.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_FILE.parent
DEFAULT_MAX_FEATURE_LAG = 50  # Empirical: max lookback for feature engineering (ticks)
DEFAULT_CRASH_WINDOW = 10  # From detector: crash is detected over 10-tick window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label-window clearance & ranking for floor-censored causal panel"
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-feature-lag", type=int, default=DEFAULT_MAX_FEATURE_LAG)
    parser.add_argument("--crash-window", type=int, default=DEFAULT_CRASH_WINDOW)
    parser.add_argument("--guard-windows", nargs="+", type=int, default=[5, 10, 20])
    return parser.parse_args()


def compute_label_clearance(
    run_df: pd.DataFrame,
    max_feature_lag: int,
    crash_window: int,
    guard_windows: list[int],
) -> dict[str, float]:
    """
    Check label-window clearance for each guard_window.

    A crash is "clear" if:
      first_crash_tick + crash_window < first_floor_tick - guard_window - max_feature_lag

    This ensures:
    1. Crash detector finishes (first_crash + crash_window)
    2. Guard region (±guard_window around first_floor)
    3. Feature lag is respected (max_feature_lag before floor)

    Returns dict: {f"clearance_gw{gw}": float (0-1)} for each guard_window
    """
    result = {}

    for gw in guard_windows:
        threshold = gw + max_feature_lag
        # Only check runs that have both crash and floor
        crash_runs = run_df[(run_df["has_crash"] == 1) & (run_df["has_floor"] == 1)]

        if len(crash_runs) == 0:
            result[f"clearance_gw{gw}"] = 1.0  # No crashes to worry about
            continue

        clear = (
            (crash_runs["first_crash_tick"] + crash_window)
            < (crash_runs["first_floor_tick"] - threshold)
        ).sum()

        result[f"clearance_gw{gw}"] = float(clear / len(crash_runs))

    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-run data
    if not args.input_file.exists():
        raise SystemExit(f"Input file not found: {args.input_file}")

    run_df = pd.read_csv(args.input_file)

    print("=== STEP 1: COMPUTE CRASH-FLOOR GAP & LABEL-WINDOW CLEARANCE ===\n")

    # Per-config aggregation
    config_rows = []

    for cfg in run_df["cfg"].unique():
        cfg_runs = run_df[run_df["cfg"] == cfg]

        # Overall metrics (already computed)
        crash_rate_total = cfg_runs["has_crash"].mean()
        pre_floor_crash_rate = cfg_runs["crash_before_floor"].mean()
        floor_hit_rate = cfg_runs["has_floor"].mean()
        avg_frac_at_min = cfg_runs["frac_at_min"].mean()
        usable_pre_floor_fraction = cfg_runs["usable_pre_floor_fraction"].mean()

        # Gap analysis: first_crash_tick to first_floor_tick
        runs_with_both = cfg_runs[(cfg_runs["has_crash"] == 1) & (cfg_runs["has_floor"] == 1)]

        if len(runs_with_both) > 0:
            gap = runs_with_both["first_floor_tick"] - runs_with_both["first_crash_tick"]
            gap_median = float(gap.median())
            gap_p25 = float(gap.quantile(0.25))
            gap_p75 = float(gap.quantile(0.75))
            n_crash_with_floor = len(runs_with_both)
        else:
            gap_median = float("nan")
            gap_p25 = float("nan")
            gap_p75 = float("nan")
            n_crash_with_floor = 0

        # Label-window clearance per guard_window
        clearance_dict = compute_label_clearance(
            cfg_runs,
            max_feature_lag=args.max_feature_lag,
            crash_window=args.crash_window,
            guard_windows=args.guard_windows,
        )

        row = {
            "cfg": cfg,
            "crash_rate_total": crash_rate_total,
            "pre_floor_crash_rate": pre_floor_crash_rate,
            "floor_hit_rate": floor_hit_rate,
            "avg_frac_at_min": avg_frac_at_min,
            "usable_pre_floor_fraction": usable_pre_floor_fraction,
            "n_runs": len(cfg_runs),
            "n_crash": int(cfg_runs["has_crash"].sum()),
            "n_crash_with_floor": n_crash_with_floor,
            "crash_floor_gap_median": gap_median,
            "crash_floor_gap_p25": gap_p25,
            "crash_floor_gap_p75": gap_p75,
        }
        row.update(clearance_dict)
        config_rows.append(row)

    df_config = pd.DataFrame(config_rows).sort_values("cfg").reset_index(drop=True)

    print("Per-config label-window clearance:")
    print(df_config[
        ["cfg", "crash_rate_total", "floor_hit_rate", "avg_frac_at_min",
         "usable_pre_floor_fraction", "crash_floor_gap_median",
         "clearance_gw5", "clearance_gw10", "clearance_gw20"]
    ].to_string(index=False))

    print("\n=== STEP 2: CHOOSE OPTIMAL GUARD_WINDOW ===\n")

    # Choose guard_window where >= 90% of crash runs pass, or if none, most permissive
    best_gw = None
    for gw in args.guard_windows:
        clearance_key = f"clearance_gw{gw}"
        min_clearance = df_config[clearance_key].min()
        mean_clearance = df_config[clearance_key].mean()
        print(f"guard_window={gw}t: min_clearance={min_clearance:.3f} mean_clearance={mean_clearance:.3f}")
        if mean_clearance >= 0.90 and best_gw is None:
            best_gw = gw

    if best_gw is None:
        # Choose most permissive (highest gw = most lenient)
        best_gw = args.guard_windows[-1]

    print(f"\n→ Selected guard_window = {best_gw} ticks")
    best_gw_key = f"clearance_gw{best_gw}"

    print("\n=== STEP 3: RANK CONFIGS BY COMPOSITE SCORE ===\n")

    # Scoring: prioritize configs that are:
    # Per thầy decision tree:
    # PRIMARY: Have crash signal (pre_floor_crash_rate > 0)
    # SECONDARY: Close to crash_rate_target = 0.10 (8–12%)
    # TERTIARY: Low floor_hit_rate, low avg_frac_at_min, high usable_pre_floor_fraction

    TARGET_CRASH_RATE = 0.10
    TARGET_RANGE = (0.08, 0.12)

    def score_fn(row):
        # Stage 1: Penalty for no crash signal (must have pre_floor_crash_rate > 0)
        if row["pre_floor_crash_rate"] <= 0:
            # Configs with no crash are last resort, large penalty
            return 999.0

        # Stage 2: Distance to target crash rate (lower is better)
        crash_dist = abs(row["pre_floor_crash_rate"] - TARGET_CRASH_RATE)
        if row["pre_floor_crash_rate"] < TARGET_RANGE[0]:
            # Too low, add penalty
            crash_dist += 0.05
        elif row["pre_floor_crash_rate"] > TARGET_RANGE[1]:
            # Too high, smaller penalty
            crash_dist += 0.02

        # Stage 3: Floor contamination penalties (lower is better)
        floor_penalty = row["floor_hit_rate"]
        frac_penalty = row["avg_frac_at_min"]

        # Stage 4: Usability bonus (higher is better, so negate)
        usable_bonus = -row["usable_pre_floor_fraction"]

        # Stage 5: Clearance bonus (higher is better)
        clearance_bonus = -row[best_gw_key]

        # Weighted composite (among configs with crash signal)
        score = (
            2.0 * crash_dist  # Primary among crash-bearing configs: closeness to target
            + 1.5 * floor_penalty  # Penalty for floor contamination
            + 1.0 * frac_penalty  # Penalty for frac_at_min
            + 0.5 * usable_bonus  # Bonus for usable fraction
            + 0.2 * clearance_bonus  # Bonus for clearance
        )
        return score

    df_config["score"] = df_config.apply(score_fn, axis=1)
    df_ranked = df_config.sort_values("score").reset_index(drop=True)

    print("Ranked configs (lower score = better):")
    print(
        df_ranked[
            [
                "cfg",
                "score",
                "pre_floor_crash_rate",
                "floor_hit_rate",
                "avg_frac_at_min",
                "usable_pre_floor_fraction",
                best_gw_key,
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    # Top candidates
    print("\n=== TOP CANDIDATES FOR STEP 4 (200-run lock test) ===\n")
    top_candidates = df_ranked.head(2)
    for idx, row in top_candidates.iterrows():
        print(f"Rank {idx+1}: {row['cfg']}")
        print(f"  crash_rate_target (pre-floor)   = {row['pre_floor_crash_rate']:.4f}")
        print(f"  floor_hit_rate                  = {row['floor_hit_rate']:.4f}")
        print(f"  avg_frac_at_min                 = {row['avg_frac_at_min']:.4f}")
        print(f"  usable_pre_floor_fraction       = {row['usable_pre_floor_fraction']:.4f}")
        print(f"  label_clearance (gw={best_gw}t)     = {row[best_gw_key]:.4f}")
        print(f"  crash_floor_gap_median (ticks)  = {row['crash_floor_gap_median']:.0f}")
        print(f"  composite_score                 = {row['score']:.4f}")
        print()

    # Save ranked output
    output_file = args.output_dir / "step3_config_ranking.csv"
    df_ranked.to_csv(output_file, index=False)
    print(f"Saved ranking table: {output_file}")

    return df_ranked


if __name__ == "__main__":
    main()
