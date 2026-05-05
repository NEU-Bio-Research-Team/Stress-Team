"""
19_analyze_dim_sweep.py
Aggregate results from the drop_impact_mult sweep.

Reads:
  data/processed/tardis/phase2_outputs/dim_sweep/lob_sweep_dim{DIM}_s{SEED}.json  (summary)
  data/processed/tardis/phase2_outputs/dim_sweep/lob_sweep_dim{DIM}_s{SEED}.csv   (tick panel)

Computes for each shard:
  - flash_crash_rate       (from JSON)
  - mean_drawdown_pct      (from JSON)
  - avg_frac_at_min        (fraction of drop-phase ticks where price <= floor_price)

Decision criterion:
  Choose lowest dim where crash_rate > 0 AND avg_frac_at_min < 0.20
"""

import json
import glob
import os
import pandas as pd
import numpy as np

SWEEP_DIR = "data/processed/tardis/phase2_outputs/dim_sweep"
FLOOR_FRAC = 0.85  # --min-price-fraction used in sweep
DIMS = [80, 100, 120, 150, 180, 220, 300, 400, 500, 600]
SEEDS = [42, 77]


def compute_frac_at_min(csv_path: str, floor_frac: float = FLOOR_FRAC) -> float:
    """Mean per-run fraction of drop-phase rows at/under the per-run floor."""
    if not os.path.exists(csv_path):
        return float("nan")
    df = pd.read_csv(csv_path)
    if "phase" not in df.columns:
        return float("nan")

    # Price column: try 'close', 'mid', 'price'
    price_col = next((c for c in ["close", "mid", "price"] if c in df.columns), None)
    if price_col is None:
        return float("nan")

    # If run_id is unavailable, fall back to file-level estimate.
    if "run_id" not in df.columns:
        drop_rows = df[df["phase"] == "drop"]
        pre_rows = df[df["phase"] == "pre"]
        if len(drop_rows) == 0 or len(pre_rows) == 0:
            return float("nan")
        init_price = pre_rows[price_col].iloc[-1]
        floor_price = init_price * floor_frac
        return float((drop_rows[price_col] <= floor_price * 1.001).mean())

    per_run_fracs = []
    for _, run_df in df.groupby("run_id"):
        drop_rows = run_df[run_df["phase"] == "drop"]
        pre_rows = run_df[run_df["phase"] == "pre"]
        if len(drop_rows) == 0 or len(pre_rows) == 0:
            continue
        init_price = pre_rows[price_col].iloc[-1]
        floor_price = init_price * floor_frac
        per_run_fracs.append(float((drop_rows[price_col] <= floor_price * 1.001).mean()))

    if len(per_run_fracs) == 0:
        return float("nan")
    return float(np.mean(per_run_fracs))


records = []
for dim in DIMS:
    dim_str = f"{dim:03d}"
    for seed in SEEDS:
        base = f"{SWEEP_DIR}/lob_sweep_dim{dim_str}_s{seed}"
        json_path = base + ".json"
        csv_path = base + ".csv"

        row = {"dim": dim / 100, "dim_int": dim, "seed": seed}

        # --- JSON summary ---
        if os.path.exists(json_path):
            with open(json_path) as f:
                summary = json.load(f)
            row["crash_rate"] = summary.get("flash_crash_rate", float("nan"))
            row["mean_drawdown_pct"] = summary.get("mean_drawdown_pct", float("nan"))
            row["p95_drawdown_pct"] = summary.get("p95_drawdown_pct", float("nan"))
            row["json_ok"] = True
        else:
            row["crash_rate"] = float("nan")
            row["mean_drawdown_pct"] = float("nan")
            row["p95_drawdown_pct"] = float("nan")
            row["json_ok"] = False

        # --- CSV panel: frac_at_min ---
        row["frac_at_min"] = compute_frac_at_min(csv_path)
        row["csv_ok"] = os.path.exists(csv_path)

        records.append(row)

df_results = pd.DataFrame(records)

print("\n=== DIM SWEEP RESULTS ===")
print(df_results.to_string(index=False))

# Average across seeds
agg = (
    df_results.groupby("dim_int")
    .agg(
        crash_rate=("crash_rate", "mean"),
        frac_at_min=("frac_at_min", "mean"),
        mean_drawdown=("mean_drawdown_pct", "mean"),
        n_json=("json_ok", "sum"),
    )
    .reset_index()
)

print("\n=== AGGREGATED (mean across seeds) ===")
print(agg.to_string(index=False))

# Decision
print("\n=== DECISION CRITERION: crash_rate > 0 AND frac_at_min < 0.20 ===")
candidates = agg[(agg["crash_rate"] > 0) & (agg["frac_at_min"] < 0.20)]
if len(candidates) == 0:
    print("NO candidate found. Consider increasing dim or adjusting resilience.")
else:
    best = candidates.sort_values("dim_int").iloc[0]
    print(f"Recommended dim_impact_mult = {best['dim_int']/100:.2f}")
    print(f"  crash_rate = {best['crash_rate']:.3f}")
    print(f"  frac_at_min = {best['frac_at_min']:.3f}")
    print(f"  mean_drawdown = {best['mean_drawdown']:.3f}%")

# Save
out_path = f"{SWEEP_DIR}/sweep_summary.csv"
df_results.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
