"""
Script 11 – Compute Prior Anchors for Phase-1 LLM Elicitation
===============================================================
Aggregates per-phase empirical statistics from Event_Dynamics_100ms_gridded.csv
into a JSON file consumed by the Phase-1 prompting pipeline.

Output: prior_anchors.json with:
    - OFI percentiles per phase (5th, 25th, 50th, 75th, 95th)
    - Noise-trader arrival rate (Poisson λ = mean trade_intensity)
    - Order-size Pareto α (Clauset–Shalizi–Newman 2009 MLE)
    - Kyle λ per phase
    - Realized vol per phase
    - Spread statistics per phase
    - Depth imbalance per phase

Usage:
    python scripts/stage2_economics/11_compute_prior_anchors.py
"""

import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import PROCESSED_DIR

# ── Paths ─────────────────────────────────────────────────────────────
INPUT_CSV  = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms_gridded.csv"
INPUT_FALLBACK = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms.csv"
OUTPUT_JSON = PROCESSED_DIR / "tardis" / "confounder_outputs" / "prior_anchors.json"

PHASES = ["pre", "drop", "recovery", "post"]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def pareto_alpha_mle(x: np.ndarray) -> float:
    """
    Pareto tail index α via MLE (Clauset, Shalizi & Newman 2009).
    α̂ = n / Σ log(x_i / x_min)
    """
    x = x[x > 0]
    if len(x) < 10:
        return np.nan
    x_min = np.min(x)
    if x_min <= 0:
        return np.nan
    n = len(x)
    alpha = n / np.sum(np.log(x / x_min))
    return float(alpha)


def phase_quantiles(series: pd.Series) -> dict:
    """Compute quantiles for a series, return as dict."""
    s = series.dropna()
    if s.empty:
        return {f"p{int(q*100):02d}": None for q in QUANTILES}
    return {f"p{int(q*100):02d}": round(float(s.quantile(q)), 8) for q in QUANTILES}


def phase_nonzero_quantiles(series: pd.Series) -> dict:
    """Compute quantiles after excluding exact zero-filled bins."""
    s = series.dropna()
    s = s[s != 0]
    if s.empty:
        return {f"p{int(q*100):02d}": None for q in QUANTILES}
    return {f"p{int(q*100):02d}": round(float(s.quantile(q)), 8) for q in QUANTILES}


def phase_stats(series: pd.Series) -> dict:
    """Compute mean/std/median for a series."""
    s = series.dropna()
    if s.empty:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean": round(float(s.mean()), 8),
        "std": round(float(s.std()), 8),
        "median": round(float(s.median()), 8),
        "n": int(len(s)),
    }


def main():
    print("=" * 70)
    print("Script 11: Compute Prior Anchors")
    print("=" * 70)

    # Try gridded first, fall back to original
    if INPUT_CSV.exists():
        input_path = INPUT_CSV
    elif INPUT_FALLBACK.exists():
        input_path = INPUT_FALLBACK
        print(f"  [WARN] Gridded file not found, using raw: {INPUT_FALLBACK}")
    else:
        print(f"[!] No input file found. Run scripts 09-10 first.")
        return

    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows, {df['event_id'].nunique()} events")

    if "phase" not in df.columns:
        print("[!] 'phase' column missing. Cannot compute per-phase anchors.")
        return

    anchors = {
        "metadata": {
            "description": "Empirical prior anchors for Phase-1 LLM elicitation",
            "source": str(input_path.name),
            "n_events": int(df["event_id"].nunique()),
            "n_rows": int(len(df)),
            "phases": PHASES,
        },
        "ofi_percentiles_per_phase": {},
        "trade_intensity_per_phase": {},
        "kyle_lambda_per_phase": {},
        "realized_vol_per_phase": {},
        "spread_bps_per_phase": {},
        "depth_imbalance_per_phase": {},
        "vpin_per_phase": {},
        "amihud_per_phase": {},
        "noise_trader_lambda": {},
        "order_size_pareto_alpha": {},
    }

    for phase in PHASES:
        pdf = df[df["phase"] == phase]
        if pdf.empty:
            continue

        # OFI percentiles
        if "ofi" in df.columns:
            anchors["ofi_percentiles_per_phase"][phase] = phase_nonzero_quantiles(pdf["ofi"])

        # Trade intensity (Poisson λ proxy)
        if "trade_intensity" in df.columns:
            stats = phase_stats(pdf["trade_intensity"])
            anchors["trade_intensity_per_phase"][phase] = stats
            # Poisson λ = mean(trade_intensity)
            anchors["noise_trader_lambda"][phase] = stats["mean"]

        # Kyle's lambda
        if "kyle_lambda" in df.columns:
            anchors["kyle_lambda_per_phase"][phase] = phase_stats(pdf["kyle_lambda"])

        # Realized volatility
        if "realized_vol_50" in df.columns:
            anchors["realized_vol_per_phase"][phase] = phase_stats(pdf["realized_vol_50"])

        # Spread
        if "spread_bps" in df.columns:
            anchors["spread_bps_per_phase"][phase] = phase_stats(pdf["spread_bps"])

        # Depth imbalance
        if "depth_imbalance" in df.columns:
            anchors["depth_imbalance_per_phase"][phase] = phase_stats(pdf["depth_imbalance"])

        # VPIN
        if "vpin" in df.columns:
            anchors["vpin_per_phase"][phase] = phase_stats(pdf["vpin"])

        # Amihud
        if "amihud_illiq" in df.columns:
            amihud_clean = pdf["amihud_illiq"].replace([np.inf, -np.inf], np.nan)
            anchors["amihud_per_phase"][phase] = phase_stats(amihud_clean)

    # Global order-size Pareto α (from OFI magnitude as proxy for order size)
    if "ofi" in df.columns:
        ofi_abs = df["ofi"].abs().values
        for phase in PHASES:
            pdf = df[df["phase"] == phase]
            if not pdf.empty:
                alpha = pareto_alpha_mle(pdf["ofi"].abs().values)
                anchors["order_size_pareto_alpha"][phase] = round(alpha, 4) if not np.isnan(alpha) else None
        anchors["order_size_pareto_alpha"]["all"] = round(pareto_alpha_mle(ofi_abs), 4)

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(anchors, f, indent=2, default=str)

    print(f"\n  Prior anchors saved: {OUTPUT_JSON}")

    # Print summary
    for phase in PHASES:
        ofi_pcts = anchors["ofi_percentiles_per_phase"].get(phase, {})
        lam = anchors["noise_trader_lambda"].get(phase)
        alpha = anchors["order_size_pareto_alpha"].get(phase)
        rv = anchors["realized_vol_per_phase"].get(phase, {}).get("mean")
        print(f"  {phase:10s}  OFI_p50={ofi_pcts.get('p50','?'):>10}  "
              f"λ={lam if lam else '?':>10}  "
              f"α_pareto={alpha if alpha else '?':>8}  "
              f"rv_mean={rv if rv else '?':>12}")

    print("\nDone.")


if __name__ == "__main__":
    main()
