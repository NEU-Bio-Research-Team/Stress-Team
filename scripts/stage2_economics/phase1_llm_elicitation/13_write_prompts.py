"""Script 13 - Generate Phase 1 LLM elicitation prompts.

Builds one prompt record per agent, phase, and run. Each record embeds:
    - the full agent prompt markdown as the system prompt
    - one sampled empirical market state row for the matching phase
    - per-phase anchors from prior_anchors.json

Resume behavior:
    - Existing run_ids in the output JSON are preserved.
    - Missing run_ids are generated and checkpointed incrementally.

Dry run behavior:
    - Writes 3 prompt records to a dry-run file by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from config.settings import PROCESSED_DIR

from common import (
    AGENT_CONFIGS,
    DRY_RUN_SAMPLES,
    PHASES,
    PROMPT_DETAILS_DIR,
    PROMPT_RECORD_COLUMNS,
    RESPONSE_SCHEMA,
    RUNS_PER_PHASE,
    SCHEMA_VERSION,
    atomic_write_json,
    existing_run_ids,
    load_agent_documents,
    load_json_records,
    safe_float,
)


INPUT_GRIDDED_CSV = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms_gridded.csv"
INPUT_FALLBACK_CSV = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms.csv"
ANCHORS_JSON = PROCESSED_DIR / "tardis" / "confounder_outputs" / "prior_anchors.json"
OUTPUT_DIR = PROCESSED_DIR / "tardis" / "phase1_outputs"
OUTPUT_JSON = OUTPUT_DIR / "phase1_prompts.json"
DRY_RUN_OUTPUT_JSON = OUTPUT_DIR / "phase1_prompts.dry_run.json"
RAW_SIGNAL_OVERLAY_COLUMNS = {
    "ofi",
    "trade_intensity",
    "amihud_illiq",
    "volume",
    "dollar_volume",
    "buy_ratio",
    "vwap",
    "log_return",
}
RAW_SIGNAL_TOLERANCE_MS = 50
SIGNAL_POOL_MIN_ROWS = 5

DEFAULT_RESPONSE_RULES = [
    "Return exactly one JSON object and no prose outside the JSON.",
    "Use only the provided market state, anchors, and role description.",
    "Keep all numeric values inside the stated bounds.",
    "Choose order_type from {market, limit} only.",
    "Choose side from {buy, sell, do_nothing} only.",
    "Keep reasoning_summary under 40 words.",
    "If the edge is weak, prefer do_nothing instead of inventing conviction.",
]

MARKET_STATE_COLUMNS = {
    "event_id": "event_id",
    "date": "date",
    "timestamp_ms": "timestamp_ms",
    "timestamp_utc": "timestamp_utc",
    "time_from_drop_start_ms": "time_from_drop_start_ms",
    "close_sample": "close",
    "mid_price_sample": "mid_price",
    "moving_average_50_sample": "moving_average_50",
    "moving_average_200_sample": "moving_average_200",
    "price_vs_ma_50_pct_sample": "price_vs_ma_50_pct",
    "price_vs_ma_200_pct_sample": "price_vs_ma_200_pct",
    "ofi_sample": "ofi",
    "trade_intensity_sample": "trade_intensity",
    "realized_vol_50_sample": "realized_vol_50",
    "kyle_lambda_sample": "kyle_lambda",
    "spread_bps_sample": "spread_bps",
    "touch_depth_sample": "touch_depth",
    "depth_imbalance_sample": "depth_imbalance",
    "vpin_sample": "vpin",
    "amihud_illiq_sample": "amihud_illiq",
    "leverage_proxy_sample": "leverage_proxy",
    "order_flow_toxicity_sample": "order_flow_toxicity",
    "drop_from_local_pct_sample": "drop_from_local_pct",
    "delta_from_news_ms_sample": "delta_from_news_ms",
}

NON_NEGATIVE_IMPUTED_COLUMNS = {
    "trade_intensity",
    "kyle_lambda",
    "realized_vol_50",
    "spread_bps",
    "touch_depth",
    "vpin",
    "amihud_illiq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 LLM elicitation prompts")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="Optional override for the market-state CSV")
    parser.add_argument("--anchors-json", type=Path, default=ANCHORS_JSON,
                        help="Per-phase anchor JSON")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional override for the output JSON")
    parser.add_argument("--runs-per-phase", type=int, default=RUNS_PER_PHASE,
                        help="Prompt count for each agent-phase pair")
    parser.add_argument("--seed", type=int, default=20260501,
                        help="Base seed used to deterministically sample rows")
    parser.add_argument("--raw-signal-tolerance-ms", type=int, default=RAW_SIGNAL_TOLERANCE_MS,
                        help="Nearest-neighbor tolerance used when overlaying raw signal columns onto gridded rows")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing output and regenerate all run_ids")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate only 3 prompt records and write a dry-run file")
    return parser.parse_args()


def choose_input_csv(override: Path | None) -> Path:
    if override is not None:
        return override
    if INPUT_GRIDDED_CSV.exists():
        return INPUT_GRIDDED_CSV
    return INPUT_FALLBACK_CSV


def load_anchors(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Anchor file not found: {path}")
    anchors = json.loads(path.read_text(encoding="utf-8"))
    phases = anchors.get("metadata", {}).get("phases", [])
    missing = [phase for phase in PHASES if phase not in phases]
    if missing:
        raise ValueError(f"Anchor file missing phases: {missing}")
    return anchors


def overlay_raw_signal_columns(gridded: pd.DataFrame, raw: pd.DataFrame, tolerance_ms: int) -> pd.DataFrame:
    required_columns = {"event_id", "timestamp_ms"}
    if not required_columns.issubset(gridded.columns) or not required_columns.issubset(raw.columns):
        return gridded

    overlay_columns = [column for column in RAW_SIGNAL_OVERLAY_COLUMNS if column in raw.columns]
    if not overlay_columns:
        return gridded

    left = gridded.copy()
    right = raw.copy()
    left["event_id"] = pd.to_numeric(left["event_id"], errors="coerce")
    left["timestamp_ms"] = pd.to_numeric(left["timestamp_ms"], errors="coerce")
    right["event_id"] = pd.to_numeric(right["event_id"], errors="coerce")
    right["timestamp_ms"] = pd.to_numeric(right["timestamp_ms"], errors="coerce")

    left = left.sort_values(["event_id", "timestamp_ms"]).reset_index(drop=True)
    right = right.sort_values(["event_id", "timestamp_ms"]).reset_index(drop=True)
    right = right[["event_id", "timestamp_ms", *overlay_columns]].rename(
        columns={column: f"{column}_raw" for column in overlay_columns}
    )

    merged = pd.merge_asof(
        left,
        right,
        on="timestamp_ms",
        by="event_id",
        direction="nearest",
        tolerance=tolerance_ms,
    )
    for column in overlay_columns:
        raw_column = f"{column}_raw"
        merged[column] = merged[raw_column].where(merged[raw_column].notna(), merged[column])
        merged = merged.drop(columns=[raw_column])
    return merged


def enrich_market_state(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["event_id"] = pd.to_numeric(enriched["event_id"], errors="coerce")
    enriched["timestamp_ms"] = pd.to_numeric(enriched["timestamp_ms"], errors="coerce")
    enriched = enriched.sort_values(["event_id", "timestamp_ms"]).reset_index(drop=True)

    close = pd.to_numeric(enriched.get("close"), errors="coerce")
    if "mid_price" not in enriched.columns:
        enriched["mid_price"] = close
    else:
        enriched["mid_price"] = pd.to_numeric(enriched["mid_price"], errors="coerce").fillna(close)
    enriched["close"] = close

    enriched["moving_average_50"] = (
        enriched.groupby("event_id", group_keys=False)["close"]
        .transform(lambda series: series.rolling(window=50, min_periods=1).mean())
    )
    enriched["moving_average_200"] = (
        enriched.groupby("event_id", group_keys=False)["close"]
        .transform(lambda series: series.rolling(window=200, min_periods=1).mean())
    )

    for source_column, target_column in (
        ("moving_average_50", "price_vs_ma_50_pct"),
        ("moving_average_200", "price_vs_ma_200_pct"),
    ):
        baseline = pd.to_numeric(enriched[source_column], errors="coerce")
        enriched[target_column] = np.where(
            baseline.abs() > 1e-9,
            (enriched["close"] / baseline - 1.0) * 100.0,
            np.nan,
        )

    return enriched


def load_market_state(path: Path, raw_signal_tolerance_ms: int) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Market-state CSV not found: {path}")
    df = pd.read_csv(path)
    merge_mode = "single_source"

    if path.resolve() == INPUT_GRIDDED_CSV.resolve() and INPUT_FALLBACK_CSV.exists():
        raw_df = pd.read_csv(INPUT_FALLBACK_CSV)
        df = overlay_raw_signal_columns(df, raw_df, tolerance_ms=raw_signal_tolerance_ms)
        merge_mode = "gridded_with_raw_signal_overlay"
    elif path.resolve() == INPUT_FALLBACK_CSV.resolve():
        merge_mode = "raw_only"
    else:
        merge_mode = "override_source"

    if "phase" not in df.columns:
        raise ValueError("Input CSV must contain a 'phase' column")
    return enrich_market_state(df), merge_mode


def choose_sampling_pool(phase_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    signal_mask = (
        pd.to_numeric(phase_df.get("ofi"), errors="coerce").fillna(0.0).abs() > 1e-6
    ) | (
        pd.to_numeric(phase_df.get("trade_intensity"), errors="coerce").fillna(0.0) > 1e-6
    )
    signal_rows = phase_df[signal_mask].copy()
    if len(signal_rows) >= SIGNAL_POOL_MIN_ROWS:
        return signal_rows, "signal_dense_phase_sample"
    return phase_df, "observed_phase_sample"


def phase_anchor_snapshot(anchors: dict[str, Any], phase: str) -> dict[str, Any]:
    return {
        "ofi_percentiles": anchors.get("ofi_percentiles_per_phase", {}).get(phase, {}),
        "trade_intensity": anchors.get("trade_intensity_per_phase", {}).get(phase, {}),
        "kyle_lambda": anchors.get("kyle_lambda_per_phase", {}).get(phase, {}),
        "realized_vol": anchors.get("realized_vol_per_phase", {}).get(phase, {}),
        "spread_bps": anchors.get("spread_bps_per_phase", {}).get(phase, {}),
        "depth_imbalance": anchors.get("depth_imbalance_per_phase", {}).get(phase, {}),
        "vpin": anchors.get("vpin_per_phase", {}).get(phase, {}),
        "amihud": anchors.get("amihud_per_phase", {}).get(phase, {}),
        "noise_trader_lambda": safe_float(anchors.get("noise_trader_lambda", {}).get(phase)),
        "order_size_pareto_alpha": safe_float(anchors.get("order_size_pareto_alpha", {}).get(phase)),
    }


def deterministic_index(phase_df: pd.DataFrame, run_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{run_id}".encode("utf-8")).hexdigest()
    local_seed = int(digest[:16], 16)
    rng = np.random.default_rng(local_seed)
    return int(rng.integers(0, len(phase_df)))


def fallback_phase_for(target_phase: str, available_phases: set[str]) -> str:
    preference = {
        "pre": ["pre", "drop", "recovery", "post"],
        "drop": ["drop", "recovery", "pre", "post"],
        "recovery": ["drop", "post", "pre", "recovery"],
        "post": ["pre", "recovery", "drop", "post"],
    }
    for candidate in preference[target_phase]:
        if candidate in available_phases:
            return candidate
    raise ValueError(f"No available donor phase for {target_phase}")


def phase_scale_from_percentiles(percentiles: dict[str, Any]) -> float:
    p25 = float(percentiles.get("p25") or 0.0)
    p75 = float(percentiles.get("p75") or 0.0)
    p05 = float(percentiles.get("p05") or 0.0)
    p95 = float(percentiles.get("p95") or 0.0)
    iqr = abs(p75 - p25)
    tail = abs(p95 - p05) / 2.0
    return max(iqr, tail, 1e-6)


def remap_numeric_to_phase(
    value: Any,
    donor_stats: dict[str, Any],
    target_stats: dict[str, Any],
) -> float | None:
    numeric = safe_float(value, decimals=12)
    donor_mean = safe_float(donor_stats.get("mean"), decimals=12)
    donor_std = safe_float(donor_stats.get("std"), decimals=12)
    target_mean = safe_float(target_stats.get("mean"), decimals=12)
    target_std = safe_float(target_stats.get("std"), decimals=12)
    if numeric is None or donor_mean is None or target_mean is None:
        return target_mean
    donor_std = donor_std if donor_std and donor_std > 1e-6 else 1.0
    target_std = target_std if target_std and target_std > 1e-6 else donor_std
    zscore = (numeric - donor_mean) / donor_std
    return round(target_mean + zscore * target_std, 8)


def impute_phase_row(
    donor_row: pd.Series,
    donor_phase: str,
    target_phase: str,
    anchors: dict[str, Any],
) -> pd.Series:
    row = donor_row.copy()
    row["phase"] = target_phase

    donor_ofi = anchors.get("ofi_percentiles_per_phase", {}).get(donor_phase, {})
    target_ofi = anchors.get("ofi_percentiles_per_phase", {}).get(target_phase, {})
    donor_ofi_center = float(donor_ofi.get("p50") or 0.0)
    target_ofi_center = float(target_ofi.get("p50") or 0.0)
    donor_ofi_scale = phase_scale_from_percentiles(donor_ofi)
    target_ofi_scale = phase_scale_from_percentiles(target_ofi)
    donor_ofi_value = safe_float(row.get("ofi"), decimals=12)
    if donor_ofi_value is not None:
        standardized_ofi = (donor_ofi_value - donor_ofi_center) / donor_ofi_scale
        row["ofi"] = round(target_ofi_center + standardized_ofi * target_ofi_scale, 8)

    stat_pairs = [
        ("trade_intensity", "trade_intensity_per_phase"),
        ("kyle_lambda", "kyle_lambda_per_phase"),
        ("realized_vol_50", "realized_vol_per_phase"),
        ("spread_bps", "spread_bps_per_phase"),
        ("depth_imbalance", "depth_imbalance_per_phase"),
        ("vpin", "vpin_per_phase"),
        ("amihud_illiq", "amihud_per_phase"),
    ]
    for column_name, anchor_key in stat_pairs:
        donor_stats = anchors.get(anchor_key, {}).get(donor_phase, {})
        target_stats = anchors.get(anchor_key, {}).get(target_phase, {})
        remapped = remap_numeric_to_phase(row.get(column_name), donor_stats, target_stats)
        if remapped is not None:
            if column_name in NON_NEGATIVE_IMPUTED_COLUMNS:
                remapped = max(0.0, remapped)
            if column_name == "vpin":
                remapped = min(remapped, 1.0)
            row[column_name] = remapped

    return row


def compute_stress_proxy(row: pd.Series, anchor_snapshot: dict[str, Any]) -> float:
    ofi_scale = max(
        abs(anchor_snapshot["ofi_percentiles"].get("p05") or 0.0),
        abs(anchor_snapshot["ofi_percentiles"].get("p95") or 0.0),
        1e-6,
    )
    vol_scale = max(abs(anchor_snapshot["realized_vol"].get("mean") or 0.0), 1e-6)
    spread_scale = max(abs(anchor_snapshot["spread_bps"].get("mean") or 0.0), 1e-6)

    ofi_component = abs(float(row.get("ofi", 0.0) or 0.0)) / ofi_scale
    vol_component = abs(float(row.get("realized_vol_50", 0.0) or 0.0)) / vol_scale
    spread_component = abs(float(row.get("spread_bps", 0.0) or 0.0)) / spread_scale
    return round((ofi_component + vol_component + spread_component) / 3.0, 6)


def build_system_prompt(agent_doc: dict[str, Any]) -> str:
    rule_block = "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(DEFAULT_RESPONSE_RULES, start=1))
    return (
        f"{agent_doc['prompt_text']}\n\n"
        "Additional response rules:\n"
        f"{rule_block}"
    )


def build_user_prompt(
    agent_doc: dict[str, Any],
    phase: str,
    row: pd.Series,
    anchor_snapshot: dict[str, Any],
    stress_proxy: float,
) -> str:
    targets = "\n".join(f"- {target}" for target in agent_doc["parameter_targets"])
    response_schema = json.dumps(RESPONSE_SCHEMA, indent=2)
    return f"""
Phase: {phase}

Empirical market state sampled from the event dynamics table:
- event_id: {int(row.get('event_id', -1))}
- timestamp_ms: {int(row.get('timestamp_ms', -1))}
- timestamp_utc: {row.get('timestamp_utc')}
- time_from_drop_start_ms: {safe_float(row.get('time_from_drop_start_ms'))}
- close: {safe_float(row.get('close'))}
- mid_price: {safe_float(row.get('mid_price'))}
- moving_average_50: {safe_float(row.get('moving_average_50'))}
- moving_average_200: {safe_float(row.get('moving_average_200'))}
- price_vs_ma_50_pct: {safe_float(row.get('price_vs_ma_50_pct'))}
- price_vs_ma_200_pct: {safe_float(row.get('price_vs_ma_200_pct'))}
- current_inventory_units: 0.0
- current_inventory_notional: 0.0
- inventory_state: flat
- ofi: {safe_float(row.get('ofi'))}
- trade_intensity: {safe_float(row.get('trade_intensity'))}
- realized_vol_50: {safe_float(row.get('realized_vol_50'))}
- kyle_lambda: {safe_float(row.get('kyle_lambda'))}
- spread_bps: {safe_float(row.get('spread_bps'))}
- touch_depth: {safe_float(row.get('touch_depth'))}
- depth_imbalance: {safe_float(row.get('depth_imbalance'))}
- vpin: {safe_float(row.get('vpin'))}
- amihud_illiq: {safe_float(row.get('amihud_illiq'))}
- leverage_proxy: {safe_float(row.get('leverage_proxy'))}
- order_flow_toxicity: {safe_float(row.get('order_flow_toxicity'))}
- drop_from_local_pct: {safe_float(row.get('drop_from_local_pct'))}
- delta_from_news_ms: {safe_float(row.get('delta_from_news_ms'))}

Per-phase empirical anchors:
{json.dumps(anchor_snapshot, indent=2)}

Agent-specific elicitation targets:
{targets}

Return one JSON object that matches this schema exactly:
{response_schema}

Use the sampled market state and the phase anchors to choose a realistic action for this specific archetype. The stress_proxy for this row is {stress_proxy}.
""".strip()


def build_prompt_record(
    agent_type: str,
    phase: str,
    run_number: int,
    row: pd.Series,
    anchor_snapshot: dict[str, Any],
    agent_doc: dict[str, Any],
    input_market_state_source: str,
    input_market_state_mode: str,
    input_anchor_source: str,
    project_root: Path,
    sample_origin: str,
    sample_source_phase: str,
) -> dict[str, Any]:
    stress_proxy = compute_stress_proxy(row, anchor_snapshot)
    system_prompt = build_system_prompt(agent_doc)
    user_prompt = build_user_prompt(agent_doc, phase, row, anchor_snapshot, stress_proxy)
    run_id = f"{agent_type}__{phase}__{run_number:02d}"

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_number": run_number,
        "agent_type": agent_type,
        "agent_label": agent_doc["label"],
        "phase": phase,
        "sample_origin": sample_origin,
        "sample_source_phase": sample_source_phase,
        "sample_row_index": int(row.name),
        "input_market_state_source": input_market_state_source,
        "input_market_state_mode": input_market_state_mode,
        "input_anchor_source": input_anchor_source,
        "prompt_path": agent_doc["prompt_path"].relative_to(project_root).as_posix(),
        "spec_path": agent_doc["spec_path"].relative_to(project_root).as_posix(),
        "stress_proxy": stress_proxy,
        "noise_trader_lambda_anchor": anchor_snapshot["noise_trader_lambda"],
        "order_size_pareto_alpha_anchor": anchor_snapshot["order_size_pareto_alpha"],
        "current_inventory_units": 0.0,
        "current_inventory_notional": 0.0,
        "inventory_state": "flat",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    for output_key, source_key in MARKET_STATE_COLUMNS.items():
        value = row.get(source_key)
        if output_key in {"event_id", "timestamp_ms", "sample_row_index"}:
            record[output_key] = None if pd.isna(value) else int(value)
        elif output_key in {"date", "timestamp_utc"}:
            record[output_key] = None if pd.isna(value) else str(value)
        else:
            record[output_key] = safe_float(value)

    return {column: record.get(column) for column in PROMPT_RECORD_COLUMNS}


def main() -> None:
    args = parse_args()

    input_csv = choose_input_csv(args.input_csv)
    output_json = args.output_json or (DRY_RUN_OUTPUT_JSON if args.dry_run else OUTPUT_JSON)

    anchors = load_anchors(args.anchors_json)
    market_state, market_state_mode = load_market_state(
        input_csv,
        raw_signal_tolerance_ms=args.raw_signal_tolerance_ms,
    )
    agent_docs = load_agent_documents(PROMPT_DETAILS_DIR)
    project_root = Path(__file__).resolve().parents[3]
    available_phases = set(market_state["phase"].dropna().astype(str).unique())
    missing_phases = [phase for phase in PHASES if phase not in available_phases]

    existing_records = [] if args.overwrite else load_json_records(output_json)
    records = list(existing_records)
    seen_run_ids = existing_run_ids(records)

    target_total = DRY_RUN_SAMPLES if args.dry_run else len(AGENT_CONFIGS) * len(PHASES) * args.runs_per_phase
    generated_now = 0

    print("=" * 70)
    print("Script 13: Generate Phase 1 Prompts")
    print("=" * 70)
    print(f"  Market state : {input_csv}")
    print(f"  Merge mode   : {market_state_mode}")
    print(f"  Anchors      : {args.anchors_json}")
    print(f"  Prompt root  : {PROMPT_DETAILS_DIR}")
    print(f"  Output       : {output_json}")
    print(f"  Existing     : {len(existing_records)} records")
    print(f"  Available phases in CSV: {sorted(available_phases)}")
    if missing_phases:
        print(f"  [WARN] Missing CSV phases: {missing_phases}")
        print("         Missing phases will use anchor-imputed snapshots based on the closest available donor phase.")

    for agent_type, agent_doc in agent_docs.items():
        for phase in PHASES:
            phase_df = market_state[market_state["phase"] == phase].copy().reset_index(drop=False)
            donor_phase = phase if not phase_df.empty else fallback_phase_for(phase, available_phases)
            donor_df = market_state[market_state["phase"] == donor_phase].copy().reset_index(drop=False)
            sampling_df, observed_origin = choose_sampling_pool(donor_df)

            runs_for_phase = 1 if args.dry_run else args.runs_per_phase
            for run_number in range(runs_for_phase):
                run_id = f"{agent_type}__{phase}__{run_number:02d}"
                if run_id in seen_run_ids:
                    continue
                if len(records) >= target_total:
                    break

                row_index = deterministic_index(sampling_df, run_id, args.seed)
                donor_row = sampling_df.iloc[row_index]
                sample_origin = observed_origin
                row = donor_row
                if phase_df.empty:
                    sample_origin = "anchor_imputed_phase_sample"
                    row = impute_phase_row(donor_row, donor_phase=donor_phase, target_phase=phase, anchors=anchors)
                anchor_snapshot = phase_anchor_snapshot(anchors, phase)
                record = build_prompt_record(
                    agent_type=agent_type,
                    phase=phase,
                    run_number=run_number,
                    row=row,
                    anchor_snapshot=anchor_snapshot,
                    agent_doc=agent_doc,
                    input_market_state_source=input_csv.name,
                    input_market_state_mode=market_state_mode,
                    input_anchor_source=args.anchors_json.name,
                    project_root=project_root,
                    sample_origin=sample_origin,
                    sample_source_phase=donor_phase,
                )
                records.append(record)
                seen_run_ids.add(run_id)
                generated_now += 1
                atomic_write_json(output_json, records)

                if args.dry_run and len(records) >= target_total:
                    break

            if len(records) >= target_total:
                break
        if len(records) >= target_total:
            break

    records = sorted(records, key=lambda record: str(record["run_id"]))
    atomic_write_json(output_json, records)

    df = pd.DataFrame(records)
    if not df.empty:
        counts = df.groupby(["agent_type", "phase"]).size().sort_index()
        print("\n  Record counts by agent and phase:")
        for (agent_type, phase), count in counts.items():
            print(f"    - {agent_type:18s} {phase:8s} {int(count):>3d}")

    print(f"\n  Generated now: {generated_now}")
    print(f"  Total records: {len(records)}")
    print(f"  Target total : {target_total}")

    expected_columns = [column for column in PROMPT_RECORD_COLUMNS if column not in set(df.columns)]
    if expected_columns:
        raise ValueError(f"Missing expected prompt columns: {expected_columns}")

    print("\nDone.")


if __name__ == "__main__":
    main()