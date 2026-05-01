"""Shared constants and helpers for the Phase 1 LLM elicitation pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PHASES = ["pre", "drop", "recovery", "post"]
RUNS_PER_PHASE = 32
DRY_RUN_SAMPLES = 3
SCHEMA_VERSION = "phase1_llm_elicitation.v1"
EPSILON = 1e-6

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DETAILS_DIR = PROJECT_ROOT / "prompt_details"

AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "momentum_trader": {
        "label": "Momentum Trader",
        "prompt_filename": "momentum_prompt.md",
        "spec_filename": "momentum_spec.md",
        "parameter_targets": [
            "Aggressiveness should stay in [0, 1] and move with continuation strength rather than collapse to one value.",
            "Cancel probability should remain low when conviction is strong, but it should still reflect uncertainty in weaker setups.",
            "Inventory sensitivity should be positive and moderate because the agent is wealth-normalized.",
            "Execution can favor market orders, but weak edges should still allow do_nothing instead of forced participation.",
        ],
        "fallback_priors": {
            "aggressiveness": {"dist": "beta", "params": {"alpha": 6.0, "beta": 2.0}},
            "cancel_probability": {"dist": "beta", "params": {"alpha": 1.0, "beta": 19.0}},
            "inventory_sensitivity_vg": {
                "dist": "gamma",
                "params": {"shape": 2.0, "scale": 0.05},
            },
            "order_size_multiplier": {
                "dist": "lognorm",
                "params": {"shape": 0.35, "loc": 0.0, "scale": 1.0},
            },
            "order_type_market_fraction": {"dist": "scalar", "params": {"value": 0.9}},
        },
    },
    "contrarian_trader": {
        "label": "Contrarian Trader",
        "prompt_filename": "prompt-for-contrarian.md",
        "spec_filename": "detailed-info-of-contrarian.md",
        "parameter_targets": [
            "Aggressiveness should rise with overshoot magnitude, but small deviations should still map to light conviction or no trade.",
            "Inventory sensitivity should stay positive, with a higher default mean than the momentum trader.",
            "Use moving-average deviation together with drop_from_local_pct when choosing conviction.",
            "Execution can lean market during sharp reversals without forcing the same order type in every response.",
        ],
        "fallback_priors": {
            "aggressiveness": {"dist": "beta", "params": {"alpha": 4.0, "beta": 3.0}},
            "cancel_probability": {"dist": "beta", "params": {"alpha": 1.0, "beta": 12.0}},
            "inventory_sensitivity_vg": {
                "dist": "gamma",
                "params": {"shape": 2.0, "scale": 0.075},
            },
            "order_size_multiplier": {
                "dist": "lognorm",
                "params": {"shape": 0.45, "loc": 0.0, "scale": 1.0},
            },
            "order_type_market_fraction": {"dist": "scalar", "params": {"value": 0.85}},
            "estar_activation_threshold_pct": {"dist": "scalar", "params": {"value": 0.10}},
            "max_size_commitment_deviation_pct": {"dist": "scalar", "params": {"value": 0.60}},
        },
    },
    "hft_market_maker": {
        "label": "HFT Market Maker",
        "prompt_filename": "MM_Prompt.md",
        "spec_filename": "MM_Detailed.md",
        "parameter_targets": [
            "Order type should usually be limit, but the market state should decide whether quoting or standing down is safer.",
            "Inventory sensitivity should stay positive because the market maker must skew back toward flat inventory.",
            "Quote updates should respect the 25% depth cap and leverage-aware caution.",
            "Event triggers should remain in the low single-digit bps range instead of collapsing to a fixed constant.",
        ],
        "fallback_priors": {
            "aggressiveness": {"dist": "beta", "params": {"alpha": 2.0, "beta": 6.0}},
            "cancel_probability": {"dist": "beta", "params": {"alpha": 6.0, "beta": 4.0}},
            "inventory_sensitivity_vg": {
                "dist": "gamma",
                "params": {"shape": 2.0, "scale": 0.08},
            },
            "order_size_multiplier": {
                "dist": "lognorm",
                "params": {"shape": 0.25, "loc": 0.0, "scale": 0.8},
            },
            "order_type_market_fraction": {"dist": "scalar", "params": {"value": 0.0}},
            "event_trigger_price_bps": {"dist": "scalar", "params": {"value": 3.0}},
            "max_order_frac_depth": {"dist": "scalar", "params": {"value": 0.25}},
            "leverage_factor": {"dist": "scalar", "params": {"value": 5.0}},
        },
    },
    "noise_trader": {
        "label": "Noise Trader",
        "prompt_filename": "Noise_Trader_Prompt.md",
        "spec_filename": "Noise_Trader_Detailed.md",
        "parameter_targets": [
            "Direction should stay approximately symmetric over many samples and remain volatility-scaled instead of following OFI or moving averages.",
            "Inventory sensitivity should stay positive and act as an absolute penalty, not a wealth-normalized one.",
            "Execution can favor market orders, but weak random draws should still allow do_nothing.",
            "Arrival rate comes from the phase anchor and should not be hallucinated.",
        ],
        "fallback_priors": {
            "aggressiveness": {"dist": "beta", "params": {"alpha": 3.0, "beta": 3.0}},
            "cancel_probability": {"dist": "beta", "params": {"alpha": 1.0, "beta": 20.0}},
            "inventory_sensitivity_vg": {
                "dist": "gamma",
                "params": {"shape": 2.0, "scale": 0.05},
            },
            "order_size_multiplier": {
                "dist": "lognorm",
                "params": {"shape": 0.55, "loc": 0.0, "scale": 1.1},
            },
            "order_type_market_fraction": {"dist": "scalar", "params": {"value": 1.0}},
            "aggressiveness_alpha": {"dist": "scalar", "params": {"value": 1.0}},
        },
    },
}

PROMPT_RECORD_COLUMNS = [
    "schema_version",
    "run_id",
    "run_number",
    "agent_type",
    "agent_label",
    "phase",
    "sample_origin",
    "sample_source_phase",
    "sample_row_index",
    "input_market_state_source",
    "input_market_state_mode",
    "input_anchor_source",
    "prompt_path",
    "spec_path",
    "event_id",
    "date",
    "timestamp_ms",
    "timestamp_utc",
    "time_from_drop_start_ms",
    "close_sample",
    "mid_price_sample",
    "moving_average_50_sample",
    "moving_average_200_sample",
    "price_vs_ma_50_pct_sample",
    "price_vs_ma_200_pct_sample",
    "current_inventory_units",
    "current_inventory_notional",
    "inventory_state",
    "ofi_sample",
    "trade_intensity_sample",
    "realized_vol_50_sample",
    "kyle_lambda_sample",
    "spread_bps_sample",
    "touch_depth_sample",
    "depth_imbalance_sample",
    "vpin_sample",
    "amihud_illiq_sample",
    "leverage_proxy_sample",
    "order_flow_toxicity_sample",
    "drop_from_local_pct_sample",
    "delta_from_news_ms_sample",
    "stress_proxy",
    "noise_trader_lambda_anchor",
    "order_size_pareto_alpha_anchor",
    "system_prompt",
    "user_prompt",
    "messages",
]

INFERENCE_RECORD_COLUMNS = PROMPT_RECORD_COLUMNS + [
    "model_name",
    "backend",
    "temperature",
    "max_tokens",
    "attempt_count",
    "inference_status",
    "raw_response_text",
]

PARSED_RECORD_COLUMNS = [
    col for col in INFERENCE_RECORD_COLUMNS if col not in {"messages", "system_prompt", "user_prompt"}
] + [
    "parse_status",
    "parse_error",
    "aggressiveness",
    "cancel_probability",
    "order_size_multiplier",
    "inventory_sensitivity",
    "order_type",
    "side",
    "reasoning_summary",
]

RESPONSE_SCHEMA = {
    "aggressiveness": "float in [0, 1]",
    "cancel_probability": "float in [0, 1]",
    "order_size_multiplier": "float in [0.1, 5.0]",
    "inventory_sensitivity": "float in [0, 1]",
    "order_type": "one of: market, limit",
    "side": "one of: buy, sell, do_nothing",
    "reasoning_summary": "short string explaining the choice",
}


def find_markdown_file(root_dir: Path, filename: str) -> Path:
    matches = sorted(root_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root_dir}")
    if len(matches) > 1:
        raise FileExistsError(f"Found multiple matches for {filename}: {matches}")
    return matches[0]


def load_agent_documents(root_dir: Path = PROMPT_DETAILS_DIR) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for agent_type, config in AGENT_CONFIGS.items():
        prompt_path = find_markdown_file(root_dir, config["prompt_filename"])
        spec_path = find_markdown_file(root_dir, config["spec_filename"])
        documents[agent_type] = {
            **config,
            "prompt_path": prompt_path,
            "spec_path": spec_path,
            "prompt_text": prompt_path.read_text(encoding="utf-8").strip(),
            "spec_text": spec_path.read_text(encoding="utf-8").strip(),
        }
    return documents


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def sanitize_json_response(text: str) -> str:
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0:
        if end >= start:
            cleaned = cleaned[start:end + 1]
        else:
            cleaned = cleaned[start:]

    if cleaned.count("{") == cleaned.count("}") + 1:
        cleaned = cleaned + "}"
    return cleaned


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    normalized_text = sanitize_json_response(text)
    if not normalized_text.strip():
        return None, "empty_response"

    start = normalized_text.find("{")
    if start < 0:
        return None, "missing_open_brace"

    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(normalized_text[start:])
    except json.JSONDecodeError as exc:
        return None, str(exc)

    if not isinstance(payload, dict):
        return None, "json_payload_not_object"
    return payload, None


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def existing_run_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record["run_id"]) for record in records if "run_id" in record}


def safe_float(value: Any, decimals: int = 8) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return round(numeric, decimals)


def clip_open_unit_interval(value: float) -> float:
    return min(max(float(value), EPSILON), 1.0 - EPSILON)
