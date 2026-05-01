# Phase 1: LLM Elicitation — Kế Hoạch Triển Khai

> **Mục tiêu:** Dùng local LLM (Mistral-7B-Instruct, RTX 3090) để elicit behavioral priors cho 4 agent archetypes dưới dạng phân phối xác suất (Beta/Gamma/Pareto), được anchored bằng empirical statistics từ 66 BTC flash crash events trong `prior_anchors.json`. Output của Phase 1 là `behavioral_priors.json` — input trực tiếp cho Phase 2 LOB Simulator.

---

## I. Tổng Quan Input / Output

### Inputs (đã có sẵn — không cần chạy thêm)

| File | Mô tả |
|------|-------|
| `data/processed/tardis/confounder_outputs/prior_anchors.json` | Empirical anchors per phase: OFI percentiles, Kyle lambda, trade intensity, realized vol, spread proxy, Pareto alpha, depth imbalance, VPIN, Amihud |
| `data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv` | 66 flash crash events với market microstructure features: `ofi`, `realized_vol_50`, `trade_intensity`, `kyle_lambda`, `spread_bps proxy`, `touch_depth proxy`, `phase label`, v.v. |
| `momentum_prompt.md`, `prompt-for-contrarian.md`, `MM_Prompt.md`, `Noise_Trader_Prompt.md` | 4 agent prompt files (Google Drive) — mô tả identity + execution style + rules của từng agent |
| `momentum_spec.md`, `detailed-info-of-contrarian.md`, `MM_Detailed.md`, `Noise_Trader_Detailed.md` | 4 agent spec files (Google Drive) — parameters cần elicit và distribution priors mục tiêu |

### Outputs (cần tạo)

| File | Mô tả |
|------|-------|
| `data/processed/tardis/phase1_outputs/phase1_prompts.json` | 512 prompts (4 agents × 4 phases × 32 runs), mỗi prompt embed thống kê thật từ `prior_anchors.json` + sample market state từ `Event_Dynamics_100ms.csv` |
| `data/processed/tardis/phase1_outputs/raw_elicited.json` | Raw JSON responses từ LLM (512 records) |
| `data/processed/tardis/phase1_outputs/raw_elicited.csv` | Parsed, validated parameters per agent per phase |
| `data/processed/tardis/phase1_outputs/behavioral_priors.json` | Distribution params (Beta α/β, Gamma shape/scale, scalar means) per agent type → **INPUT cho Phase 2 LOB Simulator** |

---

## II. Parameters Cần Elicit Per Agent

### Momentum Trader

| Parameter | Distribution | Constraint |
|-----------|-------------|------------|
| `aggressiveness` | Beta(α, β) | Right-skewed, mean ~0.75 |
| `cancel_prob` | Beta(α, β) | Near 0, mean ~0.05 |
| `inventory_sensitivity_vg` | Gamma(shape, scale) | Default mean 0.10 |
| `order_type_market_fraction` | Scalar | Expected >0.80 |

### Contrarian Trader

| Parameter | Distribution | Constraint |
|-----------|-------------|------------|
| `inventory_sensitivity_vg` | Gamma(shape, scale) | Default mean 0.15 |
| `aggressiveness` (ESTAR) | Non-linear thresholds | LLM elicits deviation % tại max size commitment |
| `order_type_market_fraction` | Scalar | Expected >0.80 |

### HFT Market Maker

| Parameter | Distribution | Constraint |
|-----------|-------------|------------|
| `inventory_sensitivity_vg` | Gamma(shape, scale) | — |
| `event_trigger_price_bps` | Scalar | Range: 2–5 bps |
| `max_order_frac_depth` | Fixed | 0.25 (from spec) |
| `leverage_factor` | Scalar | From empirical HFT scale |
| `order_type_market_fraction` | Fixed | 0 (limit orders only) |

### Noise Trader

| Parameter | Distribution | Constraint |
|-----------|-------------|------------|
| `aggressiveness_alpha` | Scalar | Fixed multiplier |
| `inventory_sensitivity_vg` | Gamma(shape, scale) | Absolute penalty — không normalize theo wealth |
| `arrival_rate_lambda` | Scalar (Poisson) | Direct từ `noise_trader_lambda` trong `prior_anchors.json` |
| `order_type_market_fraction` | Fixed | 1.0 (market orders only) |

---

## III. Cấu Trúc Thư Mục

Tạo folder mới: `scripts/stage2_economics/phase1_llm_elicitation/`

```
scripts/stage2_economics/phase1_llm_elicitation/
├── 13_write_prompts.py        # Generate 512 prompts từ prior_anchors + Event_Dynamics
├── 14_run_elicitation.py      # Batch inference với vLLM (Mistral-7B)
├── 15_extract_parameters.py   # Parse, validate, lưu raw_elicited.csv
└── 16_fit_distributions.py    # Fit Beta/Gamma/Pareto, xuất behavioral_priors.json
```

---

## IV. Bước 0: Setup Môi Trường *(~1–2 giờ)*

### Tạo Conda Environment

```bash
conda create -n comosa_phase1 python=3.11 && conda activate comosa_phase1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.1 transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 bitsandbytes==0.43.3
pip install scipy pandas numpy jsonschema huggingface_hub tqdm
mkdir -p data/processed/tardis/phase1_outputs models/
```

### Download Model Mistral-7B-Instruct-v0.3 *(~30 phút — chạy 1 lần)*

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    local_dir='models/mistral-7b-instruct',
    ignore_patterns=['*.pt', 'original/']
)
```

> **Lý do chọn Mistral-7B thay vì FinGPT:** FinGPT được train cho sentiment classification (positive/negative/neutral), không elicit behavioral parameters dạng JSON. Mistral-7B-Instruct có instruction following tốt hơn cho structured output. Zero-shot elicitation (Strategy C) được ưu tiên vì cho greater epistemic diversity — điều kiện cần thiết để fit distributions có variance phù hợp.

---

## V. Bước 1: Script 13 — Viết Prompts *(~3–4 giờ)*

### Mục tiêu

Generate **512 prompts** = 4 agents × 4 phases × 32 runs. Mỗi prompt gồm 2 phần:

- **SYSTEM:** Toàn bộ identity section từ agent prompt file tương ứng (persona: agent là ai, tin gì, trade thế nào) + 7 rules bắt buộc ở cuối
- **USER:** Market state thực được sample ngẫu nhiên từ `Event_Dynamics_100ms.csv` đúng phase đó (không dùng giá trị cố định) + thống kê anchor từ `prior_anchors.json` + yêu cầu JSON output với schema cụ thể

### JSON Output Schema (yêu cầu LLM trả về)

```json
{
  "aggressiveness": float,           // [0, 1]
  "cancel_probability": float,       // [0, 1]
  "order_size_multiplier": float,    // [0.1, 5.0]
  "inventory_sensitivity": float,    // [0, 1]
  "order_type": "market" | "limit",
  "side": "buy" | "sell" | "do_nothing",
  "reasoning_summary": string
}
```

> **Temperature = 0.8** (bắt buộc — cần behavioral diversity để fit distributions, không phải 'best answer').

---

## VI. Bước 2: Script 14 — Chạy Elicitation với vLLM *(~2–4 giờ inference)*

### Cấu hình vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model='models/mistral-7b-instruct',
    quantization='awq',
    gpu_memory_utilization=0.85,
    max_model_len=4096
)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=400,
    stop=['}']  # Guarantee JSON closed
)
```

### Xử lý Batch & Checkpointing

- **Batch size:** 32 prompts/batch
- **Checkpoint:** Lưu sau mỗi batch (resume-safe vì inference lâu)
- **Retry logic:** Nếu response không phải valid JSON → retry tối đa 3 lần, sau đó skip và log

### Output Schema per Record (`raw_elicited.json`)

```
run_id | agent_type | phase | ofi_sample | stress_proxy | [all elicited params]
```

---

## VII. Bước 3: Script 15 — Extract & Validate Parameters *(~1 giờ)*

**Mục tiêu:** Parse raw JSON responses, validate schema, tách theo `agent_type` và `phase`. Parse rate mục tiêu **>85%**.

### Logic Xử lý

```python
import re

# Recover JSON block từ raw response
match = re.search(r'\{[^{}]*\}', text, re.DOTALL)

# Schema validation
assert 0 <= aggressiveness <= 1
assert 0 <= cancel_probability <= 1
assert 0.1 <= order_size_multiplier <= 5.0
assert 0 <= inventory_sensitivity <= 1
assert order_type in {"market", "limit"}
assert side in {"buy", "sell", "do_nothing"}

# Failed records: skip + log ra extract_errors.log (không raise exception)
```

### Output

`data/processed/tardis/phase1_outputs/raw_elicited.csv` — columns:

```
run_id | agent_type | phase | ofi_sample | stress_proxy |
aggressiveness | cancel_probability | order_size_multiplier |
inventory_sensitivity | order_type | side | reasoning_summary
```

In thống kê cuối: `Parsed N/512 (X%), Failed: Y` — breakdown theo `agent_type` + `phase`.

---

## VIII. Bước 4: Script 16 — Fit Distributions *(~1 giờ)*

**Mục tiêu:** Đọc `raw_elicited.csv`, group theo `agent_type + phase`, fit `scipy.stats` distributions cho từng behavioral parameter, xuất `behavioral_priors.json`.

### Distribution Fitting per Parameter

| Parameter | scipy.stats method | Output distribution |
|-----------|-------------------|---------------------|
| `aggressiveness` | `beta.fit(data)` | Beta(α, β) |
| `cancel_probability` | `beta.fit(data)` | Beta(α, β) |
| `order_size_multiplier` | `lognorm.fit(data)` | Lognormal(shape, loc, scale) |
| `inventory_sensitivity` | `beta.fit(data)` | Beta(α, β) |
| `order_type_market_fraction` | `count(market) / total` | Scalar per agent-phase group |

> Chỉ fit khi sub-group có **≥10 records**. Nếu ít hơn: log cảnh báo, sử dụng prior mặc định từ agent spec.

### Output Structure (`behavioral_priors.json`)

```json
{
  "agent_type": {
    "phase": {
      "param": {
        "dist": "beta",
        "params": { "alpha": ..., "beta": ... }
      }
    }
  }
}
```

---

## IX. Checklist Phase 1 Hoàn Thành

Phase 1 được coi là hoàn thành khi **tất cả 4 file output** tồn tại và hợp lệ:

- [ ] `phase1_prompts.json` — đúng **512 records** (4 agents × 4 phases × 32 runs)
- [ ] `raw_elicited.json` — đúng **512 records**
- [ ] `raw_elicited.csv` — parse rate **≥85%** (≥435 records, ít nhất 1 record mỗi agent-phase combo)
- [ ] `behavioral_priors.json` — có entry cho tất cả **16 agent-phase combos** (4 agents × 4 phases)

> **Kiểm tra nhanh trước khi chạy full:**
> ```bash
> python scripts/stage2_economics/phase1_llm_elicitation/1X_script_name.py --dry-run
> ```

---

## X. Timeline Ước Tính

| Bước | Task | Thời gian |
|------|------|-----------|
| 0 | Setup conda env + download model | ~1–2 giờ *(1 lần duy nhất)* |
| 1 | `13_write_prompts.py` | ~3–4 giờ code + test |
| 2 | `14_run_elicitation.py` (inference RTX 3090) | ~2–4 giờ |
| 3 | `15_extract_parameters.py` | ~1 giờ |
| 4 | `16_fit_distributions.py` | ~1 giờ |
| **Tổng** | | **~8–12 giờ** |

> Các bước phải chạy **sequential** (không song song). `behavioral_priors.json` là output cuối — Phase 2 LOB Simulator có thể bắt đầu ngay sau khi file này sẵn sàng.
