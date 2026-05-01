Tìm ra rồi — đây là toàn bộ sơ đồ nguồn gốc và cơ chế gây zero.

***

## Nguồn gốc của `Event_Dynamics_100ms_gridded.csv`

File được tạo bởi **[`script 10`](https://github.com/NEU-Bio-Research-Team/Stress-Team/blob/a0b09aacdfd181782f92498719c5147816936524/scripts/stage2_economics/10_augment_dynamics_features.py)** — không phải script 09 hay bất kỳ script confounder nào khác. Chain đầy đủ là:

```
script 09  →  Event_Dynamics_100ms.csv          (raw, có gaps)
script 10  →  Event_Dynamics_100ms_gridded.csv  (regular 100ms grid)
script 13  →  phase1_prompts.json               (chọn gridded đầu tiên)
```

***

## Tại sao gridded bị zero-dominated

Script 10 áp dụng **3 fill strategy khác nhau** cho 3 nhóm cột, và đây chính là nguồn gốc của vấn đề:

```python
BOOK_STATE_COLS = ["close", "mid_price", "spread_bps", "depth_imbalance", "touch_depth"]
TRADE_FLOW_COLS = ["ofi", "trade_intensity"]          # ← zero-fill
DERIVED_COLS    = ["kyle_lambda", "vpin", "amihud_illiq", "realized_vol_50", ...]  # ← interpolate
```

Đoạn fill trong `reindex_event()`:
```python
for col in TRADE_FLOW_COLS:
    if col in edf.columns:
        edf[col] = edf[col].fillna(0.0)   # ← mọi gap 100ms trong ofi, trade_intensity → 0
```

Logic này **đúng về mặt tài chính**: khi không có trade nào land trong 100ms bin, OFI và trade intensity thật sự bằng 0. Vấn đề không phải zero-fill sai — mà là **tỷ lệ gap quá cao**. Nếu raw data có nhiều khoảng lặng (ví dụ: thị trường yên tĩnh ở phase `pre` và `post`), thì sau khi grid, phần lớn rows sẽ là 0. 

**`amihud_illiq` bị 0 toàn bộ** thì khác — nó thuộc `DERIVED_COLS` (được interpolate), nhưng nếu raw đã là NaN/0 toàn bộ do không có đủ volume trong crash window ở script 09, thì `interpolate + ffill().bfill()` cũng chỉ khuếch đại số 0 ban đầu đó. 

***

## Cách sửa logic đúng chỗ

Vấn đề thật sự ở **chỗ nào trong chain** và **cần sửa ở đâu**:

### Tùy chọn A — Sửa script 10 (fix tại nguồn gridded)

Thay vì zero-fill ngay lập tức, tính thêm metadata về density trước khi fill:

```python
# Thêm vào reindex_event():
n_original = len(edf)
n_grid = len(grid)
edf["trade_density"] = n_original / n_grid   # fraction of bins with real trades

# Zero-fill như cũ, nhưng thêm cột flag để script 13 biết
for col in TRADE_FLOW_COLS:
    if col in edf.columns:
        edf[f"{col}_observed"] = edf[col].notna()   # True = có real trade
        edf[col] = edf[col].fillna(0.0)
```

Script 13 sau đó có thể lọc: chỉ sample rows có `trade_density > 0.1` hoặc `ofi_observed == True`.

### Tùy chọn B — Sửa script 13 (fix tại sampling)

Đây là fix nhanh nhất và không cần rerun script 10. Trong `choose_input_csv()` và `load_market_state()`, thêm fallback theo signal density:

```python
def sample_phase_row(phase_df, run_id, seed):
    """Prefer rows with non-zero trade flow over pure gap-fill rows."""
    signal_rows = phase_df[
        (phase_df["ofi"].abs() > 1e-6) |
        (phase_df["trade_intensity"] > 1e-6)
    ]
    pool = signal_rows if len(signal_rows) >= 5 else phase_df
    # Dùng pool thay vì phase_df trong deterministic_index
    return pool
```

Điều này giữ nguyên gridded CSV (tốt cho NOTEARS/LiNGAM cần regular grid) nhưng chỉ sample từ rows có real market activity.

### Tùy chọn C — Merge raw + gridded (fix toàn diện nhất)

Dùng gridded cho các cột book-state (`spread_bps`, `touch_depth`, `close`, `mid_price`) nhưng lấy `ofi`, `trade_intensity` từ raw CSV, join theo `(event_id, timestamp_ms)`:

```python
def load_market_state_merged(gridded_path, raw_path):
    gridded = pd.read_csv(gridded_path)
    raw = pd.read_csv(raw_path)[["event_id", "timestamp_ms", "ofi",
                                  "trade_intensity", "amihud_illiq",
                                  "depth_imbalance"]]
    # Merge: gridded làm base, raw overwrite các flow columns
    merged = gridded.merge(raw, on=["event_id", "timestamp_ms"],
                            how="left", suffixes=("_grid", "_raw"))
    for col in ["ofi", "trade_intensity", "amihud_illiq", "depth_imbalance"]:
        merged[col] = merged[f"{col}_raw"].where(
            merged[f"{col}_raw"].notna(), merged[f"{col}_grid"]
        )
    return merged
```

***

## Điểm cần kiểm tra thêm trước khi chọn fix

Trước khi sửa, nên confirm thêm một điểm: **raw CSV (`Event_Dynamics_100ms.csv`) có ofi non-zero ở các phase drop/recovery không?** Nếu raw cũng toàn 0 thì vấn đề nằm sâu hơn ở script 06 (`06_micro_feature_engineering.py`) — là nơi tính OFI từ aggTrades parquet. Từ output terminal bạn đã chạy, zero-fraction của raw là khoảng ~89% — có nghĩa raw vẫn có ~11% non-zero rows, và đó là signal thật. Vậy **Tùy chọn B (fix script 13 bằng density-aware sampling)** là đường đi nhanh và an toàn nhất để unblock rerun ngay hôm nay.

Dưới đây là bản tổng hợp kỹ thuật đầy đủ dựa trên rà soát toàn bộ pipeline 4 scripts và output thực tế. Các nguyên nhân được xếp theo thứ tự causal, không phải theo mức độ dễ fix.

***

## Chuỗi lỗi causal theo 4 tầng

Vấn đề không phải là một lỗi đơn lẻ mà là **cascade failure** — mỗi tầng khuếch đại lỗi từ tầng trước. Script 16 không phải nguồn gốc; nó chỉ là nơi collapse trở nên quan sát được.

***

## Tầng 1 — Input Data (Script 13)

**Root cause: `choose_input_csv` ưu tiên gridded CSV nhưng file đó bị zero-dominated.**

Nhìn vào đúng đoạn code trong [`13_write_prompts.py`](https://github.com/NEU-Bio-Research-Team/Stress-Team/blob/a0b09aacdfd181782f92498719c5147816936524/scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py):

```python
def choose_input_csv(override: Path | None) -> Path:
    if override is not None:
        return override
    if INPUT_GRIDDED_CSV.exists():
        return INPUT_GRIDDED_CSV  # ← luôn ưu tiên gridded
    return INPUT_FALLBACK_CSV
```

File gridded có ~89.8% zero cho `ofi`, `trade_intensity`, `depth_imbalance`, và 100% zero cho `amihud_illiq`. Tất cả 18 trường này đều được inject thẳng vào `build_user_prompt()` dưới dạng literal numbers. Khi model nhìn thấy `ofi: 0.0`, `trade_intensity: 0.0`, `amihud_illiq: 0.0` cho hầu hết runs, nó không có signal gì để differentiate hành vi theo phase, nên sẽ fall back về stereotype của agent.

**Failure mode phụ:** `impute_phase_row()` có logic remap theo z-score, nhưng nó chỉ được gọi khi `phase_df.empty` — tức là khi phase không có row trong CSV. Khi phase có rows (dù toàn zero), imputation bị skip hoàn toàn.

***

## Tầng 2 — Prompt Contract (Script 13 + common.py)

**Root cause: hai lỗi độc lập cộng hưởng.**

**Lỗi 2a — Missing state variables.** Từ code `build_user_prompt()`, script 13 chỉ truyền vào: `close`, `ofi`, `trade_intensity`, `realized_vol_50`, `kyle_lambda`, `spread_bps`, `touch_depth`, `depth_imbalance`, `vpin`, `amihud_illiq`, `leverage_proxy`, `order_flow_toxicity`, `drop_from_local_pct`, `delta_from_news_ms`. Nhưng các prompt markdown yêu cầu thêm:

| Biến cần trong prompt | Agent | Script 13 có truyền? |
|---|---|---|
| `moving_average` / `long_term_MA` | Contrarian, Momentum | ❌ |
| `current_inventory` / `inventory` | MM, Noise, Contrarian | ❌ |
| `mid_price` | MM, Noise | ❌ |
| `position_pnl` / `unrealized_pnl` | MM | ❌ |

Hệ quả trực tiếp: model buộc phải hallucinate những giá trị này. Output `raw_elicited.csv` đã chứng minh — có row reasoning về "3.5x the long-term moving average" và "drop_from_local_pct = 1624.7%" dù script không cấp inventory hay MA state nào.

**Lỗi 2b — Target leakage.** Trong `common.py`, `AGENT_CONFIGS` chứa trường `parameter_targets` với các câu như:
- `"Cancel probability should stay close to zero"` (nhiều agent)
- `"Order type should usually be market"` (momentum, noise)
- `"Order type should be limit unless urgent"` (market maker)

Những câu này được embed **nguyên văn** vào user prompt qua `build_user_prompt()`. Đây là circular prior injection: pipeline đang elicit distribution từ LLM nhưng đồng thời bảo LLM output phải ra giá trị nào. Output đã confirm điều này: toàn bộ 510 parsed rows có `cancel_probability = 0.0`, MM 100% limit, còn lại 100% market.

***

## Tầng 3 — Inference Decoding (Script 14)

**Root cause: free-form generation không có structural constraint.**

Script 14 chỉ set `temperature` và `max_tokens` trong `SamplingParams` — không có `stop` sequences, không có guided decoding, không có JSON mode. Kết quả thực tế:

- 72/512 responses wrap JSON trong code fence ` ```json ... ``` `
- 5 responses append extra prose sau JSON
- 47 responses cần ít nhất 1 retry
- 2 responses fail parse hoàn toàn (contrarian drop+recovery)

`extract_json_object()` trong `common.py` chỉ scan `text.find('{')` và decode từ đó — nên code fence và extra text làm hỏng extraction logic. Với vLLM/LLM inference framework, đây là vấn đề hoàn toàn có thể tránh bằng guided decoding (outlines/lm-format-enforcer).

***

## Tầng 4 — Fitting (Script 16)

**Root cause: downstream damage, không phải independent bug.**

Script 16 không có lỗi logic riêng — nó chỉ fit đúng những gì được feed vào. Vì `cancel_probability` đã là toàn 0.0 từ tầng trước, Beta fitting bị degenerate: tham số `alpha` về `5e-05`, distribution là near-point-mass tại 0. `order_type_market_fraction = 1.0` chính xác là encode collapse từ tầng 2b. `behavioral_priors.json` về mặt kỹ thuật không sai — nó chỉ đang faithfully phản ánh corrupted elicitation.

***

## Thứ tự fix theo dependency order

Fix không đúng thứ tự sẽ tốn công rerun nhiều lần. Dependency chain buộc thứ tự sau:

1. **Fix `13_write_prompts.py`:** Merge raw + gridded thay vì chọn một. Cụ thể: dùng gridded cho `spread_bps`, `touch_depth`, `kyle_lambda` (vì gridded interpolate tốt hơn); nhưng lấy `ofi`, `trade_intensity`, `depth_imbalance`, `amihud_illiq` từ raw. Hoặc đơn giản hơn: thêm `--input-csv` flag trỏ vào raw CSV khi chạy.

2. **Thêm missing state variables vào `build_user_prompt()`:** Tối thiểu cần tính `moving_average` từ `close` column của phase window trước khi sample, và truyền `inventory=0` như initial state. Nếu không có inventory tracking thực, thì phải xóa các rule trong prompt markdown phụ thuộc vào inventory.

3. **Giảm leakage trong `common.py`:** Thay các `parameter_targets` kiểu prescriptive (`"cancel probability should stay close to zero"`) bằng descriptive (`"cancel probability reflects current market uncertainty; typical range 0.0–0.3"`). Elicitation nên capture uncertainty, không encode point estimate.

4. **Thêm structured decoding vào `14_run_inference.py`:** Với vLLM, dùng `guided_json=RESPONSE_SCHEMA` trong `SamplingParams`. Nếu framework không support, tối thiểu thêm `stop=["}"]` và strip code fences trước khi parse.

5. **Rerun 13 → 14 → 15 → 16 theo thứ tự.** Chạy 15+16 trên `raw_elicited.csv` hiện tại là vô nghĩa vì sẽ chỉ refit lên collapsed data.