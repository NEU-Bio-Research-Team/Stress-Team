# Task: Overview Plan
Tôi đã có đủ toàn bộ context. Đây là phân tích đầy đủ và actionable plan.

***

## Phân Tích Toàn Diện: Normal Week Data cho COMOSA Phase 1

***

### Phần 1 — Report về 66 Events: Khoảng Thời Gian Nào?

Từ `config/settings.py`, dataset chạy từ **`2020-06-01` đến `2024-12-31`** (4.5 năm) , được lọc với ngưỡng drop ≥ 3.0% trong 5 phút, separation ≥ 4 giờ . Đây là bức tranh tổng thể:

```
Khoảng nguồn data:  2020-06-01 → 2024-12-31  (~4.5 năm, BTCUSDT Futures)
Event detection:    drop ≥ 3% trong 5 min, recovery ≥ 50% trong 30 min
66 events ≈ trung bình ~15 events/năm → ~1.2 events/tháng
```

Dựa trên lịch sử BTC đã biết, 66 events đó **phân bố không đều** theo 5 cụm chính:

| Cụm thời gian | Regime thị trường | Ước tính số events |
|---|---|---|
| 2020-06 → 2020-12 | COVID recovery, bull start | ~8–10 |
| 2021-01 → 2021-05 | Bull peak (ATH $65k), May crash | ~15–18 |
| 2021-06 → 2021-12 | Recovery, ATH $69k (Nov), correction | ~10–12 |
| 2022-01 → 2022-12 | Bear market (LUNA, FTX collapse) | ~18–20 |
| 2023-01 → 2024-12 | Recovery + ETF approval (Jan 2024) | ~8–10 |

**Bạn có cần tôi viết report không?** Câu trả lời là **có, nên có**, vì reviewer sẽ hỏi ngay. Report tối thiểu cần: histogram phân bố events theo năm/tháng, breakdown flash vs trend crash, distribution of drop_pct và duration_min. Tôi có thể generate script cho report này (xem Phần 3).

***

### Phần 2 — Nên Tải Normal Week Ở Khoảng Thời Gian Nào?

Tiêu chí khoa học để chọn tuần "normal":

**Phải thoả mãn đồng thời:**
1. **Không có event nào trong 66 events** trong khoảng ±3 ngày
2. **Low realized volatility**: BTC 7-day realized vol dưới median của toàn bộ sample (2020–2024)
3. **Cùng time range với event data** (2020–2024) để distributional shift không quá lớn
4. **Tránh các market microstructure anomaly đã biết**: exchange maintenance, API outage, halving weeks

**Các tuần cụ thể nên tải — theo thứ tự ưu tiên:**

| Tuần | Regime | Lý do chọn |
|---|---|---|
| `2021-10-04 → 2021-10-08` | Mid-bull, stable | Sau correction tháng 9, trước ATH push; low vol, high liquidity |
| `2023-02-06 → 2023-02-10` | Post-bear recovery | Thị trường đã stabilize sau FTX; vol thấp, order flow organic |
| `2024-03-11 → 2024-03-15` | Post-ETF settling | ETF launch dust settled (Jan), trước halving (April); liquidity cao |
| `2022-08-08 → 2022-08-12` | Bear mid-consolidation | Giữa LUNA crash (May) và FTX (Nov); tương đối flat |

**Khuyến nghị chọn 2 tuần** thay vì 1, từ hai regime khác nhau (bull + bear), để prior_anchors.json có thể phân biệt `normal_bull` vs `normal_bear`. Điều này cho phép Condition A trong ablation study §5.3 có baseline realistic hơn "uniform".

***

### Phần 3 — Scripts Cần Update Để Normal Week Merge Hoàn Hảo Vào Workflow

Đây là toàn bộ delta cần làm, theo thứ tự pipeline:

#### A. `config/settings.py` — Thêm constants mới

```python
# ──────────────────── Normal-Market Baseline ──────────────────────────
# Weeks selected to represent unbiased market microstructure for
# Noise Agent + MM Agent calibration (Phase 1 prior anchors §5.3)
NORMAL_WEEK_WINDOWS = [
    {"label": "normal_bull", "start": "2021-10-04", "end": "2021-10-08"},
    {"label": "normal_bear", "start": "2023-02-06", "end": "2023-02-10"},
]
NORMAL_WEEK_DIR   = PROCESSED_DIR / "tardis" / "normal_baseline"
NORMAL_WEEK_STATS = NORMAL_WEEK_DIR / "baseline_prior_stats.json"
```

#### B. `scripts/stage2_economics/05b_download_normal_week.py` — Script mới

Script này đã có draft từ trả lời trước. Cần thêm: loop qua `NORMAL_WEEK_WINDOWS`, save theo `label`, output `baseline_prior_stats.json` thay vì CSV.

#### C. `scripts/stage2_economics/06_micro_feature_engineering.py` — Thêm `--mode` flag

Script 06 hiện chỉ chạy trên `EVENT_RAW_DIR` . Cần thêm:

```python
parser.add_argument("--mode", choices=["event", "normal"], default="event",
    help="'event' = existing crash-window pipeline; "
         "'normal' = process normal_week baseline data")
parser.add_argument("--normal-dir", default=str(NORMAL_WEEK_DIR))
```

Khi `--mode normal`, script đọc từ `NORMAL_WEEK_DIR` thay vì `EVENT_RAW_DIR`, và skip phase-labeling (không có pre/drop/recovery/post — toàn bộ là `normal`).

#### D. `scripts/stage2_economics/11_compute_prior_anchors.py` — Thêm `normal` phase

Script 11 hiện chỉ tính anchors cho `PHASES = ["pre", "drop", "recovery", "post"]` . Cần thêm:

```python
# Thêm vào đầu file:
NORMAL_STATS_PATH = PROCESSED_DIR / "tardis" / "normal_baseline" / "baseline_prior_stats.json"

# Trong main(), sau khi load event data:
if NORMAL_STATS_PATH.exists():
    with open(NORMAL_STATS_PATH) as f:
        normal_stats = json.load(f)
    # Merge vào anchors với key "normal_bull" và "normal_bear"
    anchors["normal_baseline"] = normal_stats
    anchors["metadata"]["has_normal_baseline"] = True
    print(f"  [OK] Normal baseline loaded: {NORMAL_STATS_PATH}")
else:
    anchors["metadata"]["has_normal_baseline"] = False
    print(f"  [WARN] No normal baseline found. Run 05b first.")
```

#### E. `scripts/stage2_economics/18_lob_mini_runner.py` — Dùng normal baseline trong init

Hiện tại Noise Agent và MM Agent khởi tạo từ `prior_anchors.json` phase `pre` . Sau khi có normal baseline, cần thêm:

```python
parser.add_argument("--calibration-phase",
    choices=["pre", "normal_bull", "normal_bear"],
    default="pre",
    help="Which phase to use for Noise/MM agent calibration")
```

Đây chính là điểm nối cho **Condition A vs B vs C** trong ablation §5.3:
- **Condition A**: `--calibration-phase normal_bull` (unbiased prior)
- **Condition B**: `--calibration-phase pre` (biased 60s prior, current)
- **Condition C**: LLM-elicited prior (hiện tại)

#### F. `scripts/stage2_economics/PHASE2_RUNBOOK.md` — Update runbook

Thêm bước mới vào runbook giữa Step 04 và Step 05:

```markdown
## Step 04b — Normal Baseline Download (NEW)
Run: python scripts/stage2_economics/05b_download_normal_week.py
Output: data/processed/tardis/normal_baseline/
Purpose: Provides unbiased prior for Noise/MM agent calibration.
Required before: Step 11 (prior anchors), Step 18 (LOB runner --calibration-phase)
```

***

### Tóm Tắt Thứ Tự Làm

```
1. [Decision] Xác nhận 2 tuần từ bảng gợi ý trên (hoặc bạn chọn khác)
2. [Config]   Thêm NORMAL_WEEK_WINDOWS vào settings.py
3. [Script]   Tạo 05b_download_normal_week.py và chạy
4. [Script]   Update 06 với --mode normal và chạy trên normal data
5. [Script]   Update 11 để merge normal_baseline vào prior_anchors.json
6. [Script]   Update 18 với --calibration-phase flag
7. [Optional] Viết event distribution report (nếu cần cho paper)
```

# Task: Detailed Plan
Tôi đã có đủ toàn bộ context. Đây là phân tích đầy đủ và actionable plan.

***

## Phân Tích Toàn Diện: Normal Week Data cho COMOSA Phase 1

***

### Phần 1 — Report về 66 Events: Khoảng Thời Gian Nào?

Từ `config/settings.py`, dataset chạy từ **`2020-06-01` đến `2024-12-31`** (4.5 năm) , được lọc với ngưỡng drop ≥ 3.0% trong 5 phút, separation ≥ 4 giờ . Đây là bức tranh tổng thể:

```
Khoảng nguồn data:  2020-06-01 → 2024-12-31  (~4.5 năm, BTCUSDT Futures)
Event detection:    drop ≥ 3% trong 5 min, recovery ≥ 50% trong 30 min
66 events ≈ trung bình ~15 events/năm → ~1.2 events/tháng
```

Dựa trên lịch sử BTC đã biết, 66 events đó **phân bố không đều** theo 5 cụm chính:

| Cụm thời gian | Regime thị trường | Ước tính số events |
|---|---|---|
| 2020-06 → 2020-12 | COVID recovery, bull start | ~8–10 |
| 2021-01 → 2021-05 | Bull peak (ATH $65k), May crash | ~15–18 |
| 2021-06 → 2021-12 | Recovery, ATH $69k (Nov), correction | ~10–12 |
| 2022-01 → 2022-12 | Bear market (LUNA, FTX collapse) | ~18–20 |
| 2023-01 → 2024-12 | Recovery + ETF approval (Jan 2024) | ~8–10 |

**Bạn có cần tôi viết report không?** Câu trả lời là **có, nên có**, vì reviewer sẽ hỏi ngay. Report tối thiểu cần: histogram phân bố events theo năm/tháng, breakdown flash vs trend crash, distribution of drop_pct và duration_min. Tôi có thể generate script cho report này (xem Phần 3).

***

### Phần 2 — Nên Tải Normal Week Ở Khoảng Thời Gian Nào?

Tiêu chí khoa học để chọn tuần "normal":

**Phải thoả mãn đồng thời:**
1. **Không có event nào trong 66 events** trong khoảng ±3 ngày
2. **Low realized volatility**: BTC 7-day realized vol dưới median của toàn bộ sample (2020–2024)
3. **Cùng time range với event data** (2020–2024) để distributional shift không quá lớn
4. **Tránh các market microstructure anomaly đã biết**: exchange maintenance, API outage, halving weeks

**Các tuần cụ thể nên tải — theo thứ tự ưu tiên:**

| Tuần | Regime | Lý do chọn |
|---|---|---|
| `2021-10-04 → 2021-10-08` | Mid-bull, stable | Sau correction tháng 9, trước ATH push; low vol, high liquidity |
| `2023-02-06 → 2023-02-10` | Post-bear recovery | Thị trường đã stabilize sau FTX; vol thấp, order flow organic |
| `2024-03-11 → 2024-03-15` | Post-ETF settling | ETF launch dust settled (Jan), trước halving (April); liquidity cao |
| `2022-08-08 → 2022-08-12` | Bear mid-consolidation | Giữa LUNA crash (May) và FTX (Nov); tương đối flat |

**Khuyến nghị chọn 2 tuần** thay vì 1, từ hai regime khác nhau (bull + bear), để prior_anchors.json có thể phân biệt `normal_bull` vs `normal_bear`. Điều này cho phép Condition A trong ablation study §5.3 có baseline realistic hơn "uniform".

***

### Phần 3 — Scripts Cần Update Để Normal Week Merge Hoàn Hảo Vào Workflow

Đây là toàn bộ delta cần làm, theo thứ tự pipeline:

#### A. `config/settings.py` — Thêm constants mới

```python
# ──────────────────── Normal-Market Baseline ──────────────────────────
# Weeks selected to represent unbiased market microstructure for
# Noise Agent + MM Agent calibration (Phase 1 prior anchors §5.3)
NORMAL_WEEK_WINDOWS = [
    {"label": "normal_bull", "start": "2021-10-04", "end": "2021-10-08"},
    {"label": "normal_bear", "start": "2023-02-06", "end": "2023-02-10"},
]
NORMAL_WEEK_DIR   = PROCESSED_DIR / "tardis" / "normal_baseline"
NORMAL_WEEK_STATS = NORMAL_WEEK_DIR / "baseline_prior_stats.json"
```

#### B. `scripts/stage2_economics/05b_download_normal_week.py` — Script mới

Script này đã có draft từ trả lời trước. Cần thêm: loop qua `NORMAL_WEEK_WINDOWS`, save theo `label`, output `baseline_prior_stats.json` thay vì CSV.

#### C. `scripts/stage2_economics/06_micro_feature_engineering.py` — Thêm `--mode` flag

Script 06 hiện chỉ chạy trên `EVENT_RAW_DIR` . Cần thêm:

```python
parser.add_argument("--mode", choices=["event", "normal"], default="event",
    help="'event' = existing crash-window pipeline; "
         "'normal' = process normal_week baseline data")
parser.add_argument("--normal-dir", default=str(NORMAL_WEEK_DIR))
```

Khi `--mode normal`, script đọc từ `NORMAL_WEEK_DIR` thay vì `EVENT_RAW_DIR`, và skip phase-labeling (không có pre/drop/recovery/post — toàn bộ là `normal`).

#### D. `scripts/stage2_economics/11_compute_prior_anchors.py` — Thêm `normal` phase

Script 11 hiện chỉ tính anchors cho `PHASES = ["pre", "drop", "recovery", "post"]` . Cần thêm:

```python
# Thêm vào đầu file:
NORMAL_STATS_PATH = PROCESSED_DIR / "tardis" / "normal_baseline" / "baseline_prior_stats.json"

# Trong main(), sau khi load event data:
if NORMAL_STATS_PATH.exists():
    with open(NORMAL_STATS_PATH) as f:
        normal_stats = json.load(f)
    # Merge vào anchors với key "normal_bull" và "normal_bear"
    anchors["normal_baseline"] = normal_stats
    anchors["metadata"]["has_normal_baseline"] = True
    print(f"  [OK] Normal baseline loaded: {NORMAL_STATS_PATH}")
else:
    anchors["metadata"]["has_normal_baseline"] = False
    print(f"  [WARN] No normal baseline found. Run 05b first.")
```

#### E. `scripts/stage2_economics/18_lob_mini_runner.py` — Dùng normal baseline trong init

Hiện tại Noise Agent và MM Agent khởi tạo từ `prior_anchors.json` phase `pre` . Sau khi có normal baseline, cần thêm:

```python
parser.add_argument("--calibration-phase",
    choices=["pre", "normal_bull", "normal_bear"],
    default="pre",
    help="Which phase to use for Noise/MM agent calibration")
```

Đây chính là điểm nối cho **Condition A vs B vs C** trong ablation §5.3:
- **Condition A**: `--calibration-phase normal_bull` (unbiased prior)
- **Condition B**: `--calibration-phase pre` (biased 60s prior, current)
- **Condition C**: LLM-elicited prior (hiện tại)

#### F. `scripts/stage2_economics/PHASE2_RUNBOOK.md` — Update runbook

Thêm bước mới vào runbook giữa Step 04 và Step 05:

```markdown
## Step 04b — Normal Baseline Download (NEW)
Run: python scripts/stage2_economics/05b_download_normal_week.py
Output: data/processed/tardis/normal_baseline/
Purpose: Provides unbiased prior for Noise/MM agent calibration.
Required before: Step 11 (prior anchors), Step 18 (LOB runner --calibration-phase)
```

***

### Tóm Tắt Thứ Tự Làm

```
1. [Decision] Xác nhận 2 tuần từ bảng gợi ý trên (hoặc bạn chọn khác)
2. [Config]   Thêm NORMAL_WEEK_WINDOWS vào settings.py
3. [Script]   Tạo 05b_download_normal_week.py và chạy
4. [Script]   Update 06 với --mode normal và chạy trên normal data
5. [Script]   Update 11 để merge normal_baseline vào prior_anchors.json
6. [Script]   Update 18 với --calibration-phase flag
7. [Optional] Viết event distribution report (nếu cần cho paper)
```

Tôi đã đọc đủ. Để trả lời câu hỏi "implement được đúng không?" — **có, implement được hoàn toàn**, và tôi có thể phân tích chính xác từng điểm.

***

## Đánh Giá Feasibility: Implement Được Đúng Không?

**Câu trả lời ngắn: Có, không có blocker kỹ thuật nào.** Dưới đây là phân tích từng điểm dựa trên code thực tế.

***

### Vấn Đề BBO — Không Phải Blocker Vì Đây Là Pattern Đã Có

Script 06 đã **thiết kế sẵn fallback graceful** khi không có bookTicker . Tại `process_event()`, `bbo_bars = None` khi `bkt_path` không tồn tại, và `merge_and_enrich()` nhận `bbo_bars=None` thì chỉ skip phần merge BBO — các cột `kyle_lambda`, `vpin`, `realized_vol_*`, `amihud_illiq` **vẫn được tính từ aggTrades thuần túy** . Spread và depth sẽ là `NaN`, và script 11 đã xử lý `NaN` qua `dropna()` .

Điều này có nghĩa là: **normal baseline sẽ có đủ OFI, trade_intensity, kyle_lambda, và Pareto α** — đúng 4 anchors quan trọng nhất mà script 11 dùng cho `noise_trader_lambda` và `order_size_pareto_alpha`. Spread/depth từ proxy (Roll's implicit spread qua `roll_spread_10`, `roll_spread_50`) **đã được tính sẵn trong merge_and_enrich**  và sẽ dùng được thay cho BBO trực tiếp.

***

### Phân Tích Từng Delta — Độ Phức Tạp Thực Tế

| Delta | Loại thay đổi | Rủi ro | Verdict |
|---|---|---|---|
| `settings.py` + constants | Additive, không sửa existing | Không có | ✅ Dễ |
| `05b_download_normal_week.py` | File mới hoàn toàn, copy helpers từ 05 | Thấp | ✅ Dễ |
| `06` thêm `--mode normal` | Thêm CLI arg + swap input_dir, KHÔNG sửa core logic | Thấp | ✅ Dễ |
| `11` merge `normal_baseline` | Additive section vào JSON output | Thấp | ✅ Dễ |
| `18` thêm `--calibration-phase` | Đây là điểm phức tạp nhất — cần đọc `safe_phase_metric` | Trung bình | ⚠️ Cần cẩn thận |

**Điểm duy nhất cần cẩn thận là script 18.** Agent của bạn đúng khi nói engine đang dùng `safe_phase_metric` tại line 490 để lấy anchor **per-tick theo phase mô phỏng** . Nghĩa là nếu tick đang ở phase `drop`, nó tự động dùng anchor của `drop`. Logic này phải được giữ nguyên cho Condition B và C. Chỉ Condition A (normal_bull/normal_bear) mới cần override: thay vì lấy anchor theo phase mô phỏng, nó lấy anchor từ `normal_baseline` section trong `prior_anchors.json` bất kể tick đang ở phase nào.

Patch đúng cho script 18 là:

```python
# Thêm vào argparse:
parser.add_argument(
    "--calibration-source",
    choices=["phase_dynamic", "normal_bull", "normal_bear"],
    default="phase_dynamic",
    help="phase_dynamic = current behavior (per-tick phase anchor); "
         "normal_bull/normal_bear = override with normal baseline anchors"
)

# Thêm vào safe_phase_metric (hoặc bọc ngoài):
def get_anchor(anchors, metric, phase, calibration_source):
    if calibration_source == "phase_dynamic":
        # Behavior hiện tại — không đổi gì
        return anchors.get(metric, {}).get(phase)
    else:
        # Override: lấy từ normal_baseline
        return anchors.get("normal_baseline", {}).get(
            calibration_source, {}
        ).get(metric)
```

Đây là **2 dòng logic** thêm vào một điểm duy nhất — không cần refactor toàn bộ engine.

***

### Một Vấn Đề Schema Cần Chú Ý

Agent của bạn phát hiện đúng: `Event_Dynamics_gridded.csv` thiếu `spread_bps` và `depth_imbalance` . Nhưng điều này **không ảnh hưởng đến normal baseline** vì:

1. Script 11 dùng `if "spread_bps" in df.columns` trước khi tính  — nếu cột không có thì key đó trong JSON sẽ empty, không crash.
2. Với normal_baseline, Roll's implicit spread (`roll_spread_10`) là proxy đủ tốt cho normal regime — đây là estimator standard trong Chordia et al. (2001) và Roll (1984) cho khi không có BBO.
3. Không cần rerun pipeline cũ để fix schema — normal_baseline là section riêng trong JSON.

***

### Kết Luận: 5 Delta, Tất Cả Implement Được

Không có dependency hidden nào giữa các delta. Thứ tự an toàn nhất để implement là:

```
settings.py  →  05b (file mới)  →  06 (thêm --mode)
     ↓
  chạy 05b + 06 --mode normal
     ↓
    11 (merge JSON)  →  18 (thêm --calibration-source)
     ↓
  runbook update + end-to-end sanity check
```
# Task: 4 Agent Prompts

Đã đủ thông tin. Đây là phân tích đầy đủ.

***

## Agent Prompts & Detail Markdown: Có Cần Thay Đổi Không?

**Câu trả lời ngắn: Không cần thay đổi gì ở agent prompt markdown hiện tại.** Nhưng cần thêm mới — không sửa cũ.

***

## Tại Sao Prompt Cũ Không Cần Đụng Vào

Script 13 load agent documents qua `load_agent_documents(PROMPT_DETAILS_DIR)` rồi dùng `agent_doc['prompt_text']` làm system prompt . Toàn bộ logic xây dựng prompt được **tách biệt hoàn toàn** khỏi phase data — phase chỉ xuất hiện trong `user_prompt` qua `build_user_prompt()`, còn `system_prompt` = prompt markdown + `DEFAULT_RESPONSE_RULES` cố định .

Điều này có nghĩa là: 4 agent markdown hiện tại (momentum, contrarian, hft_market_maker, noise_trader) được thiết kế để describe **vai trò và hành vi**, không embed assumption về phase nào. Chúng sẽ hoạt động đúng với normal_baseline mà không cần sửa một dòng nào.

***

## Điều Duy Nhất Cần Làm Với Prompt System

Có **2 nơi cần thêm mới**, không sửa cũ:

### 1. Thêm `normal_bull` và `normal_bear` vào `PHASES` trong `common.py`

Hiện tại `PHASES` trong `common.py` là `["pre", "drop", "recovery", "post"]` . Script 13 dùng `PHASES` để:
- Validate rằng anchor file có đủ phases  — `load_anchors()` sẽ throw nếu thiếu
- Loop qua từng phase để generate prompts 

Khi bạn thêm `normal_bull`/`normal_bear` vào `prior_anchors.json` (từ script 11 nâng cấp), thì `PHASES` cũng cần được extend. Cách làm an toàn nhất là **không sửa constant gốc** mà thêm một constant riêng:

```python
# common.py — KHÔNG sửa PHASES gốc, thêm mới
PHASES = ["pre", "drop", "recovery", "post"]  # giữ nguyên
NORMAL_PHASES = ["normal_bull", "normal_bear"]  # thêm mới
ALL_PHASES = PHASES + NORMAL_PHASES  # dùng khi cần full list
```

Lý do giữ `PHASES` gốc: script 11, 15, 16, 17 đều import `PHASES` từ `common.py`  — đổi constant gốc sẽ phá vỡ validation của các script đó với event data cũ.

### 2. Thêm anchor snapshot mapping cho normal phases trong `phase_anchor_snapshot()`

Hàm `phase_anchor_snapshot()` trong script 13 đang hardcode lấy từ `anchors["ofi_percentiles_per_phase"][phase]` . Khi `phase = "normal_bull"`, nó sẽ tự động work nếu `prior_anchors.json` đã có key `normal_bull` trong mỗi `*_per_phase` dict — không cần sửa hàm này.

Điều kiện duy nhất: script 11 khi write `normal_baseline` vào JSON phải **dùng cùng key structure** như các phase hiện tại, không phải nested dưới `normal_baseline.normal_bull` mà phải flat như `ofi_percentiles_per_phase.normal_bull`. Đây là điểm cần xác nhận khi implement script 11.

***

## Tóm Tắt Thay Đổi Cần Làm

| File | Thay đổi | Mức độ |
|---|---|---|
| Agent prompt markdown (4 file) | **Không cần đụng** | — |
| `common.py` | Thêm `NORMAL_PHASES`, `ALL_PHASES` — giữ nguyên `PHASES` gốc | Nhỏ, additive |
| `13_write_prompts.py` | Thêm `--include-normal` flag để loop `NORMAL_PHASES` khi cần | Nhỏ, optional |
| `prior_anchors.json` schema | Script 11 cần write flat key structure cho normal phases | Quan trọng, xác nhận khi impl 11 |

Không có gì trong prompt markdown cần sửa vì chúng được thiết kế đúng theo role-based principle — behavior logic nằm trong user prompt (data-driven), không trong system prompt (role-driven) .

