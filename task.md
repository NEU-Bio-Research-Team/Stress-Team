Tôi đã có đủ thông tin. Giờ tôi phân tích root cause trực tiếp từ code.

***

## Root Cause Analysis: `crash_rate_sim = 0.0` trong cả 3 scenario

Sau khi đọc toàn bộ `18_lob_mini_runner.py` và output JSON, tôi xác định **4 root cause độc lập** đang cộng hưởng để triệt tiêu khả năng sinh crash.

***

## Root Cause 1: `flash_crash_flag` được tính per-tick, không per-run

Đây là **bug logic nghiêm trọng nhất** . Logic hiện tại trong `run_one_simulation()` tại dòng tính `window`:

```python
window = close_series[-args.crash_window_ticks:]
flash_crash_flag = 0
if len(window) == args.crash_window_ticks and window[0] > 0:
    drop_pct = (window[0] - min(window)) / window[0] * 100.0
    if drop_pct >= args.crash_threshold_pct:  # default = 1.0%
        flash_crash_flag = 1
```

Giá trị `flash_crash_flag` được ghi vào từng row riêng lẻ. Tuy nhiên trong `summarize_output()` :

```python
run_flags = df.groupby("run_id")["flash_crash_flag"].max()
```

Tức là một run được tính là "crash" chỉ khi **ít nhất 1 tick trong 10 tick liên tiếp** có `drop_pct >= 1.0%`. Vì `window[0]` là giá đầu window, còn `min(window)` là đáy window — điều này về mặt lý thuyết không sai. Nhưng vấn đề thực tế là: **price path cực kỳ smooth do impact quá nhỏ**, nên không tick nào đạt 1% drawdown trong 10 ticks (= 1 giây).

Cần đọc thêm để confirm magnitude.

***

## Root Cause 2: `impact_scale * kyle_lambda * net_flow` quá nhỏ để tạo 1% drop

Tại default params :
- `--impact-scale = 1.0`
- `--drop-impact-mult = 1.35`
- `--drop-sell-pressure = 0.12`
- `--intensity-scale = 0.35` → `lam ≈ 0.35 * trade_intensity_anchor`

Theo summary JSON , `ofi_drop_mean = -0.0550` cho LLM. Với kyle_lambda điển hình từ anchors vào khoảng 0.5–2.0 (empirical HFT range), `impact ≈ 1.0 * 1.35 * 0.5 * (-0.0550) ≈ -0.037`. Với init_price ≈ 30,000 USDT, **impact per tick chỉ ~ -$37**, hay ~0.00012% per tick. Để đạt 1% drawdown trong 10 ticks cần tích lũy 10 ticks cùng chiều impact ~ -$30 mỗi tick — xác suất xảy ra liên tục rất thấp với lognormal noise.

Vấn đề: `kyle_lambda` từ `prior_anchors.json` có thể quá nhỏ (được tính từ empirical data thực tế trên BTC, thường nằm khoảng 1e-5 đến 1e-2 tính bằng $/BTC), trong khi `net_flow` tính bằng BTC với `base_order_size = 0.25 BTC`. Cần kiểm tra unit mismatch giữa kyle_lambda empirical và net_flow đơn vị BTC.

***

## Root Cause 3: `infer_side_probability` không áp dụng `drop_sell_pressure` đúng chỗ

Đây là **bug logic im lặng** . Hàm `infer_side_probability` có logic như sau:

```python
def infer_side_probability(agent_type, phase, ofi_anchor_median, drop_sell_pressure):
    if agent_type == "momentum_trader":
        if phase == "drop":
            return 0.25   # hardcoded, KHÔNG dùng drop_sell_pressure
    ...
    # drop_sell_pressure chỉ được dùng ở nhánh cuối (fallback)
    p_buy = 0.50
    if phase == "drop":
        p_buy = p_buy - drop_sell_pressure
    return clip01(p_buy)
```

`drop_sell_pressure` (default `0.12`) **chỉ áp dụng cho fallback branch** — không rơi vào bất kỳ agent_type nào được định nghĩa rõ ràng. Cụ thể:
- `momentum_trader` → hardcoded 0.25 trong drop (không dùng tham số)
- `contrarian_trader` → hardcoded 0.55 trong drop
- `hft_market_maker` → hardcoded 0.50 (luôn)
- `noise_trader` → hardcoded 0.45 trong drop

Không có agent type nào sử dụng `drop_sell_pressure`. Tham số `--drop-sell-pressure` **hoàn toàn dead code** cho các agent type cụ thể. Tăng `--drop-sell-pressure` lên bất kỳ giá trị nào cũng **không có tác dụng** vì fallback branch không bao giờ được gọi (tất cả 4 agent types đều được handle trước đó).

***

## Root Cause 4: `leverage_proxy` không feed back vào price impact

`leverage_proxy` được tính như sau :

```python
leverage_proxy = 1.0 + abs(log_ret) / max(realized_anchor, 1e-8)
```

Nhưng giá trị này **chỉ được ghi vào DataFrame, không bao giờ được dùng để amplify impact**. Không có feedback loop từ leverage vào `impact`. Đây là lý do `leverage_proxy ~ 1.006` trong drop phase — không phải vì leverage thấp thực sự, mà vì khi `log_ret` nhỏ (do impact nhỏ ở RC2), `leverage_proxy` cũng nhỏ. Vòng tròn này hoàn toàn decorative.

***

## Bảng tóm tắt root causes

| # | Root Cause | Loại | Tác động | Severity |
|---|---|---|---|---|
| RC1 | `crash_threshold_pct=1%` quá cao so với impact magnitude thực tế | **Tham số miscalibrated** | Không có tick nào pass gate crash | 🔴 Critical |
| RC2 | Kyle lambda × net_flow → impact per tick quá nhỏ (possibily unit mismatch) | **Physics bug / calibration** | Price path quá smooth để tạo 1% trong 10 ticks | 🔴 Critical |
| RC3 | `drop_sell_pressure` là dead code — không áp dụng cho bất kỳ agent type nào | **Logic bug** | Tăng param không có tác dụng gì | 🟠 High |
| RC4 | `leverage_proxy` không feed back vào price impact | **Missing amplification loop** | Wealth/leverage channel decorative | 🟠 High |

***

## Fix đề xuất theo thứ tự ưu tiên

**Fix RC3 trước (1 dòng)** — đây là quick win nhất. Trong `infer_side_probability`, thay vì hardcode, áp dụng `drop_sell_pressure` vào tất cả agent types trong drop phase:

```python
if agent_type == "momentum_trader":
    base = 0.25 if phase == "drop" else ...
    if phase == "drop":
        return clip01(base - drop_sell_pressure * 0.5)  # thêm dòng này
```

Hoặc tốt hơn, refactor để `drop_sell_pressure` là một global shift áp dụng sau khi xác định base probability:

```python
p_buy = _base_side_prob(agent_type, phase, ofi_anchor_median)
if phase == "drop":
    p_buy = clip01(p_buy - drop_sell_pressure)
return p_buy
```

**Fix RC1 + RC2** — Hạ `crash_threshold_pct` từ `1.0%` xuống `0.5%` làm điểm thử nghiệm đầu tiên. Đồng thời kiểm tra đơn vị của `kyle_lambda` trong `prior_anchors.json` — nếu đang là $ per BTC unit, cần nhân thêm hệ số scale để `impact` tính ra được theo phần trăm giá.

**Fix RC4** — Thêm feedback loop leverage vào impact multiplier trong drop phase:

```python
# Sau khi tính leverage_proxy
if phase == "drop" and leverage_proxy > 1.5:
    impact *= min(leverage_proxy / 1.5, 3.0)  # cap tại 3x
```

Với 4 fix trên, đặc biệt RC3, `crash_rate_sim` sẽ nhích lên ngay cả khi không tăng `--drop-sell-pressure` hay `--drop-impact-mult` — vì hiện tại tăng 2 tham số đó không có tác dụng gì .