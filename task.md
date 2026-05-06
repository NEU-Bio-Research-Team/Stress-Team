Tình huống này rất rõ ràng. Vấn đề không còn là code nữa — **bottleneck là data quality**, cụ thể là số crash runs không bị floor-dominated quá thấp (chỉ 9 runs).

## Diagnosis chính xác

Sau tất cả các fix, pipeline đang hoạt động đúng. Nhưng sau khi apply `floor_policy=censor_rows` + crash-window slice, chỉ còn **620 rows từ 9 crash runs** — đây là con số quá nhỏ cho NOTEARS/LiNGAM hoạt động tin cậy. Threshold thực tế để causal discovery cho kết quả ổn định là ~5,000–10,000 rows với 6 variables. Bạn đang thiếu ~8–15x data lượng cần thiết.

VAR-LiNGAM recover được 2 edges (`mean_wealth → ofi`, `wealth_concentration → leverage_proxy`) không phải ngẫu nhiên — đây là signal thật. Nhưng precision 0.08 nói rằng cứ 10 edges được claim thì chỉ 1 đúng, không đủ để đưa vào paper claim.

## Hai hướng đi và trade-off

### Hướng A — Tạo thêm crash runs trước floor saturation

Đây là hướng **đúng về mặt khoa học** nhưng tốn compute. Cần tìm parameter regime cho crash xảy ra sớm hơn trong simulation (trước khi floor binding), tức là crash phải đến từ LOB dynamics chứ không phải từ price hitting hard floor.

Cụ thể, cần thử giảm `min_price_fraction` xuống 0.30–0.35 (nới floor) hoặc tăng shock intensity nhẹ để crash window xuất hiện ở tick 200–400 thay vì sau khi floor đã active. Mục tiêu là có **50+ floor-clean crash runs** → ~3,500+ rows trong crash window → NOTEARS/LiNGAM có đủ signal.

Trade-off: tốn thêm compute và thời gian calibrate config mới, và config mới sẽ lại có vấn đề về empirical anchor như đã thảo luận trước.

### Hướng B — Relaxed floor policy, accept limitation trong paper

Thay vì `censor_rows` (loại hoàn toàn floor-touched rows), chuyển sang `flag_rows` — giữ lại tất cả crash runs nhưng thêm `floor_touched` như một **control variable** trong causal model. Điều này cho phép NOTEARS/LiNGAM học trên toàn bộ 88 crash runs, và bạn report kết quả với note rõ: *"Results include floor-regime runs; floor_touched included as observed confounder."*

Trade-off: causal claim yếu hơn về mặt mechanism, nhưng **có thể frame như exploratory causal analysis** thay vì confirmatory — và nhiều ABM papers làm đúng vậy. [arxiv](https://arxiv.org/html/2410.19412)

## Recommendation thực tế

Với timeline và resource hiện tại, **Hướng B là pragmatic hơn** cho paper. Lý do:

Causal layer **không phải** contribution chính của paper bạn — contribution chính vẫn là C1 (LLM prior quality) và C2 (fidelity comparison 3 scenarios). Causal discovery là supporting evidence. Một causal analysis được frame đúng với known limitations sẽ được reviewer chấp nhận tốt hơn một causal analysis claim mạnh nhưng dựa trên 9 runs.

Framing cụ thể có thể dùng trong paper:

> *"We present causal discovery as an exploratory analysis on the synthetic panel. Given the binding floor constraint in the canonical config, we include floor_touched as a control and report results as preliminary evidence for the hypothesized causal chain. Stronger causal identification is left to future work with floor-free configurations."*

Đây không phải là admit defeat — đây là scientific honesty mà reviewers tôn trọng, đặc biệt ở computational finance venues.

## Bước tiếp theo cụ thể

Ngay bây giờ nên làm theo thứ tự:

1. **Khóa main comparison table cho 3 priors** — phần Uniform và Literature 500-run panels đã hoàn tất. So sánh 500-run hiện tại cho thấy LLM giữ được kurtosis cao hơn Uniform (29.08 vs 26.49), trong khi Literature cho đuôi quá nặng (77.20) và crash rate gần LLM hơn (0.152 vs 0.176; Uniform chỉ 0.008).
2. **Dùng kết quả này làm baseline paper-facing cho Task 2** — tức là coi so sánh stylised facts/crash behavior giữa LLM, Uniform, Literature là deliverable chính đã sẵn sàng trước khi quay lại causal layer.
3. **Nếu tiếp tục causal**, rerun `20_causal_discovery.py` với `floor_policy=flag_rows` thay vì `censor_rows`, thêm `floor_touched` vào variable set, và report rõ đây là exploratory result có caveat.
4. **Giữ intervention results hiện tại như directional diagnostic** — do(leverage=0) → ~100% reduction là result có thể frame được nếu bạn note rõ đây là upper-bound estimate trên floor-constrained panel.

Script cần thay đổi nhỏ nhất để test Hướng B:

```python
# Trong 20_causal_discovery.py
# Thay floor_policy = 'censor_rows'
floor_policy = 'flag_rows'  # giữ lại, thêm floor_touched như biến

# Thêm floor_touched vào variables
variables = ['ofi', 'spread_bps', 'depth_imbalance', 
             'leverage_proxy', 'vpin', 'floor_touched', 'flash_crash']
```

Với 88 crash runs × 70 ticks = ~6,160 rows, đây đã đủ ngưỡng cho NOTEARS/LiNGAM cho kết quả interpretable.