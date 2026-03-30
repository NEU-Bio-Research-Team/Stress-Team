# Stage 2 Event Pipeline Documentation

## Muc tieu tai lieu
Tai lieu nay mo ta day du luong xu ly ban da dung de di tu event phat hien tren moc phut, refine ve tick-level, tao feature 100ms, va sinh bo ket qua trong thu muc confounder outputs.

Pham vi:
- Tu event catalog (Tier 1) den event tick windows (Tier 2)
- Tu aggTrades/bookTicker den micro features 100ms
- Tu event refined timestamps den bo output confounder

Luu y quan trong:
- Trong pipeline event-driven nay, ban khong reindex thanh luoi timestamp day du 100ms.
- He thong dang binning theo resolution_ms=100 de tao cac bar 100ms.

---

## 1) Tong quan luong xu ly

Thu tu chay thuc te:
1. Script 04: Detect event catalog
2. Script 05: Download tick data theo event
3. Script 06: Tao micro features (chon resolution 100ms)
4. Script 08: Refine timestamp tu kline-level ve tick-level
5. Script 09: Tao confounder outputs

Input/Output chinh:
- Input ban dau: du lieu giao dich theo ngay va event catalog
- Trung gian:
  - data/raw/tardis/events/event_xxx_yyyy-mm-dd/aggtrades.parquet
  - data/raw/tardis/events/event_xxx_yyyy-mm-dd/bookticker.parquet (neu co)
  - data/processed/tardis/micro/res_100ms/*.parquet
  - data/processed/tardis/event_catalog_tick_refined.csv
- Output cuoi:
  - data/processed/tardis/confounder_outputs/Flash_Crash_Events_Labeled.csv
  - data/processed/tardis/confounder_outputs/Empirical_Benchmarks.json
  - data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv
  - data/processed/tardis/confounder_outputs/News_Impact_Decomposition.png

---

## 2) Cau hinh trung tam (settings)

Cac bien lien quan nam trong config/settings.py:
- EVENT_CATALOG_PATH: data/processed/tardis/event_catalog.csv
- EVENT_RAW_DIR: data/raw/tardis/events
- EVENT_MICRO_DIR: data/processed/tardis/micro
- MICRO_RESOLUTIONS_MS: [10, 100, 1000]
- FLASH_CRASH_*: bo tham so detect event

Y nghia:
- 100ms la mot muc resolution trong bo MICRO_RESOLUTIONS_MS.
- Thu muc confounder output khong nam trong settings, duoc dat truc tiep trong script 09.

---

## 3) Chi tiet tung buoc

### Buoc 1 - Tao event catalog (Script 04)
File: scripts/stage2_economics/04_detect_flash_crashes.py

Tac vu:
- Quet chuoi gia theo ngay.
- Tim cac dot giam manh theo nguong drop_pct trong cua so window phut.
- Danh dau co phuc hoi hay khong de phan loai flash crash va trend crash.
- Deduplicate event theo khoang cach thoi gian.

Output:
- data/processed/tardis/event_catalog.csv

Ghi chu:
- Day la moc thoi gian cap phut/chuoi detect, chua phai moc tick refine.

### Buoc 2 - Tai tick data cho tung event (Script 05)
File: scripts/stage2_economics/05_download_event_ticks.py

Tac vu:
- Doc event_catalog.csv.
- Tai aggTrades theo ngay event.
- Thu tai them bookTicker (co the khong co tren Binance Vision futures cho mot so ngay).
- Cat theo cua so event: [-before_min, +after_min] quanh crash_start_time.

Output theo event:
- data/raw/tardis/events/event_xxx_yyyy-mm-dd/aggtrades.parquet
- data/raw/tardis/events/event_xxx_yyyy-mm-dd/bookticker.parquet (neu co)
- data/raw/tardis/events/event_xxx_yyyy-mm-dd/event_meta.csv

### Buoc 3 - Tao micro features 100ms (Script 06)
File: scripts/stage2_economics/06_micro_feature_engineering.py

Tac vu:
- Doc aggTrades (va bookTicker neu co) trong tung event folder.
- Chia bin theo cong thuc:
  - bin_id = floor((timestamp_ms - t0) / resolution_ms)
- Aggregate theo bin de tao cac cot open/high/low/close, volume, trade_count, OFI, VWAP, return...
- Merge voi BBO neu co va tinh them kyle_lambda, vpin, amihud, realized_vol, spread metrics.

Output:
- data/processed/tardis/micro/res_100ms/event_xxx_yyyy-mm-dd_100ms.parquet
- summary: data/processed/tardis/micro/micro_features_summary.csv

Luu y ky thuat:
- Day la time-binned bars 100ms, khong phai reindex full-grid 100ms co du moi moc.

### Buoc 4 - Refine timestamp event xuong tick-level (Script 08)
File: scripts/stage2_economics/08_refine_event_timestamps.py

Tac vu:
- Doc data/processed/tardis/event_catalog.csv.
- Voi moi event, doc aggtrades.parquet da tai o Buoc 2.
- Tim trong cua so tim kiem:
  - tick_bottom_time: timestamp trade co gia thap nhat
  - tick_start_time: local max truoc bottom
- Tinh lai tick_drop_pct, tick_duration_ms.

Output:
- data/processed/tardis/event_catalog_tick_refined.csv

Y nghia:
- Day la bang moc thoi gian refined de script 09 dung de tinh timeline va quan he voi news.

### Buoc 5 - Tao confounder outputs (Script 09)
File: scripts/stage2_economics/09_produce_confounder_outputs.py

Input chinh:
- Catalog refined: data/processed/tardis/event_catalog_tick_refined.csv
- Micro 100ms: data/processed/tardis/micro/res_100ms/*.parquet
- Co fallback 1000ms neu 100ms khong ton tai cho event nao do.

Tac vu:
- Gan nhan event theo news knowledge base va timing catalog (neu co).
- Trich xuat benchmark microstructure trong cua so crash.
- Tao event dynamics theo phase pre/drop/recovery.
- Ve hinh decomposition.

Output:
- data/processed/tardis/confounder_outputs/Flash_Crash_Events_Labeled.csv
- data/processed/tardis/confounder_outputs/Empirical_Benchmarks.json
- data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv
- data/processed/tardis/confounder_outputs/News_Impact_Decomposition.png

---

## 4) Lenh chay tham khao

Chay tu root project Stress-Team:

```powershell
# 1) Detect events
python scripts/stage2_economics/04_detect_flash_crashes.py

# 2) Download event ticks
python scripts/stage2_economics/05_download_event_ticks.py

# 3) Build micro features only 100ms
python scripts/stage2_economics/06_micro_feature_engineering.py --resolution 100

# 4) Refine event timestamps to tick level
python scripts/stage2_economics/08_refine_event_timestamps.py

# 5) Produce confounder outputs
python scripts/stage2_economics/09_produce_confounder_outputs.py
```

Goi y:
- Neu bookTicker hay bi thieu tren futures data source, script 05 va 06 van chay duoc voi aggTrades.
- Script 09 uu tien 100ms, neu event nao thieu se fallback 1000ms.

---

## 5) Kiem tra nhanh sau khi chay

Checklist:
- Co file data/processed/tardis/event_catalog.csv
- Co file data/processed/tardis/event_catalog_tick_refined.csv
- Co nhieu file event_*_100ms.parquet trong data/processed/tardis/micro/res_100ms
- Co du 4 file trong data/processed/tardis/confounder_outputs

Neu can doi chieu su khac nhau giua hai nhanh:
- Nhanh 1min tong hop cu: src/preprocessing/tardis_preprocess.py
- Nhanh event-driven 100ms + confounder: scripts/stage2_economics/04 -> 05 -> 06 -> 08 -> 09

---

## 6) Nguon de doc code nhanh

- scripts/stage2_economics/04_detect_flash_crashes.py
- scripts/stage2_economics/05_download_event_ticks.py
- scripts/stage2_economics/06_micro_feature_engineering.py
- scripts/stage2_economics/08_refine_event_timestamps.py
- scripts/stage2_economics/09_produce_confounder_outputs.py
- config/settings.py
- src/preprocessing/tardis_preprocess.py

---

## 7) Threshold provenance: cai nao co co so, cai nao dang heuristic

Day la phan quan trong de tranh bi reviewer bat be ve viec chon nguong co dinh.

### 7.1 Nhom nguong event detection (Script 04)

Nguong dang dung:
- FLASH_CRASH_DROP_PCT = 3.0
- FLASH_CRASH_WINDOW_MIN = 5
- FLASH_CRASH_RECOVERY_MIN = 30
- FLASH_CRASH_RECOVERY_PCT = 50%
- FLASH_CRASH_MIN_SEPARATION_HR = 4

Danh gia:
- Co co so theo thong le event-study va microstructure (dinh nghia su kien bang do lon giam gia + toc do + muc do phuc hoi).
- Tuy nhien bo gia tri cu the (3%, 5 phut, 50%, 4 gio) trong code hien tai van la calibration theo project, chua phai tham so duoc uoc luong tu optimization chinh thuc.

Ket luan reviewer-facing:
- Khong duoc trinh bay nhu "chan ly co dinh".
- Nen trinh bay nhu "baseline specification" va bat buoc co sensitivity analysis.

### 7.2 Nhom nguong cat cua so du lieu event (Script 05)

Nguong dang dung:
- EVENT_WINDOW_BEFORE_MIN = 30
- EVENT_WINDOW_AFTER_MIN = 30

Danh gia:
- Co co so ky thuat: can du context truoc/sau su kien de tinh pre-drop va recovery features.
- Gia tri 30/30 la pragmatic default, khong phai gia tri duy nhat dung.

### 7.3 Nhom nguong tinh feature vi mo (Script 06)

Nguong dang dung:
- MICRO_KYLE_WINDOW = 50
- MICRO_VPIN_WINDOW = 50
- resolution mac dinh co 100ms

Danh gia:
- Co co so thong ke: can rolling window du lon de on dinh uoc luong.
- Nhung chon 50 bars hien tai van la heuristic threshold (trade-off bias-variance), can test do ben.

### 7.4 Nhom nguong refine timestamp (Script 08)

Nguong dang dung:
- lookback-seconds = 60

Danh gia:
- Co co so nghiep vu: tim local peak ngay truoc khi roi gia.
- Muc 60 giay la heuristic co huong nghiep vu, can test 30/60/120s.

### 7.5 Nhom nguong confounder/news (Script 09)

Nguong dang dung:
- sentiment <= -0.4 de xep D_news=1
- window_hours = +-12h de match news-event
- RECOVERY_TARGET = 50% retrace de danh dau event end

Danh gia:
- Co co so event-study: can cua so thoi gian de gan tac dong tin tuc.
- Nguong cu the (-0.4, 12h, 50%) hien tai van la policy choice trong project.
- Script 09 co noi ro la "curated knowledge base" va phuong phap nghien cuu su kien, nhung reviewer van se hoi ve robustness.

Tom tat ngan gon:
- Co so khoa hoc: CO (o muc framework/phuong phap)
- Gia tri nguong cu the trong code: PHAN LON la heuristic baseline can bao cao sensitivity

---

## 8) Ke hoach chong reviewer bat be (de dua vao paper/report)

Neu muon phan nay "an chac" khi review, nen bo sung 4 lop bao ve sau:

1. Pre-specify threshold grid
- Khai bao truoc grid va khong doi sau khi xem ket qua.
- Vi du:
  - drop_pct: [2.0, 2.5, 3.0, 4.0, 5.0]
  - window_min: [3, 5, 10]
  - recovery_pct: [30, 50, 70]
  - separation_hr: [2, 4, 6]
  - sentiment_cut: [-0.3, -0.4, -0.5]
  - news_window_h: [6, 12, 24]

2. Bao cao ket qua theo vung, khong theo 1 diem
- Bao cao median/IQR cua metric chinh qua toan bo grid.
- Bao cao ty le ket luan giu nguyen tren bao nhieu phan tram cau hinh.

3. Chon baseline bang objective criterion
- Vi du chon baseline co event count nam trong khoang hop ly va quality cao nhat:
  - du so event de co power
  - ti le missing du lieu thap
  - phan bo crash duration va drop magnitude khong bi meo qua muc

4. Negative controls / placebo
- Doi timestamp event ngau nhien trong ngay hoac trong tuan de tao placebo events.
- Ky vong benchmark that phai tach biet ro voi placebo, neu khong threshold dang overfit.

Mau cau an toan de viet trong paper:
- "Thresholds are used as baseline specifications motivated by market microstructure event-study conventions; all key findings are validated under pre-specified sensitivity grids and placebo controls."

---

## 9) Ket luan thang ve cau hoi "co tu cho so khong?"

Tra loi thang:
- Co, mot phan threshold trong code hien tai la so baseline do nhom chon (heuristic but reasonable), chua phai gia tri duy nhat co chan ly.
- Nhung ban co the bien no thanh nghiem tuc hoc thuat neu bo sung sensitivity + placebo + objective selection criterion nhu muc 8.

Neu can bao ve truoc reviewer, khong nen noi:
- "Chung toi chon 3% vi thay hop ly"

Nen noi:
- "Chung toi dinh nghia baseline 3%/5min theo thong le event-study, sau do xac nhan tinh ben ket qua tren luoi tham so da pre-specify."
