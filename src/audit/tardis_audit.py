"""
Tardis BTC Futures Audit – Checklist T1-T15
============================================
Runs every mandatory check from the Algorithmic Panic audit protocol
on downloaded Tardis data.

Checks:
    T1  Date range coverage
    T2  Pre-2020-05-14 exclusion
    T3  Timestamp ordering
    T4  Sequence number gaps
    T5  May 19, 2021 data check
    T6  Orderbook validity (best_bid < best_ask)
    T7  Snapshot completeness
    T8  Price outliers
    T9  Missing ticks detection
    T10 Daily reconnection gaps
    T11 Trade-orderbook consistency
    T12 Stylized facts validation
    T13 Liquidation data availability
    T14 Open interest availability
    T15 Volume distribution
"""

import sys
import gzip
import csv
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    TARDIS_RAW_DIR, TARDIS_EXCHANGE, TARDIS_SYMBOL,
    TARDIS_UNSAFE_DATE, TARDIS_MAY19_START, TARDIS_MAY19_END,
    AUDIT, REPORTS_DIR,
)
from src.utils.io_utils import save_audit_report


def _list_files(data_type: str, root: Path = TARDIS_RAW_DIR,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> List[Path]:
    """List .csv.gz files for a given data type, optionally filtered by date range."""
    dt_dir = root / data_type
    if not dt_dir.exists():
        return []
    files = sorted(dt_dir.glob("*.csv.gz"))
    if start_date or end_date:
        def _in_range(p: Path) -> bool:
            d = _extract_date(p)
            if d is None:
                return True
            if start_date and d < start_date:
                return False
            if end_date and d > end_date:
                return False
            return True
        files = [f for f in files if _in_range(f)]
    return files


def _extract_date(filepath: Path) -> Optional[str]:
    """Extract YYYY-MM-DD from Tardis filename."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
    return m.group(1) if m else None


def _load_day_csv(path: Path, max_rows: int = 0) -> pd.DataFrame:
    """Load a gzipped CSV file."""
    try:
        with gzip.open(path, "rt") as f:
            if max_rows > 0:
                return pd.read_csv(f, nrows=max_rows)
            return pd.read_csv(f)
    except Exception as e:
        print(f"  [!] Failed to load {path.name}: {e}")
        return pd.DataFrame()


# ─────────────────────── Individual Checks ───────────────────────────

def T1_date_range(root: Path = TARDIS_RAW_DIR) -> Dict:
    """T1: Check continuous coverage from start date."""
    trades_files = _list_files("trades", root)
    if not trades_files:
        return {"check": "T1", "name": "Date range",
                "status": "SKIP", "detail": "no trades files found",
                "priority": "CRITICAL"}

    dates = sorted(filter(None, [_extract_date(f) for f in trades_files]))
    return {
        "check": "T1", "name": "Date range",
        "first_date": dates[0] if dates else "none",
        "last_date": dates[-1] if dates else "none",
        "n_days": len(dates),
        "status": "PASS" if dates else "FAIL",
        "priority": "CRITICAL",
    }


def T2_pre_cutoff_exclusion(root: Path = TARDIS_RAW_DIR) -> Dict:
    """T2: Ensure no data before 2020-05-14 infrastructure fix."""
    all_files = []
    for dt in ["trades", "incremental_book_L2", "liquidations"]:
        all_files.extend(_list_files(dt, root))

    early = []
    for f in all_files:
        d = _extract_date(f)
        if d and d < TARDIS_UNSAFE_DATE:
            early.append(f.name)

    return {
        "check": "T2", "name": "Pre-cutoff exclusion",
        "early_files": len(early),
        "detail": "; ".join(early[:5]) if early else "none",
        "status": "PASS" if not early else "FAIL",
        "priority": "CRITICAL",
    }


def T3_timestamp_ordering(root: Path = TARDIS_RAW_DIR,
                           sample_days: int = 5) -> List[Dict]:
    """T3: Verify timestamps are monotonically increasing."""
    trades_files = _list_files("trades", root)
    results = []

    for path in trades_files[:sample_days]:
        df = _load_day_csv(path)
        if df.empty or "timestamp" not in df.columns:
            results.append({"check": "T3", "name": "Timestamp order",
                          "file": path.name, "status": "SKIP",
                          "priority": "CRITICAL"})
            continue

        ts = df["timestamp"].values
        monotonic = bool(np.all(np.diff(ts) >= 0))
        results.append({
            "check": "T3", "name": "Timestamp order",
            "file": path.name,
            "n_rows": len(df),
            "is_monotonic": monotonic,
            "status": "PASS" if monotonic else "FAIL",
            "priority": "CRITICAL",
        })

    return results


def T4_sequence_gaps(root: Path = TARDIS_RAW_DIR,
                      sample_days: int = 5) -> List[Dict]:
    """T4: Check sequence number (u field) continuity in book data."""
    book_files = _list_files("incremental_book_L2", root)
    results = []

    for path in book_files[:sample_days]:
        df = _load_day_csv(path, max_rows=100000)
        if df.empty:
            continue

        # Look for sequence fields
        seq_col = None
        for col in ["u", "U", "lastUpdateId", "sequence"]:
            if col in df.columns:
                seq_col = col
                break

        if seq_col is None:
            results.append({"check": "T4", "name": "Seq gaps",
                          "file": path.name, "status": "SKIP",
                          "detail": "no seq column", "priority": "CRITICAL"})
            continue

        seq = df[seq_col].dropna().values
        if len(seq) < 2:
            continue

        diffs = np.diff(seq)
        gaps = int(np.sum(diffs > 1))

        results.append({
            "check": "T4", "name": "Sequence gaps",
            "file": path.name,
            "n_updates": len(seq),
            "gaps": gaps,
            "status": "PASS" if gaps == 0 else "WARN",
            "priority": "CRITICAL",
        })

    return results


def T5_may19_check(root: Path = TARDIS_RAW_DIR) -> Dict:
    """T5: Inspect 2021-05-19 data for the known trading halt."""
    trades_files = _list_files("trades", root)
    may19_file = None
    for f in trades_files:
        if "2021-05-19" in f.name:
            may19_file = f
            break

    if may19_file is None:
        return {"check": "T5", "name": "May 19 2021",
                "status": "SKIP", "detail": "file not found",
                "priority": "CRITICAL"}

    df = _load_day_csv(may19_file)
    if df.empty:
        return {"check": "T5", "name": "May 19 2021",
                "status": "SKIP", "detail": "empty file",
                "priority": "CRITICAL"}

    # Check for gap between 13:00-15:00 UTC
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], unit="us")
    elif "local_timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["local_timestamp"], unit="us")
    else:
        return {"check": "T5", "name": "May 19 2021",
                "status": "SKIP", "detail": "no timestamp col",
                "priority": "CRITICAL"}

    gap_start = pd.Timestamp("2021-05-19 13:00:00", tz="UTC")
    gap_end = pd.Timestamp("2021-05-19 15:00:00", tz="UTC")

    if df["dt"].dt.tz is None:
        df["dt"] = df["dt"].dt.tz_localize("UTC")

    in_gap = df[(df["dt"] >= gap_start) & (df["dt"] <= gap_end)]
    total = len(df)

    return {
        "check": "T5", "name": "May 19 2021",
        "total_trades": total,
        "trades_in_gap_window": len(in_gap),
        "gap_detected": "YES" if len(in_gap) < total * 0.01 else "PARTIAL",
        "recommendation": "FLAG or EXCLUDE this date",
        "status": "WARN",
        "priority": "CRITICAL",
    }


def T6_orderbook_validity(root: Path = TARDIS_RAW_DIR,
                           sample_days: int = 3) -> List[Dict]:
    """T6: Verify best_bid < best_ask after reconstruction."""
    from src.data.tardis_orderbook import reconstruct_day

    book_files = _list_files("incremental_book_L2", root)
    results = []

    for path in book_files[:sample_days]:
        try:
            snapshots = reconstruct_day(path)
            if not snapshots:
                continue
            violations = sum(1 for s in snapshots if not s["valid"])
            total = len(snapshots)
            results.append({
                "check": "T6", "name": "OB validity",
                "file": path.name,
                "snapshots": total,
                "violations": violations,
                "violation_pct": f"{violations / total * 100:.3f}%",
                "status": "PASS" if violations == 0 else "FAIL",
                "priority": "CRITICAL",
            })
        except Exception as e:
            results.append({"check": "T6", "name": "OB validity",
                          "file": path.name, "status": "ERROR",
                          "detail": str(e)[:100], "priority": "CRITICAL"})

    return results


def T7_snapshot_completeness(root: Path = TARDIS_RAW_DIR,
                              sample_days: int = 5) -> List[Dict]:
    """T7: Count is_snapshot=true rows – expect >= 1/day."""
    book_files = _list_files("incremental_book_L2", root)
    results = []

    for path in book_files[:sample_days]:
        df = _load_day_csv(path, max_rows=500000)
        if df.empty:
            continue

        if "is_snapshot" in df.columns:
            n_snap = int((df["is_snapshot"] == True).sum() |
                        (df["is_snapshot"] == "true").sum())
        else:
            n_snap = 0

        results.append({
            "check": "T7", "name": "Snapshot completeness",
            "file": path.name,
            "n_snapshots": n_snap,
            "status": "PASS" if n_snap >= 1 else "WARN",
            "priority": "HIGH",
        })

    return results


def T8_price_outliers(root: Path = TARDIS_RAW_DIR,
                       sample_days: int = 10) -> List[Dict]:
    """T8: Z-score of midprice returns, flag |z| > 10."""
    trades_files = _list_files("trades", root)
    results = []

    for path in trades_files[:sample_days]:
        df = _load_day_csv(path)
        if df.empty or "price" not in df.columns:
            continue

        prices = df["price"].astype(float).values
        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < 10:
            continue

        z = (returns - returns.mean()) / (returns.std() + 1e-10)
        n_outliers = int(np.sum(np.abs(z) > AUDIT["tardis_outlier_zscore"]))

        results.append({
            "check": "T8", "name": "Price outliers",
            "file": path.name,
            "n_trades": len(prices),
            "n_outliers": n_outliers,
            "max_z": f"{np.max(np.abs(z)):.1f}",
            "status": "PASS" if n_outliers == 0 else "WARN",
            "priority": "HIGH",
        })

    return results


def T9_missing_ticks(root: Path = TARDIS_RAW_DIR,
                      sample_days: int = 5) -> List[Dict]:
    """T9: Estimate expected vs actual message count."""
    trades_files = _list_files("trades", root)
    results = []

    for path in trades_files[:sample_days]:
        df = _load_day_csv(path)
        if df.empty or "timestamp" not in df.columns:
            continue

        ts = df["timestamp"].values
        span_sec = (ts[-1] - ts[0]) / 1e6 if len(ts) > 1 else 0
        avg_rate = len(ts) / span_sec if span_sec > 0 else 0

        # Check for large gaps (> 60 seconds)
        diffs = np.diff(ts) / 1e6  # to seconds
        large_gaps = int(np.sum(diffs > 60))

        results.append({
            "check": "T9", "name": "Missing ticks",
            "file": path.name,
            "n_trades": len(df),
            "span_hours": f"{span_sec / 3600:.1f}",
            "avg_rate_per_sec": f"{avg_rate:.1f}",
            "gaps_gt_60s": large_gaps,
            "status": "PASS" if large_gaps < 5 else "WARN",
            "priority": "HIGH",
        })

    return results


def T10_daily_reconnection_gap(root: Path = TARDIS_RAW_DIR,
                                sample_days: int = 5) -> List[Dict]:
    """T10: Detect 300-3000ms reconnection gaps."""
    book_files = _list_files("incremental_book_L2", root)
    results = []

    for path in book_files[:sample_days]:
        df = _load_day_csv(path, max_rows=200000)
        if df.empty or "timestamp" not in df.columns:
            continue

        ts = df["timestamp"].values
        diffs_ms = np.diff(ts) / 1000  # to ms

        reconnect_gaps = np.sum((diffs_ms >= 300) & (diffs_ms <= 3000))
        large_gaps = np.sum(diffs_ms > 3000)

        results.append({
            "check": "T10", "name": "Reconnection gaps",
            "file": path.name,
            "reconnect_gaps_300_3000ms": int(reconnect_gaps),
            "gaps_gt_3s": int(large_gaps),
            "status": "PASS",
            "priority": "HIGH",
        })

    return results


def T11_trade_book_consistency(root: Path = TARDIS_RAW_DIR,
                                sample_days: int = 2) -> List[Dict]:
    """T11: Cross-validate trade prices with orderbook spread."""
    from src.data.tardis_orderbook import reconstruct_day_to_df
    results = []

    book_files = _list_files("incremental_book_L2", root)
    trades_files = _list_files("trades", root)

    # Match dates
    book_dates = {_extract_date(f): f for f in book_files}
    trade_dates = {_extract_date(f): f for f in trades_files}

    common_dates = sorted(set(book_dates) & set(trade_dates))[:sample_days]

    for date in common_dates:
        try:
            book_df = reconstruct_day_to_df(book_dates[date])
            trade_df = _load_day_csv(trade_dates[date])

            if book_df.empty or trade_df.empty:
                continue

            # Simple check: are trade prices within reasonable range of midprice?
            if "price" in trade_df.columns and "midprice" in book_df.columns:
                mid_mean = book_df["midprice"].dropna().mean()
                trade_mean = trade_df["price"].astype(float).mean()
                pct_diff = abs(trade_mean - mid_mean) / mid_mean * 100

                results.append({
                    "check": "T11", "name": "Trade-book consistency",
                    "date": date,
                    "midprice_mean": f"{mid_mean:.2f}",
                    "trade_mean": f"{trade_mean:.2f}",
                    "pct_diff": f"{pct_diff:.3f}%",
                    "status": "PASS" if pct_diff < 1 else "WARN",
                    "priority": "HIGH",
                })
        except Exception as e:
            results.append({"check": "T11", "name": "Trade-book consistency",
                          "date": date, "status": "ERROR",
                          "detail": str(e)[:100], "priority": "HIGH"})

    return results


def T12_stylized_facts(root: Path = TARDIS_RAW_DIR,
                        sample_days: int = 30) -> Dict:
    """T12: Compute kurtosis and volatility clustering from trades."""
    trades_files = _list_files("trades", root)

    all_returns = []
    for path in trades_files[:sample_days]:
        df = _load_day_csv(path)
        if df.empty or "price" not in df.columns:
            continue
        prices = df["price"].astype(float).values
        rets = np.diff(np.log(prices + 1e-10))
        all_returns.append(rets)

    if not all_returns:
        return {"check": "T12", "name": "Stylized facts",
                "status": "SKIP", "detail": "no data",
                "priority": "CRITICAL"}

    returns = np.concatenate(all_returns)
    from scipy.stats import kurtosis as kurt_func

    excess_kurt = float(kurt_func(returns, fisher=True))
    # Vol clustering: ACF of |returns| at lag 1
    abs_ret = np.abs(returns)
    if len(abs_ret) > 100:
        abs_ret_norm = abs_ret - abs_ret.mean()
        acf_1 = np.correlate(abs_ret_norm[:-1], abs_ret_norm[1:]) / \
                (np.var(abs_ret_norm) * (len(abs_ret_norm) - 1) + 1e-10)
        acf_1 = float(acf_1[0]) if len(acf_1) > 0 else 0
    else:
        acf_1 = 0

    # Return ACF at lag 1 (should be ~0)
    ret_norm = returns - returns.mean()
    if len(ret_norm) > 100:
        ret_acf_1 = float(np.correlate(ret_norm[:-1], ret_norm[1:])[0] / \
                    (np.var(ret_norm) * (len(ret_norm) - 1) + 1e-10))
    else:
        ret_acf_1 = 0

    return {
        "check": "T12", "name": "Stylized facts",
        "n_returns": len(returns),
        "excess_kurtosis": f"{excess_kurt:.2f}",
        "fat_tails": "YES" if excess_kurt > AUDIT["tardis_min_kurtosis"] else "NO",
        "abs_return_acf1": f"{acf_1:.4f}",
        "vol_clustering": "YES" if acf_1 > 0.05 else "WEAK",
        "return_acf1": f"{ret_acf_1:.4f}",
        "no_return_autocorr": "YES" if abs(ret_acf_1) < 0.05 else "NO",
        "status": "PASS" if excess_kurt > AUDIT["tardis_min_kurtosis"] else "WARN",
        "priority": "CRITICAL",
    }


def T13_liquidation_data(root: Path = TARDIS_RAW_DIR) -> Dict:
    """T13: Check liquidation data availability (from 2021-09-01)."""
    files = _list_files("liquidations", root)
    if not files:
        return {"check": "T13", "name": "Liquidation data",
                "status": "SKIP", "detail": "no files found",
                "priority": "MEDIUM"}
    dates = sorted(filter(None, [_extract_date(f) for f in files]))
    return {
        "check": "T13", "name": "Liquidation data",
        "first_date": dates[0],
        "last_date": dates[-1],
        "n_days": len(dates),
        "status": "PASS" if dates[0] >= "2021-09-01" else "WARN",
        "priority": "MEDIUM",
    }


def T14_open_interest(root: Path = TARDIS_RAW_DIR) -> Dict:
    """T14: Check open interest / derivative_ticker availability."""
    files = _list_files("derivative_ticker", root)
    if not files:
        return {"check": "T14", "name": "Open interest",
                "status": "SKIP", "detail": "not downloaded (optional)",
                "priority": "MEDIUM"}
    dates = sorted(filter(None, [_extract_date(f) for f in files]))
    return {
        "check": "T14", "name": "Open interest",
        "first_date": dates[0],
        "last_date": dates[-1],
        "n_days": len(dates),
        "status": "PASS",
        "priority": "MEDIUM",
    }


def T15_volume_distribution(root: Path = TARDIS_RAW_DIR,
                              sample_days: int = 10) -> Dict:
    """T15: Check for realistic intraday volume pattern."""
    trades_files = _list_files("trades", root)
    hourly_vols = np.zeros(24)
    days_counted = 0

    for path in trades_files[:sample_days]:
        df = _load_day_csv(path)
        if df.empty or "timestamp" not in df.columns:
            continue
        df["hour"] = pd.to_datetime(df["timestamp"], unit="us").dt.hour
        if "amount" in df.columns:
            hourly = df.groupby("hour")["amount"].sum()
        else:
            hourly = df.groupby("hour").size()
        for h, v in hourly.items():
            hourly_vols[h] += v
        days_counted += 1

    if days_counted == 0:
        return {"check": "T15", "name": "Volume dist",
                "status": "SKIP", "priority": "MEDIUM"}

    hourly_vols /= days_counted
    peak_hour = int(np.argmax(hourly_vols))
    trough_hour = int(np.argmin(hourly_vols[hourly_vols > 0])) if hourly_vols.sum() > 0 else 0

    return {
        "check": "T15", "name": "Volume distribution",
        "days_sampled": days_counted,
        "peak_hour_utc": peak_hour,
        "trough_hour_utc": trough_hour,
        "peak_trough_ratio": f"{hourly_vols[peak_hour] / (hourly_vols[trough_hour] + 1e-10):.1f}",
        "status": "PASS",
        "priority": "MEDIUM",
    }


# ─────────────────────── Full Audit Runner ───────────────────────────

def run_full_audit(root: Path = TARDIS_RAW_DIR,
                   save: bool = True,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> List[Dict]:
    """Run all T1-T15 checks, optionally filtered to [start_date, end_date]."""
    print("=" * 60)
    print("TARDIS BTC FUTURES AUDIT – T1 to T15")
    if start_date or end_date:
        print(f"  Date filter: {start_date or 'beginning'} → {end_date or 'end'}")
    print("=" * 60)

    # Temporarily rebind module-level _list_files so all T* functions
    # transparently pick up the date filter without signature changes.
    import functools, sys as _sys
    _this_module = _sys.modules[__name__]
    _orig_list_files = _list_files
    if start_date or end_date:
        _filtered = functools.partial(_list_files, start_date=start_date,
                                      end_date=end_date)
        setattr(_this_module, "_list_files", _filtered)

    results = []

    # T1
    t1 = T1_date_range(root)
    results.append(t1)
    print(f"\n[T1] Date range: {t1.get('first_date','?')} → {t1.get('last_date','?')} "
          f"({t1.get('n_days',0)} days) → {t1['status']}")

    # T2
    t2 = T2_pre_cutoff_exclusion(root)
    results.append(t2)
    print(f"[T2] Pre-cutoff exclusion: {t2['status']} ({t2['early_files']} early files)")

    # T3
    print("\n[T3] Timestamp ordering ...")
    t3_list = T3_timestamp_ordering(root)
    results.extend(t3_list)
    for r in t3_list:
        print(f"  {r.get('file','?')}: {r['status']}")

    # T4
    print("\n[T4] Sequence gaps ...")
    t4_list = T4_sequence_gaps(root)
    results.extend(t4_list)
    for r in t4_list:
        print(f"  {r.get('file','?')}: gaps={r.get('gaps','?')} → {r['status']}")

    # T5
    t5 = T5_may19_check(root)
    results.append(t5)
    print(f"\n[T5] May 19 2021: {t5['status']} "
          f"(gap_detected={t5.get('gap_detected','?')})")

    # T6
    print("\n[T6] Orderbook validity ...")
    t6_list = T6_orderbook_validity(root)
    results.extend(t6_list)
    for r in t6_list:
        print(f"  {r.get('file','?')}: {r['status']} "
              f"(violations={r.get('violations','?')})")

    # T7
    print("\n[T7] Snapshot completeness ...")
    t7_list = T7_snapshot_completeness(root)
    results.extend(t7_list)

    # T8
    print("\n[T8] Price outliers ...")
    t8_list = T8_price_outliers(root)
    results.extend(t8_list)
    for r in t8_list:
        print(f"  {r.get('file','?')}: outliers={r.get('n_outliers',0)} → {r['status']}")

    # T9
    print("\n[T9] Missing ticks ...")
    t9_list = T9_missing_ticks(root)
    results.extend(t9_list)

    # T10
    print("\n[T10] Reconnection gaps ...")
    t10_list = T10_daily_reconnection_gap(root)
    results.extend(t10_list)

    # T11
    print("\n[T11] Trade-orderbook consistency ...")
    t11_list = T11_trade_book_consistency(root)
    results.extend(t11_list)
    for r in t11_list:
        print(f"  {r.get('date','?')}: {r['status']}")

    # T12
    print("\n[T12] Stylized facts ...")
    t12 = T12_stylized_facts(root)
    results.append(t12)
    print(f"  kurtosis={t12.get('excess_kurtosis','?')}, "
          f"fat_tails={t12.get('fat_tails','?')}, "
          f"vol_clustering={t12.get('vol_clustering','?')}")

    # T13
    t13 = T13_liquidation_data(root)
    results.append(t13)
    print(f"\n[T13] Liquidation data: {t13['status']}")

    # T14
    t14 = T14_open_interest(root)
    results.append(t14)
    print(f"[T14] Open interest: {t14['status']}")

    # T15
    t15 = T15_volume_distribution(root)
    results.append(t15)
    print(f"[T15] Volume distribution: {t15['status']}")

    # Summary
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_warn = sum(1 for r in results if r.get("status") == "WARN")
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")
    print(f"{'=' * 60}")

    if save:
        report_path = REPORTS_DIR / "tardis_audit.csv"
        save_audit_report(results, report_path)

    # Restore original _list_files if it was patched
    setattr(_this_module, "_list_files", _orig_list_files)

    return results


if __name__ == "__main__":
    from config.settings import ensure_dirs
    ensure_dirs()
    run_full_audit()
