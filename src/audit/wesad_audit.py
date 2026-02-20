"""
WESAD Audit – Checklist W1-W12
===============================
Runs every mandatory dataset check from the Algorithmic Panic audit protocol
BEFORE any modelling begins.

Checks:
    W1  Subject count verification
    W2  Sampling rate consistency
    W3  Label distribution
    W4  Class imbalance ratio
    W5  Missing data per channel
    W6  RR-interval completeness
    W7  Device synchronization (pkl only)
    W8  ECG signal quality (SNR)
    W9  EDA artifact rate (ACC-correlated noise)
    W10 Label reliability (protocol vs self-report)
    W11 Stress distribution shape
    W12 Subject demographic balance
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    WESAD_RAW_DIR, WESAD_EXPECTED_SUBJECTS, WESAD_LABELS,
    WESAD_CHEST_SR, WESAD_WRIST_BVP_SR, WESAD_WRIST_EDA_SR,
    WESAD_WRIST_TEMP_SR, WESAD_WRIST_ACC_SR,
    WESAD_ECG_BANDPASS, WESAD_IBI_RANGE, AUDIT, REPORTS_DIR,
)
from src.data.wesad_loader import (
    discover_subjects, load_subject, load_all_subjects,
    WESADSubject, get_label_counts,
)
from src.utils.io_utils import save_audit_report, print_audit_table


# ─────────────────────── Individual Checks ───────────────────────────

def W1_subject_count(root: Path = WESAD_RAW_DIR) -> Dict:
    """W1: Verify 15 subjects (S2-S17, minus S1, S12)."""
    found = discover_subjects(root)
    expected = WESAD_EXPECTED_SUBJECTS
    missing = sorted(set(expected) - set(found))
    extra   = sorted(set(found) - set(expected))
    passed  = len(found) == AUDIT["wesad_min_subjects"] and not missing
    return {
        "check": "W1", "name": "Subject count",
        "expected": AUDIT["wesad_min_subjects"],
        "found": len(found),
        "missing": ",".join(missing) if missing else "none",
        "extra": ",".join(extra) if extra else "none",
        "status": "PASS" if passed else "FAIL",
        "priority": "CRITICAL",
    }


def W2_sampling_rates(subj: WESADSubject) -> Dict:
    """W2: Check sampling rate consistency."""
    issues = []
    if subj.source == "pkl" and subj.labels is not None:
        # Chest signals should be same length as labels (all at 700 Hz)
        label_len = len(subj.labels)
        for name, arr in [("ECG", subj.chest_ecg), ("EDA", subj.chest_eda),
                          ("EMG", subj.chest_emg), ("Temp", subj.chest_temp),
                          ("Resp", subj.chest_resp)]:
            if arr is not None and len(arr) != label_len:
                issues.append(f"{name}:{len(arr)}vs{label_len}")
        # Chest ACC – 3 cols
        if subj.chest_acc is not None and subj.chest_acc.shape[0] != label_len:
            issues.append(f"ACC:{subj.chest_acc.shape[0]}vs{label_len}")

    return {
        "check": "W2", "name": "Sampling rate consistency",
        "subject": subj.subject_id,
        "source": subj.source,
        "issues": "; ".join(issues) if issues else "none",
        "status": "PASS" if not issues else "FAIL",
        "priority": "CRITICAL",
    }


def W3_label_distribution(subj: WESADSubject) -> Dict:
    """W3: Document exact % per class."""
    if subj.labels is None:
        return {"check": "W3", "name": "Label dist", "subject": subj.subject_id,
                "status": "SKIP", "priority": "CRITICAL", "detail": "no labels"}
    dist = get_label_counts(subj.labels)
    detail = "; ".join(f"{k}={v['pct']}%" for k, v in dist.items())
    return {
        "check": "W3", "name": "Label distribution",
        "subject": subj.subject_id,
        "detail": detail,
        "status": "PASS",
        "priority": "CRITICAL",
    }


def W4_class_imbalance(subj: WESADSubject) -> Dict:
    """W4: If stress < 15% → flag."""
    if subj.labels is None:
        return {"check": "W4", "name": "Imbalance", "subject": subj.subject_id,
                "status": "SKIP", "priority": "CRITICAL"}
    total = len(subj.labels)
    stress = np.sum(subj.labels == 2)
    pct = stress / total * 100 if total > 0 else 0
    return {
        "check": "W4", "name": "Class imbalance",
        "subject": subj.subject_id,
        "stress_pct": f"{pct:.1f}%",
        "needs_balancing": "YES" if pct < 15 else "NO",
        "status": "WARN" if pct < 15 else "PASS",
        "priority": "CRITICAL",
    }


def W5_missing_data(subj: WESADSubject) -> Dict:
    """W5: Count NaN/None per channel, flag if >5%."""
    issues = []
    for name, arr in [("ECG", subj.chest_ecg), ("EDA", subj.chest_eda),
                      ("EMG", subj.chest_emg), ("Temp", subj.chest_temp),
                      ("Resp", subj.chest_resp)]:
        if arr is not None:
            nan_pct = np.isnan(arr).sum() / len(arr) * 100 if len(arr) > 0 else 0
            if nan_pct > AUDIT["wesad_max_missing_pct"]:
                issues.append(f"{name}:{nan_pct:.1f}%NaN")

    if subj.chest_acc is not None:
        nan_pct = np.isnan(subj.chest_acc).any(axis=1).sum() / len(subj.chest_acc) * 100
        if nan_pct > AUDIT["wesad_max_missing_pct"]:
            issues.append(f"ACC:{nan_pct:.1f}%NaN")

    return {
        "check": "W5", "name": "Missing data",
        "subject": subj.subject_id,
        "issues": "; ".join(issues) if issues else "none",
        "status": "PASS" if not issues else "FAIL",
        "priority": "HIGH",
    }


def W6_rr_interval_completeness(subj: WESADSubject) -> Dict:
    """W6: Check HRV feature feasibility – can we detect R-peaks?"""
    if subj.chest_ecg is None:
        return {"check": "W6", "name": "RR completeness", "subject": subj.subject_id,
                "status": "SKIP", "detail": "no ECG", "priority": "HIGH"}

    # Quick check: is signal mostly valid (non-NaN, non-zero)?
    ecg = subj.chest_ecg
    valid_pct = (~np.isnan(ecg) & (ecg != 0)).sum() / len(ecg) * 100
    return {
        "check": "W6", "name": "RR completeness",
        "subject": subj.subject_id,
        "valid_ecg_pct": f"{valid_pct:.1f}%",
        "status": "PASS" if valid_pct > 80 else "WARN",
        "priority": "HIGH",
    }


def W7_device_sync(subj: WESADSubject) -> Dict:
    """W7: Label-duration temporal consistency check (pkl only).

    WESAD pkl files are pre-synchronized by the dataset authors (RespiBAN chest
    at 700 Hz and Empatica E4 wrist at their respective rates are already aligned
    to the same label vector).  ACC cross-correlation is unreliable here because
    the protocol consists of quasi-static tasks (sitting, writing, reading) where
    both ACC channels are nearly flat and the correlation peak is dominated by noise.

    Instead we verify that each wrist modality contains the expected number of
    samples for the total recording duration implied by the label array:

        expected_wrist_samples = n_labels * wrist_sr / WESAD_CHEST_SR

    A relative deviation > wesad_sync_tol_pct (default 2 %) indicates a genuine
    length mismatch / truncation that would misalign labels with wrist features.
    """
    if subj.source != "pkl" or subj.labels is None:
        return {"check": "W7", "name": "Device sync", "subject": subj.subject_id,
                "status": "SKIP", "detail": "not pkl or no labels",
                "priority": "CRITICAL"}

    n_labels = len(subj.labels)
    tol = AUDIT["wesad_sync_tol_pct"] / 100.0

    # (modality_name, array, sampling_rate_hz)
    modalities = [
        ("BVP",  subj.wrist_bvp,  WESAD_WRIST_BVP_SR),
        ("EDA",  subj.wrist_eda,  WESAD_WRIST_EDA_SR),
        ("Temp", subj.wrist_temp, WESAD_WRIST_TEMP_SR),
        ("ACC",  subj.wrist_acc,  WESAD_WRIST_ACC_SR),
    ]

    issues = []
    details = []
    for name, arr, sr in modalities:
        if arr is None:
            continue
        actual = arr.shape[0]
        expected = n_labels * sr / WESAD_CHEST_SR
        rel_dev = abs(actual - expected) / expected
        details.append(f"{name}:{actual}(exp {expected:.0f}, dev {rel_dev*100:.2f}%)")
        if rel_dev > tol:
            issues.append(f"{name}:dev={rel_dev*100:.1f}%>tol")

    return {
        "check": "W7", "name": "Device sync",
        "subject": subj.subject_id,
        "issues": "; ".join(issues) if issues else "none",
        "detail": "; ".join(details),
        "status": "PASS" if not issues else "FAIL",
        "priority": "CRITICAL",
    }


def W8_ecg_snr(subj: WESADSubject) -> Dict:
    """W8: Estimate ECG signal-to-noise ratio."""
    if subj.chest_ecg is None:
        return {"check": "W8", "name": "ECG SNR", "subject": subj.subject_id,
                "status": "SKIP", "priority": "HIGH"}

    ecg = subj.chest_ecg
    ecg = ecg[~np.isnan(ecg)]
    if len(ecg) < WESAD_CHEST_SR * 10:
        return {"check": "W8", "name": "ECG SNR", "subject": subj.subject_id,
                "status": "SKIP", "detail": "too short", "priority": "HIGH"}

    # Simple SNR estimate: power in 5-15 Hz (QRS) / power in 50-100 Hz (noise)
    from scipy.signal import welch
    freqs, psd = welch(ecg[:WESAD_CHEST_SR * 60], fs=WESAD_CHEST_SR, nperseg=4096)
    signal_band = (freqs >= 5) & (freqs <= 15)
    noise_band  = (freqs >= 50) & (freqs <= 100)

    sig_power = np.trapz(psd[signal_band], freqs[signal_band])
    noise_power = np.trapz(psd[noise_band], freqs[noise_band])

    if noise_power > 0:
        snr_db = 10 * np.log10(sig_power / noise_power)
    else:
        snr_db = float("inf")

    return {
        "check": "W8", "name": "ECG SNR",
        "subject": subj.subject_id,
        "snr_db": f"{snr_db:.1f}",
        "status": "PASS" if snr_db > AUDIT["wesad_snr_min_db"] else "FAIL",
        "priority": "HIGH",
    }


def W9_eda_artifact_rate(subj: WESADSubject) -> Dict:
    """W9: Motion-correlated EDA noise (ACC-EDA correlation)."""
    if subj.chest_eda is None or subj.chest_acc is None:
        return {"check": "W9", "name": "EDA artifacts", "subject": subj.subject_id,
                "status": "SKIP", "priority": "HIGH"}

    eda = subj.chest_eda
    acc_mag = np.sqrt(np.sum(subj.chest_acc ** 2, axis=1))
    min_len = min(len(eda), len(acc_mag))
    eda = eda[:min_len]
    acc_mag = acc_mag[:min_len]

    # Remove NaN
    valid = ~(np.isnan(eda) | np.isnan(acc_mag))
    if valid.sum() < 100:
        return {"check": "W9", "name": "EDA artifacts", "subject": subj.subject_id,
                "status": "SKIP", "detail": "too few valid samples", "priority": "HIGH"}

    corr = np.abs(np.corrcoef(eda[valid], acc_mag[valid])[0, 1])
    return {
        "check": "W9", "name": "EDA artifacts",
        "subject": subj.subject_id,
        "acc_eda_corr": f"{corr:.3f}",
        "status": "WARN" if corr > 0.3 else "PASS",
        "priority": "HIGH",
    }


def W10_label_reliability(subj: WESADSubject) -> Dict:
    """W10: Compare protocol labels with self-report (if available)."""
    if subj.questionnaire is None:
        return {"check": "W10", "name": "Label reliability",
                "subject": subj.subject_id,
                "status": "SKIP", "detail": "no questionnaire",
                "priority": "MEDIUM"}

    # We can only note that questionnaire data exists
    n_rows = len(subj.questionnaire)
    return {
        "check": "W10", "name": "Label reliability",
        "subject": subj.subject_id,
        "quest_rows": n_rows,
        "status": "INFO",
        "detail": "questionnaire available for manual review",
        "priority": "MEDIUM",
    }


def W11_stress_distribution(subj: WESADSubject) -> Dict:
    """W11: Check that stress segment distribution matches expectations."""
    if subj.labels is None:
        return {"check": "W11", "name": "Stress dist", "subject": subj.subject_id,
                "status": "SKIP", "priority": "HIGH"}

    stress_mask = subj.labels == 2
    stress_pct = stress_mask.sum() / len(subj.labels)
    return {
        "check": "W11", "name": "Stress distribution",
        "subject": subj.subject_id,
        "stress_fraction": f"{stress_pct:.3f}",
        "in_expected_range": "YES"
            if AUDIT["wesad_stress_mean_range"][0] <= stress_pct <= AUDIT["wesad_stress_mean_range"][1]
            else "NO",
        "status": "PASS"
            if AUDIT["wesad_stress_mean_range"][0] <= stress_pct <= AUDIT["wesad_stress_mean_range"][1]
            else "WARN",
        "priority": "HIGH",
    }


def W12_demographics(subj: WESADSubject) -> Dict:
    """W12: Report age, gender from readme."""
    age, gender = None, None
    if subj.readme:
        import re
        age_match = re.search(r"age[:\s]+(\d+)", subj.readme, re.IGNORECASE)
        gender_match = re.search(r"gender[:\s]+(male|female|m|f)", subj.readme, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
        if gender_match:
            gender = gender_match.group(1).strip()

    return {
        "check": "W12", "name": "Demographics",
        "subject": subj.subject_id,
        "age": age or "unknown",
        "gender": gender or "unknown",
        "status": "PASS" if age and gender else "INFO",
        "priority": "MEDIUM",
    }


# ─────────────────────── Full Audit Runner ───────────────────────────

def run_full_audit(root: Path = WESAD_RAW_DIR,
                   save: bool = True) -> List[Dict]:
    """Run all W1-W12 checks and return results."""
    print("=" * 60)
    print("WESAD AUDIT – W1 to W12")
    print("=" * 60)

    results = []

    # W1 – global check
    w1 = W1_subject_count(root)
    results.append(w1)
    print(f"\n[W1] Subject count: {w1['found']}/{w1['expected']} → {w1['status']}")
    if w1["missing"] != "none":
        print(f"     Missing: {w1['missing']}")

    # Load all subjects
    subjects = load_all_subjects(root)
    if not subjects:
        print("[!] No subjects loaded – cannot continue W2-W12.")
        return results

    # Per-subject checks
    for sid, subj in sorted(subjects.items()):
        print(f"\n--- {sid} (source={subj.source}) ---")
        checks = [
            W2_sampling_rates(subj),
            W3_label_distribution(subj),
            W4_class_imbalance(subj),
            W5_missing_data(subj),
            W6_rr_interval_completeness(subj),
            W7_device_sync(subj),
            W8_ecg_snr(subj),
            W9_eda_artifact_rate(subj),
            W10_label_reliability(subj),
            W11_stress_distribution(subj),
            W12_demographics(subj),
        ]
        for c in checks:
            status_icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠",
                          "SKIP": "○", "INFO": "ℹ"}.get(c["status"], "?")
            print(f"  [{c['check']}] {status_icon} {c['name']}: {c['status']}")
            results.append(c)

    # Summary
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")
    print(f"{'=' * 60}")

    if save:
        report_path = REPORTS_DIR / "wesad_audit.csv"
        save_audit_report(results, report_path)

    return results


if __name__ == "__main__":
    from config.settings import ensure_dirs
    ensure_dirs()
    run_full_audit()
