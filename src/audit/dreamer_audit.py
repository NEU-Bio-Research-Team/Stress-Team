"""
DREAMER Audit – Checklist D1-D12
================================
Runs every mandatory dataset check from the Algorithmic Panic audit protocol.

Checks:
    D1  Subject count (23)
    D2  Channel count & order (14 channels, 10-20 system)
    D3  Sampling rate (128 Hz)
    D4  Label distribution (V/A/D histograms)
    D5  Stress proxy definition & validation
    D6  EEG artifact detection (variance per channel)
    D7  ICA component quality (if MNE available)
    D8  Baseline signal integrity (61s per trial)
    D9  Cross-subject variance
    D10 ECG signal availability
    D11 Trial completeness (18 trials × 23 subjects = 414)
    D12 Frequency band power (PSD shape)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DREAMER_MAT_PATH, DREAMER_N_SUBJECTS, DREAMER_N_TRIALS,
    DREAMER_EEG_SR, DREAMER_EEG_CHANNELS, DREAMER_BASELINE_SEC,
    DREAMER_STRESS_AROUSAL_THR, DREAMER_STRESS_VALENCE_THR,
    DREAMER_ICA_N_COMPONENTS, AUDIT, REPORTS_DIR,
)
from src.data.dreamer_loader import (
    load_dreamer, get_all_labels, get_stress_labels,
    get_subject_demographics, DREAMERSubject,
)
from src.utils.io_utils import save_audit_report, print_audit_table


# ─────────────────────── Individual Checks ───────────────────────────

def D1_subject_count(subjects: List[DREAMERSubject]) -> Dict:
    """D1: Verify 23 subjects."""
    n = len(subjects)
    return {
        "check": "D1", "name": "Subject count",
        "expected": AUDIT["dreamer_min_subjects"],
        "found": n,
        "status": "PASS" if n == AUDIT["dreamer_min_subjects"] else "FAIL",
        "priority": "CRITICAL",
    }


def D2_channel_count(subjects: List[DREAMERSubject]) -> List[Dict]:
    """D2: Verify 14 EEG channels per subject/trial."""
    results = []
    for subj in subjects:
        for trial in subj.trials:
            if trial.eeg_stimulus is not None:
                n_ch = trial.eeg_stimulus.shape[1] if trial.eeg_stimulus.ndim == 2 else 0
                ok = n_ch == AUDIT["dreamer_n_channels"]
                if not ok:
                    results.append({
                        "check": "D2", "name": "Channel count",
                        "subject": subj.subject_id, "trial": trial.trial_id,
                        "channels": n_ch,
                        "status": "FAIL",
                        "priority": "CRITICAL",
                    })
    if not results:
        results.append({
            "check": "D2", "name": "Channel count",
            "detail": f"All trials have {AUDIT['dreamer_n_channels']} channels",
            "status": "PASS",
            "priority": "CRITICAL",
        })
    return results


def D3_sampling_rate(subjects: List[DREAMERSubject]) -> Dict:
    """D3: Verify data consistent with 128 Hz."""
    # Check by examining baseline length vs expected
    issues = []
    for subj in subjects:
        for trial in subj.trials:
            if trial.eeg_baseline is not None:
                n_samples = trial.eeg_baseline.shape[0]
                expected_min = int(DREAMER_BASELINE_SEC * DREAMER_EEG_SR * 0.9)
                expected_max = int(DREAMER_BASELINE_SEC * DREAMER_EEG_SR * 1.1)
                if not (expected_min <= n_samples <= expected_max):
                    issues.append(
                        f"S{subj.subject_id}/T{trial.trial_id}: "
                        f"baseline={n_samples} samples "
                        f"(expected ~{DREAMER_BASELINE_SEC * DREAMER_EEG_SR})"
                    )
    return {
        "check": "D3", "name": "Sampling rate",
        "expected_hz": DREAMER_EEG_SR,
        "issues": "; ".join(issues[:5]) if issues else "none",
        "n_issues": len(issues),
        "status": "PASS" if not issues else "WARN",
        "priority": "CRITICAL",
    }


def D4_label_distribution(subjects: List[DREAMERSubject]) -> Dict:
    """D4: Histogram of V/A/D labels, check skewness."""
    labels = get_all_labels(subjects)
    detail = {}
    for dim in ["valence", "arousal", "dominance"]:
        arr = labels[dim]
        detail[dim] = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "median": float(np.median(arr)),
        }
        # Count per value
        unique, counts = np.unique(arr, return_counts=True)
        detail[dim]["distribution"] = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "check": "D4", "name": "Label distribution",
        "valence_mean": detail["valence"]["mean"],
        "arousal_mean": detail["arousal"]["mean"],
        "dominance_mean": detail["dominance"]["mean"],
        "detail": str(detail),
        "status": "PASS",
        "priority": "CRITICAL",
    }


def D5_stress_proxy(subjects: List[DREAMERSubject]) -> Dict:
    """D5: Define and validate stress proxy (low V + high A)."""
    stress_labels = get_stress_labels(subjects)
    n_stress = int(stress_labels.sum())
    n_total = len(stress_labels)
    pct = n_stress / n_total * 100 if n_total > 0 else 0

    return {
        "check": "D5", "name": "Stress proxy",
        "definition": f"valence<={DREAMER_STRESS_VALENCE_THR} AND arousal>={DREAMER_STRESS_AROUSAL_THR}",
        "n_stress": n_stress,
        "n_total": n_total,
        "stress_pct": f"{pct:.1f}%",
        "status": "PASS" if 5 < pct < 50 else "WARN",
        "priority": "CRITICAL",
    }


def D6_eeg_artifact_detection(subjects: List[DREAMERSubject]) -> List[Dict]:
    """D6: Flag channels with variance > 3 std from mean (per subject)."""
    results = []
    thr = AUDIT["dreamer_artifact_std_thr"]

    for subj in subjects:
        all_stim = [t.eeg_stimulus for t in subj.trials
                     if t.eeg_stimulus is not None]
        if not all_stim:
            continue
        concat = np.vstack(all_stim)  # (total_samples, 14)
        ch_var = np.var(concat, axis=0)
        mean_var = np.mean(ch_var)
        std_var = np.std(ch_var)

        flagged = []
        for i, v in enumerate(ch_var):
            if abs(v - mean_var) > thr * std_var:
                ch_name = DREAMER_EEG_CHANNELS[i] if i < len(DREAMER_EEG_CHANNELS) else f"ch{i}"
                flagged.append(ch_name)

        results.append({
            "check": "D6", "name": "EEG artifacts",
            "subject": subj.subject_id,
            "flagged_channels": ",".join(flagged) if flagged else "none",
            "status": "WARN" if flagged else "PASS",
            "priority": "HIGH",
        })
    return results


def D7_ica_quality_check(subjects: List[DREAMERSubject]) -> Dict:
    """D7: Check if ICA is feasible (enough data, correct dims).
    Actual ICA runs in preprocessing – here we just verify prerequisites."""
    total_samples = 0
    for subj in subjects:
        for trial in subj.trials:
            if trial.eeg_stimulus is not None:
                total_samples += trial.eeg_stimulus.shape[0]

    min_for_ica = DREAMER_ICA_N_COMPONENTS * 20 * DREAMER_EEG_SR
    feasible = total_samples > min_for_ica

    return {
        "check": "D7", "name": "ICA feasibility",
        "total_eeg_samples": total_samples,
        "min_required": min_for_ica,
        "feasible": "YES" if feasible else "NO",
        "recommended_components": DREAMER_ICA_N_COMPONENTS,
        "status": "PASS" if feasible else "FAIL",
        "priority": "HIGH",
    }


def D8_baseline_integrity(subjects: List[DREAMERSubject]) -> List[Dict]:
    """D8: Check 61s baseline per trial – no missing/corrupted."""
    results = []
    expected_samples = DREAMER_BASELINE_SEC * DREAMER_EEG_SR

    for subj in subjects:
        missing = 0
        corrupted = 0
        for trial in subj.trials:
            if trial.eeg_baseline is None:
                missing += 1
            elif np.isnan(trial.eeg_baseline).any():
                corrupted += 1
            elif trial.eeg_baseline.shape[0] < expected_samples * 0.5:
                corrupted += 1

        results.append({
            "check": "D8", "name": "Baseline integrity",
            "subject": subj.subject_id,
            "missing_baselines": missing,
            "corrupted_baselines": corrupted,
            "status": "PASS" if missing == 0 and corrupted == 0 else "FAIL",
            "priority": "HIGH",
        })
    return results


def D9_cross_subject_variance(subjects: List[DREAMERSubject]) -> Dict:
    """D9: Compute inter-subject variability for normalization."""
    subj_means = []
    subj_stds = []
    for subj in subjects:
        all_stim = [t.eeg_stimulus for t in subj.trials
                     if t.eeg_stimulus is not None]
        if all_stim:
            concat = np.vstack(all_stim)
            subj_means.append(np.mean(concat))
            subj_stds.append(np.std(concat))

    if not subj_means:
        return {"check": "D9", "name": "Cross-subject var",
                "status": "SKIP", "priority": "HIGH"}

    return {
        "check": "D9", "name": "Cross-subject variance",
        "mean_range": f"[{min(subj_means):.2f}, {max(subj_means):.2f}]",
        "std_range": f"[{min(subj_stds):.2f}, {max(subj_stds):.2f}]",
        "cv": f"{np.std(subj_means) / (np.mean(subj_means) + 1e-10):.3f}",
        "needs_normalization": "YES" if max(subj_stds) / (min(subj_stds) + 1e-10) > 2 else "NO",
        "status": "PASS",
        "priority": "HIGH",
    }


def D10_ecg_availability(subjects: List[DREAMERSubject]) -> Dict:
    """D10: Check all 23 subjects have ECG data."""
    missing_ecg = []
    for subj in subjects:
        has_ecg = any(t.ecg_stimulus is not None for t in subj.trials)
        if not has_ecg:
            missing_ecg.append(subj.subject_id)

    return {
        "check": "D10", "name": "ECG availability",
        "missing": ",".join(map(str, missing_ecg)) if missing_ecg else "none",
        "status": "PASS" if not missing_ecg else "FAIL",
        "priority": "CRITICAL",
    }


def D11_trial_completeness(subjects: List[DREAMERSubject]) -> Dict:
    """D11: Count trials per subject → total should be 18×23=414."""
    total = 0
    issues = []
    for subj in subjects:
        n = len(subj.trials)
        total += n
        if n != DREAMER_N_TRIALS:
            issues.append(f"S{subj.subject_id}:{n}")

    expected_total = DREAMER_N_SUBJECTS * DREAMER_N_TRIALS
    return {
        "check": "D11", "name": "Trial completeness",
        "expected": expected_total,
        "found": total,
        "incomplete_subjects": "; ".join(issues) if issues else "none",
        "status": "PASS" if total == expected_total else "WARN",
        "priority": "MEDIUM",
    }


def D12_frequency_band_power(subjects: List[DREAMERSubject],
                              n_subjects: int = 3) -> List[Dict]:
    """D12: PSD per frequency band for a sample of subjects."""
    from scipy.signal import welch
    results = []
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    for subj in subjects[:n_subjects]:
        all_stim = [t.eeg_stimulus for t in subj.trials
                     if t.eeg_stimulus is not None]
        if not all_stim:
            continue
        concat = np.vstack(all_stim)  # (N, 14)
        # Average across channels
        avg_signal = concat.mean(axis=1)

        freqs, psd = welch(avg_signal, fs=DREAMER_EEG_SR, nperseg=256)

        band_powers = {}
        for band_name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            band_powers[band_name] = float(np.trapz(psd[mask], freqs[mask]))

        results.append({
            "check": "D12", "name": "Freq band power",
            "subject": subj.subject_id,
            **{f"power_{k}": f"{v:.4f}" for k, v in band_powers.items()},
            "status": "PASS",
            "priority": "MEDIUM",
        })

    return results


# ─────────────────────── Full Audit Runner ───────────────────────────

def run_full_audit(mat_path: Path = DREAMER_MAT_PATH,
                   save: bool = True) -> List[Dict]:
    """Run all D1-D12 checks."""
    print("=" * 60)
    print("DREAMER AUDIT – D1 to D12")
    print("=" * 60)

    print(f"\n[dreamer] Loading {mat_path} ...")
    subjects = load_dreamer(mat_path)
    print(f"[dreamer] Loaded {len(subjects)} subjects.")

    results = []

    # D1
    d1 = D1_subject_count(subjects)
    results.append(d1)
    print(f"\n[D1] Subject count: {d1['found']}/{d1['expected']} → {d1['status']}")

    # D2
    print("\n[D2] Channel count ...")
    d2_list = D2_channel_count(subjects)
    results.extend(d2_list)
    for r in d2_list:
        print(f"  {r['status']}: {r.get('detail', r.get('subject', ''))}")

    # D3
    d3 = D3_sampling_rate(subjects)
    results.append(d3)
    print(f"\n[D3] Sampling rate: {d3['status']} (issues: {d3['n_issues']})")

    # D4
    d4 = D4_label_distribution(subjects)
    results.append(d4)
    print(f"\n[D4] Labels: V={d4['valence_mean']}, A={d4['arousal_mean']}, D={d4['dominance_mean']}")

    # D5
    d5 = D5_stress_proxy(subjects)
    results.append(d5)
    print(f"\n[D5] Stress proxy: {d5['n_stress']}/{d5['n_total']} ({d5['stress_pct']})")

    # D6
    print("\n[D6] EEG artifact detection ...")
    d6_list = D6_eeg_artifact_detection(subjects)
    results.extend(d6_list)
    flagged_count = sum(1 for r in d6_list if r["status"] == "WARN")
    print(f"  {flagged_count}/{len(d6_list)} subjects have flagged channels")

    # D7
    d7 = D7_ica_quality_check(subjects)
    results.append(d7)
    print(f"\n[D7] ICA feasibility: {d7['feasible']} ({d7['total_eeg_samples']} samples)")

    # D8
    print("\n[D8] Baseline integrity ...")
    d8_list = D8_baseline_integrity(subjects)
    results.extend(d8_list)
    d8_fails = sum(1 for r in d8_list if r["status"] == "FAIL")
    print(f"  {d8_fails} subjects with baseline issues")

    # D9
    d9 = D9_cross_subject_variance(subjects)
    results.append(d9)
    print(f"\n[D9] Cross-subject var: mean_range={d9.get('mean_range','?')}, "
          f"needs_norm={d9.get('needs_normalization','?')}")

    # D10
    d10 = D10_ecg_availability(subjects)
    results.append(d10)
    print(f"\n[D10] ECG availability: {d10['status']} (missing: {d10['missing']})")

    # D11
    d11 = D11_trial_completeness(subjects)
    results.append(d11)
    print(f"\n[D11] Trials: {d11['found']}/{d11['expected']} → {d11['status']}")

    # D12
    print("\n[D12] Frequency band power (sample) ...")
    d12_list = D12_frequency_band_power(subjects, n_subjects=3)
    results.extend(d12_list)
    for r in d12_list:
        print(f"  S{r['subject']}: δ={r['power_delta']} θ={r['power_theta']} "
              f"α={r['power_alpha']} β={r['power_beta']} γ={r['power_gamma']}")

    # Summary
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_warn = sum(1 for r in results if r.get("status") == "WARN")
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")
    print(f"{'=' * 60}")

    if save:
        report_path = REPORTS_DIR / "dreamer_audit.csv"
        save_audit_report(results, report_path)

    return results


if __name__ == "__main__":
    from config.settings import ensure_dirs
    ensure_dirs()
    run_full_audit()
