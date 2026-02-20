"""
Script 16 – DREAMER Recovery Experiment
=========================================
Implements the advisor's recommended strategy to rescue DREAMER:

  Step 1: Within-subject z-normalization (mandatory)
  Step 2: Test 3 target variants: stress, arousal, valence
  Step 3: LOSOCV with LogReg + RF for each combination
  Step 4: Compare with pre-normalization NO_SIGNAL baseline

Expected outcome (advisor's prediction):
  - Subject probe: 92% → ~20-40%
  - Balanced accuracy: 0.54 → 0.60-0.68 (arousal target)

Usage:
    python scripts/phase2_validation/16_dreamer_recovery.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Force unbuffered output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from config.settings import PROCESSED_DIR, VALIDATION_DIR
from src.validation.losocv import losocv_evaluate
from src.validation.scaling import get_scaler_factory


def load_dreamer_with_options(
    normalize_within_subject: bool = False,
    target: str = "stress",           # "stress", "arousal", "valence"
    arousal_threshold: int = 3,       # >=thr = high arousal
    valence_threshold: int = 3,       # <=thr = low valence
):
    """
    Load DREAMER features with optional within-subject z-normalization
    and configurable target variable.
    
    Returns:
        X, y, subjects, feature_cols
    """
    out_dir = PROCESSED_DIR / "dreamer"
    npz_files = sorted(out_dir.glob("S*_preprocessed.npz"))
    
    all_X, all_y, all_subjects = [], [], []
    
    for npz_file in npz_files:
        subj = npz_file.stem.replace("_preprocessed", "")
        data = np.load(npz_file)
        
        X_subj = data["de_features"].astype(np.float64)  # (N, 70)
        
        # ── Within-subject z-normalization ──
        if normalize_within_subject:
            mu = X_subj.mean(axis=0, keepdims=True)
            sigma = X_subj.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0  # avoid div-by-zero
            X_subj = (X_subj - mu) / sigma
        
        # ── Target selection ──
        if target == "stress":
            y_subj = data["stress_labels"].astype(np.int32)
        elif target == "arousal":
            y_subj = (data["arousal"] >= arousal_threshold).astype(np.int32)
        elif target == "valence":
            y_subj = (data["valence"] <= valence_threshold).astype(np.int32)
        else:
            raise ValueError(f"Unknown target: {target}")
        
        n = len(y_subj)
        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subjects.extend([subj] * n)
    
    feature_cols = [f"f{i}" for i in range(70)]
    return (
        np.vstack(all_X),
        np.concatenate(all_y),
        np.array(all_subjects),
        feature_cols,
    )


def run_subject_probe(X, subjects, max_samples=10000):
    """Quick subject classification probe to measure subject encoding.
    
    Uses subsampling for speed (85K samples with 23-class LogReg is slow).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    y_subj = le.fit_transform(subjects)
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes
    
    # Subsample for speed if dataset is large
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X_sub, y_sub = X[idx], y_subj[idx]
    else:
        X_sub, y_sub = X, y_subj
    
    clf = LogisticRegression(max_iter=500, solver="saga", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_sub, y_sub, cv=cv, scoring="accuracy")
    probe_acc = scores.mean()
    
    return {
        "probe_accuracy": round(probe_acc, 4),
        "chance_level": round(chance, 4),
        "encoding_ratio": round(probe_acc / chance, 1),
        "n_classes": n_classes,
    }


def run_experiment(X, y, subjects, feature_cols, label, dataset_tag="dreamer"):
    """Run LogReg + RF LOSOCV and return results dict."""
    results = {}
    
    for model_name, factory in [
        ("LogisticRegression", lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42)),
        ("RandomForest", lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]:
        print(f"\n  {model_name} ...")
        
        scaler = get_scaler_factory(dataset_tag, feature_cols)
        
        r = losocv_evaluate(
            model_factory=factory,
            X=X, y=y, subjects=subjects,
            scaler_factory=scaler,
            verbose=True,
        )
        
        results[model_name] = {
            "bal_acc": round(r["aggregate"]["balanced_accuracy_mean"], 4),
            "f1": round(r["aggregate"]["f1_mean"], 4),
            "auc": round(r["aggregate"].get("auc_roc_mean", 0), 4),
            "std": round(np.std([s["balanced_accuracy"] for s in r["per_subject"]]), 4),
        }
        
        print(f"    bal_acc = {results[model_name]['bal_acc']}")
    
    return results


def main():
    print("=" * 70)
    print("#  DREAMER RECOVERY EXPERIMENT")
    print("#  Strategy: within-subject z-norm + target redefinition")
    print("=" * 70)
    
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    
    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Original (no normalization, stress target) — baseline
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Original (no z-norm, stress target) -- BASELINE")
    print("=" * 70)
    
    X, y, subjects, fcols = load_dreamer_with_options(
        normalize_within_subject=False, target="stress")
    
    n_stress = y.sum()
    print(f"  Samples: {len(y):,} | Stress: {n_stress:,} ({n_stress/len(y)*100:.1f}%)")
    
    probe = run_subject_probe(X, subjects)
    print(f"  Subject probe: {probe['probe_accuracy']*100:.1f}% "
          f"({probe['encoding_ratio']}x chance)")
    
    exp1 = run_experiment(X, y, subjects, fcols, "original_stress")
    all_results["1_original_stress"] = {
        "normalization": "none",
        "target": "stress",
        "subject_probe": probe,
        "models": exp1,
        "class_balance": f"{n_stress/len(y)*100:.1f}% positive",
    }
    
    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: z-norm + stress target
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Within-subject z-norm + stress target")
    print("=" * 70)
    
    X, y, subjects, fcols = load_dreamer_with_options(
        normalize_within_subject=True, target="stress")
    
    probe = run_subject_probe(X, subjects)
    print(f"  Subject probe: {probe['probe_accuracy']*100:.1f}% "
          f"({probe['encoding_ratio']}x chance)")
    
    exp2 = run_experiment(X, y, subjects, fcols, "znorm_stress")
    all_results["2_znorm_stress"] = {
        "normalization": "within_subject_zscore",
        "target": "stress",
        "subject_probe": probe,
        "models": exp2,
        "class_balance": f"{y.sum()/len(y)*100:.1f}% positive",
    }
    
    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: z-norm + arousal target (advisor's top recommendation)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Within-subject z-norm + AROUSAL target")
    print("=" * 70)
    
    X, y, subjects, fcols = load_dreamer_with_options(
        normalize_within_subject=True, target="arousal")
    
    n_pos = y.sum()
    print(f"  Samples: {len(y):,} | High arousal: {n_pos:,} ({n_pos/len(y)*100:.1f}%)")
    
    probe = run_subject_probe(X, subjects)
    print(f"  Subject probe: {probe['probe_accuracy']*100:.1f}% "
          f"({probe['encoding_ratio']}x chance)")
    
    exp3 = run_experiment(X, y, subjects, fcols, "znorm_arousal")
    all_results["3_znorm_arousal"] = {
        "normalization": "within_subject_zscore",
        "target": "arousal_binary",
        "subject_probe": probe,
        "models": exp3,
        "class_balance": f"{n_pos/len(y)*100:.1f}% positive",
    }
    
    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: z-norm + valence target
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Within-subject z-norm + VALENCE target")
    print("=" * 70)
    
    X, y, subjects, fcols = load_dreamer_with_options(
        normalize_within_subject=True, target="valence")
    
    n_pos = y.sum()
    print(f"  Samples: {len(y):,} | Low valence: {n_pos:,} ({n_pos/len(y)*100:.1f}%)")
    
    probe = run_subject_probe(X, subjects)
    print(f"  Subject probe: {probe['probe_accuracy']*100:.1f}% "
          f"({probe['encoding_ratio']}x chance)")
    
    exp4 = run_experiment(X, y, subjects, fcols, "znorm_valence")
    all_results["4_znorm_valence"] = {
        "normalization": "within_subject_zscore",
        "target": "valence_binary",
        "subject_probe": probe,
        "models": exp4,
        "class_balance": f"{n_pos/len(y)*100:.1f}% positive",
    }
    
    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: NO z-norm + arousal target (ablation: is z-norm needed?)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: NO z-norm + AROUSAL target (ablation)")
    print("=" * 70)
    
    X, y, subjects, fcols = load_dreamer_with_options(
        normalize_within_subject=False, target="arousal")
    
    n_pos = y.sum()
    print(f"  Samples: {len(y):,} | High arousal: {n_pos:,} ({n_pos/len(y)*100:.1f}%)")
    
    probe = run_subject_probe(X, subjects)
    print(f"  Subject probe: {probe['probe_accuracy']*100:.1f}% "
          f"({probe['encoding_ratio']}x chance)")
    
    exp5 = run_experiment(X, y, subjects, fcols, "raw_arousal")
    all_results["5_raw_arousal"] = {
        "normalization": "none",
        "target": "arousal_binary",
        "subject_probe": probe,
        "models": exp5,
        "class_balance": f"{n_pos/len(y)*100:.1f}% positive",
    }
    
    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DREAMER RECOVERY SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Experiment':<30} {'Norm':<10} {'Target':<10} {'Probe':<10} "
          f"{'LR bal_acc':<12} {'RF bal_acc':<12}")
    print("-" * 90)
    
    for key, res in all_results.items():
        norm = "z-norm" if "zscore" in res["normalization"] else "none"
        target = res["target"]
        probe = f"{res['subject_probe']['probe_accuracy']*100:.1f}%"
        lr = res["models"]["LogisticRegression"]["bal_acc"]
        rf = res["models"]["RandomForest"]["bal_acc"]
        print(f"  {key:<30} {norm:<10} {target:<10} {probe:<10} {lr:<12} {rf:<12}")
    
    # Decision logic
    best_key = max(all_results.keys(),
                   key=lambda k: max(all_results[k]["models"]["LogisticRegression"]["bal_acc"],
                                      all_results[k]["models"]["RandomForest"]["bal_acc"]))
    best = all_results[best_key]
    best_acc = max(best["models"]["LogisticRegression"]["bal_acc"],
                   best["models"]["RandomForest"]["bal_acc"])
    
    if best_acc >= 0.65:
        decision = "RECOVERED"
        action = "DREAMER is viable. Proceed with deep model using this configuration."
    elif best_acc >= 0.58:
        decision = "PARTIAL_RECOVERY"
        action = "Marginal improvement. Consider PSD features or raw EEG CNN."
    else:
        decision = "NO_RECOVERY"
        action = "Accept DREAMER as negative control. Focus on WESAD."
    
    all_results["_summary"] = {
        "best_experiment": best_key,
        "best_bal_acc": best_acc,
        "decision": decision,
        "action": action,
        "original_baseline": all_results["1_original_stress"]["models"]["LogisticRegression"]["bal_acc"],
        "improvement": round(best_acc - all_results["1_original_stress"]["models"]["LogisticRegression"]["bal_acc"], 4),
    }
    
    print(f"\n  BEST: {best_key} -> bal_acc = {best_acc}")
    print(f"  DECISION: {decision}")
    print(f"  ACTION: {action}")
    
    # Save
    out_path = VALIDATION_DIR / "dreamer_recovery_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
