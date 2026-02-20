"""
WESAD Dataset Loader
====================
Loads WESAD data from either .pkl files (preferred) or raw respiban.txt + quest.csv.
Handles 15 subjects (S2-S17, excluding S1 and S12 due to sensor malfunction).

References:
    Schmidt et al. (2018) "Introducing WESAD, a Multimodal Dataset for
    Wearable Stress and Affect Detection" – ACM ICMI 2018.
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    WESAD_RAW_DIR, WESAD_EXPECTED_SUBJECTS, WESAD_LABELS,
    WESAD_CHEST_SR, WESAD_WRIST_BVP_SR, WESAD_WRIST_EDA_SR,
    WESAD_WRIST_TEMP_SR, WESAD_WRIST_ACC_SR,
)


@dataclass
class WESADSubject:
    """Container for one WESAD subject's data."""
    subject_id: str
    # Chest signals (700 Hz)
    chest_ecg: Optional[np.ndarray] = None      # (N,)
    chest_eda: Optional[np.ndarray] = None      # (N,)
    chest_emg: Optional[np.ndarray] = None      # (N,)
    chest_temp: Optional[np.ndarray] = None     # (N,)
    chest_acc: Optional[np.ndarray] = None      # (N, 3)
    chest_resp: Optional[np.ndarray] = None     # (N,)
    # Wrist signals (variable SR)
    wrist_bvp: Optional[np.ndarray] = None      # 64 Hz
    wrist_eda: Optional[np.ndarray] = None      # 4 Hz
    wrist_temp: Optional[np.ndarray] = None     # 4 Hz
    wrist_acc: Optional[np.ndarray] = None      # 32 Hz, (N, 3)
    # Labels (700 Hz, aligned with chest)
    labels: Optional[np.ndarray] = None         # (N,)
    # Questionnaire data
    questionnaire: Optional[pd.DataFrame] = None
    # Metadata
    readme: Optional[str] = None
    source: str = "unknown"  # "pkl" or "respiban"


def discover_subjects(root: Optional[Path] = None) -> List[str]:
    """Return sorted list of available subject folder names (e.g. ['S2','S3',...])."""
    root = root or WESAD_RAW_DIR
    if not root.exists():
        return []
    return sorted(
        [d.name for d in root.iterdir()
         if d.is_dir() and re.match(r"^S\d+$", d.name)],
        key=lambda s: int(s[1:])
    )


def load_subject_pkl(subject_dir: Path) -> WESADSubject:
    """
    Load from the pre-synchronized .pkl file.
    Structure: {'signal': {'chest': {...}, 'wrist': {...}}, 'label': ndarray}
    """
    sid = subject_dir.name
    pkl_path = subject_dir / f"{sid}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    subj = WESADSubject(subject_id=sid, source="pkl")
    subj.labels = np.asarray(data["label"]).ravel()

    # Chest signals
    chest = data.get("signal", {}).get("chest", {})
    if "ECG" in chest:
        subj.chest_ecg = np.asarray(chest["ECG"]).ravel()
    if "EDA" in chest:
        subj.chest_eda = np.asarray(chest["EDA"]).ravel()
    if "EMG" in chest:
        subj.chest_emg = np.asarray(chest["EMG"]).ravel()
    if "Temp" in chest:
        subj.chest_temp = np.asarray(chest["Temp"]).ravel()
    if "ACC" in chest:
        subj.chest_acc = np.asarray(chest["ACC"])
        if subj.chest_acc.ndim == 1:
            subj.chest_acc = subj.chest_acc.reshape(-1, 3)
    if "Resp" in chest:
        subj.chest_resp = np.asarray(chest["Resp"]).ravel()

    # Wrist signals
    wrist = data.get("signal", {}).get("wrist", {})
    if "BVP" in wrist:
        subj.wrist_bvp = np.asarray(wrist["BVP"]).ravel()
    if "EDA" in wrist:
        subj.wrist_eda = np.asarray(wrist["EDA"]).ravel()
    if "TEMP" in wrist:
        subj.wrist_temp = np.asarray(wrist["TEMP"]).ravel()
    if "ACC" in wrist:
        subj.wrist_acc = np.asarray(wrist["ACC"])
        if subj.wrist_acc.ndim == 1:
            subj.wrist_acc = subj.wrist_acc.reshape(-1, 3)

    # Questionnaire
    quest_path = subject_dir / f"{sid}_quest.csv"
    if quest_path.exists():
        subj.questionnaire = pd.read_csv(quest_path)

    # Readme
    readme_path = subject_dir / f"{sid}_readme.txt"
    if readme_path.exists():
        subj.readme = readme_path.read_text(encoding="utf-8", errors="replace")

    return subj


def _parse_respiban_header(header_lines: List[str]) -> Dict:
    """Parse RespiBAN header metadata."""
    meta = {}
    for line in header_lines:
        line = line.strip().lstrip("#").strip()
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip()
    return meta


def load_subject_respiban(subject_dir: Path) -> WESADSubject:
    """
    Fallback loader: parse *_respiban.txt (chest RespiBAN data).
    Wrist data NOT available from this format.
    
    RespiBAN txt format:
        - Header lines start with '#'
        - Data columns: nSeq  DI  ECG  EDA  EMG  Temp  Accel_x  Accel_y  Accel_z  Resp
    """
    sid = subject_dir.name
    resp_path = subject_dir / f"{sid}_respiban.txt"
    if not resp_path.exists():
        raise FileNotFoundError(f"Respiban file not found: {resp_path}")

    header_lines = []
    data_start = 0
    with open(resp_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                header_lines.append(line)
                data_start = i + 1
            else:
                break

    meta = _parse_respiban_header(header_lines)

    # Parse data – tab-separated, skip header rows
    try:
        raw = np.loadtxt(resp_path, comments="#", delimiter="\t")
    except Exception:
        raw = np.genfromtxt(resp_path, comments="#", delimiter="\t",
                            filling_values=np.nan)

    subj = WESADSubject(subject_id=sid, source="respiban")

    # Expected column order: nSeq DI ECG EDA EMG Temp Accel_x Accel_y Accel_z Resp
    if raw.ndim == 2 and raw.shape[1] >= 10:
        subj.chest_ecg  = raw[:, 2]
        subj.chest_eda  = raw[:, 3]
        subj.chest_emg  = raw[:, 4]
        subj.chest_temp = raw[:, 5]
        subj.chest_acc  = raw[:, 6:9]
        subj.chest_resp = raw[:, 9]
    elif raw.ndim == 2:
        # Fewer columns – assign what we can
        if raw.shape[1] > 2:
            subj.chest_ecg = raw[:, 2]
        if raw.shape[1] > 3:
            subj.chest_eda = raw[:, 3]

    # Questionnaire
    quest_path = subject_dir / f"{sid}_quest.csv"
    if quest_path.exists():
        subj.questionnaire = pd.read_csv(quest_path)

    # Readme
    readme_path = subject_dir / f"{sid}_readme.txt"
    if readme_path.exists():
        subj.readme = readme_path.read_text(encoding="utf-8", errors="replace")

    return subj


def load_subject(subject_dir: Path) -> WESADSubject:
    """
    Load a subject – prefer .pkl, fallback to respiban.txt.
    """
    sid = subject_dir.name
    pkl_path = subject_dir / f"{sid}.pkl"
    if pkl_path.exists():
        return load_subject_pkl(subject_dir)
    else:
        return load_subject_respiban(subject_dir)


def load_all_subjects(root: Optional[Path] = None) -> Dict[str, WESADSubject]:
    """Load all available subjects. Returns dict {subject_id: WESADSubject}."""
    root = root or WESAD_RAW_DIR
    subjects = {}
    for sid in discover_subjects(root):
        subj_dir = root / sid
        try:
            subjects[sid] = load_subject(subj_dir)
            print(f"  [wesad] Loaded {sid} (source={subjects[sid].source})")
        except Exception as e:
            print(f"  [wesad] FAILED {sid}: {e}")
    return subjects


def get_label_counts(labels: np.ndarray) -> Dict[str, dict]:
    """Return {label_name: {count, pct}} for each label value."""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    result = {}
    for u, c in zip(unique, counts):
        name = WESAD_LABELS.get(int(u), f"unknown_{u}")
        result[name] = {"value": int(u), "count": int(c),
                        "pct": round(c / total * 100, 2)}
    return result
