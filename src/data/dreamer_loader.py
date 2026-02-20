"""
DREAMER Dataset Loader
======================
Loads the DREAMER.mat file containing 23 subjects × 18 trials of
EEG (14ch, 128 Hz) + ECG + self-report labels (Valence, Arousal, Dominance).

References:
    Katsigiannis & Ramzan (2018) "DREAMER: A Database for Emotion Recognition
    Through EEG and ECG Signals From Wireless Low-cost Off-the-Shelf Devices"
    – IEEE JBHI.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DREAMER_MAT_PATH, DREAMER_N_SUBJECTS, DREAMER_N_TRIALS,
    DREAMER_EEG_SR, DREAMER_EEG_CHANNELS, DREAMER_BASELINE_SEC,
    DREAMER_STRESS_AROUSAL_THR, DREAMER_STRESS_VALENCE_THR,
)


@dataclass
class DREAMERTrial:
    """Container for one trial of one subject."""
    subject_id: int
    trial_id: int
    eeg_baseline: Optional[np.ndarray] = None   # (n_baseline, 14)
    eeg_stimulus: Optional[np.ndarray] = None   # (n_stimulus, 14)
    ecg_baseline: Optional[np.ndarray] = None   # (n_baseline, 2)
    ecg_stimulus: Optional[np.ndarray] = None   # (n_stimulus, 2)
    valence: Optional[int] = None               # 1-5
    arousal: Optional[int] = None               # 1-5
    dominance: Optional[int] = None             # 1-5

    @property
    def is_stress_proxy(self) -> bool:
        """Stress = low valence AND high arousal."""
        if self.valence is None or self.arousal is None:
            return False
        return (self.valence <= DREAMER_STRESS_VALENCE_THR and
                self.arousal >= DREAMER_STRESS_AROUSAL_THR)


@dataclass
class DREAMERSubject:
    """Container for one subject (all 18 trials)."""
    subject_id: int
    age: Optional[int] = None
    gender: Optional[str] = None
    trials: List[DREAMERTrial] = field(default_factory=list)


def load_dreamer(mat_path: Optional[Path] = None) -> List[DREAMERSubject]:
    """
    Load DREAMER.mat and return list of DREAMERSubject objects.

    The .mat file has structure:
        DREAMER.Data[0,s].EEG.stimuli[0,t]     → (n_samples, 14)
        DREAMER.Data[0,s].EEG.baseline[0,t]    → (n_baseline, 14)
        DREAMER.Data[0,s].ECG.stimuli[0,t]     → (n_samples, 2)
        DREAMER.Data[0,s].ECG.baseline[0,t]    → (n_baseline, 2)
        DREAMER.Data[0,s].Age                   → scalar
        DREAMER.Data[0,s].Gender                → string
        DREAMER.Data[0,s].ScoreValence[0,t]     → scalar 1-5
        DREAMER.Data[0,s].ScoreArousal[0,t]     → scalar 1-5
        DREAMER.Data[0,s].ScoreDominance[0,t]   → scalar 1-5
    """
    mat_path = mat_path or DREAMER_MAT_PATH
    if not mat_path.exists():
        raise FileNotFoundError(f"DREAMER.mat not found at {mat_path}")

    raw = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    dreamer_struct = raw["DREAMER"]
    data_arr = dreamer_struct.Data

    # data_arr may be 1-D array of subject structs
    if not hasattr(data_arr, "__len__"):
        data_arr = [data_arr]

    subjects = []
    for s_idx, subj_data in enumerate(data_arr):
        subj = DREAMERSubject(subject_id=s_idx + 1)

        # Demographics
        try:
            subj.age = int(subj_data.Age)
        except Exception:
            pass
        try:
            gender = subj_data.Gender
            if hasattr(gender, "item"):
                gender = gender.item()
            subj.gender = str(gender).strip()
        except Exception:
            pass

        # Trials
        eeg_stim = subj_data.EEG.stimuli
        eeg_base = subj_data.EEG.baseline
        ecg_stim = subj_data.ECG.stimuli
        ecg_base = subj_data.ECG.baseline

        valence_arr = subj_data.ScoreValence
        arousal_arr = subj_data.ScoreArousal
        dominance_arr = subj_data.ScoreDominance

        # Ensure iterable
        if not hasattr(eeg_stim, "__len__"):
            eeg_stim = [eeg_stim]
            eeg_base = [eeg_base]
            ecg_stim = [ecg_stim]
            ecg_base = [ecg_base]
            valence_arr = [valence_arr]
            arousal_arr = [arousal_arr]
            dominance_arr = [dominance_arr]

        n_trials = len(eeg_stim)
        for t_idx in range(n_trials):
            trial = DREAMERTrial(
                subject_id=s_idx + 1,
                trial_id=t_idx + 1,
            )
            try:
                trial.eeg_stimulus = np.asarray(eeg_stim[t_idx], dtype=np.float64)
                trial.eeg_baseline = np.asarray(eeg_base[t_idx], dtype=np.float64)
            except Exception:
                pass
            try:
                trial.ecg_stimulus = np.asarray(ecg_stim[t_idx], dtype=np.float64)
                trial.ecg_baseline = np.asarray(ecg_base[t_idx], dtype=np.float64)
            except Exception:
                pass

            # Labels – may be ndarray or scalar
            def _scalar(x, idx):
                if hasattr(x, "__len__"):
                    v = x[idx] if idx < len(x) else x[-1]
                else:
                    v = x
                if hasattr(v, "item"):
                    v = v.item()
                return int(v)

            try:
                trial.valence = _scalar(valence_arr, t_idx)
                trial.arousal = _scalar(arousal_arr, t_idx)
                trial.dominance = _scalar(dominance_arr, t_idx)
            except Exception:
                pass

            subj.trials.append(trial)

        subjects.append(subj)

    return subjects


def get_all_labels(subjects: List[DREAMERSubject]) -> Dict[str, np.ndarray]:
    """Extract all V/A/D labels as arrays."""
    valence, arousal, dominance = [], [], []
    for s in subjects:
        for t in s.trials:
            if t.valence is not None:
                valence.append(t.valence)
                arousal.append(t.arousal)
                dominance.append(t.dominance)
    return {
        "valence": np.array(valence),
        "arousal": np.array(arousal),
        "dominance": np.array(dominance),
    }


def get_stress_labels(subjects: List[DREAMERSubject]) -> np.ndarray:
    """Binary stress proxy: low valence + high arousal → 1, else 0."""
    labels = []
    for s in subjects:
        for t in s.trials:
            labels.append(1 if t.is_stress_proxy else 0)
    return np.array(labels)


def get_subject_demographics(subjects: List[DREAMERSubject]) -> pd.DataFrame:
    """Return demographics table."""
    import pandas as pd
    rows = []
    for s in subjects:
        rows.append({
            "subject_id": s.subject_id,
            "age": s.age,
            "gender": s.gender,
            "n_trials": len(s.trials),
        })
    return pd.DataFrame(rows)
