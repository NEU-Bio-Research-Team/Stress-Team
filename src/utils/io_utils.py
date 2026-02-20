"""
I/O utilities – loading, saving, report generation.
"""
import json, csv, hashlib, gzip, pickle
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pickle(path: Path, encoding: str = "latin1") -> Any:
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding)


def save_pickle(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_csv_gz(path: Path) -> pd.DataFrame:
    """Load a gzip-compressed CSV."""
    return pd.read_csv(path, compression="gzip")


def save_audit_report(results: List[Dict], path: Path):
    """Save audit results as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    # Collect all unique keys across every row (order preserved, Python 3.7+)
    seen: dict = {}
    for row in results:
        seen.update(dict.fromkeys(row.keys()))
    keys = list(seen.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(results)
    print(f"[io] Audit report saved → {path}")


def print_audit_table(results: List[Dict]):
    """Pretty-print audit results."""
    if not results:
        print("  (no results)")
        return
    # Determine column widths
    keys = list(results[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in results))
              for k in keys}
    header = " | ".join(k.ljust(widths[k]) for k in keys)
    sep = "-+-".join("-" * widths[k] for k in keys)
    print(header)
    print(sep)
    for r in results:
        row = " | ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys)
        print(row)
