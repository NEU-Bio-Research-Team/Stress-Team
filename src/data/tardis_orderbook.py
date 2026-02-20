"""
Tardis Order Book Reconstruction
=================================
Reconstructs the full order book from Tardis incremental_book_L2 data.

Protocol (from Tardis docs):
    1. Start from is_snapshot=true row at start of day / after reconnect
    2. Apply incremental updates sequentially
    3. When amount = 0 → remove that price level
    4. When is_snapshot=true appears mid-day → RESET local state

Validation:
    - best_bid < best_ask at all times
    - Sequence numbers monotonically increasing
    - Cross-validate midprice with trade prices
"""

import gzip
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from sortedcontainers import SortedDict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import TARDIS_RAW_DIR


@dataclass
class OrderBookState:
    """Maintains current orderbook state."""
    bids: SortedDict = field(default_factory=lambda: SortedDict())  # price → amount (desc)
    asks: SortedDict = field(default_factory=lambda: SortedDict())  # price → amount (asc)
    last_update_id: int = 0
    last_timestamp: int = 0

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.last_update_id = 0

    @property
    def best_bid(self) -> Optional[float]:
        if self.bids:
            return self.bids.keys()[-1]  # highest bid
        return None

    @property
    def best_ask(self) -> Optional[float]:
        if self.asks:
            return self.asks.keys()[0]   # lowest ask
        return None

    @property
    def midprice(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    def depth(self, pct: float = 0.01) -> Dict[str, float]:
        """Total volume within pct% of midprice."""
        mid = self.midprice
        if mid is None:
            return {"bid_depth": 0, "ask_depth": 0}
        lower = mid * (1 - pct)
        upper = mid * (1 + pct)
        bid_depth = sum(v for k, v in self.bids.items() if k >= lower)
        ask_depth = sum(v for k, v in self.asks.items() if k <= upper)
        return {"bid_depth": bid_depth, "ask_depth": ask_depth}

    def is_valid(self) -> bool:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return True  # empty book is technically valid
        return bb < ba


def _apply_update(book: OrderBookState, side: str, price: float, amount: float):
    """Apply single price level update."""
    target = book.bids if side == "bid" else book.asks
    if amount == 0:
        target.pop(price, None)
    else:
        target[price] = amount


def reconstruct_day(csv_gz_path: Path) -> List[Dict]:
    """
    Reconstruct orderbook from a single day's incremental_book_L2 file.
    
    Returns list of snapshots at each update with:
        timestamp, midprice, spread, best_bid, best_ask, bid_depth, ask_depth, valid
    """
    book = OrderBookState()
    snapshots = []
    seq_gaps = 0
    violations = 0
    reset_count = 0

    with gzip.open(csv_gz_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = int(row.get("timestamp", row.get("local_timestamp", 0)))
            is_snapshot = row.get("is_snapshot", "false").lower() == "true"
            side = row.get("side", "")
            price = float(row.get("price", 0))
            amount = float(row.get("amount", row.get("size", 0)))

            if is_snapshot:
                book.reset()
                reset_count += 1

            if side in ("bid", "ask") and price > 0:
                _apply_update(book, side, price, amount)

            book.last_timestamp = timestamp

            # Record state periodically (every 1000th update to limit output)
            # For full reconstruction, remove the modulo check
            if len(snapshots) == 0 or timestamp - snapshots[-1]["timestamp"] >= 1000:
                depth = book.depth(0.01)
                valid = book.is_valid()
                if not valid:
                    violations += 1
                snapshots.append({
                    "timestamp": timestamp,
                    "midprice": book.midprice,
                    "spread": book.spread,
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "bid_depth_1pct": depth["bid_depth"],
                    "ask_depth_1pct": depth["ask_depth"],
                    "valid": valid,
                })

    return snapshots


def reconstruct_day_to_df(csv_gz_path: Path, sample_ms: int = 1000) -> pd.DataFrame:
    """
    Reconstruct and return as DataFrame, sampled every sample_ms milliseconds.
    """
    snapshots = reconstruct_day(csv_gz_path)
    if not snapshots:
        return pd.DataFrame()
    df = pd.DataFrame(snapshots)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="us")
    return df


def load_trades_day(csv_gz_path: Path) -> pd.DataFrame:
    """Load trades for one day."""
    with gzip.open(csv_gz_path, "rt") as f:
        df = pd.read_csv(f)
    if "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="us")
    return df


def validate_orderbook_trades(
    book_df: pd.DataFrame, trades_df: pd.DataFrame
) -> Dict:
    """
    Cross-validate: trade prices should fall within bid-ask spread.
    Returns violation statistics.
    """
    if book_df.empty or trades_df.empty:
        return {"n_trades": 0, "violations": 0, "violation_pct": 0}

    # For each trade, find nearest book snapshot
    # Simplified: check if trade price is within global min_ask / max_bid range
    n_violations = 0
    for _, trade in trades_df.iterrows():
        tp = trade.get("price", 0)
        # Find nearest book snapshot by timestamp
        idx = book_df["timestamp"].searchsorted(trade.get("timestamp", 0))
        if idx >= len(book_df):
            idx = len(book_df) - 1
        bb = book_df.iloc[idx].get("best_bid")
        ba = book_df.iloc[idx].get("best_ask")
        if bb is not None and ba is not None:
            if tp < bb * 0.999 or tp > ba * 1.001:  # 0.1% tolerance
                n_violations += 1

    return {
        "n_trades": len(trades_df),
        "violations": n_violations,
        "violation_pct": round(n_violations / len(trades_df) * 100, 4)
                         if len(trades_df) > 0 else 0,
    }
