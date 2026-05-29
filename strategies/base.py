"""
strategies/base.py

Common interface for all 10 strategies.

Every strategy:
- Accepts a DataFrame of OHLCV
- Returns (signal_series, confidence_series) of identical length & index
- Signals are computed on bar close — the backtest enforces next-bar-open
  execution, so strategies should NOT shift signals themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """Container for what every strategy returns."""
    name: str
    signal: pd.Series          # +1 / 0 / -1
    confidence: pd.Series      # 0.0–1.0

    def latest(self) -> Tuple[int, float]:
        """Return the most recent (signal, confidence) pair."""
        if self.signal.empty:
            return 0, 0.0
        s = self.signal.iloc[-1]
        c = self.confidence.iloc[-1] if not self.confidence.empty else 0.0
        return int(s) if pd.notna(s) else 0, float(c) if pd.notna(c) else 0.0


class BaseStrategy(ABC):
    """Subclass and implement `generate`."""

    name: str = "BASE"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> StrategyResult:
        """Return a StrategyResult for the supplied OHLCV frame."""
        raise NotImplementedError

    # ---- shared helpers ----
    @staticmethod
    def _empty(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        idx = df.index
        return (pd.Series(0, index=idx, dtype=float),
                pd.Series(0.0, index=idx, dtype=float))

    @staticmethod
    def _clip01(s: pd.Series) -> pd.Series:
        return s.clip(lower=0.0, upper=1.0).fillna(0.0)

    @staticmethod
    def _ensure_close(df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        return df["close"].astype(float)
