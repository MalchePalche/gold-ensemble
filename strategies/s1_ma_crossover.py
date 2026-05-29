"""
S1 — MA Crossover (Trend)

Fast EMA(20) vs Slow EMA(50).
  +1 when fast > slow
  -1 when fast < slow
Confidence scales with |fast - slow| / close (capped at ~1% spread = 1.0 conf).
"""

from __future__ import annotations
import pandas as pd
from .base import BaseStrategy, StrategyResult


class S1MACrossover(BaseStrategy):
    name = "S1"

    def __init__(self, fast: int = 20, slow: int = 50, **kw):
        super().__init__(fast=fast, slow=slow, **kw)
        self.fast = fast
        self.slow = slow

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        close = self._ensure_close(df)
        if len(close) < self.slow + 1:
            sig, conf = self._empty(df)
            return StrategyResult(self.name, sig, conf)

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        signal = pd.Series(0, index=df.index, dtype=float)
        signal[ema_fast > ema_slow] = 1.0
        signal[ema_fast < ema_slow] = -1.0

        # confidence: spread between EMAs as fraction of price, scaled
        # 1% spread = full confidence (1.0)
        spread_pct = (ema_fast - ema_slow).abs() / close.replace(0, pd.NA)
        confidence = self._clip01(spread_pct * 100.0)

        # only show confidence where there's a directional signal
        confidence = confidence.where(signal != 0, 0.0)

        return StrategyResult(self.name, signal.fillna(0), confidence.fillna(0))
