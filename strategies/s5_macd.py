"""
S5 — MACD Signal Cross (Momentum)

MACD(12, 26, 9).
  Long  when MACD line crosses ABOVE signal line AND both > 0
  Short when MACD line crosses BELOW signal line AND both < 0

Position is held until the opposite cross. Confidence is 1.0 on the
cross bar and decays by 0.05 per bar while in position (floor 0.1).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseStrategy, StrategyResult


class S5MACD(BaseStrategy):
    name = "S5"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **kw):
        super().__init__(fast=fast, slow=slow, signal=signal, **kw)
        self.fast = fast
        self.slow = slow
        self.sig_p = signal

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        close = self._ensure_close(df)
        if len(close) < self.slow + self.sig_p + 2:
            return StrategyResult(self.name, *self._empty(df))

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.sig_p, adjust=False).mean()

        diff = macd - signal_line
        prev_diff = diff.shift(1)

        # both lines above/below zero confirms the regime
        bull_cross = (prev_diff <= 0) & (diff > 0) & (macd > 0) & (signal_line > 0)
        bear_cross = (prev_diff >= 0) & (diff < 0) & (macd < 0) & (signal_line < 0)

        position = np.zeros(len(df), dtype=float)
        bars_in = np.zeros(len(df), dtype=int)
        for i in range(1, len(df)):
            prev = position[i - 1]
            new_pos = prev
            new_bars = bars_in[i - 1] + 1 if prev != 0 else 0
            if bull_cross.iloc[i]:
                new_pos, new_bars = 1.0, 0
            elif bear_cross.iloc[i]:
                new_pos, new_bars = -1.0, 0
            # exit on opposite zero-line cross
            elif prev == 1.0 and diff.iloc[i] < 0:
                new_pos, new_bars = 0.0, 0
            elif prev == -1.0 and diff.iloc[i] > 0:
                new_pos, new_bars = 0.0, 0
            position[i] = new_pos
            bars_in[i] = new_bars

        signal = pd.Series(position, index=df.index)
        confidence = pd.Series(
            np.where(position == 0, 0.0, np.maximum(1.0 - 0.05 * bars_in, 0.1)),
            index=df.index,
        )
        return StrategyResult(self.name, signal, self._clip01(confidence))
