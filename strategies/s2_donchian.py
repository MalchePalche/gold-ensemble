"""
S2 — Donchian Channel Breakout (Turtle-style)

Long entry: close above prior 55-day high
Short entry: close below prior 55-day low
Exit: opposite 20-day channel

Confidence: 1.0 on the breakout bar, decays by 0.1 per bar afterward
(floors at 0.1 while position remains open).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseStrategy, StrategyResult


class S2Donchian(BaseStrategy):
    name = "S2"

    def __init__(self, lookback: int = 55, exit_lookback: int = 20, **kw):
        super().__init__(lookback=lookback, exit_lookback=exit_lookback, **kw)
        self.lookback = lookback
        self.exit_lookback = exit_lookback

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        if not {"high", "low", "close"}.issubset(df.columns):
            return StrategyResult(self.name, *self._empty(df))
        if len(df) < self.lookback + 2:
            return StrategyResult(self.name, *self._empty(df))

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        # Use prior bars only — shift(1) so today's close is compared to
        # yesterday's window (avoids using today's own H/L)
        roll_high = high.shift(1).rolling(self.lookback).max()
        roll_low = low.shift(1).rolling(self.lookback).min()
        exit_high = high.shift(1).rolling(self.exit_lookback).max()
        exit_low = low.shift(1).rolling(self.exit_lookback).min()

        position = np.zeros(len(df), dtype=float)
        bars_in = np.zeros(len(df), dtype=int)

        for i in range(1, len(df)):
            prev_pos = position[i - 1]
            prev_bars = bars_in[i - 1]
            c = close.iloc[i]
            new_pos = prev_pos
            new_bars = prev_bars + 1 if prev_pos != 0 else 0

            if prev_pos == 0:
                # entry only on fresh breakout
                if pd.notna(roll_high.iloc[i]) and c > roll_high.iloc[i]:
                    new_pos = 1.0
                    new_bars = 0
                elif pd.notna(roll_low.iloc[i]) and c < roll_low.iloc[i]:
                    new_pos = -1.0
                    new_bars = 0
            elif prev_pos == 1.0:
                # exit when close below 20-day low
                if pd.notna(exit_low.iloc[i]) and c < exit_low.iloc[i]:
                    new_pos = 0.0
                    new_bars = 0
            elif prev_pos == -1.0:
                if pd.notna(exit_high.iloc[i]) and c > exit_high.iloc[i]:
                    new_pos = 0.0
                    new_bars = 0

            position[i] = new_pos
            bars_in[i] = new_bars

        signal = pd.Series(position, index=df.index)
        confidence = pd.Series(
            np.where(position == 0, 0.0, np.maximum(1.0 - 0.1 * bars_in, 0.1)),
            index=df.index,
        )
        return StrategyResult(self.name, signal, self._clip01(confidence))
