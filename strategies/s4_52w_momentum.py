"""
S4 — 52-Week High / Low Momentum

George & Hwang (2004) — assets near 52-week highs continue higher more
often than they reverse. Same anomaly is documented on commodities.

Long  when close is within 2% of the trailing 252-day high
Short when close is within 2% of the trailing 252-day low

A signal is held for `hold_bars` (default 10) unless invalidated — i.e.
price moves more than `tolerance` away from the extreme during the hold.

Confidence is 1.0 at the exact high/low and decays linearly to 0 at the
2% boundary.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseStrategy, StrategyResult


class S452WeekMomentum(BaseStrategy):
    name = "S4"

    def __init__(self, lookback: int = 252, tolerance: float = 0.02, hold_bars: int = 10, **kw):
        super().__init__(lookback=lookback, tolerance=tolerance, hold_bars=hold_bars, **kw)
        self.lookback = lookback
        self.tolerance = tolerance
        self.hold_bars = hold_bars

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        if "close" not in df.columns:
            return StrategyResult(self.name, *self._empty(df))
        if len(df) < self.lookback + 2:
            return StrategyResult(self.name, *self._empty(df))

        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close

        # Use prior 252 bars (shift(1)) — today's bar must not be in the window
        roll_high = high.shift(1).rolling(self.lookback).max()
        roll_low = low.shift(1).rolling(self.lookback).min()

        near_high = close >= roll_high * (1.0 - self.tolerance)
        near_low = close <= roll_low * (1.0 + self.tolerance)

        n = len(df)
        position = np.zeros(n, dtype=float)
        confidence = np.zeros(n, dtype=float)
        hold_remaining = 0
        active_dir = 0

        for i in range(n):
            rh = roll_high.iloc[i]
            rl = roll_low.iloc[i]
            c = close.iloc[i]
            if pd.isna(rh) or pd.isna(rl):
                continue

            # decide fresh entry
            if hold_remaining <= 0:
                if near_high.iloc[i]:
                    active_dir = 1
                    hold_remaining = self.hold_bars
                elif near_low.iloc[i]:
                    active_dir = -1
                    hold_remaining = self.hold_bars
                else:
                    active_dir = 0

            # invalidate if price moves > 2x tolerance from extreme
            if active_dir == 1 and c < rh * (1.0 - 2 * self.tolerance):
                active_dir = 0
                hold_remaining = 0
            elif active_dir == -1 and c > rl * (1.0 + 2 * self.tolerance):
                active_dir = 0
                hold_remaining = 0

            position[i] = active_dir

            # confidence: linear with proximity to the extreme
            if active_dir == 1:
                dist = max((rh - c) / rh, 0.0)
                confidence[i] = max(1.0 - dist / self.tolerance, 0.0)
            elif active_dir == -1:
                dist = max((c - rl) / rl, 0.0)
                confidence[i] = max(1.0 - dist / self.tolerance, 0.0)

            if hold_remaining > 0:
                hold_remaining -= 1

        return StrategyResult(
            self.name,
            pd.Series(position, index=df.index),
            self._clip01(pd.Series(confidence, index=df.index)),
        )
