"""
ensemble/model.py — Weighted voting machine (V4 production).

Active strategies: S1 (MA Crossover), S2 (Donchian), S4 (52-week momentum), S5 (MACD).
Weights are locked from V3 grid-search optimization — no regime modifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from strategies import (
    S1MACrossover, S2Donchian, S452WeekMomentum, S5MACD, StrategyResult,
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "S1": 1.5, "S2": 0.5, "S4": 1.5, "S5": 2.0,
}


@dataclass
class EnsembleOutput:
    """Latest snapshot from the ensemble (single bar)."""
    date: pd.Timestamp
    bias: str
    confidence_pct: float
    final_score: float
    active_signals: List[str] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    per_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class EnsembleSeries:
    """Full timeseries output for use by the backtest and daily runner."""
    score: pd.Series
    signal: pd.Series
    confidence_pct: pd.Series
    per_strategy: Dict[str, StrategyResult]


class EnsembleModel:
    """The weighted voting machine."""

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        dead_zone: float = 0.05,
        strategy_params: dict | None = None,
    ):
        self.weights  = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
        self.dead_zone = dead_zone
        sp = strategy_params or {}

        self.strategies = {
            "S1": S1MACrossover(**sp.get("s1_ma", {})),
            "S2": S2Donchian(**sp.get("s2_donchian", {})),
            "S4": S452WeekMomentum(**sp.get("s4_52w", {})),
            "S5": S5MACD(**sp.get("s5_macd", {})),
        }

    def run(self, gold_df: pd.DataFrame) -> EnsembleSeries:
        """Compute ensemble signal across the entire frame."""
        results: Dict[str, StrategyResult] = {
            k: s.generate(gold_df) for k, s in self.strategies.items()
        }

        idx        = gold_df.index
        score      = pd.Series(0.0, index=idx)
        weight_sum = pd.Series(0.0, index=idx)

        for k, w in self.weights.items():
            if k not in results:
                continue
            r = results[k]
            contrib    = r.signal * r.confidence * w
            score      = score.add(contrib, fill_value=0.0)
            active_w   = pd.Series(w, index=idx, dtype=float).where(r.signal != 0, 0.0)
            weight_sum = weight_sum.add(active_w, fill_value=0.0)

        normalized = (score / weight_sum.replace(0, np.nan)).fillna(0.0)

        total_possible = sum(self.weights.values())
        participation  = (weight_sum / total_possible).clip(0, 1).fillna(0.0)
        magnitude      = normalized.abs() * np.sqrt(participation)

        signal = pd.Series(0, index=idx, dtype=int)
        signal[normalized >  self.dead_zone] =  1
        signal[normalized < -self.dead_zone] = -1

        confidence_pct = (magnitude * 100.0).clip(0, 100)

        return EnsembleSeries(
            score=normalized,
            signal=signal,
            confidence_pct=confidence_pct,
            per_strategy=results,
        )

    def latest(self, gold_df: pd.DataFrame) -> EnsembleOutput:
        """Convenience: most recent bar as a clean snapshot."""
        series = self.run(gold_df)
        if series.score.empty:
            return EnsembleOutput(date=pd.NaT, bias="NEUTRAL",
                                  confidence_pct=0.0, final_score=0.0)

        i     = -1
        date  = series.score.index[i]
        score = float(series.score.iloc[i])
        sig   = int(series.signal.iloc[i])
        bias  = "BULLISH" if sig > 0 else ("BEARISH" if sig < 0 else "NEUTRAL")
        conf  = float(series.confidence_pct.iloc[i])

        active, conflicting = [], []
        per_strat: Dict[str, Dict[str, float]] = {}
        for k, r in series.per_strategy.items():
            s_v = int(r.signal.iloc[i])   if not r.signal.empty     else 0
            c_v = float(r.confidence.iloc[i]) if not r.confidence.empty else 0.0
            per_strat[k] = {"signal": s_v, "confidence": round(c_v, 3)}
            if s_v == sig and sig != 0:
                active.append(k)
            elif s_v == -sig and sig != 0:
                conflicting.append(k)

        return EnsembleOutput(
            date=date, bias=bias,
            confidence_pct=round(conf, 1), final_score=round(score, 4),
            active_signals=active, conflicting_signals=conflicting,
            per_strategy=per_strat,
        )
