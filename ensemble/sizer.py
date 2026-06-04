"""
ensemble/sizer.py — V4 vol regime detection, sizing matrix, and simulation.

These functions are extracted here so run_daily.py and any other production
code can import them without depending on the v4.py experiment script.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── Layer 1: volatility regime ─────────────────────────────────────────────

def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _iv_rv_spread(iv: float | None, rv: float | None) -> dict:
    """
    Returns spread signal when IV > RV by meaningful margin.
    spread > 0.05 (5pp annualized) = elevated fear premium
    spread > 0.15 = extreme fear premium
    """
    if iv is None or rv is None:
        return {"spread": None, "signal": "NEUTRAL", "adjustment": 0.0}
    spread = iv - rv
    if spread > 0.15:
        return {"spread": round(spread, 4), "signal": "RISK_OFF_EXTREME", "adjustment": -10.0}
    elif spread > 0.05:
        return {"spread": round(spread, 4), "signal": "RISK_OFF", "adjustment": -5.0}
    elif spread < -0.05:
        return {"spread": round(spread, 4), "signal": "RISK_ON", "adjustment": +5.0}
    else:
        return {"spread": round(spread, 4), "signal": "NEUTRAL", "adjustment": 0.0}


def compute_vol_regime(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_period: int = 14,
    atr_avg_period: int = 20,
    atr_elevated_mult: float = 1.5,
    atr_extreme_mult: float = 2.0,
    rv_period: int = 20,
    rv_lookback: int = 252,
    rv_high_pct: float = 0.75,
    rv_extreme_pct: float = 0.90,
    implied_vol: float | None = None,   # current IV (from options, annualized)
    realized_vol: float | None = None,  # current RV (annualized, already computed)
) -> Tuple[pd.Series, pd.Series, pd.Series, dict]:
    """
    Returns (regime, atr_ratio, rv_percentile, iv_rv).
    regime values: 'normal' | 'elevated' | 'extreme'
    iv_rv: IV-vs-RV spread signal (see _iv_rv_spread); NEUTRAL/zero-adjustment
    when implied_vol or realized_vol is not supplied.
    """
    atr       = _wilder_atr(high, low, close, atr_period)
    avg_atr   = atr.rolling(atr_avg_period).mean()
    atr_ratio = (atr / avg_atr.replace(0, np.nan)).fillna(1.0)

    daily_ret = close.pct_change()
    rv        = daily_ret.rolling(rv_period).std(ddof=0) * np.sqrt(252)
    rv_pct    = rv.rolling(rv_lookback, min_periods=60).rank(pct=True).fillna(0.5)

    is_extreme  = (atr_ratio > atr_extreme_mult)  | (rv_pct > rv_extreme_pct)
    is_elevated = (atr_ratio > atr_elevated_mult) | (rv_pct > rv_high_pct)

    regime = pd.Series("normal", index=close.index, dtype=object)
    regime[is_elevated] = "elevated"
    regime[is_extreme]  = "extreme"

    iv_rv = _iv_rv_spread(implied_vol, realized_vol)

    return regime, atr_ratio, rv_pct, iv_rv


# ── Layer 2: sizing matrix ─────────────────────────────────────────────────

def target_size(
    sig: int,
    conf: float,
    vol: str,
    bullish_high_normal: float   = 1.5,
    bullish_high_elevated: float = 1.0,
    bullish_high_extreme: float  = 0.5,
    bullish_base_normal: float   = 1.0,
    bullish_base_elevated: float = 0.75,
    bullish_base_extreme: float  = 0.25,
    neutral: float               = 0.5,
    bearish: float               = 0.0,
    conf_high_threshold: float   = 60.0,
) -> float:
    """Return target position size from the 6-tier matrix."""
    if sig == -1:
        return bearish
    if sig == 0:
        return neutral
    if conf > conf_high_threshold:
        return {"normal": bullish_high_normal, "elevated": bullish_high_elevated,
                "extreme": bullish_high_extreme}[vol]
    return {"normal": bullish_base_normal, "elevated": bullish_base_elevated,
            "extreme": bullish_base_extreme}[vol]


# ── Layers 3 + 4: circuit breaker + entry smoothing ───────────────────────

def simulate_v4(
    signal: pd.Series,
    conf_pct: pd.Series,
    vol_regime: pd.Series,
    bar_returns: pd.Series,
    tc: float = 0.001,
    cb_threshold: float = -0.08,
    cb_freeze_bars: int = 10,
    sizing_kwargs: dict | None = None,
) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """
    Bar-by-bar V4 simulation with all 4 layers.

    Returns (net_returns, executed_positions, cb_log).
    Execution lag: position decided at bar-close t executes at bar t+1.
    """
    skw = sizing_kwargs or {}
    n            = len(signal)
    intended     = np.zeros(n)
    actual       = np.zeros(n)
    net          = np.zeros(n)
    cb_active    = False
    cb_remaining = 0
    cb_log: List[Dict] = []
    sm_mode      = None
    sm_day       = 0
    sm_out_init  = 0.0
    prev_sig     = 0

    for i in range(n):
        actual[i] = intended[i - 1] if i > 0 else 0.0

        prev_actual = actual[i - 1] if i > 0 else 0.0
        cost_i      = abs(actual[i] - prev_actual) * tc
        net[i]      = actual[i] * bar_returns.iloc[i] - cost_i

        if i >= 19 and not cb_active:
            roll_ret = float(np.prod(1.0 + net[i - 19 : i + 1]) - 1.0)
            if roll_ret < cb_threshold:
                cb_active    = True
                cb_remaining = cb_freeze_bars
                cb_log.append({
                    "date"        : signal.index[i],
                    "20d_return"  : round(roll_ret * 100, 2),
                    "triggered_by": f"20d return {roll_ret:.1%}",
                })

        sig_i  = int(signal.iloc[i])
        conf_i = float(conf_pct.iloc[i])
        vol_i  = vol_regime.iloc[i]
        tgt    = target_size(sig_i, conf_i, vol_i, **skw)

        if sig_i == 1 and prev_sig != 1:
            sm_mode     = "in"
            sm_day      = 1
        elif sig_i != 1 and prev_sig == 1:
            sm_mode     = "out"
            sm_day      = 1
            sm_out_init = intended[i - 1] if i > 0 else 0.0

        if sm_mode == "in":
            if sm_day == 1:
                smoothed = min(0.5, tgt)
            elif sm_day == 2:
                smoothed = min(1.0, tgt)
            else:
                smoothed = tgt
            sm_day += 1
            if sm_day > 3:
                sm_mode = None
        elif sm_mode == "out":
            smoothed = sm_out_init * 0.5 if sm_day == 1 else 0.0
            sm_day  += 1
            if sm_day > 2:
                sm_mode = None
        else:
            smoothed = tgt

        if cb_active:
            intended[i]   = 0.0
            cb_remaining -= 1
            if cb_remaining <= 0:
                cb_active    = False
                cb_remaining = 0
                sm_mode      = None
        else:
            intended[i] = max(smoothed, 0.0)

        prev_sig = sig_i

    return (
        pd.Series(net,    index=signal.index),
        pd.Series(actual, index=signal.index),
        cb_log,
    )
