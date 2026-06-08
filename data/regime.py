import numpy as np
import pandas as pd

# Regime states
TRENDING_UP   = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
RANGING       = "RANGING"

# ADX thresholds
ADX_TRENDING_STRONG  = 25.0   # ADX > 25 = trending
ADX_TRENDING_WEAK    = 20.0   # ADX 20-25 = weakly trending
ADX_RANGING          = 20.0   # ADX < 20 = ranging


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    delta_high = high.diff()
    delta_low  = -low.diff()

    plus_dm  = pd.Series(np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0.0), index=close.index)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _compute_regime_scores(
    close: pd.Series,
    adx: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    lookback: int = 20
) -> dict:
    """
    Score-based regime detection combining ADX + price structure.
    Returns scores for each regime state.
    """
    latest_adx   = float(adx.iloc[-1])
    latest_plus  = float(plus_di.iloc[-1])
    latest_minus = float(minus_di.iloc[-1])

    # Price structure: higher highs/lower lows over lookback
    recent_close = close.iloc[-lookback:]
    rolling_high = recent_close.rolling(5).max()
    rolling_low  = recent_close.rolling(5).min()
    hh_count = int((rolling_high.diff() > 0).sum())   # higher highs
    ll_count = int((rolling_low.diff()  < 0).sum())   # lower lows
    hl_count = int((rolling_low.diff()  > 0).sum())   # higher lows
    lh_count = int((rolling_high.diff() < 0).sum())   # lower highs

    # SMA slope
    sma_50   = close.rolling(50).mean()
    sma_slope = float((sma_50.iloc[-1] - sma_50.iloc[-10]) / sma_50.iloc[-10] * 100)

    trending_up_score = 0.0
    trending_dn_score = 0.0
    ranging_score     = 0.0

    # ADX component (most important)
    if latest_adx > ADX_TRENDING_STRONG:
        if latest_plus > latest_minus:
            trending_up_score += 3.0
        else:
            trending_dn_score += 3.0
    elif latest_adx > ADX_TRENDING_WEAK:
        if latest_plus > latest_minus:
            trending_up_score += 1.5
        else:
            trending_dn_score += 1.5
    else:
        ranging_score += 3.0

    # Price structure component
    if hh_count > lh_count and hl_count > ll_count:
        trending_up_score += 1.5
    elif ll_count > hl_count and lh_count > hh_count:
        trending_dn_score += 1.5
    else:
        ranging_score += 1.5

    # SMA slope component
    if sma_slope > 0.5:
        trending_up_score += 1.0
    elif sma_slope < -0.5:
        trending_dn_score += 1.0
    else:
        ranging_score += 1.0

    return {
        "adx":               round(latest_adx, 2),
        "plus_di":           round(latest_plus, 2),
        "minus_di":          round(latest_minus, 2),
        "sma_slope":         round(sma_slope, 3),
        "trending_up_score": round(trending_up_score, 1),
        "trending_dn_score": round(trending_dn_score, 1),
        "ranging_score":     round(ranging_score, 1),
    }


def classify_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    lookback: int = 20
) -> dict:
    """
    Main entry point. Accepts OHLCV DataFrame with columns:
      Open, High, Low, Close, Volume (case-insensitive)
    Returns regime classification + metadata + weight multipliers.

    Weight multipliers for ensemble:
      TRENDING_UP   → 1.0x (no penalty)
      TRENDING_DOWN → 1.0x (no penalty)
      RANGING       → 0.6x (reduce all signals — false signals common)

    Confidence adjustment:
      TRENDING_UP   + BULLISH bias  → +5%
      TRENDING_DOWN + BEARISH bias  → +5%
      TRENDING_UP   + BEARISH bias  → -5%
      TRENDING_DOWN + BULLISH bias  → -5%
      RANGING                       →  0% (no boost, no penalty on confidence)
    """
    try:
        # Normalise column names. yfinance returns MultiIndex columns
        # (field, ticker) for a single-ticker download, so flatten to the
        # field level before normalising case.
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).capitalize() for c in df.columns]

        high  = df["High"]
        low   = df["Low"]
        close = df["Close"]

        if len(close) < 60:
            return _ranging_fallback("Insufficient data for regime detection")

        adx, plus_di, minus_di = _compute_adx(high, low, close, adx_period)
        scores = _compute_regime_scores(close, adx, plus_di, minus_di, lookback)

        # Determine regime from scores
        max_score = max(
            scores["trending_up_score"],
            scores["trending_dn_score"],
            scores["ranging_score"]
        )

        if max_score == scores["trending_up_score"]:
            regime = TRENDING_UP
        elif max_score == scores["trending_dn_score"]:
            regime = TRENDING_DOWN
        else:
            regime = RANGING

        # Tie-break: if trending scores are equal, use DI
        if scores["trending_up_score"] == scores["trending_dn_score"]:
            regime = TRENDING_UP if scores["plus_di"] > scores["minus_di"] else TRENDING_DOWN

        weight_multiplier = 0.6 if regime == RANGING else 1.0

        return {
            "regime":           regime,
            "adx":              scores["adx"],
            "plus_di":          scores["plus_di"],
            "minus_di":         scores["minus_di"],
            "sma_slope":        scores["sma_slope"],
            "weight_multiplier": weight_multiplier,
            "scores":           scores,
            "error":            None
        }

    except Exception as e:
        return _ranging_fallback(f"Regime classification error: {e}")


def get_confidence_adjustment(regime: str, current_bias: str) -> float:
    """
    Returns confidence adjustment based on regime + bias alignment.
      TRENDING_UP   + BULLISH → +5%
      TRENDING_DOWN + BEARISH → +5%
      TRENDING_UP   + BEARISH → -5%
      TRENDING_DOWN + BULLISH → -5%
      RANGING                 →  0%
    """
    if regime == RANGING:
        return 0.0
    if regime == TRENDING_UP and current_bias == "BULLISH":
        return 5.0
    if regime == TRENDING_DOWN and current_bias == "BEARISH":
        return 5.0
    if regime == TRENDING_UP and current_bias == "BEARISH":
        return -5.0
    if regime == TRENDING_DOWN and current_bias == "BULLISH":
        return -5.0
    return 0.0


def _ranging_fallback(reason: str) -> dict:
    return {
        "regime":            RANGING,
        "adx":               None,
        "plus_di":           None,
        "minus_di":          None,
        "sma_slope":         None,
        "weight_multiplier": 0.6,
        "scores":            None,
        "error":             reason
    }
