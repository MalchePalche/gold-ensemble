"""
data/macro.py — real-yields macro layer for the gold ensemble.

Real yields (nominal Treasury yield minus inflation expectations) are gold's
single strongest macro driver: falling real yields are bullish for gold, rising
real yields bearish. This layer pulls the 10-year TIPS yield (a direct real-rate
read), the 10-year nominal yield, the 10-year breakeven inflation rate and the
Fed Funds rate from FRED (free, no key via the public CSV endpoint), classifies
the regime by both level and trend, and returns a small confidence adjustment
for the V4 ensemble (capped at +/- 8%, in line with the CB layer).

If FRED is unavailable every entry point degrades gracefully to a neutral /
zero-adjustment result so the dashboard and daily runner never break.
"""

import io

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# FRED API (free, no key needed for these series via direct URL)
# Series used:
#   DGS10    = 10-year nominal Treasury yield
#   DFII10   = 10-year TIPS yield (real yield directly)
#   T10YIE   = 10-year breakeven inflation rate
#   FEDFUNDS = Fed Funds effective rate

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
CACHE_TTL_HOURS = 6

_cache = {"data": None, "fetched_at": None}


# FRED's WAF can 504/stall the default urllib User-Agent used by pd.read_csv,
# and read_csv(url) takes no timeout — a slow FRED would hang the whole daily
# run. Fetch via requests with a browser UA + a bounded timeout (matching the
# central_banks layer), retry once on failure, then parse the CSV text locally.
_FRED_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_FRED_TIMEOUT = 15


def _fetch_fred_series(series_id: str, days: int = 90) -> pd.Series | None:
    """Fetch a single FRED series, return as pd.Series indexed by date."""
    url = f"{FRED_BASE}{series_id}"
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            r = requests.get(url, headers=_FRED_HEADERS, timeout=_FRED_TIMEOUT)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["DATE"],
                             index_col="DATE")
            df = df[df.iloc[:, 0] != "."]  # remove missing value markers
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            df = df.dropna()
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            return df.iloc[:, 0][df.index >= cutoff]
        except Exception as e:
            last_err = e
    print(f"[macro] FRED fetch failed for {series_id}: {last_err}")
    return None


def _classify_real_yield_regime(
    real_yield: float,
    real_yield_1m_ago: float | None,
    real_yield_3m_ago: float | None
) -> dict:
    """
    Classify real yield regime and direction.

    Real yield levels for gold:
      < 0%   = strongly bullish (negative real rates)
      0-1%   = neutral-bullish
      1-2%   = neutral-bearish
      > 2%   = strongly bearish

    Direction (trend) matters as much as level:
      falling = bullish regardless of level
      rising  = bearish regardless of level
    """
    level_signal = "NEUTRAL"
    if real_yield < 0.0:
        level_signal = "STRONGLY_BULLISH"
    elif real_yield < 1.0:
        level_signal = "BULLISH"
    elif real_yield < 2.0:
        level_signal = "BEARISH"
    else:
        level_signal = "STRONGLY_BEARISH"

    trend = "FLAT"
    if real_yield_1m_ago is not None:
        delta_1m = real_yield - real_yield_1m_ago
        if delta_1m < -0.15:
            trend = "FALLING"
        elif delta_1m > 0.15:
            trend = "RISING"

    momentum = "NEUTRAL"
    if real_yield_3m_ago is not None:
        delta_3m = real_yield - real_yield_3m_ago
        if delta_3m < -0.30:
            momentum = "FALLING_STRONG"
        elif delta_3m < -0.10:
            momentum = "FALLING"
        elif delta_3m > 0.30:
            momentum = "RISING_STRONG"
        elif delta_3m > 0.10:
            momentum = "RISING"

    return {
        "level_signal":  level_signal,
        "trend":         trend,
        "momentum":      momentum,
    }


def get_macro_analysis(current_bias: str) -> dict:
    """
    Main entry point. Fetches real yields + fed funds rate,
    classifies macro regime, returns confidence adjustment.

    Adjustment logic (max ±8%):
      Real yield falling + level bullish + bias BULLISH  → +8%
      Real yield falling + level bullish + bias BEARISH  → -8%
      Real yield rising  + level bearish + bias BULLISH  → -8%
      Real yield rising  + level bearish + bias BEARISH  → +8%
      Mixed signals (level vs trend disagree)            → ±4%
      Flat/neutral                                       → 0%

    Returns dict with keys: analysis, adjustment, error
    """
    global _cache

    # Cache check (6h TTL — FRED updates daily)
    if (_cache["data"] is not None and _cache["fetched_at"] is not None and
            datetime.utcnow() - _cache["fetched_at"] < timedelta(hours=CACHE_TTL_HOURS)):
        cached = _cache["data"]
        real_yield_series = cached["real_yield"]
        nominal_series    = cached["nominal"]
        breakeven_series  = cached["breakeven"]
        fedfunds_series   = cached["fedfunds"]
    else:
        real_yield_series = _fetch_fred_series("DFII10", days=120)
        nominal_series    = _fetch_fred_series("DGS10",  days=120)
        breakeven_series  = _fetch_fred_series("T10YIE", days=120)
        fedfunds_series   = _fetch_fred_series("FEDFUNDS", days=120)
        _cache = {
            "data": {
                "real_yield": real_yield_series,
                "nominal":    nominal_series,
                "breakeven":  breakeven_series,
                "fedfunds":   fedfunds_series,
            },
            "fetched_at": datetime.utcnow()
        }

    # Graceful failure
    if real_yield_series is None or len(real_yield_series) < 5:
        return {
            "analysis": {
                "real_yield": None, "nominal": None,
                "breakeven": None, "fedfunds": None,
                "regime": "UNKNOWN", "trend": "UNKNOWN",
                "signal": "NEUTRAL", "summary": "Real yield data unavailable"
            },
            "adjustment": {"adjustment": 0.0, "reason": "Macro data unavailable"},
            "error": "FRED data fetch failed"
        }

    # Current values
    real_yield  = float(real_yield_series.iloc[-1])
    nominal     = float(nominal_series.iloc[-1])    if nominal_series    is not None and len(nominal_series)    > 0 else None
    breakeven   = float(breakeven_series.iloc[-1])  if breakeven_series  is not None and len(breakeven_series)  > 0 else None
    fedfunds    = float(fedfunds_series.iloc[-1])   if fedfunds_series   is not None and len(fedfunds_series)   > 0 else None

    # Historical comparison points
    def _get_n_days_ago(series: pd.Series, days: int) -> float | None:
        try:
            target = series.index[-1] - pd.Timedelta(days=days)
            idx = series.index.get_indexer([target], method="nearest")[0]
            return float(series.iloc[idx])
        except Exception:
            return None

    ry_1m_ago = _get_n_days_ago(real_yield_series, 30)
    ry_3m_ago = _get_n_days_ago(real_yield_series, 90)

    regime = _classify_real_yield_regime(real_yield, ry_1m_ago, ry_3m_ago)
    level_signal = regime["level_signal"]
    trend        = regime["trend"]
    momentum     = regime["momentum"]

    # Compute adjustment
    bullish_level = level_signal in ("STRONGLY_BULLISH", "BULLISH")
    bearish_level = level_signal in ("STRONGLY_BEARISH", "BEARISH")
    falling       = trend in ("FALLING",) or momentum in ("FALLING_STRONG", "FALLING")
    rising        = trend in ("RISING",)  or momentum in ("RISING_STRONG",  "RISING")

    adjustment = 0.0
    reason     = "Neutral macro environment"

    if bullish_level and falling:
        adjustment = 8.0 if current_bias == "BULLISH" else -8.0
        reason = f"Real yields negative/low AND falling → strong gold tailwind"
    elif bearish_level and rising:
        adjustment = -8.0 if current_bias == "BULLISH" else 8.0
        reason = f"Real yields high AND rising → strong gold headwind"
    elif bullish_level or falling:
        adjustment = 4.0 if current_bias == "BULLISH" else -4.0
        reason = f"Partial bullish macro signal (level or trend)"
    elif bearish_level or rising:
        adjustment = -4.0 if current_bias == "BULLISH" else 4.0
        reason = f"Partial bearish macro signal (level or trend)"

    delta_str = ""
    if ry_1m_ago is not None:
        delta_str = f" ({real_yield - ry_1m_ago:+.2f}% vs 1m ago)"

    summary = (
        f"Real yield {real_yield:.2f}%{delta_str} | "
        f"Nominal {nominal:.2f}% | Breakeven {breakeven:.2f}% | "
        f"Fed Funds {fedfunds:.2f}% | "
        f"Regime: {level_signal} / Trend: {trend}"
    )

    return {
        "analysis": {
            "real_yield":   round(real_yield, 3),
            "nominal":      round(nominal,   3) if nominal   is not None else None,
            "breakeven":    round(breakeven, 3) if breakeven is not None else None,
            "fedfunds":     round(fedfunds,  3) if fedfunds  is not None else None,
            "regime":       level_signal,
            "trend":        trend,
            "momentum":     momentum,
            "signal":       "BULLISH" if adjustment > 0 else ("BEARISH" if adjustment < 0 else "NEUTRAL"),
            "summary":      summary
        },
        "adjustment": {
            "adjustment": round(adjustment, 1),
            "reason":     reason
        },
        "error": None
    }
