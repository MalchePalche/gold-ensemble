"""
data/options_flow.py — options / open-interest layer for the gold ensemble.

Pulls a live GLD options chain from yfinance, rolls it into institutional-
positioning metrics (OI- and volume-based put/call ratios, an IV skew, a net
positioning score, and an unusual-activity flag), then translates that into a
small confidence adjustment for the V4 ensemble signal.

GLD is used as the gold proxy: GC=F futures options need a paid data tier, and
GLD is the most liquid free options market that tracks gold positioning.
Polygon's free tier does not serve OI snapshots, so this layer reads the chain
from yfinance (no API key required).

If the chain is unavailable or the call fails, every entry point degrades
gracefully to a neutral / zero-adjustment result so the dashboard and daily
runner never break.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# Gold ETF — GLD is the most liquid options market for gold.
# GC=F futures options exist but need a paid tier; GLD options are the best
# free proxy for gold institutional positioning.
GLD_TICKER = "GLD"


def fetch_options_chain(ticker: str = GLD_TICKER) -> dict:
    """
    Fetch options chain via yfinance.
    Gets nearest 2 expiry dates for most liquid contracts.
    Returns calls and puts with OI and volume.
    """
    try:
        t = yf.Ticker(ticker)
        expirations = t.options

        if not expirations:
            return {"error": "No options data available"}

        # Use nearest 2 expirations — most liquid
        all_calls = []
        all_puts = []

        for exp in expirations[:2]:
            chain = t.option_chain(exp)
            calls = chain.calls[["strike", "openInterest",
                                 "volume", "impliedVolatility"]].copy()
            puts = chain.puts[["strike", "openInterest",
                               "volume", "impliedVolatility"]].copy()
            calls["contract_type"] = "call"
            puts["contract_type"] = "put"
            calls["expiry"] = exp
            puts["expiry"] = exp
            all_calls.append(calls)
            all_puts.append(puts)

        calls_df = pd.concat(all_calls, ignore_index=True)
        puts_df = pd.concat(all_puts, ignore_index=True)

        return {
            "calls": calls_df,
            "puts": puts_df,
            "expirations_used": list(expirations[:2]),
            "error": None,
        }

    except Exception as e:
        return {"error": str(e)}


def analyze_positioning(chain_data: dict) -> dict:
    """
    Analyze yfinance options chain.

    Metrics:
    1. Put/Call Ratio (OI-based) — > 1.2 bearish, < 0.8 bullish, else neutral
    2. Put/Call Ratio (Volume-based) — same thresholds, more responsive
    3. Net positioning score — OI (60%) + Volume (40%), in -1..+1
    4. Unusual activity flag — volume/OI ratio > 3 on liquid strikes
    5. IV skew — avg put IV minus avg call IV (positive = fear)
    """
    if chain_data.get("error"):
        return {
            "error": chain_data["error"],
            "score": 0,
            "signal": "NEUTRAL",
            "pcr_oi": None,
            "pcr_vol": None,
        }

    calls = chain_data["calls"]
    puts = chain_data["puts"]

    # Clean NaN
    calls = calls.fillna(0)
    puts = puts.fillna(0)

    total_call_oi = int(calls["openInterest"].sum())
    total_put_oi = int(puts["openInterest"].sum())
    total_call_vol = int(calls["volume"].sum())
    total_put_vol = int(puts["volume"].sum())

    # PCR
    pcr_oi = (total_put_oi / total_call_oi
              if total_call_oi > 0 else 1.0)
    pcr_vol = (total_put_vol / total_call_vol
               if total_call_vol > 0 else 1.0)

    # Unusual activity — volume/OI ratio > 3 (only on liquid strikes, OI > 100)
    calls["vol_oi_ratio"] = calls.apply(
        lambda r: r["volume"] / r["openInterest"]
        if r["openInterest"] > 100 else 0, axis=1)
    puts["vol_oi_ratio"] = puts.apply(
        lambda r: r["volume"] / r["openInterest"]
        if r["openInterest"] > 100 else 0, axis=1)

    unusual_calls = int((calls["vol_oi_ratio"] > 3).sum())
    unusual_puts = int((puts["vol_oi_ratio"] > 3).sum())

    unusual = ""
    if unusual_calls > 3:
        unusual = f"Unusual call buying ({unusual_calls} strikes)"
    elif unusual_puts > 3:
        unusual = f"Unusual put buying ({unusual_puts} strikes)"

    # Normalize PCR to -1..+1 score
    # PCR > 1.2 → bearish → negative score
    # PCR < 0.8 → bullish → positive score
    def pcr_to_score(pcr):
        if pcr > 1.5:
            return -1.0
        elif pcr > 1.2:
            return -0.5
        elif pcr < 0.6:
            return 1.0
        elif pcr < 0.8:
            return 0.5
        else:
            return 0.0

    oi_score = pcr_to_score(pcr_oi)
    vol_score = pcr_to_score(pcr_vol)

    # Weight: OI (60%) + Volume (40%)
    net_score = (oi_score * 0.6) + (vol_score * 0.4)

    # Signal
    if net_score > 0.3:
        signal = "BULLISH"
    elif net_score < -0.3:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    # Average IV — high IV = uncertainty/fear
    avg_iv_calls = float(calls["impliedVolatility"].mean())
    avg_iv_puts = float(puts["impliedVolatility"].mean())
    iv_skew = avg_iv_puts - avg_iv_calls  # positive = put premium = fear

    return {
        "error": None,
        "score": round(net_score, 3),
        "signal": signal,
        "pcr_oi": round(pcr_oi, 3),
        "pcr_vol": round(pcr_vol, 3),
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_call_vol": total_call_vol,
        "total_put_vol": total_put_vol,
        "unusual": unusual,
        "unusual_calls": unusual_calls,
        "unusual_puts": unusual_puts,
        "iv_skew": round(iv_skew, 4),
        "expirations": chain_data.get("expirations_used", []),
    }


def get_confidence_adjustment(options_signal: str,
                              ensemble_bias: str,
                              options_score: float) -> dict:
    """
    Compute confidence adjustment based on options alignment.

    Rules:
    - Options BULLISH + Ensemble BULLISH → +5 to +10% confidence
    - Options BEARISH + Ensemble BEARISH → +5 to +10% confidence
    - Options BULLISH + Ensemble BEARISH → -5 to -10% confidence
    - Options BEARISH + Ensemble BULLISH → -5 to -10% confidence
    - Options NEUTRAL → 0% adjustment

    Adjustment magnitude scales with options score strength.
    Max adjustment: +/- 10% to avoid options dominating.
    """
    if options_signal == "NEUTRAL" or ensemble_bias == "NEUTRAL":
        return {
            "adjustment": 0,
            "direction": "neutral",
            "message": "Options positioning neutral",
        }

    agrees = options_signal == ensemble_bias
    magnitude = min(10, abs(options_score) * 15)

    if agrees:
        return {
            "adjustment": round(magnitude, 1),
            "direction": "confirms",
            "message": (f"Options confirm {ensemble_bias} bias "
                        f"(PCR supports direction)"),
        }
    else:
        return {
            "adjustment": round(-magnitude, 1),
            "direction": "conflicts",
            "message": (f"Options diverge from {ensemble_bias} bias "
                        f"— institutional positioning suggests caution"),
        }


def get_options_analysis(ensemble_bias: str) -> dict:
    """Main entry point."""
    chain = fetch_options_chain()
    positioning = analyze_positioning(chain)
    if positioning.get("error"):
        return {
            "positioning": positioning,
            "adjustment": {"adjustment": 0, "direction": "neutral",
                           "message": "Options data unavailable"},
            "error": positioning["error"],
        }
    adjustment = get_confidence_adjustment(
        positioning["signal"], ensemble_bias,
        positioning["score"])
    return {
        "positioning": positioning,
        "adjustment": adjustment,
        "error": None,
    }
