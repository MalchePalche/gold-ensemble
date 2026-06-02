"""
data/options_flow.py — options / open-interest layer for the gold ensemble.

Pulls a live GLD options-chain snapshot from Polygon.io (free tier), rolls it
into institutional-positioning metrics (OI- and volume-based put/call ratios,
a net positioning score, and an unusual-activity flag), then translates that
into a small confidence adjustment for the V4 ensemble signal.

GLD is used as the gold proxy: GC=F futures options need a paid Polygon tier,
and GLD is the most liquid free options market that tracks gold positioning.

Requires the POLYGON_KEY environment variable. If it is missing or the call
fails, every entry point degrades gracefully to a neutral / zero-adjustment
result so the dashboard and daily runner never break.
"""

from __future__ import annotations

import os

import requests

POLYGON_KEY = os.getenv("POLYGON_KEY")

# Gold ETF — GLD is the most liquid options market for gold.
# GC=F futures options exist but need a paid tier; GLD options are the best
# free proxy for gold institutional positioning.
GLD_TICKER = "GLD"


def fetch_options_chain(ticker: str = GLD_TICKER) -> dict:
    """
    Fetch current options chain snapshot for GLD.
    Returns calls and puts with open interest and volume.
    """
    if not POLYGON_KEY:
        return {"error": "No Polygon API key", "results": []}

    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        params = {
            "apiKey": POLYGON_KEY,
            "limit": 250,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if "results" not in data:
            return {"error": f"No results: {data.get('status')}", "results": []}

        return {"results": data["results"], "error": None}

    except Exception as e:
        return {"error": str(e), "results": []}


def analyze_positioning(chain_data: dict) -> dict:
    """
    Analyze options chain for institutional positioning signals.

    Metrics:
    1. Put/Call Ratio (OI-based)
       - OI PCR > 1.2 = bearish positioning (more puts than calls)
       - OI PCR < 0.8 = bullish positioning (more calls than puts)
       - 0.8-1.2 = neutral

    2. Put/Call Ratio (Volume-based)
       - Same thresholds but uses today's volume
       - More responsive than OI

    3. Net positioning score
       - Combines both into -1 to +1 score
       - Negative = bearish institutional positioning
       - Positive = bullish

    4. Unusual activity flag
       - Volume/OI ratio > 3 on calls = unusual call buying
       - Volume/OI ratio > 3 on puts = unusual put buying
    """
    if chain_data.get("error"):
        return {
            "error": chain_data["error"],
            "score": 0,
            "signal": "NEUTRAL",
            "pcr_oi": None,
            "pcr_vol": None,
        }

    results = chain_data["results"]

    total_call_oi = 0
    total_put_oi = 0
    total_call_vol = 0
    total_put_vol = 0
    unusual_calls = 0
    unusual_puts = 0

    for opt in results:
        details = opt.get("details", {})
        day = opt.get("day", {})
        oi = opt.get("open_interest", 0) or 0
        vol = day.get("volume", 0) or 0

        contract_type = details.get("contract_type", "")

        if contract_type == "call":
            total_call_oi += oi
            total_call_vol += vol
            if oi > 0 and vol / oi > 3:
                unusual_calls += 1
        elif contract_type == "put":
            total_put_oi += oi
            total_put_vol += vol
            if oi > 0 and vol / oi > 3:
                unusual_puts += 1

    # Put/Call ratios
    pcr_oi = (total_put_oi / total_call_oi
              if total_call_oi > 0 else 1.0)
    pcr_vol = (total_put_vol / total_call_vol
               if total_call_vol > 0 else 1.0)

    # Normalize to -1 to +1 score
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

    # Unusual activity
    unusual = ""
    if unusual_calls > 3:
        unusual = f"Unusual call buying ({unusual_calls} contracts)"
    elif unusual_puts > 3:
        unusual = f"Unusual put buying ({unusual_puts} contracts)"

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
