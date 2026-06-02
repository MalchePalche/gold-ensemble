"""
data/correlations.py — real-time correlation monitor for gold vs key assets.

Computes 30-day and 5-day rolling return correlations of XAU/USD against the
dollar, equities, yields, silver, oil and the VIX, and flags when a normally
inverse/positive relationship breaks down (a possible regime-change tell).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

ASSETS = {
    "DXY":    {"ticker": "DX-Y.NYB", "label": "DXY",        "normal": "inverse"},
    "SPX":    {"ticker": "^GSPC",    "label": "S&P 500",    "normal": "inverse"},
    "TNX":    {"ticker": "^TNX",     "label": "10Y Yields", "normal": "inverse"},
    "SILVER": {"ticker": "SI=F",     "label": "Silver",     "normal": "positive"},
    "OIL":    {"ticker": "CL=F",     "label": "Oil",        "normal": "positive"},
    "VIX":    {"ticker": "^VIX",     "label": "VIX",        "normal": "positive"},
}


def fetch_correlations(lookback_days: int = 30) -> dict:
    """
    Fetch 30-day rolling correlation of each asset vs XAU/USD.
    Also compute 5-day correlation to detect recent breakdown.
    Returns dict with correlation data per asset.
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_days + 10)

    # Fetch gold
    gold = yf.download("GC=F", start=start, end=end,
                       progress=False)["Close"].squeeze()
    gold_returns = gold.pct_change().dropna()

    results = {}
    for key, meta in ASSETS.items():
        try:
            asset = yf.download(meta["ticker"], start=start,
                               end=end, progress=False)["Close"].squeeze()
            asset_returns = asset.pct_change().dropna()

            # Align
            aligned = pd.DataFrame({
                "gold": gold_returns,
                "asset": asset_returns,
            }).dropna()

            if len(aligned) < 5:
                continue

            # 30-day correlation
            corr_30d = aligned["gold"].corr(aligned["asset"])

            # 5-day correlation (recent behavior)
            corr_5d = aligned.tail(5)["gold"].corr(
                aligned.tail(5)["asset"])

            # Current price + change
            current = float(asset.iloc[-1])
            prev = float(asset.iloc[-2])
            change_pct = (current - prev) / prev * 100

            # Breakdown detection
            normal = meta["normal"]
            breakdown = False
            breakdown_msg = ""

            if normal == "inverse":
                # Should be negative correlation
                if corr_30d > 0.3:
                    breakdown = True
                    breakdown_msg = "Moving WITH gold (unusual)"
                elif corr_5d > 0.2 and corr_30d < -0.2:
                    breakdown = True
                    breakdown_msg = "Recent correlation flip"
            elif normal == "positive":
                # Should be positive correlation
                if corr_30d < -0.3:
                    breakdown = True
                    breakdown_msg = "Moving AGAINST gold (unusual)"
                elif corr_5d < -0.2 and corr_30d > 0.2:
                    breakdown = True
                    breakdown_msg = "Recent correlation flip"

            results[key] = {
                "label": meta["label"],
                "normal": normal,
                "corr_30d": round(corr_30d, 3),
                "corr_5d": round(corr_5d, 3),
                "current": current,
                "change_pct": round(change_pct, 2),
                "breakdown": breakdown,
                "breakdown_msg": breakdown_msg,
            }
        except Exception:
            continue

    return results


def correlation_summary(corr_data: dict) -> str:
    """
    Returns overall correlation health:
    ALIGNED   — most assets behaving as expected vs gold
    MIXED     — some breakdowns detected
    BREAKDOWN — multiple assets behaving abnormally
    """
    breakdowns = sum(1 for v in corr_data.values() if v["breakdown"])
    if breakdowns == 0:
        return "ALIGNED"
    elif breakdowns <= 2:
        return "MIXED"
    return "BREAKDOWN"
