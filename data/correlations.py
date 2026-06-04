"""
data/correlations.py — real-time correlation monitor for gold vs key assets.

Computes 5/30/90/250-day rolling return correlations of XAU/USD against the
dollar, equities, yields, silver, oil, the VIX and EUR/USD, and flags when a
normally inverse/positive relationship breaks down or when the short-term
(30d) correlation structurally diverges from its long-term (250d) baseline —
both possible regime-change tells.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

ASSETS = {
    "DXY":    {"ticker": "DX-Y.NYB",  "label": "DXY",        "normal": "inverse"},
    "SPX":    {"ticker": "^GSPC",     "label": "S&P 500",    "normal": "inverse"},
    "TNX":    {"ticker": "^TNX",      "label": "10Y Yields", "normal": "inverse"},
    "SILVER": {"ticker": "SI=F",      "label": "Silver",     "normal": "positive"},
    "OIL":    {"ticker": "CL=F",      "label": "Oil",        "normal": "positive"},
    "VIX":    {"ticker": "^VIX",      "label": "VIX",        "normal": "positive"},
    # Gold up ⇒ dollar down ⇒ EUR/USD up, so gold and EUR/USD should be
    # positively correlated (expected_sign +1). `normal` drives the existing
    # breakdown logic; expected_sign documents the directional convention.
    "EURUSD": {"ticker": "EURUSD=X",  "label": "EUR/USD",    "normal": "positive",
               "expected_sign": 1},
}


def _corr_tail(aligned: pd.DataFrame, n: int) -> float:
    """Correlation of gold vs asset over the most recent `n` aligned bars.

    Returns 0.0 when there are too few points / no variance (corr is NaN),
    so downstream comparisons never have to special-case NaN.
    """
    window = aligned.tail(n)
    if len(window) < 2:
        return 0.0
    c = window["gold"].corr(window["asset"])
    return 0.0 if pd.isna(c) else float(c)


def _structural_shift(corr_5d, corr_30d, corr_90d, corr_250d) -> dict:
    """Detects when recent correlation deviates from long-term norm."""
    shift = abs(corr_30d - corr_250d) > 0.35  # 30d vs 250d divergence
    direction = None
    if shift:
        direction = "WEAKENING" if abs(corr_30d) < abs(corr_250d) else "STRENGTHENING"
    return {"shift": shift, "direction": direction}


def fetch_correlations(lookback_days: int = 30) -> dict:
    """
    Fetch rolling return correlations of each asset vs XAU/USD across multiple
    windows: 5-day (recent), 30-day, 90-day and 250-day (long-term baseline).
    Flags both classic breakdowns and structural shifts (30d diverging from the
    250d baseline). Returns dict with correlation data per asset.
    """
    end = datetime.today()
    # Pull enough history to cover the 250-trading-day window regardless of the
    # caller's `lookback_days`; ~1.6 calendar days per trading day + buffer.
    days_needed = max(lookback_days, 250)
    start = end - timedelta(days=int(days_needed * 1.6) + 15)

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

            # Rolling-window correlations (most-recent N aligned bars each).
            corr_30d  = _corr_tail(aligned, 30)
            corr_5d   = _corr_tail(aligned, 5)
            corr_90d  = _corr_tail(aligned, 90)
            corr_250d = _corr_tail(aligned, 250)

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

            # Structural-shift detection (30d vs 250d baseline divergence).
            structural_shift = _structural_shift(
                corr_5d, corr_30d, corr_90d, corr_250d)

            results[key] = {
                "label": meta["label"],
                "normal": normal,
                "corr_30d": round(corr_30d, 3),
                "corr_5d": round(corr_5d, 3),
                "corr_90d": round(corr_90d, 3),
                "corr_250d": round(corr_250d, 3),
                "current": current,
                "change_pct": round(change_pct, 2),
                "breakdown": breakdown,
                "breakdown_msg": breakdown_msg,
                "structural_shift": structural_shift,
            }
        except Exception as e:
            print(f"[correlations] Warning: {meta.get('label', key)} failed: {e}")
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
        summary = "ALIGNED"
    elif breakdowns <= 2:
        summary = "MIXED"
    else:
        summary = "BREAKDOWN"

    # Flag a regime change when 2+ assets have a 30d-vs-250d structural shift.
    shifts = sum(1 for v in corr_data.values()
                 if v.get("structural_shift", {}).get("shift"))
    if shifts >= 2:
        summary += " + STRUCTURAL SHIFT"

    return summary
