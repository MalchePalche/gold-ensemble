"""
data/sentiment.py — news sentiment analysis for gold.

Pulls gold-relevant headlines from NewsAPI (last 24h), scores each with
TextBlob polarity, and rolls them into a BULLISH/BEARISH/NEUTRAL sentiment
signal. Also flags divergence between news sentiment and the ensemble bias —
a high-value tell that price may reverse within a few days.

Requires the NEWSAPI_KEY environment variable. If it is missing or the call
fails, every entry point degrades gracefully to an empty/neutral result.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytz
import requests
from textblob import TextBlob

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SOFIA_TZ = pytz.timezone("Europe/Sofia")

GOLD_QUERIES = [
    "gold price", "XAU USD", "gold trading",
    "gold rally", "gold selloff", "Federal Reserve gold",
    "inflation gold", "gold futures",
]


def fetch_headlines(hours_back: int = 24) -> list[dict]:
    """
    Fetch gold-relevant headlines from last 24 hours.
    Returns list of dicts with title, source, published, url.
    """
    if not NEWSAPI_KEY:
        return []

    from_time = (datetime.utcnow() - timedelta(hours=hours_back))\
                .strftime("%Y-%m-%dT%H:%M:%S")

    query = 'gold price OR XAU/USD OR "gold futures" OR "gold rally"'

    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_time,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 30,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )
        data = r.json()
        articles = data.get("articles", [])
    except Exception:
        return []

    results = []
    for a in articles:
        title = a.get("title", "") or ""
        description = a.get("description", "") or ""
        if not title:
            continue
        results.append({
            "title": title,
            "description": description,
            "source": a.get("source", {}).get("name", ""),
            "published": a.get("publishedAt", ""),
            "url": a.get("url", ""),
            "text": f"{title}. {description}",
        })

    return results


def score_sentiment(headlines: list[dict]) -> dict:
    """
    Score each headline with TextBlob polarity.
    Polarity: -1.0 (very negative) to +1.0 (very positive)

    Returns a dict with avg_polarity, signal, confidence, per-sentiment
    counts, the scored headlines, and the top 3 bullish/bearish headlines.
    """
    if not headlines:
        return {
            "avg_polarity": 0,
            "signal": "NEUTRAL",
            "confidence": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "scores": [],
            "top_bullish": [],
            "top_bearish": [],
            "total": 0,
        }

    scores = []
    for h in headlines:
        blob = TextBlob(h["text"])
        polarity = blob.sentiment.polarity
        scores.append({
            **h,
            "polarity": round(polarity, 3),
            "sentiment": "bullish" if polarity > 0.05
                         else "bearish" if polarity < -0.05
                         else "neutral",
        })

    bullish = [s for s in scores if s["sentiment"] == "bullish"]
    bearish = [s for s in scores if s["sentiment"] == "bearish"]
    neutral = [s for s in scores if s["sentiment"] == "neutral"]

    avg_polarity = sum(s["polarity"] for s in scores) / len(scores)

    # Signal
    if avg_polarity > 0.05:
        signal = "BULLISH"
    elif avg_polarity < -0.05:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    # Confidence: scale avg_polarity to 0-100
    confidence = min(100, abs(avg_polarity) * 300)

    # Top headlines
    top_bullish = sorted(bullish, key=lambda x: x["polarity"],
                         reverse=True)[:3]
    top_bearish = sorted(bearish, key=lambda x: x["polarity"])[:3]

    return {
        "avg_polarity": round(avg_polarity, 4),
        "signal": signal,
        "confidence": round(confidence, 1),
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "scores": scores,
        "top_bullish": top_bullish,
        "top_bearish": top_bearish,
        "total": len(scores),
    }


def get_sentiment() -> dict:
    """Main entry point — fetch + score."""
    headlines = fetch_headlines(hours_back=24)
    return score_sentiment(headlines)


def divergence_check(sentiment_signal: str,
                     ensemble_bias: str) -> dict:
    """
    Check if sentiment diverges from ensemble signal.
    Divergence = sentiment and ensemble point in opposite directions.
    This is a HIGH VALUE signal — price often follows sentiment
    divergence within 1-3 days.
    """
    if sentiment_signal == "NEUTRAL" or ensemble_bias == "NEUTRAL":
        return {"divergence": False, "message": ""}

    diverges = sentiment_signal != ensemble_bias
    if diverges:
        msg = (f"⚡ Divergence: News is {sentiment_signal} but "
               f"ensemble is {ensemble_bias}. "
               f"Watch for potential reversal in 1-3 days.")
    else:
        msg = (f"✓ Aligned: News sentiment confirms "
               f"ensemble {ensemble_bias} bias.")

    return {"divergence": diverges, "message": msg}
