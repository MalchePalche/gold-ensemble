"""
data/sentiment.py — news sentiment analysis for gold.

Pulls gold-relevant headlines from NewsAPI (last 48h), filters them down to
genuine gold-commodity stories, scores each with FinBERT (a finance-tuned BERT)
polarity weighted by relevance, and rolls them into a BULLISH/BEARISH/NEUTRAL
sentiment signal. Falls back to TextBlob polarity if `transformers` is not
installed, so the system never hard-crashes. Also flags divergence between news
sentiment and the ensemble bias — a high-value tell that price may reverse
within a few days.

Requires the NEWSAPI_KEY environment variable. If it is missing or the call
fails, every entry point degrades gracefully to an empty/neutral result.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import requests
from textblob import TextBlob

# FinBERT (finance-tuned BERT) is the primary scorer. Import only the pipeline
# factory at module level so a missing `transformers` install degrades to the
# TextBlob fallback instead of hard-crashing the runner. The heavy part — the
# ~400MB model download/load — stays lazy in _get_finbert().
try:
    from transformers import pipeline as hf_pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Module-level cache for the loaded FinBERT pipeline (loaded once, on first use).
_finbert = None


def _get_finbert():
    """Lazily build and cache the FinBERT text-classification pipeline."""
    global _finbert
    if _finbert is None:
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,  # CPU; no GPU required
        )
    return _finbert


def _score_finbert(text: str) -> float:
    """Return a polarity float in [-1, +1] for `text`.

    Uses FinBERT when available (positive_score - negative_score); otherwise
    falls back to TextBlob polarity. Any runtime failure degrades to neutral
    (0.0) so a single bad headline never breaks the whole scoring pass.
    """
    if not FINBERT_AVAILABLE:
        return TextBlob(text).sentiment.polarity
    try:
        results = _get_finbert()(text[:512])  # FinBERT max 512 tokens
        # results is a list of [{"label": ..., "score": ...}]
        scores = {r["label"].lower(): r["score"] for r in results[0]}
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)
    except Exception:
        return 0.0  # graceful fallback to neutral

# Commodity-context keywords. A headline must contain at least one of these to
# count as gold-relevant; the more it contains, the higher its relevance weight.
COMMODITY_KEYWORDS = [
    "xau", "gold price", "spot gold", "gold futures", "gold rally",
    "gold selloff", "gold demand", "gold reserves", "gold trading",
    "gold surges", "gold drops", "gold climbs", "gold falls",
    "gold hits", "gold bulls", "gold bears", "bullion",
    "precious metal", "comex gold", "gold ounce", "per ounce",
    "federal reserve", "fed rate", "inflation", "dollar index",
    "treasury yield", "safe haven", "risk off", "central bank",
]

# Hard-block phrases — non-commodity uses of "gold" (sports, entertainment …).
IRRELEVANT_KEYWORDS = [
    "gold medal", "golden globe", "gold award", "gold album",
    "gold record", "gold coast", "heart of gold", "gold mining stock",
    "olympic gold", "gold trophy", "gold standard test",
    "bachelor gold", "age of gold",
]


def is_relevant(headline: dict) -> bool:
    """
    Returns True only if headline is clearly about gold commodity.
    Checks title + description for commodity keywords.
    Filters out known irrelevant patterns.
    """
    text = headline["text"].lower()

    # Hard filter — if irrelevant keyword present, skip
    if any(kw in text for kw in IRRELEVANT_KEYWORDS):
        return False

    # Must contain at least one commodity keyword
    if not any(kw in text for kw in COMMODITY_KEYWORDS):
        return False

    return True


def relevance_score(headline: dict) -> float:
    """
    Score 0.0-1.0 how relevant the headline is to gold trading.
    Used to weight the sentiment score.
    More commodity keywords = higher weight in final avg.
    """
    text = headline["text"].lower()
    hits = sum(1 for kw in COMMODITY_KEYWORDS if kw in text)
    return min(1.0, hits / 3)  # 3+ keywords = full weight


def fetch_headlines(hours_back: int = 48) -> list[dict]:
    """
    Fetch candidate gold headlines from the last `hours_back` hours.

    Uses a strict boolean NewsAPI query (gold-commodity phrases, with sports/
    entertainment uses excluded). Returns the raw matched set; relevance
    filtering + weighting happens in score_sentiment().
    Returns list of dicts with title, source, published, url, text.
    """
    if not NEWSAPI_KEY:
        return []

    from_time = (datetime.utcnow() - timedelta(hours=hours_back))\
                .strftime("%Y-%m-%dT%H:%M:%S")

    query = (
        '("gold price" OR "XAU" OR "gold futures" OR "gold rally" '
        'OR "gold selloff" OR "gold demand" OR "gold reserves" '
        'OR "gold trading" OR "spot gold" OR "gold bulls" '
        'OR "gold bears" OR "gold hits" OR "gold surges" '
        'OR "gold drops" OR "gold climbs" OR "gold falls") '
        'AND NOT ("gold medal" OR "gold award" OR "gold record" '
        'OR "golden globe" OR "gold album" OR "gold standard test" '
        'OR "heart of gold" OR "gold coast" OR "gold mining stock")'
    )

    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_time,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 50,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )
        data = r.json()
        articles = data.get("articles", [])
    except Exception as e:
        print(f"[sentiment] Warning: NewsAPI fetch failed: {e}")
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


def _neutral_result(raw_count: int = 0) -> dict:
    """Empty/neutral sentiment payload (no relevant headlines)."""
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
        "raw_count": raw_count,
        "no_data_msg": "No relevant gold headlines found in last 48h",
    }


def score_sentiment(headlines: list[dict]) -> dict:
    """
    Filter headlines to gold-commodity stories, then score each with FinBERT
    polarity (TextBlob fallback) weighted by relevance. Polarity: -1.0 (very
    negative) to +1.0.

    Returns a dict with avg_polarity (relevance-weighted), signal, confidence,
    per-sentiment counts, the scored headlines (each with a relevance score),
    and the top 3 bullish/bearish headlines. raw_count is the pre-filter fetch
    size so the UI can show "filtered from raw fetch".
    """
    raw_count = len(headlines)

    # 1. Keep only headlines clearly about the gold commodity.
    headlines = [h for h in headlines if is_relevant(h)]
    if not headlines:
        return _neutral_result(raw_count)

    # 2. Score + relevance-weight each headline.
    scores = []
    for h in headlines:
        polarity = _score_finbert(h["text"])
        weight = relevance_score(h)
        scores.append({
            **h,
            "polarity": round(polarity, 3),
            "weight": round(weight, 3),
            "relevance": round(weight, 2),   # for display
            "sentiment": "bullish" if polarity > 0.05
                         else "bearish" if polarity < -0.05
                         else "neutral",
        })

    bullish = [s for s in scores if s["sentiment"] == "bullish"]
    bearish = [s for s in scores if s["sentiment"] == "bearish"]
    neutral = [s for s in scores if s["sentiment"] == "neutral"]

    # 3. Relevance-weighted average polarity.
    total_weight = sum(s["weight"] for s in scores)
    if total_weight > 0:
        avg_polarity = sum(s["polarity"] * s["weight"] for s in scores) / total_weight
    else:
        avg_polarity = 0

    # Signal
    if avg_polarity > 0.05:
        signal = "BULLISH"
    elif avg_polarity < -0.05:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    # Confidence: scale avg_polarity to 0-100
    confidence = min(100, abs(avg_polarity) * 300)

    # Top headlines (by raw polarity)
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
        "raw_count": raw_count,
        "no_data_msg": "",
    }


def get_sentiment() -> dict:
    """Main entry point — fetch + score."""
    headlines = fetch_headlines(hours_back=48)
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
