"""
data/central_banks.py — central-bank gold-buying layer for the gold ensemble.

Pulls annual gold-reserve levels (in USD) for the world's most active
official-sector buyers from the World Bank API (indicator FI.RES.XGLD.CD,
free, no key), then rolls them into a slow-moving structural signal: which
central banks are accumulating vs reducing, the net reserve change across the
tracked set, and a small confidence adjustment for the V4 ensemble.

Central-bank accumulation is one of gold's strongest structural drivers, but
World Bank data is annual (refreshed quarterly), so this is macro *context*,
not a timing signal — the confidence adjustment is capped tighter than the
options layer (+/- 8% vs +/- 10%).

If the API is unavailable or a call fails, every entry point degrades
gracefully to a neutral / zero-adjustment result so the dashboard and daily
runner never break.
"""

from __future__ import annotations

import requests

# Top official-sector gold buyers (plus the two largest holders, US/DE, as
# stable anchors). Source: World Gold Council central-bank statistics.
COUNTRIES = {
    "CN": "China",
    "RU": "Russia",
    "IN": "India",
    "TR": "Turkey",
    "PL": "Poland",
    "SG": "Singapore",
    "SA": "Saudi Arabia",
    "KZ": "Kazakhstan",
    "US": "United States",
    "DE": "Germany",
}

# Flag emoji per country — used by the dashboard buyer/seller lists.
COUNTRY_FLAGS = {
    "China": "🇨🇳",
    "Russia": "🇷🇺",
    "India": "🇮🇳",
    "Turkey": "🇹🇷",
    "Poland": "🇵🇱",
    "Singapore": "🇸🇬",
    "Saudi Arabia": "🇸🇦",
    "Kazakhstan": "🇰🇿",
    "United States": "🇺🇸",
    "Germany": "🇩🇪",
}

_WB_URL = "https://api.worldbank.org/v2/country/{}/indicator/FI.RES.XGLD.CD"


def _fetch_one_country(code: str, name: str) -> tuple[str, dict | None]:
    """Fetch one country's gold-reserve history from the World Bank API."""
    r = requests.get(
        _WB_URL.format(code),
        params={"format": "json", "per_page": 10, "mrv": 8},
        timeout=5,
    )
    data = r.json()

    if not data or len(data) < 2:
        return code, None

    records = data[1]
    if not records:
        return code, None

    # World Bank returns newest-first; keep only non-null observations.
    values = [
        {"year": rec["date"], "value": float(rec["value"]), "country": name}
        for rec in records
        if rec.get("value") is not None
    ]

    if not values:
        return code, None

    return code, {
        "name": name,
        "latest": values[0]["value"],
        "previous": values[1]["value"] if len(values) > 1 else None,
        "history": values,
    }


def fetch_wb_gold_reserves() -> dict:
    """
    Fetch gold reserves (USD) from the World Bank API, concurrently.

    Indicator FI.RES.XGLD.CD — free, no API key required. Returns a dict
    keyed by country code with the latest value, the prior year's value, and
    the recent history. Countries are fetched in parallel (5s per-request
    timeout) so a single slow endpoint can't stall the whole layer; failures
    are logged and skipped individually.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_fetch_one_country, code, name): code
            for code, name in COUNTRIES.items()
        }
        try:
            for future in as_completed(futures, timeout=15):
                try:
                    code, result = future.result()
                    if result:
                        results[code] = result
                except Exception as e:
                    print(f"[central_banks] Warning: {futures[future]} failed: {e}")
        except Exception as e:
            # as_completed itself timed out — keep whatever finished in time.
            print(f"[central_banks] Warning: WB fetch timed out: {e}")

    return results


def fetch_annual_gold_prices() -> dict:
    """Average GC=F close per calendar year, keyed by year string.

    Used to deflate the World Bank's USD-valued gold reserves into a
    tonnage-equivalent so the YoY change reflects actual buying/selling rather
    than gold-price appreciation.
    """
    import yfinance as yf
    from datetime import datetime

    try:
        df = yf.download(
            "GC=F",
            start="2015-01-01",
            end=datetime.today().strftime("%Y-%m-%d"),
            progress=False,
        )["Close"]
        if hasattr(df, "columns"):       # flatten MultiIndex / DataFrame → Series
            df = df.squeeze()
        annual = df.resample("YE").mean()
        return {str(idx.year): float(px) for idx, px in annual.items()}
    except Exception as e:
        print(f"[central_banks] Warning: annual gold price fetch failed: {e}")
        return {}


def deflate_by_gold_price(reserves_data: dict, gold_prices: dict) -> dict:
    """Divide each country's USD reserves by that year's gold price.

    Removes the price-appreciation effect so a country only counts as a buyer
    when it actually added tonnage. Adds a `value_deflated` field to every
    history record (falls back to the raw value when no price is available).
    """
    for data in reserves_data.values():
        for record in data.get("history", []):
            gold_px = gold_prices.get(record["year"])
            if gold_px and gold_px > 0:
                record["value_deflated"] = record["value"] / gold_px
            else:
                record["value_deflated"] = record["value"]
    return reserves_data


def analyze_cb_trend(reserves_data: dict) -> dict:
    """
    Analyze the central-bank buying trend from reserve levels.

    A central bank counts as a buyer/seller when its USD reserves moved more
    than +/- 2% year-over-year (a band that filters out gold-price noise).
    Returns trend (ACCUMULATING / STABLE / REDUCING), signal (BULLISH /
    NEUTRAL / BEARISH), a -1..+1 score, the buyer/seller lists, and the net
    reserve change across the tracked set.
    """
    if not reserves_data:
        return {
            "trend": "UNKNOWN",
            "signal": "NEUTRAL",
            "score": 0,
            "buyers": [],
            "sellers": [],
            "stable": [],
            "net_change_pct": 0,
            "buyer_count": 0,
            "seller_count": 0,
            "momentum": "UNKNOWN",
            "summary": "No central bank data available",
            "data_note": "World Bank annual data — updates quarterly",
        }

    buyers = []
    sellers = []
    stable = []
    total_latest = 0.0
    total_previous = 0.0

    for data in reserves_data.values():
        name = data["name"]
        flag = COUNTRY_FLAGS.get(name, "")

        # Prefer gold-price-deflated values (tonnage-equivalent) when available
        # so the YoY change isn't just gold-price appreciation. History is
        # newest-first, so [0] is latest and [1] the prior year.
        history = data.get("history", [])
        defl = [h for h in history if h.get("value_deflated") is not None]
        if len(defl) >= 2:
            latest = defl[0]["value_deflated"]
            previous = defl[1]["value_deflated"]
        else:
            latest = data["latest"]
            previous = data["previous"]

        if previous is None or previous == 0:
            stable.append(name)
            continue

        pct_change = (latest - previous) / previous * 100
        total_latest += latest
        total_previous += previous

        if pct_change > 2:
            buyers.append({
                "country": name,
                "flag": flag,
                "change_pct": round(pct_change, 1),
                "direction": "buying",
            })
        elif pct_change < -2:
            sellers.append({
                "country": name,
                "flag": flag,
                "change_pct": round(pct_change, 1),
                "direction": "selling",
            })
        else:
            stable.append(name)

    net_change_pct = (
        (total_latest - total_previous) / total_previous * 100
        if total_previous > 0 else 0
    )

    buyer_count = len(buyers)
    seller_count = len(sellers)

    # Trend / signal / score. Broad accumulation (>=4 CBs) or a >3% net jump
    # is bullish; the mirror image is bearish; everything else is structural
    # stability.
    if buyer_count >= 4 or net_change_pct > 3:
        trend, signal = "ACCUMULATING", "BULLISH"
        score = min(1.0, 0.3 + (buyer_count * 0.1))
    elif seller_count >= 4 or net_change_pct < -3:
        trend, signal = "REDUCING", "BEARISH"
        score = max(-1.0, -0.3 - (seller_count * 0.1))
    else:
        trend, signal = "STABLE", "NEUTRAL"
        score = 0.0

    if buyers:
        top_buyers = ", ".join(
            b["country"]
            for b in sorted(buyers, key=lambda x: x["change_pct"], reverse=True)[:3]
        )
        summary = f"{buyer_count} CBs accumulating led by {top_buyers}"
    elif sellers:
        summary = f"{seller_count} CBs reducing gold reserves"
    else:
        summary = "No significant central bank accumulation detected"

    return {
        "trend": trend,
        "signal": signal,
        "score": round(score, 2),
        "buyers": buyers,
        "sellers": sellers,
        "stable": stable,
        "net_change_pct": round(net_change_pct, 2),
        "buyer_count": buyer_count,
        "seller_count": seller_count,
        "summary": summary,
        "data_note": "World Bank annual data — updates quarterly",
    }


def get_confidence_adjustment(cb_signal: str,
                              ensemble_bias: str,
                              cb_score: float) -> dict:
    """
    Confidence adjustment from the central-bank trend.

    Capped at +/- 8% (tighter than the options layer's +/- 10%): CB reserve
    data is annual/quarterly, so it is structural context, not a timing
    signal. Confirms the bias when the CB signal agrees, dents it when they
    conflict, and stays flat when either side is neutral.
    """
    if cb_signal == "NEUTRAL" or ensemble_bias == "NEUTRAL":
        return {
            "adjustment": 0,
            "direction": "neutral",
            "message": "Central bank positioning neutral",
        }

    agrees = cb_signal == ensemble_bias
    magnitude = min(8, abs(cb_score) * 10)

    if agrees:
        return {
            "adjustment": round(magnitude, 1),
            "direction": "confirms",
            "message": f"CB accumulation confirms {ensemble_bias} structural bias",
        }
    return {
        "adjustment": round(-magnitude, 1),
        "direction": "conflicts",
        "message": (f"CB trend conflicts with {ensemble_bias} bias "
                    f"— structural caution"),
    }


def get_cb_analysis(ensemble_bias: str) -> dict:
    """Main entry point — fetch reserves, analyze, and size the adjustment."""
    try:
        reserves = fetch_wb_gold_reserves()
        # Deflate USD reserves by annual gold price so "buying" reflects real
        # tonnage changes, not gold-price appreciation (false-bullish in rallies).
        gold_prices = fetch_annual_gold_prices()
        if gold_prices:
            reserves = deflate_by_gold_price(reserves, gold_prices)
        analysis = analyze_cb_trend(reserves)
        adjustment = get_confidence_adjustment(
            analysis["signal"], ensemble_bias, analysis["score"])
        return {
            "analysis": analysis,
            "adjustment": adjustment,
            "reserves": reserves,
            "error": None,
        }
    except Exception as e:
        return {
            "analysis": {"trend": "UNKNOWN", "signal": "NEUTRAL",
                         "score": 0, "summary": str(e)},
            "adjustment": {"adjustment": 0, "direction": "neutral",
                           "message": "CB data unavailable"},
            "reserves": {},
            "error": str(e),
        }
