"""
data/calendar.py — Forex Factory economic calendar (this week).

Pulls the public Forex Factory weekly calendar JSON and filters it down to the
USD, HIGH-impact events that move gold. Times are converted to Europe/Sofia so
the dashboard and daily runner can show them in local time.
"""

from __future__ import annotations

from datetime import datetime

import pytz
import requests

SOFIA_TZ = pytz.timezone("Europe/Sofia")
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

GOLD_RELEVANT_EVENTS = [
    "Non-Farm Payrolls", "NFP", "CPI", "Core CPI",
    "FOMC", "Fed", "Interest Rate", "GDP",
    "Unemployment", "PPI", "Retail Sales",
    "ISM", "PMI", "Treasury", "Inflation",
    "Powell", "Jobless Claims", "JOLTS",
    "Consumer Confidence", "Durable Goods",
]


def fetch_calendar() -> list[dict]:
    """
    Fetch this week's Forex Factory calendar.
    Filter for: USD events only + HIGH impact only.
    Returns list of dicts sorted by datetime ascending.
    """
    try:
        r = requests.get(FF_URL, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        events = r.json()
    except Exception as e:
        print(f"[calendar] Warning: Forex Factory fetch failed: {e}")
        return []

    filtered = []
    for e in events:
        # USD only
        if e.get("country", "").upper() != "USD":
            continue
        # High impact only
        if e.get("impact", "").lower() != "high":
            continue

        # Parse datetime
        try:
            dt_str = e.get("date", "")
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            dt_sofia = dt.astimezone(SOFIA_TZ)
        except Exception:
            continue

        filtered.append({
            "title": e.get("title", ""),
            "datetime_utc": dt,
            "datetime_sofia": dt_sofia,
            "date": dt_sofia.date(),
            "time_sofia": dt_sofia.strftime("%H:%M"),
            "impact": e.get("impact", ""),
            "forecast": e.get("forecast", ""),
            "previous": e.get("previous", ""),
            "actual": e.get("actual", ""),
            "currency": e.get("country", ""),
        })

    # Sort by time
    filtered.sort(key=lambda x: x["datetime_utc"])
    return filtered


def get_todays_events() -> list[dict]:
    """Return only today's high-impact USD events."""
    all_events = fetch_calendar()
    today = datetime.now(SOFIA_TZ).date()
    return [e for e in all_events if e["date"] == today]


def get_week_events() -> list[dict]:
    """Return all high-impact USD events this week."""
    return fetch_calendar()


def get_next_event() -> dict | None:
    """Return the next upcoming event (not yet passed)."""
    now = datetime.now(pytz.UTC)
    upcoming = [e for e in fetch_calendar()
                if e["datetime_utc"] > now]
    return upcoming[0] if upcoming else None


def event_risk_score(events: list[dict]) -> str:
    """
    Given a list of today's events, return risk level:
    HIGH   — FOMC, NFP, CPI, rate decision present
    MEDIUM — other high-impact events present
    LOW    — no events today
    """
    titles = " ".join(e["title"] for e in events).upper()
    if any(k in titles for k in ["FOMC", "NON-FARM", "NFP", "CPI", "RATE DECISION"]):
        return "HIGH"
    elif events:
        return "MEDIUM"
    return "LOW"
