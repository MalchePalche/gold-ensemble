"""
data/cot.py — CFTC Commitment of Traders (COT) positioning layer for gold.

COT shows the net positioning of commercial hedgers (producers/merchants) vs
large speculators (managed money) in COMEX gold futures. Commercials are the
"smart money" hedgers: relative to their own 3-year range, an extreme reading
in their net positioning has historically been a contrarian tell for gold.
This module turns that into a ±6% confidence adjustment, degrading gracefully
to neutral if CFTC data is unavailable.

Data source — CFTC Socrata API (disaggregated futures-only report, resource
72hh-3qpy). We use the JSON API rather than the flat newcot/f_disagg.txt file
because that file is headerless AND contains only the single most-recent week,
which gives no history to compute a percentile from. The API is headered and
returns the full multi-year weekly history we need to normalize extremes.

CFTC publishes the report every Friday for the prior Tuesday.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# COMEX full-size gold = CFTC contract market code 088691 (NOT the commodity
# code 088, which also covers Micro Gold). We filter on the contract code so the
# series is the single full-size gold contract.
GOLD_CONTRACT_CODE = "088691"
GOLD_MARKET_NAME = "GOLD - COMMODITY EXCHANGE INC."
COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
CACHE_TTL_HOURS = 24

# Position columns we pull from the disaggregated report.
_PROD_LONG = "prod_merc_positions_long"
_PROD_SHORT = "prod_merc_positions_short"
_MM_LONG = "m_money_positions_long_all"
_MM_SHORT = "m_money_positions_short_all"
_REPORT_DATE = "report_date_as_yyyy_mm_dd"

_cache = {"data": None, "fetched_at": None}


def _fetch_cot_raw() -> pd.DataFrame | None:
    """Fetch the full-size gold COT history (futures only), oldest → newest.

    Returns ~3 years of weekly rows so the percentile range can be computed,
    or None on any failure (network, API change) so the caller degrades to a
    neutral 0% adjustment.
    """
    try:
        params = {
            "cftc_contract_market_code": GOLD_CONTRACT_CODE,
            # Newest-first + limit grabs the most RECENT reports (ordering ASC
            # with a limit would return the oldest rows instead). We reverse to
            # oldest → newest below so iloc[-1] is current and tail() is recent.
            "$order": f"{_REPORT_DATE} DESC",
            "$limit": 160,  # ~3 years of weekly reports
        }
        r = requests.get(COT_URL, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            print("[cot] No gold rows returned from CFTC API")
            return None
        # Reverse to chronological order (oldest → newest).
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        print(f"[cot] Fetch failed: {e}")
        return None


def _gold_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive gold filter — the fetch is already gold-only, but if a broader
    frame is ever passed in, keep only the full-size gold contract."""
    if "cftc_contract_market_code" in df.columns:
        g = df[df["cftc_contract_market_code"].astype(str).str.strip()
               == GOLD_CONTRACT_CODE]
        if not g.empty:
            return g
    if "market_and_exchange_names" in df.columns:
        return df[df["market_and_exchange_names"].astype(str).str.strip()
                  == GOLD_MARKET_NAME]
    return df


def _parse_gold_cot(df: pd.DataFrame) -> dict | None:
    """Extract the most recent gold COT row and compute net positioning."""
    try:
        gold = _gold_rows(df)
        if gold.empty:
            return None

        row = gold.iloc[-1]  # most recent (fetch is ordered oldest → newest)

        prod_long  = float(row.get(_PROD_LONG, 0))
        prod_short = float(row.get(_PROD_SHORT, 0))
        mm_long    = float(row.get(_MM_LONG, 0))
        mm_short   = float(row.get(_MM_SHORT, 0))

        commercial_net = prod_long - prod_short
        mm_net         = mm_long - mm_short
        # Socrata dates look like '2026-05-26T00:00:00.000' — keep the date part.
        report_date    = str(row.get(_REPORT_DATE, "unknown"))[:10]

        return {
            "commercial_net": int(commercial_net),
            "mm_net":         int(mm_net),
            "report_date":    report_date,
            "prod_long":      int(prod_long),
            "prod_short":     int(prod_short),
            "mm_long":        int(mm_long),
            "mm_short":       int(mm_short),
        }
    except Exception as e:
        print(f"[cot] Parse failed: {e}")
        return None


def _get_historical_range(df: pd.DataFrame, lookback_rows: int = 156) -> dict | None:
    """
    Get 3-year range (156 weekly reports) of commercial net positioning to
    normalize extremes. Used to compute the percentile rank of current
    positioning.
    """
    try:
        gold = _gold_rows(df)
        if gold.empty:
            return None

        recent = gold.tail(lookback_rows)
        nets = (recent[_PROD_LONG].astype(float) -
                recent[_PROD_SHORT].astype(float))

        return {
            "min": float(nets.min()),
            "max": float(nets.max()),
            "mean": float(nets.mean()),
            "std": float(nets.std()),
        }
    except Exception:
        return None


def get_cot_analysis(current_bias: str) -> dict:
    """
    Main entry point. Returns COT positioning analysis + confidence adjustment.

    Confidence adjustment logic (percentile of commercial net vs 3yr range):
    - Very low percentile (≤20th) = commercials at extreme net-short end =
      historically bullish (contrarian) → +6% if bias BULLISH, -6% if BEARISH
    - Very high percentile (≥80th) = extreme net-long end = bearish signal →
      -6% if BULLISH, +3% if BEARISH
    - Middle range → 0% adjustment

    Returns dict with keys: analysis, adjustment, error
    """
    global _cache

    # Cache check (24h TTL)
    if (_cache["data"] is not None and _cache["fetched_at"] is not None and
            datetime.utcnow() - _cache["fetched_at"] < timedelta(hours=CACHE_TTL_HOURS)):
        raw = _cache["data"]
    else:
        raw = _fetch_cot_raw()
        if raw is not None:
            _cache = {"data": raw, "fetched_at": datetime.utcnow()}

    if raw is None:
        return {
            "analysis": {"signal": "NEUTRAL", "positioning": "UNKNOWN",
                         "commercial_net": None, "mm_net": None,
                         "percentile": None, "report_date": None},
            "adjustment": {"adjustment": 0.0, "reason": "COT data unavailable"},
            "error": "Failed to fetch COT data"
        }

    gold_data = _parse_gold_cot(raw)
    hist_range = _get_historical_range(raw)

    if gold_data is None:
        return {
            "analysis": {"signal": "NEUTRAL", "positioning": "UNKNOWN",
                         "commercial_net": None, "mm_net": None,
                         "percentile": None, "report_date": None},
            "adjustment": {"adjustment": 0.0, "reason": "Gold COT data not found"},
            "error": "Gold not found in COT data"
        }

    commercial_net = gold_data["commercial_net"]
    percentile = None
    positioning = "NEUTRAL"
    adjustment = 0.0
    signal = "NEUTRAL"

    if hist_range and hist_range["max"] != hist_range["min"]:
        percentile = round((commercial_net - hist_range["min"]) /
                            (hist_range["max"] - hist_range["min"]), 3)

        if percentile <= 0.20:
            # Commercials at their net-short extreme = historically bullish.
            positioning = "EXTREME_SHORT"
            signal = "BULLISH"
            adjustment = 6.0 if current_bias == "BULLISH" else -6.0

        elif percentile >= 0.80:
            # Commercials at their net-long extreme = bearish signal.
            positioning = "EXTREME_LONG"
            signal = "BEARISH"
            adjustment = -6.0 if current_bias == "BULLISH" else 3.0

        elif percentile <= 0.35:
            positioning = "MODERATELY_SHORT"
            signal = "MILDLY_BULLISH"
            adjustment = 3.0 if current_bias == "BULLISH" else -3.0

        elif percentile >= 0.65:
            positioning = "MODERATELY_LONG"
            signal = "MILDLY_BEARISH"
            adjustment = -3.0 if current_bias == "BULLISH" else 2.0

    return {
        "analysis": {
            "signal":         signal,
            "positioning":    positioning,
            "commercial_net": commercial_net,
            "mm_net":         gold_data["mm_net"],
            "percentile":     percentile,
            "report_date":    gold_data["report_date"],
            "summary": (f"Commercials net {commercial_net:+,} contracts "
                        f"({positioning}, {percentile:.0%} of 3yr range)"
                        if percentile is not None else
                        f"Commercials net {commercial_net:+,} contracts")
        },
        "adjustment": {
            "adjustment": round(adjustment, 1),
            "reason": f"COT {positioning} → {adjustment:+.1f}%"
        },
        "error": None
    }
