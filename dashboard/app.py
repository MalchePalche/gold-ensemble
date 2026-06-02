"""
dashboard/app.py — Gold Ensemble V4 Streamlit dashboard (dark theme).

Reads precomputed signals from Supabase (written by run_daily.py) and shows a
live XAU/USD price. Designed to run on Railway — it does NOT recompute the
ensemble, it just renders what the daily runner stored.

Run locally with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.queries import get_latest_signal, get_recent_signals


# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="XAU/USD · V4 Ensemble", page_icon="🥇")

# Re-run the script every 60s so live price + signals stay fresh.
st_autorefresh(interval=60000, key="dash_refresh")

# ── colors ────────────────────────────────────────────────────────────────────
GREEN   = "#639922"
RED     = "#E24B4A"
NEUTRAL = "#888780"
AMBER   = "#BA7517"


def bias_key(b: str | None) -> str:
    """Map a stored bias string to a CSS suffix."""
    return {"BULLISH": "bullish", "BEARISH": "bearish"}.get(b or "", "neutral")


def bias_hex(b: str | None) -> str:
    return {"BULLISH": GREEN, "BEARISH": RED}.get(b or "", NEUTRAL)


# ── global styling ────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  /* hide streamlit default chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2rem 1rem 2rem; max-width: 900px; margin: 0 auto; }

  /* global */
  body, .stApp { background-color: #0d0d0d; color: #e8e8e8; font-family: 'Inter', sans-serif; }

  /* signal card */
  .signal-card { border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.5rem;
                 display: flex; justify-content: space-between; align-items: center;
                 margin-bottom: 1rem; background: #111; }
  .signal-bearish { border-left: 3px solid #E24B4A; }
  .signal-bullish { border-left: 3px solid #639922; }
  .signal-neutral { border-left: 3px solid #888780; }
  .signal-label-bearish { font-size: 2rem; font-weight: 500; color: #E24B4A; }
  .signal-label-bullish { font-size: 2rem; font-weight: 500; color: #639922; }
  .signal-label-neutral { font-size: 2rem; font-weight: 500; color: #888780; }
  .signal-sub { font-size: 0.8rem; color: #888; margin-top: 4px; }
  .conf-pct { font-size: 1.8rem; font-weight: 500; color: #e8e8e8; text-align: right; }
  .conf-label { font-size: 0.75rem; color: #888; text-align: right; }
  .conf-tag { font-size: 0.7rem; text-align: right; margin-top: 4px; }
  .conf-tag-pos { color: #639922; }
  .conf-tag-neg { color: #E24B4A; }
  .conf-tag-neutral { color: #888; }

  /* metric cards */
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 1rem; }
  .metric-card { background: #161616; border-radius: 8px; padding: 0.9rem 1rem; }
  .metric-label { font-size: 0.72rem; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 1.1rem; font-weight: 500; color: #e8e8e8; }
  .metric-positive { color: #639922; }
  .metric-negative { color: #E24B4A; }
  .metric-neutral { color: #888780; }
  .metric-amber { color: #BA7517; }

  /* section headers */
  .section-header { font-size: 0.72rem; color: #888; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.75rem; font-weight: 500; }

  /* strategy rows */
  .strat-card { border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.25rem;
                margin-bottom: 1rem; background: #111; }
  .strat-row { display: flex; justify-content: space-between; align-items: center;
               padding: 8px 0; border-bottom: 1px solid #1a1a1a; }
  .strat-row:last-child { border-bottom: none; }
  .strat-name { font-size: 0.875rem; color: #e8e8e8; }
  .strat-driver { font-size: 0.75rem; color: #888; margin-top: 2px; }
  .badge-bearish { background: #2a1010; color: #E24B4A; font-size: 0.75rem;
                   padding: 3px 10px; border-radius: 20px; font-weight: 500; }
  .badge-bullish { background: #0f1f08; color: #639922; font-size: 0.75rem;
                   padding: 3px 10px; border-radius: 20px; font-weight: 500; }
  .badge-neutral { background: #1a1a1a; color: #888780; font-size: 0.75rem;
                   padding: 3px 10px; border-radius: 20px; font-weight: 500; }

  /* history rows */
  .hist-card { border: 1px solid #1e1e1e; border-radius: 12px; padding: 1.25rem;
               margin-bottom: 1rem; background: #111; }
  .hist-row { display: flex; gap: 12px; padding: 7px 0; border-bottom: 1px solid #1a1a1a;
              font-size: 0.8rem; align-items: center; }
  .hist-row:last-child { border-bottom: none; }
  .hist-date { color: #888; width: 70px; }
  .hist-bearish { color: #E24B4A; font-weight: 500; width: 65px; }
  .hist-bullish { color: #639922; font-weight: 500; width: 65px; }
  .hist-neutral { color: #888780; font-weight: 500; width: 65px; }
  .hist-size { color: #e8e8e8; width: 40px; }
  .hist-conf { color: #e8e8e8; flex: 1; }
  .hist-vol { color: #888; font-size: 0.75rem; }

  /* live price */
  .price-positive { color: #639922; font-size: 0.85rem; }
  .price-negative { color: #E24B4A; font-size: 0.85rem; }
  .live-dot { display: inline-block; width: 8px; height: 8px; background: #639922;
              border-radius: 50%; margin-right: 4px; }

  /* economic calendar */
  .risk-high { background: #2a1010; border: 1px solid #E24B4A; border-radius: 8px;
               padding: 0.75rem 1rem; color: #E24B4A; font-size: 0.85rem;
               margin-bottom: 1rem; }
  .risk-medium { background: #1f1a0a; border: 1px solid #BA7517; border-radius: 8px;
                 padding: 0.75rem 1rem; color: #BA7517; font-size: 0.85rem;
                 margin-bottom: 1rem; }
  .next-event { font-size: 0.78rem; color: #888; margin-top: 0.5rem;
                margin-bottom: 1rem; }
  .next-event span { color: #e8e8e8; }
  .cal-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  .cal-table th { color: #888; text-transform: uppercase; font-size: 0.7rem;
                  letter-spacing: 0.05em; text-align: left; padding: 6px 10px;
                  border-bottom: 1px solid #1e1e1e; font-weight: 500; }
  .cal-table td { padding: 7px 10px; border-bottom: 1px solid #1a1a1a; }
  .cal-row-today td { background: #181818; }
  .cal-row-past td { color: #5a5a5a; }
  .cal-row-future td { color: #e8e8e8; }
  .cal-actual-better { color: #639922; }
  .cal-actual-worse { color: #E24B4A; }
  .cal-actual-pending { color: #888; }

  /* correlation monitor */
  .corr-card { background: #161616; border-radius: 8px; padding: 0.9rem;
               border: 1px solid #1e1e1e; }
  .corr-card.breakdown { border: 1px solid #BA7517; background: #1a160a; }
  .corr-label { font-size: 0.72rem; color: #888; text-transform: uppercase;
                letter-spacing: 0.05em; }
  .corr-price { font-size: 1rem; font-weight: 500; color: #e8e8e8; margin: 4px 0; }
  .corr-stat { font-size: 0.75rem; color: #888; }
  .corr-breakdown-msg { font-size: 0.72rem; color: #BA7517; margin-top: 4px; }
  .badge-aligned { background: #0f1f08; color: #639922; font-size: 0.75rem;
                   padding: 3px 10px; border-radius: 20px; }
  .badge-mixed { background: #1f1a0a; color: #BA7517; font-size: 0.75rem;
                 padding: 3px 10px; border-radius: 20px; }
  .badge-breakdown { background: #2a1010; color: #E24B4A; font-size: 0.75rem;
                     padding: 3px 10px; border-radius: 20px; }
  .corr-interp { font-size: 0.8rem; color: #888; margin-top: 0.75rem;
                 padding: 0.6rem 0.75rem; background: #161616;
                 border-radius: 6px; border-left: 2px solid #1e1e1e; }

  /* options / OI positioning */
  .opt-unusual { background: #1f1a0a; border: 1px solid #BA7517;
                 border-radius: 8px; padding: 0.75rem 1rem; color: #BA7517;
                 font-size: 0.85rem; margin: 0.75rem 0; }
  .opt-confirms { color: #639922; font-size: 0.82rem; margin: 0.75rem 0;
                  padding: 0.6rem 0.75rem; background: #0f1f08;
                  border-radius: 6px; border-left: 2px solid #639922; }
  .opt-conflicts { background: #1f1a0a; border: 1px solid #BA7517;
                   border-radius: 8px; padding: 0.75rem 1rem; color: #BA7517;
                   font-size: 0.82rem; margin: 0.75rem 0; }
  .opt-neutral-msg { color: #888; font-size: 0.82rem; margin: 0.75rem 0;
                     padding: 0.6rem 0.75rem; background: #161616;
                     border-radius: 6px; border-left: 2px solid #1e1e1e; }

  /* news sentiment */
  .sentiment-card { border: 1px solid #1e1e1e; border-radius: 12px;
                    padding: 1.25rem; background: #111;
                    margin-bottom: 1rem; }
  .polarity-track { height: 6px; background: #1e1e1e; border-radius: 3px;
                    position: relative; margin: 0.75rem 0; }
  .polarity-fill-bull { height: 6px; background: #639922; border-radius: 3px; }
  .polarity-fill-bear { height: 6px; background: #E24B4A; border-radius: 3px; }
  .sentiment-label { font-size: 0.72rem; color: #888; }
  .headline-item { padding: 6px 0; border-bottom: 1px solid #1a1a1a;
                   font-size: 0.78rem; }
  .headline-item:last-child { border-bottom: none; }
  .headline-title { color: #e8e8e8; text-decoration: none; }
  .headline-title:hover { color: #639922; }
  .headline-meta { color: #888; font-size: 0.72rem; margin-top: 2px; }
  .divergence-alert { background: #1f1a0a; border: 1px solid #BA7517;
                      border-radius: 8px; padding: 0.75rem 1rem;
                      color: #BA7517; font-size: 0.82rem;
                      margin: 0.75rem 0; }
  .sentiment-aligned { color: #639922; font-size: 0.78rem;
                       margin: 0.5rem 0; }

  /* intraday confirmation */
  .confirm-enter { background: #0f1f08; border: 1px solid #639922;
                   border-radius: 8px; padding: 0.9rem 1rem;
                   color: #639922; margin-bottom: 1rem; }
  .confirm-wait { background: #1f1a0a; border: 1px solid #BA7517;
                  border-radius: 8px; padding: 0.9rem 1rem;
                  color: #BA7517; margin-bottom: 1rem; }
  .confirm-against { background: #2a1010; border: 1px solid #E24B4A;
                     border-radius: 8px; padding: 0.9rem 1rem;
                     color: #E24B4A; margin-bottom: 1rem; }
  .confirm-enter-strong { background: #0a2008; border: 1px solid #4CAF50;
                          border-radius: 8px; padding: 0.9rem 1rem;
                          color: #4CAF50; margin-bottom: 1rem; }
  .confirm-enter-weak { background: #1f1a0a; border: 1px solid #BA7517;
                        border-radius: 8px; padding: 0.9rem 1rem;
                        color: #BA7517; margin-bottom: 1rem; }
  .confirm-reversal { background: #1a0a2a; border: 1px solid #7C3AED;
                      border-radius: 8px; padding: 0.9rem 1rem;
                      color: #7C3AED; margin-bottom: 1rem; }
  .vol-strip { display: flex; gap: 1.5rem; font-size: 0.8rem;
               color: #888; padding: 0.5rem 0; }
  .vol-strip span { color: #e8e8e8; }
  .vol-high { color: #639922; }
  .vol-low { color: #888780; }
  .vol-climax { background: #1f1a0a; border: 1px solid #BA7517;
                border-radius: 6px; padding: 0.5rem 0.75rem;
                color: #BA7517; font-size: 0.78rem; margin-top: 0.5rem; }
  .size-note { background: #161616; border-radius: 6px;
               padding: 0.5rem 0.75rem; color: #BA7517;
               font-size: 0.78rem; margin-top: 0.5rem;
               border-left: 2px solid #BA7517; }
  .confirm-reason { font-size: 0.85rem; font-weight: 500; }
  .confirm-sub { font-size: 0.78rem; margin-top: 4px; opacity: 0.8; }
  .session-card { background: #161616; border-radius: 8px;
                  padding: 0.9rem; border: 1px solid #1e1e1e; }
  .session-label { font-size: 0.72rem; color: #888; text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 6px; }
  .session-levels { font-size: 0.82rem; color: #e8e8e8; }
  .price-strip { display: flex; gap: 1.5rem; font-size: 0.8rem;
                 color: #888; padding: 0.6rem 0; }
  .price-strip span { color: #e8e8e8; }
</style>
""",
    unsafe_allow_html=True,
)


# ── cached data access ───────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def latest_signal() -> dict | None:
    return get_latest_signal()


@st.cache_data(ttl=300, show_spinner=False)
def recent_signals(n: int = 14) -> list[dict]:
    return get_recent_signals(n)


@st.cache_data(ttl=60, show_spinner=False)
def live_price() -> float | None:
    """Live XAU/USD price via yfinance fast_info (refreshes every 60s)."""
    try:
        return float(yf.Ticker("GC=F").fast_info.last_price)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)  # refresh hourly
def load_calendar():
    """This week's high-impact USD calendar (week, today, next, risk)."""
    from data.calendar import (
        get_week_events,
        get_todays_events,
        get_next_event,
        event_risk_score,
    )
    week = get_week_events()
    today = get_todays_events()
    next_evt = get_next_event()
    risk = event_risk_score(today)
    return week, today, next_evt, risk


@st.cache_data(ttl=300, show_spinner=False)  # refresh every 5 min
def load_correlations():
    """Gold-vs-asset correlation data + overall health summary."""
    from data.correlations import fetch_correlations, correlation_summary
    data = fetch_correlations()
    summary = correlation_summary(data)
    return data, summary


@st.cache_data(ttl=1800, show_spinner=False)  # refresh every 30 min
def load_sentiment():
    """News sentiment over the last 48h (signal, counts, headlines)."""
    from data.sentiment import get_sentiment
    return get_sentiment()


@st.cache_data(ttl=60, show_spinner=False)  # refresh every 60 seconds
def load_intraday(bias: str):
    """Intraday confirmation layer: session levels, price action, ENTER/WAIT/AGAINST."""
    from data.intraday import get_intraday_analysis
    return get_intraday_analysis(bias)


@st.cache_data(ttl=900, show_spinner=False)  # refresh every 15 min
def load_options(bias: str):
    """Options/OI layer: GLD put/call positioning + confidence adjustment."""
    from data.options_flow import get_options_analysis
    return get_options_analysis(bias)


@st.cache_data(ttl=86400, show_spinner=False)  # refresh once per day — annual data
def load_cb_data(bias: str):
    """Central-bank layer: official-sector gold reserves + confidence adjustment."""
    from data.central_banks import get_cb_analysis
    return get_cb_analysis(bias)


# ── load ─────────────────────────────────────────────────────────────────────
try:
    signal = latest_signal()
except Exception as e:
    st.error(f"Failed to read signals from Supabase: {e}")
    st.stop()

if not signal:
    st.markdown("## XAU/USD · V4 Ensemble")
    st.warning("No signals found yet. Run `python run_daily.py` to populate Supabase.")
    st.stop()

bias         = signal["bias"]
stored_price = float(signal["price"]) if signal.get("price") is not None else None
price_now    = live_price()

try:
    week_events, today_events, next_event, risk_score = load_calendar()
except Exception:
    week_events, today_events, next_event, risk_score = [], [], None, "LOW"

try:
    corr_data, corr_summary = load_correlations()
except Exception:
    corr_data, corr_summary = {}, "ALIGNED"

try:
    sentiment = load_sentiment()
except Exception:
    sentiment = None

try:
    intraday = load_intraday(bias)
except Exception as e:
    intraday = {"error": str(e)}

try:
    options_data = load_options(bias)
except Exception as e:
    options_data = {"positioning": {}, "adjustment": {}, "error": str(e)}

positioning = options_data.get("positioning", {}) or {}
adjustment = options_data.get("adjustment", {}) or {}

try:
    cb_data = load_cb_data(bias)
except Exception as e:
    cb_data = {"analysis": {}, "adjustment": {}, "error": str(e)}

cb_analysis = cb_data.get("analysis", {}) or {}
cb_adjustment = cb_data.get("adjustment", {}) or {}

# Flag lookup for the stable-CB list (buyers/sellers already carry their flag).
try:
    from data.central_banks import COUNTRY_FLAGS as COUNTRY_FLAGS_DASH
except Exception:
    COUNTRY_FLAGS_DASH = {}


# ── top bar ────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([3, 1])

with top_l:
    disp_price = price_now if price_now is not None else stored_price
    price_html = f"${disp_price:,.2f}" if disp_price is not None else "—"
    change_html = ""
    if price_now is not None and stored_price:
        chg = price_now - stored_price
        pct = chg / stored_price * 100.0
        cls = "price-positive" if chg >= 0 else "price-negative"
        change_html = (
            f"<div class='{cls}'><span class='live-dot' "
            f"style='background:{GREEN if chg >= 0 else RED};'></span>"
            f"{chg:+,.2f} ({pct:+.2f}%) · live</div>"
        )
    st.markdown(
        f"""
        <div style="color:#888;font-size:0.8rem;letter-spacing:0.05em;">XAU/USD · V4 ENSEMBLE</div>
        <div style="font-size:2.2rem;font-weight:600;color:#e8e8e8;line-height:1.2;">{price_html}</div>
        {change_html}
        """,
        unsafe_allow_html=True,
    )

with top_r:
    st.markdown(
        f"<div style='color:#888;font-size:0.75rem;text-align:right;margin-bottom:6px;'>"
        f"Signal {signal['date']}</div>",
        unsafe_allow_html=True,
    )
    if st.button("↻ Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── signal card ────────────────────────────────────────────────────────────────
conf = float(signal["confidence"])
pos  = float(signal["position_size"])

# Positioning adjustments (display only — Supabase keeps the clean signal).
# Combine the options layer (timing) and the central-bank layer (structural).
opt_adj = float(adjustment.get("adjustment", 0) or 0)
cb_adj = float(cb_adjustment.get("adjustment", 0) or 0)
total_adjustment = opt_adj + cb_adj
adjusted_confidence = min(100, max(0, conf + total_adjustment))

# Small "base · options · CB" tag under the headline confidence number,
# coloured by the net adjustment.
if total_adjustment > 0:
    tag_cls = "conf-tag-pos"
elif total_adjustment < 0:
    tag_cls = "conf-tag-neg"
else:
    tag_cls = "conf-tag-neutral"
conf_tag_html = (
    f"<div class='conf-tag {tag_cls}'>base: {conf:.1f}% · "
    f"options: {opt_adj:+.1f}% · CB: {cb_adj:+.1f}%</div>"
)

# Count strategies aligned with the headline bias.
strat_keys = ("s1", "s2", "s4", "s5")
aligned = [k for k in strat_keys if signal.get(f"{k}_signal") == bias]
if bias == "NEUTRAL":
    aligned_txt = "Mixed signals — no directional bias"
else:
    names = ", ".join(k.upper() for k in aligned) or "none"
    aligned_txt = f"{len(aligned)}/4 strategies aligned · {names}"

bk = bias_key(bias)
st.markdown(
    f"""
<div class="signal-card signal-{bk}">
  <div>
    <div class="signal-label-{bk}">{bias}</div>
    <div class="signal-sub">{aligned_txt}</div>
    <div class="signal-sub">Position size · <b style="color:#e8e8e8;">{pos:.2f}x</b></div>
  </div>
  <div style="min-width:160px;">
    <div class="conf-pct">{adjusted_confidence:.1f}%</div>
    <div class="conf-label">CONFIDENCE</div>
    <div style="background:#1a1a1a;border-radius:6px;height:6px;width:100%;margin-top:8px;">
      <div style="background:{bias_hex(bias)};height:6px;border-radius:6px;width:{min(max(adjusted_confidence,0),100):.0f}%;"></div>
    </div>
    {conf_tag_html}
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── intraday confirmation (most actionable — directly under the signal card) ───
st.markdown('<div class="section-header">Intraday confirmation</div>', unsafe_allow_html=True)

if not intraday or intraday.get("error"):
    err = (intraday or {}).get("error") or "Intraday data unavailable"
    st.markdown(
        f"<div class='session-card' style='color:#888;font-size:0.85rem;text-align:center;'>"
        f"Intraday layer unavailable — {err}</div>",
        unsafe_allow_html=True,
    )
else:
    confirm = intraday.get("confirmation") or {}
    pa = intraday.get("price_action") or {}
    sess = intraday.get("levels") or {}

    # 1. Confirmation banner — full width. Signal may have been upgraded /
    #    downgraded by the volume filter (ENTER STRONG / WEAK, WATCH CLOSELY,
    #    POTENTIAL REVERSAL) so map each variant to its own color + icon.
    sig = confirm.get("signal", "WAIT")
    reason = confirm.get("reason", "")
    vol_note = confirm.get("volume_note", "")

    _SIG_BANNER = {
        "ENTER STRONG":       ("confirm-enter-strong", "✅✅ ENTRY CONFIRMED (STRONG)"),
        "ENTER":              ("confirm-enter",        "✅ ENTRY CONFIRMED"),
        "ENTER WEAK":         ("confirm-enter-weak",   "⚠️ ENTRY CONFIRMED (WEAK)"),
        "WATCH CLOSELY":      ("confirm-wait",         "👁️ WATCH CLOSELY"),
        "WAIT":               ("confirm-wait",         "⏳ WAITING FOR CONFIRMATION"),
        "AGAINST":            ("confirm-against",      "🚫 PRICE ACTING AGAINST BIAS"),
        "POTENTIAL REVERSAL": ("confirm-reversal",     "⚡ POTENTIAL REVERSAL"),
    }
    banner_cls, banner_label = _SIG_BANNER.get(sig, _SIG_BANNER["WAIT"])

    # Sub-line: entry zone for ENTER variants, watched level while waiting.
    if sig.startswith("ENTER"):
        zone = confirm.get("entry_zone")
        sub = f"Entry zone: {zone}" if zone else ""
    elif sig in ("WAIT", "WATCH CLOSELY"):
        key_level = confirm.get("key_level")
        sub = f"Watching: ${key_level}" if key_level is not None else ""
    elif sig == "AGAINST":
        sub = "Do not enter until price returns to bias direction"
    else:  # POTENTIAL REVERSAL
        sub = ""

    sub_html = f"<div class='confirm-sub'>{sub}</div>" if sub else ""
    note_html = f"<div class='confirm-sub'>{vol_note}</div>" if vol_note else ""
    st.markdown(
        f"<div class='{banner_cls}'>"
        f"<div class='confirm-reason'>{banner_label} — {reason}</div>"
        f"{sub_html}{note_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 2. Session levels row — London + NY, side by side.
    def _session_card(label: str, lv: dict | None) -> str:
        if not lv:
            return (
                f"<div class='session-card' style='opacity:0.45;'>"
                f"<div class='session-label'>{label} session</div>"
                f"<div class='session-levels'>Not active yet today</div>"
                f"</div>"
            )
        return (
            f"<div class='session-card'>"
            f"<div class='session-label'>{label} session</div>"
            f"<div class='session-levels'>High <b>${lv['high']:,.2f}</b> · "
            f"Low <b>${lv['low']:,.2f}</b> · Range <b>${lv['range']:,.2f}</b></div>"
            f"</div>"
        )

    c_lon, c_ny = st.columns(2)
    c_lon.markdown(_session_card("London", sess.get("london")), unsafe_allow_html=True)
    c_ny.markdown(_session_card("NY", sess.get("ny")), unsafe_allow_html=True)

    ticker_used = intraday.get("ticker")
    if ticker_used:
        label = {"XAUUSD=X": "XAU/USD spot", "GC=F": "GC=F futures",
                 "MGC=F": "MGC=F micro futures", "GLD": "GLD ETF"}.get(
                     ticker_used, ticker_used)
        premium = intraday.get("spot_premium") or 0.0
        offset_txt = f", −${premium:.0f} spot offset" if premium else ""
        st.markdown(
            f"<div style='color:#555;font-size:0.72rem;margin:2px 0 0.75rem 0;'>"
            f"Intraday data: {label} ({ticker_used}){offset_txt}</div>",
            unsafe_allow_html=True,
        )

    # 3. Price action strip — single row.
    if pa:
        current = pa.get("current")
        change = pa.get("change_5m", 0.0)
        slope = pa.get("ema9_slope", 0.0)
        vol_ratio = pa.get("vol_ratio", 1.0)

        if slope > 0.05:
            slope_txt = "↑ positive"
        elif slope < -0.05:
            slope_txt = "↓ negative"
        else:
            slope_txt = "→ flat"

        chg_sign = f"{change:+,.2f}"
        st.markdown(
            f"<div class='price-strip'>"
            f"Current: <span>${current:,.2f}</span>"
            f"5m change: <span>${chg_sign}</span>"
            f"EMA9 slope: <span>{slope_txt}</span>"
            f"Volume: <span>{vol_ratio:.1f}x avg</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # 4. Volume strip — volume context for the confirmation signal.
    vol = intraday.get("volume") or {}
    if vol:
        vr = vol.get("vol_ratio", 1.0)
        vt = vol.get("vol_trend", "flat")
        vreg = vol.get("vol_regime", "NORMAL")
        hvb = vol.get("high_vol_bars", 0)
        trend_txt = {"rising": "↑rising", "falling": "↓falling",
                     "flat": "→flat"}.get(vt, "→flat")
        reg_cls = {"HIGH": "vol-high", "LOW": "vol-low"}.get(vreg, "")
        reg_html = (f"<span class='{reg_cls}'>{vreg}</span>" if reg_cls
                    else f"<span>{vreg}</span>")
        st.markdown(
            f"<div class='vol-strip'>"
            f"Vol ratio: <span>{vr:.1f}x avg</span>"
            f"Trend: <span>{trend_txt}</span>"
            f"Regime: {reg_html}"
            f"High vol bars (last 20): <span>{hvb}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # 5. Climax warning bar — volume exhaustion flag.
        if vol.get("climax"):
            cdir = vol.get("climax_direction") or ""
            st.markdown(
                f"<div class='vol-climax'>⚠️ Volume climax detected — "
                f"{cdir} exhaustion</div>",
                unsafe_allow_html=True,
            )

    # 6. Suggested size note — when volume reduces the V4 position size.
    size_mod = confirm.get("size_modifier", 1.0)
    if size_mod < 1.0 and "ENTER" in sig:
        vq = confirm.get("volume_quality", "")
        st.markdown(
            f"<div class='size-note'>Suggested size: {pos * size_mod:.2f}x "
            f"(reduced for {vq} volume)</div>",
            unsafe_allow_html=True,
        )


# ── next event countdown (under the signal card) ───────────────────────────────
if next_event is not None:
    now_utc = datetime.now(next_event["datetime_utc"].tzinfo)
    delta = next_event["datetime_utc"] - now_utc
    total_min = max(int(delta.total_seconds() // 60), 0)
    hrs, mins = divmod(total_min, 60)
    when = f"{hrs}h {mins}m" if hrs else f"{mins}m"
    st.markdown(
        f"<div class='next-event'>Next: <span>{next_event['title']}</span> "
        f"in <span>{when}</span> · {next_event['time_sofia']} Sofia</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='next-event'>No high-impact events remaining this week</div>",
        unsafe_allow_html=True,
    )


# ── risk banner (full width, above everything below the signal card) ───────────
if risk_score in ("HIGH", "MEDIUM"):
    titles = " · ".join(e["title"] for e in today_events)
    if risk_score == "HIGH":
        st.markdown(
            f"<div class='risk-high'>⚠️ HIGH IMPACT EVENT TODAY — {titles}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='risk-medium'>📅 Market event today — {titles}</div>",
            unsafe_allow_html=True,
        )


# ── metrics row ────────────────────────────────────────────────────────────────
vol = str(signal.get("vol_regime") or "—")
vol_cls = {"normal": "metric-neutral", "elevated": "metric-amber", "extreme": "metric-negative"}.get(vol, "metric-neutral")

sma = signal.get("sma_200")
sma_txt = f"${float(sma):,.2f}" if sma is not None else "—"

cb_active = bool(signal.get("circuit_breaker_active"))
cb_txt = "Active" if cb_active else "Inactive"
cb_cls = "metric-negative" if cb_active else "metric-positive"

score = signal.get("signal_score")
if score is not None:
    score_val = float(score)
    score_txt = f"{score_val:+.3f}"
    score_cls = "metric-positive" if score_val > 0 else ("metric-negative" if score_val < 0 else "metric-neutral")
else:
    score_txt, score_cls = "—", "metric-neutral"

st.markdown(
    f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Vol regime</div>
    <div class="metric-value {vol_cls}">{vol.upper()}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">200d SMA</div>
    <div class="metric-value">{sma_txt}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Circuit breaker</div>
    <div class="metric-value {cb_cls}">{cb_txt}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Signal score</div>
    <div class="metric-value {score_cls}">{score_txt}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── strategy breakdown ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Strategy breakdown</div>', unsafe_allow_html=True)

strat_rows_html = ""
for k in strat_keys:
    sig = signal.get(f"{k}_signal")
    if sig is None:
        continue
    driver = signal.get(f"{k}_driver", "")
    sk = bias_key(sig)
    strat_rows_html += (
        f"<div class='strat-row'>"
        f"<div><div class='strat-name'>{k.upper()}</div>"
        f"<div class='strat-driver'>{driver}</div></div>"
        f"<div class='badge-{sk}'>{sig}</div>"
        f"</div>"
    )
st.markdown(f'<div class="strat-card">{strat_rows_html}</div>', unsafe_allow_html=True)


# ── economic calendar (this week) ──────────────────────────────────────────────
def _to_float(v: str) -> float | None:
    """Parse a Forex Factory numeric string (e.g. '116K', '0.3%', '4.3%')."""
    if not v:
        return None
    s = str(v).strip().replace("%", "").replace(",", "").replace("$", "")
    mult = 1.0
    if s and s[-1].upper() in ("K", "M", "B", "T"):
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[s[-1].upper()]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _actual_cell(ev: dict) -> str:
    """Actual value cell: green if beats forecast, red if misses, gray if pending."""
    actual = ev.get("actual", "")
    if not actual:
        return "<span class='cal-actual-pending'>—</span>"
    a, f = _to_float(actual), _to_float(ev.get("forecast", ""))
    if a is None or f is None or a == f:
        cls = "cal-actual-pending"
    elif a > f:
        cls = "cal-actual-better"
    else:
        cls = "cal-actual-worse"
    return f"<span class='{cls}'>{actual}</span>"


st.markdown('<div class="section-header">Economic calendar</div>', unsafe_allow_html=True)

with st.expander("This week · high-impact USD events", expanded=False):
    if not week_events:
        st.markdown(
            "<div style='color:#888;font-size:0.85rem;'>Calendar unavailable.</div>",
            unsafe_allow_html=True,
        )
    else:
        today = datetime.now().date()
        now_utc = datetime.now(week_events[0]["datetime_utc"].tzinfo)
        rows_html = ""
        for ev in week_events:
            if ev["date"] == today:
                row_cls = "cal-row-today"
            elif ev["datetime_utc"] < now_utc:
                row_cls = "cal-row-past"
            else:
                row_cls = "cal-row-future"
            day = ev["datetime_sofia"].strftime("%a %b %d")
            forecast = ev.get("forecast") or "—"
            previous = ev.get("previous") or "—"
            rows_html += (
                f"<tr class='{row_cls}'>"
                f"<td>{day}</td>"
                f"<td>{ev['time_sofia']}</td>"
                f"<td>{ev['title']}</td>"
                f"<td>{forecast}</td>"
                f"<td>{previous}</td>"
                f"<td>{_actual_cell(ev)}</td>"
                f"</tr>"
            )
        st.markdown(
            "<table class='cal-table'>"
            "<thead><tr>"
            "<th>Day</th><th>Time (Sofia)</th><th>Event</th>"
            "<th>Forecast</th><th>Previous</th><th>Actual</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )


# ── correlation monitor ────────────────────────────────────────────────────────
_CORR_BADGE = {"ALIGNED": "aligned", "MIXED": "mixed", "BREAKDOWN": "breakdown"}
_CORR_BADGE_TXT = {
    "ALIGNED": "ALIGNED",
    "MIXED": "MIXED",
    "BREAKDOWN": "⚠️ BREAKDOWN",
}
_CORR_INTERP = {
    "ALIGNED": "All major correlations holding. Macro context supports signal.",
    "MIXED": "Some correlation breakdowns detected. Verify signal with context.",
    "BREAKDOWN": "⚠️ Multiple correlation breakdowns. Regime change possible — "
                 "reduce confidence in ensemble signal.",
}

badge_cls = _CORR_BADGE.get(corr_summary, "aligned")
st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:center;"
    f"margin-bottom:0.75rem;'>"
    f"<div class='section-header' style='margin-bottom:0;'>Correlation monitor</div>"
    f"<div class='badge-{badge_cls}'>{_CORR_BADGE_TXT.get(corr_summary, corr_summary)}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

if not corr_data:
    st.markdown(
        "<div class='strat-card' style='color:#888;font-size:0.85rem;text-align:center;'>"
        "Correlation data unavailable</div>",
        unsafe_allow_html=True,
    )
else:
    # 6 cards in 2 rows of 3 (Streamlit columns).
    items = list(corr_data.values())
    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, v in zip(cols, items[row_start:row_start + 3]):
            chg = v["change_pct"]
            chg_cls = "price-positive" if chg >= 0 else "price-negative"
            card_cls = "corr-card breakdown" if v["breakdown"] else "corr-card"
            msg_html = (
                f"<div class='corr-breakdown-msg'>{v['breakdown_msg']}</div>"
                if v["breakdown"] else ""
            )
            col.markdown(
                f"<div class='{card_cls}'>"
                f"<div class='corr-label'>{v['label']}</div>"
                f"<div class='corr-price'>{v['current']:,.2f} "
                f"<span class='{chg_cls}' style='font-size:0.8rem;'>{chg:+.2f}%</span></div>"
                f"<div class='corr-stat'>30d: {v['corr_30d']:+.2f}</div>"
                f"<div class='corr-stat'>5d: {v['corr_5d']:+.2f}</div>"
                f"<div class='corr-stat'>normally {v['normal']}</div>"
                f"{msg_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown(
    f"<div class='corr-interp'>{_CORR_INTERP.get(corr_summary, '')}</div>",
    unsafe_allow_html=True,
)


# ── options positioning (GLD put/call) ─────────────────────────────────────────
opt_signal = positioning.get("signal", "NEUTRAL")
opt_sk = bias_key(opt_signal)

st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:center;"
    f"margin-bottom:0.75rem;'>"
    f"<div class='section-header' style='margin-bottom:0;'>Options positioning</div>"
    f"<div class='badge-{opt_sk}'>{opt_signal}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

if positioning.get("error") or positioning.get("pcr_oi") is None:
    err = positioning.get("error") or "Options data unavailable"
    st.markdown(
        f"<div class='strat-card' style='color:#888;font-size:0.85rem;text-align:center;'>"
        f"Options layer unavailable — {err}</div>",
        unsafe_allow_html=True,
    )
else:
    pcr_oi = float(positioning["pcr_oi"])
    pcr_vol = float(positioning["pcr_vol"])
    opt_score = float(positioning.get("score", 0) or 0)
    iv_skew = float(positioning.get("iv_skew", 0) or 0)

    def _pcr_cls(pcr: float) -> str:
        # Low PCR = more calls = bullish (green); high PCR = bearish (red).
        if pcr < 0.8:
            return "metric-positive"
        if pcr > 1.2:
            return "metric-negative"
        return "metric-neutral"

    score_cls = ("metric-positive" if opt_score > 0
                 else "metric-negative" if opt_score < 0 else "metric-neutral")

    # IV skew: positive = put premium = fear (bearish) → red; negative → green.
    skew_cls = ("metric-negative" if iv_skew > 0.05
                else "metric-positive" if iv_skew < -0.05 else "metric-neutral")

    # Four metric cards: OI PCR, Volume PCR, net positioning score, IV skew.
    st.markdown(
        f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">OI Put/Call</div>
    <div class="metric-value {_pcr_cls(pcr_oi)}">{pcr_oi:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Volume Put/Call</div>
    <div class="metric-value {_pcr_cls(pcr_vol)}">{pcr_vol:.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Net positioning</div>
    <div class="metric-value {score_cls}">{opt_score:+.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">IV Skew (puts−calls)</div>
    <div class="metric-value {skew_cls}">{iv_skew:+.3f}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#555;font-size:0.72rem;margin:-0.5rem 0 0.75rem 0;'>"
        "Positive IV skew = put buyers paying premium = fear</div>",
        unsafe_allow_html=True,
    )

    # Expirations used — so the trader knows how fresh the chain is.
    exps = positioning.get("expirations") or []
    if exps:
        st.markdown(
            f"<div style='color:#555;font-size:0.72rem;margin:-0.5rem 0 0.75rem 0;'>"
            f"Using expirations: {' · '.join(str(e) for e in exps)}</div>",
            unsafe_allow_html=True,
        )

    # Unusual activity → amber alert.
    unusual = positioning.get("unusual")
    if unusual:
        st.markdown(
            f"<div class='opt-unusual'>⚡ {unusual}</div>",
            unsafe_allow_html=True,
        )

    # Adjustment line — confirms (green) / conflicts (amber) / neutral (gray).
    direction = adjustment.get("direction", "neutral")
    msg = adjustment.get("message", "")
    if direction == "confirms":
        st.markdown(f"<div class='opt-confirms'>✓ {msg}</div>", unsafe_allow_html=True)
    elif direction == "conflicts":
        st.markdown(f"<div class='opt-conflicts'>⚠️ {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='opt-neutral-msg'>{msg}</div>", unsafe_allow_html=True)


# ── central bank positioning (official-sector gold reserves) ────────────────────
cb_trend = cb_analysis.get("trend", "UNKNOWN")
# ACCUMULATING → green, REDUCING → red, STABLE / UNKNOWN → gray.
_CB_BADGE = {"ACCUMULATING": "bullish", "REDUCING": "bearish"}
cb_badge = _CB_BADGE.get(cb_trend, "neutral")

st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:center;"
    f"margin-bottom:0.4rem;'>"
    f"<div class='section-header' style='margin-bottom:0;'>Central bank positioning</div>"
    f"<div class='badge-{cb_badge}'>{cb_trend}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

if cb_data.get("error") or not cb_analysis or cb_trend == "UNKNOWN":
    err = cb_data.get("error") or "Central bank data unavailable"
    st.markdown(
        f"<div class='strat-card' style='color:#888;font-size:0.85rem;text-align:center;'>"
        f"Central bank layer unavailable — {err}</div>",
        unsafe_allow_html=True,
    )
else:
    # 1. Summary line + freshness caption.
    st.markdown(
        f"<div style='color:#e8e8e8;font-size:0.9rem;margin-bottom:2px;'>"
        f"{cb_analysis.get('summary', '')}</div>"
        f"<div style='color:#555;font-size:0.72rem;margin-bottom:0.75rem;'>"
        f"{cb_analysis.get('data_note', 'World Bank annual data · updates quarterly')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 2. Two columns — buyers (green) on the left, sellers/stable on the right.
    buyers = cb_analysis.get("buyers", []) or []
    sellers = cb_analysis.get("sellers", []) or []
    stable = cb_analysis.get("stable", []) or []

    def _cb_rows(items, color):
        if not items:
            return "<div style='color:#555;font-size:0.8rem;'>None</div>"
        rows = []
        for it in items:
            flag = it.get("flag", "")
            rows.append(
                f"<div style='font-size:0.85rem;margin:3px 0;'>"
                f"{flag} {it['country']} "
                f"<span style='color:{color};'>{it['change_pct']:+.1f}%</span></div>"
            )
        return "".join(rows)

    stable_rows = (
        "".join(
            f"<div style='color:#888780;font-size:0.85rem;margin:3px 0;'>"
            f"{COUNTRY_FLAGS_DASH.get(name, '')} {name}</div>"
            for name in stable
        )
        if stable else "<div style='color:#555;font-size:0.8rem;'>None</div>"
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f"<div class='metric-label'>Buyers</div>{_cb_rows(buyers, GREEN)}",
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            f"<div class='metric-label'>Sellers</div>{_cb_rows(sellers, RED)}"
            f"<div class='metric-label' style='margin-top:0.75rem;'>Stable</div>"
            f"{stable_rows}",
            unsafe_allow_html=True,
        )

    # 3. Net reserve change across the tracked central banks.
    net_pct = float(cb_analysis.get("net_change_pct", 0) or 0)
    net_cls = ("metric-positive" if net_pct > 0
               else "metric-negative" if net_pct < 0 else "metric-neutral")
    st.markdown(
        f"<div style='margin:0.9rem 0 0.4rem 0;font-size:0.85rem;color:#888;'>"
        f"Net reserves change: "
        f"<span class='{net_cls}' style='font-weight:500;'>{net_pct:+.2f}%</span> "
        f"across tracked central banks</div>",
        unsafe_allow_html=True,
    )

    # 4. Adjustment line — confirms (green) / conflicts (amber) / neutral (gray).
    cb_dir = cb_adjustment.get("direction", "neutral")
    cb_msg = cb_adjustment.get("message", "")
    if cb_dir == "confirms":
        st.markdown(f"<div class='opt-confirms'>✓ {cb_msg}</div>", unsafe_allow_html=True)
    elif cb_dir == "conflicts":
        st.markdown(f"<div class='opt-conflicts'>⚠️ {cb_msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='opt-neutral-msg'>{cb_msg}</div>", unsafe_allow_html=True)

    # 5. Structural-context caveat.
    st.markdown(
        "<div style='color:#666;font-size:0.72rem;font-style:italic;margin-top:0.5rem;'>"
        "⚠️ Annual data — use as structural context only, not a timing signal</div>",
        unsafe_allow_html=True,
    )


# ── news sentiment (last 48h) ──────────────────────────────────────────────────
import html as _html

st.markdown('<div class="section-header">News sentiment</div>', unsafe_allow_html=True)

if not sentiment or sentiment.get("total", 0) == 0:
    raw_n = sentiment.get("raw_count", 0) if sentiment else 0
    msg = (sentiment.get("no_data_msg") if sentiment else None) \
        or "No gold-relevant headlines found (or NEWSAPI_KEY not configured)"
    if raw_n:
        msg += f" · {raw_n} headlines fetched but none passed the relevance filter"
    st.info(msg)
else:
    sig = sentiment["signal"]
    sk = bias_key(sig)                       # bullish / bearish / neutral CSS suffix
    sconf = float(sentiment["confidence"])
    pol = float(sentiment["avg_polarity"])

    # 1. Header + sentiment badge + confidence
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:0.75rem;'>"
        f"<div class='sentiment-label' style='font-size:0.8rem;'>Last 48h</div>"
        f"<div><span class='badge-{sk}'>{sig}</span>"
        f"<span class='sentiment-label' style='margin-left:8px;'>{sconf:.0f}% conf</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Sub-line: how many relevant headlines were analyzed vs raw fetch.
    raw_n = sentiment.get("raw_count", sentiment["total"])
    st.markdown(
        f"<div class='sentiment-label' style='margin:-0.25rem 0 0.75rem 0;'>"
        f"Analyzed {sentiment['total']} relevant headlines "
        f"(filtered from {raw_n} fetched)</div>",
        unsafe_allow_html=True,
    )

    # 2. Stats row — total / bullish / bearish
    st.markdown(
        f"""
<div class="metric-grid" style="grid-template-columns: repeat(3, 1fr);">
  <div class="metric-card">
    <div class="metric-label">Headlines</div>
    <div class="metric-value">{sentiment['total']}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Bullish</div>
    <div class="metric-value metric-positive">{sentiment['bullish_count']}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Bearish</div>
    <div class="metric-value metric-negative">{sentiment['bearish_count']}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # 3. Polarity bar (-1 .. +1), filled from center toward the current avg.
    pct = min(abs(pol), 1.0) * 50.0          # half-width fraction
    if pol >= 0:
        bar = (f"<div class='polarity-fill-bull' style='position:absolute;left:50%;"
               f"width:{pct:.1f}%;'></div>")
    else:
        bar = (f"<div class='polarity-fill-bear' style='position:absolute;"
               f"right:50%;width:{pct:.1f}%;'></div>")
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;'>"
        f"<span class='sentiment-label'>Bearish</span>"
        f"<span class='sentiment-label'>avg polarity {pol:+.3f}</span>"
        f"<span class='sentiment-label'>Bullish</span></div>"
        f"<div class='polarity-track'>"
        f"<div style='position:absolute;left:50%;top:-2px;width:1px;height:10px;"
        f"background:#444;'></div>{bar}</div>",
        unsafe_allow_html=True,
    )

    # 4. Divergence vs ensemble bias
    from data.sentiment import divergence_check
    div = divergence_check(sig, bias)
    if div["message"]:
        if div["divergence"]:
            st.markdown(
                f"<div class='divergence-alert'>{div['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='sentiment-aligned'>{div['message']}</div>",
                unsafe_allow_html=True,
            )

    # 5. Top headlines — two columns, collapsible.
    def _headlines_html(items: list[dict]) -> str:
        if not items:
            return "<div class='headline-meta'>None</div>"
        out = ""
        for h in items:
            title = _html.escape(h.get("title", ""))
            url = _html.escape(h.get("url", ""), quote=True)
            src = _html.escape(h.get("source", ""))
            title_html = (f"<a class='headline-title' href='{url}' target='_blank'>{title}</a>"
                          if url else f"<span class='headline-title'>{title}</span>")
            out += (
                f"<div class='headline-item'>{title_html}"
                f"<div class='headline-meta'>{src} · polarity {h.get('polarity', 0):+.2f}"
                f" · rel {h.get('relevance', 0):.1f}</div>"
                f"</div>"
            )
        return out

    with st.expander("Top headlines", expanded=False):
        c_bull, c_bear = st.columns(2)
        c_bull.markdown(
            "<div class='sentiment-label' style='margin-bottom:6px;'>📈 Most bullish</div>"
            f"{_headlines_html(sentiment['top_bullish'])}",
            unsafe_allow_html=True,
        )
        c_bear.markdown(
            "<div class='sentiment-label' style='margin-bottom:6px;'>📉 Most bearish</div>"
            f"{_headlines_html(sentiment['top_bearish'])}",
            unsafe_allow_html=True,
        )


# ── confidence chart (last 14 days) ────────────────────────────────────────────
st.markdown('<div class="section-header">Confidence · last 14 days</div>', unsafe_allow_html=True)

hist14 = list(reversed(recent_signals(14)))  # oldest → newest

if len(hist14) < 2:
    st.markdown(
        "<div class='strat-card' style='color:#888;font-size:0.85rem;text-align:center;'>"
        "Chart will populate after 2+ days of signals</div>",
        unsafe_allow_html=True,
    )
else:
    # X axis = date only (category axis keeps plotly from showing a timestamp).
    x = [str(r["date"]) for r in hist14]
    y = [float(r["confidence"]) for r in hist14]
    biases = [r["bias"] for r in hist14]

    fig = go.Figure()
    # Line chart with markers; each segment colored by the day it arrives at.
    for i in range(len(x) - 1):
        fig.add_trace(go.Scatter(
            x=x[i:i + 2], y=y[i:i + 2], mode="lines",
            line=dict(color=bias_hex(biases[i + 1]), width=2),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=[bias_hex(b) for b in biases], size=6),
        showlegend=False,
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        height=160,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(type="category", gridcolor="#1e1e1e",
                   tickfont=dict(color="#888", size=11)),
        yaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#888", size=11),
                   ticksuffix="%", range=[40, 85]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── signal history log (last 7 days) ───────────────────────────────────────────
st.markdown('<div class="section-header">Signal history</div>', unsafe_allow_html=True)

hist7 = recent_signals(7)  # newest first
hist_rows_html = ""
for r in hist7:
    b = r["bias"]
    bk = bias_key(b)
    try:
        d = datetime.strptime(str(r["date"]), "%Y-%m-%d").strftime("%b %d")
    except Exception:
        d = str(r["date"])
    pos_r  = float(r["position_size"]) if r.get("position_size") is not None else 0.0
    conf_r = float(r["confidence"]) if r.get("confidence") is not None else 0.0
    vol_r  = str(r.get("vol_regime") or "")
    hist_rows_html += (
        f"<div class='hist-row'>"
        f"<div class='hist-date'>{d}</div>"
        f"<div class='hist-{bk}'>{b}</div>"
        f"<div class='hist-size'>{pos_r:.1f}x</div>"
        f"<div class='hist-conf'>{conf_r:.1f}%</div>"
        f"<div class='hist-vol'>{vol_r}</div>"
        f"</div>"
    )
st.markdown(f'<div class="hist-card">{hist_rows_html}</div>', unsafe_allow_html=True)


# ── last updated line ──────────────────────────────────────────────────────────
st.markdown(
    f"<div style='color:#555;font-size:0.72rem;text-align:center;margin-top:0.5rem;'>"
    f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S} · Auto-refreshes every 60s</div>",
    unsafe_allow_html=True,
)
