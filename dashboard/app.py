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
    <div class="conf-pct">{conf:.1f}%</div>
    <div class="conf-label">CONFIDENCE</div>
    <div style="background:#1a1a1a;border-radius:6px;height:6px;width:100%;margin-top:8px;">
      <div style="background:{bias_hex(bias)};height:6px;border-radius:6px;width:{min(max(conf,0),100):.0f}%;"></div>
    </div>
  </div>
</div>
""",
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
