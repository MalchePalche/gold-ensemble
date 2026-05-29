"""
dashboard/app.py — Gold Ensemble V4 Streamlit dashboard.

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

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.queries import get_latest_signal, get_recent_signals


# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Ensemble V4 — XAU/USD",
    page_icon="G",
    layout="wide",
)

# Re-run the script every 60s so live price + signals stay fresh.
st_autorefresh(interval=60_000, key="refresh")

BIAS_COLOR = {"BULLISH": "#16a34a", "BEARISH": "#dc2626", "NEUTRAL": "#6b7280"}


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


# ── sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Gold Ensemble V4")
if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

# ── load ─────────────────────────────────────────────────────────────────────
try:
    signal = latest_signal()
except Exception as e:
    st.error(f"Failed to read signals from Supabase: {e}")
    st.stop()

if not signal:
    st.title("XAU/USD Daily Bias — Gold Ensemble V4")
    st.warning("No signals found yet. Run `python run_daily.py` to populate Supabase.")
    st.stop()

bias        = signal["bias"]
signal_date = signal["date"]
stored_price = float(signal["price"]) if signal.get("price") is not None else None
price_now    = live_price()

# ── header ───────────────────────────────────────────────────────────────────
st.title("XAU/USD Daily Bias — Gold Ensemble V4")
head_l, head_r = st.columns([3, 2])
with head_l:
    st.subheader(f"Signal as of {signal_date}")

with head_r:
    if price_now is not None:
        # Change vs. the price stored with the latest signal.
        delta_txt = None
        if stored_price:
            chg = price_now - stored_price
            pct = chg / stored_price * 100.0
            delta_txt = f"{chg:+,.2f} ({pct:+.2f}%)"
        c_price, c_badge = st.columns([4, 1])
        with c_price:
            st.metric("XAU/USD (live)", f"${price_now:,.2f}", delta=delta_txt)
        with c_badge:
            st.markdown(
                "<div style='margin-top:18px;padding:3px 8px;border-radius:6px;"
                "background:#dc2626;color:white;font-size:11px;font-weight:700;"
                "text-align:center;'>● LIVE</div>",
                unsafe_allow_html=True,
            )
    elif stored_price:
        st.metric("XAU/USD (last signal)", f"${stored_price:,.2f}")

# ── bias card + metrics ───────────────────────────────────────────────────────
st.markdown("---")
b_col, conf_col, vol_col, pos_col = st.columns([3, 1, 1, 1])

with b_col:
    st.markdown(
        f"""
        <div style="padding:24px;border-radius:10px;background:{BIAS_COLOR.get(bias, '#6b7280')};color:white;text-align:center;">
            <div style="font-size:13px;opacity:0.85;">TODAY'S BIAS</div>
            <div style="font-size:42px;font-weight:700;margin-top:6px;">{bias}</div>
            <div style="font-size:17px;opacity:0.9;margin-top:4px;">{float(signal['confidence']):.1f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with conf_col:
    score = signal.get("signal_score")
    st.metric("Signal score", f"{float(score):+.3f}" if score is not None else "—")
    sma = signal.get("sma_200")
    if sma is not None:
        st.caption(f"200-DMA ${float(sma):,.2f}")

with vol_col:
    st.metric("Vol regime", str(signal.get("vol_regime", "—")).upper())

with pos_col:
    st.metric("V4 position", f"{float(signal['position_size']):.2f}x")
    if signal.get("circuit_breaker_active"):
        st.caption("Circuit breaker ACTIVE")

# ── strategy breakdown ───────────────────────────────────────────────────────
st.markdown("### Strategy breakdown")
strat_rows = []
for key in ("s1", "s2", "s4", "s5"):
    sig = signal.get(f"{key}_signal")
    if sig is None:
        continue
    arrow = {"BULLISH": "UP", "BEARISH": "DN", "NEUTRAL": "--"}.get(sig, "--")
    aligned = ""
    if bias != "NEUTRAL" and sig != "NEUTRAL":
        aligned = "YES" if sig == bias else "no"
    strat_rows.append({
        "Strategy": f"{key.upper()} — {signal.get(f'{key}_driver', '')}",
        "Signal"  : arrow,
        "Aligned" : aligned,
    })
if strat_rows:
    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

# ── confidence chart (last 14 days) ───────────────────────────────────────────
st.markdown("### Confidence — last 14 days")
hist14 = recent_signals(14)
if hist14:
    df14 = pd.DataFrame(hist14).iloc[::-1]  # oldest → newest for the chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df14["date"], y=df14["confidence"].astype(float),
        marker_color=[BIAS_COLOR.get(b, "#6b7280") for b in df14["bias"]],
        name="Confidence",
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Confidence (%)", range=[0, 100]),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── history log (last 7 days) ──────────────────────────────────────────────────
st.markdown("### History — last 7 days")
hist7 = recent_signals(7)
if hist7:
    log = pd.DataFrame(hist7)[
        ["date", "price", "bias", "confidence", "vol_regime", "position_size"]
    ].copy()
    log = log.rename(columns={
        "price": "close", "confidence": "confidence_pct", "position_size": "position_v4",
    })
    st.dataframe(log, use_container_width=True, hide_index=True)

st.caption(
    f"Refreshed {datetime.now():%Y-%m-%d %H:%M:%S}  ·  "
    "signals via Supabase  ·  live price via yfinance  ·  Gold Ensemble V4"
)
