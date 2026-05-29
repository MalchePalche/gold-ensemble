"""
dashboard/web_dashboard.py — Flask web dashboard for Gold Ensemble V4.

Start:  python dashboard/web_dashboard.py
        — or double-click run_dashboard.bat —

Serves:  http://localhost:5000
  GET /          Full dashboard (page refresh every 60s, data cached 15min)
  GET /api/price Live XAU/USD price JSON (cached 60s, polled every 30s by JS)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template_string

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.loader import DataLoader
from ensemble.model import EnsembleModel
from ensemble.sizer import compute_vol_regime, target_size, simulate_v4

app = Flask(__name__)

# signal data — expensive, 15-min TTL
_sig_cache: dict = {"d": None, "ts": 0.0}
SIG_TTL = 15 * 60

# live price — cheap, 60-sec TTL
_price_cache: dict = {"d": None, "ts": 0.0}
PRICE_TTL = 60


# ── live price ─────────────────────────────────────────────────────────────

def _fetch_price() -> dict:
    """Fetch spot price + change via yfinance fast_info. Cached 60s."""
    now = time.time()
    if _price_cache["d"] is not None and (now - _price_cache["ts"]) < PRICE_TTL:
        return _price_cache["d"]

    try:
        fi   = yf.Ticker("GC=F").fast_info
        last = float(fi.last_price)
        prev = float(fi.previous_close)
        if not last or not prev:
            raise ValueError("null price")
        chg     = last - prev
        chg_pct = chg / prev * 100
        result  = {
            "price"      : round(last, 2),
            "change"     : round(chg, 2),
            "change_pct" : round(chg_pct, 2),
            "updated"    : datetime.now().strftime("%H:%M:%S"),
            "ok"         : True,
        }
    except Exception:
        # Return stale data if we have it, otherwise signal an error
        if _price_cache["d"] is not None:
            result = {**_price_cache["d"], "ok": False,
                      "updated": datetime.now().strftime("%H:%M:%S")}
        else:
            result = {"price": None, "change": 0.0, "change_pct": 0.0,
                      "updated": datetime.now().strftime("%H:%M:%S"), "ok": False}

    _price_cache["d"]  = result
    _price_cache["ts"] = now
    return result


# ── signal computation helpers ─────────────────────────────────────────────

def _driver(key: str, sig: int, close: pd.Series) -> str:
    if key == "S1":
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        gap   = ema20 - ema50
        if sig > 0: return f"EMA 20 above EMA 50  (+${gap:,.0f})"
        if sig < 0: return f"EMA 20 below EMA 50  (${gap:,.0f})"
        return "EMA 20 and EMA 50 converging"

    if key == "S2":
        u = float(close.rolling(55).max().iloc[-1])
        l = float(close.rolling(55).min().iloc[-1])
        if sig > 0: return f"Price above 55-day upper band  (${u:,.0f})"
        if sig < 0: return f"Price below 55-day lower band  (${l:,.0f})"
        return f"No fresh Donchian breakout  ${l:,.0f} – ${u:,.0f}"

    if key == "S4":
        h = float(close.rolling(252).max().iloc[-1])
        l = float(close.rolling(252).min().iloc[-1])
        if sig > 0: return f"Near 52-week high  (${h:,.0f})"
        if sig < 0: return f"Near 52-week low  (${l:,.0f})"
        return f"Mid-range  ${l:,.0f} – ${h:,.0f}"

    if key == "S5":
        macd = (close.ewm(span=12, adjust=False).mean()
                - close.ewm(span=26, adjust=False).mean())
        hist = float(macd.iloc[-1] - macd.ewm(span=9, adjust=False).mean().iloc[-1])
        if sig > 0: return f"MACD above signal line  (hist {hist:+.1f})"
        if sig < 0: return f"MACD below signal line  (hist {hist:+.1f})"
        return "MACD near signal line"

    return ""


def _pos_display(pos: float) -> tuple[str, str]:
    if pos == 0.0:  return "flat", "— stay flat"
    if pos <= 0.5:  return "half", "— half position"
    if pos <= 1.0:  return "full", ""
    return "over", "— high-conviction long"


def _compute_signals() -> dict:
    loader = DataLoader(use_cache=True, cache_ttl_hours=1)
    gold   = loader.load_gold(years=15)
    close  = gold["close"]
    br     = close.pct_change().fillna(0.0)

    model  = EnsembleModel()
    series = model.run(gold)
    signal = series.signal
    conf   = series.confidence_pct

    vol_regime, _, _ = compute_vol_regime(close, gold["high"], gold["low"])
    _, pos_v4, _     = simulate_v4(signal, conf, vol_regime, br)

    i          = -1
    today_date = gold.index[i].date()
    today_cls  = float(close.iloc[i])
    today_sig  = int(signal.iloc[i])
    today_conf = float(conf.iloc[i])
    today_vol  = vol_regime.iloc[i]
    today_pos  = float(pos_v4.iloc[i])
    today_scr  = float(series.score.iloc[i])
    bias_str   = {1: "BULLISH", 0: "NEUTRAL", -1: "BEARISH"}[today_sig]
    matrix_tgt = target_size(today_sig, today_conf, today_vol)
    cb_active  = (matrix_tgt > 0.0) and (today_pos == 0.0)
    sma200     = float(close.rolling(200).mean().iloc[i])
    pos_dot, pos_hint = _pos_display(today_pos)

    strat_labels = {
        "S1": "S1 — MA crossover",
        "S2": "S2 — Donchian 55d",
        "S4": "S4 — 52-week momentum",
        "S5": "S5 — MACD",
    }
    strategies = []
    for k, r in series.per_strategy.items():
        sv = int(r.signal.iloc[i])
        strategies.append({
            "name"  : strat_labels.get(k, k),
            "signal": sv,
            "dir"   : "bullish" if sv > 0 else ("bearish" if sv < 0 else "neutral"),
            "label" : "Bullish" if sv > 0 else ("Bearish" if sv < 0 else "Neutral"),
            "driver": _driver(k, sv, close),
        })

    short    = lambda n: n.split("—")[0].strip()
    aligned  = [short(s["name"]) for s in strategies
                if s["signal"] == today_sig and today_sig != 0]
    neutrals = [short(s["name"]) for s in strategies if s["signal"] == 0]
    conflict = [short(s["name"]) for s in strategies
                if s["signal"] != 0 and s["signal"] != today_sig]
    parts = []
    if aligned:  parts.append(", ".join(aligned) + " aligned")
    if neutrals: parts.append(", ".join(neutrals) + " neutral")
    if conflict: parts.append(", ".join(conflict) + " conflicting")
    signal_sub = "  ·  ".join(parts) or "No strategies aligned"

    last14   = gold.index[-14:]
    hist_all = []
    for dt in last14:
        loc = gold.index.get_loc(dt)
        s   = int(signal.iloc[loc])
        c   = float(conf.iloc[loc])
        p   = float(pos_v4.iloc[loc])
        v   = vol_regime.iloc[loc]
        hist_all.append({
            "date"    : f"{dt.day} {dt.strftime('%b')}",
            "bias"    : {1: "Bullish", 0: "Neutral", -1: "Bearish"}[s],
            "dir"     : "bullish" if s > 0 else ("bearish" if s < 0 else "neutral"),
            "position": round(p, 2),
            "conf"    : round(c, 1),
            "vol"     : v.capitalize() + " vol",
            "is_bear" : s == -1,
        })

    conf_labels   = [h["date"]    for h in hist_all]
    conf_values   = [h["conf"]    for h in hist_all]
    bearish_flags = [h["is_bear"] for h in hist_all]
    ymin = max(0,   min(conf_values) - 15) if conf_values else 0
    ymax = min(100, max(conf_values) + 15) if conf_values else 100

    return {
        "date"         : f"{today_date.day} {today_date.strftime('%B %Y')}",
        "close"        : f"${today_cls:,.2f}",
        "bias"         : bias_str,
        "bias_dir"     : bias_str.lower(),
        "confidence"   : round(today_conf, 1),
        "position"     : round(today_pos, 2),
        "pos_dot"      : pos_dot,
        "pos_hint"     : pos_hint,
        "signal_sub"   : signal_sub,
        "score"        : f"{today_scr:+.3f}",
        "score_neg"    : today_scr < 0,
        "vol_regime"   : today_vol.capitalize(),
        "sma200"       : f"${sma200:,.0f}",
        "cb_active"    : cb_active,
        "strategies"   : strategies,
        "history7"     : list(reversed(hist_all))[:7],
        "chart_labels" : json.dumps(conf_labels),
        "chart_values" : json.dumps(conf_values),
        "chart_bearish": json.dumps(bearish_flags),
        "chart_ymin"   : ymin,
        "chart_ymax"   : ymax,
        "updated"      : datetime.now().strftime(
                             f"{today_date.day} %b %Y  ·  %H:%M"),
    }


def _get_signals() -> dict:
    now = time.time()
    if _sig_cache["d"] is None or (now - _sig_cache["ts"]) > SIG_TTL:
        print("[dashboard] Computing signals…", flush=True)
        _sig_cache["d"]  = _compute_signals()
        _sig_cache["ts"] = now
        print("[dashboard] Ready.", flush=True)
    return _sig_cache["d"]


# ── HTML template ──────────────────────────────────────────────────────────

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>XAU/USD &middot; V4 Ensemble</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.31.0/dist/tabler-icons.min.css">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --sans:    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --txt:     #ffffff;
      --muted:   #888780;
      --border:  #1e1e1e;
      --border2: #2a2a2a;
      --bg2:     #141414;
      --r-lg:    8px;
      --r-md:    6px;
      --bear:    #E24B4A;
      --bull:    #639922;
      --neut:    #888780;
    }
    html, body { background: #0d0d0d; color: var(--txt); font-family: var(--sans); }
    .page { max-width: 900px; margin: 0 auto; padding: 28px 24px 48px; }

    /* ── top bar ─────────────────────────────────────────────────────── */
    .topbar { display: flex; justify-content: space-between; align-items: center;
              margin-bottom: 20px; }
    .topbar-title { font-size: 13px; font-weight: 500; color: var(--muted);
                    letter-spacing: .06em; text-transform: uppercase; }
    .topbar-right { display: flex; align-items: center; gap: 14px; }

    /* live price block */
    .price-wrap { display: flex; align-items: center; gap: 7px; }
    .live-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
                background: var(--muted); }
    .live-dot.live    { background: var(--bull);
                        animation: pulse-dot 2s ease-in-out infinite; }
    .live-dot.delayed { background: var(--muted); animation: none; }
    @keyframes pulse-dot {
      0%, 100% { opacity: 1;    transform: scale(1);    }
      50%       { opacity: 0.4; transform: scale(0.72); }
    }
    .price { font-size: 22px; font-weight: 500; }
    .price-change { font-size: 13px; font-weight: 500; }
    .price-change.pos { color: var(--bull); }
    .price-change.neg { color: var(--bear); }
    .price-change.neu { color: var(--muted); }
    .delay-tag { font-size: 11px; color: var(--muted); background: var(--bg2);
                 padding: 1px 6px; border-radius: 3px; letter-spacing: .03em; }
    .price-date { font-size: 13px; color: var(--muted); }

    .refresh-btn { display: flex; align-items: center; gap: 5px; font-size: 13px;
                   padding: 6px 12px; border: 0.5px solid var(--border2);
                   border-radius: var(--r-md); background: transparent;
                   color: var(--muted); cursor: pointer; font-family: var(--sans);
                   transition: background .15s, color .15s; }
    .refresh-btn:hover { background: var(--bg2); color: var(--txt); }
    .cd { font-size: 11px; opacity: .5; }

    /* ── signal card ─────────────────────────────────────────────────── */
    .signal-card { border: 0.5px solid var(--border); border-radius: var(--r-lg);
                   padding: 20px 24px; margin-bottom: 12px;
                   display: flex; align-items: center; justify-content: space-between; }
    .signal-card.bearish { border-left: 3px solid var(--bear); }
    .signal-card.bullish { border-left: 3px solid var(--bull); }
    .signal-card.neutral { border-left: 3px solid var(--neut); }
    .signal-label { font-size: 32px; font-weight: 500; }
    .signal-label.bearish { color: var(--bear); }
    .signal-label.bullish { color: var(--bull); }
    .signal-label.neutral { color: var(--neut); }
    .signal-sub { font-size: 13px; color: var(--muted); margin-top: 5px; }
    .pos-row { display: flex; align-items: center; gap: 8px; margin-top: 14px; }
    .pos-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .pos-dot.flat { background: var(--neut); }
    .pos-dot.half { background: #BA7517; }
    .pos-dot.full { background: var(--bull); }
    .pos-dot.over { background: #185FA5; }
    .pos-size { font-size: 14px; font-weight: 500; }
    .pos-hint { font-size: 13px; color: var(--muted); margin-left: 2px; }
    .conf-block { text-align: right; flex-shrink: 0; }
    .conf-pct { font-size: 28px; font-weight: 500; }
    .conf-lbl { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .conf-bar-bg   { width: 120px; height: 4px; background: var(--border);
                     border-radius: 2px; margin-top: 8px; margin-left: auto; }
    .conf-bar-fill { height: 4px; border-radius: 2px; }
    .conf-bar-fill.bearish { background: var(--bear); }
    .conf-bar-fill.bullish { background: var(--bull); }
    .conf-bar-fill.neutral { background: var(--neut); }

    /* ── metric cards ────────────────────────────────────────────────── */
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
               margin-bottom: 12px; }
    .metric { background: var(--bg2); border-radius: var(--r-md); padding: 14px 16px; }
    .metric-lbl { font-size: 12px; color: var(--muted); margin-bottom: 7px; }
    .metric-val { font-size: 18px; font-weight: 500; }
    .metric-val.pos  { color: var(--bull); }
    .metric-val.neg  { color: var(--bear); }
    .metric-val.neut { color: var(--neut); }

    /* ── shared card ─────────────────────────────────────────────────── */
    .card { border: 0.5px solid var(--border); border-radius: var(--r-lg);
            padding: 20px; margin-bottom: 12px; }
    .card-title { font-size: 12px; font-weight: 500; color: var(--muted);
                  margin-bottom: 16px; text-transform: uppercase; letter-spacing: .07em; }

    /* ── strategy rows ───────────────────────────────────────────────── */
    .strat-row { display: flex; align-items: center; justify-content: space-between;
                 padding: 9px 0; border-bottom: 0.5px solid var(--border); }
    .strat-row:last-child { border-bottom: none; }
    .strat-name   { font-size: 14px; }
    .strat-driver { font-size: 12px; color: var(--muted); margin-top: 3px; }
    .badge { font-size: 12px; font-weight: 500; padding: 3px 12px;
             border-radius: 20px; white-space: nowrap; }
    .badge.bearish { background: rgba(226,75,74,.13); color: var(--bear); }
    .badge.neutral { background: var(--bg2); color: var(--muted); }
    .badge.bullish { background: rgba(99,153,34,.13); color: var(--bull); }

    /* ── chart ───────────────────────────────────────────────────────── */
    .chart-wrap { position: relative; width: 100%; height: 140px; }

    /* ── history log ─────────────────────────────────────────────────── */
    .hist-row { display: flex; align-items: center; gap: 12px; padding: 8px 0;
                border-bottom: 0.5px solid var(--border); font-size: 13px; }
    .hist-row:last-child { border-bottom: none; }
    .hist-date { color: var(--muted); width: 68px; flex-shrink: 0; }
    .hist-bias { font-weight: 500; width: 68px; flex-shrink: 0; }
    .hist-bias.bearish { color: var(--bear); }
    .hist-bias.bullish { color: var(--bull); }
    .hist-bias.neutral { color: var(--neut); }
    .hist-pos  { color: var(--muted); width: 40px; flex-shrink: 0; }
    .hist-conf { flex: 1; }
    .hist-vol  { color: var(--muted); font-size: 12px; }

    /* ── footer ──────────────────────────────────────────────────────── */
    .footer { font-size: 12px; color: var(--muted); text-align: right; margin-top: 6px; }
  </style>
</head>
<body>
<div class="page">

  <!-- Top bar -->
  <div class="topbar">
    <div class="topbar-title">XAU/USD &middot; V4 Ensemble</div>
    <div class="topbar-right">

      <!-- Live price block (JS takes over after first poll) -->
      <div class="price-wrap">
        <div class="live-dot" id="liveDot"></div>
        <span class="price" id="livePrice">{{ close }}</span>
        <span class="price-change neu" id="priceChange"></span>
        <span class="delay-tag" id="delayTag" style="display:none">delayed</span>
        <span class="price-date">{{ date }}</span>
      </div>

      <button class="refresh-btn" onclick="location.reload()">
        <i class="ti ti-refresh"></i>&nbsp;Refresh
        <span class="cd" id="cd">60s</span>
      </button>
    </div>
  </div>

  <!-- Signal card -->
  <div class="signal-card {{ bias_dir }}">
    <div>
      <div class="signal-label {{ bias_dir }}">{{ bias }}</div>
      <div class="signal-sub">{{ signal_sub }}</div>
      <div class="pos-row">
        <div class="pos-dot {{ pos_dot }}"></div>
        <span class="pos-size">{{ position }}x position</span>
        <span class="pos-hint">{{ pos_hint }}</span>
      </div>
    </div>
    <div class="conf-block">
      <div class="conf-pct">{{ confidence }}%</div>
      <div class="conf-lbl">confidence</div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill {{ bias_dir }}" style="width:{{ confidence }}%"></div>
      </div>
    </div>
  </div>

  <!-- Metric cards -->
  <div class="metrics">
    <div class="metric">
      <div class="metric-lbl">Vol regime</div>
      <div class="metric-val neut">{{ vol_regime }}</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">200d SMA</div>
      <div class="metric-val">{{ sma200 }}</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">Circuit breaker</div>
      <div class="metric-val {{ 'neg' if cb_active else 'pos' }}">
        {{ 'Active' if cb_active else 'Inactive' }}
      </div>
    </div>
    <div class="metric">
      <div class="metric-lbl">Signal score</div>
      <div class="metric-val {{ 'neg' if score_neg else 'pos' }}">{{ score }}</div>
    </div>
  </div>

  <!-- Strategy breakdown -->
  <div class="card">
    <div class="card-title">Strategy breakdown</div>
    {% for s in strategies %}
    <div class="strat-row">
      <div>
        <div class="strat-name">{{ s.name }}</div>
        <div class="strat-driver">{{ s.driver }}</div>
      </div>
      <span class="badge {{ s.dir }}">{{ s.label }}</span>
    </div>
    {% endfor %}
  </div>

  <!-- Confidence chart -->
  <div class="card">
    <div class="card-title">Confidence history &mdash; last 14 days</div>
    <div class="chart-wrap">
      <canvas id="confChart"
              aria-label="Daily confidence percentage over the last 14 trading days">
      </canvas>
    </div>
  </div>

  <!-- History log -->
  <div class="card">
    <div class="card-title">Recent signals log</div>
    {% for row in history7 %}
    <div class="hist-row">
      <span class="hist-date">{{ row.date }}</span>
      <span class="hist-bias {{ row.dir }}">{{ row.bias }}</span>
      <span class="hist-pos">{{ "%.2f"|format(row.position) }}x</span>
      <span class="hist-conf">{{ row.conf }}%</span>
      <span class="hist-vol">{{ row.vol }}</span>
    </div>
    {% endfor %}
  </div>

  <div class="footer" id="footer">Last updated: {{ updated }}</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
// ── confidence chart ──────────────────────────────────────────────────────
const labels  = {{ chart_labels  | safe }};
const vals    = {{ chart_values  | safe }};
const isBear  = {{ chart_bearish | safe }};
const yMin    = {{ chart_ymin }};
const yMax    = {{ chart_ymax }};

new Chart(document.getElementById('confChart'), {
  type: 'line',
  data: {
    labels,
    datasets: [{
      data: vals,
      borderColor: '#555',
      backgroundColor: 'transparent',
      pointBackgroundColor: vals.map((_, i) => isBear[i] ? '#E24B4A' : '#639922'),
      pointBorderColor: 'transparent',
      pointRadius: 5,
      pointHoverRadius: 6,
      borderWidth: 1.5,
      tension: 0.35,
      segment: {
        borderColor: ctx => isBear[ctx.p1DataIndex] ? '#E24B4A' : '#639922'
      }
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1e1e1e',
        titleColor: '#888780',
        bodyColor: '#fff',
        callbacks: { label: c => c.parsed.y.toFixed(1) + '% confidence' }
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: '#888780', font: { size: 11 } }
      },
      y: {
        min: yMin, max: yMax,
        grid: { color: 'rgba(136,135,128,.1)' },
        ticks: { color: '#888780', font: { size: 11 }, callback: v => v + '%' }
      }
    }
  }
});

// ── live price ticker ─────────────────────────────────────────────────────
const priceEl  = document.getElementById('livePrice');
const changeEl = document.getElementById('priceChange');
const dotEl    = document.getElementById('liveDot');
const delayEl  = document.getElementById('delayTag');
let lastKnown  = null;

function applyPrice(data) {
  if (data.price !== null) {
    priceEl.textContent = '$' + data.price.toLocaleString('en-US',
        { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    const sign = data.change >= 0 ? '+' : '';
    changeEl.textContent =
        sign + data.change.toFixed(2) +
        ' (' + sign + data.change_pct.toFixed(2) + '%)';
    changeEl.className = 'price-change ' +
        (data.change > 0 ? 'pos' : data.change < 0 ? 'neg' : 'neu');

    lastKnown = data;
  }

  if (data.ok) {
    dotEl.className   = 'live-dot live';
    delayEl.style.display = 'none';
  } else {
    dotEl.className   = 'live-dot delayed';
    delayEl.style.display = 'inline';
  }
}

async function pollPrice() {
  try {
    const resp = await fetch('/api/price');
    if (!resp.ok) throw new Error('non-200');
    applyPrice(await resp.json());
  } catch (_) {
    // Network failure — mark delayed, keep last known values visible
    if (lastKnown) {
      applyPrice({ ...lastKnown, ok: false });
    } else {
      dotEl.className   = 'live-dot delayed';
      delayEl.style.display = 'inline';
    }
  }
}

pollPrice();                         // fire immediately on load
setInterval(pollPrice, 30_000);      // then every 30s

// ── page data countdown (full reload every 60s) ───────────────────────────
let cd = 60;
const cdEl = document.getElementById('cd');
setInterval(() => {
  if (--cd <= 0) { location.reload(); return; }
  if (cdEl) cdEl.textContent = cd + 's';
}, 1000);
</script>
</body>
</html>"""


# ── Flask routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        data = _get_signals()
    except Exception as e:
        return (f"<pre style='color:#E24B4A;background:#0d0d0d;padding:2rem'>"
                f"Error computing signals:\n{e}</pre>"), 500
    return render_template_string(TEMPLATE, **data)


@app.route("/api/price")
def api_price():
    return jsonify(_fetch_price())


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  Gold Ensemble V4 Dashboard")
    print("  http://localhost:5000")
    print("  GET /           full dashboard  (cached 15min)")
    print("  GET /api/price  live price JSON (cached 60s)")
    print("  Ctrl+C to stop")
    print("=" * 56)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
