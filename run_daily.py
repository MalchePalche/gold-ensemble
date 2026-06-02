"""
run_daily.py — V4 Gold Ensemble daily runner.

Run once after market close each day:
    python run_daily.py

What it does:
  1. Fetches latest gold data (cached, 12h TTL)
  2. Computes ensemble signal + confidence
  3. Detects volatility regime (ATR + realised-vol percentile)
  4. Runs full V4 simulation to derive today's position with CB + smoothing
  5. Compares with yesterday's signal from Supabase
  6. Prints a one-page console summary
  7. Sends Telegram alert if bias flipped OR position changed >= 0.5x
  8. Upserts today's row into the Supabase `signals` table
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import yaml

from data.loader import DataLoader
from ensemble.model import EnsembleModel
from ensemble.sizer import compute_vol_regime, target_size, simulate_v4
from db.queries import save_signal, get_recent_signals

STRAT_NAMES = {
    "S1": "MA Crossover (EMA 20/50)",
    "S2": "Donchian Breakout (55d)",
    "S4": "52-Week Momentum",
    "S5": "MACD (12/26/9)",
}
BIAS_OF = {1: "BULLISH", 0: "NEUTRAL", -1: "BEARISH"}


# ── helpers ────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(os.path.join(HERE, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _load_previous(today_date: str) -> dict | None:
    """Most recent stored signal from a day other than today (Supabase)."""
    try:
        for row in get_recent_signals(2):
            if str(row.get("date")) != today_date:
                return row
        return None
    except Exception as e:
        print(f"[run_daily] Could not load previous signal: {e}")
        return None


def _send_telegram(bot_token: str, chat_id: str, message: str) -> None:
    import requests
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        print("[telegram] Alert sent.")
    except Exception as e:
        print(f"[telegram] Failed: {e}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    cfg      = _load_config()
    data_cfg = cfg.get("data", {})
    ens_cfg  = cfg.get("ensemble", {})
    vol_cfg  = cfg.get("volatility", {})
    siz_cfg  = cfg.get("sizing", {})
    cb_cfg   = cfg.get("circuit_breaker", {})
    tg_cfg   = cfg.get("telegram", {})

    # ── 1. Gold data ───────────────────────────────────────────────────────
    print("[run_daily] Fetching gold data…")
    loader = DataLoader(
        gold_ticker     = data_cfg.get("gold_ticker", "GC=F"),
        cache_ttl_hours = data_cfg.get("cache_ttl_hours", 12),
        use_cache       = True,
    )
    gold        = loader.load_gold(years=data_cfg.get("history_years", 15))
    close       = gold["close"]
    bar_returns = close.pct_change().fillna(0.0)

    # ── 2. Ensemble signals ────────────────────────────────────────────────
    print("[run_daily] Computing ensemble signals…")
    model = EnsembleModel(
        weights   = ens_cfg.get("weights", {"S1": 1.5, "S2": 0.5, "S4": 1.5, "S5": 2.0}),
        dead_zone = ens_cfg.get("dead_zone", 0.05),
    )
    series = model.run(gold)
    signal = series.signal
    conf   = series.confidence_pct

    # ── 3. Vol regime ──────────────────────────────────────────────────────
    print("[run_daily] Computing volatility regime…")
    vol_regime, atr_ratio, rv_pct = compute_vol_regime(
        close, gold["high"], gold["low"],
        atr_period        = vol_cfg.get("atr_period",        14),
        atr_avg_period    = vol_cfg.get("atr_avg_period",    20),
        atr_elevated_mult = vol_cfg.get("atr_elevated_mult", 1.5),
        atr_extreme_mult  = vol_cfg.get("atr_extreme_mult",  2.0),
        rv_period         = vol_cfg.get("rv_period",         20),
        rv_lookback       = vol_cfg.get("rv_lookback",       252),
        rv_high_pct       = vol_cfg.get("rv_high_pct",       0.75),
        rv_extreme_pct    = vol_cfg.get("rv_extreme_pct",    0.90),
    )

    # ── 4. V4 simulation (determines CB state + smoothed position) ─────────
    print("[run_daily] Running V4 simulation…")
    sizing_kwargs = {
        "bullish_high_normal"   : siz_cfg.get("bullish_high_normal",   1.5),
        "bullish_high_elevated" : siz_cfg.get("bullish_high_elevated", 1.0),
        "bullish_high_extreme"  : siz_cfg.get("bullish_high_extreme",  0.5),
        "bullish_base_normal"   : siz_cfg.get("bullish_base_normal",   1.0),
        "bullish_base_elevated" : siz_cfg.get("bullish_base_elevated", 0.75),
        "bullish_base_extreme"  : siz_cfg.get("bullish_base_extreme",  0.25),
        "neutral"               : siz_cfg.get("neutral",               0.5),
        "bearish"               : siz_cfg.get("bearish",               0.0),
        "conf_high_threshold"   : siz_cfg.get("conf_high_threshold",   60.0),
    }
    net_v4, pos_v4, cb_log = simulate_v4(
        signal, conf, vol_regime, bar_returns,
        tc            = cfg.get("backtest", {}).get("transaction_cost", 0.001),
        cb_threshold  = cb_cfg.get("return_threshold", -0.08),
        cb_freeze_bars= cb_cfg.get("freeze_bars", 10),
        sizing_kwargs = sizing_kwargs,
    )

    # ── 5. Today's values ──────────────────────────────────────────────────
    today_date   = gold.index[-1].date()
    today_close  = float(close.iloc[-1])
    today_sig    = int(signal.iloc[-1])
    today_conf   = float(conf.iloc[-1])
    today_vol    = vol_regime.iloc[-1]
    today_atr    = float(atr_ratio.iloc[-1])
    today_rv     = float(rv_pct.iloc[-1])
    today_pos    = float(pos_v4.iloc[-1])
    today_matrix = target_size(today_sig, today_conf, today_vol, **sizing_kwargs)
    bias_str     = BIAS_OF[today_sig]

    sma_200_series = close.rolling(200).mean()
    sma_200        = float(sma_200_series.iloc[-1]) if sma_200_series.notna().iloc[-1] else None

    # Per-strategy latest signal + driver (for the signals table / dashboard)
    per_strat = {}
    for k, r in series.per_strategy.items():
        sv = int(r.signal.iloc[-1])
        per_strat[k] = {"bias": BIAS_OF[sv], "driver": STRAT_NAMES.get(k, k)}

    # CB is active when the matrix target is > 0 but the simulation forced 0
    cb_active = (today_matrix > 0.0) and (today_pos == 0.0)

    # ── 6. Yesterday comparison ────────────────────────────────────────────
    yesterday  = _load_previous(str(today_date))
    prev_bias  = yesterday.get("bias") if yesterday else None
    prev_pos   = float(yesterday.get("position_size", today_pos)) if yesterday else today_pos
    pos_change = today_pos - prev_pos
    bias_flip  = (prev_bias is not None) and (prev_bias != bias_str)

    # ── 7. Console summary ─────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  Gold Ensemble V4  --  {today_date}")
    print("=" * 62)
    print(f"  {'XAU/USD close':<22} ${today_close:,.2f}")
    print(f"  {'Bias':<22} {bias_str}")
    print(f"  {'Confidence':<22} {today_conf:.1f}%")
    print(f"  {'Vol regime':<22} {today_vol.upper()}"
          f"  (ATR {today_atr:.2f}x  RV {today_rv:.0%})")
    print(f"  {'Matrix target':<22} {today_matrix}x")
    print(f"  {'Position (V4)':<22} {today_pos:.2f}x"
          f"{'  [CB ACTIVE]' if cb_active else ''}")
    print(f"  {'Prev position':<22} {prev_pos:.2f}x")

    change_flag = ""
    if abs(pos_change) >= tg_cfg.get("size_change_threshold", 0.5):
        change_flag = "  << ALERT"
    print(f"  {'Change':<22} {pos_change:+.2f}x{change_flag}")

    if bias_flip:
        print(f"  {'Bias flip':<22} {prev_bias} -> {bias_str}  << ALERT")

    print()
    print(f"  {'Strategy':<8}  {'Dir':<4}  {'Conf':>6}  Description")
    print(f"  {'-'*50}")
    for k, r in series.per_strategy.items():
        sv    = int(r.signal.iloc[-1])
        cv    = float(r.confidence.iloc[-1])
        arrow = "UP  " if sv > 0 else "DN  " if sv < 0 else "--  "
        print(f"  {k:<8}  {arrow}  {cv:>6.3f}  {STRAT_NAMES.get(k, k)}")
    print("=" * 62)

    # ── Economic calendar (high-impact USD events) ──────────────────────────
    try:
        from data.calendar import get_todays_events, event_risk_score
        today_events = get_todays_events()
        risk = event_risk_score(today_events)
        print(f"\n  Economic risk today: {risk}")
        if today_events:
            for e in today_events:
                print(f"  {e['time_sofia']} Sofia — {e['title']}")
    except Exception as e:
        print(f"\n  Economic calendar unavailable: {e}")

    # ── Correlation monitor (gold vs key assets) ────────────────────────────
    corr_summary = "ALIGNED"
    corr_breaks  = []
    try:
        from data.correlations import fetch_correlations, correlation_summary
        corr_data    = fetch_correlations()
        corr_summary = correlation_summary(corr_data)
        corr_breaks  = [v for v in corr_data.values() if v["breakdown"]]
        print(f"\n  Correlation health: {corr_summary}")
        for v in corr_breaks:
            print(f"  {v['label']}: {v['breakdown_msg']} "
                  f"(30d {v['corr_30d']:+.2f}, 5d {v['corr_5d']:+.2f})")
    except Exception as e:
        print(f"\n  Correlation monitor unavailable: {e}")

    # ── News sentiment (last 24h) ───────────────────────────────────────────
    sentiment   = None
    sent_diverge = False
    try:
        from data.sentiment import get_sentiment, divergence_check
        sentiment = get_sentiment()
        div = divergence_check(sentiment["signal"], bias_str)
        sent_diverge = div["divergence"]
        print(f"\n  Sentiment: {sentiment['signal']} ({sentiment['confidence']}%) — "
              f"{sentiment['bullish_count']} bullish, "
              f"{sentiment['bearish_count']} bearish, "
              f"{sentiment['neutral_count']} neutral headlines")
        if sent_diverge:
            print(f"  DIVERGENCE: News {sentiment['signal']} vs Ensemble {bias_str} "
                  f"— watch for reversal in 1-3 days")
    except Exception as e:
        print(f"\n  News sentiment unavailable: {e}")

    # ── 8. Telegram alert ──────────────────────────────────────────────────
    size_thresh  = tg_cfg.get("size_change_threshold", 0.5)
    corr_alert   = corr_summary == "BREAKDOWN"
    sent_alert   = sent_diverge
    should_alert = (
        tg_cfg.get("enabled", False) and
        (bias_flip or abs(pos_change) >= size_thresh or corr_alert or sent_alert)
    )

    if should_alert:
        # Secrets come from the environment first, config.yaml as fallback.
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or tg_cfg.get("bot_token", "")
        chat_id   = os.getenv("TELEGRAM_CHAT_ID")   or tg_cfg.get("chat_id", "")
        if bot_token and "YOUR" not in bot_token and chat_id and "YOUR" not in chat_id:
            lines = [
                f"<b>Gold Ensemble V4  --  {today_date}</b>",
                f"XAU/USD: <b>${today_close:,.2f}</b>",
                "",
            ]
            if bias_flip:
                lines.append(f"BIAS FLIP: {prev_bias} -> <b>{bias_str}</b>")
            lines += [
                f"Bias:  <b>{bias_str}</b>  ({today_conf:.1f}% conf)",
                f"Vol:   {today_vol.upper()}  (ATR {today_atr:.2f}x  RV {today_rv:.0%})",
                f"Pos:   <b>{today_pos:.2f}x</b>  (was {prev_pos:.2f}x, "
                f"change {pos_change:+.2f}x)",
            ]
            if cb_active:
                lines.append("Circuit breaker ACTIVE")
            if corr_alert:
                lines.append("")
                lines.append("⚠️ Correlation breakdown detected — regime change possible")
                for v in corr_breaks:
                    lines.append(f"  {v['label']}: {v['breakdown_msg']}")
            if sent_alert and sentiment is not None:
                lines.append("")
                lines.append(f"⚡ SENTIMENT DIVERGENCE: News {sentiment['signal']} "
                             f"vs Ensemble {bias_str}")
                lines.append("Price often follows sentiment within 1-3 days")
            _send_telegram(bot_token, chat_id, "\n".join(lines))
        else:
            print("[telegram] Alert triggered but bot_token/chat_id not configured.")
    elif tg_cfg.get("enabled", False):
        print("[telegram] No significant change — alert suppressed.")

    # ── 9. Upsert today's row into Supabase ─────────────────────────────────
    row = {
        "date"                   : str(today_date),
        "price"                  : round(today_close, 2),
        "bias"                   : bias_str,
        "confidence"             : round(today_conf, 2),
        "signal_score"           : round(float(series.score.iloc[-1]), 4),
        "position_size"          : round(today_pos, 2),
        "vol_regime"             : today_vol,
        "sma_200"                : round(sma_200, 2) if sma_200 is not None else None,
        "circuit_breaker_active" : bool(cb_active),
        "s1_signal"              : per_strat["S1"]["bias"], "s1_driver": per_strat["S1"]["driver"],
        "s2_signal"              : per_strat["S2"]["bias"], "s2_driver": per_strat["S2"]["driver"],
        "s4_signal"              : per_strat["S4"]["bias"], "s4_driver": per_strat["S4"]["driver"],
        "s5_signal"              : per_strat["S5"]["bias"], "s5_driver": per_strat["S5"]["driver"],
    }
    try:
        save_signal(row)
        print(f"\n[run_daily] Upserted signal for {today_date} to Supabase.")
    except Exception as e:
        print(f"\n[run_daily] Failed to save signal to Supabase: {e}")
        raise


if __name__ == "__main__":
    main()
