"""
v4.py — Volatility-Aware Position Sizing

Layer 1  Vol regime    ATR ratio (14/20) + realized-vol percentile (20d/252d)
Layer 2  Sizing matrix 6-tier based on signal + confidence + vol regime
Layer 3  Circuit breaker  20-day rolling portfolio return < -8% → 0x for 10 bars
Layer 4  Entry smoothing  scale-in 3 days, scale-out 2 days
"""

from __future__ import annotations
import os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data.loader import load_all
from ensemble.model import EnsembleModel
from ensemble.sizer import compute_vol_regime, target_size, simulate_v4
from backtest.engine import compute_stats

# ── V4 weights (V3 grid-search optimised, locked for production)
V3_WEIGHTS = {
    "S1": 1.5, "S2": 0.5, "S4": 1.5, "S5": 2.0,
}
TC            = 0.001
DRAWDOWN_WIN  = ("2022-03-01", "2022-11-30")
TEST          = ("2020-01-01", "2025-12-31")


# ════════════════════════════════════════════ helpers
def _section(title: str) -> None:
    print(f"\n{'='*66}")
    print(f"  {title}")
    print(f"{'='*66}")


def _comparison_row(label, stats, extra=None):
    row = {
        "version"      : label,
        "sharpe"       : f"{stats.sharpe:+.3f}",
        "win_rate"     : f"{stats.win_rate:.1%}",
        "max_dd"       : f"{stats.max_drawdown:.1%}",
        "total_return" : f"{stats.total_return:+.1%}",
        "trades"       : stats.total_trades,
    }
    if extra:
        row.update(extra)
    return row


# ════════════════════════════════════════════ main
def main() -> None:
    print("[v4] Loading data…")
    data  = load_all(years=15, use_cache=True)
    gold  = data["gold"]
    close = gold["close"]
    bar_returns = close.pct_change().fillna(0.0)

    model = EnsembleModel(weights=V3_WEIGHTS)
    print("[v4] Running ensemble signals…")
    series = model.run(gold)
    signal = series.signal
    conf   = series.confidence_pct

    print("[v4] Computing volatility regimes…")
    vol_regime, atr_ratio, rv_pct = compute_vol_regime(
        close, gold["high"], gold["low"]
    )

    test_mask = (gold.index >= TEST[0]) & (gold.index <= TEST[1])
    dd_mask   = (gold.index >= DRAWDOWN_WIN[0]) & (gold.index <= DRAWDOWN_WIN[1])

    # ──────────────────────────────────── run V4 on full history
    print("[v4] Simulating V4 (all 4 layers)…")
    net_v4, pos_v4, cb_log = simulate_v4(signal, conf, vol_regime, bar_returns)
    st_v4_test = compute_stats(net_v4[test_mask], pos_v4[test_mask])

    # ──────────────────────────────────── V3 overlay baseline (for comparison)
    # reproduce the V3 analysis.py sizing exactly
    def v3_sizing(sig, cp):
        pos = pd.Series(0.5, index=sig.index)
        pos[sig ==  1] = 1.0
        pos[(sig == 1) & (cp >= 40)] = 1.0
        pos[(sig == 1) & (cp >  60)] = 1.5
        pos[sig == -1] = 0.0
        return pos

    pos_v3     = v3_sizing(signal, conf)
    exe_v3     = pos_v3.shift(1).fillna(0.0)
    net_v3     = exe_v3 * bar_returns - exe_v3.diff().abs().fillna(exe_v3.abs()) * TC
    st_v3_test = compute_stats(net_v3[test_mask], exe_v3[test_mask])

    # ──────────────────────────────────── Buy & Hold
    bh_ret  = bar_returns[test_mask]
    bh_eq   = (1 + bh_ret).cumprod()
    bh_sharpe = bh_ret.mean() / bh_ret.std(ddof=0) * np.sqrt(252)
    bh_dd   = ((bh_eq / bh_eq.cummax()) - 1).min()
    bh_return = bh_eq.iloc[-1] - 1

    # ──────────────────────────────────── SECTION: Vol regime distribution
    _section("Vol Regime Distribution — TEST period")
    vr_test = vol_regime[test_mask]
    n_test  = test_mask.sum()
    for r in ["normal", "elevated", "extreme"]:
        n_r = (vr_test == r).sum()
        print(f"  {r:<10}  {n_r:4d} days  ({n_r / n_test:.1%})")
    print(f"\n  ATR ratio  — mean {atr_ratio[test_mask].mean():.2f}  "
          f"max {atr_ratio[test_mask].max():.2f}")
    print(f"  RV pct     — mean {rv_pct[test_mask].mean():.2f}  "
          f"max {rv_pct[test_mask].max():.2f}")

    # ──────────────────────────────────── SECTION: Circuit breaker
    _section("Circuit Breaker Log — TEST period")
    cb_test = [e for e in cb_log
               if pd.to_datetime(e["date"]) >= pd.to_datetime(TEST[0])
               and pd.to_datetime(e["date"]) <= pd.to_datetime(TEST[1])]
    print(f"  Total triggers on TEST: {len(cb_test)}")
    for e in cb_test:
        print(f"    {e['date'].date() if hasattr(e['date'],'date') else str(e['date'])[:10]}"
              f"  20d-return={e['20d_return']:+.1f}%  → froze 10 bars")

    # ──────────────────────────────────── SECTION: Position size distribution
    _section("Position Size Distribution — TEST period")
    pos_test = pos_v4[test_mask]
    for sz, label in [
        (1.5,  "1.5x  BULLISH high-conf + normal vol"),
        (1.0,  "1.0x  BULLISH base      + normal vol  (or high-conf elevated)"),
        (0.75, "0.75x BULLISH base      + elevated vol"),
        (0.5,  "0.5x  NEUTRAL or high-conf extreme vol"),
        (0.25, "0.25x BULLISH base      + extreme vol"),
        (0.0,  "0.0x  BEARISH or CB freeze"),
    ]:
        n_sz = (pos_test.round(3) == sz).sum()
        if n_sz > 0 or sz in [1.5, 1.0, 0.5, 0.0]:
            print(f"  {label:<50}  {n_sz:4d} days  ({n_sz / n_test:.1%})")

    # ──────────────────────────────────── SECTION: Main comparison table
    _section("TEST Period Comparison (2020–2025)")
    rows = [
        _comparison_row("V4 (vol-aware + CB + smooth)", st_v4_test,
                        {"cb_triggers": len(cb_test)}),
        _comparison_row("V3 Overlay (prev baseline)",  st_v3_test,
                        {"cb_triggers": "—"}),
        {
            "version": "Buy & Hold Gold", "sharpe": f"{bh_sharpe:+.3f}",
            "win_rate": "—", "max_dd": f"{bh_dd:.1%}",
            "total_return": f"{bh_return:+.1%}", "trades": 1, "cb_triggers": "—",
        },
    ]
    print(pd.DataFrame(rows).set_index("version").to_string())

    # ──────────────────────────────────── SECTION: 2022 drawdown window
    _section("2022 Drawdown Window — Mar–Nov 2022")

    dd_price = close[dd_mask]
    peak_p   = dd_price.max()
    trough_p = dd_price.min()
    print(f"\n  Gold: peak ${peak_p:,.0f} → trough ${trough_p:,.0f}  "
          f"({(trough_p-peak_p)/peak_p:.1%})")

    # Monthly detail
    dd_df = pd.DataFrame({
        "price"   : dd_price,
        "pos_v4"  : pos_v4[dd_mask],
        "pos_v3"  : exe_v3[dd_mask],
        "net_v4"  : net_v4[dd_mask],
        "bh_ret"  : bar_returns[dd_mask],
    }).assign(month=lambda d: d.index.to_period("M"))

    print(f"\n  {'Month':<8}  {'Gold':>7}  {'V4 pos':>9}  {'V4 ret':>7}  {'BH ret':>7}")
    print(f"  {'─'*50}")
    cum_v4 = 1.0
    cum_bh = 1.0
    for month, grp in dd_df.groupby("month"):
        m_gold = (1 + grp["bh_ret"]).prod() - 1
        m_v4   = (1 + grp["net_v4"]).prod() - 1
        avg_pos = grp["pos_v4"].mean()
        cum_v4 *= (1 + m_v4)
        cum_bh *= (1 + m_gold)
        print(f"  {str(month):<8}  {m_gold:>+7.1%}  {avg_pos:>9.2f}x  "
              f"{m_v4:>+7.1%}  {m_gold:>+7.1%}")
    print(f"  {'─'*50}")
    print(f"  {'Total':<8}  {cum_bh-1:>+7.1%}  {'':>9}  {cum_v4-1:>+7.1%}  {cum_bh-1:>+7.1%}")

    cb_dd = [e for e in cb_log
             if str(e["date"])[:7] >= DRAWDOWN_WIN[0][:7]
             and str(e["date"])[:7] <= DRAWDOWN_WIN[1][:7]]
    if cb_dd:
        print(f"\n  Circuit breaker triggers in 2022 window: {len(cb_dd)}")
        for e in cb_dd:
            print(f"    {str(e['date'])[:10]}  20d={e['20d_return']:+.1f}%")
    else:
        print(f"\n  No circuit breaker triggers in 2022 window.")

    # Max DD for V4 in this specific window
    eq_v4_dd  = (1 + net_v4[dd_mask]).cumprod()
    peak_eq   = eq_v4_dd.cummax()
    v4_dd_win = ((eq_v4_dd / peak_eq) - 1).min()
    print(f"\n  V4 max DD in window:        {v4_dd_win:.1%}  (target: below -5%)")

    # ──────────────────────────────────── SECTION: Today's signal
    _section("Today's V4 Signal (2026-05-26)")
    latest_i     = -1
    latest_date  = gold.index[latest_i].date()
    latest_close = float(close.iloc[latest_i])
    latest_sig   = int(signal.iloc[latest_i])
    latest_conf  = float(conf.iloc[latest_i])
    latest_vol   = vol_regime.iloc[latest_i]
    latest_atr   = float(atr_ratio.iloc[latest_i])
    latest_rv    = float(rv_pct.iloc[latest_i])
    today_target = target_size(latest_sig, latest_conf, latest_vol)

    # Check if CB is currently active (look at last 10 bars of actual portfolio)
    # Reuse the last n_cb_remaining from the simulation
    last_20_net   = net_v4.iloc[-20:]
    roll_ret_now  = float(np.prod(1 + last_20_net) - 1)
    cb_now_active = roll_ret_now < -0.08

    bias_str = {1: "BULLISH", 0: "NEUTRAL", -1: "BEARISH"}[latest_sig]
    size_str  = {1.5: "1.5x", 1.0: "1.0x", 0.75: "0.75x",
                 0.5: "0.5x", 0.25: "0.25x", 0.0: "0.0x"}
    display_size = "0.0x (CB freeze)" if cb_now_active else size_str.get(today_target, f"{today_target}x")

    print(f"\n  Date:              {latest_date}")
    print(f"  XAU/USD close:     ${latest_close:,.2f}")
    print()
    print(f"  Ensemble bias:     {bias_str}")
    print(f"  Confidence:        {latest_conf:.1f}%")
    print()
    print(f"  Vol regime:        {latest_vol.upper()}")
    print(f"    ATR ratio:       {latest_atr:.2f}x  "
          f"({'ELEVATED' if latest_atr > 1.5 else 'normal'})")
    print(f"    RV percentile:   {latest_rv:.0%}  "
          f"({'HIGH' if latest_rv > 0.75 else 'normal'})")
    print()
    print(f"  Circuit breaker:   {'ACTIVE (20d return = ' + f'{roll_ret_now:.1%})' if cb_now_active else 'inactive  (20d portfolio return = ' + f'{roll_ret_now:.1%})'}")
    print()
    print(f"  Suggested size:    {display_size}")
    print(f"  (matrix says {today_target}x before smoothing/CB)")


if __name__ == "__main__":
    main()
