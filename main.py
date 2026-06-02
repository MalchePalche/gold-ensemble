"""
main.py — Gold Ensemble V4 entry point.

Usage:
    python main.py              # today's signal (one-shot)
    python main.py --backtest   # full walk-forward backtest (v4.py)
    python main.py --dashboard  # launch Streamlit dashboard
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data.loader import load_all
from ensemble.model import EnsembleModel
from ensemble.sizer import compute_vol_regime, target_size, simulate_v4

STRAT_NAMES = {
    "S1": "MA Crossover (EMA 20/50)",
    "S2": "Donchian Breakout (55d)",
    "S4": "52-Week Momentum",
    "S5": "MACD (12/26/9)",
}
BIAS_OF = {1: "BULLISH", 0: "NEUTRAL", -1: "BEARISH"}


def load_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = path or os.path.join(HERE, "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_model(cfg: Dict[str, Any]) -> EnsembleModel:
    weights   = cfg.get("ensemble", {}).get("weights")
    dead_zone = cfg.get("ensemble", {}).get("dead_zone", 0.05)
    return EnsembleModel(
        weights        = weights or None,
        dead_zone      = dead_zone,
        strategy_params= cfg.get("strategies", {}),
    )


def run_once(cfg: Dict[str, Any]) -> None:
    history = cfg.get("data", {}).get("history_years", 15)
    print(f"[main] Loading data ({history} years)…")
    data  = load_all(years=history, use_cache=True)
    gold  = data["gold"]
    close = gold["close"]
    bar_returns = close.pct_change().fillna(0.0)

    model  = build_model(cfg)
    series = model.run(gold)
    signal = series.signal
    conf   = series.confidence_pct

    vol_regime, atr_ratio, rv_pct = compute_vol_regime(
        close, gold["high"], gold["low"]
    )

    net_v4, pos_v4, cb_log = simulate_v4(
        signal, conf, vol_regime, bar_returns,
        tc            = cfg.get("backtest", {}).get("transaction_cost", 0.001),
        cb_threshold  = cfg.get("circuit_breaker", {}).get("return_threshold", -0.08),
        cb_freeze_bars= cfg.get("circuit_breaker", {}).get("freeze_bars", 10),
    )

    # Print snapshot
    i            = -1
    today_date   = gold.index[i].date()
    today_close  = float(close.iloc[i])
    today_sig    = int(signal.iloc[i])
    today_conf   = float(conf.iloc[i])
    today_vol    = vol_regime.iloc[i]
    today_pos    = float(pos_v4.iloc[i])
    today_matrix = target_size(today_sig, today_conf, today_vol)
    bias_str     = BIAS_OF[today_sig]

    print()
    print("=" * 62)
    print(f"  Gold Ensemble V4  --  {today_date}")
    print("=" * 62)
    print(f"  XAU/USD close:   ${today_close:,.2f}")
    print(f"  Bias:            {bias_str}")
    print(f"  Confidence:      {today_conf:.1f}%")
    print(f"  Vol regime:      {today_vol.upper()}")
    print(f"  Matrix target:   {today_matrix}x")
    print(f"  Position (V4):   {today_pos:.2f}x")
    print()
    for k, r in series.per_strategy.items():
        sv    = int(r.signal.iloc[i])
        cv    = float(r.confidence.iloc[i])
        arrow = "UP " if sv > 0 else "DN " if sv < 0 else "-- "
        print(f"    {k}  {arrow}  conf={cv:.3f}  [{STRAT_NAMES.get(k, k)}]")
    print("=" * 62)

    # Preview mode only — main.py must NOT write to Supabase. It recomputes the
    # signal on a different code path (and without the options/CB/forward-test
    # overlays), so persisting from here would clobber run_daily.py's richer row.
    print("\n[main] Preview mode — signal NOT saved to Supabase.")
    print("[main] Use run_daily.py for production signal generation.")


def launch_dashboard() -> None:
    app = os.path.join(HERE, "dashboard", "app.py")
    print(f"[main] Launching dashboard: streamlit run {app}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app])


def run_backtest() -> None:
    import importlib.util
    v4_path = os.path.join(HERE, "v4.py")
    spec = importlib.util.spec_from_file_location("v4", v4_path)
    v4   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    v4.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Ensemble V4 — XAU/USD daily bias")
    parser.add_argument("--backtest",  action="store_true", help="Run V4 backtest")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--config",    type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dashboard:
        launch_dashboard()
    elif args.backtest:
        run_backtest()
    else:
        run_once(cfg)


if __name__ == "__main__":
    main()
