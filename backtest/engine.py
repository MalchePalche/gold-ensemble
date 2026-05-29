"""
backtest/engine.py

Walk-forward backtester with strict next-bar-open execution.

Conventions
-----------
- Strategies emit signals on bar close at time t.
- The engine shifts those signals by 1 bar before applying returns —
  i.e. a signal generated at close-of-day t opens a position on the next
  bar's open and earns the return from open(t+1) → open(t+2).
- For simplicity & to stay aligned with the daily close-driven strategy
  outputs, returns are measured close-to-close on the post-shift series.
  This is conservative: it understates how quickly a stop-out happens but
  does not introduce lookahead.
- Transaction cost: 0.1% per round trip (applied when the position size
  CHANGES, scaled by |Δpos|).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ensemble.model import EnsembleModel, EnsembleSeries
from strategies.base import StrategyResult


# ----------------------------------------------------------------- result dc
@dataclass
class BacktestStats:
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0

    def as_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 5),
            "avg_loss": round(self.avg_loss, 5),
            "expectancy": round(self.expectancy, 5),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe": round(self.sharpe, 3),
            "profit_factor": round(self.profit_factor, 3),
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
        }


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series                  # equity curve (starts at 1.0)
    returns: pd.Series                 # per-bar net returns
    positions: pd.Series               # post-shift, executed positions
    trades: pd.DataFrame               # trade-by-trade log
    stats: BacktestStats = field(default_factory=BacktestStats)


# ----------------------------------------------------------------- stats
def compute_stats(returns: pd.Series, positions: pd.Series, name: str = "") -> BacktestStats:
    """Compute headline stats from a return series + position series."""
    stats = BacktestStats()
    if returns is None or returns.empty:
        return stats

    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()

    # Trades = positions where the sign FLIPS (close + reopen counts as 1 trade)
    pos = positions.fillna(0.0)
    trade_open_mask = (pos != 0) & (pos.shift(1, fill_value=0) != pos)
    n_trades = int(trade_open_mask.sum())
    stats.total_trades = n_trades

    # Per-trade returns: chunk r by constant-position runs
    runs = (pos != pos.shift(1, fill_value=0)).cumsum()
    trade_rets: List[float] = []
    for _, idx in pos.groupby(runs).groups.items():
        if len(idx) == 0:
            continue
        if pos.loc[idx[0]] == 0:
            continue
        chunk = r.loc[idx]
        trade_rets.append(float((1.0 + chunk).prod() - 1.0))

    if trade_rets:
        wins = [t for t in trade_rets if t > 0]
        losses = [t for t in trade_rets if t < 0]
        stats.win_rate = len(wins) / len(trade_rets)
        stats.avg_win = float(np.mean(wins)) if wins else 0.0
        stats.avg_loss = float(np.mean(losses)) if losses else 0.0
        stats.expectancy = float(np.mean(trade_rets))
        sum_wins = sum(wins)
        sum_losses = -sum(losses)
        stats.profit_factor = float(sum_wins / sum_losses) if sum_losses > 0 else float("inf") if sum_wins > 0 else 0.0

    # Drawdown
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    stats.max_drawdown = float(dd.min()) if not dd.empty else 0.0

    # Sharpe (daily → annualized assuming 252)
    if r.std(ddof=0) > 0:
        stats.sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(252))

    stats.total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(r) / 252.0, 1e-9)
    if equity.iloc[-1] > 0:
        stats.cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)

    return stats


# ----------------------------------------------------------------- engine
class BacktestEngine:
    """Run individual-strategy and ensemble backtests with walk-forward split."""

    def __init__(
        self,
        gold_df: pd.DataFrame,
        ensemble: EnsembleModel,
        transaction_cost: float = 0.001,
        train_period: Optional[tuple] = None,
        test_period: Optional[tuple] = None,
    ):
        self.gold_df = gold_df.copy()
        self.ensemble = ensemble
        self.tc = transaction_cost
        self.train_period = train_period
        self.test_period = test_period

        # Pre-compute close-to-close returns once
        self.bar_returns = self.gold_df["close"].pct_change().fillna(0.0)

    # ---- core simulator ----
    def _simulate(self, signal: pd.Series, name: str) -> BacktestResult:
        """
        Apply next-bar-open execution + transaction costs.

        Steps:
          1. Forward-fill position from the signal (carry until changed/zeroed)
          2. Shift by 1 bar to enforce "execute on next bar's open"
          3. PnL = position * bar_return; cost = tc * |Δposition|
        """
        # Position from signal — strategies already maintain position via their state
        position = signal.reindex(self.gold_df.index).fillna(0).astype(float)
        # Execution lag of 1 bar
        executed = position.shift(1).fillna(0.0)

        # Per-bar position change drives cost
        delta_pos = executed.diff().abs().fillna(executed.abs())
        cost = delta_pos * self.tc

        gross = executed * self.bar_returns
        net = gross - cost
        equity = (1.0 + net).cumprod()

        # Trade log — entries on each new non-zero position run
        runs = (executed != executed.shift(1, fill_value=0)).cumsum()
        trades_rows: List[dict] = []
        for _, idx_list in executed.groupby(runs).groups.items():
            idx_list = list(idx_list)
            if not idx_list:
                continue
            pos_val = executed.loc[idx_list[0]]
            if pos_val == 0:
                continue
            chunk_ret = net.loc[idx_list]
            ret = float((1.0 + chunk_ret).prod() - 1.0)
            trades_rows.append({
                "entry_date": idx_list[0],
                "exit_date": idx_list[-1],
                "direction": "LONG" if pos_val > 0 else "SHORT",
                "bars_held": len(idx_list),
                "return": ret,
            })
        trades = pd.DataFrame(trades_rows)

        stats = compute_stats(net, executed, name=name)
        return BacktestResult(
            name=name, equity=equity, returns=net, positions=executed,
            trades=trades, stats=stats,
        )

    # ---- public runs ----
    def run_strategy(self, key: str) -> BacktestResult:
        """Backtest a single strategy in isolation (raw +1/-1 vote, no weighting)."""
        result: StrategyResult = self.ensemble.strategies[key].generate(self.gold_df)
        return self._simulate(result.signal.astype(float), name=key)

    def run_ensemble(self) -> tuple[BacktestResult, EnsembleSeries]:
        series = self.ensemble.run(self.gold_df)
        # cast +1/0/-1 ensemble signal into a position series
        sig = series.signal.astype(float)
        bt = self._simulate(sig, name="ENSEMBLE")
        return bt, series

    def run_all(self) -> Dict[str, BacktestResult]:
        """Run all strategies + ensemble together."""
        out: Dict[str, BacktestResult] = {}
        for key in ["S1", "S2", "S3", "S4", "S5", "S7", "S8", "S9", "S10"]:
            try:
                out[key] = self.run_strategy(key)
            except Exception as e:
                print(f"[backtest] {key} failed: {e}")
        ens, _ = self.run_ensemble()
        out["ENSEMBLE"] = ens
        return out

    # ---- walk-forward ----
    def walk_forward(self) -> Dict[str, Dict[str, BacktestResult]]:
        """
        Split into train + test windows using configured dates and run each
        strategy + the ensemble on both. Returns {"train": {...}, "test": {...}}.
        """
        if not self.train_period or not self.test_period:
            # If unset, do a single full-history run under "test"
            return {"test": self.run_all()}

        tr_start, tr_end = self.train_period
        te_start, te_end = self.test_period
        full_gold = self.gold_df

        out: Dict[str, Dict[str, BacktestResult]] = {}
        for label, (start, end) in [("train", (tr_start, tr_end)),
                                    ("test", (te_start, te_end))]:
            mask = (full_gold.index >= pd.to_datetime(start)) & (full_gold.index <= pd.to_datetime(end))
            window = full_gold.loc[mask]
            if window.empty:
                continue
            sub_engine = BacktestEngine(
                gold_df=window, ensemble=self.ensemble,
                transaction_cost=self.tc,
            )
            out[label] = sub_engine.run_all()
        return out

    # ---- persistence ----
    @staticmethod
    def save_signals(series: EnsembleSeries, gold_df: pd.DataFrame,
                     path: str = "results/signals.csv") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = pd.DataFrame({
            "close": gold_df["close"],
            "score": series.score,
            "signal": series.signal,
            "confidence_pct": series.confidence_pct,
            "regime": series.regime,
        })
        for k, r in series.per_strategy.items():
            df[f"{k}_signal"] = r.signal
            df[f"{k}_conf"] = r.confidence.round(3)
        df.to_csv(path)

    @staticmethod
    def save_summary(results: Dict[str, BacktestResult],
                     path: str = "results/summary.csv") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rows = []
        for name, r in results.items():
            row = {"strategy": name}
            row.update(r.stats.as_dict())
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
