"""
data/forward_test.py — forward-test analytics for the V4 ensemble.

Tracks system accuracy on live signals going forward: pulls the stored
`signals` rows from Supabase and compares each day's bias against the actual
next-day XAU/USD return. Surfaces win rate, expectancy, streaks, per-bias
accuracy and confidence calibration to the dashboard.

Unlike the backtest layer this evaluates only signals the system has actually
emitted live, so it is an honest out-of-sample track record.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_gold_returns(days_back: int = 90) -> pd.Series:
    """
    Fetch daily XAU/USD returns for the last `days_back` days.
    Used to evaluate past signal accuracy.

    Returns a Series of fractional daily returns indexed by `datetime.date`.
    """
    end = datetime.today()
    start = end - timedelta(days=days_back + 5)
    df = yf.download("GC=F", start=start, end=end,
                     progress=False)["Close"].squeeze()
    returns = df.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index).date
    return returns


def evaluate_signals(signals: list[dict],
                     returns: pd.Series) -> pd.DataFrame:
    """
    For each signal in Supabase, check if the bias matched the actual
    next-day return direction.

    Signal is CORRECT if:
    - BULLISH and next day return > 0
    - BEARISH and next day return < 0
    - NEUTRAL counts as abstain (excluded from win rate)

    Returns DataFrame with evaluation per signal.
    """
    rows = []
    for sig in signals:
        sig_date = pd.to_datetime(sig["date"]).date()
        bias = sig.get("bias", "NEUTRAL")
        confidence = sig.get("confidence_pct") or sig.get("confidence", 0)

        if bias == "NEUTRAL":
            continue

        # Find next trading day return (look up to 5 days for weekends/holidays).
        next_return = None
        check_date = sig_date + timedelta(days=1)
        for _ in range(5):
            if check_date in returns.index:
                next_return = float(returns[check_date])
                break
            check_date += timedelta(days=1)

        if next_return is None:
            # Future signal — not yet evaluable.
            rows.append({
                "date": sig_date,
                "bias": bias,
                "confidence": confidence,
                "next_return": None,
                "correct": None,
                "status": "pending",
            })
            continue

        # Evaluate.
        if bias == "BULLISH":
            correct = next_return > 0
        else:  # BEARISH
            correct = next_return < 0

        rows.append({
            "date": sig_date,
            "bias": bias,
            "confidence": confidence,
            "next_return": round(next_return * 100, 3),
            "correct": correct,
            "status": "evaluated",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_stats(eval_df: pd.DataFrame) -> dict:
    """
    Compute forward-test statistics from evaluated signals.

    Metrics:
    - total_signals: total non-neutral signals
    - evaluated: signals with known outcome
    - pending: signals awaiting next day
    - win_rate: correct / evaluated
    - win_rate_high_conf: win rate when confidence > 65%
    - win_rate_low_conf: win rate when confidence <= 65%
    - avg_return_correct: avg next-day return on correct calls
    - avg_return_incorrect: avg next-day return on incorrect calls
    - expectancy: (win_rate * avg_win) - (loss_rate * avg_loss)
    - streak_current: current win/loss streak
    - best_streak: longest win streak
    - by_bias: win rate split by BULLISH vs BEARISH
    - confidence_calibration: does higher confidence = higher accuracy?
    """
    if eval_df.empty:
        return {"error": "No evaluated signals yet"}

    evaluated = eval_df[eval_df["status"] == "evaluated"].copy()
    pending = eval_df[eval_df["status"] == "pending"]

    if evaluated.empty:
        return {
            "error": None,
            "total_signals": len(eval_df),
            "evaluated": 0,
            "pending": len(pending),
            "win_rate": None,
            "message": "Signals logged but no outcomes yet — check back tomorrow",
        }

    total_eval = len(evaluated)
    correct = int(evaluated["correct"].sum())
    win_rate = correct / total_eval if total_eval > 0 else 0

    # High vs low confidence.
    high_conf = evaluated[evaluated["confidence"] > 65]
    low_conf = evaluated[evaluated["confidence"] <= 65]

    wr_high = (high_conf["correct"].sum() / len(high_conf)
               if len(high_conf) > 0 else None)
    wr_low = (low_conf["correct"].sum() / len(low_conf)
              if len(low_conf) > 0 else None)

    # Returns analysis.
    correct_returns = evaluated[evaluated["correct"] == True]["next_return"]
    incorrect_returns = evaluated[evaluated["correct"] == False]["next_return"]

    avg_win = float(correct_returns.mean()) if len(correct_returns) > 0 else 0
    avg_loss = float(incorrect_returns.mean()) if len(incorrect_returns) > 0 else 0

    loss_rate = 1 - win_rate
    expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

    # Current streak (walk back from the most recent evaluated signal).
    streak = 0
    streak_type = None
    for _, row in evaluated.sort_values("date", ascending=False).iterrows():
        if streak == 0:
            streak_type = "WIN" if row["correct"] else "LOSS"
            streak = 1
        elif (row["correct"] and streak_type == "WIN") or \
             (not row["correct"] and streak_type == "LOSS"):
            streak += 1
        else:
            break

    # Best win streak.
    best_streak = 0
    current = 0
    for _, row in evaluated.sort_values("date").iterrows():
        if row["correct"]:
            current += 1
            best_streak = max(best_streak, current)
        else:
            current = 0

    # By bias.
    bull_eval = evaluated[evaluated["bias"] == "BULLISH"]
    bear_eval = evaluated[evaluated["bias"] == "BEARISH"]
    wr_bull = (bull_eval["correct"].sum() / len(bull_eval)
               if len(bull_eval) > 0 else None)
    wr_bear = (bear_eval["correct"].sum() / len(bear_eval)
               if len(bear_eval) > 0 else None)

    # Confidence calibration buckets.
    buckets = []
    for low, high in [(40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]:
        bucket = evaluated[
            (evaluated["confidence"] >= low) &
            (evaluated["confidence"] < high)
        ]
        if len(bucket) > 0:
            buckets.append({
                "range": f"{low}-{high}%",
                "count": len(bucket),
                "win_rate": round(
                    bucket["correct"].sum() / len(bucket) * 100, 1),
            })

    return {
        "error": None,
        "total_signals": len(eval_df),
        "evaluated": total_eval,
        "pending": len(pending),
        "win_rate": round(win_rate * 100, 1),
        "win_rate_high_conf": round(wr_high * 100, 1) if wr_high is not None else None,
        "win_rate_low_conf": round(wr_low * 100, 1) if wr_low is not None else None,
        "avg_return_correct": round(avg_win, 3),
        "avg_return_incorrect": round(avg_loss, 3),
        "expectancy": round(expectancy, 3),
        "streak_current": streak,
        "streak_type": streak_type,
        "best_streak": best_streak,
        "wr_bullish": round(wr_bull * 100, 1) if wr_bull is not None else None,
        "wr_bearish": round(wr_bear * 100, 1) if wr_bear is not None else None,
        "confidence_calibration": buckets,
        "evaluated_signals": evaluated.to_dict("records"),
        "pending_signals": len(pending),
    }


def get_forward_test_analysis() -> dict:
    """Main entry point — pull signals, evaluate, and summarize."""
    try:
        from db.queries import get_recent_signals
        signals = get_recent_signals(90)
        if not signals:
            return {
                "stats": {"error": "No signals in database yet"},
                "eval_df": pd.DataFrame(),
                "error": None,
            }
        returns = fetch_gold_returns(90)
        eval_df = evaluate_signals(signals, returns)
        stats = compute_stats(eval_df)
        return {
            "stats": stats,
            "eval_df": eval_df,
            "error": None,
        }
    except Exception as e:
        return {
            "stats": {"error": str(e)},
            "eval_df": pd.DataFrame(),
            "error": str(e),
        }
