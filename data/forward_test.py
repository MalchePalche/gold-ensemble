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

from datetime import date, datetime, timedelta

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


def auto_evaluate_pending(supabase_client) -> dict:
    """
    Runs automatically each day as part of run_daily.py.

    Steps:
    1. Fetch all unevaluated signals from Supabase
       (evaluated = false, bias != NEUTRAL, date < today)
    2. Fetch recent gold returns via yfinance
    3. For each unevaluated signal find next trading day return
    4. Mark correct/incorrect and save back to Supabase
    5. Return summary of what was evaluated
    """
    today = date.today()

    # Fetch unevaluated signals older than today.
    try:
        res = (
            supabase_client.table("signals")
            .select("id, date, bias, confidence")
            .eq("evaluated", False)
            .neq("bias", "NEUTRAL")
            .lt("date", str(today))
            .execute()
        )
        pending = res.data
    except Exception as e:
        return {"error": str(e), "evaluated": 0}

    if not pending:
        return {
            "error": None,
            "evaluated": 0,
            "message": "No pending signals to evaluate",
        }

    # Fetch daily returns over a window wide enough to cover the OLDEST pending
    # signal — a fixed 30-day window silently drops any signal older than that
    # (e.g. after the runner was down for a stretch), so it would never be scored.
    oldest = min(pd.to_datetime(s["date"]).date() for s in pending)
    days_needed = (date.today() - oldest).days + 10
    days_needed = max(30, min(days_needed, 365))
    returns = fetch_gold_returns(days_needed)
    evaluated_count = 0
    results = []

    for sig in pending:
        sig_date = pd.to_datetime(sig["date"]).date()
        bias = sig["bias"]

        # Find next trading day return (look up to 5 days for weekends/holidays).
        next_return = None
        next_date = sig_date + timedelta(days=1)
        for _ in range(5):
            if next_date in returns.index:
                next_return = float(returns[next_date])
                break
            next_date += timedelta(days=1)

        if next_return is None:
            continue  # Still no next-day data — try again tomorrow.

        # Evaluate direction.
        if bias == "BULLISH":
            correct = bool(next_return > 0)
        else:  # BEARISH
            correct = bool(next_return < 0)

        # Persist the outcome back to Supabase.
        try:
            supabase_client.table("signals").update({
                "next_day_return": round(next_return * 100, 4),
                "evaluated": True,
                "correct": correct,
                "evaluation_date": str(today),
            }).eq("id", sig["id"]).execute()
            evaluated_count += 1
            results.append({
                "date": str(sig_date),
                "bias": bias,
                "return": round(next_return * 100, 3),
                "correct": correct,
            })
        except Exception:
            continue

    return {
        "error": None,
        "evaluated": evaluated_count,
        "results": results,
    }


def _compute_stats_core(evaluated: pd.DataFrame, total_signals: int,
                        pending_count: int) -> dict:
    """Shared metric computation over an already-evaluated frame.

    `evaluated` must carry the columns: bias, date, confidence (float),
    correct (bool), next_return (float, percentage points). Both
    `compute_stats()` (on-the-fly evaluation) and `compute_stats_from_db()`
    (stored outcomes) normalise their input to this shape and delegate here,
    so the win-rate / streak / calibration logic lives in exactly one place.
    """
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
        "total_signals": total_signals,
        "evaluated": total_eval,
        "pending": pending_count,
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
        "pending_signals": pending_count,
    }


def compute_stats(eval_df: pd.DataFrame) -> dict:
    """
    Compute forward-test statistics from on-the-fly evaluated signals (the
    `evaluate_signals()` output, with a `status` column). Delegates the metric
    math to `_compute_stats_core()`.
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

    return _compute_stats_core(evaluated, total_signals=len(eval_df),
                               pending_count=len(pending))


def compute_stats_from_db(df: pd.DataFrame) -> dict:
    """
    Compute forward-test statistics from already-evaluated signals stored in
    Supabase. Same metrics as `compute_stats()`, but the DB columns differ:
    `next_day_return` is already a percentage and `correct` is already boolean,
    so we just normalise the column names and delegate to the shared core.
    """
    if df.empty:
        return {"error": "No evaluated signals yet"}

    evaluated = df.copy()
    # Normalise the column names the core expects.
    evaluated["next_return"] = evaluated["next_day_return"].astype(float)
    evaluated["correct"] = evaluated["correct"].astype(bool)
    evaluated["confidence"] = evaluated["confidence"].fillna(0).astype(float)

    return _compute_stats_core(evaluated, total_signals=len(evaluated),
                               pending_count=0)


def _empty_confidence_buckets() -> list[dict]:
    """A zero-data confidence-bucket list (one entry per bucket)."""
    return [
        {"bucket": name, "count": 0, "win_rate": None, "avg_return": None}
        for name in ("low", "mid", "high")
    ]


def _compute_confidence_buckets(eval_df: pd.DataFrame) -> list[dict]:
    """Group evaluated signals by confidence band and score each band.

    Bands: low (<45%), mid (45–60% inclusive), high (>60%). Per band we report
    count, win_rate (%) and avg_return (% — `next_day_return` is already stored
    in percentage points). A band with no signals comes back as
    count=0 / win_rate=None / avg_return=None so the shape is always stable.
    """
    if eval_df.empty:
        return _empty_confidence_buckets()

    df = eval_df.copy()
    df["confidence"] = df["confidence"].fillna(0).astype(float)
    df["correct"] = df["correct"].astype(bool)
    df["next_day_return"] = df["next_day_return"].astype(float)

    band_masks = [
        ("low", df["confidence"] < 45),
        ("mid", (df["confidence"] >= 45) & (df["confidence"] <= 60)),
        ("high", df["confidence"] > 60),
    ]

    out = []
    for name, mask in band_masks:
        bucket = df[mask]
        if len(bucket) == 0:
            out.append({"bucket": name, "count": 0,
                        "win_rate": None, "avg_return": None})
        else:
            out.append({
                "bucket": name,
                "count": int(len(bucket)),
                "win_rate": round(bucket["correct"].sum() / len(bucket) * 100, 1),
                "avg_return": round(float(bucket["next_day_return"].mean()), 3),
            })
    return out


def _compute_strategy_accuracy() -> dict:
    """Per-strategy hit rate when a strategy agreed with the ensemble bias.

    For each of S1/S2/S4/S5 we pull the evaluated rows where that strategy's
    stored signal matched the ensemble `bias`, then report count, win_rate (%)
    and avg_return (%). Each strategy is queried independently so that a missing
    column degrades to None for just that strategy rather than wiping the lot.
    """
    from db.supabase_client import supabase

    columns = {
        "S1": "s1_signal",
        "S2": "s2_signal",
        "S4": "s4_signal",
        "S5": "s5_signal",
    }

    result: dict = {}
    for label, col in columns.items():
        try:
            res = (
                supabase.table("signals")
                .select(f"{col}, bias, correct, next_day_return")
                .eq("evaluated", True)
                .execute()
            )
            rows = res.data or []
            matched = [
                r for r in rows
                if r.get(col) is not None
                and r.get(col) == r.get("bias")
                and r.get("correct") is not None
                and r.get("next_day_return") is not None
            ]
            count = len(matched)
            if count == 0:
                result[label] = {"count": 0, "win_rate": None,
                                 "avg_return": None}
            else:
                wins = sum(1 for r in matched if r["correct"])
                avg_ret = sum(float(r["next_day_return"])
                              for r in matched) / count
                result[label] = {
                    "count": count,
                    "win_rate": round(wins / count * 100, 1),
                    "avg_return": round(avg_ret, 3),
                }
        except Exception:
            # Column missing (or any query error) — degrade gracefully.
            result[label] = None
    return result


def get_signal_duration_stats(supabase_client) -> dict:
    """Run-length analytics: how long the system holds a bias and how those
    runs perform.

    Walks every stored signal in date order, groups consecutive days that share
    the same bias into "runs" (e.g. four BULLISH days in a row = duration 4),
    and records each completed run's bias, duration and the next-day return of
    its last bar. Only runs whose last bar has been evaluated (non-null
    next_day_return) are counted.

    Returns:
        {
          "avg_duration": {"BULLISH": X, "BEARISH": Y, "NEUTRAL": Z},
          "by_duration_bucket": {
              "short_1_2":   {"count": N, "win_rate": X},
              "medium_3_5":  {"count": N, "win_rate": X},
              "long_6_plus": {"count": N, "win_rate": X},
          },
        }

    NEUTRAL is an abstain everywhere else in the forward test, so it carries an
    avg_duration but is excluded from the directional win-rate buckets.
    """
    from collections import defaultdict

    empty = {
        "avg_duration": {"BULLISH": None, "BEARISH": None, "NEUTRAL": None},
        "by_duration_bucket": {
            "short_1_2": {"count": 0, "win_rate": None},
            "medium_3_5": {"count": 0, "win_rate": None},
            "long_6_plus": {"count": 0, "win_rate": None},
        },
    }

    try:
        res = (
            supabase_client.table("signals")
            .select("date, bias, next_day_return")
            .order("date", desc=False)
            .execute()
        )
        rows = res.data or []
    except Exception:
        return empty

    if not rows:
        return empty

    # ── Collapse consecutive same-bias days into runs ───────────────────────
    runs: list[dict] = []
    cur_bias = None
    cur_len = 0
    cur_last_ret = None
    for r in rows:
        bias = r.get("bias")
        if bias == cur_bias:
            cur_len += 1
            cur_last_ret = r.get("next_day_return")
        else:
            if cur_bias is not None:
                runs.append({"bias": cur_bias, "duration": cur_len,
                             "next_day_return": cur_last_ret})
            cur_bias = bias
            cur_len = 1
            cur_last_ret = r.get("next_day_return")
    if cur_bias is not None:
        runs.append({"bias": cur_bias, "duration": cur_len,
                     "next_day_return": cur_last_ret})

    # Keep only evaluated runs (last bar has a recorded next-day return).
    runs = [r for r in runs if r["next_day_return"] is not None]
    if not runs:
        return empty

    # ── Average duration per bias ───────────────────────────────────────────
    durations = defaultdict(list)
    for r in runs:
        durations[r["bias"]].append(r["duration"])

    avg_duration = {}
    for bias in ("BULLISH", "BEARISH", "NEUTRAL"):
        vals = durations.get(bias, [])
        avg_duration[bias] = (round(sum(vals) / len(vals), 2)
                              if vals else None)

    # ── Win rate by duration bucket (directional runs only) ─────────────────
    buckets: dict[str, list[bool]] = {
        "short_1_2": [],
        "medium_3_5": [],
        "long_6_plus": [],
    }
    for r in runs:
        if r["bias"] not in ("BULLISH", "BEARISH"):
            continue  # NEUTRAL abstains — no directional win/loss.
        d = r["duration"]
        if d <= 2:
            key = "short_1_2"
        elif d <= 5:
            key = "medium_3_5"
        else:
            key = "long_6_plus"
        ret = float(r["next_day_return"])
        win = ((r["bias"] == "BULLISH" and ret > 0) or
               (r["bias"] == "BEARISH" and ret < 0))
        buckets[key].append(win)

    by_duration_bucket = {}
    for key, wins in buckets.items():
        if wins:
            by_duration_bucket[key] = {
                "count": len(wins),
                "win_rate": round(sum(1 for w in wins if w) / len(wins) * 100, 1),
            }
        else:
            by_duration_bucket[key] = {"count": 0, "win_rate": None}

    return {"avg_duration": avg_duration,
            "by_duration_bucket": by_duration_bucket}


def get_forward_test_analysis() -> dict:
    """Main entry point — read evaluated signals from Supabase and summarize.

    Evaluation itself now happens once per day in `auto_evaluate_pending()`
    (called from run_daily.py), so the dashboard just reads the stored
    outcomes instead of recomputing returns on every page load.
    """
    try:
        from db.queries import get_evaluated_signals, get_pending_signals
        evaluated = get_evaluated_signals(90)
        pending = get_pending_signals()

        if not evaluated:
            return {
                "stats": {
                    "error": None,
                    "evaluated": 0,
                    "pending": len(pending),
                    "message": (f"Tracking {len(pending)} pending signal(s). "
                                f"First evaluation tomorrow."),
                },
                "eval_df": pd.DataFrame(),
                "confidence_buckets": _empty_confidence_buckets(),
                "strategy_accuracy": {"S1": None, "S2": None,
                                      "S4": None, "S5": None},
            }

        eval_df = pd.DataFrame(evaluated)
        stats = compute_stats_from_db(eval_df)

        # New analytics layered on top of the existing stats (additive — the
        # original keys are untouched).
        confidence_buckets = _compute_confidence_buckets(eval_df)
        strategy_accuracy = _compute_strategy_accuracy()

        return {
            "stats": stats,
            "eval_df": eval_df,
            "error": None,
            "confidence_buckets": confidence_buckets,
            "strategy_accuracy": strategy_accuracy,
        }
    except Exception as e:
        return {
            "stats": {"error": str(e)},
            "eval_df": pd.DataFrame(),
            "error": str(e),
        }
