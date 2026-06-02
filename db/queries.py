"""
db/queries.py — read/write helpers for the `signals` table.

Used by the daily runner (save + previous-signal lookup) and the dashboard
(latest signal, recent history).
"""

from __future__ import annotations

from db.supabase_client import supabase


def save_signal(data: dict) -> None:
    """Upsert a signal row keyed on `date`.

    Re-running the same trading day overwrites that day's row instead of
    creating a duplicate. `data` must contain keys matching the `signals`
    table columns (see db/migrations.sql).
    """
    supabase.table("signals").upsert(data, on_conflict="date").execute()


def get_latest_signal() -> dict | None:
    """Most recent signal row, or None if the table is empty."""
    res = (
        supabase.table("signals")
        .select("*")
        .order("date", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def get_recent_signals(n: int = 14) -> list[dict]:
    """The `n` most recent signal rows, newest first."""
    res = (
        supabase.table("signals")
        .select("*")
        .order("date", desc=True)
        .limit(n)
        .execute()
    )
    return res.data


def get_evaluated_signals(limit: int = 90) -> list[dict]:
    """The most recent signals that have already been scored.

    Reads the stored forward-test columns written by
    `data.forward_test.auto_evaluate_pending()`, so the dashboard can render
    the live track record without recomputing returns on the fly.
    """
    res = (
        supabase.table("signals")
        .select(
            "date, bias, confidence, next_day_return, "
            "correct, evaluated, evaluation_date"
        )
        .eq("evaluated", True)
        .order("date", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data


def get_pending_signals() -> list[dict]:
    """Non-neutral signals still awaiting their next-day outcome."""
    res = (
        supabase.table("signals")
        .select("date, bias, confidence")
        .eq("evaluated", False)
        .neq("bias", "NEUTRAL")
        .order("date", desc=True)
        .execute()
    )
    return res.data
