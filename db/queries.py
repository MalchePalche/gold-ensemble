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
