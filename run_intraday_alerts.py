"""
run_intraday_alerts.py — intraday NY-session Telegram alerts (thin wrapper).

Runs every 5 minutes during the NY session only (13:30–20:00 UTC /
16:30–23:00 Sofia in summer, 15:30–22:00 in winter — the check is UTC-anchored
so it stays correct across DST). Reads today's daily bias from Supabase,
computes the current intraday confirmation signal, and sends a Telegram alert
ONLY when the signal STATE changes. Stays silent on WAIT → WAIT so it never
spams.

The shared logic lives in data/intraday_runner.py (see run_session_alerts);
this file only supplies the NY-specific session window + flag. The last known
state is persisted to data/cache/intraday_state_ny.json so the next 5-minute
run can detect a change.

Intended to be driven by Windows Task Scheduler — see README.md "Scheduling".
"""

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

# Anchor paths and imports to this file so the script works regardless of the
# directory Task Scheduler launches it from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data.intraday_runner import run_session_alerts


def is_ny_session() -> bool:
    now_utc = datetime.now(timezone.utc)
    # NY session: 13:30–20:00 UTC. Skip weekends.
    if now_utc.weekday() >= 5:
        return False
    session_start = now_utc.replace(
        hour=13, minute=30, second=0, microsecond=0)
    session_end = now_utc.replace(
        hour=20, minute=0, second=0, microsecond=0)
    return session_start <= now_utc <= session_end


if __name__ == '__main__':
    run_session_alerts('NY', '🗽', is_ny_session)
