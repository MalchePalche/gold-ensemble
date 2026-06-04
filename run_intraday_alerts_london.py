# Windows Task Scheduler setup for London session:
# Name:     Gold London Session Alerts
# Trigger:  Daily, repeat every 5 minutes
#           Start: 09:00 Sofia time
#           Stop after: 4 hours (covers 09:00-13:00)
# Action:   python D:\path\to\gold_ensemble\run_intraday_alerts_london.py
# Start in: D:\path\to\gold_ensemble
"""
run_intraday_alerts_london.py — intraday London-session Telegram alerts.

Gold moves most during the London open and the London/NY overlap, so this
mirrors run_intraday_alerts.py (NY) for the London session. Runs every 5
minutes during the London window only (09:00–13:00 Sofia — wide enough to cover
the London open across both DST cases). Reads today's daily bias from Supabase,
computes the current intraday confirmation signal, and sends a Telegram alert
ONLY when the signal STATE changes. Stays silent on WAIT → WAIT so it never
spams.

The shared logic lives in data/intraday_runner.py (see run_session_alerts);
this file only supplies the London-specific session window + flag. The last
known state is persisted to data/cache/intraday_state_london.json — a separate
file from the NY run, so the two sessions never cross-trigger each other's
state-change detection.

Intended to be driven by Windows Task Scheduler — see README.md "Scheduling".
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Anchor paths and imports to this file so the script works regardless of the
# directory Task Scheduler launches it from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data.intraday_runner import run_session_alerts, SOFIA_TZ


def is_london_session() -> bool:
    now_sofia = datetime.now(SOFIA_TZ)
    # London session: 09:00–13:00 Sofia (wide enough for London open across
    # both DST cases). Skip weekends.
    if now_sofia.weekday() >= 5:
        return False
    session_start = now_sofia.replace(
        hour=9, minute=0, second=0, microsecond=0)
    session_end = now_sofia.replace(
        hour=13, minute=0, second=0, microsecond=0)
    return session_start <= now_sofia <= session_end


if __name__ == '__main__':
    run_session_alerts('LONDON', '🇬🇧', is_london_session)
