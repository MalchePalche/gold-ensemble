"""
data/intraday_runner.py — shared intraday-session Telegram alert logic.

Both run_intraday_alerts.py (NY) and run_intraday_alerts_london.py (London) are
thin wrappers around run_session_alerts(). Each session passes its own name,
flag emoji, and a session-active check, then gets identical downstream logic:
read today's daily bias from Supabase, compute the intraday confirmation
signal, and send a Telegram alert ONLY when the signal STATE changes (e.g.
WAIT → ENTER, ENTER → AGAINST, ENTER STRONG → ENTER WEAK). Stays silent on
WAIT → WAIT so it never spams.

The session-active check is passed in (not derived from fixed Sofia hours)
because the two sessions are anchored differently: NY is UTC-anchored (US
market hours, DST-safe in UTC) while London is Sofia-anchored. Each session
also persists its own state file (data/cache/intraday_state_<session>.json) so
the two runs never cross-trigger each other's state-change detection.
"""

import json
import os
from datetime import datetime, timezone

import pytz

# gold_ensemble root = parent of this data/ package. Anchored to this file so
# everything works regardless of the directory Task Scheduler launches from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOFIA_TZ = pytz.timezone('Europe/Sofia')


def _state_file(session_name: str) -> str:
    return os.path.join(
        ROOT, 'data', 'cache',
        f'intraday_state_{session_name.lower()}.json')


def load_state(session_name: str) -> dict:
    try:
        with open(_state_file(session_name)) as f:
            return json.load(f)
    except Exception:
        return {"signal": None, "timestamp": None, "key_level": None}


def save_state(session_name: str, signal: str, key_level, timestamp: str):
    os.makedirs(os.path.join(ROOT, 'data', 'cache'), exist_ok=True)
    with open(_state_file(session_name), 'w') as f:
        json.dump({
            "signal": signal,
            "timestamp": timestamp,
            "key_level": key_level,
        }, f)


def signal_changed(old_state: dict, new_signal: str) -> bool:
    old = old_state.get("signal")
    if old == new_signal:
        return False
    # Ignore minor WAIT variants staying as WAIT
    wait_variants = {"WAIT", None}
    if old in wait_variants and new_signal == "WAIT":
        return False
    return True


def build_message(confirmation: dict,
                  volume: dict,
                  bias: str,
                  confidence: float,
                  position_size: float,
                  session_name: str,
                  session_flag: str) -> str:
    now_sofia = datetime.now(SOFIA_TZ)
    date_str = now_sofia.strftime('%Y-%m-%d')
    time_str = now_sofia.strftime('%H:%M')
    sig = confirmation.get('signal', 'WAIT')
    reason = confirmation.get('reason', '')
    entry_zone = confirmation.get('entry_zone', '')
    key_level = confirmation.get('key_level', '')
    vol_ratio = volume.get('vol_ratio', 1.0) if volume else 1.0
    vol_quality = confirmation.get('volume_quality', 'NORMAL')
    size_mod = confirmation.get('size_modifier', 0.0)
    suggested_size = round(position_size * size_mod, 2)

    # Session prefix so the two alert streams (NY / London) are distinguishable.
    prefix = f"{session_flag} {session_name} SESSION"

    if 'ENTER' in sig:
        emoji = '✅✅' if sig == 'ENTER STRONG' else '✅'
        return (
            f"{prefix}\n"
            f"{emoji} INTRADAY ENTRY — XAU/USD\n"
            f"{date_str} {time_str} Sofia\n\n"
            f"Bias: {bias} {confidence:.1f}%\n"
            f"Signal: {sig}\n"
            f"{reason}\n"
            f"Entry zone: {entry_zone}\n"
            f"Key level: ${key_level}\n"
            f"Volume: {vol_ratio:.1f}x avg ({vol_quality})\n\n"
            f"Size suggestion: {suggested_size}x"
        )
    elif sig == 'AGAINST':
        return (
            f"{prefix}\n"
            f"🚫 PRICE AGAINST BIAS — XAU/USD\n"
            f"{date_str} {time_str} Sofia\n\n"
            f"Bias: {bias} {confidence:.1f}%\n"
            f"{reason}\n\n"
            f"Do not enter. Wait for price to return to bias direction."
        )
    elif sig == 'WATCH CLOSELY':
        direction = 'above' if bias == 'BULLISH' else 'below'
        return (
            f"{prefix}\n"
            f"👁️ WATCH CLOSELY — XAU/USD\n"
            f"{time_str} Sofia\n\n"
            f"Volume rising {vol_ratio:.1f}x avg\n"
            f"Watching for break {direction} ${key_level}"
        )
    return None


def run_session_alerts(session_name: str,
                       session_flag: str,
                       is_session_active) -> None:
    """Run one intraday alert check for the given session.

    session_name      — e.g. "NY" / "LONDON" (also names the state file).
    session_flag       — flag emoji used in the Telegram message prefix.
    is_session_active — zero-arg callable returning True if the session window
                         is currently open (NY is UTC-anchored, London Sofia-
                         anchored — each wrapper supplies its own).
    """
    if not is_session_active():
        print(f"[intraday_alerts:{session_name}] "
              f"Outside {session_name} session — exiting")
        return

    # Ensure the gold_ensemble root is importable regardless of launch dir.
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import requests

    from db.queries import get_latest_signal
    from data.intraday import get_intraday_analysis

    latest = get_latest_signal()
    if not latest:
        print(f"[intraday_alerts:{session_name}] No signal in DB")
        return

    bias = latest.get('bias', 'NEUTRAL')
    confidence = latest.get('confidence', 0)
    position_size = latest.get('position_size', 0)

    if bias == 'NEUTRAL':
        print(f"[intraday_alerts:{session_name}] Bias neutral — no alerts")
        return

    # Get intraday analysis
    intraday = get_intraday_analysis(bias)
    if intraday.get('error'):
        print(f"[intraday_alerts:{session_name}] Error: {intraday['error']}")
        sys.exit(1)

    confirmation = intraday.get('confirmation', {})
    volume = intraday.get('volume', {})
    current_signal = confirmation.get('signal', 'WAIT')

    # Check state change (per-session state file)
    old_state = load_state(session_name)
    if not signal_changed(old_state, current_signal):
        print(f"[intraday_alerts:{session_name}] "
              f"No change: {current_signal} — silent")
        return

    # Build and send message
    msg = build_message(
        confirmation, volume, bias, confidence, position_size,
        session_name, session_flag)

    if msg:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        try:
            r = requests.post(
                f'https://api.telegram.org/bot{token}/sendMessage',
                json={'chat_id': chat_id, 'text': msg},
                timeout=10,
            )
            if r.json().get('ok'):
                print(f"[intraday_alerts:{session_name}] Sent: {current_signal}")
            else:
                print(f"[intraday_alerts:{session_name}] "
                      f"Telegram error: {r.json()}")
        except Exception as e:
            print(f"[intraday_alerts:{session_name}] Send failed: {e}")

    # Save new state
    save_state(
        session_name,
        current_signal,
        confirmation.get('key_level'),
        datetime.now(timezone.utc).isoformat(),
    )
