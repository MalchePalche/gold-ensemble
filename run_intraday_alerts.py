"""
run_intraday_alerts.py — intraday NY-session Telegram alerts.

Runs every 5 minutes during the NY session only (13:30–20:00 UTC /
16:30–23:00 Sofia). Reads today's daily bias from Supabase, computes the
current intraday confirmation signal, and sends a Telegram alert ONLY when the
signal STATE changes (e.g. WAIT → ENTER, ENTER → AGAINST, ENTER STRONG →
ENTER WEAK). Stays silent on WAIT → WAIT so it never spams.

The last known state is persisted to data/cache/intraday_state.json so the next
5-minute run can detect a change.

Intended to be driven by Windows Task Scheduler — see README.md "Scheduling".
"""

import json
import os
import sys
from datetime import datetime, timezone

import pytz
from dotenv import load_dotenv

load_dotenv()

# Anchor paths and imports to this file so the script works regardless of the
# directory Task Scheduler launches it from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SOFIA_TZ = pytz.timezone('Europe/Sofia')
STATE_FILE = os.path.join(HERE, 'data', 'cache', 'intraday_state.json')


def is_ny_session() -> bool:
    now_utc = datetime.now(timezone.utc)
    # NY session: 13:30–20:00 UTC
    # Skip weekends
    if now_utc.weekday() >= 5:
        return False
    session_start = now_utc.replace(
        hour=13, minute=30, second=0, microsecond=0)
    session_end = now_utc.replace(
        hour=20, minute=0, second=0, microsecond=0)
    return session_start <= now_utc <= session_end


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"signal": None, "timestamp": None, "key_level": None}


def save_state(signal: str, key_level, timestamp: str):
    os.makedirs(os.path.join(HERE, 'data', 'cache'), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
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
                  position_size: float) -> str:
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

    if 'ENTER' in sig:
        emoji = '✅✅' if sig == 'ENTER STRONG' else '✅'
        return (
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
            f"🚫 PRICE AGAINST BIAS — XAU/USD\n"
            f"{date_str} {time_str} Sofia\n\n"
            f"Bias: {bias} {confidence:.1f}%\n"
            f"{reason}\n\n"
            f"Do not enter. Wait for price to return to bias direction."
        )
    elif sig == 'WATCH CLOSELY':
        direction = 'above' if bias == 'BULLISH' else 'below'
        return (
            f"👁️ WATCH CLOSELY — XAU/USD\n"
            f"{time_str} Sofia\n\n"
            f"Volume rising {vol_ratio:.1f}x avg\n"
            f"Watching for break {direction} ${key_level}"
        )
    return None


def main():
    if not is_ny_session():
        print("[intraday_alerts] Outside NY session — exiting")
        sys.exit(0)

    import requests

    from db.queries import get_latest_signal
    from data.intraday import get_intraday_analysis

    latest = get_latest_signal()
    if not latest:
        print("[intraday_alerts] No signal in DB")
        sys.exit(0)

    bias = latest.get('bias', 'NEUTRAL')
    confidence = latest.get('confidence', 0)
    position_size = latest.get('position_size', 0)

    if bias == 'NEUTRAL':
        print("[intraday_alerts] Bias neutral — no alerts")
        sys.exit(0)

    # Get intraday analysis
    intraday = get_intraday_analysis(bias)
    if intraday.get('error'):
        print(f"[intraday_alerts] Error: {intraday['error']}")
        sys.exit(1)

    confirmation = intraday.get('confirmation', {})
    volume = intraday.get('volume', {})
    current_signal = confirmation.get('signal', 'WAIT')

    # Check state change
    old_state = load_state()
    if not signal_changed(old_state, current_signal):
        print(f"[intraday_alerts] No change: {current_signal} — silent")
        sys.exit(0)

    # Build and send message
    msg = build_message(
        confirmation, volume, bias, confidence, position_size)

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
                print(f"[intraday_alerts] Sent: {current_signal}")
            else:
                print(f"[intraday_alerts] Telegram error: {r.json()}")
        except Exception as e:
            print(f"[intraday_alerts] Send failed: {e}")

    # Save new state
    save_state(
        current_signal,
        confirmation.get('key_level'),
        datetime.now(timezone.utc).isoformat(),
    )


if __name__ == '__main__':
    main()
