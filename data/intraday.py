"""
data/intraday.py — intraday confirmation layer for the gold ensemble.

The daily ensemble gives a directional *bias*; this module gives the *timing*
within the day. It fetches 5-minute XAU/USD bars, computes London / NY opening
range levels, reads short-term price action (EMA9 slope, volume, momentum) and
turns the daily bias into an actionable ENTER / WAIT / AGAINST confirmation.

Data source: spot XAU/USD where available, else GC=F futures shifted down by an
estimated premium so levels read in spot terms (see fetch_intraday). Via yfinance.
"""

from __future__ import annotations

import os

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

SOFIA_TZ = pytz.timezone('Europe/Sofia')
LONDON_OPEN_UTC = 8   # 08:00 UTC
NY_OPEN_UTC = 13      # 13:30 UTC

# yfinance has no intraday data for spot XAU/USD (XAUUSD=X is empty), so we fall
# back to GC=F futures, which trade ~$15-30 above spot. To make session levels
# match the trader's spot charts we subtract an estimated premium from futures
# OHLC. Override the estimate with the INTRADAY_SPOT_PREMIUM env var.
FUTURES_TICKERS = {'GC=F', 'MGC=F'}
SPOT_PREMIUM = float(os.getenv('INTRADAY_SPOT_PREMIUM', '25.0'))


def fetch_intraday(interval: str = '5m',
                   hours_back: int = 48) -> tuple[pd.DataFrame, str | None, float]:
    """
    Fetch 5-minute gold bars for last 48 hours, expressed in spot terms.

    Tries spot XAU/USD first so session levels match the trader's spot charts;
    GC=F futures are the fallback (yfinance has no intraday spot). When a futures
    source is used, OHLC is shifted down by SPOT_PREMIUM so levels read in spot.
    Returns (df, ticker_used, premium_applied) — ticker_used is None if every
    source failed; premium_applied is the dollar offset subtracted (0 for spot).
    """
    end = datetime.utcnow()
    start = end - timedelta(hours=hours_back)

    tickers = ['XAUUSD=X', 'GC=F']
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval=interval, progress=False)
            if not df.empty and len(df) > 10:
                # Recent yfinance returns MultiIndex columns (Price, Ticker)
                # even for a single ticker; flatten so df['High'] is a Series.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index, utc=True)

                # Convert futures prices to spot by subtracting the premium.
                # Volume is left alone; price *differences* (range, 5m change,
                # EMA slope) are unaffected by a constant shift.
                premium = SPOT_PREMIUM if ticker in FUTURES_TICKERS else 0.0
                if premium:
                    price_cols = [c for c in ('Open', 'High', 'Low', 'Close')
                                  if c in df.columns]
                    df[price_cols] = df[price_cols] - premium

                return df, ticker, premium
        except Exception:
            continue

    return pd.DataFrame(), None, 0.0


def get_session_levels(df: pd.DataFrame) -> dict:
    """
    Compute London and NY opening range levels.
    London: 08:00–09:00 UTC
    NY: 13:30–14:30 UTC
    Uses most recent completed sessions.
    """
    now_utc = datetime.now(pytz.UTC)
    today = now_utc.date()

    results = {}

    for session_name, open_h, open_m, dur_h in [
        ('london', 8, 0, 1),
        ('ny', 13, 30, 1)
    ]:
        session_start = datetime(today.year, today.month, today.day,
                                  open_h, open_m, tzinfo=pytz.UTC)
        session_end = session_start + timedelta(hours=dur_h)

        # If session hasn't started yet use yesterday's
        if now_utc < session_start:
            session_start -= timedelta(days=1)
            session_end -= timedelta(days=1)

        mask = (df.index >= session_start) & (df.index <= session_end)
        session_bars = df[mask]

        if len(session_bars) < 3:
            results[session_name] = None
            continue

        high = float(session_bars['High'].max())
        low = float(session_bars['Low'].min())
        mid = (high + low) / 2

        results[session_name] = {
            'high': round(high, 2),
            'low': round(low, 2),
            'mid': round(mid, 2),
            'range': round(high - low, 2),
            'session_start': session_start,
            'session_end': session_end,
        }

    return results


def get_current_price_action(df: pd.DataFrame) -> dict:
    """
    Analyze last 15 bars (75 min on 5m chart) for:
    - Current price
    - Short-term trend (last 15 bars EMA slope)
    - Momentum (last bar vs 15-bar avg)
    - Recent high/low
    """
    if len(df) < 15:
        return {}

    recent = df.tail(15)
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])

    # EMA slope on recent bars
    closes = recent['Close'].squeeze()
    ema9 = closes.ewm(span=9).mean()
    slope = float(ema9.iloc[-1]) - float(ema9.iloc[-5])

    # Volume context
    avg_vol = float(recent['Volume'].mean())
    last_vol = float(df['Volume'].iloc[-1])
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

    return {
        'current': round(current_price, 2),
        'prev': round(prev_price, 2),
        'change_5m': round(current_price - prev_price, 2),
        'ema9_slope': round(slope, 3),
        'vol_ratio': round(vol_ratio, 2),
        'recent_high': round(float(recent['High'].max()), 2),
        'recent_low': round(float(recent['Low'].min()), 2),
    }


def get_confirmation_signal(
        df: pd.DataFrame,
        daily_bias: str,
        session_levels: dict,
        price_action: dict
) -> dict:
    """
    Generate intraday entry confirmation.

    Rules:
    BULLISH daily bias → look for:
    1. Price breaks above London or NY session high
    2. EMA9 slope positive (momentum up)
    3. Volume above average on the break

    BEARISH daily bias → look for:
    1. Price breaks below London or NY session low
    2. EMA9 slope negative (momentum down)
    3. Volume above average on the break

    Output:
    - confirmed: bool
    - signal: ENTER / WAIT / AGAINST
    - reason: explanation string
    - key_level: the level being watched
    - entry_zone: specific price zone to watch
    """
    if not price_action or daily_bias == 'NEUTRAL':
        return {
            'confirmed': False,
            'signal': 'WAIT',
            'reason': 'No daily bias or insufficient data',
            'key_level': None,
            'entry_zone': None,
        }

    current = price_action['current']
    slope = price_action['ema9_slope']
    vol_ratio = price_action['vol_ratio']

    # Get best available session levels
    levels = session_levels.get('ny') or session_levels.get('london')
    if not levels:
        return {
            'confirmed': False,
            'signal': 'WAIT',
            'reason': 'Session levels not yet available',
            'key_level': None,
            'entry_zone': None,
        }

    session_high = levels['high']
    session_low = levels['low']

    if daily_bias == 'BULLISH':
        key_level = session_high
        if current > session_high and slope > 0 and vol_ratio > 1.0:
            return {
                'confirmed': True,
                'signal': 'ENTER',
                'reason': (f'Price broke above session high '
                           f'${session_high} with positive momentum '
                           f'and {vol_ratio:.1f}x volume'),
                'key_level': key_level,
                'entry_zone': f'${session_high} – ${session_high + levels["range"] * 0.2:.2f}',
            }
        elif current > session_high and slope > 0:
            return {
                'confirmed': True,
                'signal': 'ENTER',
                'reason': (f'Price above session high ${session_high}, '
                           f'momentum positive. Volume weak — '
                           f'consider smaller size.'),
                'key_level': key_level,
                'entry_zone': f'Near ${session_high:.2f}',
            }
        elif current < session_low:
            return {
                'confirmed': False,
                'signal': 'AGAINST',
                'reason': (f'Price below session low ${session_low} '
                           f'despite BULLISH bias — do not enter'),
                'key_level': session_low,
                'entry_zone': None,
            }
        else:
            return {
                'confirmed': False,
                'signal': 'WAIT',
                'reason': (f'Watching for break above '
                           f'${session_high} to confirm entry'),
                'key_level': key_level,
                'entry_zone': None,
            }

    elif daily_bias == 'BEARISH':
        key_level = session_low
        if current < session_low and slope < 0 and vol_ratio > 1.0:
            return {
                'confirmed': True,
                'signal': 'ENTER',
                'reason': (f'Price broke below session low '
                           f'${session_low} with negative momentum '
                           f'and {vol_ratio:.1f}x volume'),
                'key_level': key_level,
                'entry_zone': f'${session_low - levels["range"] * 0.2:.2f} – ${session_low}',
            }
        elif current < session_low and slope < 0:
            return {
                'confirmed': True,
                'signal': 'ENTER',
                'reason': (f'Price below session low ${session_low}, '
                           f'momentum negative. Volume weak — '
                           f'consider smaller size.'),
                'key_level': key_level,
                'entry_zone': f'Near ${session_low:.2f}',
            }
        elif current > session_high:
            return {
                'confirmed': False,
                'signal': 'AGAINST',
                'reason': (f'Price above session high ${session_high} '
                           f'despite BEARISH bias — do not enter'),
                'key_level': session_high,
                'entry_zone': None,
            }
        else:
            return {
                'confirmed': False,
                'signal': 'WAIT',
                'reason': (f'Watching for break below '
                           f'${session_low} to confirm entry'),
                'key_level': key_level,
                'entry_zone': None,
            }

    return {
        'confirmed': False,
        'signal': 'WAIT',
        'reason': 'No active signal',
        'key_level': None,
        'entry_zone': None,
    }


def get_intraday_analysis(daily_bias: str) -> dict:
    """Main entry point."""
    try:
        df, ticker, premium = fetch_intraday()
        if df.empty:
            return {'error': 'No intraday data available'}
        levels = get_session_levels(df)
        price_action = get_current_price_action(df)
        confirmation = get_confirmation_signal(
            df, daily_bias, levels, price_action)
        return {
            'ticker': ticker,
            'spot_premium': premium,
            'levels': levels,
            'price_action': price_action,
            'confirmation': confirmation,
            'error': None
        }
    except Exception as e:
        return {'error': str(e)}
