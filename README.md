# Gold Ensemble V4 — XAU/USD Daily Bias Engine

Weighted ensemble of four trend-following strategies with volatility-aware position
sizing, a circuit breaker, and entry smoothing. Generates a daily BULLISH / NEUTRAL /
BEARISH signal with a suggested position size (0x – 1.5x).

## Architecture

```
Gold OHLCV (yfinance GC=F)
        |
        v
  EnsembleModel  ─── S1: EMA 20/50 crossover       (weight 1.5)
                  ─── S2: Donchian breakout 55d      (weight 0.5)
                  ─── S4: 52-week high/low momentum  (weight 1.5)
                  ─── S5: MACD 12/26/9               (weight 2.0)
        |
        v
  Volatility regime  ── ATR ratio (14d / 20d avg)
                     ── Realised vol percentile (20d / 252d rank)
        |
        v
  6-tier sizing matrix  (signal x confidence x vol regime)
        |
        v
  Circuit breaker  ── 20d portfolio return < -8%  -> 0x for 10 bars
        |
        v
  Entry smoothing  ── scale in over 3 days (bull flip)
                   ── scale out over 2 days (non-bull flip)
        |
        v
  results/signals.csv  /  Telegram alert
```

## Sizing matrix

| Signal  | Confidence | Vol regime | Size  |
|---------|------------|------------|-------|
| BULLISH | > 60%      | normal     | 1.5x  |
| BULLISH | > 60%      | elevated   | 1.0x  |
| BULLISH | > 60%      | extreme    | 0.5x  |
| BULLISH | <= 60%     | normal     | 1.0x  |
| BULLISH | <= 60%     | elevated   | 0.75x |
| BULLISH | <= 60%     | extreme    | 0.25x |
| NEUTRAL | —          | any        | 0.5x  |
| BEARISH | —          | any        | 0.0x  |

## Backtest results (V4, test period 2020–2025)

| Metric       | V4      | Buy & Hold |
|--------------|---------|------------|
| Sharpe       | +0.980  | +0.887     |
| Max drawdown | -21.6%  | -18.9%     |
| Total return | +110.5% | +83.2%     |
| CB triggers  | 0       | —          |

## Setup

```bash
cd gold_ensemble
pip install -r requirements.txt
```

Python 3.9+ required. Tested on 3.11.

## Running

### Daily signal (one-shot)

```bash
python main.py
```

### Daily runner (full output + Telegram alerts)

```bash
python run_daily.py
```

### Full V4 backtest

```bash
python main.py --backtest
```

### Streamlit dashboard

```bash
python main.py --dashboard
# or directly:
streamlit run dashboard/app.py
```

## Telegram alerts

Alerts fire only when:
- The bias flips (e.g. BULLISH -> BEARISH)
- The position size changes by >= 0.5x

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and copy the token.
2. Send your bot a message, then get the `chat_id`:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

3. Edit `config.yaml`:

```yaml
telegram:
  enabled: true
  bot_token: "123456789:ABCdef..."
  chat_id: "987654321"
```

## Scheduling on Windows (Task Scheduler)

Run `run_daily.py` automatically after US market close (e.g. 18:30 ET):

```
Program:   C:\Users\usr\AppData\Local\Programs\Python\Python311\python.exe
Arguments: D:\MyDesktop\Trading_Ensemble\gold_ensemble\run_daily.py
Trigger:   Daily at 18:30
```

## Signal output — results/signals.csv

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| date           | Bar date (trading day)                           |
| close          | XAU/USD closing price                            |
| bias           | BULLISH / NEUTRAL / BEARISH                      |
| confidence_pct | 0–100; how strongly the ensemble agrees          |
| vol_regime     | normal / elevated / extreme                      |
| atr_ratio      | ATR(14) / 20d avg ATR                            |
| rv_pct         | Realised vol percentile rank (252d lookback)     |
| matrix_target  | Raw sizing matrix output (before smoothing / CB) |
| position_size  | Final suggested size after all layers            |
| cb_active      | True if circuit breaker is forcing 0x            |
| bias_flipped   | True if bias changed vs previous row             |
| pos_change     | Position change vs previous row                  |

## File structure

```
gold_ensemble/
  config.yaml           -- all parameters
  run_daily.py          -- daily runner (use this for production)
  main.py               -- CLI entry point
  requirements.txt

  data/
    loader.py           -- gold OHLCV via yfinance (pickle cache)

  ensemble/
    model.py            -- weighted voting machine (S1/S2/S4/S5)
    sizer.py            -- vol regime, sizing matrix, V4 simulation

  strategies/
    s1_ma_crossover.py
    s2_donchian.py
    s4_52w_momentum.py
    s5_macd.py

  backtest/
    engine.py           -- walk-forward backtest utilities

  dashboard/
    app.py              -- Streamlit UI

  v4.py                 -- V4 backtest reference script
  results/
    signals.csv         -- daily signal log (auto-created on first run)
```
