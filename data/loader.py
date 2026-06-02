"""
data/loader.py — Gold OHLCV loader (GC=F via yfinance) with pickle cache.

V4 uses only gold OHLCV — all macro loaders (DXY, silver, TIPS, COT) removed.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError("yfinance is required: pip install yfinance") from e


DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class DataLoader:
    def __init__(
        self,
        gold_ticker: str = "GC=F",
        cache_dir: str = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
        cache_ttl_hours: int = 12,
    ):
        self.gold_ticker      = gold_ticker
        self.cache_dir        = cache_dir
        self.use_cache        = use_cache
        self.cache_ttl_hours  = cache_ttl_hours
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── cache helpers ──────────────────────────────────────────────────────
    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{name}.pkl")

    def _read_cache(self, name: str) -> Optional[pd.DataFrame]:
        if not self.use_cache:
            return None
        p = self._cache_path(name)
        if not os.path.exists(p):
            return None
        age_h = (time.time() - os.path.getmtime(p)) / 3600.0
        if age_h > self.cache_ttl_hours:
            return None
        try:
            return pd.read_pickle(p)
        except Exception as e:
            print(f"[loader] Warning: cache read for '{name}' failed: {e}")
            return None

    def _write_cache(self, name: str, df: pd.DataFrame) -> None:
        try:
            df.to_pickle(self._cache_path(name))
        except Exception as e:
            print(f"[loader] Warning: cache write for '{name}' failed: {e}")

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        keep = [c for c in ["open", "high", "low", "close", "adj close", "volume"]
                if c in df.columns]
        df = df[keep].copy()
        if "adj close" in df.columns and "close" in df.columns:
            df.drop(columns=["adj close"], inplace=True)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        full_idx = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(full_idx).ffill()
        df.index.name = "date"
        return df

    # ── public ─────────────────────────────────────────────────────────────
    def load_gold(self, years: int = 15) -> pd.DataFrame:
        """Daily OHLCV for XAU/USD futures."""
        cached = self._read_cache("gold")
        if cached is not None:
            return cached

        end   = datetime.utcnow()
        start = end - timedelta(days=int(365.25 * years))
        df = yf.download(
            self.gold_ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
        df = self._normalize_ohlcv(df)
        if df.empty:
            raise RuntimeError(f"Failed to load gold data for {self.gold_ticker}")
        self._write_cache("gold", df)
        return df


def load_all(years: int = 15, use_cache: bool = True, cache_ttl_hours: int = 12) -> dict:
    """Returns {'gold': DataFrame}."""
    loader = DataLoader(use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)
    return {"gold": loader.load_gold(years=years)}


if __name__ == "__main__":
    data = load_all(years=15)
    gold = data["gold"]
    print(f"gold: {gold.shape}  cols={list(gold.columns)}")
    print(gold.tail(3))
