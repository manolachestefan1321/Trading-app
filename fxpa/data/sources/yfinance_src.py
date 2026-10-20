"""yfinance adapter.

Quirks handled here:
- Yahoo serves 1H bars only for roughly the last 730 days, so the start is clamped and the
  range is requested in chunks.
- FX tickers look like EURUSD=X; gold uses the COMEX future GC=F (close to spot, not identical).
- FX volume from Yahoo is always 0, so volume confirmation is neutral with this source.
- Index time zone varies by ticker; everything is converted to UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from fxpa.config import Instrument
from fxpa.data.common import DataSourceError, chunk_ranges, empty_frame, normalize_ohlcv, retry_call, utcnow
from fxpa.data.sources.base import DataSource

MAX_DAYS = 729
CHUNK_DAYS = 120


class YFinanceSource(DataSource):
    name = "yfinance"

    def is_available(self) -> tuple[bool, str]:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False, "yfinance not installed"
        return True, ""

    def max_history_days(self) -> int | None:
        return MAX_DAYS

    def fetch(self, inst: Instrument, start: datetime, end: datetime, progress: bool = True) -> pd.DataFrame:
        import logging

        import yfinance as yf

        # yfinance logs every failed chunk at ERROR level; our own fallback logging is enough.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        earliest = utcnow() - timedelta(days=MAX_DAYS)
        start = max(start, earliest)
        if start >= end:
            return empty_frame()
        frames = []
        for a, b in tqdm(chunk_ranges(start, end, CHUNK_DAYS), desc=f"yfinance {inst.symbol}",
                         unit="chunk", disable=not progress):
            raw = retry_call(
                yf.download, inst.yfinance, start=a, end=b, interval="1h", progress=False,
                auto_adjust=False, multi_level_index=False, threads=False,
                retries=self.retries, base_delay=self.backoff, label=f"yfinance {inst.yfinance}",
            )
            if raw is None or raw.empty:
                continue
            raw = raw.rename(columns=str.lower)
            raw.index.name = "timestamp"
            frames.append(raw.reset_index()[["timestamp", "open", "high", "low", "close", "volume"]])
        if not frames:
            raise DataSourceError(f"yfinance returned no data for {inst.yfinance}")
        df = normalize_ohlcv(pd.concat(frames, ignore_index=True))
        return df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] < pd.Timestamp(end))]
