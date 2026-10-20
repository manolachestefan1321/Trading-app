"""Dukascopy adapter (free, no API key, long history).

Two paths, tried in order:
1. The `dukascopy-python` package, which serves 1H BID candles from Dukascopy's chart API.
2. Raw daily 1-minute BID candle files from datafeed.dukascopy.com (LZMA-compressed .bi5),
   resampled to 1H here. Used if the package is missing or its endpoint fails.

Quirks handled here:
- Dukascopy reports BID prices; we keep BID consistently (spread is modeled separately).
- Raw .bi5 month folders are 0-indexed (January = 00); prices are integers scaled by
  `dukascopy_point` (1e5 for most pairs, 1e3 for JPY pairs and gold).
- Dukascopy emits flat zero-volume candles while the market is closed; those are dropped.
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from fxpa.config import Instrument
from fxpa.data.common import (
    DataSourceError,
    chunk_ranges,
    empty_frame,
    log,
    market_closed_mask,
    normalize_ohlcv,
    retry_call,
)
from fxpa.data.sources.base import DataSource

DATAFEED_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
_RECORD = struct.Struct(">5if")  # seconds-from-midnight, open, close, low, high (scaled ints), volume


def parse_bi5_minute_candles(raw: bytes, day: datetime, point: float) -> pd.DataFrame:
    """Decode one day of Dukascopy 1-minute candles (.bi5) into a DataFrame."""
    if not raw:
        return empty_frame()
    data = lzma.decompress(raw)
    n = len(data) // _RECORD.size
    if n == 0:
        return empty_frame()
    arr = np.frombuffer(data[: n * _RECORD.size], dtype=np.dtype(
        [("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]
    ))
    base = pd.Timestamp(day.date(), tz="UTC")
    return pd.DataFrame({
        "timestamp": base + pd.to_timedelta(arr["t"].astype("int64"), unit="s"),
        "open": arr["o"] / point,
        "high": arr["h"] / point,
        "low": arr["l"] / point,
        "close": arr["c"] / point,
        "volume": arr["v"].astype("float64"),
    })


def resample_minutes_to_hours(m1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute candles into 1H candles (no filling of empty hours)."""
    if m1.empty:
        return empty_frame()
    g = m1.set_index("timestamp").resample("1h", label="left", closed="left")
    h1 = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna(subset=["open"])
    return h1.reset_index()


def drop_closed_flat_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the flat, zero-volume placeholder candles Dukascopy emits on weekends."""
    if df.empty:
        return df
    flat = (df["volume"] <= 0) & (df["high"] == df["low"])
    closed = market_closed_mask(df["timestamp"])
    return df[~(flat & closed)].reset_index(drop=True)


class DukascopySource(DataSource):
    name = "dukascopy"

    def _fetch_package(self, inst: Instrument, start: datetime, end: datetime, progress: bool) -> pd.DataFrame:
        import logging

        import dukascopy_python as dk  # imported lazily: optional dependency

        # The package resets its logger to INFO on every call and logs each request; keep it out
        # of our output (its errors still surface as exceptions handled below).
        logging.getLogger("DUKASCRIPT").propagate = False

        frames = []
        chunks = chunk_ranges(start, end, 90)
        for a, b in tqdm(chunks, desc=f"dukascopy {inst.symbol}", unit="chunk", disable=not progress):
            raw = retry_call(
                dk.fetch, inst.dukascopy, dk.INTERVAL_HOUR_1, dk.OFFER_SIDE_BID, a, b,
                max_retries=1, retries=self.retries, base_delay=self.backoff,
                label=f"dukascopy {inst.symbol} {a:%Y-%m-%d}",
            )
            if raw is not None and len(raw):
                frames.append(raw.reset_index())
        if not frames:
            return empty_frame()
        return normalize_ohlcv(pd.concat(frames, ignore_index=True))

    def _fetch_raw(self, inst: Instrument, start: datetime, end: datetime, progress: bool) -> pd.DataFrame:
        sym = inst.dukascopy.replace("/", "")
        days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end), freq="D")
        session = requests.Session()
        frames = []
        for day in tqdm(days, desc=f"dukascopy-raw {inst.symbol}", unit="day", disable=not progress):
            if day.weekday() == 5:  # Saturday: never any data
                continue
            url = DATAFEED_URL.format(sym=sym, y=day.year, m=day.month - 1, d=day.day)

            def _get() -> bytes:
                r = session.get(url, timeout=30)
                if r.status_code == 404:
                    return b""
                r.raise_for_status()
                return r.content

            raw = retry_call(_get, retries=self.retries, base_delay=self.backoff, label=f"GET {url}",
                             exceptions=(requests.RequestException,))
            m1 = parse_bi5_minute_candles(raw, day.to_pydatetime(), inst.dukascopy_point)
            if not m1.empty:
                frames.append(m1)
        if not frames:
            return empty_frame()
        return normalize_ohlcv(resample_minutes_to_hours(pd.concat(frames, ignore_index=True)))

    def fetch(self, inst: Instrument, start: datetime, end: datetime, progress: bool = True) -> pd.DataFrame:
        errors = []
        for label, fn in (("package", self._fetch_package), ("raw bi5", self._fetch_raw)):
            try:
                df = fn(inst, start, end, progress)
            except ImportError as exc:
                errors.append(f"{label}: package not installed ({exc})")
                continue
            except Exception as exc:  # network / parsing failures -> try next path
                errors.append(f"{label}: {exc.__class__.__name__}: {str(exc)[:160]}")
                log.warning("dukascopy %s path failed for %s: %s", label, inst.symbol, str(exc)[:160])
                continue
            df = drop_closed_flat_candles(df)
            df = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] < pd.Timestamp(end))]
            return df.reset_index(drop=True)
        raise DataSourceError("; ".join(errors))

