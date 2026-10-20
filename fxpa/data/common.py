"""Shared helpers for the data layer: schema normalization, retries, market-hours math."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

log = logging.getLogger("fxpa.data")

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
NY_TZ = "America/New_York"
T = TypeVar("T")


class DataSourceError(Exception):
    """A source could not deliver data (network, auth, unknown symbol, empty answer)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def floor_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS})
    df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any adapter output to the canonical schema.

    timestamp (tz-aware UTC, candle open time), open, high, low, close, volume (float64).
    Timestamps are floored to the hour so every source lines up on the same grid.
    """
    if df is None or len(df) == 0:
        return empty_frame()
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "timestamp" not in out.columns:
        out = out.reset_index()
        out.columns = [str(c).lower() for c in out.columns]
        for cand in ("datetime", "date", "time", "index"):
            if cand in out.columns:
                out = out.rename(columns={cand: "timestamp"})
                break
    if "volume" not in out.columns:
        out["volume"] = 0.0
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = ts.dt.floor("h").astype("datetime64[ns, UTC]")
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out[COLUMNS].sort_values("timestamp").reset_index(drop=True)


def retry_call(
    fn: Callable[..., T],
    *args,
    retries: int = 4,
    base_delay: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    label: str = "request",
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Call fn with exponential backoff (base, 2*base, 4*base ...) on the given exceptions."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:  # noqa: PERF203 - retry loop
            attempt += 1
            if attempt > retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.warning("%s failed (%s); retry %d/%d in %.0fs", label, exc, attempt, retries, delay)
            sleep(delay)


def market_closed(ts: pd.Timestamp | datetime) -> bool:
    """True when the spot FX market is closed: Friday 17:00 to Sunday 17:00 New York time."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    ny = t.tz_convert(NY_TZ)
    wd, h = ny.weekday(), ny.hour
    if wd == 5:  # Saturday
        return True
    if wd == 4 and h >= 17:  # Friday after the close
        return True
    if wd == 6 and h < 17:  # Sunday before the open
        return True
    return False


def market_closed_mask(ts: pd.Series) -> np.ndarray:
    """Vectorized market_closed over a UTC timestamp series."""
    ny = pd.DatetimeIndex(ts).tz_convert(NY_TZ)
    wd, h = ny.weekday, ny.hour
    return np.asarray((wd == 5) | ((wd == 4) & (h >= 17)) | ((wd == 6) & (h < 17)))


def open_hours_between(start: datetime, end: datetime) -> int:
    """Number of hourly candles in [start, end) during which the market is open."""
    if end <= start:
        return 0
    rng = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end), freq="h", inclusive="left")
    if len(rng) == 0:
        return 0
    return int((~market_closed_mask(pd.Series(rng))).sum())


def chunk_ranges(start: datetime, end: datetime, days: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into consecutive chunks of at most `days` days."""
    out = []
    cur = start
    step = timedelta(days=days)
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out
