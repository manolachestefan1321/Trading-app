"""Higher-timeframe resampling aligned to the forex day (17:00 New York close).

A "trading clock" is New York wall time shifted by +7h, so 17:00 NY becomes 00:00.
Daily bars therefore run 17:00 NY -> 17:00 NY (DST-aware), 4H bars start at
17/21/01/05/09/13 NY, and weeks start Sunday 17:00 NY.
"""

from __future__ import annotations

import pandas as pd

from fxpa.data.common import NY_TZ

_SHIFT = pd.Timedelta(hours=7)
# Time after the period start at which the bar is finished. A week ends at the Friday close
# (5 trading days), not when the next week opens.
_PERIOD = {"4h": pd.Timedelta(hours=4), "D": pd.Timedelta(days=1), "W": pd.Timedelta(days=5)}


def trading_clock(ts: pd.Series) -> pd.Series:
    """UTC timestamps -> naive NY wall time + 7h (17:00 NY maps to midnight)."""
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = pd.to_datetime(ts, utc=True)
    return ts.dt.tz_convert(NY_TZ).dt.tz_localize(None) + _SHIFT


def period_key(ts: pd.Series, tf: str) -> pd.Series:
    clock = trading_clock(ts)
    if tf == "4h":
        return clock.dt.floor("4h")
    if tf == "D":
        return clock.dt.floor("D")
    if tf == "W":
        day = clock.dt.floor("D")
        return day - pd.to_timedelta(day.dt.weekday, unit="D")  # Monday 00:00 clock = Sunday 17:00 NY
    raise ValueError(f"unsupported timeframe {tf}")


def resample(df: pd.DataFrame, tf: str, complete_only: bool = True) -> pd.DataFrame:
    """Resample 1H candles to 4h / D / W bars.

    Output columns: timestamp (UTC of the first 1H candle), key (trading-clock period start),
    open, high, low, close, volume, n_bars, complete.
    A bar is complete when the last 1H candle's close time reaches the period end, so the
    still-forming bar is never used for decisions (no lookahead).
    """
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "key", "open", "high", "low", "close", "volume",
                                     "n_bars", "complete"])
    key = period_key(df["timestamp"], tf)
    g = df.assign(key=key.values).groupby("key", sort=True)
    out = pd.DataFrame({
        "timestamp": g["timestamp"].first(),
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "n_bars": g["open"].size(),
    }).reset_index()
    last_close_clock = trading_clock(df["timestamp"].iloc[[-1]] + pd.Timedelta(hours=1)).iloc[0]
    out["complete"] = (out["key"] + _PERIOD[tf]) <= last_close_clock
    if complete_only:
        out = out[out["complete"]].reset_index(drop=True)
    return out
