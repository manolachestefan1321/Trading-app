"""Trading sessions, entry windows, rollover, Asian range and London sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxpa.data.common import NY_TZ, market_closed

SESSION_ORDER = ("asia", "london", "newyork", "overlap")


def _minutes(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def in_session(ts: pd.Timestamp, sess_cfg: dict, name: str) -> bool:
    start, end = sess_cfg[name]
    m = _minutes(ts)
    return start * 60 <= m < end * 60


def session_label(ts: pd.Timestamp, sess_cfg: dict) -> str:
    """Single label for reporting: overlap > london > newyork > asia > off."""
    for name in ("overlap", "london", "newyork", "asia"):
        if in_session(ts, sess_cfg, name):
            return name
    return "off"


def rollover_time(ts: pd.Timestamp, sess_cfg: dict) -> pd.Timestamp:
    """The daily rollover (17:00 New York) on the NY calendar day of `ts`, in UTC."""
    ny = ts.tz_convert(NY_TZ)
    roll = ny.normalize() + pd.Timedelta(hours=int(sess_cfg.get("rollover_ny_hour", 17)))
    return roll.tz_convert("UTC")


def entry_allowed(entry_ts: pd.Timestamp, allowed: tuple[str, ...], sess_cfg: dict) -> tuple[bool, str]:
    """May a new trade be opened at `entry_ts` (the close of the signal candle)?

    Requires: market open, inside one of the pair's liquid sessions but not in its first
    N minutes, and not in the last hour before the 17:00 NY rollover.
    """
    if market_closed(entry_ts):
        return False, "market closed"
    roll = rollover_time(entry_ts, sess_cfg)
    before = pd.Timedelta(minutes=int(sess_cfg.get("avoid_before_rollover_minutes", 60)))
    if roll - before <= entry_ts < roll:
        return False, "last hour before daily rollover"
    avoid = int(sess_cfg.get("avoid_first_minutes", 15))
    m = _minutes(entry_ts)
    for name in allowed:
        start, end = sess_cfg[name]
        if start * 60 + avoid <= m < end * 60:
            return True, name
    opening = [n for n in allowed if sess_cfg[n][0] * 60 <= m < sess_cfg[n][0] * 60 + avoid]
    if opening:
        return False, f"first {avoid} min of {opening[0]} session"
    return False, f"outside liquid sessions ({'/'.join(allowed)})"


@dataclass
class AsianRange:
    high: float
    low: float
    date: pd.Timestamp


def asian_range(df: pd.DataFrame, t: int, sess_cfg: dict) -> AsianRange | None:
    """Completed Asian range for the UTC day of bar t (previous day's if Asia is still open)."""
    ts = df["timestamp"]
    a_start, a_end = sess_cfg["asia"]
    close_t = ts.iat[t] + pd.Timedelta(hours=1)
    day = close_t.normalize()
    if close_t < day + pd.Timedelta(hours=a_end):
        day = day - pd.Timedelta(days=1)
    lo_ts, hi_ts = day + pd.Timedelta(hours=a_start), day + pd.Timedelta(hours=a_end)
    left = ts.searchsorted(lo_ts, side="left")
    right = min(ts.searchsorted(hi_ts, side="left"), t + 1)
    if right - left < max(2, (a_end - a_start) // 2):
        return None
    return AsianRange(float(df["high"].iloc[left:right].max()), float(df["low"].iloc[left:right].min()), day)


@dataclass
class LondonEvent:
    kind: str        # "sweep" (fakeout / stop hunt) or "breakout"
    direction: int   # implied trade direction
    text: str


def london_asian_event(df: pd.DataFrame, t: int, rng: AsianRange | None, atr_t: float,
                       sess_cfg: dict, window: tuple[int, int]) -> LondonEvent | None:
    """Flag London breakouts / fakeouts of the Asian range during the London open window."""
    if rng is None:
        return None
    ts = df["timestamp"]
    hour = ts.iat[t].hour
    if not (window[0] <= hour < window[1]) or ts.iat[t].normalize() != rng.date:
        return None
    start = ts.searchsorted(rng.date + pd.Timedelta(hours=window[0]), side="left")
    hi = df["high"].iloc[start:t + 1].max()
    lo = df["low"].iloc[start:t + 1].min()
    c = df["close"].iat[t]
    if hi > rng.high and c < rng.high:
        return LondonEvent("sweep", -1, f"London swept Asian high {rng.high:.5g} and closed back inside")
    if lo < rng.low and c > rng.low:
        return LondonEvent("sweep", 1, f"London swept Asian low {rng.low:.5g} and closed back inside")
    margin = 0.2 * atr_t if np.isfinite(atr_t) else 0.0
    if c > rng.high + margin:
        return LondonEvent("breakout", 1, f"London breakout above Asian high {rng.high:.5g}")
    if c < rng.low - margin:
        return LondonEvent("breakout", -1, f"London breakout below Asian low {rng.low:.5g}")
    return None
