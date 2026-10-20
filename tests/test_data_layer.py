"""Data layer: cache + incremental updates (network mocked), fallback chain, validation, resampling."""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from fxpa.data.common import DataSourceError, market_closed, normalize_ohlcv, retry_call
from fxpa.data.manager import DataManager
from fxpa.data.resample import resample
from fxpa.data.sources.base import DataSource
from fxpa.data.sources.dukascopy import parse_bi5_minute_candles, resample_minutes_to_hours
from fxpa.data.validate import validate
from fxpa.data.calendar import parse_forexfactory


class FakeSource(DataSource):
    """Serves slices of a fixed 'market' frame and records every requested range."""

    def __init__(self, cfg, market: pd.DataFrame, name: str = "fake", fail: bool = False):
        super().__init__(cfg)
        self.name = name
        self.market = market
        self.fail = fail
        self.calls: list[tuple[datetime, datetime]] = []

    def fetch(self, inst, start, end, progress=True):
        self.calls.append((start, end))
        if self.fail:
            raise DataSourceError("simulated outage")
        m = self.market
        return m[(m["timestamp"] >= pd.Timestamp(start)) & (m["timestamp"] < pd.Timestamp(end))].copy()


@pytest.fixture
def market():
    return make_candles(3000, start="2024-01-01 22:00")


def _clock(ts):
    return lambda: ts


def test_first_run_downloads_full_range_then_only_missing(cfg, instruments, market):
    t1 = market["timestamp"].iloc[2000].to_pydatetime()
    src = FakeSource(cfg, market)
    dm = DataManager(cfg, instruments, source="fake", sources={"fake": src}, progress=False,
                     now_fn=_clock(t1))
    df = dm.get("EURUSD", years=0.25)
    assert len(src.calls) == 1
    assert df["timestamp"].iloc[-1] < pd.Timestamp(t1)  # forming candle excluded
    assert dm.cache.path("EURUSD").exists()

    # Second run 50 hours later: only the tail since the last cached candle is requested.
    t2 = market["timestamp"].iloc[2050].to_pydatetime()
    src2 = FakeSource(cfg, market)
    dm2 = DataManager(cfg, instruments, source="fake", sources={"fake": src2}, progress=False,
                      now_fn=_clock(t2))
    df2 = dm2.get("EURUSD", years=0.25)
    assert len(src2.calls) == 1
    req_start, req_end = src2.calls[0]
    assert pd.Timestamp(req_start) == df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    assert pd.Timestamp(req_end) == pd.Timestamp(t2)
    assert df2["timestamp"].is_monotonic_increasing and df2["timestamp"].is_unique
    assert df2["timestamp"].iloc[-1] == market["timestamp"].iloc[2049]


def test_no_download_when_cache_is_current(cfg, instruments, market):
    t = market["timestamp"].iloc[1500].to_pydatetime()
    src = FakeSource(cfg, market)
    DataManager(cfg, instruments, source="fake", sources={"fake": src}, progress=False,
                now_fn=_clock(t)).get("EURUSD", years=0.2)
    src2 = FakeSource(cfg, market)
    DataManager(cfg, instruments, source="fake", sources={"fake": src2}, progress=False,
                now_fn=_clock(t)).get("EURUSD", years=0.2)
    assert src2.calls == []


def test_refresh_ignores_cache(cfg, instruments, market):
    t = market["timestamp"].iloc[1500].to_pydatetime()
    for _ in range(2):
        src = FakeSource(cfg, market)
        DataManager(cfg, instruments, source="fake", sources={"fake": src}, progress=False,
                    refresh=True, now_fn=_clock(t)).get("EURUSD", years=0.2)
        assert len(src.calls) == 1
        expected_start = pd.Timestamp(t) - pd.Timedelta(days=round(365.25 * 0.2))
        assert pd.Timestamp(src.calls[0][0]) == expected_start


def test_fallback_chain_uses_next_source(cfg, instruments, market):
    t = market["timestamp"].iloc[1200].to_pydatetime()
    bad = FakeSource(cfg, market, name="bad", fail=True)
    good = FakeSource(cfg, market, name="good")
    cfg["data"]["sources"] = ["bad", "good"]
    dm = DataManager(cfg, instruments, sources={"bad": bad, "good": good}, progress=False,
                     now_fn=_clock(t))
    df = dm.get("EURUSD", years=0.1)
    assert len(df) > 0
    assert dm.last_result["EURUSD"].sources_used == ["good"]
    assert any("bad" in e for e in dm.last_result["EURUSD"].errors)


def test_all_sources_fail_without_cache_raises(cfg, instruments, market):
    t = market["timestamp"].iloc[1200].to_pydatetime()
    bad = FakeSource(cfg, market, name="bad", fail=True)
    dm = DataManager(cfg, instruments, source="bad", sources={"bad": bad}, progress=False,
                     now_fn=_clock(t))
    with pytest.raises(DataSourceError):
        dm.get("EURUSD", years=0.1)


def test_all_sources_fail_with_cache_uses_cache(cfg, instruments, market):
    t = market["timestamp"].iloc[1200].to_pydatetime()
    DataManager(cfg, instruments, source="f", sources={"f": FakeSource(cfg, market)}, progress=False,
                now_fn=_clock(t)).get("EURUSD", years=0.1)
    t2 = market["timestamp"].iloc[1300].to_pydatetime()
    dm = DataManager(cfg, instruments, source="bad", sources={"bad": FakeSource(cfg, market, fail=True)},
                     progress=False, now_fn=_clock(t2))
    df = dm.get("EURUSD", years=0.1)
    assert dm.last_result["EURUSD"].stale
    assert df["timestamp"].iloc[-1] == market["timestamp"].iloc[1199]


def test_weekend_only_gap_triggers_no_request(cfg, instruments, market):
    # Last cached candle is Friday 21:00 UTC (16:00 NY, winter); "now" is Saturday noon -> nothing to fetch.
    fri = market[(market["timestamp"].dt.weekday == 4) & (market["timestamp"].dt.hour == 21)].iloc[3]
    t = (fri["timestamp"] + pd.Timedelta(hours=1)).to_pydatetime()
    DataManager(cfg, instruments, source="f", sources={"f": FakeSource(cfg, market)}, progress=False,
                now_fn=_clock(t)).get("EURUSD", years=0.1)
    sat = (fri["timestamp"] + pd.Timedelta(hours=15)).to_pydatetime()
    src = FakeSource(cfg, market)
    DataManager(cfg, instruments, source="f", sources={"f": src}, progress=False,
                now_fn=_clock(sat)).get("EURUSD", years=0.1)
    assert src.calls == []


def test_validate_dedupes_sorts_and_drops_bad_ohlc():
    df = make_candles(300)
    dup = df.iloc[[10, 20]]
    bad = df.iloc[[30]].copy()
    bad["high"] = bad["low"] - 0.001  # high below low
    shuffled = pd.concat([df.drop(index=30), dup, bad]).sample(frac=1, random_state=0)
    out, rep = validate(shuffled)
    assert rep.duplicates_dropped == 2
    assert rep.invalid_ohlc_dropped == 1
    assert out["timestamp"].is_monotonic_increasing and out["timestamp"].is_unique
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()


def test_validate_flags_spikes_and_counts_missing_without_interpolating():
    df = make_candles(400)
    df.loc[200, "high"] = df.loc[200, "high"] + 0.05  # huge spike
    removed = df.drop(index=[100, 101])
    out, rep = validate(removed)
    assert bool(out.loc[out["timestamp"] == df.loc[200, "timestamp"], "spike"].iloc[0])
    assert rep.missing_candles == 2
    assert len(out) == len(removed)  # nothing interpolated


def test_market_closed_hours():
    assert market_closed(pd.Timestamp("2024-01-06 12:00", tz="UTC"))       # Saturday
    assert market_closed(pd.Timestamp("2024-01-05 22:00", tz="UTC"))       # Fri 17:00 NY (EST)
    assert not market_closed(pd.Timestamp("2024-01-05 21:00", tz="UTC"))   # Fri 16:00 NY
    assert not market_closed(pd.Timestamp("2024-01-07 22:00", tz="UTC"))   # Sun 17:00 NY open
    assert not market_closed(pd.Timestamp("2024-07-07 21:00", tz="UTC"))   # Sun 17:00 NY (EDT)


def test_daily_candle_uses_new_york_close():
    # Winter: 17:00 NY = 22:00 UTC. A bar at 21:00 UTC belongs to day D, 22:00 UTC to day D+1.
    df = make_candles(200, start="2024-01-08 00:00")
    d = resample(df, "D", complete_only=False)
    first_bar_hours = d["timestamp"].iloc[1:].dt.hour.unique()
    assert list(first_bar_hours) == [22]
    # Summer (EDT): the day starts 21:00 UTC.
    df2 = make_candles(200, start="2024-07-08 00:00")
    d2 = resample(df2, "D", complete_only=False)
    assert list(d2["timestamp"].iloc[1:].dt.hour.unique()) == [21]
    # The last (forming) day is excluded when complete_only.
    assert len(resample(df, "D")) == len(d) - (0 if d["complete"].iloc[-1] else 1)


def test_four_hour_and_weekly_resample_alignment():
    df = make_candles(400, start="2024-01-08 00:00")
    h4 = resample(df, "4h", complete_only=False)
    assert set(h4["timestamp"].iloc[1:].dt.hour.unique()) <= {22, 2, 6, 10, 14, 18}
    w = resample(df, "W", complete_only=False)
    # weekly bars start Sunday 22:00 UTC in winter
    starts = w["timestamp"].iloc[1:]
    assert ((starts.dt.weekday == 6) & (starts.dt.hour == 22)).all()


def test_dukascopy_bi5_parser_and_minute_resample():
    recs = []
    for minute in range(120):
        p = 108000 + minute
        recs.append(struct.pack(">5if", minute * 60, p, p + 1, p - 2, p + 3, 1.5))
    raw = lzma.compress(b"".join(recs), format=lzma.FORMAT_ALONE)
    m1 = parse_bi5_minute_candles(raw, datetime(2024, 1, 3, tzinfo=timezone.utc), 100000)
    assert len(m1) == 120
    assert m1["open"].iloc[0] == pytest.approx(1.08)
    h1 = resample_minutes_to_hours(m1)
    assert len(h1) == 2
    assert h1["high"].iloc[0] == pytest.approx((108000 + 59 + 3) / 1e5)
    assert h1["low"].iloc[1] == pytest.approx((108060 - 2) / 1e5)
    assert h1["volume"].iloc[0] == pytest.approx(90.0)


def test_normalize_handles_index_and_timezones():
    idx = pd.date_range("2024-01-02 03:00", periods=3, freq="h", tz="America/New_York")
    raw = pd.DataFrame({"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2, 3]},
                       index=idx)
    raw.index.name = "Datetime"
    out = normalize_ohlcv(raw)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].iloc[0].hour == 8
    assert (out["volume"] == 0).all()


def test_retry_backoff_delays():
    delays, calls = [], {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert retry_call(flaky, retries=4, base_delay=2, sleep=delays.append) == "ok"
    assert delays == [2, 4]


def test_forexfactory_parser():
    items = [{"title": "Non-Farm Employment Change", "country": "USD", "date": "2024-01-05T08:30:00-05:00",
              "impact": "High"}, {"title": "bad", "country": "EUR", "date": "not a date"}]
    df = parse_forexfactory(items)
    assert len(df) == 1
    assert df["time"].iloc[0] == pd.Timestamp("2024-01-05 13:30", tz="UTC")
