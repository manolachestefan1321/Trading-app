"""Zones, key levels, behavior, sessions, news, correlation guard and pip math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import bars, make_candles
from fxpa.analysis.behavior import evaluate_behavior
from fxpa.analysis.levels import key_levels, round_numbers
from fxpa.analysis.zones import Zone, build_zones
from fxpa.forex.correlation import apply_correlation_guard
from fxpa.forex.news import NewsFilter
from fxpa.forex.pips import pip_value_per_lot, position_size
from fxpa.forex.sessions import asian_range, entry_allowed, london_asian_event, session_label

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


# ---------------------------------------------------------------- zones
def _swings(rows):
    return pd.DataFrame([{"idx": i, "timestamp": ts("2024-01-02") + pd.Timedelta(hours=i), "price": p,
                          "kind": k, "confirmed_idx": i + 3, "across_gap": False} for i, p, k in rows])


def test_swings_cluster_into_scored_zone(cfg):
    df = make_candles(200, price=1.0850)
    df.loc[199, "close"] = 1.0900
    sw = _swings([(20, 1.0810, "L"), (60, 1.0814, "L"), (100, 1.0812, "L"), (140, 1.0990, "H")])
    zones = build_zones(df, sw, 0.0010, cfg["zones"], levels=[])
    sup = [z for z in zones if z.touches == 3]
    assert len(sup) == 1
    z = sup[0]
    assert z.low <= 1.0810 and z.high >= 1.0814 and z.role == "support" and not z.flipped
    lone = [z for z in zones if z.touches == 1][0]
    assert z.score > lone.score


def test_role_reversal_and_key_level_confluence(cfg):
    df = make_candles(200, price=1.0850)
    df.loc[199, "close"] = 1.0900
    sw = _swings([(20, 1.0849, "H"), (60, 1.0851, "H")])  # old resistance, price now above
    lvl = [type("L", (), {"name": "round 1.085", "price": 1.0850, "weight": 1.0})()]
    zones = build_zones(df, sw, 0.0010, cfg["zones"], levels=lvl)
    z = [z for z in zones if z.touches == 2][0]
    assert z.role == "support" and z.flipped
    assert "round 1.085" in z.key_levels


def test_unconfirmed_swings_are_ignored(cfg):
    df = make_candles(100)
    sw = _swings([(97, 1.2, "H")])  # confirmed at 100 > last index 99
    zones = build_zones(df, sw, 0.001, cfg["zones"], levels=[])
    assert zones == []


def test_round_numbers():
    lv = round_numbers(1.0823, (0.01, 0.005), 2)
    prices = sorted(l.price for l in lv)
    assert 1.08 in prices and 1.09 in prices and 1.085 in prices
    gold = round_numbers(2023.0, (100.0, 50.0), 1)
    assert {2000.0, 2050.0} <= {l.price for l in gold}


def test_key_levels_previous_day_from_ny_close(cfg):
    df = make_candles(60, start="2024-01-08 00:00")  # ends 2024-01-10 11:00 UTC
    lv = {l.name: l.price for l in key_levels(df, (0.01,), cfg["sessions"])}
    for k in ("PDH", "PDL", "PDC", "weekly open", "Asian high", "Asian low"):
        assert k in lv
    # Last complete NY day: 2024-01-08 22:00 UTC (17:00 NY) .. 2024-01-09 21:00 UTC.
    day = df[(df["timestamp"] >= ts("2024-01-08 22:00")) & (df["timestamp"] < ts("2024-01-09 22:00"))]
    assert lv["PDH"] == pytest.approx(day["high"].max())
    assert lv["PDL"] == pytest.approx(day["low"].min())
    assert lv["PDC"] == pytest.approx(day["close"].iloc[-1])


# ---------------------------------------------------------------- behavior
def _zone(lo, hi, role="support", lows=3, highs=0):
    return Zone(low=lo, high=hi, touches=lows + highs, n_highs=highs, n_lows=lows, last_touch_idx=0,
                reaction_atr=2, volume_ratio=1, score=70, role=role)


def test_bounce_from_support(cfg):
    df = bars([(1.1010, 1.1012, 1.1004, 1.1006), (1.1006, 1.1008, 1.0999, 1.1001),
               (1.1001, 1.1012, 1.0997, 1.1010)])
    hits = evaluate_behavior(df, [_zone(1.0995, 1.1000)], np.full(3, 0.001), 2, 1, cfg["behavior"], 0.35)
    assert hits and hits[0].kind == "bounce" and hits[0].direction == 1


def test_failed_bounce_decisive_close_through_support(cfg):
    df = bars([(1.1010, 1.1012, 1.1004, 1.1006), (1.1006, 1.1008, 1.1001, 1.1003),
               (1.1003, 1.1004, 1.0980, 1.0982)])
    hits = evaluate_behavior(df, [_zone(1.0995, 1.1000)], np.full(3, 0.001), 2, 1, cfg["behavior"], 0.35)
    assert hits[0].kind == "failed bounce" and hits[0].direction == -1


def test_fakeout_below_support(cfg):
    df = bars([(1.1010, 1.1012, 1.1004, 1.1006), (1.1006, 1.1008, 1.1001, 1.1003),
               (1.1003, 1.1006, 1.0985, 1.1004)])
    hits = evaluate_behavior(df, [_zone(1.0995, 1.1000)], np.full(3, 0.001), 2, 1, cfg["behavior"], 0.35)
    assert hits[0].kind == "fakeout" and hits[0].direction == 1


def test_pullback_retest_of_broken_resistance(cfg):
    rows = [(1.0990, 1.0995, 1.0985, 1.0992), (1.0992, 1.1020, 1.0990, 1.1018),   # breakout
            (1.1018, 1.1025, 1.1010, 1.1015), (1.1015, 1.1016, 1.1002, 1.1012)]  # retest + hold
    df = bars(rows)
    z = _zone(1.0995, 1.1000, role="support", lows=0, highs=3)
    hits = evaluate_behavior(df, [z], np.full(4, 0.001), 3, 1, cfg["behavior"], 0.35)
    assert hits[0].kind == "pullback" and hits[0].value == pytest.approx(1.0)


# ---------------------------------------------------------------- sessions
def test_entry_windows(cfg):
    s = cfg["sessions"]
    assert entry_allowed(ts("2024-01-03 09:00"), ("london", "overlap"), s)[0]
    ok, why = entry_allowed(ts("2024-01-03 07:00"), ("london", "overlap"), s)
    assert not ok and "first 15 min" in why
    assert not entry_allowed(ts("2024-01-03 03:00"), ("london", "overlap"), s)[0]
    assert entry_allowed(ts("2024-01-03 03:00"), ("asia", "london"), s)[0]
    # Last hour before 17:00 NY rollover (21:00-22:00 UTC in winter) is blocked.
    ok, why = entry_allowed(ts("2024-01-03 21:00"), ("newyork", "asia"), s)
    assert not ok and "rollover" in why
    assert not entry_allowed(ts("2024-01-06 10:00"), ("london",), s)[0]  # Saturday


def test_session_label(cfg):
    s = cfg["sessions"]
    assert session_label(ts("2024-01-03 13:00"), s) == "overlap"
    assert session_label(ts("2024-01-03 08:00"), s) == "london"
    assert session_label(ts("2024-01-03 18:00"), s) == "newyork"
    assert session_label(ts("2024-01-03 02:00"), s) == "asia"


def test_asian_range_and_london_sweep(cfg):
    rows = [(1.1000, 1.1010, 1.0995, 1.1005)] * 7  # 00:00-06:00 Asia: high 1.1010, low 1.0995
    rows += [(1.1005, 1.1020, 1.1004, 1.1006)]     # 07:00 sweeps the Asian high, closes back inside
    df = bars(rows, start="2024-01-03 00:00")
    rng = asian_range(df, 7, cfg["sessions"])
    assert rng.high == pytest.approx(1.1010) and rng.low == pytest.approx(1.0995)
    ev = london_asian_event(df, 7, rng, 0.001, cfg["sessions"], (7, 10))
    assert ev.kind == "sweep" and ev.direction == -1


# ---------------------------------------------------------------- news
def test_news_blackout_window(cfg, instruments):
    ev = pd.DataFrame([{"time": ts("2024-01-05 13:30"), "currency": "USD", "impact": "High", "title": "NFP"},
                       {"time": ts("2024-01-05 09:00"), "currency": "JPY", "impact": "Low", "title": "x"}])
    nf = NewsFilter(ev, cfg)
    eur = instruments["EURUSD"]
    assert nf.blackout(eur, ts("2024-01-05 13:00"))[0]       # 30 min before
    assert nf.blackout(eur, ts("2024-01-05 14:30"))[0]       # 60 min after
    assert not nf.blackout(eur, ts("2024-01-05 14:31"))[0]
    assert not nf.blackout(eur, ts("2024-01-05 12:59"))[0]
    assert not nf.blackout(instruments["EURJPY"], ts("2024-01-05 13:30"))[0]  # no USD leg
    assert nf.blackout(instruments["XAUUSD"], ts("2024-01-05 13:30"))[0]


# ---------------------------------------------------------------- correlation
def test_correlation_guard(instruments):
    kept, warns = apply_correlation_guard(
        [("EURUSD", 1, 75), ("USDCHF", -1, 65), ("GBPUSD", 1, 80), ("USDJPY", 1, 62)], instruments)
    # GBPUSD BUY (short USD) is strongest; EURUSD BUY and USDCHF SELL are also short USD -> dropped.
    # USDJPY BUY is long USD, which offsets rather than duplicates -> kept.
    assert kept == ["GBPUSD", "USDJPY"]
    assert any("EURUSD" in w for w in warns) and any("USDCHF" in w for w in warns)


def test_correlation_guard_allows_offsetting(instruments):
    kept, warns = apply_correlation_guard([("EURUSD", 1, 70), ("USDJPY", 1, 65)], instruments)
    assert kept == ["EURUSD", "USDJPY"] and not warns  # short USD then long USD: no duplication


# ---------------------------------------------------------------- pips
def test_pip_values_and_position_size(instruments):
    assert pip_value_per_lot(instruments["EURUSD"], 1.08) == pytest.approx(10.0)
    assert pip_value_per_lot(instruments["USDJPY"], 150.0) == pytest.approx(1000 / 150)
    assert pip_value_per_lot(instruments["XAUUSD"], 2000.0) == pytest.approx(10.0)
    assert pip_value_per_lot(instruments["EURJPY"], 160.0, {"USDJPY": 150.0}) == pytest.approx(1000 / 150)
    assert position_size(10000, 1.0, 20, 10.0) == pytest.approx(0.5)
    assert instruments["USDJPY"].to_pips(0.25) == pytest.approx(25)
