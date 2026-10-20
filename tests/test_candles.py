"""Every candlestick pattern on hand-made OHLC fixtures (ATR fixed at 10 pips)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import bars
from fxpa.analysis.candles import patterns_at
from fxpa.config import load_config

P = load_config()["candles"]
ATR = 0.0010


def names(rows):
    df = bars(rows)
    return [h.name for h in patterns_at(df, np.full(len(df), ATR), len(df) - 1, P)], df


def hit(rows, name):
    df = bars(rows)
    for h in patterns_at(df, np.full(len(df), ATR), len(df) - 1, P):
        if h.name == name:
            return h
    return None


FILLER = (1.1000, 1.1005, 1.0995, 1.1001)


def test_hammer_bullish_pin():
    h = hit([FILLER, (1.1000, 1.1003, 1.0985, 1.1002)], "bullish pin bar")
    assert h and h.direction == 1 and h.low == pytest.approx(1.0985)


def test_shooting_star_bearish_pin():
    h = hit([FILLER, (1.1000, 1.1015, 1.0998, 1.0999)], "bearish pin bar")
    assert h and h.direction == -1


def test_weak_pin_rejected_by_atr():
    # Same shape but only 3 pips of range (< 0.5 ATR) -> ignored.
    n, _ = names([FILLER, (1.1000, 1.10007, 1.0997, 1.10004)])
    assert "bullish pin bar" not in n


def test_bullish_engulfing():
    h = hit([FILLER, (1.1005, 1.1006, 1.0998, 1.0999), (1.0998, 1.1012, 1.0997, 1.1010)], "bullish engulfing")
    assert h and h.direction == 1 and h.n_bars == 2


def test_bearish_engulfing():
    h = hit([FILLER, (1.0999, 1.1006, 1.0998, 1.1005), (1.1006, 1.1007, 1.0990, 1.0992)], "bearish engulfing")
    assert h and h.direction == -1


def test_engulfing_requires_opposite_colors():
    n, _ = names([FILLER, (1.0999, 1.1006, 1.0998, 1.1005), (1.1000, 1.1015, 1.0997, 1.1012)])
    assert "bullish engulfing" not in n and "bearish engulfing" not in n


def test_inside_bar():
    h = hit([FILLER, (1.1000, 1.1020, 1.0990, 1.1010), (1.1005, 1.1012, 1.0998, 1.1008)], "inside bar")
    assert h and h.direction == 0


def test_doji():
    h = hit([FILLER, (1.1000, 1.1008, 1.0992, 1.10005)], "doji")
    assert h and h.direction == 0


def test_morning_star():
    rows = [FILLER, (1.1020, 1.1022, 1.0998, 1.1000), (1.0999, 1.1002, 1.0994, 1.0998),
            (1.0999, 1.1018, 1.0997, 1.1015)]
    h = hit(rows, "morning star")
    assert h and h.direction == 1 and h.n_bars == 3 and h.low == pytest.approx(1.0994)


def test_evening_star():
    rows = [FILLER, (1.1000, 1.1022, 1.0998, 1.1020), (1.1021, 1.1026, 1.1018, 1.1022),
            (1.1021, 1.1023, 1.1002, 1.1005)]
    h = hit(rows, "evening star")
    assert h and h.direction == -1


def test_tweezer_bottom():
    rows = [FILLER, (1.1010, 1.1012, 1.0990, 1.0995), (1.0995, 1.1008, 1.09905, 1.1006)]
    h = hit(rows, "tweezer bottom")
    assert h and h.direction == 1


def test_tweezer_top():
    rows = [FILLER, (1.0990, 1.1010, 1.0988, 1.1006), (1.1006, 1.10095, 1.0992, 1.0994)]
    h = hit(rows, "tweezer top")
    assert h and h.direction == -1


def test_plain_trend_candle_has_no_pattern():
    n, _ = names([FILLER, (1.1000, 1.1006, 1.0999, 1.1005)])
    assert n == []
