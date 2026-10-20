"""Swing detection and market structure classification on hand-made price paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import bars
from fxpa.analysis.structure import analyze_structure, label_swings
from fxpa.analysis.swings import detect_swings, fractal_pivots
from fxpa.data.validate import add_gap_flags


def path_to_bars(points: list[float], steps: int = 5, start: str = "2024-01-02 00:00") -> pd.DataFrame:
    """Piecewise-linear close path through `points`, with small wicks."""
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        closes.extend(np.linspace(a, b, steps, endpoint=False))
    closes.append(points[-1])
    closes = np.array(closes)
    opens = np.r_[closes[0], closes[:-1]]
    rows = [(o, max(o, c) + 0.0001, min(o, c) - 0.0001, c) for o, c in zip(opens, closes)]
    return bars(rows, start=start)


def test_fractal_pivots_basic():
    high = np.array([1, 2, 3, 6, 3, 2, 1, 2, 3], dtype=float)
    low = high - 0.5
    is_h, is_l = fractal_pivots(high, low, 3)
    assert list(np.flatnonzero(is_h)) == [3]
    assert not is_l[:3].any()  # edges can't be pivots


def test_fractal_tie_goes_to_earlier_bar():
    high = np.array([1, 2, 3, 5, 5, 3, 2, 1, 0.5], dtype=float)
    is_h, _ = fractal_pivots(high, high - 1, 3)
    assert list(np.flatnonzero(is_h)) == [3]


def test_detect_swings_alternates_and_confirms_late():
    df = path_to_bars([1.10, 1.12, 1.11, 1.13, 1.12, 1.14, 1.13])
    atr = np.full(len(df), 0.002)
    sw = detect_swings(df, atr, window=3, min_swing_atr=0.5)
    kinds = "".join(sw["kind"])
    assert "HH" not in kinds and "LL" not in kinds
    assert kinds.startswith("H")
    assert (sw["confirmed_idx"] == sw["idx"] + 3).all()
    assert sw["price"].iloc[0] > 1.1195


def test_noise_filter_drops_small_swings():
    # A tiny 5-pip wiggle inside an up-leg is not a swing when ATR is 20 pips.
    df = path_to_bars([1.1000, 1.1100, 1.1095, 1.1200, 1.1100], steps=6)
    atr = np.full(len(df), 0.0020)
    loose = detect_swings(df, atr, window=2, min_swing_atr=0.0)
    strict = detect_swings(df, atr, window=2, min_swing_atr=1.0)
    assert len(strict) < len(loose)
    assert strict[strict["kind"] == "H"]["price"].max() > 1.1199


def test_swings_marked_across_weekend_gap():
    df = path_to_bars([1.10, 1.12, 1.10, 1.12, 1.10])
    df.loc[df.index >= 7, "timestamp"] += pd.Timedelta(hours=48)  # gap between bar 6 and 7
    df = add_gap_flags(df)
    sw = detect_swings(df, np.full(len(df), 0.002), window=3, min_swing_atr=0.5)
    first_high = sw.iloc[0]
    assert first_high["idx"] == 5 and bool(first_high["across_gap"])  # gap at bar 7 inside (5-3, 5+3]


def test_uptrend_classification_and_strength():
    pts = [1.1000, 1.1100, 1.1050, 1.1150, 1.1100, 1.1200, 1.1150, 1.1250, 1.1200]
    df = path_to_bars(pts)
    sw = detect_swings(df, np.full(len(df), 0.002), window=2, min_swing_atr=0.5)
    st = analyze_structure(sw, df, strength_full_count=6)
    assert st.trend == "up"
    assert st.confirming >= 4
    assert st.strength >= 60
    assert st.describe().startswith("HH/HL")


def test_downtrend_classification():
    pts = [1.1300, 1.1200, 1.1250, 1.1150, 1.1200, 1.1100, 1.1150, 1.1050, 1.1100]
    df = path_to_bars(pts)
    sw = detect_swings(df, np.full(len(df), 0.002), window=2, min_swing_atr=0.5)
    st = analyze_structure(sw, df)
    assert st.trend == "down"
    assert st.sign == -1


def test_range_classification():
    pts = [1.1000, 1.1100, 1.1000, 1.1100, 1.1000, 1.1100, 1.1000]
    df = path_to_bars(pts)
    sw = detect_swings(df, np.full(len(df), 0.002), window=2, min_swing_atr=0.5)
    assert analyze_structure(sw, df).trend == "range"


def test_bos_then_choch():
    # Uptrend breaking highs (BOS up), then a close below the last higher low (bearish CHoCH).
    pts = [1.1000, 1.1100, 1.1050, 1.1150, 1.1100, 1.1200, 1.1120, 1.1220, 1.1150, 1.1250, 1.1000]
    df = path_to_bars(pts)
    sw = detect_swings(df, np.full(len(df), 0.002), window=2, min_swing_atr=0.5)
    st = analyze_structure(sw, df)
    kinds = [(e.kind, e.direction) for e in st.events]
    assert ("BOS", 1) in kinds
    assert kinds[-1] == ("CHoCH", -1)
    choch = st.last_event("CHoCH")
    assert choch.level < 1.1200


def test_label_swings():
    sw = pd.DataFrame({"kind": ["H", "L", "H", "L", "H", "L"], "price": [2, 1, 3, 1.5, 2.5, 1.2]})
    assert label_swings(sw) == ["H", "L", "HH", "HL", "LH", "LL"]
