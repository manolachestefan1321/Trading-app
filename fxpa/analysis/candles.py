"""Candlestick patterns from body/wick geometry, normalized by ATR.

Each detector looks at the pattern *ending* at bar t and returns a PatternHit or None.
Direction: +1 bullish, -1 bearish, 0 neutral (doji / inside bar: direction comes from
location, decided by the scoring engine). Patterns are only meaningful at S/R zones or
swing points; that validity check also lives in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PatternHit:
    name: str
    direction: int
    strength: float        # 0-1 base reliability x size quality
    idx: int               # last bar of the pattern
    n_bars: int            # bars in the pattern
    low: float             # extreme low across the pattern bars
    high: float            # extreme high across the pattern bars


# Base reliability of each pattern (classical ranking; configurable in code only).
BASE_STRENGTH = {
    "bullish pin bar": 1.0, "bearish pin bar": 1.0,
    "bullish engulfing": 1.0, "bearish engulfing": 1.0,
    "morning star": 0.9, "evening star": 0.9,
    "tweezer bottom": 0.7, "tweezer top": 0.7,
    "inside bar": 0.4, "doji": 0.3,
}


class CandleGeometry:
    """Pre-computed body / wick arrays for fast pattern checks."""

    def __init__(self, df: pd.DataFrame, atr_values: np.ndarray):
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.atr = np.asarray(atr_values, dtype=float)
        self.rng = self.h - self.l
        self.body = np.abs(self.c - self.o)
        self.upper = self.h - np.maximum(self.o, self.c)
        self.lower = np.minimum(self.o, self.c) - self.l
        self.bull = self.c > self.o
        self.bear = self.c < self.o


def _quality(g: CandleGeometry, t: int) -> float:
    """Size quality: a candle of 1.5 ATR scores 1.0; 0.5 ATR scores ~0.33."""
    a = g.atr[t]
    if not np.isfinite(a) or a <= 0:
        return 0.5
    return float(np.clip(g.rng[t] / (1.5 * a), 0.0, 1.0))


def _hit(g: CandleGeometry, name: str, direction: int, t: int, n: int, q: float) -> PatternHit:
    s = slice(t - n + 1, t + 1)
    return PatternHit(name, direction, round(BASE_STRENGTH[name] * q, 3), t, n,
                      float(g.l[s].min()), float(g.h[s].max()))


def pin_bar(g: CandleGeometry, t: int, p: dict) -> PatternHit | None:
    """Hammer / bullish pin (long lower wick) or shooting star / bearish pin (long upper wick)."""
    if g.rng[t] < p["min_range_atr"] * g.atr[t] or g.rng[t] <= 0:
        return None
    body = max(g.body[t], 1e-12)
    ratio_body, ratio_rng, opp = p["pin_wick_body_ratio"], p["pin_wick_range_ratio"], p["pin_opposite_wick_ratio"]
    if g.lower[t] >= ratio_body * body and g.lower[t] >= ratio_rng * g.rng[t] and g.upper[t] <= opp * g.rng[t]:
        return _hit(g, "bullish pin bar", 1, t, 1, _quality(g, t))
    if g.upper[t] >= ratio_body * body and g.upper[t] >= ratio_rng * g.rng[t] and g.lower[t] <= opp * g.rng[t]:
        return _hit(g, "bearish pin bar", -1, t, 1, _quality(g, t))
    return None


def engulfing(g: CandleGeometry, t: int, p: dict) -> PatternHit | None:
    """Body of bar t fully engulfs the opposite-colored body of bar t-1."""
    if t < 1 or g.body[t] < p["engulf_min_body_atr"] * g.atr[t]:
        return None
    po, pc, o, c = g.o[t - 1], g.c[t - 1], g.o[t], g.c[t]
    if g.bear[t - 1] and g.bull[t] and o <= pc and c >= po and g.body[t] > g.body[t - 1]:
        return _hit(g, "bullish engulfing", 1, t, 2, _quality(g, t))
    if g.bull[t - 1] and g.bear[t] and o >= pc and c <= po and g.body[t] > g.body[t - 1]:
        return _hit(g, "bearish engulfing", -1, t, 2, _quality(g, t))
    return None


def inside_bar(g: CandleGeometry, t: int, p: dict) -> PatternHit | None:
    """Bar t's range sits inside bar t-1's (the mother bar must be a meaningful candle)."""
    if t < 1 or g.rng[t - 1] < p["min_range_atr"] * g.atr[t]:
        return None
    if g.h[t] < g.h[t - 1] and g.l[t] > g.l[t - 1]:
        return _hit(g, "inside bar", 0, t, 2, _quality(g, t - 1))
    return None


def doji(g: CandleGeometry, t: int, p: dict) -> PatternHit | None:
    if g.rng[t] < p["min_range_atr"] * g.atr[t] or g.rng[t] <= 0:
        return None
    if g.body[t] <= p["doji_body_ratio"] * g.rng[t]:
        return _hit(g, "doji", 0, t, 1, _quality(g, t))
    return None


def star(g: CandleGeometry, t: int, p: dict) -> PatternHit | None:
    """Morning star (bearish big, small, bullish closing past bar-1 body midpoint) / evening star."""
    if t < 2:
        return None
    a, b = t - 2, t - 1
    big = p["star_big_body_ratio"]
    first_big = g.rng[a] > 0 and g.body[a] >= big * g.rng[a] and g.body[a] >= p["engulf_min_body_atr"] * g.atr[t]
    small_mid = g.body[b] <= 0.35 * g.body[a]
    if not (first_big and small_mid):
        return None
    mid = (g.o[a] + g.c[a]) / 2
    if g.bear[a] and g.bull[t] and g.c[t] > mid and g.body[t] >= 0.5 * g.body[a]:
        return _hit(g, "morning star", 1, t, 3, _quality(g, a))
    if g.bull[a] and g.bear[t] and g.c[t] < mid and g.body[t] >= 0.5 * g.body[a]:
        return _hit(g, "evening star", -1, t, 3, _quality(g, a))
    return None


def tweezer(g: CandleGeometry, t: int, p: dict) -> PatternHit | None:
    """Two bars with matching lows (bear then bull) or matching highs (bull then bear)."""
    if t < 1:
        return None
    tol = p["tweezer_tolerance_atr"] * g.atr[t]
    big_enough = min(g.rng[t], g.rng[t - 1]) >= p["min_range_atr"] * g.atr[t]
    if not big_enough:
        return None
    if g.bear[t - 1] and g.bull[t] and abs(g.l[t] - g.l[t - 1]) <= tol:
        return _hit(g, "tweezer bottom", 1, t, 2, _quality(g, t))
    if g.bull[t - 1] and g.bear[t] and abs(g.h[t] - g.h[t - 1]) <= tol:
        return _hit(g, "tweezer top", -1, t, 2, _quality(g, t))
    return None


DETECTORS = (pin_bar, engulfing, star, tweezer, inside_bar, doji)


def patterns_at(df: pd.DataFrame, atr_values: np.ndarray, t: int, params: dict,
                geometry: CandleGeometry | None = None) -> list[PatternHit]:
    """All patterns ending at bar t, strongest first."""
    g = geometry or CandleGeometry(df, atr_values)
    hits = [h for det in DETECTORS if (h := det(g, t, params)) is not None]
    return sorted(hits, key=lambda h: -h.strength)


def scan_patterns(df: pd.DataFrame, atr_values: np.ndarray, params: dict, start: int = 0) -> list[PatternHit]:
    """Patterns ending at every bar from `start` (used by the chart)."""
    g = CandleGeometry(df, atr_values)
    out: list[PatternHit] = []
    for t in range(max(start, 0), len(df)):
        out.extend(patterns_at(df, atr_values, t, params, g))
    return out
