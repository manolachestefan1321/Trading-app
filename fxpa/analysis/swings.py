"""Swing high / low detection with fractals and an ATR noise filter.

A fractal swing high at bar i has a high strictly above the `window` highs before it and
at least as high as the `window` highs after it (ties go to the earlier bar). It is only
*known* at bar i + window, recorded as `confirmed_idx`, which keeps everything causal.

The noise filter turns raw fractals into an alternating high/low sequence: consecutive
same-type fractals keep the more extreme one, and an opposite swing closer than
`min_swing_atr * ATR` to the previous swing is ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SWING_COLUMNS = ["idx", "timestamp", "price", "kind", "confirmed_idx", "across_gap"]


def fractal_pivots(high: np.ndarray, low: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays marking raw fractal highs and lows."""
    n = len(high)
    is_h = np.zeros(n, dtype=bool)
    is_l = np.zeros(n, dtype=bool)
    if n < 2 * window + 1:
        return is_h, is_l
    w = window
    hv = sliding_window_view(high, 2 * w + 1)
    lv = sliding_window_view(low, 2 * w + 1)
    center_h = hv[:, w]
    center_l = lv[:, w]
    left_h, right_h = hv[:, :w].max(axis=1), hv[:, w + 1:].max(axis=1)
    left_l, right_l = lv[:, :w].min(axis=1), lv[:, w + 1:].min(axis=1)
    is_h[w:n - w] = (center_h > left_h) & (center_h >= right_h)
    is_l[w:n - w] = (center_l < left_l) & (center_l <= right_l)
    return is_h, is_l


def detect_swings(
    df: pd.DataFrame,
    atr_values: np.ndarray | pd.Series,
    window: int = 3,
    min_swing_atr: float = 0.75,
) -> pd.DataFrame:
    """Return the filtered, alternating swing sequence for `df`.

    Columns: idx (row position), timestamp, price, kind ('H'/'L'), confirmed_idx,
    across_gap (a weekend/missing-data gap lies inside the fractal window).
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = np.asarray(atr_values, dtype=float)
    bullish = df["close"].to_numpy(dtype=float) >= df["open"].to_numpy(dtype=float)
    is_h, is_l = fractal_pivots(high, low, window)

    cands: list[tuple[int, str, float]] = []
    for i in np.flatnonzero(is_h | is_l):
        # An outside bar can be both; order by which extreme is more pronounced is unknowable
        # from OHLC, so emit the low first for a bullish close and the high first otherwise.
        both = is_h[i] and is_l[i]
        if both and bullish[i]:
            cands += [(i, "L", low[i]), (i, "H", high[i])]
        elif both:
            cands += [(i, "H", high[i]), (i, "L", low[i])]
        elif is_h[i]:
            cands.append((i, "H", high[i]))
        else:
            cands.append((i, "L", low[i]))

    seq: list[list] = []
    for i, kind, price in cands:
        if not seq:
            seq.append([i, kind, price])
            continue
        li, lkind, lprice = seq[-1]
        if kind == lkind:
            more_extreme = price > lprice if kind == "H" else price < lprice
            if more_extreme:
                seq[-1] = [i, kind, price]
            continue
        thr = min_swing_atr * (atr_arr[i] if np.isfinite(atr_arr[i]) else 0.0)
        if abs(price - lprice) < thr:
            continue
        seq.append([i, kind, price])

    if not seq:
        return pd.DataFrame(columns=SWING_COLUMNS)

    gaps = df["gap_before"].to_numpy(dtype=bool) if "gap_before" in df.columns else np.zeros(len(df), bool)
    gap_cum = np.cumsum(gaps)
    times = df["timestamp"].to_numpy(dtype="datetime64[ns]")  # UTC, avoids object boxing
    n = len(df)
    rows = []
    for i, kind, price in seq:
        lo, hi = max(i - window + 1, 0), min(i + window, n - 1)
        across = bool(gap_cum[hi] - (gap_cum[lo - 1] if lo > 0 else 0) > 0)
        rows.append({
            "idx": int(i),
            "timestamp": times[i],
            "price": float(price),
            "kind": kind,
            "confirmed_idx": int(i + window),
            "across_gap": across,
        })
    out = pd.DataFrame(rows, columns=SWING_COLUMNS)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out
