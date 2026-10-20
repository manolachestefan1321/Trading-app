"""Average True Range (Wilder). The only indicator used, and only for volatility normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR. Causal: the value at bar t uses bars <= t only."""
    tr = true_range(df)
    out = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    # Before `period` bars exist, fall back to the expanding mean so early bars aren't NaN.
    return out.fillna(tr.expanding().mean()).astype(float)


def safe_atr(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) and value > 0 else float(fallback)
