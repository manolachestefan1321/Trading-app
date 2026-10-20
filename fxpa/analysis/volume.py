"""Tick volume confirmation.

Spot forex has no central volume: this is *tick volume* (number of price updates or the
provider's traded volume), a weak proxy for participation. Its weight is capped at 5%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_ratio(df: pd.DataFrame, t: int, rolling: int = 20) -> float | None:
    """Signal-candle volume / average of the previous `rolling` candles. None if no volume data."""
    v = df["volume"].to_numpy(dtype=float)
    hist = v[max(0, t - rolling):t]
    if hist.size < max(5, rolling // 2) or hist.mean() <= 0 or not np.isfinite(v[t]):
        return None
    return float(v[t] / hist.mean())


def volume_component(ratio: float | None, direction: int) -> float:
    """Above-average volume confirms the candidate direction, below-average weakens it."""
    if ratio is None or direction == 0:
        return 0.0
    return float(np.clip(ratio - 1.0, -1.0, 1.0)) * direction
