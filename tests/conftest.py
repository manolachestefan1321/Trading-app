"""Shared fixtures. Synthetic data here is only for tests; it never reaches the cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fxpa.config import load_config, load_instruments  # noqa: E402
from fxpa.data.common import market_closed_mask  # noqa: E402


def make_candles(n: int = 500, start: str = "2024-01-02 00:00", seed: int = 1,
                 price: float = 1.1000, vol: float = 0.0012, drift: float = 0.0) -> pd.DataFrame:
    """Random-walk 1H candles on open-market hours only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(n * 1.6) + 100, freq="h", tz="UTC")
    idx = idx[~market_closed_mask(pd.Series(idx))][:n]
    rets = rng.normal(drift, vol, len(idx))
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, vol * 0.6, (2, len(idx)))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.integers(500, 3000, len(idx)).astype(float)})


def bars(rows: list[tuple[float, float, float, float]], start: str = "2024-01-02 08:00",
         volume: float = 1000.0) -> pd.DataFrame:
    """Hand-made OHLC fixture: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"timestamp": idx, "open": o, "high": h, "low": l, "close": c,
                         "volume": [volume] * len(rows)})


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"data": {"cache_dir": str(tmp_path / "cache")}})


@pytest.fixture
def instruments():
    return load_instruments()
