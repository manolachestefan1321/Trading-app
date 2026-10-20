"""Data source adapters, registered by name for the fallback chain."""

from __future__ import annotations

from fxpa.data.sources.base import DataSource
from fxpa.data.sources.dukascopy import DukascopySource
from fxpa.data.sources.oanda import OandaSource
from fxpa.data.sources.twelvedata import TwelveDataSource
from fxpa.data.sources.yfinance_src import YFinanceSource

REGISTRY: dict[str, type[DataSource]] = {
    "dukascopy": DukascopySource,
    "yfinance": YFinanceSource,
    "oanda": OandaSource,
    "twelvedata": TwelveDataSource,
}

__all__ = ["DataSource", "REGISTRY"]
