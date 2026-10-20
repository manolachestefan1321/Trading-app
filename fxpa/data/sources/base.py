"""Abstract interface every data source adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from fxpa.config import Instrument


class DataSource(ABC):
    """A provider of 1H OHLCV candles.

    Adapters handle all source-specific quirks (symbols, time zones, pagination,
    price scaling) and return frames already passed through `normalize_ohlcv`.
    """

    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.retries = int(cfg["data"].get("max_retries", 4))
        self.backoff = float(cfg["data"].get("backoff_base_seconds", 2.0))

    def is_available(self) -> tuple[bool, str]:
        """(available, reason). Sources needing API keys or packages override this."""
        return True, ""

    def max_history_days(self) -> int | None:
        """How far back this source can serve 1H data (None = unlimited)."""
        return None

    @abstractmethod
    def fetch(self, inst: Instrument, start: datetime, end: datetime, progress: bool = True) -> pd.DataFrame:
        """Return 1H candles with open time in [start, end), canonical schema, UTC."""
