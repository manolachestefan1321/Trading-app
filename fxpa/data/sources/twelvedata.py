"""Twelve Data adapter (requires TWELVEDATA_API_KEY).

Quirks handled here:
- outputsize max 5000 per request -> chunked by 200 days.
- Free tier allows 8 requests/minute, so chunks are spaced out.
- Times are requested in UTC explicitly; volume is often missing for FX (set to 0).
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

from fxpa.config import Instrument, env
from fxpa.data.common import DataSourceError, chunk_ranges, empty_frame, normalize_ohlcv, retry_call
from fxpa.data.sources.base import DataSource

URL = "https://api.twelvedata.com/time_series"


class TwelveDataSource(DataSource):
    name = "twelvedata"
    request_spacing_s = 8.0

    def is_available(self) -> tuple[bool, str]:
        if not env("TWELVEDATA_API_KEY"):
            return False, "TWELVEDATA_API_KEY not set"
        return True, ""

    def fetch(self, inst: Instrument, start: datetime, end: datetime, progress: bool = True) -> pd.DataFrame:
        rows = []
        chunks = chunk_ranges(start, end, 200)
        for i, (a, b) in enumerate(tqdm(chunks, desc=f"twelvedata {inst.symbol}", unit="chunk",
                                        disable=not progress)):
            if i:
                time.sleep(self.request_spacing_s)
            params = {
                "symbol": inst.twelvedata, "interval": "1h", "timezone": "UTC", "outputsize": 5000,
                "start_date": a.strftime("%Y-%m-%d %H:%M:%S"), "end_date": b.strftime("%Y-%m-%d %H:%M:%S"),
                "order": "ASC", "apikey": env("TWELVEDATA_API_KEY"),
            }

            def _get() -> dict:
                r = requests.get(URL, params=params, timeout=30)
                r.raise_for_status()
                return r.json()

            data = retry_call(_get, retries=self.retries, base_delay=self.backoff, label="twelvedata",
                              exceptions=(requests.RequestException,))
            if data.get("status") == "error":
                if "No data" in str(data.get("message", "")):
                    continue
                raise DataSourceError(f"Twelve Data: {data.get('message')}")
            for v in data.get("values", []):
                rows.append({"timestamp": v["datetime"], "open": v["open"], "high": v["high"],
                             "low": v["low"], "close": v["close"], "volume": v.get("volume", 0)})
        if not rows:
            return empty_frame()
        return normalize_ohlcv(pd.DataFrame(rows))
