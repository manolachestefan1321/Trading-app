"""OANDA v20 adapter (requires OANDA_API_KEY; OANDA_ENV=practice|live).

Quirks handled here:
- Max 5000 candles per request, so ranges are chunked (~200 days of H1).
- BID prices are requested for consistency with Dukascopy.
- Only `complete` candles are kept.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

from fxpa.config import Instrument, env
from fxpa.data.common import DataSourceError, chunk_ranges, empty_frame, normalize_ohlcv, retry_call
from fxpa.data.sources.base import DataSource

HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}


class OandaSource(DataSource):
    name = "oanda"

    def is_available(self) -> tuple[bool, str]:
        if not env("OANDA_API_KEY"):
            return False, "OANDA_API_KEY not set"
        return True, ""

    def fetch(self, inst: Instrument, start: datetime, end: datetime, progress: bool = True) -> pd.DataFrame:
        host = HOSTS.get((env("OANDA_ENV") or "practice").lower(), HOSTS["practice"])
        headers = {"Authorization": f"Bearer {env('OANDA_API_KEY')}"}
        url = f"{host}/v3/instruments/{inst.oanda}/candles"
        rows = []
        for a, b in tqdm(chunk_ranges(start, end, 200), desc=f"oanda {inst.symbol}", unit="chunk",
                         disable=not progress):
            params = {"granularity": "H1", "price": "B", "from": a.isoformat(), "to": b.isoformat()}

            def _get() -> dict:
                r = requests.get(url, headers=headers, params=params, timeout=30)
                if r.status_code in (401, 403, 404):
                    raise DataSourceError(f"OANDA HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()

            data = retry_call(_get, retries=self.retries, base_delay=self.backoff, label="oanda candles",
                              exceptions=(requests.RequestException,))
            for c in data.get("candles", []):
                if not c.get("complete", False):
                    continue
                p = c["bid"]
                rows.append({"timestamp": c["time"], "open": p["o"], "high": p["h"], "low": p["l"],
                             "close": p["c"], "volume": c.get("volume", 0)})
        if not rows:
            return empty_frame()
        return normalize_ohlcv(pd.DataFrame(rows))
