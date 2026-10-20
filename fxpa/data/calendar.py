"""Economic calendar download (ForexFactory weekly JSON feed) with a local accumulating cache.

The free feed only covers the current week, so every fetch is merged into
data/cache/calendar.parquet; over time this builds a history the backtest can use.
If the feed is unreachable the program continues with a warning.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from fxpa.data.common import log, retry_call, utcnow

CAL_COLUMNS = ["time", "currency", "impact", "title"]


def parse_forexfactory(items: list[dict]) -> pd.DataFrame:
    """ForexFactory JSON items -> DataFrame[time (UTC), currency, impact, title]."""
    rows = []
    for it in items:
        try:
            t = pd.Timestamp(it["date"])
        except (KeyError, ValueError, TypeError):
            continue
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        rows.append({"time": t, "currency": str(it.get("country", "")).upper(),
                     "impact": str(it.get("impact", "")), "title": str(it.get("title", ""))})
    if not rows:
        return pd.DataFrame(columns=CAL_COLUMNS)
    return pd.DataFrame(rows, columns=CAL_COLUMNS)


class EconomicCalendar:
    def __init__(self, cfg: dict, now_fn=utcnow):
        self.cfg = cfg
        self.url = cfg["data"]["calendar_url"]
        self.path = Path(cfg["data"]["cache_dir"]) / "calendar.parquet"
        self.refresh_after = timedelta(hours=float(cfg["data"].get("calendar_refresh_hours", 6)))
        self.now_fn = now_fn
        self.events = pd.DataFrame(columns=CAL_COLUMNS)
        self.available = False
        self.warning: str | None = None

    def _load_cache(self) -> pd.DataFrame:
        if self.path.exists():
            df = pd.read_parquet(self.path)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df
        return pd.DataFrame(columns=CAL_COLUMNS)

    def load(self, fetch: bool = True) -> "EconomicCalendar":
        cached = self._load_cache()
        stale = True
        if self.path.exists():
            age = self.now_fn() - datetime.fromtimestamp(self.path.stat().st_mtime, tz=self.now_fn().tzinfo)
            stale = age > self.refresh_after
        if fetch and stale:
            try:
                def _get():
                    r = requests.get(self.url, timeout=20, headers={"User-Agent": "fxpa/0.1"})
                    r.raise_for_status()
                    return r.json()

                items = retry_call(_get, retries=2, base_delay=2.0, label="economic calendar",
                                   exceptions=(requests.RequestException, ValueError))
                fresh = parse_forexfactory(items)
                cached = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
                cached = cached.drop_duplicates(["time", "currency", "title"]).sort_values("time")
                self.path.parent.mkdir(parents=True, exist_ok=True)
                cached.to_parquet(self.path, index=False)
                log.info("economic calendar updated: %d events this week", len(fresh))
            except Exception as exc:
                self.warning = (f"economic calendar unavailable ({exc.__class__.__name__}); "
                                "news blackout filter uses cached events only")
                log.warning(self.warning)
        self.events = cached.reset_index(drop=True)
        self.available = len(self.events) > 0
        if not self.available and not self.warning:
            self.warning = "no economic calendar data: news blackout filter is inactive"
        return self

    def coverage(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if self.events.empty:
            return None
        return self.events["time"].min(), self.events["time"].max()
