"""DataManager: cache-first loading with incremental updates over a source fallback chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

from fxpa.config import Instrument
from fxpa.data.cache import CandleCache
from fxpa.data.common import (
    COLUMNS,
    DataSourceError,
    empty_frame,
    floor_hour,
    log,
    open_hours_between,
    utcnow,
)
from fxpa.data.sources import REGISTRY, DataSource
from fxpa.data.validate import ValidationReport, validate

ONE_HOUR = timedelta(hours=1)


@dataclass
class UpdateResult:
    symbol: str
    rows: int
    fetched_rows: int = 0
    ranges: list[tuple[datetime, datetime]] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report: ValidationReport | None = None
    stale: bool = False


class DataManager:
    """Owns the fetch -> validate -> cache cycle for every instrument.

    On each call to `get`, the cache is loaded and only missing candles are requested:
    the tail since the last stored candle and, if --years grew, the head before the first one.
    """

    def __init__(
        self,
        cfg: dict,
        instruments: dict[str, Instrument],
        source: str | None = None,
        refresh: bool = False,
        progress: bool = True,
        now_fn: Callable[[], datetime] = utcnow,
        sources: dict[str, DataSource] | None = None,
    ):
        self.cfg = cfg
        self.instruments = instruments
        self.cache = CandleCache(cfg["data"]["cache_dir"])
        self.refresh = refresh
        self.progress = progress
        self.now_fn = now_fn
        names = [source] if source else list(cfg["data"]["sources"])
        unknown = [n for n in names if n not in REGISTRY and not (sources and n in sources)]
        if unknown:
            raise ValueError(f"unknown data source(s) {unknown}; choose from {sorted(REGISTRY)}")
        self.forced = source is not None
        self.sources: list[DataSource] = []
        for n in names:
            if sources and n in sources:
                self.sources.append(sources[n])
            else:
                self.sources.append(REGISTRY[n](cfg))
        self.last_result: dict[str, UpdateResult] = {}

    # ------------------------------------------------------------------ ranges
    def _missing_ranges(self, cached: pd.DataFrame, meta: dict, start: datetime,
                        end: datetime) -> list[tuple[datetime, datetime]]:
        if cached.empty:
            return [(start, end)]
        ranges = []
        first = cached["timestamp"].iloc[0].to_pydatetime()
        last = cached["timestamp"].iloc[-1].to_pydatetime()
        requested_from = pd.Timestamp(meta.get("requested_start", first)).to_pydatetime()
        if start < requested_from and start < first:
            ranges.append((start, min(first, requested_from)))
        if last + ONE_HOUR < end:
            ranges.append((last + ONE_HOUR, end))
        return ranges

    # ------------------------------------------------------------------ fetch
    def _fetch_chain(self, inst: Instrument, start: datetime, end: datetime,
                     res: UpdateResult) -> pd.DataFrame:
        for src in self.sources:
            ok, why = src.is_available()
            if not ok:
                log.info("skip %s for %s: %s", src.name, inst.symbol, why)
                res.errors.append(f"{src.name}: {why}")
                continue
            lim = src.max_history_days()
            if lim is not None and (self.now_fn() - start).days > lim + 2:
                log.warning("%s serves only ~%d days of 1H history; the start of the range will be "
                            "missing", src.name, lim)
            log.info("fetching %s %s -> %s from %s", inst.symbol, f"{start:%Y-%m-%d %H:%M}",
                     f"{end:%Y-%m-%d %H:%M}", src.name)
            try:
                df = src.fetch(inst, start, end, progress=self.progress)
            except Exception as exc:  # each source failure falls through to the next one
                msg = f"{src.name}: {exc.__class__.__name__}: {str(exc)[:200]}"
                log.warning("source failed -> %s", msg)
                res.errors.append(msg)
                continue
            if df is None or df.empty:
                res.errors.append(f"{src.name}: no candles returned")
                log.warning("%s returned no candles for %s", src.name, inst.symbol)
                continue
            log.info("source used for %s: %s (%d candles)", inst.symbol, src.name, len(df))
            res.sources_used.append(src.name)
            return df
        return empty_frame()

    # ------------------------------------------------------------------ public
    def get(self, symbol: str, years: float | None = None) -> pd.DataFrame:
        """Return validated 1H candles for `symbol`, updating the cache as needed.

        Only closed candles are returned (the currently forming hour is excluded).
        """
        symbol = symbol.upper()
        if symbol not in self.instruments:
            raise KeyError(f"{symbol} is not configured in instruments.yaml")
        inst = self.instruments[symbol]
        years = float(years if years is not None else self.cfg["data"]["years"])
        now = floor_hour(self.now_fn())
        end = now  # exclusive: the candle opening at `now` is still forming
        start = now - timedelta(days=round(365.25 * years))

        cached = empty_frame() if self.refresh else self.cache.load(symbol)
        meta = {} if self.refresh else self.cache.load_meta(symbol)
        res = UpdateResult(symbol=symbol, rows=len(cached))

        ranges = self._missing_ranges(cached, meta, start, end)
        frames = [cached[COLUMNS]] if not cached.empty else []
        for a, b in ranges:
            if open_hours_between(a, b) == 0:
                continue  # whole range falls in the weekend closure: nothing to download
            res.ranges.append((a, b))
            new = self._fetch_chain(inst, a, b, res)
            res.fetched_rows += len(new)
            if not new.empty:
                frames.append(new)

        if res.ranges and res.fetched_rows == 0:
            res.stale = True
            if cached.empty:
                raise DataSourceError(
                    f"no data for {symbol}: every source failed -> " + " | ".join(res.errors))
            log.warning("could not update %s; using cached data up to %s", symbol,
                        cached["timestamp"].iloc[-1])

        merged = pd.concat(frames, ignore_index=True) if frames else empty_frame()
        merged = merged[merged["timestamp"] < pd.Timestamp(end)]
        clean, report = validate(merged, float(self.cfg["data"].get("spike_range_multiple", 10.0)))
        res.report = report
        res.rows = len(clean)

        if res.fetched_rows or self.refresh or (not cached.empty and len(clean) != len(cached)):
            self.cache.save(symbol, clean)
            prev_start = pd.Timestamp(meta.get("requested_start", start))
            meta.update({
                "symbol": symbol,
                "requested_start": str(min(prev_start, pd.Timestamp(start))),
                "last_update": str(self.now_fn()),
                "last_candle": str(clean["timestamp"].iloc[-1]) if len(clean) else None,
                "sources": sorted(set(meta.get("sources", [])) | set(res.sources_used)),
                "validation": report.summary(),
            })
            self.cache.save_meta(symbol, meta)
        elif not meta.get("requested_start"):
            meta["requested_start"] = str(start)
            self.cache.save_meta(symbol, meta)

        if res.ranges or self.refresh:
            log.info("%s: %s", symbol, report.summary())
        self.last_result[symbol] = res
        return clean[clean["timestamp"] >= pd.Timestamp(start)].reset_index(drop=True)

    def latest_prices(self) -> dict[str, float]:
        """Last cached close per symbol (used for cross-currency pip value conversion)."""
        out = {}
        for sym in self.instruments:
            df = self.cache.load(sym)
            if not df.empty:
                out[sym] = float(df["close"].iloc[-1])
        return out
