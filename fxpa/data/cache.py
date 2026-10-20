"""Local parquet cache: ./data/cache/{symbol}_1H.parquet plus a small JSON metadata sidecar."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from fxpa.data.common import COLUMNS, empty_frame


class CandleCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str) -> Path:
        return self.dir / f"{symbol.upper()}_1H.parquet"

    def meta_path(self, symbol: str) -> Path:
        return self.dir / f"{symbol.upper()}_1H.meta.json"

    def load(self, symbol: str) -> pd.DataFrame:
        p = self.path(symbol)
        if not p.exists():
            return empty_frame()
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        cols = COLUMNS + [c for c in ("spike",) if c in df.columns]
        return df[cols].sort_values("timestamp").reset_index(drop=True)

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Atomic write (temp file + rename) so an interrupted run never corrupts the cache."""
        p = self.path(symbol)
        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)

    def load_meta(self, symbol: str) -> dict:
        p = self.meta_path(symbol)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}

    def save_meta(self, symbol: str, meta: dict) -> None:
        self.meta_path(symbol).write_text(json.dumps(meta, indent=2, default=str))
