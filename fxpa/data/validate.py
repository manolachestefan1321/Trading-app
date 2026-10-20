"""Data validation: dedupe, sort, OHLC sanity, spikes, missing candles. Never interpolates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxpa.data.common import COLUMNS, log, market_closed_mask


@dataclass
class ValidationReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    invalid_ohlc_dropped: int = 0
    closed_market_dropped: int = 0
    spikes_flagged: int = 0
    missing_candles: int = 0              # open-market hours with no candle
    missing_examples: list[str] = field(default_factory=list)
    weekend_gaps: int = 0

    def summary(self) -> str:
        return (f"{self.rows_out} candles (dropped: {self.duplicates_dropped} dup, "
                f"{self.invalid_ohlc_dropped} bad OHLC, {self.closed_market_dropped} closed-market); "
                f"{self.spikes_flagged} spikes flagged; {self.missing_candles} missing open-market candles "
                f"(not interpolated); {self.weekend_gaps} weekend gaps")


def validate(df: pd.DataFrame, spike_range_multiple: float = 10.0) -> tuple[pd.DataFrame, ValidationReport]:
    """Clean a candle frame and report problems.

    - drops exact-timestamp duplicates (keeps the last, i.e. newest download)
    - sorts by time
    - drops candles violating high >= max(open, close) / low <= min(open, close)
    - drops candles stamped while the market is closed (weekend placeholders)
    - flags (does not drop) spikes: range > N x rolling median range -> column `spike`
    - counts missing hours outside the weekend closure; they are NOT filled
    """
    rep = ValidationReport(rows_in=len(df))
    if df.empty:
        out = df.copy()
        out["spike"] = pd.Series(dtype=bool)
        return out, rep

    out = df[COLUMNS].copy()
    before = len(out)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    rep.duplicates_dropped = before - len(out)

    o, h, l, c = (out[k].to_numpy() for k in ("open", "high", "low", "close"))
    eps = 1e-12
    bad = (h + eps < np.maximum(o, c)) | (l - eps > np.minimum(o, c)) | (h < l) | ~np.isfinite(o + h + l + c)
    bad |= (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    rep.invalid_ohlc_dropped = int(bad.sum())
    if rep.invalid_ohlc_dropped:
        log.warning("dropping %d candles with inconsistent OHLC", rep.invalid_ohlc_dropped)
    out = out[~bad]

    closed = market_closed_mask(out["timestamp"])
    # Some brokers print a Sunday pre-open candle; only drop closed-market candles that are flat.
    flat = (out["high"] == out["low"]).to_numpy()
    drop_closed = closed & flat
    rep.closed_market_dropped = int(drop_closed.sum())
    out = out[~drop_closed].reset_index(drop=True)

    rng = (out["high"] - out["low"]).to_numpy()
    med = pd.Series(rng).rolling(100, min_periods=20).median().shift(1).to_numpy()
    spike = np.where(np.isfinite(med) & (med > 0), rng > spike_range_multiple * med, False)
    out["spike"] = spike
    rep.spikes_flagged = int(spike.sum())
    if rep.spikes_flagged:
        log.warning("%d suspicious spike candles flagged (kept, column 'spike')", rep.spikes_flagged)

    # Missing-candle detection: every open-market hour between first and last candle should exist.
    if len(out) > 1:
        full = pd.date_range(out["timestamp"].iloc[0], out["timestamp"].iloc[-1], freq="h")
        expected = full[~market_closed_mask(pd.Series(full))]
        missing = expected.difference(pd.DatetimeIndex(out["timestamp"]))
        rep.missing_candles = len(missing)
        rep.missing_examples = [str(t) for t in missing[:5]]
        gaps = out["timestamp"].diff() >= pd.Timedelta(hours=24)
        rep.weekend_gaps = int(gaps.sum())

    rep.rows_out = len(out)
    return out, rep


def add_gap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add `gap_before` (any missing hour before this bar) and `weekend_gap` (>= 24h gap).

    Swings and structure across these are marked rather than silently computed through.
    """
    out = df.copy()
    dt = out["timestamp"].diff()
    out["gap_before"] = (dt > pd.Timedelta(hours=1)).fillna(False).astype(bool)
    out["weekend_gap"] = (dt >= pd.Timedelta(hours=24)).fillna(False).astype(bool)
    return out
