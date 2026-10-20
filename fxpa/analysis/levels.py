"""Key reference levels: previous day/week, weekly open, Asian range, psychological round numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from fxpa.data.resample import period_key, resample
from fxpa.forex.sessions import asian_range


@dataclass
class KeyLevel:
    name: str
    price: float
    weight: float     # 0-1, how much extra weight the level adds to a zone


def round_numbers(price: float, steps: tuple[float, ...], each_side: int = 2) -> list[KeyLevel]:
    """Nearest round numbers above/below price. steps = (major, minor, ...)."""
    out: list[KeyLevel] = []
    seen: set[float] = set()
    for rank, step in enumerate(steps):
        weight = 1.0 if rank == 0 else 0.6
        base = math.floor(price / step)
        for k in range(-each_side + 1, each_side + 1):
            lvl = round((base + k) * step, 10)
            key = round(lvl, 8)
            if key in seen:
                continue
            seen.add(key)
            out.append(KeyLevel(f"round {lvl:g}", lvl, weight))
    return out


def key_levels(df: pd.DataFrame, round_steps: tuple[float, ...], sess_cfg: dict,
               each_side: int = 2) -> list[KeyLevel]:
    """Levels known at the close of the last bar in df (completed periods only)."""
    if df.empty:
        return []
    out: list[KeyLevel] = []
    daily = resample(df, "D")
    if len(daily):
        d = daily.iloc[-1]
        out += [KeyLevel("PDH", float(d["high"]), 1.0), KeyLevel("PDL", float(d["low"]), 1.0),
                KeyLevel("PDC", float(d["close"]), 0.6)]
    weekly = resample(df, "W")
    if len(weekly):
        w = weekly.iloc[-1]
        out += [KeyLevel("PWH", float(w["high"]), 1.0), KeyLevel("PWL", float(w["low"]), 1.0)]
    # Weekly open: first candle of the current trading week (known as soon as it opens).
    wk = period_key(df["timestamp"], "W")
    cur = df[wk == wk.iloc[-1]]
    if len(cur):
        out.append(KeyLevel("weekly open", float(cur["open"].iloc[0]), 0.7))
    rng = asian_range(df, len(df) - 1, sess_cfg)
    if rng is not None:
        out += [KeyLevel("Asian high", rng.high, 0.7), KeyLevel("Asian low", rng.low, 0.7)]
    out += round_numbers(float(df["close"].iloc[-1]), round_steps, each_side)
    return out
