"""Market structure: HH/HL vs LH/LL classification, BOS / CHoCH events, trend strength."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StructureEvent:
    kind: str          # "BOS" (continuation) or "CHoCH" (change of character)
    direction: int     # +1 bullish, -1 bearish
    idx: int           # bar index of the breaking close
    timestamp: pd.Timestamp
    level: float       # swing price that was broken


@dataclass
class StructureState:
    trend: str = "range"                 # "up" | "down" | "range"
    strength: float = 0.0                # 0-100
    confirming: int = 0                  # consecutive confirming swing labels
    labels: list[str] = field(default_factory=list)
    last_high: float | None = None
    last_low: float | None = None
    events: list[StructureEvent] = field(default_factory=list)

    @property
    def sign(self) -> int:
        return {"up": 1, "down": -1}.get(self.trend, 0)

    def last_event(self, kind: str, since_idx: int | None = None) -> StructureEvent | None:
        for ev in reversed(self.events):
            if ev.kind == kind and (since_idx is None or ev.idx >= since_idx):
                return ev
        return None

    def describe(self) -> str:
        if self.trend == "up":
            return f"HH/HL x{self.confirming}"
        if self.trend == "down":
            return f"LH/LL x{self.confirming}"
        return "range"


def label_swings(swings: pd.DataFrame) -> list[str]:
    """Label each swing vs the previous swing of the same kind: HH/LH/EH and HL/LL/EL."""
    labels = []
    last = {"H": None, "L": None}
    for kind, price in zip(swings["kind"], swings["price"]):
        prev = last[kind]
        if prev is None:
            labels.append(kind)  # first of its kind: no comparison possible
        elif kind == "H":
            labels.append("HH" if price > prev else "LH" if price < prev else "EH")
        else:
            labels.append("HL" if price > prev else "LL" if price < prev else "EL")
        last[kind] = price
    return labels


def find_events(swings: pd.DataFrame, close: np.ndarray, timestamps: pd.Series) -> list[StructureEvent]:
    """Walk the bars and record breaks of the latest confirmed swing high / low.

    A close above the last swing high is a bullish BOS if structure was already bullish and a
    bullish CHoCH if it was bearish (and vice versa). Only swings already confirmed at the
    breaking bar are used. The very first break has no prior bias and is labeled a BOS.
    """
    events: list[StructureEvent] = []
    if swings.empty:
        return events
    sw = swings.sort_values("confirmed_idx")
    conf = sw["confirmed_idx"].to_numpy()
    kinds = sw["kind"].to_numpy()
    prices = sw["price"].to_numpy()
    k = 0
    cur_h = cur_l = None
    h_broken = l_broken = True
    bias = 0
    for j in range(int(conf[0]), len(close)):
        while k < len(conf) and conf[k] <= j:
            if kinds[k] == "H":
                cur_h, h_broken = prices[k], False
            else:
                cur_l, l_broken = prices[k], False
            k += 1
        c = close[j]
        if cur_h is not None and not h_broken and c > cur_h:
            events.append(StructureEvent("CHoCH" if bias == -1 else "BOS", 1, j, timestamps.iat[j], cur_h))
            bias, h_broken = 1, True
        elif cur_l is not None and not l_broken and c < cur_l:
            events.append(StructureEvent("CHoCH" if bias == 1 else "BOS", -1, j, timestamps.iat[j], cur_l))
            bias, l_broken = -1, True
    return events


def analyze_structure(swings: pd.DataFrame, df: pd.DataFrame, strength_full_count: int = 6) -> StructureState:
    """Classify trend from the last two highs and lows and count confirming swings."""
    st = StructureState()
    if swings.empty:
        return st
    labels = label_swings(swings)
    st.labels = labels
    highs = swings[swings["kind"] == "H"]["price"].to_numpy()
    lows = swings[swings["kind"] == "L"]["price"].to_numpy()
    st.last_high = float(highs[-1]) if len(highs) else None
    st.last_low = float(lows[-1]) if len(lows) else None

    if len(highs) >= 2 and len(lows) >= 2:
        hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
        lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
        st.trend = "up" if (hh and hl) else "down" if (lh and ll) else "range"

    good = {"up": {"HH", "HL"}, "down": {"LH", "LL"}}.get(st.trend)
    if good:
        n = 0
        for lab in reversed(labels):
            if lab in good:
                n += 1
            else:
                break
        st.confirming = n
        st.strength = float(min(100.0, 100.0 * n / max(strength_full_count, 1)))

    st.events = find_events(swings, df["close"].to_numpy(dtype=float), df["timestamp"])
    return st
