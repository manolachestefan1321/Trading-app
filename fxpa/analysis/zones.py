"""Support / resistance zones: swing clustering, scoring, role reversal, key-level confluence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxpa.analysis.levels import KeyLevel


@dataclass
class Zone:
    low: float
    high: float
    touches: int
    n_highs: int                 # swing highs in the cluster (resistance origin)
    n_lows: int                  # swing lows in the cluster (support origin)
    last_touch_idx: int
    reaction_atr: float          # mean move away from the zone after a touch, in ATR
    volume_ratio: float          # tick volume at touches / average tick volume
    score: float = 0.0           # 0-100
    role: str = ""               # "support" | "resistance" relative to the last close
    flipped: bool = False        # role reversal (broken resistance now support, or vice versa)
    key_levels: list[str] = field(default_factory=list)
    origin: str = "swings"       # "swings" or "key" (built from a key level only)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2

    def contains(self, price: float, pad: float = 0.0) -> bool:
        return self.low - pad <= price <= self.high + pad

    def distance(self, price: float) -> float:
        if price > self.high:
            return price - self.high
        if price < self.low:
            return self.low - price
        return 0.0

    def label(self, fmt) -> str:
        extras = f", {'/'.join(self.key_levels)} confluence" if self.key_levels else ""
        flip = ", role reversal" if self.flipped else ""
        if self.origin == "key":
            return f"{self.role} {fmt(self.low)}-{fmt(self.high)} ({'/'.join(self.key_levels)})"
        touches = f"{self.touches} touch{'es' if self.touches != 1 else ''}"
        return f"{self.role} {fmt(self.low)}-{fmt(self.high)} ({touches}{extras}{flip})"


def _cluster(prices: np.ndarray, max_gap: float, max_width: float) -> list[np.ndarray]:
    """Greedy 1-D clustering of sorted swing prices (indices into `prices`)."""
    order = np.argsort(prices)
    clusters: list[list[int]] = []
    for i in order:
        if clusters:
            cur = clusters[-1]
            if prices[i] - prices[cur[-1]] <= max_gap and prices[i] - prices[cur[0]] <= max_width:
                cur.append(int(i))
                continue
        clusters.append([int(i)])
    return [np.array(c) for c in clusters]


def build_zones(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    atr_now: float,
    params: dict,
    levels: list[KeyLevel] | None = None,
    now_idx: int | None = None,
) -> list[Zone]:
    """Cluster confirmed swings into horizontal zones and score them.

    Score (0-100) = 40% touches + 25% recency + 25% reaction strength + 10% tick volume,
    plus a bonus when a key level (PDH, round number, ...) sits inside the zone.
    """
    now_idx = len(df) - 1 if now_idx is None else now_idx
    close = float(df["close"].iat[now_idx])
    vol = df["volume"].to_numpy(dtype=float)
    avg_vol = float(vol.mean()) if vol.size and vol.mean() > 0 else 0.0
    sw = swings[swings["confirmed_idx"] <= now_idx].reset_index(drop=True)
    zones: list[Zone] = []

    if len(sw):
        prices = sw["price"].to_numpy(dtype=float)
        idxs = sw["idx"].to_numpy(dtype=int)
        kinds = sw["kind"].to_numpy()
        # Reaction = move to the next swing (always the opposite kind), in ATR at that time.
        nxt = np.r_[prices[1:], np.nan]
        reaction = np.abs(nxt - prices) / max(atr_now, 1e-12)
        min_w = params["min_width_atr"] * atr_now
        for members in _cluster(prices, params["cluster_atr"] * atr_now, params["max_width_atr"] * atr_now):
            lo, hi = float(prices[members].min()), float(prices[members].max())
            if hi - lo < min_w:
                mid = (hi + lo) / 2
                lo, hi = mid - min_w / 2, mid + min_w / 2
            r = reaction[members]
            r = r[np.isfinite(r)]
            v_ratio = float(vol[idxs[members]].mean() / avg_vol) if avg_vol > 0 else 1.0
            zones.append(Zone(
                low=lo, high=hi, touches=len(members),
                n_highs=int((kinds[members] == "H").sum()), n_lows=int((kinds[members] == "L").sum()),
                last_touch_idx=int(idxs[members].max()),
                reaction_atr=float(r.mean()) if r.size else 0.0,
                volume_ratio=v_ratio,
            ))

    halflife = float(params["recency_halflife_bars"])
    for z in zones:
        recency = 0.5 ** ((now_idx - z.last_touch_idx) / halflife)
        z.score = 100 * (0.40 * min(z.touches, 5) / 5 + 0.25 * recency
                         + 0.25 * min(z.reaction_atr, 4) / 4 + 0.10 * min(z.volume_ratio, 2) / 2)

    # Key-level confluence; key levels that don't fall in any zone become thin zones of their own.
    pad = params.get("near_atr", 0.35) * atr_now * 0.5
    bonus = float(params.get("key_level_bonus", 15))
    for lvl in levels or []:
        inside = [z for z in zones if z.origin == "swings" and z.contains(lvl.price, pad)]
        if inside:
            for z in inside:
                if lvl.name not in z.key_levels:
                    z.key_levels.append(lvl.name)
                    z.score += bonus * lvl.weight
        else:
            half = params["min_width_atr"] * atr_now / 2
            zones.append(Zone(low=lvl.price - half, high=lvl.price + half, touches=0, n_highs=0, n_lows=0,
                              last_touch_idx=now_idx, reaction_atr=0.0, volume_ratio=1.0,
                              score=35.0 * lvl.weight, key_levels=[lvl.name], origin="key"))

    for z in zones:
        z.score = float(min(z.score, 100.0))
        if close > z.high:
            z.role = "support"
        elif close < z.low:
            z.role = "resistance"
        else:  # price inside the zone: keep the role it was formed with
            z.role = "support" if z.n_lows >= z.n_highs else "resistance"
        # Role reversal: formed mostly by swing highs but price now above it (or the reverse).
        z.flipped = (z.role == "support" and z.n_highs > z.n_lows) or \
                    (z.role == "resistance" and z.n_lows > z.n_highs)

    return select_zones(zones, close, int(params.get("max_zones", 12)))


def select_zones(zones: list[Zone], close: float, max_zones: int, keep_nearest: int = 3) -> list[Zone]:
    """Keep the best-scored zones plus the `keep_nearest` closest on each side of price
    (targets and stops need the nearby structure even when it scores low)."""
    if len(zones) <= max_zones:
        return sorted(zones, key=lambda z: z.mid)
    keep = sorted(zones, key=lambda z: -z.score)[:max_zones]
    sup = sorted((z for z in zones if z.role == "support"), key=lambda z: -z.high)[:keep_nearest]
    res = sorted((z for z in zones if z.role == "resistance"), key=lambda z: z.low)[:keep_nearest]
    for z in sup + res:
        if z not in keep:
            keep.append(z)
    return sorted(keep, key=lambda z: z.mid)
