"""Price behavior around zones: bounces, breakouts, fakeouts, retests, failed bounces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxpa.analysis.zones import Zone


@dataclass
class BehaviorHit:
    kind: str          # bounce | pullback | breakout | failed bounce | fakeout
    direction: int
    value: float       # signed contribution in [-1, 1]
    text: str
    zone: Zone | None = None


def zone_breakouts(df: pd.DataFrame, zone: Zone, atr: np.ndarray, t: int, lookback: int,
                   close_atr: float) -> list[tuple[int, int]]:
    """(bar index, direction) of decisive closes through the zone within the lookback window."""
    c = df["close"].to_numpy(dtype=float)
    out = []
    for j in range(max(1, t - lookback + 1), t + 1):
        m = close_atr * atr[j]
        if c[j] > zone.high + m and c[j - 1] <= zone.high + m:
            out.append((j, 1))
        elif c[j] < zone.low - m and c[j - 1] >= zone.low - m:
            out.append((j, -1))
    return out


def evaluate_behavior(
    df: pd.DataFrame,
    zones: list[Zone],
    atr: np.ndarray,
    t: int,
    trend_sign: int,
    params: dict,
    near_atr: float,
    fmt=lambda p: f"{p:.5f}",
) -> list[BehaviorHit]:
    """All behavior signals at bar t, strongest first."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = atr[t]
    near = near_atr * a
    hits: list[BehaviorHit] = []
    for z in zones:
        if z.distance(c[t]) > 2.5 * a:
            continue
        zl = f"{fmt(z.low)}-{fmt(z.high)}"
        body = abs(c[t] - o[t])
        decisive = body >= params["decisive_body_atr"] * a
        m = params["breakout_close_atr"] * a

        # Failed bounce / decisive break: a strong close through the zone on this bar, starting
        # from inside the zone or the other side of it (not from a bar already beyond it).
        if decisive and c[t] < z.low - m and c[t - 1] >= z.low:
            kind = "failed bounce" if z.n_lows >= z.n_highs else "breakout"
            hits.append(BehaviorHit(kind, -1, -0.8, f"decisive close below zone {zl}", z))
            continue
        if decisive and c[t] > z.high + m and c[t - 1] <= z.high:
            kind = "failed bounce" if z.n_highs >= z.n_lows else "breakout"
            hits.append(BehaviorHit(kind, 1, 0.8, f"decisive close above zone {zl}", z))
            continue

        # Fakeout (false breakout / liquidity sweep): wick through, close back inside or beyond.
        if l[t] < z.low - 0.1 * a and c[t] > z.low and c[t - 1] >= z.low:
            hits.append(BehaviorHit("fakeout", 1, 0.7, f"false break below {zl} (stop hunt), closed back above", z))
            continue
        if h[t] > z.high + 0.1 * a and c[t] < z.high and c[t - 1] <= z.high:
            hits.append(BehaviorHit("fakeout", -1, -0.7, f"false break above {zl} (stop hunt), closed back below", z))
            continue

        touched_from_above = min(l[max(0, t - 1):t + 1]) <= z.high + near and c[t] > z.high
        touched_from_below = max(h[max(0, t - 1):t + 1]) >= z.low - near and c[t] < z.low
        # Pullback / retest of a level broken earlier, in the direction of the break.
        brk = zone_breakouts(df, z, atr, t - 1, int(params["retest_bars"]), params["breakout_close_atr"])
        last_brk = brk[-1] if brk else None
        if last_brk and last_brk[1] == 1 and touched_from_above:
            v = 1.0 if trend_sign >= 0 else 0.6
            hits.append(BehaviorHit("pullback", 1, v, f"retest of broken resistance {zl} held as support", z))
            continue
        if last_brk and last_brk[1] == -1 and touched_from_below:
            v = -1.0 if trend_sign <= 0 else -0.6
            hits.append(BehaviorHit("pullback", -1, v, f"retest of broken support {zl} held as resistance", z))
            continue

        # Plain bounce off the zone with a close in the bounce direction.
        if z.role == "support" and touched_from_above and c[t] > o[t]:
            hits.append(BehaviorHit("bounce", 1, 0.4 + 0.4 * z.score / 100, f"bounce from support {zl}", z))
        elif z.role == "resistance" and touched_from_below and c[t] < o[t]:
            hits.append(BehaviorHit("bounce", -1, -(0.4 + 0.4 * z.score / 100), f"rejection from resistance {zl}", z))

    return sorted(hits, key=lambda x: -abs(x.value))
