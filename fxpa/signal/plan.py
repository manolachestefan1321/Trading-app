"""Trade plan construction: entry, stop beyond the invalidation, target at the next opposing zone."""

from __future__ import annotations

from dataclasses import dataclass

from fxpa.analysis.zones import Zone
from fxpa.config import Instrument


@dataclass
class TradePlan:
    direction: int
    entry: float               # signal candle close (market order at the next open)
    stop: float
    target: float
    rr: float                  # reward / risk after spread and slippage
    risk_pips: float           # distance entry -> stop incl. costs
    reward_pips: float
    stop_reason: str
    target_reason: str
    lots: float | None = None
    risk_amount: float | None = None

    @property
    def side(self) -> str:
        return "BUY" if self.direction > 0 else "SELL"


def build_plan(
    direction: int,
    close: float,
    atr_t: float,
    inst: Instrument,
    zones: list[Zone],
    invalidation: list[tuple[float, str]],
    params: dict,
) -> tuple[TradePlan | None, str]:
    """Return (plan, reason). `invalidation` holds candidate (price, description) levels whose
    break invalidates the idea (pattern extreme, bounced zone edge, last swing)."""
    pip = inst.pip_size
    cost = (inst.spread_pips + inst.slippage_pips) * pip
    buffer = params["sl_atr_buffer"] * atr_t
    min_stop = 0.5 * atr_t

    if direction > 0:
        cands = [(p, d) for p, d in invalidation if p < close]
        if not cands:
            return None, "no invalidation level below price"
        base, why = min(cands, key=lambda x: x[0])
        stop = min(base - buffer, close - min_stop)
        eff_entry = close + cost
        targets = sorted((z for z in zones if z.low > close + params["min_tp_atr"] * atr_t),
                         key=lambda z: z.low)
        if not targets:
            return None, "no opposing zone above for a target"
        tz = targets[0]
        target = tz.low  # take profit at the near edge of the opposing zone
        risk = eff_entry - stop + inst.slippage_pips * pip
        reward = target - eff_entry
    else:
        cands = [(p, d) for p, d in invalidation if p > close]
        if not cands:
            return None, "no invalidation level above price"
        base, why = max(cands, key=lambda x: x[0])
        stop = max(base + buffer, close + min_stop)
        eff_entry = close - inst.slippage_pips * pip
        targets = sorted((z for z in zones if z.high < close - params["min_tp_atr"] * atr_t),
                         key=lambda z: -z.high)
        if not targets:
            return None, "no opposing zone below for a target"
        tz = targets[0]
        target = tz.high
        # A short is closed at the ask: spread is paid on exit.
        risk = stop - eff_entry + cost
        reward = eff_entry - target - inst.spread_pips * pip

    if risk <= 0:
        return None, "invalid stop distance"
    if risk > params["max_sl_atr"] * atr_t:
        return None, f"stop too wide ({risk / atr_t:.1f} ATR > {params['max_sl_atr']} ATR)"
    rr = reward / risk
    zone_desc = "/".join(tz.key_levels) if tz.origin == "key" else f"zone {inst.fmt(tz.low)}-{inst.fmt(tz.high)}"
    return TradePlan(
        direction=direction, entry=close, stop=stop, target=target, rr=float(rr),
        risk_pips=float(risk / pip), reward_pips=float(reward / pip),
        stop_reason=f"beyond {why} + {params['sl_atr_buffer']} ATR",
        target_reason=f"next {'resistance' if direction > 0 else 'support'} {zone_desc}",
    ), ""
