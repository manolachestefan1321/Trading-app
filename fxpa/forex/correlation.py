"""Correlation guard: drop signals that duplicate currency exposure of a stronger signal.

A BUY on BASE/QUOTE is long BASE and short QUOTE. Two signals overlap when they hold the
same currency on the same side, e.g. BUY EURUSD + SELL USDCHF (both short USD) or
BUY EURUSD + BUY GBPUSD (both short USD). The strongest |score| survives.
"""

from __future__ import annotations

from dataclasses import dataclass

from fxpa.config import Instrument


@dataclass
class Exposure:
    symbol: str
    direction: int
    score: float
    legs: dict[str, int]


def exposure(inst: Instrument, direction: int) -> dict[str, int]:
    return {inst.base: direction, inst.quote: -direction}


def apply_correlation_guard(
    candidates: list[tuple[str, int, float]], instruments: dict[str, Instrument]
) -> tuple[list[str], list[str]]:
    """candidates: (symbol, direction, score). Returns (kept symbols, warning messages)."""
    ordered = sorted(candidates, key=lambda c: -abs(c[2]))
    kept: list[Exposure] = []
    warnings: list[str] = []
    for sym, d, score in ordered:
        legs = exposure(instruments[sym], d)
        clash = next((k for k in kept
                      for cur, side in legs.items() if k.legs.get(cur) == side), None)
        if clash is not None:
            shared = [c for c, s in legs.items() if clash.legs.get(c) == s]
            act = "BUY" if d > 0 else "SELL"
            kact = "BUY" if clash.direction > 0 else "SELL"
            warnings.append(f"{act} {sym} suppressed: duplicates {'/'.join(shared)} exposure of stronger "
                            f"{kact} {clash.symbol} ({clash.score:+.0f} vs {score:+.0f})")
            continue
        kept.append(Exposure(sym, d, score, legs))
    return [k.symbol for k in kept], warnings
