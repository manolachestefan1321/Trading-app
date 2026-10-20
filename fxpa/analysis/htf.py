"""Higher-timeframe bias: Daily trend filter with a 4H CHoCH override."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fxpa.analysis.atr import atr
from fxpa.analysis.structure import StructureState, analyze_structure
from fxpa.analysis.swings import detect_swings
from fxpa.data.resample import resample


@dataclass
class HTFBias:
    daily: StructureState
    h4: StructureState
    h4_choch: int = 0                 # +1 / -1 if a 4H CHoCH happened recently, else 0
    daily_bars: int = 0

    def allows(self, direction: int, allow_range: bool = False) -> tuple[bool, str]:
        d = self.daily.sign
        if d == direction:
            return True, f"Daily {self.daily.trend}trend"
        if self.h4_choch == direction:
            return True, f"4H CHoCH {'bullish' if direction > 0 else 'bearish'} overrides Daily {self.daily.trend}"
        if d == 0 and allow_range:
            return True, "Daily range (allowed by config)"
        side = "BUY" if direction > 0 else "SELL"
        return False, f"{side} against Daily {self.daily.trend}{'trend' if d else ''} without 4H CHoCH"


def _structure_for(bars: pd.DataFrame, window: int, min_swing_atr: float, full_count: int) -> StructureState:
    if len(bars) < 2 * window + 3:
        return StructureState()
    a = atr(bars, 14).to_numpy()
    sw = detect_swings(bars, a, window=window, min_swing_atr=min_swing_atr)
    return analyze_structure(sw, bars, full_count)


def htf_bias(df_1h: pd.DataFrame, cfg: dict) -> HTFBias:
    """Daily and 4H structure from completed higher-timeframe bars only."""
    p = cfg["htf"]
    full = int(cfg["structure"]["strength_full_count"])
    daily = resample(df_1h, "D")
    h4 = resample(df_1h, "4h")
    d_state = _structure_for(daily, int(p["daily_fractal_window"]), float(p["min_swing_atr"]), full)
    h_state = _structure_for(h4, int(p["h4_fractal_window"]), float(p["min_swing_atr"]), full)
    choch = 0
    ev = h_state.last_event("CHoCH", since_idx=len(h4) - int(p["choch_lookback_h4_bars"]))
    if ev is not None:
        choch = ev.direction
    return HTFBias(daily=d_state, h4=h_state, h4_choch=choch, daily_bars=len(daily))
