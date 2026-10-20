"""Signal engine: runs every analysis module on data up to the last closed candle,
scores the setup from -100 to +100, applies filters and builds a trade plan.

The engine only ever sees the frame it is given. The live CLI passes the full history;
the backtester passes `df.iloc[:t+1]` for each bar t, so there is no lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxpa.analysis.atr import atr as compute_atr
from fxpa.analysis.behavior import BehaviorHit, evaluate_behavior
from fxpa.analysis.candles import PatternHit, patterns_at
from fxpa.analysis.htf import HTFBias, htf_bias
from fxpa.analysis.levels import KeyLevel, key_levels, round_numbers
from fxpa.analysis.structure import StructureState, analyze_structure
from fxpa.analysis.swings import detect_swings
from fxpa.analysis.volume import volume_component, volume_ratio
from fxpa.analysis.zones import Zone, build_zones
from fxpa.config import Instrument
from fxpa.data.resample import period_key
from fxpa.data.validate import add_gap_flags
from fxpa.forex.news import NewsFilter
from fxpa.forex.sessions import asian_range, entry_allowed, london_asian_event, session_label
from fxpa.signal.plan import TradePlan, build_plan

COMPONENTS = ("structure", "location", "candle", "behavior", "volume")


@dataclass
class Context:
    """Everything the chart needs to draw the analysis (window-relative indices)."""
    window: pd.DataFrame
    atr: np.ndarray
    swings: pd.DataFrame
    zones: list[Zone]
    levels: list[KeyLevel]
    structure: StructureState
    htf: HTFBias | None
    patterns: list[PatternHit]
    behavior: list[BehaviorHit]


@dataclass
class Signal:
    symbol: str
    bar_time: pd.Timestamp             # open time of the signal candle
    entry_time: pd.Timestamp           # its close = earliest possible entry
    close: float
    atr: float
    score: float
    direction: int                     # sign of the raw score
    components: dict[str, float]
    factors: list[str]                 # independent factors agreeing with `direction`
    gates: dict[str, tuple[bool, str]] # session / news / htf
    plan: TradePlan | None
    plan_reason: str
    session: str
    facts: dict[str, str] = field(default_factory=dict)
    action: str = "NO TRADE"
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    context: Context | None = None

    @property
    def confluence(self) -> int:
        return len(self.factors)


class SignalEngine:
    def __init__(self, cfg: dict, inst: Instrument, news: NewsFilter | None = None):
        self.cfg = cfg
        self.inst = inst
        self.news = news or NewsFilter(None, cfg)
        self._htf_cache: dict = {}
        self._levels_cache: dict = {}

    # ------------------------------------------------------------------ helpers
    def _htf(self, df: pd.DataFrame) -> HTFBias | None:
        """HTF bias from completed 4H/Daily bars, cached per 4H period.

        The window is "the last `htf_lookback_bars` 1H candles before the current 4H period",
        a pure function of past data, so cached and uncached evaluation are identical.
        """
        if not self.cfg["htf"].get("enabled", True):
            return None
        span = int(self.cfg["analysis"]["htf_lookback_bars"])
        tail = df.tail(span + 8)
        close_ts = tail["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        cur_key = period_key(pd.Series([close_ts]), "4h").iloc[0]
        complete = tail[period_key(tail["timestamp"], "4h").to_numpy() < cur_key]
        if complete.empty:
            return None
        fingerprint = (cur_key, complete["timestamp"].iloc[-1], float(complete["close"].iloc[-1]))
        hit = self._htf_cache.get(fingerprint)
        if hit is not None:
            return hit
        bias = htf_bias(complete.tail(span).reset_index(drop=True), self.cfg)
        if len(self._htf_cache) > 64:
            self._htf_cache.clear()
        self._htf_cache[fingerprint] = bias
        return bias

    def _levels(self, w: pd.DataFrame) -> list[KeyLevel]:
        """Key levels. The period-based ones (PDH/PDL/PDC, PWH/PWL, weekly open, Asian range)
        only change when a day/week/Asian session completes, so they are cached on that."""
        close_ts = w["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        day_key = period_key(pd.Series([close_ts]), "D").iloc[0]
        asia_done = close_ts.hour >= int(self.cfg["sessions"]["asia"][1])
        fp = (day_key, close_ts.normalize(), asia_done, w["timestamp"].iloc[0] <= close_ts - pd.Timedelta(days=10))
        static = self._levels_cache.get(fp)
        if static is None:
            static = [lv for lv in key_levels(w, (), self.cfg["sessions"]) if not lv.name.startswith("round")]
            if len(self._levels_cache) > 64:
                self._levels_cache.clear()
            self._levels_cache[fp] = static
        each = int(self.cfg["key_levels"].get("round_numbers_each_side", 2))
        return static + round_numbers(float(w["close"].iloc[-1]), self.inst.round_steps, each)

    def _fmt(self, p: float) -> str:
        return self.inst.fmt(p)

    # ------------------------------------------------------------------ main
    def evaluate(self, df: pd.DataFrame, keep_context: bool = False) -> Signal:
        cfg, inst = self.cfg, self.inst
        lookback = int(cfg["analysis"]["lookback_bars"])
        w = add_gap_flags(df.tail(lookback).reset_index(drop=True))
        t = len(w) - 1
        bar_time = w["timestamp"].iat[t]
        entry_time = bar_time + pd.Timedelta(hours=1)
        a_arr = compute_atr(w, int(cfg["analysis"]["atr_period"])).to_numpy()
        a = float(a_arr[t]) if np.isfinite(a_arr[t]) and a_arr[t] > 0 else float(np.nanmean(a_arr) or 1e-9)
        o, h, l, c = (w[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        close = float(c[t])
        facts: dict[str, str] = {}

        # 1. Gates that don't depend on the setup.
        sess_ok, sess_txt = entry_allowed(entry_time, inst.sessions, cfg["sessions"])
        news_blocked, news_txt = self.news.blackout(inst, entry_time)
        gates = {"session": (sess_ok, sess_txt),
                 "news": (not news_blocked, f"news blackout: {news_txt}" if news_blocked else "")}
        facts["session"] = session_label(entry_time, cfg["sessions"])
        facts["news"] = self.news.next_event_text(inst, entry_time) if not news_blocked else news_txt

        # 2. Swings and 1H structure.
        sw_p = cfg["swings"]
        swings = detect_swings(w, a_arr, int(sw_p["fractal_window"]), float(sw_p["min_swing_atr"]))
        swings = swings[swings["confirmed_idx"] <= t].reset_index(drop=True)
        st = analyze_structure(swings, w, int(cfg["structure"]["strength_full_count"]))
        ev_since = t - int(cfg["structure"]["event_lookback_bars"])
        choch, bos = st.last_event("CHoCH", ev_since), st.last_event("BOS", ev_since)
        last_ev = st.events[-1] if st.events else None
        if last_ev is not None and last_ev.idx < ev_since:
            last_ev = None

        # 3. Higher-timeframe bias.
        htf = self._htf(df)
        daily_sign = htf.daily.sign if htf else 0

        # 4. Key levels and S/R zones.
        levels = self._levels(w) if cfg["key_levels"].get("enabled", True) else []
        zp = cfg["zones"]
        zones = build_zones(w, swings, a, zp, levels, t)
        near = float(zp["near_atr"]) * a

        # 5. Location: are we at support or resistance right now?
        recent_low, recent_high = float(l[max(0, t - 2):t + 1].min()), float(h[max(0, t - 2):t + 1].max())
        # A zone's role already accounts for price sitting inside it (see build_zones).
        at_sup = [z for z in zones if z.role == "support"
                  and recent_low <= z.high + near and close >= z.low - near]
        at_res = [z for z in zones if z.role == "resistance"
                  and recent_high >= z.low - near and close <= z.high + near]
        sup = max(at_sup, key=lambda z: z.score, default=None)
        res = max(at_res, key=lambda z: z.score, default=None)
        loc = ((sup.score if sup else 0.0) - (res.score if res else 0.0)) / 100.0
        if loc > 0:
            above = [z.low - close for z in zones if z.low > close and z is not sup]
            if above and min(above) < float(zp["room_min_atr"]) * a:
                loc *= 0.5
                facts["room"] = "little room to the next resistance"
        elif loc < 0:
            below = [close - z.high for z in zones if z.high < close and z is not res]
            if below and min(below) < float(zp["room_min_atr"]) * a:
                loc *= 0.5
                facts["room"] = "little room to the next support"

        # 6. Candlestick confirmation (only at zones or swing points).
        cp = cfg["candles"]
        prox = int(cp["swing_proximity_bars"]) * 2
        lo_ext = float(l[max(0, t - prox):t + 1].min())
        hi_ext = float(h[max(0, t - prox):t + 1].max())
        valid: list[tuple[PatternHit, int, str]] = []  # (pattern, direction, where)
        for p in patterns_at(w, a_arr, t, cp):
            d = p.direction if p.direction != 0 else int(np.sign(loc))
            if d == 0:
                continue
            if d > 0:
                z = next((z for z in zones if z.low - near <= p.low <= z.high + near), None)
                where = f"at {z.label(self._fmt)}" if z else ("at swing low" if p.low <= lo_ext else None)
            else:
                z = next((z for z in zones if z.low - near <= p.high <= z.high + near), None)
                where = f"at {z.label(self._fmt)}" if z else ("at swing high" if p.high >= hi_ext else None)
            if where:
                valid.append((p, d, where))
        best_pat = max(valid, key=lambda x: x[0].strength, default=None)
        candle = best_pat[1] * best_pat[0].strength if best_pat else 0.0

        # 7. Bounce / breakout behavior, plus London sweeps of the Asian range.
        beh_hits = evaluate_behavior(w, zones, a_arr, t, st.sign, cfg["behavior"], float(zp["near_atr"]),
                                     self._fmt)
        behavior = beh_hits[0].value if beh_hits else 0.0
        rng = asian_range(w, t, cfg["sessions"])
        lev = london_asian_event(w, t, rng, a, cfg["sessions"], tuple(cfg["behavior"]["london_sweep_hours"]))
        if lev is not None:
            behavior = float(np.clip(behavior + (0.4 if lev.kind == "sweep" else 0.3) * lev.direction, -1, 1))
            facts["london"] = lev.text

        # 8. Structure component (1H trend, recent BOS/CHoCH, Daily alignment).
        s1 = st.sign * st.strength / 100.0
        if last_ev is not None and last_ev.kind == "CHoCH" and last_ev.direction != np.sign(s1):
            s1 = 0.5 * last_ev.direction
        elif last_ev is not None and last_ev.kind == "BOS" and last_ev.direction == np.sign(s1):
            s1 += 0.15 * last_ev.direction
        structure_1h = float(np.clip(s1, -1, 1))
        structure = float(np.clip(structure_1h + 0.25 * daily_sign, -1, 1))

        # 9. Weighted score; tick volume confirms the provisional direction.
        wts = cfg["scoring"]["weights"]
        comps = {"structure": structure, "location": float(np.clip(loc, -1, 1)),
                 "candle": float(candle), "behavior": float(behavior)}
        prelim = sum(wts[k] * v for k, v in comps.items())
        v_ratio = volume_ratio(w, t, int(cfg["volume"]["rolling"]))
        comps["volume"] = volume_component(v_ratio, int(np.sign(prelim)))
        score = float(np.clip(100.0 * sum(wts[k] * comps[k] for k in COMPONENTS), -100, 100))
        direction = int(np.sign(score))

        # 10. Confluence: independent factors pointing the same way.
        fmin = float(cfg["scoring"]["factor_min"])
        factors = []
        if direction:
            checks = {"1H structure": structure_1h, "Daily trend": float(daily_sign),
                      "S/R location": comps["location"], "candlestick": comps["candle"],
                      "behavior": comps["behavior"], "tick volume": comps["volume"]}
            factors = [k for k, v in checks.items() if v * direction >= fmin]

        # 11. HTF gate for the candidate direction.
        if htf is not None and direction:
            gates["htf"] = htf.allows(direction, bool(cfg["htf"].get("allow_daily_range", False)))
        elif direction:
            gates["htf"] = (True, "HTF filter disabled")

        # 12. Trade plan.
        plan, plan_reason = None, ""
        if direction and abs(score) >= float(cfg["scoring"]["plan_min_score"]):
            inval: list[tuple[float, str]] = []
            if best_pat and best_pat[1] == direction:
                p = best_pat[0]
                inval.append((p.low if direction > 0 else p.high, best_pat[0].name))
            for z in ([sup] if direction > 0 else [res]) + [b.zone for b in beh_hits[:1] if b.direction == direction]:
                if z is not None:
                    inval.append((z.low if direction > 0 else z.high, "zone"))
            if not inval:  # fall back to the last swing that invalidates the idea
                kind = "L" if direction > 0 else "H"
                prev = swings[swings["kind"] == kind]
                if len(prev):
                    inval.append((float(prev["price"].iloc[-1]), "last swing " + ("low" if direction > 0 else "high")))
            plan, plan_reason = build_plan(direction, close, a, inst, zones, inval, cfg["trade"])

        # Facts for the explanation.
        if htf is not None:
            facts["daily"] = f"Daily {htf.daily.trend}{'trend' if htf.daily.sign else ''}"
        facts["structure"] = f"1H {st.describe()}"
        if last_ev is not None:
            facts["event"] = f"{'bullish' if last_ev.direction > 0 else 'bearish'} {last_ev.kind} at {self._fmt(last_ev.level)}"
        if best_pat:
            facts["pattern"] = f"{best_pat[0].name} {best_pat[2]}"
        zone_for_text = sup if direction > 0 else res if direction < 0 else None
        if zone_for_text is not None:
            facts["zone"] = zone_for_text.label(self._fmt)
        if beh_hits:
            facts["behavior"] = beh_hits[0].text
        facts["volume"] = (f"{cfg['volume']['label']} {v_ratio:.1f}x avg" if v_ratio is not None
                           else "no tick volume from this source")
        if swings["across_gap"].tail(2).any() if len(swings) else False:
            facts["gap"] = "latest swing spans a weekend/data gap"

        sig = Signal(
            symbol=inst.symbol, bar_time=bar_time, entry_time=entry_time, close=close, atr=a,
            score=round(score, 1), direction=direction, components={k: round(v, 3) for k, v in comps.items()},
            factors=factors, gates=gates, plan=plan, plan_reason=plan_reason, session=facts["session"],
            facts=facts,
        )
        if keep_context:
            sig.context = Context(w, a_arr, swings, zones, levels, st, htf,
                                  patterns_at(w, a_arr, t, cp), beh_hits)
        decide(sig, cfg)
        return sig


def decide(sig: Signal, cfg: dict, threshold: float | None = None, min_rr: float | None = None) -> Signal:
    """Turn a scored setup into BUY / SELL / NO TRADE. Threshold and min R:R can be overridden
    (the walk-forward optimizer does this without re-running the analysis)."""
    sc = cfg["scoring"]
    buy_thr = threshold if threshold is not None else float(sc["buy_threshold"])
    sell_thr = -threshold if threshold is not None else float(sc["sell_threshold"])
    rr_min = min_rr if min_rr is not None else float(cfg["trade"]["min_rr"])
    reasons: list[str] = []
    d = 1 if sig.score >= buy_thr else -1 if sig.score <= sell_thr else 0
    if d == 0:
        reasons.append(f"score {sig.score:+.0f} inside ±{buy_thr:.0f}")
    for name in ("session", "news"):
        ok, txt = sig.gates.get(name, (True, ""))
        if not ok:
            reasons.append(txt)
    if d:
        ok, txt = sig.gates.get("htf", (True, ""))
        if not ok:
            reasons.append(txt)
        if sig.confluence < int(sc["min_confluence"]):
            reasons.append(f"only {sig.confluence} confluence factor(s) (< {sc['min_confluence']})")
        if sig.plan is None:
            reasons.append(sig.plan_reason or "no valid trade plan")
        elif sig.plan.rr < rr_min:
            reasons.append(f"R:R {sig.plan.rr:.2f} < {rr_min}")
    sig.reasons = reasons
    sig.action = ("BUY" if d > 0 else "SELL") if d and not reasons else "NO TRADE"
    sig.explanation = explain(sig)
    return sig


def explain(sig: Signal) -> str:
    f = sig.facts
    pattern = f.get("pattern")
    # The pattern text already names its zone; only add the zone separately when it doesn't.
    zone = f.get("zone") if not (pattern and " at " in pattern and "swing" not in pattern) else None
    parts = [f.get("daily"), f.get("structure"), f.get("event"), pattern, zone, f.get("behavior"),
             f.get("london"), f.get("room"), f"{f.get('session')} session", f.get("news"), f.get("volume"),
             f.get("gap")]
    body = ", ".join(p for p in parts if p)
    if sig.action != "NO TRADE" and sig.plan:
        body += f", R:R {sig.plan.rr:.1f}, risk {sig.plan.risk_pips:.0f} pips"
        return f"{sig.action} {sig.symbol}: {body}"
    why = "; ".join(sig.reasons) if sig.reasons else "no setup"
    return f"NO TRADE {sig.symbol} (score {sig.score:+.0f}): {why}. Context: {body}"
