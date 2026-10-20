"""Event-driven backtest on 1H candles with no lookahead.

1. `generate_signals` replays history: for each bar t the engine sees only df.iloc[:t+1]
   (the candle that just closed and everything before it).
2. `simulate` turns signals into trades: market entry at the next candle's open, spread and
   slippage applied, stop / target checked on every later candle. If one candle touches both
   stop and target, the stop is assumed to have been hit first (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from fxpa.config import Instrument
from fxpa.forex.sessions import entry_allowed, session_label
from fxpa.signal.engine import Signal, SignalEngine, decide


@dataclass
class SignalRecord:
    """A scored bar kept for simulation (the analysis context is dropped to save memory)."""
    idx: int
    signal: Signal


def generate_signals(df: pd.DataFrame, engine: SignalEngine, start_idx: int | None = None,
                     progress: bool = True) -> list[SignalRecord]:
    """Evaluate every bar whose close falls in an allowed entry window.

    Bars outside the pair's sessions can never produce a trade, so they are skipped
    (purely a speed-up: the session gate would reject them anyway).
    """
    cfg, inst = engine.cfg, engine.inst
    warmup = max(int(cfg["analysis"]["lookback_bars"]) // 4, 200)
    start = warmup if start_idx is None else max(start_idx, warmup)
    ts = df["timestamp"]
    out: list[SignalRecord] = []
    for t in tqdm(range(start, len(df) - 1), desc=f"replay {inst.symbol}", unit="bar", disable=not progress,
                  mininterval=1.0):
        if not entry_allowed(ts.iat[t] + pd.Timedelta(hours=1), inst.sessions, cfg["sessions"])[0]:
            continue
        sig = engine.evaluate(df.iloc[:t + 1])
        out.append(SignalRecord(t, sig))
    return out


def simulate(df: pd.DataFrame, records: list[SignalRecord], inst: Instrument, cfg: dict,
             threshold: float | None = None, min_rr: float | None = None,
             idx_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """One position at a time per pair. Returns a trade log (R multiples incl. costs)."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    ts = df["timestamp"]
    pip = inst.pip_size
    spread, slip = inst.spread_pips * pip, inst.slippage_pips * pip
    max_hold = int(cfg["backtest"]["max_hold_bars"])
    n = len(df)
    trades = []
    busy_until = -1
    for rec in records:
        t = rec.idx
        if idx_range and not (idx_range[0] <= t < idx_range[1]):
            continue
        if t <= busy_until or t + 1 >= n:
            continue
        sig = decide(rec.signal, cfg, threshold, min_rr)
        if sig.action == "NO TRADE" or sig.plan is None:
            continue
        p = sig.plan
        d = p.direction
        e = t + 1
        # Entry at the next open: buys pay the spread (ask = bid + spread) plus slippage.
        entry = o[e] + spread + slip if d > 0 else o[e] - slip
        risk = (entry - p.stop) if d > 0 else (p.stop - entry)
        # Skip if the next open already gapped through the stop or the target.
        if risk <= 0 or (d > 0 and o[e] >= p.target) or (d < 0 and o[e] + spread <= p.target):
            continue
        exit_px, outcome, k = None, "time", e
        last = min(e + max_hold, n - 1)
        if idx_range:
            last = min(last, idx_range[1] - 1) if idx_range[1] < n else last
        for k in range(e, last + 1):
            if d > 0:
                hit_sl, hit_tp = l[k] <= p.stop, h[k] >= p.target
            else:  # shorts are closed at the ask
                hit_sl, hit_tp = h[k] + spread >= p.stop, l[k] + spread <= p.target
            if hit_sl:  # also when both are hit inside the same candle: stop first
                gap = o[k] < p.stop if d > 0 else o[k] + spread > p.stop
                fill = (o[k] if d > 0 else o[k] + spread) if gap else p.stop
                exit_px, outcome = fill - slip * d, "sl"
                break
            if hit_tp:
                exit_px, outcome = p.target, "tp"
                break
        if exit_px is None:
            k = last
            exit_px = c[k] if d > 0 else c[k] + spread
        r = (exit_px - entry) * d / risk
        trades.append({
            "symbol": inst.symbol, "signal_time": ts.iat[t], "entry_time": ts.iat[e], "exit_time": ts.iat[k] +
            pd.Timedelta(hours=1), "side": "BUY" if d > 0 else "SELL", "score": sig.score,
            "entry": entry, "stop": p.stop, "target": p.target, "exit": exit_px, "outcome": outcome,
            "r": r, "planned_rr": p.rr, "risk_pips": risk / pip, "bars_held": k - e + 1,
            "session": session_label(ts.iat[e], cfg["sessions"]), "explanation": sig.explanation,
        })
        busy_until = k
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame, risk_percent: float = 1.0, initial_equity: float = 10000.0) -> dict:
    """Win rate, profit factor, expectancy (R), max drawdown (R and %), average planned R:R."""
    if trades is None or trades.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0, "total_r": 0.0,
                "max_dd_r": 0.0, "max_dd_pct": 0.0, "avg_planned_rr": 0.0, "avg_win_r": 0.0,
                "avg_loss_r": 0.0, "final_equity": initial_equity}
    r = trades["r"].to_numpy(dtype=float)
    wins, losses = r[r > 0], r[r <= 0]
    cum = np.cumsum(r)
    dd_r = float(np.max(np.maximum.accumulate(np.r_[0.0, cum]) - np.r_[0.0, cum]))
    eq = equity_curve(trades, risk_percent, initial_equity)["equity"].to_numpy()
    peak = np.maximum.accumulate(eq)
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(r)),
        "win_rate": float(len(wins) / len(r)),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_r": float(r.mean()),
        "total_r": float(r.sum()),
        "max_dd_r": dd_r,
        "max_dd_pct": float(np.max((peak - eq) / peak) * 100),
        "avg_planned_rr": float(trades["planned_rr"].mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "final_equity": float(eq[-1]),
    }


def equity_curve(trades: pd.DataFrame, risk_percent: float = 1.0, initial_equity: float = 10000.0) -> pd.DataFrame:
    """Compounded equity when risking `risk_percent` of equity per trade."""
    if trades.empty:
        return pd.DataFrame({"time": [], "equity": []})
    growth = 1.0 + trades["r"].to_numpy(dtype=float) * risk_percent / 100.0
    eq = initial_equity * np.cumprod(growth)
    return pd.DataFrame({"time": trades["exit_time"].to_numpy(), "equity": eq})


def breakdown(trades: pd.DataFrame, by: str, risk_percent: float, initial_equity: float) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for key, g in trades.groupby(by):
        m = metrics(g.reset_index(drop=True), risk_percent, initial_equity)
        rows.append({by: key, **m})
    return pd.DataFrame(rows)
