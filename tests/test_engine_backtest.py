"""Scoring engine, trade simulation rules, walk-forward, and the no-lookahead guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import bars, make_candles
from fxpa.backtest.engine import SignalRecord, generate_signals, metrics, simulate
from fxpa.backtest.walkforward import walk_forward
from fxpa.signal.engine import Signal, SignalEngine, decide
from fxpa.signal.plan import TradePlan


def _signal(direction=1, score=75.0, rr=2.0, entry=1.1000, stop=1.0980, target=1.1040, factors=3):
    plan = TradePlan(direction, entry, stop, target, rr, 20, 40, "test", "test")
    return Signal(symbol="EURUSD", bar_time=pd.Timestamp("2024-01-02 08:00", tz="UTC"),
                  entry_time=pd.Timestamp("2024-01-02 09:00", tz="UTC"), close=entry, atr=0.001,
                  score=score, direction=direction, components={}, factors=["a", "b", "c", "d"][:factors],
                  gates={"session": (True, "london"), "news": (True, ""), "htf": (True, "ok")},
                  plan=plan, plan_reason="", session="london")


# ---------------------------------------------------------------- decision rules
def test_decide_thresholds_and_filters(cfg):
    assert decide(_signal(score=75), cfg).action == "BUY"
    assert decide(_signal(direction=-1, score=-75, stop=1.1020, target=1.0960), cfg).action == "SELL"
    s = decide(_signal(score=55), cfg)
    assert s.action == "NO TRADE" and "inside" in s.reasons[0]
    assert decide(_signal(rr=1.2), cfg).action == "NO TRADE"                # R:R < 1.5
    assert decide(_signal(factors=2), cfg).action == "NO TRADE"             # confluence < 3
    sig = _signal()
    sig.gates["news"] = (False, "news blackout: USD NFP")
    assert decide(sig, cfg).action == "NO TRADE"
    sig = _signal()
    sig.gates["htf"] = (False, "BUY against Daily downtrend")
    assert decide(sig, cfg).action == "NO TRADE"
    assert decide(_signal(score=55), cfg, threshold=50).action == "BUY"     # optimizer override


# ---------------------------------------------------------------- simulation rules
def _sim(cfg, instruments, rows, sig):
    df = bars(rows, start="2024-01-02 08:00")
    return simulate(df, [SignalRecord(0, sig)], instruments["EURUSD"], cfg)


def test_same_candle_stop_and_target_counts_as_loss(cfg, instruments):
    rows = [(1.1000, 1.1002, 1.0998, 1.1000), (1.1000, 1.1050, 1.0970, 1.1000)]  # next bar hits both
    tr = _sim(cfg, instruments, rows, _signal())
    assert tr["outcome"].iloc[0] == "sl" and tr["r"].iloc[0] < -1.0  # costs make it worse than -1R


def test_target_hit_and_costs(cfg, instruments):
    rows = [(1.1000, 1.1002, 1.0998, 1.1000), (1.1000, 1.1010, 1.0995, 1.1008), (1.1008, 1.1045, 1.1005, 1.1040)]
    tr = _sim(cfg, instruments, rows, _signal())
    t = tr.iloc[0]
    inst = instruments["EURUSD"]
    expected_entry = 1.1000 + (inst.spread_pips + inst.slippage_pips) * inst.pip_size
    assert t["outcome"] == "tp" and t["entry"] == pytest.approx(expected_entry)
    assert t["r"] == pytest.approx((1.1040 - expected_entry) / (expected_entry - 1.0980))
    assert t["r"] < 2.0  # costs reduce R below the gross 2R


def test_short_closed_at_ask(cfg, instruments):
    sig = _signal(direction=-1, score=-75, stop=1.1020, target=1.0960)
    inst = instruments["EURUSD"]
    spread = inst.spread_pips * inst.pip_size
    # bid high reaches 1.1020 - spread/2: ask = bid + spread crosses the stop
    rows = [(1.1000, 1.1002, 1.0998, 1.1000), (1.1000, 1.1020 - spread / 2, 1.0995, 1.1001)]
    tr = _sim(cfg, instruments, rows, sig)
    assert tr["outcome"].iloc[0] == "sl"


def test_time_stop(cfg, instruments):
    cfg["backtest"]["max_hold_bars"] = 3
    rows = [(1.1000, 1.1002, 1.0998, 1.1000)] * 8
    tr = _sim(cfg, instruments, rows, _signal())
    assert tr["outcome"].iloc[0] == "time" and tr["bars_held"].iloc[0] == 4


def test_metrics():
    tr = pd.DataFrame({"r": [2.0, -1.0, -1.0, 3.0], "planned_rr": [2, 2, 2, 3],
                       "exit_time": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")})
    m = metrics(tr, 1.0, 10000)
    assert m["trades"] == 4 and m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(2.5) and m["expectancy_r"] == pytest.approx(0.75)
    assert m["max_dd_r"] == pytest.approx(2.0)


# ---------------------------------------------------------------- engine
def test_engine_produces_complete_signal(cfg, instruments):
    df = make_candles(1500, seed=7)
    sig = SignalEngine(cfg, instruments["EURUSD"]).evaluate(df, keep_context=True)
    assert -100 <= sig.score <= 100
    assert set(sig.components) == {"structure", "location", "candle", "behavior", "volume"}
    assert sig.action in {"BUY", "SELL", "NO TRADE"}
    assert sig.explanation.startswith(sig.action)
    assert sig.entry_time == df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    assert sig.context is not None and len(sig.context.zones) > 0
    assert (sig.context.swings["confirmed_idx"] <= len(sig.context.window) - 1).all()


def test_volume_weight_is_capped(cfg):
    assert cfg["scoring"]["weights"]["volume"] <= 0.05


# ---------------------------------------------------------------- no lookahead
def _fingerprint(rec: SignalRecord):
    s = rec.signal
    plan = None if s.plan is None else (round(s.plan.stop, 8), round(s.plan.target, 8), round(s.plan.rr, 6))
    return (rec.idx, s.score, tuple(sorted(s.components.items())), tuple(s.factors), plan, s.action)


def test_no_lookahead_future_data_cannot_change_past_signals(cfg, instruments):
    df = make_candles(1300, seed=11)
    cut = 1150
    poisoned = df.copy()
    rng = np.random.default_rng(0)
    # Wildly different future after the cut (prices x1.2, new volume).
    for col in ("open", "high", "low", "close"):
        poisoned.loc[cut:, col] = poisoned.loc[cut:, col] * 1.2
    poisoned.loc[cut:, "volume"] = rng.integers(1, 10, len(df) - cut)

    inst = instruments["EURUSD"]
    a = generate_signals(df, SignalEngine(cfg, inst), start_idx=1000, progress=False)
    b = generate_signals(poisoned, SignalEngine(cfg, inst), start_idx=1000, progress=False)
    pa = [_fingerprint(r) for r in a if r.idx < cut]
    pb = [_fingerprint(r) for r in b if r.idx < cut]
    assert pa and pa == pb


def test_backtest_signal_equals_live_signal_at_that_bar(cfg, instruments):
    df = make_candles(1300, seed=5)
    inst = instruments["EURUSD"]
    recs = generate_signals(df, SignalEngine(cfg, inst), start_idx=1200, progress=False)
    for rec in recs[::7]:
        live = SignalEngine(cfg, inst).evaluate(df.iloc[:rec.idx + 1])  # fresh engine, no caches
        assert live.score == rec.signal.score
        assert live.components == rec.signal.components


def test_walk_forward_runs(cfg, instruments):
    df = make_candles(1100, seed=9)
    inst = instruments["EURUSD"]
    cfg["backtest"]["min_trades"] = 1
    recs = generate_signals(df, SignalEngine(cfg, inst), progress=False)
    wf = walk_forward(df, recs, inst, cfg)
    assert len(wf.grid) == len(cfg["backtest"]["grid"]["threshold"]) * len(cfg["backtest"]["grid"]["min_rr"])
    assert wf.split_time == df["timestamp"].iat[int(len(df) * 0.6)]
