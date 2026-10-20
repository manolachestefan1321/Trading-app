"""Application layer behind the CLI: signal, scan, backtest, plot, watch."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from fxpa.backtest.engine import generate_signals, metrics, simulate
from fxpa.backtest.report import save_report
from fxpa.backtest.walkforward import walk_forward
from fxpa.config import ROOT, Instrument
from fxpa.data.calendar import EconomicCalendar
from fxpa.data.common import floor_hour, market_closed, utcnow
from fxpa.data.manager import DataManager
from fxpa.forex.correlation import apply_correlation_guard
from fxpa.forex.news import NewsFilter
from fxpa.forex.pips import pip_value_per_lot, position_size
from fxpa.signal.engine import Signal, SignalEngine, explain

log = logging.getLogger("fxpa")
REPORTS = ROOT / "reports"
DISCLAIMER = "Educational tool. Signals are not financial advice; trading forex and gold carries a high risk of loss."


@dataclass
class Options:
    years: float
    account_size: float | None
    risk_percent: float
    source: str | None
    refresh: bool
    progress: bool = True


class App:
    def __init__(self, cfg: dict, instruments: dict[str, Instrument], opts: Options):
        self.cfg = cfg
        self.instruments = instruments
        self.opts = opts
        self.cfg["trade"]["risk_percent"] = opts.risk_percent
        self.dm = DataManager(cfg, instruments, source=opts.source, refresh=opts.refresh, progress=opts.progress)
        self.calendar = EconomicCalendar(cfg).load(fetch=bool(cfg["news"].get("enabled", True)))
        self.news = NewsFilter(self.calendar.events, cfg)
        self.warnings: list[str] = [self.calendar.warning] if self.calendar.warning else []
        self._frames: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ data
    def data(self, symbol: str, cached: bool = True) -> pd.DataFrame:
        """Candles for symbol, updated at most once per command (watch mode passes cached=False)."""
        if cached and symbol in self._frames:
            return self._frames[symbol]
        df = self.dm.get(symbol, self.opts.years)
        self._frames[symbol] = df
        res = self.dm.last_result.get(symbol)
        if res and res.stale:
            self.warnings.append(f"{symbol}: update failed, using cached data to {df['timestamp'].iloc[-1]:%Y-%m-%d %H:%M}")
        last_close = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        now = utcnow()
        if not market_closed(now) and now - last_close > timedelta(hours=3):
            self.warnings.append(f"{symbol}: latest candle closed {last_close:%Y-%m-%d %H:%M} UTC (data is stale)")
        return df

    def source_label(self, symbol: str) -> str:
        res = self.dm.last_result.get(symbol)
        if res and res.sources_used:
            return ", ".join(res.sources_used)
        meta = self.dm.cache.load_meta(symbol)
        return "cache (" + ", ".join(meta.get("sources", ["?"])) + ")"

    # ------------------------------------------------------------------ signals
    def signal(self, symbol: str, keep_context: bool = False) -> tuple[Signal, pd.DataFrame]:
        df = self.data(symbol)
        sig = SignalEngine(self.cfg, self.instruments[symbol], self.news).evaluate(df, keep_context=keep_context)
        self._size(sig)
        return sig, df

    def _size(self, sig: Signal) -> None:
        if sig.plan is None or not self.opts.account_size:
            return
        inst = self.instruments[sig.symbol]
        pv = pip_value_per_lot(inst, sig.close, self.dm.latest_prices(), self.cfg["trade"]["account_currency"])
        sig.plan.lots = position_size(self.opts.account_size, self.opts.risk_percent, sig.plan.risk_pips, pv)
        sig.plan.risk_amount = self.opts.account_size * self.opts.risk_percent / 100

    def scan(self, symbols: list[str]) -> tuple[list[Signal], list[str]]:
        sigs: list[Signal] = []
        for sym in symbols:
            try:
                sigs.append(self.signal(sym)[0])
            except Exception as exc:  # one broken pair must not stop the scan
                self.warnings.append(f"{sym}: {exc.__class__.__name__}: {str(exc)[:200]}")
        guard_msgs: list[str] = []
        if self.cfg["correlation"].get("enabled", True):
            actives = [(s.symbol, 1 if s.action == "BUY" else -1, s.score) for s in sigs if s.action != "NO TRADE"]
            kept, guard_msgs = apply_correlation_guard(actives, self.instruments)
            for s in sigs:
                if s.action != "NO TRADE" and s.symbol not in kept:
                    s.reasons.append("correlation guard: duplicates exposure of a stronger signal")
                    s.action = "NO TRADE"
                    s.explanation = explain(s)
        return sorted(sigs, key=lambda s: -s.score), guard_msgs

    # ------------------------------------------------------------------ backtest
    def backtest(self, symbol: str) -> Path:
        df = self.data(symbol)
        inst = self.instruments[symbol]
        engine = SignalEngine(self.cfg, inst, self.news)
        log.info("replaying %d candles for %s (no lookahead: each bar sees only closed candles)", len(df), symbol)
        recs = generate_signals(df, engine, progress=self.opts.progress)
        trades = simulate(df, recs, inst, self.cfg)
        wf = walk_forward(df, recs, inst, self.cfg)
        notes = []
        cov = self.calendar.coverage()
        if cov is None:
            notes.append("no economic calendar history: the news blackout filter was inactive in this backtest")
        else:
            notes.append(f"news blackout applied only where calendar history exists ({cov[0]:%Y-%m-%d} → {cov[1]:%Y-%m-%d}); "
                         "the free feed covers the current week only, so earlier trades were not news-filtered")
        if df["volume"].sum() == 0:
            notes.append("source provides no tick volume: the 5% volume factor was neutral")
        out = save_report(inst, df, trades, wf, self.cfg, REPORTS, self.source_label(symbol), notes)
        self.last_backtest = (metrics(trades, self.opts.risk_percent, float(self.cfg["backtest"]["initial_equity"])), wf)
        return out

    # ------------------------------------------------------------------ plot
    def plot(self, symbol: str, bars: int = 150) -> Path:
        from fxpa.viz.plot import plot_signal

        sig, df = self.signal(symbol, keep_context=True)
        engine = SignalEngine(self.cfg, self.instruments[symbol], self.news)
        history = [engine.evaluate(df.iloc[:t + 1]) for t in range(max(300, len(df) - bars), len(df) - 1)]
        out = REPORTS / "charts" / f"{symbol}_{sig.bar_time:%Y%m%d_%H%M}.png"
        return plot_signal(sig, out, self.cfg, bars=bars, history=history)

    # ------------------------------------------------------------------ watch
    def watch(self, symbols: list[str], printer, delay_s: int = 45) -> None:
        """Evaluate on every new 1H candle close and print new BUY/SELL signals."""
        seen: set[tuple[str, pd.Timestamp]] = set()
        self.dm.progress = False
        while True:
            self._frames.clear()  # force a data update on every new candle
            sigs, guard = self.scan(symbols)
            printer(sigs, guard, [s for s in sigs if (s.symbol, s.bar_time) not in seen and s.action != "NO TRADE"])
            seen |= {(s.symbol, s.bar_time) for s in sigs}
            self.warnings.clear()
            nxt = floor_hour(utcnow()) + timedelta(hours=1, seconds=delay_s)
            log.info("next check at %s UTC (after the candle close)", nxt.strftime("%H:%M:%S"))
            while utcnow() < nxt:
                time.sleep(min(30.0, max(1.0, (nxt - utcnow()).total_seconds())))
