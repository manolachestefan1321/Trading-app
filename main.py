#!/usr/bin/env python3
"""forex-price-action-signal CLI. One command downloads/updates data and prints signals.

    python main.py --symbol EURUSD             latest signal + explanation
    python main.py --all                       scan every configured pair
    python main.py --symbol GBPUSD --backtest  2-year backtest + walk-forward report
    python main.py --symbol EURUSD --plot      save a chart
    python main.py --watch                     re-run on every new 1H candle close
"""

from __future__ import annotations

import argparse
import logging
import sys

from fxpa.app import DISCLAIMER, App, Options
from fxpa.config import load_config, load_env, load_instruments
from fxpa.signal.engine import COMPONENTS, Signal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical price action BUY / SELL / NO TRADE signals for 1H forex.")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--symbol", help="pair from instruments.yaml, e.g. EURUSD")
    target.add_argument("--all", action="store_true", help="scan all configured pairs")
    p.add_argument("--backtest", action="store_true", help="run the backtest + walk-forward for --symbol")
    p.add_argument("--plot", action="store_true", help="save a chart for --symbol")
    p.add_argument("--watch", action="store_true", help="evaluate on every new 1H candle close")
    p.add_argument("--years", type=float, default=None, help="history to download/keep (default from config: 2)")
    p.add_argument("--account-size", type=float, default=None, help="account size for position sizing")
    p.add_argument("--risk-percent", type=float, default=None, help="risk per trade in %% (default 1)")
    p.add_argument("--source", choices=["dukascopy", "yfinance", "oanda", "twelvedata"],
                   help="force one data source instead of the fallback chain")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and re-download everything")
    p.add_argument("--bars", type=int, default=150, help="candles shown on the chart")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--instruments", default=None, help="path to instruments.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    if (a.backtest or a.plot) and not a.symbol:
        p.error("--backtest and --plot need --symbol")
    if not (a.symbol or a.all or a.watch):
        p.error("give --symbol, --all or --watch")
    return a


# ---------------------------------------------------------------------- printing
def _sign(v: float) -> str:
    return f"{v:+.2f}"


def print_signal(sig: Signal, app: App) -> None:
    inst = app.instruments[sig.symbol]
    print("=" * 96)
    print(f"{sig.symbol}  candle {sig.bar_time:%Y-%m-%d %H:%M} UTC (closed {sig.entry_time:%H:%M})   "
          f"data: {app.source_label(sig.symbol)}")
    print(f"Signal: {sig.action}   score {sig.score:+.0f}   confluence {sig.confluence} "
          f"({', '.join(sig.factors) or 'none'})")
    print("Components: " + " | ".join(f"{k} {_sign(sig.components[k])}" for k in COMPONENTS)
          + "   (volume = tick volume, weak confirmation)")
    if sig.plan is not None:
        p = sig.plan
        tag = "" if sig.action != "NO TRADE" else "  [not taken]"
        print(f"Plan{tag}: {p.side} entry {inst.fmt(p.entry)}  SL {inst.fmt(p.stop)} ({p.stop_reason})  "
              f"TP {inst.fmt(p.target)} ({p.target_reason})")
        print(f"      R:R {p.rr:.2f}   risk {p.risk_pips:.1f} pips   reward {p.reward_pips:.1f} pips")
        if p.lots is not None:
            print(f"      position size {p.lots:.2f} lots (risking {p.risk_amount:,.2f} = "
                  f"{app.opts.risk_percent:g}% of {app.opts.account_size:,.0f})")
    print(sig.explanation)


def print_table(sigs: list[Signal], guard: list[str], app: App) -> None:
    print(f"{'pair':<8}{'action':<10}{'score':>6}  {'conf':>4}  {'R:R':>5}  {'risk':>7}  {'session':<9} why / setup")
    print("-" * 110)
    for s in sigs:
        rr = f"{s.plan.rr:.1f}" if s.plan else "-"
        risk = f"{s.plan.risk_pips:.0f}p" if s.plan else "-"
        why = s.explanation.split(": ", 1)[1] if s.action != "NO TRADE" else "; ".join(s.reasons[:2])
        print(f"{s.symbol:<8}{s.action:<10}{s.score:>+6.0f}  {s.confluence:>4}  {rr:>5}  {risk:>7}  "
              f"{s.session:<9} {why[:80]}")
    for g in guard:
        print(f"⚠ correlation guard: {g}")


def print_warnings(app: App) -> None:
    for w in dict.fromkeys(app.warnings):
        print(f"⚠ {w}")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # watch mode output must appear immediately, even when piped
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_env()
    cfg = load_config(a.config)
    instruments = load_instruments(a.instruments)
    if a.symbol:
        a.symbol = a.symbol.upper().replace("/", "")
        if a.symbol not in instruments:
            print(f"unknown symbol {a.symbol}; configured: {', '.join(instruments)}", file=sys.stderr)
            return 2
    opts = Options(
        years=a.years if a.years is not None else float(cfg["data"]["years"]),
        account_size=a.account_size,
        risk_percent=a.risk_percent if a.risk_percent is not None else float(cfg["trade"]["risk_percent"]),
        source=a.source, refresh=a.refresh,
    )
    app = App(cfg, instruments, opts)

    try:
        if a.watch:
            symbols = [a.symbol] if a.symbol else list(instruments)

            def printer(sigs, guard, new):
                print_table(sigs, guard, app)
                for s in new:
                    print_signal(s, app)
                print_warnings(app)

            print(DISCLAIMER)
            app.watch(symbols, printer)
            return 0
        if a.all:
            sigs, guard = app.scan(list(instruments))
            print_table(sigs, guard, app)
            for s in sigs:
                if s.action != "NO TRADE":
                    print_signal(s, app)
        else:
            if a.backtest:
                out = app.backtest(a.symbol)
                m, wf = app.last_backtest
                print(f"Backtest {a.symbol}: {m['trades']} trades, win rate {m['win_rate']:.1%}, "
                      f"PF {m['profit_factor']:.2f}, expectancy {m['expectancy_r']:+.3f}R, max DD {m['max_dd_r']:.1f}R")
                print(f"Walk-forward OOS ({wf.best_params}): {wf.out_of_sample['trades']} trades, "
                      f"expectancy {wf.out_of_sample['expectancy_r']:+.3f}R, PF {wf.out_of_sample['profit_factor']:.2f}")
                for w in wf.warnings:
                    print(f"⚠ {w}")
                print(f"Report saved to {out}")
            if a.plot:
                path = app.plot(a.symbol, a.bars)
                print(f"Chart saved to {path}")
            if not a.backtest:
                sig, _ = app.signal(a.symbol)
                print_signal(sig, app)
        print_warnings(app)
        print(DISCLAIMER)
        return 0
    except KeyboardInterrupt:
        print("\nstopped")
        return 130
    except Exception as exc:
        logging.getLogger("fxpa").error("%s: %s", exc.__class__.__name__, exc)
        if a.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
