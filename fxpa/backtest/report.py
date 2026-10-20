"""Backtest report: markdown summary, trade log CSV, equity curve PNG, JSON metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fxpa.backtest.engine import breakdown, equity_curve, metrics  # noqa: E402
from fxpa.backtest.walkforward import WalkForwardResult  # noqa: E402
from fxpa.config import Instrument  # noqa: E402

COLS = ["trades", "win_rate", "profit_factor", "expectancy_r", "total_r", "max_dd_r", "max_dd_pct", "avg_planned_rr"]


def _fmt_metrics(m: dict) -> str:
    return (f"trades {m['trades']}, win rate {m['win_rate']:.1%}, profit factor {m['profit_factor']:.2f}, "
            f"expectancy {m['expectancy_r']:+.3f}R, total {m['total_r']:+.1f}R, max DD {m['max_dd_r']:.1f}R "
            f"({m['max_dd_pct']:.1f}%), avg planned R:R {m['avg_planned_rr']:.2f}")


def _table(df: pd.DataFrame, key: str) -> str:
    if df.empty:
        return "_no trades_\n"
    out = df[[key] + COLS].copy()
    out["win_rate"] = (out["win_rate"] * 100).round(1)
    lines = ["| " + " | ".join([key, "trades", "win %", "PF", "exp R", "total R", "DD R", "DD %", "avg R:R"]) + " |",
             "|" + "---|" * (len(COLS) + 1)]
    for _, r in out.iterrows():
        lines.append(f"| {r[key]} | {int(r['trades'])} | {r['win_rate']:.1f} | {r['profit_factor']:.2f} | "
                     f"{r['expectancy_r']:+.3f} | {r['total_r']:+.1f} | {r['max_dd_r']:.1f} | {r['max_dd_pct']:.1f} | "
                     f"{r['avg_planned_rr']:.2f} |")
    return "\n".join(lines) + "\n"


def save_report(inst: Instrument, df: pd.DataFrame, trades: pd.DataFrame, wf: WalkForwardResult | None,
                cfg: dict, out_root: Path, data_source: str, notes: list[str]) -> Path:
    symbol = inst.symbol
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = out_root / f"{symbol}_backtest_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    risk, eq0 = float(cfg["trade"]["risk_percent"]), float(cfg["backtest"]["initial_equity"])
    m = metrics(trades, risk, eq0)
    by_session = breakdown(trades, "session", risk, eq0)
    by_side = breakdown(trades, "side", risk, eq0)
    trades.to_csv(out / "trades.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    eq = equity_curve(trades, risk, eq0)
    if len(eq):
        ax.plot(pd.to_datetime(eq["time"]), eq["equity"], color="#2563eb", lw=1.6, label="all trades")
    if wf is not None:
        ax.axvline(wf.split_time, color="#6b7280", ls="--", lw=1)
        ax.text(wf.split_time, ax.get_ylim()[1] if len(eq) else eq0, " out-of-sample →", va="top", fontsize=9,
                color="#374151")
    ax.axhline(eq0, color="#9ca3af", lw=0.8)
    ax.set_title(f"{symbol} equity curve ({risk:g}% risk per trade, compounded)")
    ax.set_ylabel("equity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)

    md = [f"# Backtest {symbol}", "",
          f"Data: {df['timestamp'].iloc[0]:%Y-%m-%d} → {df['timestamp'].iloc[-1]:%Y-%m-%d %H:%M} UTC, "
          f"{len(df)} 1H candles, source: {data_source}.",
          f"Costs: spread {inst.spread_pips} pips, slippage {inst.slippage_pips} pips per fill. "
          "Same-candle stop and target: stop assumed first. Entries at the next candle's open.", "",
          "## All trades (configured parameters)", "", _fmt_metrics(m), "",
          "## Per session", "", _table(by_session, "session"),
          "## Per side", "", _table(by_side, "side")]
    if wf is not None:
        md += ["## Walk-forward (optimize first 60%, validate last 40%)", "",
               f"Split at {wf.split_time:%Y-%m-%d %H:%M} UTC. Best in-sample parameters: "
               f"threshold ±{wf.best_params['threshold']:.0f}, min R:R {wf.best_params['min_rr']}.", "",
               f"- In-sample: {_fmt_metrics(wf.in_sample)}",
               f"- Out-of-sample (optimized): {_fmt_metrics(wf.out_of_sample)}",
               f"- Out-of-sample (configured defaults): {_fmt_metrics(wf.default_oos)}", ""]
        if wf.warnings:
            md += ["**Warnings:**", ""] + [f"- ⚠ {w}" for w in wf.warnings] + [""]
        g = wf.grid[["threshold", "min_rr", "trades", "win_rate", "profit_factor", "expectancy_r"]].copy()
        md += ["### In-sample grid", "", "| threshold | min R:R | trades | win % | PF | exp R |", "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            md.append(f"| {r['threshold']} | {r['min_rr']} | {int(r['trades'])} | {r['win_rate'] * 100:.1f} | "
                      f"{r['profit_factor']:.2f} | {r['expectancy_r']:+.3f} |")
        md.append("")
    if notes:
        md += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    md += ["![equity](equity.png)", "",
           "_Backtest results are hypothetical and do not guarantee future performance. Not financial advice._"]
    (out / "report.md").write_text("\n".join(md))
    summary = {"symbol": symbol, "overall": m,
               "per_session": by_session.to_dict("records") if len(by_session) else [],
               "walk_forward": None if wf is None else {
                   "split_time": str(wf.split_time), "best_params": wf.best_params,
                   "in_sample": wf.in_sample, "out_of_sample": wf.out_of_sample,
                   "default_oos": wf.default_oos, "warnings": wf.warnings}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return out
