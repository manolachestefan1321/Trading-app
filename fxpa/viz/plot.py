"""Static chart: candles, swings, S/R zones, key levels, session shading, patterns, signals."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from fxpa.analysis.candles import scan_patterns  # noqa: E402
from fxpa.signal.engine import Signal  # noqa: E402

UP, DOWN = "#0f9d76", "#d9483b"          # candle bodies
SUP, RES = "#0f9d76", "#d9483b"          # zone fills (low alpha)
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
SESSION_FILL = {"london": "#dbeafe", "newyork": "#fde7cf"}
BUY_C, SELL_C = "#1d4ed8", "#b45309"


def plot_signal(sig: Signal, out_path: Path, cfg: dict, bars: int = 150,
                history: list[Signal] | None = None) -> Path:
    """Draw the last `bars` candles of the signal's analysis window and save a PNG."""
    ctx = sig.context
    if ctx is None:
        raise ValueError("signal was evaluated without keep_context=True")
    w = ctx.window
    n = len(w)
    start = max(0, n - bars)
    view = w.iloc[start:].reset_index(drop=True)
    x = np.arange(len(view))
    o, h, l, c = (view[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))

    fig, ax = plt.subplots(figsize=(15, 7.5))
    fig.patch.set_facecolor("white")
    lo_y, hi_y = l.min(), h.max()
    pad = (hi_y - lo_y) * 0.08
    ylim = (lo_y - pad, hi_y + pad)

    # Session shading (London, New York; the overlap shows as both).
    sess_cfg = cfg["sessions"]
    hours = view["timestamp"].dt.hour.to_numpy()
    for name, color in SESSION_FILL.items():
        s0, s1 = sess_cfg[name]
        mask = (hours >= s0) & (hours < s1)
        for i in np.flatnonzero(mask):
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.45, lw=0, zorder=0)

    # Zones (alpha by score) and key levels.
    for z in ctx.zones:
        if z.high < ylim[0] or z.low > ylim[1]:
            continue
        col = SUP if z.role == "support" else RES
        alpha = 0.04 + 0.16 * z.score / 100 if z.origin == "swings" else 0.05
        ax.add_patch(Rectangle((-0.5, z.low), len(view), z.high - z.low, color=col, alpha=alpha, lw=0, zorder=1))
        if z.origin == "swings":
            ax.text(len(view) + 0.5, z.mid, f"{z.role[:3]} {z.score:.0f}", fontsize=7.5, color=MUTED,
                    va="center", ha="left")
    for lv in ctx.levels:
        if ylim[0] <= lv.price <= ylim[1]:
            ax.axhline(lv.price, color=MUTED, lw=0.8, ls=(0, (4, 4)), zorder=1.5)
            ax.text(0, lv.price, f" {lv.name}", fontsize=7.5, color=MUTED, va="bottom", ha="left")

    # Weekend / data gaps.
    gaps = view.index[view["timestamp"].diff() > pd.Timedelta(hours=1)]
    for g in gaps:
        ax.axvline(g - 0.5, color=MUTED, lw=0.8, ls=":", zorder=1.5)

    # Candles.
    width = 0.62
    for i in x:
        col = UP if c[i] >= o[i] else DOWN
        ax.vlines(i, l[i], h[i], color=col, lw=0.9, zorder=3)
        body_lo, body_h = min(o[i], c[i]), max(abs(c[i] - o[i]), (hi_y - lo_y) * 0.0008)
        ax.add_patch(Rectangle((i - width / 2, body_lo), width, body_h, facecolor=col, edgecolor=col,
                               lw=0.5, zorder=4))

    # Swings.
    sw = ctx.swings[ctx.swings["idx"] >= start]
    for _, s in sw.iterrows():
        xi = s["idx"] - start
        if s["kind"] == "H":
            ax.plot(xi, s["price"] + pad * 0.12, marker="v", ms=8, color=INK, zorder=5,
                    markeredgecolor="white", markeredgewidth=1)
        else:
            ax.plot(xi, s["price"] - pad * 0.12, marker="^", ms=8, color=INK, zorder=5,
                    markeredgecolor="white", markeredgewidth=1)

    # Strong candlestick patterns (pins, engulfings, stars) at swing extremes, as small labels.
    # (Mirrors the engine's rule that patterns only count at zones / swing points.)
    swing_bars = set(ctx.swings["idx"].tolist())
    labelled: set[int] = set()  # one label per bar (scan_patterns lists strongest first)
    abbrev = {"bullish pin bar": "PIN", "bearish pin bar": "PIN", "bullish engulfing": "ENG",
              "bearish engulfing": "ENG", "morning star": "MS", "evening star": "ES",
              "tweezer bottom": "TWZ", "tweezer top": "TWZ"}
    for p in scan_patterns(w, ctx.atr, cfg["candles"], start):
        if p.name not in abbrev or p.strength < 0.5:
            continue
        if p.idx in labelled or not any(b in swing_bars for b in range(p.idx - p.n_bars - 1, p.idx + 2)):
            continue
        labelled.add(p.idx)
        xi = p.idx - start
        y = (p.low - pad * 0.35) if p.direction > 0 else (p.high + pad * 0.35)
        ax.text(xi, y, abbrev[p.name], fontsize=6.5, color=INK, ha="center",
                va="top" if p.direction > 0 else "bottom", zorder=6)

    # Historical signals over the window.
    ts_to_x = {t: i for i, t in enumerate(view["timestamp"])}
    for hs in history or []:
        if hs.action == "NO TRADE" or hs.bar_time not in ts_to_x:
            continue
        xi = ts_to_x[hs.bar_time]
        if hs.action == "BUY":
            ax.annotate("", xy=(xi, l[xi] - pad * 0.15), xytext=(xi, l[xi] - pad * 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=BUY_C, lw=1.8), zorder=7)
        else:
            ax.annotate("", xy=(xi, h[xi] + pad * 0.15), xytext=(xi, h[xi] + pad * 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=SELL_C, lw=1.8), zorder=7)

    # Current signal with entry / SL / TP.
    last = len(view) - 1
    if sig.plan is not None:
        p = sig.plan
        span = (last - 6, last + 8)
        for price, col, lab in ((p.entry, INK, "entry"), (p.stop, DOWN, "SL"), (p.target, UP, "TP")):
            ax.hlines(price, *span, color=col, lw=2, zorder=8)
            ax.text(span[1] + 0.3, price, f"{lab} {price:.5g}", fontsize=8.5, color=INK, va="center", zorder=8)
        col = BUY_C if p.direction > 0 else SELL_C
        if sig.action != "NO TRADE":
            y0 = l[last] - pad * 0.9 if p.direction > 0 else h[last] + pad * 0.9
            y1 = l[last] - pad * 0.15 if p.direction > 0 else h[last] + pad * 0.15
            ax.annotate(sig.action, xy=(last, y1), xytext=(last, y0), ha="center", fontsize=10, color=col,
                        fontweight="bold", arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2), zorder=9)

    # Axes.
    ax.set_xlim(-1, len(view) + 12)
    ax.set_ylim(*ylim)
    step = max(1, len(view) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([t.strftime("%b %d\n%H:%M") for t in view["timestamp"].iloc[::step]], fontsize=8,
                       color=MUTED)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    title = (f"{sig.symbol} 1H  ·  {sig.action}  ·  score {sig.score:+.0f}  ·  "
             f"{sig.bar_time:%Y-%m-%d %H:%M} UTC")
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=28)
    wrapped = _wrap(sig.explanation, 190)
    fig.text(0.01, 0.01, wrapped, fontsize=8, color=MUTED, va="bottom")
    legend = [Patch(color=SUP, alpha=0.3, label="support zone"), Patch(color=RES, alpha=0.3, label="resistance zone"),
              Patch(color=SESSION_FILL["london"], label="London"), Patch(color=SESSION_FILL["newyork"], label="New York"),
              Line2D([], [], color=MUTED, ls=(0, (4, 4)), label="key level"),
              Line2D([], [], marker="v", color=INK, ls="", label="swing"),
              Line2D([], [], color=BUY_C, marker=r"$\uparrow$", ls="", ms=10, label="BUY"),
              Line2D([], [], color=SELL_C, marker=r"$\downarrow$", ls="", ms=10, label="SELL")]
    ax.legend(handles=legend, loc="lower left", fontsize=8, frameon=False, ncol=8, bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _wrap(text: str, width: int) -> str:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    lines.append(cur)
    return "\n".join(lines[:3])
