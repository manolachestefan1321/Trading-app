"""Walk-forward validation: optimize on the first 60% of bars, validate on the last 40%.

Signals are generated once (the analysis does not depend on the optimized parameters);
only the decision thresholds are re-applied per parameter set, so the grid search is cheap.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from fxpa.backtest.engine import SignalRecord, metrics, simulate
from fxpa.config import Instrument


@dataclass
class WalkForwardResult:
    split_time: pd.Timestamp
    best_params: dict
    grid: pd.DataFrame
    in_sample: dict
    out_of_sample: dict
    default_oos: dict
    oos_trades: pd.DataFrame
    warnings: list[str]


def walk_forward(df: pd.DataFrame, records: list[SignalRecord], inst: Instrument, cfg: dict) -> WalkForwardResult:
    bt = cfg["backtest"]
    split = int(len(df) * float(bt["train_fraction"]))
    risk, eq0 = float(cfg["trade"]["risk_percent"]), float(bt["initial_equity"])
    rows = []
    for thr, rr in itertools.product(bt["grid"]["threshold"], bt["grid"]["min_rr"]):
        tr = simulate(df, records, inst, cfg, float(thr), float(rr), idx_range=(0, split))
        m = metrics(tr, risk, eq0)
        rows.append({"threshold": thr, "min_rr": rr, **m})
    grid = pd.DataFrame(rows)
    warnings: list[str] = []
    eligible = grid[grid["trades"] >= int(bt["min_trades"])]
    default = {"threshold": float(cfg["scoring"]["buy_threshold"]), "min_rr": float(cfg["trade"]["min_rr"])}
    if eligible.empty:
        warnings.append(f"no parameter set produced >= {bt['min_trades']} in-sample trades; "
                        "using the configured defaults (results are not statistically meaningful)")
        best = default
    else:
        b = eligible.sort_values(["expectancy_r", "profit_factor"], ascending=False).iloc[0]
        best = {"threshold": float(b["threshold"]), "min_rr": float(b["min_rr"])}

    is_tr = simulate(df, records, inst, cfg, best["threshold"], best["min_rr"], idx_range=(0, split))
    oos_tr = simulate(df, records, inst, cfg, best["threshold"], best["min_rr"], idx_range=(split, len(df)))
    def_oos = simulate(df, records, inst, cfg, default["threshold"], default["min_rr"], idx_range=(split, len(df)))
    m_is, m_oos = metrics(is_tr, risk, eq0), metrics(oos_tr, risk, eq0)

    ratio = float(bt["overfit_ratio"])
    if m_is["trades"] and m_oos["trades"]:
        if m_is["expectancy_r"] > 0 and m_oos["expectancy_r"] <= 0:
            warnings.append(f"OVERFITTING: in-sample expectancy {m_is['expectancy_r']:+.2f}R collapses to "
                            f"{m_oos['expectancy_r']:+.2f}R out-of-sample")
        elif m_is["expectancy_r"] > 0 and m_oos["expectancy_r"] < ratio * m_is["expectancy_r"]:
            warnings.append(f"possible overfitting: out-of-sample expectancy {m_oos['expectancy_r']:+.2f}R is below "
                            f"{ratio:.0%} of in-sample {m_is['expectancy_r']:+.2f}R")
        if m_is["profit_factor"] > 1 and m_oos["profit_factor"] < 1:
            warnings.append(f"profit factor falls from {m_is['profit_factor']:.2f} to {m_oos['profit_factor']:.2f} OOS")
    elif not m_oos["trades"]:
        warnings.append("no out-of-sample trades with the optimized parameters")

    return WalkForwardResult(
        split_time=df["timestamp"].iat[min(split, len(df) - 1)], best_params=best, grid=grid,
        in_sample=m_is, out_of_sample=m_oos, default_oos=metrics(def_oos, risk, eq0), oos_trades=oos_tr,
        warnings=warnings,
    )
