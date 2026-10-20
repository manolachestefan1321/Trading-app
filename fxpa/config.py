"""Configuration loading: config.yaml (strategy/data settings) and instruments.yaml (pairs)."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
DEFAULT_INSTRUMENTS_PATH = ROOT / "instruments.yaml"


@dataclass(frozen=True)
class Instrument:
    """Static description of a tradable pair (pip math, costs, source symbols)."""

    symbol: str
    base: str
    quote: str
    pip_size: float
    price_decimals: int
    contract_size: float
    spread_pips: float
    slippage_pips: float
    sessions: tuple[str, ...]
    round_steps: tuple[float, ...]
    pip_value_usd: float
    dukascopy: str
    dukascopy_point: float
    yfinance: str
    oanda: str
    twelvedata: str

    @property
    def currencies(self) -> tuple[str, str]:
        return (self.base, self.quote)

    def to_pips(self, price_distance: float) -> float:
        return price_distance / self.pip_size

    def fmt(self, price: float) -> str:
        return f"{price:.{self.price_decimals}f}"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load config.yaml, optionally deep-merging a dict of overrides on top."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cache_dir = Path(cfg["data"]["cache_dir"])
    if not cache_dir.is_absolute():
        cfg["data"]["cache_dir"] = str(ROOT / cache_dir)
    return cfg


def load_instruments(path: str | Path | None = None) -> dict[str, Instrument]:
    """Load instruments.yaml into Instrument objects keyed by symbol."""
    path = Path(path) if path else DEFAULT_INSTRUMENTS_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, Instrument] = {}
    for sym, d in raw.items():
        sym = sym.upper()
        out[sym] = Instrument(
            symbol=sym,
            base=d["base"],
            quote=d["quote"],
            pip_size=float(d["pip_size"]),
            price_decimals=int(d.get("price_decimals", 5)),
            contract_size=float(d.get("contract_size", 100000)),
            spread_pips=float(d.get("spread_pips", 1.0)),
            slippage_pips=float(d.get("slippage_pips", 0.0)),
            sessions=tuple(d.get("sessions", ["london", "overlap"])),
            round_steps=tuple(float(x) for x in d.get("round_steps", [])),
            pip_value_usd=float(d.get("pip_value_usd", 10.0)),
            dukascopy=d.get("dukascopy", f"{d['base']}/{d['quote']}"),
            dukascopy_point=float(d.get("dukascopy_point", 100000)),
            yfinance=d.get("yfinance", f"{sym}=X"),
            oanda=d.get("oanda", f"{d['base']}_{d['quote']}"),
            twelvedata=d.get("twelvedata", f"{d['base']}/{d['quote']}"),
        )
    return out


def load_env() -> None:
    """Load a .env file (API keys) from the project root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(ROOT / ".env", override=False)


def env(name: str) -> str | None:
    val = os.environ.get(name, "").strip()
    return val or None
