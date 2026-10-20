"""Pip math, pip value in account currency, and position sizing."""

from __future__ import annotations

from fxpa.config import Instrument


def pip_value_per_lot(inst: Instrument, price: float, prices: dict[str, float] | None = None,
                      account: str = "USD") -> float:
    """Value of one pip for one standard lot, in the account currency (USD supported natively).

    quote == USD   -> pip_size * contract
    base  == USD   -> pip_size * contract / price
    crosses        -> convert via a cached USD pair (e.g. USDJPY for EURJPY), else the
                      `pip_value_usd` fallback from instruments.yaml.
    """
    raw = inst.pip_size * inst.contract_size  # in quote currency
    if account != "USD":
        return inst.pip_value_usd
    if inst.quote == "USD":
        return raw
    if inst.base == "USD":
        return raw / price
    prices = prices or {}
    if f"USD{inst.quote}" in prices:
        return raw / prices[f"USD{inst.quote}"]
    if f"{inst.quote}USD" in prices:
        return raw * prices[f"{inst.quote}USD"]
    return inst.pip_value_usd


def position_size(account_size: float, risk_percent: float, risk_pips: float, pip_value: float) -> float:
    """Lots so that hitting the stop loses `risk_percent` of the account."""
    if risk_pips <= 0 or pip_value <= 0:
        return 0.0
    return account_size * risk_percent / 100.0 / (risk_pips * pip_value)


def cost_pips(inst: Instrument) -> float:
    """Round-trip cost assumption in pips: spread plus entry and exit slippage."""
    return inst.spread_pips + 2 * inst.slippage_pips
