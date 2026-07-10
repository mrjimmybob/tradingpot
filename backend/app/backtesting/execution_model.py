"""Backtest execution model: fees, spread, slippage, no-lookahead fills.

Reuses the production ExecutionCostModel for the fee calculation so live and
backtest cost semantics stay consistent, rather than a second fee formula.
Spread/slippage are applied as a price adjustment (the production model only
estimates them as a risk-sizing dollar figure, not a fill-price shift), which
is what a real order book actually does to a market order's execution price.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.execution_cost_model import ExecutionCostModel


@dataclass
class Fill:
    price: float
    fee_usd: float
    notional_usd: float


class BacktestExecutionModel:
    """Fills a market order against the NEXT candle's open price (no
    lookahead - the caller must never pass the same candle used for the
    decision), worsened by configured spread/slippage, with a per-side fee.
    """

    def __init__(
        self,
        fee_pct: float = 0.1,
        spread_pct: float = 0.0,
        slippage_pct: float = 0.0,
    ):
        self.fee_pct = fee_pct
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self._cost_model = ExecutionCostModel(exchange_fee_pct=fee_pct)

    def fill(self, side: str, amount_base: float, next_candle_open: float) -> Fill:
        """Simulate a market fill at ``next_candle_open``, the only price the
        strategy's decision (made on the prior candle) could not have seen."""
        adjustment_pct = (self.spread_pct + self.slippage_pct) / 100.0
        if side == "buy":
            fill_price = next_candle_open * (1 + adjustment_pct)
        else:
            fill_price = next_candle_open * (1 - adjustment_pct)

        notional_usd = amount_base * fill_price
        fee_usd = self._cost_model.estimate_cost(side, notional_usd, fill_price).exchange_fee

        return Fill(price=fill_price, fee_usd=fee_usd, notional_usd=notional_usd)
