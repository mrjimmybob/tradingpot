"""Simple long-only spot portfolio ledger for the backtest engine.

Deliberately NOT the production wallet/tax-lot/ledger-invariant stack (see
design.md Non-Goals) - that machinery exists to operate real capital under
real accounting constraints across concurrently-running bots. This is a
purpose-built, single-position cash/base ledger whose only job is to make a
strategy's historical expectancy measurable.
"""
from __future__ import annotations

from typing import List, Optional

from .execution_model import Fill
from .results import TradeRecord


class BacktestPortfolio:
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.cash = starting_balance
        self.base_amount = 0.0
        self.entry_price: Optional[float] = None
        self.entry_timestamp: Optional[int] = None
        self.entry_fees = 0.0

        self.trades: List[TradeRecord] = []
        self.equity_curve: List[tuple] = []
        self.total_fees_paid = 0.0

    @property
    def has_position(self) -> bool:
        return self.base_amount > 1e-12

    def apply_buy(self, timestamp: int, amount_base: float, fill: Fill) -> None:
        if not self.has_position:
            self.entry_timestamp = timestamp
            self.entry_price = fill.price
            self.entry_fees = 0.0
        else:
            total_value = self.base_amount * self.entry_price + amount_base * fill.price
            self.entry_price = total_value / (self.base_amount + amount_base)

        self.base_amount += amount_base
        self.cash -= fill.notional_usd + fill.fee_usd
        self.entry_fees += fill.fee_usd
        self.total_fees_paid += fill.fee_usd

    def apply_sell(
        self,
        timestamp: int,
        amount_base: float,
        fill: Fill,
        strategy: str,
        exit_reason: str,
    ) -> None:
        if not self.has_position:
            return

        sell_amount = min(amount_base, self.base_amount)
        gross_pnl = (fill.price - self.entry_price) * sell_amount
        self.cash += fill.notional_usd - fill.fee_usd
        self.total_fees_paid += fill.fee_usd
        self.base_amount -= sell_amount

        is_full_close = self.base_amount <= 1e-12
        if is_full_close:
            # Attribute the full entry fee to this closing trade record. For a
            # partial exit we do not yet record a closed TradeRecord (this
            # foundation supports single-position, typically full-exit
            # strategies; partial-exit P&L attribution is a known
            # simplification, not a correctness bug for the strategies this
            # repo currently runs).
            net_pnl = gross_pnl - fill.fee_usd - self.entry_fees
            self.trades.append(TradeRecord(
                entry_timestamp=self.entry_timestamp,
                exit_timestamp=timestamp,
                strategy=strategy,
                entry_price=self.entry_price,
                exit_price=fill.price,
                fees=fill.fee_usd + self.entry_fees,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                exit_reason=exit_reason,
            ))
            self.base_amount = 0.0
            self.entry_price = None
            self.entry_timestamp = None
            self.entry_fees = 0.0

    def equity(self, current_price: float) -> float:
        return self.cash + self.base_amount * current_price

    def mark_to_market(self, timestamp: int, current_price: float) -> None:
        self.equity_curve.append((timestamp, self.equity(current_price)))
