"""Backtest result data structures and metrics computation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TradeRecord:
    entry_timestamp: int
    exit_timestamp: int
    strategy: str
    entry_price: float
    exit_price: float
    fees: float
    gross_pnl: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    starting_balance: float
    ending_balance: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy_per_trade: float
    total_fees_paid: float
    buy_and_hold_return_pct: float
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[tuple] = field(default_factory=list)  # (timestamp, equity)


def _max_drawdown_pct(equity_curve: List[tuple]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0][1]
    max_dd = 0.0
    for _, equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def compute_result(
    starting_balance: float,
    ending_balance: float,
    trades: List[TradeRecord],
    equity_curve: List[tuple],
    total_fees_paid: float,
    buy_and_hold_return_pct: float,
) -> BacktestResult:
    """Pure computation over recorded trades/equity - no I/O, deterministic."""
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]

    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    avg_win = (sum(t.net_pnl for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t.net_pnl for t in losses) / len(losses)) if losses else 0.0
    largest_win = max((t.net_pnl for t in wins), default=0.0)
    largest_loss = min((t.net_pnl for t in losses), default=0.0)

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    expectancy_per_trade = (sum(t.net_pnl for t in trades) / len(trades)) if trades else 0.0
    total_return_pct = (
        (ending_balance - starting_balance) / starting_balance * 100.0
        if starting_balance > 0 else 0.0
    )

    return BacktestResult(
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        total_return_pct=total_return_pct,
        num_trades=len(trades),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        max_drawdown_pct=_max_drawdown_pct(equity_curve),
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy_per_trade,
        total_fees_paid=total_fees_paid,
        buy_and_hold_return_pct=buy_and_hold_return_pct,
        trades=trades,
        equity_curve=equity_curve,
    )
