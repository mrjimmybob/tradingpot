"""Clock abstraction for TradingEngine's time source.

TradingEngine depends on a Clock instance (injected at construction) rather
than calling datetime.utcnow() directly, so its time source can be swapped
without global monkeypatching:

- SystemClock: production - live and dry-run bots both use this (dry-run is
  real-time paper trading against real market data, so it needs the real
  clock same as live).
- BacktestClock (app/backtesting/clock.py): historical replay time, held on
  the TradingEngine instance the backtest engine constructs and advanced as
  it steps through candles. No shared/global state - a backtest and a live
  bot each hold their own TradingEngine with their own clock, so they cannot
  affect each other even in the same process.
"""
from abc import ABC, abstractmethod
from datetime import datetime


class Clock(ABC):
    """Returns naive UTC datetimes, matching this codebase's
    datetime.utcnow() convention throughout."""

    @abstractmethod
    def now(self) -> datetime:
        ...


class SystemClock(Clock):
    """Real wall-clock time. Used by live trading and dry-run."""

    def now(self) -> datetime:
        return datetime.utcnow()
