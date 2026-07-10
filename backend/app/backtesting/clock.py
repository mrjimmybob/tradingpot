"""BacktestClock: a Clock implementation controlled by the replay engine.

Passed to TradingEngine via constructor dependency injection (see
app/services/clock.py for the Clock interface and SystemClock) instead of
monkeypatching datetime.utcnow(). Each backtest constructs its own
TradingEngine with its own BacktestClock instance - there is no shared or
global time state, so running a backtest cannot affect a live or dry-run
TradingEngine's clock even when both run in the same process.
"""
from __future__ import annotations

from datetime import datetime

from app.services.clock import Clock

from .candle import ms_to_naive_utc


class BacktestClock(Clock):
    """Holds the replay engine's current position in historical time.

    Usage:
        clock = BacktestClock(candles[0].timestamp)
        engine = TradingEngine(clock=clock)
        ...
        clock.set(candles[i].timestamp)
        signal = await engine._get_strategy_executor(strategy)(...)
    """

    def __init__(self, initial_ts_ms: int):
        self._now = ms_to_naive_utc(initial_ts_ms)

    def now(self) -> datetime:
        return self._now

    def set(self, ts_ms: int) -> None:
        self._now = ms_to_naive_utc(ts_ms)
