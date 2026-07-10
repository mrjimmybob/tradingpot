"""Regression tests for the Clock dependency-injection fix (no more global
datetime monkeypatching).

Covers:
1. A backtest's clock cannot affect a separate live/dry-run TradingEngine's
   clock - proven both at rest and under concurrent async execution in the
   same process/event loop.
2. The CLI (app/backtesting/run.py) executes a backtest successfully.
3. Backtest results are still deterministic with the dependency-injected
   clock (including a cooldown-sensitive strategy, which is exactly the kind
   of logic that would silently misbehave if the clock source were wrong).
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.backtesting.candle import Candle
from app.backtesting.clock import BacktestClock
from app.backtesting.data_provider import CsvHistoricalDataProvider
from app.backtesting.engine import BacktestEngine
from app.services.clock import SystemClock
from app.services.trading_engine import TradingEngine

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = Path(__file__).resolve().parents[1]


class TestClockIsolation:
    def test_backtest_clock_is_not_the_system_clock(self):
        """A BacktestClock set to a historical time must never equal real
        wall-clock time, and each TradingEngine must hold its own instance -
        not a shared/global one."""
        live_engine = TradingEngine()
        bt_engine = TradingEngine(clock=BacktestClock(1704067200000))  # 2024-01-01

        assert isinstance(live_engine.clock, SystemClock)
        assert isinstance(bt_engine.clock, BacktestClock)
        assert live_engine.clock is not bt_engine.clock

        real_now = live_engine.clock.now()
        historical_now = bt_engine.clock.now()
        assert historical_now == datetime(2024, 1, 1, 0, 0, 0)
        # Real "now" in this test environment is years after the fixed
        # historical backtest time - proves they are not the same clock.
        assert real_now > historical_now + timedelta(days=365)

    def test_mutating_backtest_clock_does_not_affect_live_engine(self):
        live_engine = TradingEngine()
        clock = BacktestClock(1704067200000)
        bt_engine = TradingEngine(clock=clock)

        before = live_engine.clock.now()
        clock.set(1600000000000)  # advance the backtest clock far into the past
        after = live_engine.clock.now()

        assert bt_engine.clock.now() == datetime(2020, 9, 13, 12, 26, 40)
        # The live engine's clock reads real time both before and after -
        # completely unaffected by the backtest engine's clock mutation.
        assert (after - before).total_seconds() < 5  # both are "now", not frozen historical time

    @pytest.mark.asyncio
    async def test_no_cross_contamination_under_concurrent_execution(self):
        """The historical failure mode this fix closes: the old
        implementation patched app.services.trading_engine.datetime as a
        module attribute, so any `await` inside a backtest run could hand
        control to a concurrently-running live/dry-run engine in the same
        event loop, which would then read the frozen historical time. Proves
        that no longer happens: a live-like loop and a backtest-like loop
        interleaved via asyncio.gather never see each other's clock."""
        live_engine = TradingEngine()
        bt_clock = BacktestClock(1704067200000)
        bt_engine = TradingEngine(clock=bt_clock)

        live_observations = []
        bt_observations = []

        async def live_loop():
            for _ in range(20):
                live_observations.append(live_engine.clock.now())
                await asyncio.sleep(0)  # yield control, same as a real await point

        async def backtest_loop():
            ts = 1704067200000
            for _ in range(20):
                ts += 60_000
                bt_clock.set(ts)
                bt_observations.append(bt_engine.clock.now())
                await asyncio.sleep(0)  # yield control mid-"replay", same as engine.py does

        await asyncio.gather(live_loop(), backtest_loop())

        assert len(live_observations) == 20
        assert len(bt_observations) == 20
        # Every live observation must be real, recent wall-clock time -
        # never a frozen historical timestamp from the concurrently-running
        # backtest loop.
        real_now = datetime.utcnow()
        for obs in live_observations:
            assert abs((real_now - obs).total_seconds()) < 60
            assert obs.year >= real_now.year
        # And the backtest's own observations must be exactly the historical
        # sequence it set, unaffected by the live loop running alongside it.
        assert bt_observations == [
            datetime(2024, 1, 1, 0, 0, 0) + timedelta(minutes=i + 1) for i in range(20)
        ]

    @pytest.mark.asyncio
    async def test_backtest_engine_clock_does_not_affect_sibling_live_engine(self):
        """Same proof at the level actually used in production: run a real
        BacktestEngine replay (many awaits per candle) concurrently with a
        plain TradingEngine standing in for a live/dry-run bot, and confirm
        the live engine's clock is never affected."""
        live_engine = TradingEngine()

        candles = []
        base_ms = 1704067200000
        for i in range(60):
            price = 100.0 + (i % 5)
            candles.append(Candle(
                timestamp=base_ms + i * 60_000, datetime="d", symbol="TEST",
                open=price, high=price + 1, low=price - 1, close=price,
                base_volume=1.0, quote_volume=price, trade_count=1,
            ))

        live_clock_samples = []

        async def poll_live_clock():
            for _ in range(50):
                live_clock_samples.append(live_engine.clock.now())
                await asyncio.sleep(0)

        bt = BacktestEngine(data_provider=CsvHistoricalDataProvider())
        await asyncio.gather(
            poll_live_clock(),
            bt.run_candles(candles, "TEST/USD", "dca_accumulator",
                            {"interval_minutes": 0, "amount_percent": 5,
                             "immediate_first_buy": True, "regime_filter_enabled": False},
                            10_000.0),
        )

        real_now = datetime.utcnow()
        assert all(abs((real_now - s).total_seconds()) < 60 for s in live_clock_samples)


class TestCliEntryPoint:
    def _write_fake_dataset(self, tmp_path: Path) -> Path:
        root = tmp_path / "data" / "backtest"
        directory = root / "zaplex" / "FOOBAR" / "1m"
        directory.mkdir(parents=True)
        base_ms = 1704067200000
        rows = ["Unix,Date,Symbol,Open,High,Low,Close,Volume FOO,Volume BAR,tradecount\n"]
        price = 100.0
        for i in range(300):
            price *= 1.001 if i % 20 < 10 else 0.999
            ts = base_ms + i * 60_000
            rows.append(f"{ts},d,FOOBAR,{price},{price*1.001},{price*0.999},{price},1.0,{price},1\n")
        (directory / "data.csv").write_text("".join(rows))
        return root

    def test_cli_runs_a_backtest_successfully(self, tmp_path):
        data_root = self._write_fake_dataset(tmp_path)
        result = subprocess.run(
            [
                sys.executable, "-m", "app.backtesting.run",
                "--exchange", "zaplex", "--symbol", "FOOBAR", "--timeframe", "1m",
                "--strategy", "mean_reversion",
                "--params", '{"bar_interval_seconds": 0, "regime_filter_enabled": false, '
                             '"bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0}',
                "--data-root", str(data_root),
            ],
            cwd=str(_BACKEND_DIR), capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "Backtest: mean_reversion on zaplex/FOOBAR" in result.stdout
        assert "Return %" in result.stdout
        assert "Buy-and-hold return %" in result.stdout
        assert "Trades" in result.stdout
        assert "Win rate" in result.stdout
        assert "Profit factor" in result.stdout
        assert "Expectancy per trade" in result.stdout
        assert "Max drawdown %" in result.stdout
        assert "Total fees paid" in result.stdout

    def test_cli_rejects_unknown_strategy(self, tmp_path):
        data_root = self._write_fake_dataset(tmp_path)
        result = subprocess.run(
            [
                sys.executable, "-m", "app.backtesting.run",
                "--exchange", "zaplex", "--symbol", "FOOBAR", "--timeframe", "1m",
                "--strategy", "funding_carry", "--data-root", str(data_root),
            ],
            cwd=str(_BACKEND_DIR), capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "Unknown strategy" in result.stderr

    def test_cli_rejects_unknown_exchange(self, tmp_path):
        data_root = self._write_fake_dataset(tmp_path)
        result = subprocess.run(
            [
                sys.executable, "-m", "app.backtesting.run",
                "--exchange", "nope", "--symbol", "FOOBAR", "--timeframe", "1m",
                "--strategy", "mean_reversion", "--data-root", str(data_root),
            ],
            cwd=str(_BACKEND_DIR), capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "Unknown exchange" in result.stderr
        assert "zaplex" in result.stderr


class TestDeterminismWithInjectedClock:
    @pytest.mark.asyncio
    async def test_cooldown_sensitive_strategy_is_deterministic(self):
        """A non-zero cooldown_seconds exercises the exact logic
        (self.clock.now() deltas) that would silently break if the injected
        BacktestClock weren't actually driving the strategy's time source."""
        candles = []
        base_ms = 1704067200000
        import math
        for i in range(500):
            price = 100.0 * (1 + 0.06 * math.sin(i / 10.0))
            candles.append(Candle(
                timestamp=base_ms + i * 60_000, datetime="d", symbol="TEST",
                open=price, high=price * 1.002, low=price * 0.998, close=price,
                base_volume=1.0, quote_volume=price, trade_count=1,
            ))

        params = {
            "bar_interval_seconds": 0, "regime_filter_enabled": False,
            "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 300,
        }

        async def run_once():
            engine = BacktestEngine(data_provider=CsvHistoricalDataProvider())
            return await engine.run_candles(candles, "TEST/USD", "mean_reversion", params, 10_000.0)

        r1 = await run_once()
        r2 = await run_once()

        assert r1.num_trades == r2.num_trades
        assert r1.ending_balance == r2.ending_balance
        assert [(t.entry_timestamp, t.exit_timestamp, t.net_pnl) for t in r1.trades] == \
               [(t.entry_timestamp, t.exit_timestamp, t.net_pnl) for t in r2.trades]

    @pytest.mark.asyncio
    async def test_cooldown_is_governed_by_historical_time_not_real_time(self):
        """If the clock injection were broken (e.g. silently falling back to
        SystemClock), a 300s cooldown would never appear to expire during a
        real-time-microseconds-long test run, even though ~4+ minutes of
        HISTORICAL time pass between candles. Confirms trades resume after
        the cooldown's historical window, not never."""
        import math
        candles = []
        base_ms = 1704067200000
        for i in range(600):
            price = 100.0 * (1 + 0.06 * math.sin(i / 10.0))
            candles.append(Candle(
                timestamp=base_ms + i * 60_000, datetime="d", symbol="TEST",
                open=price, high=price * 1.002, low=price * 0.998, close=price,
                base_volume=1.0, quote_volume=price, trade_count=1,
            ))
        params = {
            "bar_interval_seconds": 0, "regime_filter_enabled": False,
            "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 300,
        }
        engine = BacktestEngine(data_provider=CsvHistoricalDataProvider())
        result = await engine.run_candles(candles, "TEST/USD", "mean_reversion", params, 10_000.0)

        # If the clock were wrong (real time instead of historical), the
        # cooldown would latch after the first exit and no further trade
        # could occur within this test's real-world runtime of milliseconds.
        assert result.num_trades > 1, (
            "Only one trade occurred - the cooldown never appeared to expire, "
            "suggesting the strategy is reading real time instead of the "
            "injected BacktestClock's historical time"
        )
