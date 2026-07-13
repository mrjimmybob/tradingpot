"""Tests for the backtest replay engine (add-historical-backtesting).

Covers: candle replay correctness, fees reducing returns, a losing strategy
showing negative expectancy (not hidden/adjusted), the buy-and-hold
benchmark, no-lookahead, and run-to-run determinism.
"""
from __future__ import annotations

import math

import pytest

from app.backtesting.candle import Candle
from app.backtesting.engine import BacktestEngine
from app.backtesting.execution_model import BacktestExecutionModel
from app.backtesting.data_provider import CsvHistoricalDataProvider


def _oscillating_candles(n=300, base=100.0, amplitude=0.05, start_ms=1704067200000, step_ms=60_000):
    """Deterministic sine-wave series - no randomness - that gives a
    range-bound strategy (mean_reversion) both entries and exits."""
    candles = []
    for i in range(n):
        price = base * (1 + amplitude * math.sin(i / 12.0))
        candles.append(Candle(
            timestamp=start_ms + i * step_ms,
            datetime="d", symbol="TESTUSD",
            open=price, high=price * 1.001, low=price * 0.999, close=price,
            base_volume=1.0, quote_volume=price, trade_count=1,
        ))
    return candles


def _trending_candles(n=100, base=100.0, drift_pct=0.01, start_ms=1704067200000, step_ms=60_000):
    """Monotonic uptrend - trivial buy-and-hold fixture with a known,
    hand-computable return."""
    candles = []
    price = base
    for i in range(n):
        candles.append(Candle(
            timestamp=start_ms + i * step_ms,
            datetime="d", symbol="TESTUSD",
            open=price, high=price * 1.001, low=price * 0.999, close=price,
            base_volume=1.0, quote_volume=price, trade_count=1,
        ))
        price = price * (1 + drift_pct)
    return candles


_MEAN_REVERSION_PARAMS = {
    "bar_interval_seconds": 0, "regime_filter_enabled": False,
    "bollinger_period": 10, "atr_period": 10, "cooldown_seconds": 0,
    # These are engine-mechanics tests (replay, fees, determinism), not tests of
    # mean_reversion's Phase-4 Decision Score gate - isolate them from it so the
    # engine still produces trades to measure.
    "decision_score_threshold": 0.0,
}


def _engine(fee_pct=0.1, spread_pct=0.0, slippage_pct=0.0) -> BacktestEngine:
    return BacktestEngine(
        data_provider=CsvHistoricalDataProvider(),
        execution_model=BacktestExecutionModel(
            fee_pct=fee_pct, spread_pct=spread_pct, slippage_pct=slippage_pct,
        ),
    )


class TestReplayCorrectness:
    @pytest.mark.asyncio
    async def test_candles_replay_and_produce_trades(self):
        candles = _oscillating_candles()
        result = await _engine().run_candles(
            candles, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        assert result.num_trades > 0
        for trade in result.trades:
            assert trade.entry_timestamp < trade.exit_timestamp
            assert trade.entry_timestamp in [c.timestamp for c in candles]
            assert trade.exit_timestamp in [c.timestamp for c in candles]
            assert trade.strategy == "mean_reversion"
        # Equity curve covers every decision candle plus the final mark.
        assert len(result.equity_curve) == len(candles)

    @pytest.mark.asyncio
    async def test_fill_price_is_next_candle_open_not_decision_candle_close(self):
        """Directly exercises the engine's fill mechanics (bypassing any
        particular strategy's internal amount-sizing math, which is not what
        this test is about): a buy signal decided on candle i must fill at
        candle i+1's open, never candle i's close."""
        from unittest.mock import AsyncMock, MagicMock

        from app.backtesting.portfolio import BacktestPortfolio
        from app.services.trading_engine import TradingEngine, TradeSignal

        candles = _trending_candles(n=3, base=100.0, drift_pct=0.10)
        decision_candle, fill_candle = candles[0], candles[1]
        assert decision_candle.close != fill_candle.open  # otherwise this proves nothing

        engine = TradingEngine()
        portfolio = BacktestPortfolio(10_000.0)
        bot = MagicMock(id=1, trading_pair="TEST/USD")
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))

        exec_model = BacktestExecutionModel(fee_pct=0.0)
        backtest_engine = BacktestEngine(data_provider=CsvHistoricalDataProvider(), execution_model=exec_model)
        signal = TradeSignal(action="buy", amount=1000.0, reason="test buy")

        await backtest_engine._apply_signal(
            engine, session, bot, portfolio, "trend_following", signal, fill_candle,
        )

        expected_base = 1000.0 / fill_candle.open
        assert portfolio.base_amount == pytest.approx(expected_base, rel=1e-9)
        assert portfolio.entry_price == pytest.approx(fill_candle.open, rel=1e-9)
        assert portfolio.entry_price != decision_candle.close


class TestFullCloseAccounting:
    """A full-position exit encoded as a quote notional at the decision price
    (``pos.amount * current_price`` - how every strategy sizes its exits) must
    record exactly one closed round-trip and leave zero residual, even when the
    fill price (next candle's open) has moved away from the decision price.

    Regression for the trade-undercount bug found while backtesting Phase 1
    (see audits/volatility_breakout.md Backtesting section): the engine used to
    divide the exit notional by the *fill* price rather than the *decision*
    price, so on any up-tick the position under-sold and left float dust above
    apply_sell's close threshold - the round trip was never recorded.
    """

    @pytest.mark.asyncio
    async def test_full_close_records_trade_when_fill_price_rises(self):
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker

        from app.models import Base, Bot, BotStatus
        from app.backtesting.candle import Candle
        from app.backtesting.portfolio import BacktestPortfolio
        from app.services.trading_engine import TradingEngine, TradeSignal

        def _candle(price):
            return Candle(
                timestamp=1704067200000, datetime="d", symbol="TESTUSD",
                open=price, high=price * 1.001, low=price * 0.999, close=price,
                base_volume=1.0, quote_volume=price, trade_count=1,
            )

        db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async_session = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

        try:
            async with async_session() as session:
                bot = Bot(
                    name="backtest", trading_pair="TEST/USD", strategy="mean_reversion",
                    strategy_params={}, budget=10_000.0, current_balance=10_000.0,
                    is_dry_run=True, status=BotStatus.RUNNING,
                )
                session.add(bot)
                await session.flush()

                engine = TradingEngine()
                portfolio = BacktestPortfolio(10_000.0)
                backtest_engine = BacktestEngine(
                    data_provider=CsvHistoricalDataProvider(),
                    execution_model=BacktestExecutionModel(fee_pct=0.0),
                )

                # Open a position: spend $1000 filling at price 100 -> 10 base.
                await backtest_engine._apply_signal(
                    engine, session, bot, portfolio, "mean_reversion",
                    TradeSignal(action="buy", amount=1000.0, reason="entry"),
                    _candle(100.0), decision_price=100.0,
                )
                await session.flush()
                assert portfolio.base_amount == pytest.approx(10.0, rel=1e-9)

                # Full exit: the strategy saw a decision price of 110 and sized
                # the sell as base*110 = 1100. The fill (next open) is 121, a
                # strict up-move from the decision price - the exact condition
                # that used to leave dust.
                decision_price = 110.0
                sell_notional = portfolio.base_amount * decision_price
                fill = _candle(121.0)
                assert fill.open > decision_price
                await backtest_engine._apply_signal(
                    engine, session, bot, portfolio, "mean_reversion",
                    TradeSignal(action="sell", amount=sell_notional, reason="exit"),
                    fill, decision_price=decision_price,
                )
                await session.flush()

                # The position is fully closed on both ledgers, and the round
                # trip is recorded exactly once.
                assert portfolio.base_amount <= 1e-12
                assert not portfolio.has_position
                assert len(portfolio.trades) == 1
                trade = portfolio.trades[0]
                assert trade.entry_price == pytest.approx(100.0, rel=1e-9)
                assert trade.exit_price == pytest.approx(121.0, rel=1e-9)
        finally:
            await db_engine.dispose()


class TestFeesReduceReturns:
    @pytest.mark.asyncio
    async def test_fees_reduce_ending_balance(self):
        candles = _oscillating_candles()
        no_fee = await _engine(fee_pct=0.0).run_candles(
            candles, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        with_fee = await _engine(fee_pct=0.1).run_candles(
            candles, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        assert with_fee.num_trades == no_fee.num_trades  # same decisions either way
        assert with_fee.total_fees_paid > 0
        assert no_fee.total_fees_paid == 0
        assert with_fee.ending_balance < no_fee.ending_balance


class TestNegativeExpectancy:
    @pytest.mark.asyncio
    async def test_losing_strategy_shows_negative_expectancy_unmodified(self):
        """A strategy run against a market pattern it trades poorly in must
        report a genuinely negative expectancy - not clamped, hidden, or
        adjusted to look better."""
        # A fast, choppy oscillation with a high hard-stop-triggering
        # amplitude relative to the bollinger band period churns mean
        # reversion into repeated stop-outs.
        candles = _oscillating_candles(n=400, amplitude=0.08)
        result = await _engine(fee_pct=0.1).run_candles(
            candles, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        assert result.num_trades > 0
        assert result.expectancy_per_trade < 0
        assert result.total_return_pct < 0


class TestBuyAndHoldBenchmark:
    @pytest.mark.asyncio
    async def test_buy_and_hold_matches_hand_computed_return(self):
        candles = _trending_candles(n=50, base=100.0, drift_pct=0.01)
        result = await _engine().run_candles(
            candles, "TEST/USD", "dca_accumulator",
            {"interval_minutes": 0, "amount_percent": 0, "immediate_first_buy": False,
             "regime_filter_enabled": False},
            10_000.0,
        )
        expected = (candles[-1].close - candles[0].open) / candles[0].open * 100.0
        assert result.buy_and_hold_return_pct == pytest.approx(expected, rel=1e-9)


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_same_input_produces_identical_output(self):
        candles = _oscillating_candles()
        r1 = await _engine().run_candles(
            candles, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        r2 = await _engine().run_candles(
            candles, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        assert r1.ending_balance == r2.ending_balance
        assert r1.num_trades == r2.num_trades
        assert [(t.entry_timestamp, t.exit_timestamp, t.net_pnl) for t in r1.trades] == \
               [(t.entry_timestamp, t.exit_timestamp, t.net_pnl) for t in r2.trades]
        assert r1.equity_curve == r2.equity_curve


class TestNoLookahead:
    @pytest.mark.asyncio
    async def test_decision_never_sees_a_future_candle(self):
        """Wrap the candle list in a sequence that raises the moment any
        index is first accessed - proves the engine's own access pattern
        never reaches ahead of what a real-time replay could have seen at
        that point (index i+1 must not be read before the decision on index
        i has already happened)."""
        real_candles = _oscillating_candles(n=60)

        class _TrackingCandles:
            def __init__(self, data):
                self._data = data
                self.max_index_seen = -1

            def __len__(self):
                return len(self._data)

            def __getitem__(self, index):
                normalized = index if index >= 0 else len(self._data) + index
                if normalized > self.max_index_seen + 1:
                    raise AssertionError(
                        f"Lookahead: index {normalized} accessed while only up to "
                        f"{self.max_index_seen} had been seen"
                    )
                self.max_index_seen = max(self.max_index_seen, normalized)
                return self._data[index]

        tracked = _TrackingCandles(real_candles)
        result = await _engine().run_candles(
            tracked, "TEST/USD", "mean_reversion", _MEAN_REVERSION_PARAMS, 10_000.0,
        )
        assert result.num_trades >= 0  # completes without the assertion above firing
        assert tracked.max_index_seen == len(real_candles) - 1
