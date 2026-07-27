"""dip_recovery measures volatility over time, not over evaluation ticks.

The defect this fixes: ATR was the mean absolute change between consecutive
*evaluations*, so its value depended on how often the strategy happened to be
called. The live loop calls it about once a second, which made ATR ~30x smaller
than the defaults assume and left every ATR-derived distance inside the
round-trip fee hurdle - so the strategy's own fee-viability gate refused every
entry, permanently. trend_following had the identical bug and was moved to
60-second bar ranges; this asserts dip_recovery now does the same.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.backtesting.clock import BacktestClock
from app.services.trading_engine import (
    _VIABILITY_SAFETY_MARGIN_PCT,
    TradingEngine,
)

START_MS = 1704067200000
FEE_PCT = 0.1  # bot default; the floor is 2 x fee + safety margin


def _bot(bot_id: int = 1, balance: float = 100_000.0) -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "dip_recovery"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    bot.exchange_fee = FEE_PCT
    return bot


def _engine_at(start_ms: int = START_MS) -> tuple:
    clock = BacktestClock(start_ms)
    engine = TradingEngine(clock=clock)
    engine._get_bot_positions = AsyncMock(return_value=[])
    return engine, clock


async def _drive(engine, clock, bot, samples, params=None):
    """Feed (seconds_from_start, price) samples; return the last signal."""
    session = AsyncMock()
    signal = None
    for offset_s, price in samples:
        clock.set(START_MS + int(offset_s * 1000))
        signal = await engine._strategy_dip_recovery(bot, price, params or {}, session)
    return signal


def _atr_of(engine, bot_id: int) -> float:
    """The ATR the strategy last used, read from its own explanation metrics."""
    return engine._explain(bot_id).to_dict()["metrics"]["atr"]


def _floor_pct() -> float:
    return 2.0 * (FEE_PCT / 100.0) + _VIABILITY_SAFETY_MARGIN_PCT


class TestAtrIsIndependentOfEvaluationCadence:
    """The core regression. Same market, same elapsed time, same price extremes -
    only the number of evaluations differs."""

    def _sweep(self, ticks_per_bar: int, bars: int = 30, low=100.0, high=101.0):
        """Sweep low->high inside every 60-second bar, `ticks_per_bar` times."""
        samples = []
        for bar in range(bars):
            for i in range(ticks_per_bar):
                frac = i / max(ticks_per_bar - 1, 1)
                t = bar * 60 + (i * 60.0 / max(ticks_per_bar, 1))
                samples.append((t, low + (high - low) * frac))
        samples.append((bars * 60, low))  # close the final bar
        return samples

    @pytest.mark.asyncio
    async def test_sixty_evaluations_per_bar_give_the_same_atr_as_two(self):
        coarse_engine, coarse_clock = _engine_at()
        fine_engine, fine_clock = _engine_at()

        await _drive(coarse_engine, coarse_clock, _bot(), self._sweep(2))
        await _drive(fine_engine, fine_clock, _bot(), self._sweep(60))

        coarse = _atr_of(coarse_engine, 1)
        fine = _atr_of(fine_engine, 1)
        assert coarse == pytest.approx(fine, rel=1e-9), (
            "ATR must describe the bar's range, not how often the strategy was called"
        )

    @pytest.mark.asyncio
    async def test_the_old_tick_proxy_would_have_disagreed_wildly(self):
        """Guards the test above from passing vacuously: under the superseded
        tick proxy these two cadences differ by more than an order of magnitude,
        which is exactly why the strategy could not trade live."""
        engine, _ = _engine_at()
        coarse_prices = [p for _, p in self._sweep(2)]
        fine_prices = [p for _, p in self._sweep(60)]

        coarse = engine._calc_price_atr_proxy(coarse_prices, 14)
        fine = engine._calc_price_atr_proxy(fine_prices, 14)
        assert coarse > fine * 10

    @pytest.mark.asyncio
    async def test_atr_equals_the_mean_bar_range(self):
        engine, clock = _engine_at()
        bot = _bot()
        await _drive(engine, clock, bot, self._sweep(10, bars=30, low=100.0, high=105.0))

        bars = engine._dip_recovery_states[bot.id]["dr_bars"]
        assert len(bars) >= 14
        expected = sum(b["high"] - b["low"] for b in bars[-14:]) / 14
        assert _atr_of(engine, bot.id) == pytest.approx(expected, rel=1e-9)
        # 5.0-wide sweeps must dominate the ~0.25% floor (~0.26 at this price).
        assert expected > 105.0 * _floor_pct()


class TestFeeCoverageFloor:
    @pytest.mark.asyncio
    async def test_a_dead_calm_market_never_produces_a_stop_inside_the_fee_hurdle(self):
        """Measured volatility can be far smaller than the cost of trading. A
        near-zero ATR would place every exit inside the round-trip fee."""
        engine, clock = _engine_at()
        bot = _bot()
        # 20 bars whose ranges are negligible next to the ~0.25% fee hurdle.
        await _drive(engine, clock, bot, [(i * 60, 100.0 + i * 0.0001) for i in range(20)])

        bars = engine._dip_recovery_states[bot.id]["dr_bars"]
        assert len(bars) >= 14
        assert max(b["high"] - b["low"] for b in bars) < 0.01, "fixture must be near-flat"
        assert _atr_of(engine, bot.id) == pytest.approx(100.0 * _floor_pct(), rel=1e-3)

    @pytest.mark.asyncio
    async def test_warm_up_holds_until_enough_bars_exist(self):
        """With no position open, the strategy waits for a real ATR window
        rather than acting on one bar."""
        engine, clock = _engine_at()
        bot = _bot()
        signal = await _drive(engine, clock, bot, [(i, 100.0 + i * 0.001) for i in range(20)])

        assert engine._dip_recovery_states[bot.id]["dr_bars"] == []
        assert signal.action == "hold"
        assert "Collecting data" in signal.reason

    @pytest.mark.asyncio
    async def test_a_larger_measured_volatility_takes_over_from_the_floor(self):
        engine, clock = _engine_at()
        bot = _bot()
        samples = []
        for bar in range(30):
            samples.append((bar * 60, 100.0))
            samples.append((bar * 60 + 30, 110.0))  # 10-wide bar range
        samples.append((30 * 60, 100.0))
        await _drive(engine, clock, bot, samples)

        atr = _atr_of(engine, bot.id)
        assert atr > 100.0 * _floor_pct() * 10, "measured volatility must dominate the floor"

    @pytest.mark.asyncio
    async def test_the_floor_scales_with_the_bots_own_fee(self):
        engine, clock = _engine_at()
        bot = _bot()
        bot.exchange_fee = 0.5  # a pricier venue needs a wider minimum
        await _drive(engine, clock, bot, [(i * 60, 100.0) for i in range(20)])

        expected_pct = 2.0 * 0.005 + _VIABILITY_SAFETY_MARGIN_PCT
        assert _atr_of(engine, bot.id) == pytest.approx(100.0 * expected_pct, rel=1e-6)


class TestLiveCadenceRegression:
    @pytest.mark.asyncio
    async def test_a_one_second_cadence_no_longer_starves_the_viability_gate(self):
        """The defect, end to end. At ~1 evaluation/second with realistic
        per-second moves, the take-profit target (3 x ATR) must clear the
        round-trip fee hurdle - it previously measured ~0.02% against 0.25%."""
        engine, clock = _engine_at()
        bot = _bot()

        # 40 minutes of 1-second ticks drifting within a ~0.15%-per-minute band:
        # tiny second-to-second moves, ordinary minute-to-minute volatility.
        samples = []
        price = 60_000.0
        for second in range(40 * 60):
            price += 1.5 if (second // 30) % 2 == 0 else -1.5
            samples.append((second, price))
        await _drive(engine, clock, bot, samples)

        atr = _atr_of(engine, bot.id)
        take_profit_pct = (atr * 3.0) / price
        hurdle = 2.0 * (FEE_PCT / 100.0) + _VIABILITY_SAFETY_MARGIN_PCT
        assert take_profit_pct >= hurdle, (
            f"take-profit {take_profit_pct * 100:.4f}% must clear the "
            f"{hurdle * 100:.3f}% fee hurdle at the live cadence"
        )


class TestSetupAdvancesInMarketTime:
    """fix-dip-recovery-setup-cadence: the setup lifecycle and its confirmation
    counters are paced by bars, so they denote market time rather than however
    often the engine happened to call the strategy."""

    def _bars(self, n=20, price=100.0):
        return [{"high": price, "low": price, "close": price}] * n

    @pytest.mark.asyncio
    async def test_the_no_new_low_counter_advances_at_most_once_per_bar(self):
        """It is `min_ticks_without_new_low`, and at the live ~1s cadence it used
        to be satisfied in 2 seconds rather than the 2 minutes it describes."""
        engine, clock = _engine_at()
        bot = _bot()
        engine._dip_recovery_states = {
            bot.id: {
                **engine._dip_recovery_default_state(),
                "dr_bars": self._bars(),
                "state": "TRACKING_DROP",
                "reference_high": 100.0,
                "reference_high_time": engine.clock.now(),
                "lowest_price": 95.0,
                "lowest_price_time": engine.clock.now(),
                "tracking_started_at": engine.clock.now(),
                "ticks_since_new_low": 0,
            }
        }
        # 30 evaluations inside a single 60-second bar.
        await _drive(engine, clock, bot, [(i, 96.0) for i in range(30)])

        counter = engine._dip_recovery_states[bot.id]["ticks_since_new_low"]
        assert counter <= 1, (
            f"30 evaluations inside one bar advanced the counter to {counter}"
        )

    @pytest.mark.asyncio
    async def test_setup_state_does_not_change_mid_bar(self):
        engine, clock = _engine_at()
        bot = _bot()
        await _drive(engine, clock, bot, [(i * 60, 100.0) for i in range(20)])
        before = dict(engine._dip_recovery_states[bot.id])

        # A sharp drop, but delivered entirely inside one unfinished bar.
        await _drive(engine, clock, bot, [(19 * 60 + i, 90.0) for i in range(1, 30)])
        after = engine._dip_recovery_states[bot.id]

        assert after["state"] == before["state"]
        assert after["reference_high"] == before["reference_high"]

    @pytest.mark.asyncio
    async def test_a_mid_bar_evaluation_reports_why_it_did_nothing(self):
        engine, clock = _engine_at()
        bot = _bot()
        await _drive(engine, clock, bot, [(i * 60, 100.0) for i in range(20)])
        signal = await _drive(engine, clock, bot, [(19 * 60 + 1, 100.0)])

        assert signal.action == "hold"
        assert "Bar in progress" in signal.reason

    @pytest.mark.asyncio
    async def test_the_reference_high_spans_bars_not_evaluations(self):
        """A high set 60 bars ago must still be the reference, however many
        evaluations happened in between."""
        engine, clock = _engine_at()
        bot = _bot()
        samples = [(0, 130.0), (60, 130.0)]  # the peak, held for a full bar
        samples += [(i * 60, 100.0) for i in range(2, 40)]
        # Many intra-bar evaluations that must not age the window.
        samples += [(39 * 60 + i, 100.0) for i in range(1, 50)]
        await _drive(engine, clock, bot, samples)

        bars = engine._dip_recovery_states[bot.id]["dr_bars"]
        assert len(bars) <= 60, "fixture must fit inside the lookback window"
        assert max(b["high"] for b in bars[-60:]) == pytest.approx(130.0), (
            "the 60-BAR window must still contain the peak"
        )


class TestExitsAreNotDeferredToBarClose:
    @pytest.mark.asyncio
    async def test_a_stop_fires_mid_bar(self):
        """The per-bar gate must sit behind exit management: a stop has to react
        when price moves, not at the end of a bar."""
        from types import SimpleNamespace

        engine, clock = _engine_at()
        bot = _bot()
        engine._get_bot_positions = AsyncMock(
            return_value=[SimpleNamespace(amount=1.0)]
        )
        entry = 100.0
        engine._dip_recovery_states = {
            bot.id: {
                **engine._dip_recovery_default_state(),
                "dr_bars": [{"high": 101.0, "low": 99.0, "close": 100.0}] * 20,
                "state": "LONG_OPEN",
                "entry_price": entry,
                "entry_time": engine.clock.now(),
                "entry_atr": 1.0,
                "highest_price_since_entry": entry,
                "trailing_stop": entry - 1.5,
                "take_profit": entry + 3.0,
                "emergency_stop": entry - 5.0,
            }
        }
        # One evaluation, one second into a fresh bar, at the take-profit level.
        signal = await _drive(engine, clock, bot, [(1, 104.0)])

        assert signal.action == "sell", "an exit must not wait for the bar to close"


class TestPersistedStateCompatibility:
    @pytest.mark.asyncio
    async def test_state_saved_before_this_change_restores_without_error(self):
        """A bot resumed from a state dict that predates bar aggregation must
        backfill the new keys rather than KeyError on the first tick."""
        engine, clock = _engine_at()
        bot = _bot()
        legacy = engine._dip_recovery_default_state()
        legacy.pop("dr_bars")
        legacy.pop("dr_current_bar")
        engine._dip_recovery_states = {bot.id: legacy}

        await _drive(engine, clock, bot, [(i, 100.0 + i) for i in range(20)])

        state = engine._dip_recovery_states[bot.id]
        assert "dr_bars" in state and "dr_current_bar" in state

    @pytest.mark.asyncio
    async def test_a_lifecycle_reset_keeps_the_accumulated_bars(self):
        """Regression. Bar history is observed market data, not lifecycle state.
        Resetting it whenever a setup expires or a position closes would drop
        the strategy back onto the fee floor for `atr_period` bars every time -
        and setups expire often enough that it would sit on the floor almost
        permanently, silently undoing this fix."""
        engine, _ = _engine_at()
        bot = _bot()
        state = engine._dip_recovery_default_state()
        state["dr_bars"] = [{"high": 110.0, "low": 100.0, "close": 105.0}] * 20
        state["dr_current_bar"] = {"high": 1.0, "low": 0.0, "close": 0.5, "start_ts": None}
        state["state"] = "TRACKING_DROP"
        state["reference_high"] = 123.0

        reset = engine._dip_recovery_reset_state(state)

        assert len(reset["dr_bars"]) == 20, "volatility history must survive a reset"
        assert reset["dr_current_bar"] is state["dr_current_bar"]
        # ...while the lifecycle itself really is reset.
        assert reset["reference_high"] is None
        assert reset["state"] != "TRACKING_DROP"

    def test_the_default_state_declares_the_bar_keys(self):
        engine, _ = _engine_at()
        state = engine._dip_recovery_default_state()
        assert state["dr_bars"] == []
        assert state["dr_current_bar"] is None

    @pytest.mark.asyncio
    async def test_bars_are_trimmed_so_state_cannot_grow_without_bound(self):
        engine, clock = _engine_at()
        bot = _bot()
        await _drive(engine, clock, bot, [(i * 60, 100.0 + (i % 3)) for i in range(180)])
        assert len(engine._dip_recovery_states[bot.id]["dr_bars"]) <= 100


class TestNoOtherStrategyIsAffected:
    def test_the_superseded_tick_proxy_has_no_remaining_strategy_callers(self):
        """`_calc_price_atr_proxy` is kept only as a shared utility (and is used
        by these tests to demonstrate the regression). No strategy may compute
        volatility with it any more."""
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "app" / "services" / "trading_engine.py"
        ).read_text(encoding="utf-8")
        callers = re.findall(r"self\._calc_price_atr_proxy\(", source)
        assert not callers, (
            "a strategy is still deriving volatility from evaluation ticks: "
            f"{len(callers)} call site(s)"
        )
