"""Tests for the funding_carry strategy and its configuration.

Covers:
- Parameter validation (validate_funding_carry_params)
- Strategy registration and config exposure
- Signal generation: entry, funding/regime gating, exit, cooldown
- Failure scenarios: missing funding data

Strategy logic is tested directly against the engine method with a fake
exchange and seeded price history (no DB, no real exchange).
"""

import pytest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.trading_engine import TradingEngine, validate_funding_carry_params
from app.services.exchange import ExchangeService, FundingRate
from app.routers.config import STRATEGIES
from app.models import Bot, BotStatus


# ============================================================================
# Helpers
# ============================================================================


class FakeFundingExchange:
    """Exchange stand-in returning preset funding rates."""

    def __init__(self, rates, interval_hours=8.0):
        self._rates = rates
        self._interval = interval_hours

    def to_swap_symbol(self, symbol):
        return ExchangeService.to_swap_symbol(symbol)

    async def get_funding_rate_history(self, symbol, limit=200, since=None):
        return [
            FundingRate(symbol, r, datetime.utcnow(), self._interval)
            for r in self._rates
        ]


def make_bot(strategy_params=None, balance=10000.0):
    return Bot(
        id=1,
        name="Funding Carry Bot",
        trading_pair="BTC/USDT",
        strategy="funding_carry",
        strategy_params=strategy_params or {},
        budget=balance,
        current_balance=balance,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        total_pnl=0.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def make_engine(bot, rates, positions=None, history=None):
    """Build an engine wired with a fake exchange, price history and positions."""
    engine = TradingEngine()
    engine._exchange_services[bot.id] = FakeFundingExchange(rates)
    engine._get_bot_positions = AsyncMock(return_value=positions or [])
    # Seed a strong uptrend by default so the regime filter passes.
    if history is None:
        history = [100.0 + i for i in range(60)]
    engine._price_histories = {bot.id: list(history)}
    return engine


FLAT_HISTORY = [100.0] * 60
# Steep downtrend (−1 per tick × 60 ticks): clearly below the −0.5% EMA slope
# threshold that _detect_market_regime uses for trend_down.
DOWN_HISTORY = [100.0 - i for i in range(250)]


# ============================================================================
# Parameter validation
# ============================================================================


class TestValidation:
    def test_valid_params(self):
        assert validate_funding_carry_params({
            "min_funding_rate": -0.0005,
            "max_funding_rate": 0.0005,
            "funding_lookback_periods": 3,
            "max_allocation_percent": 20,
            "cooldown_seconds": 300,
            "funding_refresh_seconds": 300,
            "allowed_regimes": ["trend_up"],
        }) == []

    def test_empty_params_valid(self):
        assert validate_funding_carry_params({}) == []

    def test_min_above_max_rejected(self):
        errors = validate_funding_carry_params(
            {"min_funding_rate": 0.001, "max_funding_rate": -0.001}
        )
        assert any("min_funding_rate" in e for e in errors)

    def test_bad_lookback_rejected(self):
        assert validate_funding_carry_params({"funding_lookback_periods": 0})
        assert validate_funding_carry_params({"funding_lookback_periods": 2.5})

    def test_bad_allocation_rejected(self):
        assert validate_funding_carry_params({"max_allocation_percent": 0})
        assert validate_funding_carry_params({"max_allocation_percent": 150})

    def test_bad_cooldown_and_refresh_rejected(self):
        assert validate_funding_carry_params({"cooldown_seconds": -1})
        assert validate_funding_carry_params({"funding_refresh_seconds": 0})

    def test_bad_allowed_regimes_rejected(self):
        assert validate_funding_carry_params({"allowed_regimes": "trend_up"})


# ============================================================================
# Registration / configuration
# ============================================================================


class TestRegistration:
    def test_registered_in_engine(self):
        engine = TradingEngine()
        assert engine._get_strategy_executor("funding_carry") is not None

    def test_exposed_in_config(self):
        info = next((s for s in STRATEGIES if s.name == "funding_carry"), None)
        assert info is not None
        for key in (
            "min_funding_rate", "max_funding_rate", "funding_lookback_periods",
            "allowed_regimes", "max_allocation_percent", "cooldown_seconds",
            "funding_refresh_seconds",
        ):
            assert key in info.parameters

    def test_config_default_band_is_coherent(self):
        info = next(s for s in STRATEGIES if s.name == "funding_carry")
        lo = info.parameters["min_funding_rate"]["default"]
        hi = info.parameters["max_funding_rate"]["default"]
        assert lo <= hi


# ============================================================================
# Signal generation
# ============================================================================


class TestSignals:
    @pytest.mark.asyncio
    async def test_entry_when_funding_and_trend_favourable(self):
        bot = make_bot()
        engine = make_engine(bot, rates=[0.0001, 0.0001, 0.0001])

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "buy"
        # default max_allocation_percent = 20
        assert signal.amount == pytest.approx(2000.0)

    @pytest.mark.asyncio
    async def test_no_entry_when_funding_outside_band(self):
        bot = make_bot()
        # 0.2%/period is well above the default 0.05% upper bound
        engine = make_engine(bot, rates=[0.002, 0.002, 0.002])

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "hold"
        assert "outside favourable band" in signal.reason

    @pytest.mark.asyncio
    async def test_no_entry_when_trend_downtrend(self):
        """Funding Carry must still block entry during a downtrend.

        The default allowed_regimes is now ["trend_up", "trend_flat"] so flat
        markets allow participation.  A downtrend (trend_down) must still be
        blocked because it is not in the allowed list.
        """
        bot = make_bot()
        engine = make_engine(
            bot, rates=[0.0001, 0.0001, 0.0001], history=DOWN_HISTORY
        )

        signal = await engine._strategy_funding_carry(bot, DOWN_HISTORY[-1], bot.strategy_params, None)

        assert signal.action == "hold"
        assert "regime" in signal.reason

    @pytest.mark.asyncio
    async def test_entry_allowed_in_flat_regime_with_favourable_funding(self):
        """trend_flat is now included in the default allowed_regimes so the
        strategy can participate during sideways BTC markets with good funding."""
        bot = make_bot()
        engine = make_engine(
            bot, rates=[0.0001, 0.0001, 0.0001], history=FLAT_HISTORY
        )

        signal = await engine._strategy_funding_carry(bot, 100.0, bot.strategy_params, None)

        assert signal.action == "buy", (
            f"Expected buy in flat regime with favourable funding; got: {signal.reason}"
        )

    @pytest.mark.asyncio
    async def test_hold_when_funding_data_unavailable(self):
        bot = make_bot()
        engine = make_engine(bot, rates=[])  # exchange returns no funding history

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "hold"
        assert "unavailable" in signal.reason

    @pytest.mark.asyncio
    async def test_exit_when_funding_leaves_band(self):
        bot = make_bot()
        position = SimpleNamespace(amount=0.01)
        engine = make_engine(
            bot, rates=[0.002, 0.002, 0.002], positions=[position]
        )

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "sell"
        assert signal.amount == pytest.approx(0.01 * 160.0)

    @pytest.mark.asyncio
    async def test_hold_position_when_conditions_still_favourable(self):
        bot = make_bot()
        position = SimpleNamespace(amount=0.01)
        engine = make_engine(
            bot, rates=[0.0001, 0.0001, 0.0001], positions=[position]
        )

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "hold"
        assert "holding" in signal.reason

    @pytest.mark.asyncio
    async def test_cooldown_blocks_reentry(self):
        bot = make_bot()
        engine = make_engine(bot, rates=[0.0001, 0.0001, 0.0001])
        engine._funding_states = {bot.id: {"last_exit_time": datetime.utcnow()}}

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "hold"
        assert "cooldown" in signal.reason

    @pytest.mark.asyncio
    async def test_invalid_band_holds(self):
        bot = make_bot({"min_funding_rate": 0.001, "max_funding_rate": -0.001})
        engine = make_engine(bot, rates=[0.0001, 0.0001, 0.0001])

        signal = await engine._strategy_funding_carry(bot, 160.0, bot.strategy_params, None)

        assert signal.action == "hold"
        assert "invalid config" in signal.reason
