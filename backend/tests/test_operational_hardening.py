"""Tests for operational/observability/config fixes: M1, M2, M3, L1, L3, L4.

- M1: the loop trips a circuit breaker (pause + alert) after consecutive failures.
- M2: operational alerts are persisted (circuit breaker, accounting failure).
- M3: per-strategy cross-field config validation is wired into create/update.
- L1: funding_carry warns once when funding data is unavailable.
- L3: the trades CSV path is absolute and CWD-independent.
- L4: in-memory per-bot state is released on cleanup.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import select

from app.models import Alert, Bot, BotStatus
from app.routers.bots import validate_strategy_params
from app.services.exchange import ExchangeService, FundingRate
from app.services.trading_engine import TradingEngine


class _Ctx:
    """Async context manager yielding a fixed session (does not close it)."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


async def _make_db_bot(test_db, strategy="funding_carry"):
    bot = Bot(
        name="b", trading_pair="BTC/USDT", strategy=strategy, strategy_params={},
        budget=1000.0, current_balance=1000.0, is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    await test_db.flush()
    return bot


# ============================================================================
# M1 - Failure circuit breaker
# ============================================================================


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_loop_trips_breaker_after_threshold(self):
        from app.services.config import config_service

        engine = TradingEngine()
        bot_id = 1
        engine._stop_flags[bot_id] = False

        bad_session = AsyncMock()
        bad_session.execute = AsyncMock(side_effect=RuntimeError("db down"))

        pause = AsyncMock()

        async def _stop(*_a, **_k):
            engine._stop_flags[bot_id] = True

        pause.side_effect = _stop
        engine._pause_bot_for_failures = pause

        def cfg_get(key):
            if "max_consecutive_failures" in key:
                return 3
            if "failure_backoff_max_seconds" in key:
                return 60
            return None

        with patch("app.services.trading_engine.async_session_maker",
                   return_value=_Ctx(bad_session)), \
             patch.object(config_service, "get", side_effect=cfg_get), \
             patch("app.services.trading_engine.asyncio.sleep", new=AsyncMock()):
            await engine._run_bot_loop(bot_id)

        pause.assert_awaited_once()
        # Tripped exactly at the configured threshold.
        assert pause.await_args.args[1] == 3

    @pytest.mark.asyncio
    async def test_pause_for_failures_pauses_and_alerts(self, test_db):
        bot = await _make_db_bot(test_db)
        engine = TradingEngine()
        engine._stop_flags[bot.id] = False

        with patch("app.services.trading_engine.async_session_maker",
                   return_value=_Ctx(test_db)):
            await engine._pause_bot_for_failures(bot.id, 7, "kaboom")

        refreshed = (
            await test_db.execute(select(Bot).where(Bot.id == bot.id))
        ).scalar_one()
        assert refreshed.status == BotStatus.PAUSED
        assert engine._stop_flags[bot.id] is True

        alerts = (
            await test_db.execute(select(Alert).where(Alert.bot_id == bot.id))
        ).scalars().all()
        assert any(a.alert_type == "failure_circuit_breaker" for a in alerts)


# ============================================================================
# M2 - Operational alerting
# ============================================================================


class TestAlerting:
    @pytest.mark.asyncio
    async def test_emit_alert_persists_without_email(self, test_db):
        bot = await _make_db_bot(test_db)
        engine = TradingEngine()

        await engine._emit_alert(test_db, bot.id, "test_alert", "hello there")

        alerts = (
            await test_db.execute(select(Alert).where(Alert.bot_id == bot.id))
        ).scalars().all()
        assert len(alerts) == 1
        assert alerts[0].alert_type == "test_alert"
        assert alerts[0].message == "hello there"
        assert alerts[0].email_sent is False  # email disabled in tests

    @pytest.mark.asyncio
    async def test_emit_alert_never_raises_on_db_error(self):
        engine = TradingEngine()
        session = AsyncMock()
        session.add = Mock(side_effect=RuntimeError("boom"))
        # Must swallow the error - alerting cannot take down the loop.
        await engine._emit_alert(session, 1, "t", "m")


# ============================================================================
# M3 - Cross-field config validation wired into the router
# ============================================================================


class TestStrategyValidation:
    def test_inverted_funding_band_rejected(self):
        errors = validate_strategy_params(
            "funding_carry", {"min_funding_rate": 0.001, "max_funding_rate": -0.001}
        )
        assert any("min_funding_rate" in e for e in errors)

    def test_valid_funding_band_accepted(self):
        assert validate_strategy_params(
            "funding_carry", {"min_funding_rate": -0.0005, "max_funding_rate": 0.0005}
        ) == []

    def test_strategy_without_cross_field_validator_unaffected(self):
        assert validate_strategy_params("trend_following", {"short_period": 50}) == []


# ============================================================================
# L1 - funding_carry warns once when data unavailable
# ============================================================================


class _FundingExchange:
    def __init__(self, rates):
        self._rates = rates

    def to_swap_symbol(self, symbol):
        return ExchangeService.to_swap_symbol(symbol)

    async def get_funding_rate_history(self, symbol, limit=200, since=None):
        return [FundingRate(symbol, r, datetime.utcnow()) for r in self._rates]


class TestFundingUnavailableWarning:
    @pytest.mark.asyncio
    async def test_warns_once_then_relatches_when_data_returns(self):
        engine = TradingEngine()
        bot = Bot(
            id=1, name="f", trading_pair="BTC/USDT", strategy="funding_carry",
            strategy_params={}, budget=1000.0, current_balance=1000.0,
            is_dry_run=True, status=BotStatus.RUNNING,
        )
        engine._exchange_services[bot.id] = _FundingExchange([])  # no data -> None
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._price_histories = {bot.id: [100.0] * 60}
        bot_logger = Mock()
        engine._bot_loggers[bot.id] = bot_logger

        s1 = await engine._strategy_funding_carry(bot, 100.0, {}, None)
        s2 = await engine._strategy_funding_carry(bot, 100.0, {}, None)

        assert s1.action == "hold" and "unavailable" in s1.reason
        assert s2.action == "hold"
        assert bot.id in engine._funding_unavailable_warned
        bot_logger.log_activity.assert_called_once()  # warned once, not twice

        # When funding data returns, the latch clears so a later outage warns again.
        engine._exchange_services[bot.id] = _FundingExchange([0.0001, 0.0001, 0.0001])
        await engine._strategy_funding_carry(bot, 100.0, {}, None)
        assert bot.id not in engine._funding_unavailable_warned


# ============================================================================
# L3 - Absolute trades CSV path
# ============================================================================


class TestTradesCsvPath:
    def test_path_is_absolute_and_named_by_mode(self):
        engine = TradingEngine()
        sim = engine._trades_csv_path(SimpleNamespace(id=99, is_dry_run=True))
        assert sim.is_absolute()
        assert sim.name == "trades_simulated.csv"
        assert "logs" in sim.parts

        live = engine._trades_csv_path(SimpleNamespace(id=99, is_dry_run=False))
        assert live.is_absolute()
        assert live.name == "trades_live.csv"


# ============================================================================
# L4 - In-memory state cleanup
# ============================================================================


class TestStateCleanup:
    def test_cleanup_clears_only_target_bot(self):
        engine = TradingEngine()
        bid, other = 5, 6
        engine._trend_states = {bid: {"x": 1}, other: {"y": 2}}
        engine._price_histories = {bid: [1.0]}
        engine._funding_states = {bid: {"last_exit_time": None}}
        engine._funding_cache = {bid: {"mean_rate": 0.0}}
        engine._last_pending_resolve = {bid: datetime.utcnow()}
        engine._bot_loggers = {bid: Mock()}
        engine._funding_unavailable_warned = {bid}

        engine.cleanup_bot_state(bid)

        assert bid not in engine._trend_states
        assert other in engine._trend_states  # other bots untouched
        assert bid not in engine._price_histories
        assert bid not in engine._funding_states
        assert bid not in engine._funding_cache
        assert bid not in engine._last_pending_resolve
        assert bid not in engine._bot_loggers
        assert bid not in engine._funding_unavailable_warned
