"""Regression tests for the application startup auto-resume path.

Defect: ``app/main.py`` called
``trading_engine.trading_engine.resume_bots_on_startup()``. But
``from app.services import trading_engine`` resolves to the TradingEngine
*singleton instance* (``app/services/__init__.py`` re-exports it under that
name, shadowing the submodule), so the extra ``.trading_engine`` hop raised
``AttributeError: 'TradingEngine' object has no attribute 'trading_engine'``,
which the lifespan handler swallowed as
``WARNING: Failed to resume bots: ...`` — bots were never resumed after a
restart (and, via the identical bug at shutdown, state was never saved).

These tests pin both the integration contract main.py relies on and the
end-to-end resume of a persisted RUNNING bot.
"""

from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import Bot, BotStatus
from app.services.trading_engine import TradingEngine


class _Ctx:
    """Async context manager yielding a fixed session (does not close it)."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


def test_startup_imports_engine_singleton_directly():
    """The name main.py imports must BE the engine, callable without a hop.

    Guards the exact regression: ``from app.services import trading_engine``
    is the singleton instance; ``resume_bots_on_startup`` /
    ``graceful_shutdown`` are methods on it; and the erroneous
    ``trading_engine.trading_engine`` double-hop is invalid.
    """
    from app.services import trading_engine

    assert isinstance(trading_engine, TradingEngine)
    assert callable(getattr(trading_engine, "resume_bots_on_startup", None))
    assert callable(getattr(trading_engine, "graceful_shutdown", None))
    # The old buggy access path must not resolve.
    assert not hasattr(trading_engine, "trading_engine")


@pytest.mark.asyncio
async def test_resume_starts_persisted_running_bot(test_db):
    """End-to-end: a persisted RUNNING bot is resumed on startup.

    Calls ``resume_bots_on_startup`` on the singleton object exactly as
    ``main.py`` does. Heavy/IO collaborators (exchange, order reconciliation,
    per-bot logging, the run loop) are stubbed so the test stays hermetic; the
    DB-touching resume logic runs for real against the in-memory test session.
    """
    from app.services import trading_engine  # the singleton, as main.py imports it

    bot = Bot(
        name="resume-me",
        trading_pair="BTC/USDT",
        strategy="funding_carry",
        strategy_params={},
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    await test_db.flush()
    bot_id = bot.id

    fake_exchange = AsyncMock()
    fake_exchange.connect = AsyncMock(return_value=True)
    fake_exchange.disconnect = AsyncMock(return_value=None)

    async def _noop_loop(_self, _bot_id):  # patched onto the class -> bound, gets self
        return None

    try:
        with patch(
            "app.services.trading_engine.async_session_maker",
            return_value=_Ctx(test_db),
        ), patch.object(
            TradingEngine, "_make_simulated_exchange", return_value=fake_exchange
        ), patch.object(
            TradingEngine, "_recover_bot_orders", new=AsyncMock()
        ), patch.object(
            TradingEngine, "_run_bot_loop", new=_noop_loop
        ), patch(
            "app.services.trading_engine.BotLoggingService"
        ), patch(
            "app.services.trading_engine.ensure_bot_log_directory"
        ):
            resumed = await trading_engine.resume_bots_on_startup()

        assert resumed == 1
        assert bot_id in trading_engine._running_bots
        # The bot stays RUNNING (resume only flips to STOPPED on failure).
        refreshed = await test_db.get(Bot, bot_id)
        assert refreshed.status == BotStatus.RUNNING
    finally:
        # Never leak singleton state into other tests.
        task = trading_engine._running_bots.pop(bot_id, None)
        if task is not None:
            task.cancel()
        trading_engine._exchange_services.pop(bot_id, None)
        trading_engine._stop_flags.pop(bot_id, None)
        trading_engine._bot_loggers.pop(bot_id, None)
        trading_engine.cleanup_bot_state(bot_id)


@pytest.mark.asyncio
async def test_resume_starts_persisted_recovery_mode_bot(test_db):
    """End-to-end: a persisted RECOVERY_MODE bot is resumed on startup.

    Regression guard for the defect where Recovery bots froze for days with 0
    paper trades: ``resume_bots_on_startup`` queried only ``BotStatus.RUNNING``,
    so after any server restart a RECOVERY_MODE bot got NO loop task and stopped
    evaluating the market entirely. RECOVERY_MODE is an *active* state and must
    be resumed exactly like RUNNING. If the query ever drops RECOVERY_MODE
    again, ``resumed`` is 0 and the bot is absent from ``_running_bots`` — this
    test fails.
    """
    from app.services import trading_engine  # the singleton, as main.py imports it

    bot = Bot(
        name="recover-me",
        trading_pair="BTC/USDT",
        strategy="funding_carry",
        strategy_params={},
        # Recovery state persisted exactly as _enter_recovery_mode writes it;
        # the loop restores it on the first iteration after resume.
        strategy_state={
            "recovery_mode": {
                "active": True,
                "entered_at": "2026-06-24T22:24:11",
                "trigger_reason": "3 consecutive losses",
                "paper_position": None,
                "paper_trades": [],
                "consecutive_paper_wins": 0,
            }
        },
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
        status=BotStatus.RECOVERY_MODE,
    )
    test_db.add(bot)
    await test_db.flush()
    bot_id = bot.id

    fake_exchange = AsyncMock()
    fake_exchange.connect = AsyncMock(return_value=True)
    fake_exchange.disconnect = AsyncMock(return_value=None)

    async def _noop_loop(_self, _bot_id):  # patched onto the class -> bound, gets self
        return None

    try:
        with patch(
            "app.services.trading_engine.async_session_maker",
            return_value=_Ctx(test_db),
        ), patch.object(
            TradingEngine, "_make_simulated_exchange", return_value=fake_exchange
        ), patch.object(
            TradingEngine, "_recover_bot_orders", new=AsyncMock()
        ), patch.object(
            TradingEngine, "_run_bot_loop", new=_noop_loop
        ), patch(
            "app.services.trading_engine.BotLoggingService"
        ), patch(
            "app.services.trading_engine.ensure_bot_log_directory"
        ):
            resumed = await trading_engine.resume_bots_on_startup()

        assert resumed == 1, "RECOVERY_MODE bot must be resumed on startup"
        assert bot_id in trading_engine._running_bots, (
            "a loop task must be created for the recovery bot — without it the "
            "bot never evaluates the market again"
        )
        # Resume must not change the status: still recovering, still paper-trading.
        refreshed = await test_db.get(Bot, bot_id)
        assert refreshed.status == BotStatus.RECOVERY_MODE
    finally:
        task = trading_engine._running_bots.pop(bot_id, None)
        if task is not None:
            task.cancel()
        trading_engine._exchange_services.pop(bot_id, None)
        trading_engine._stop_flags.pop(bot_id, None)
        trading_engine._bot_loggers.pop(bot_id, None)
        trading_engine._recovery_states.pop(bot_id, None)
        trading_engine.cleanup_bot_state(bot_id)


@pytest.mark.asyncio
async def test_resume_returns_zero_when_no_running_bots(test_db):
    """No persisted RUNNING bots -> resume is a no-op returning 0."""
    from app.services import trading_engine

    with patch(
        "app.services.trading_engine.async_session_maker",
        return_value=_Ctx(test_db),
    ):
        assert await trading_engine.resume_bots_on_startup() == 0


@pytest.mark.asyncio
async def test_lifespan_invokes_engine_resume_and_shutdown():
    """The app lifespan must actually reach the engine resume/shutdown calls.

    This is the precise guard for the reported defect: main.py used to call
    ``trading_engine.trading_engine.resume_bots_on_startup()``, whose
    ``AttributeError`` was swallowed by the surrounding ``try/except`` (bot
    never resumed). Here the startup IO collaborators are stubbed and the
    engine methods are spies; driving the lifespan must await both. A
    reintroduced ``.trading_engine`` hop would raise before the spy is reached
    and fail this test.
    """
    import app.main as main_mod

    resume_spy = AsyncMock(return_value=1)
    shutdown_spy = AsyncMock(return_value=1)

    with ExitStack() as es:
        # Spy on the engine methods main.py is supposed to call.
        es.enter_context(patch.object(TradingEngine, "resume_bots_on_startup", new=resume_spy))
        es.enter_context(patch.object(TradingEngine, "graceful_shutdown", new=shutdown_spy))
        # Stub the heavy/side-effecting startup + shutdown collaborators.
        es.enter_context(patch.object(main_mod, "init_db", new=AsyncMock()))
        es.enter_context(patch.object(main_mod, "binding_failsafe_error", return_value=None))
        es.enter_context(patch.object(main_mod, "get_api_token", return_value=None))
        es.enter_context(patch.object(main_mod.config_service, "load_and_validate", return_value={}))
        es.enter_context(patch.object(main_mod.config_service, "get", return_value="127.0.0.1"))
        es.enter_context(patch.object(main_mod.db_backup_service, "backup_once", new=AsyncMock()))
        es.enter_context(patch.object(main_mod.db_backup_service, "start", new=Mock()))
        es.enter_context(patch.object(main_mod.db_backup_service, "stop", new=AsyncMock()))
        es.enter_context(patch.object(main_mod.ws_manager, "start", new=AsyncMock()))
        es.enter_context(patch.object(main_mod.ws_manager, "stop", new=AsyncMock()))

        async with main_mod.lifespan(main_mod.app):
            pass

    resume_spy.assert_awaited_once()
    shutdown_spy.assert_awaited_once()
