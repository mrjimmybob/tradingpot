"""Regression tests for the mean-reversion "holding" status formatter.

Root cause covered here:
A conditional was placed *inside* an f-string format spec on the holding-status
line of ``_strategy_mean_reversion``::

    f"... stop ${hard_stop:.2f if hard_stop else 'N/A'} ..."

Python treats everything after ``:`` as the format specifier, so this raised
``ValueError: Invalid format specifier '.2f if hard_stop else 'N/A'' for object
of type 'float'`` every time the strategy held an OPEN position (the only code
path that reaches this line). Because the exception escaped strategy evaluation,
the execution loop's failure circuit breaker counted it as a consecutive
failure and, after the threshold, paused the bot.

These tests:
1. Execute the affected hold path with ``hard_stop`` as a float and as ``None``
   and assert no exception is raised (and the stop is rendered sensibly).
2. Verify a single isolated exception cannot pause a bot (the breaker requires
   ``max_consecutive_failures`` *consecutive* failures and resets on success).
3. Verify the decision-status publish/log block is isolated so a presentation
   error there cannot trip the breaker.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import BotStatus
from app.services.trading_engine import TradeSignal, TradingEngine


def _bar(close: float) -> dict:
    """A completed pseudo-bar with flat OHLC at ``close``."""
    return {
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "start_ts": datetime.utcnow(),
    }


def _seed_holding_state(engine: TradingEngine, bot_id: int, hard_stop) -> None:
    """Seed mean-reversion state so the strategy reaches the holding return.

    Bars [100, 100, 90] give SMA ~= 96.67 with last close 90, so the
    mean-reached exit does not fire; ``bars_since_entry`` is below the time
    stop; and (when set) ``hard_stop`` is below the evaluation price so the
    hard-stop exit does not fire either. The only remaining branch is the
    "holding" return that builds the previously-broken status string.
    """
    if not hasattr(engine, "_mean_reversion_states"):
        engine._mean_reversion_states = {}
    engine._mean_reversion_states[bot_id] = {
        "bars": [_bar(100.0), _bar(100.0), _bar(90.0)],
        "current_bar": None,
        "entry_price": 90.0,
        "entry_atr": 1.0,
        "hard_stop": hard_stop,
        "bars_since_entry": 1,
        "last_exit_time": None,
    }


_HOLD_PARAMS = {
    "bollinger_period": 3,
    "atr_period": 3,
    "bar_interval_seconds": 600,  # current bar never completes mid-test
    "max_hold_bars": 10,
    "exit_at_mean": True,
    "regime_filter_enabled": False,  # avoids regime helpers; force_exit stays False
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hard_stop, expected_stop_text",
    [
        (80.0, "stop $80.00"),   # hard_stop = float
        (None, "stop N/A"),      # hard_stop = None
    ],
)
async def test_mean_reversion_holding_status_does_not_raise(hard_stop, expected_stop_text):
    engine = TradingEngine()
    bot = SimpleNamespace(id=1, trading_pair="BTC/USDT")
    _seed_holding_state(engine, bot.id, hard_stop)

    # Hold path requires an open position; the conditions never consult the
    # position object itself, so a lightweight stand-in is sufficient.
    engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.01)])

    # Must not raise (was: ValueError "Invalid format specifier ...").
    signal = await engine._strategy_mean_reversion(
        bot, current_price=95.0, params=dict(_HOLD_PARAMS), session=None
    )

    assert signal is not None
    assert signal.action == "hold"
    assert expected_stop_text in signal.reason
    assert "Holding" in signal.reason


class _Ctx:
    """Async context manager yielding a fixed session (does not close it)."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


class _SeqSessionMaker:
    """Yield a different session context per ``async_session_maker()`` call."""

    def __init__(self, *sessions):
        self._sessions = sessions
        self._i = 0

    def __call__(self):
        session = self._sessions[min(self._i, len(self._sessions) - 1)]
        self._i += 1
        return _Ctx(session)


@pytest.mark.asyncio
async def test_single_exception_does_not_pause_bot():
    """One isolated exception (e.g. a formatting bug) must NOT pause a bot.

    The circuit breaker only trips after ``max_consecutive_failures`` failures
    IN A ROW. A single failure followed by a clean iteration leaves the bot
    running and never calls ``_pause_bot_for_failures``.
    """
    from app.services.config import config_service

    engine = TradingEngine()
    bot_id = 1
    engine._stop_flags[bot_id] = False

    pause = AsyncMock()
    engine._pause_bot_for_failures = pause

    # Iteration 1: raise the exact formatting error class/shape, once.
    failing = AsyncMock()
    failing.execute = AsyncMock(
        side_effect=ValueError(
            "Invalid format specifier '.2f if hard_stop else 'N/A'' for object of type 'float'"
        )
    )

    # Iteration 2: bot is no longer RUNNING -> loop breaks cleanly (a stand-in
    # for "a subsequent healthy iteration"), so the breaker never accumulates.
    stopped_bot = SimpleNamespace(id=bot_id, status=BotStatus.PAUSED, trading_pair="BTC/USDT")
    ok_result = Mock()
    ok_result.scalar_one_or_none = Mock(return_value=stopped_bot)
    healthy = AsyncMock()
    healthy.execute = AsyncMock(return_value=ok_result)

    def cfg_get(key):
        if "max_consecutive_failures" in key:
            return 10
        if "failure_backoff_max_seconds" in key:
            return 60
        return None

    with patch("app.services.trading_engine.async_session_maker",
               new=_SeqSessionMaker(failing, healthy)), \
         patch.object(config_service, "get", side_effect=cfg_get), \
         patch("app.services.trading_engine.asyncio.sleep", new=AsyncMock()):
        await engine._run_bot_loop(bot_id)

    pause.assert_not_awaited()


@pytest.mark.asyncio
async def test_decision_status_publish_error_cannot_trip_breaker():
    """A failure while publishing/logging decision status is presentation-only
    and must not bubble to the failure circuit breaker.

    The store's ``update_from_signal`` is forced to raise; the loop must keep
    going and not pause the bot for it.
    """
    from app.services.config import config_service

    engine = TradingEngine()
    bot_id = 1
    engine._stop_flags[bot_id] = False

    pause = AsyncMock()
    engine._pause_bot_for_failures = pause

    running_bot = SimpleNamespace(
        id=bot_id, status=BotStatus.RUNNING, trading_pair="BTC/USDT",
        is_dry_run=True, total_pnl=0.0, name="b",
    )
    result = Mock()
    result.scalar_one_or_none = Mock(return_value=running_bot)
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()

    # Risk check passes; exchange returns a ticker; strategy holds.
    risk_ok = SimpleNamespace(action=None, reason="")
    exchange = AsyncMock()
    exchange.get_ticker = AsyncMock(return_value=SimpleNamespace(last=100.0))
    engine._exchange_services[bot_id] = exchange
    engine._execute_strategy = AsyncMock(
        return_value=TradeSignal(action="hold", amount=0, reason="holding")
    )

    # Stop after one full healthy iteration so the loop terminates.
    async def _stop_after(*_a, **_k):
        engine._stop_flags[bot_id] = True

    engine._check_positions_stop_loss = AsyncMock(side_effect=_stop_after)
    engine._take_pnl_snapshot = AsyncMock()
    engine._resolve_pending_orders = AsyncMock()
    engine._save_bot_state = AsyncMock()
    engine._reconcile_live_account = AsyncMock()

    def cfg_get(key):
        if "max_consecutive_failures" in key:
            return 10
        if "failure_backoff_max_seconds" in key:
            return 60
        return None

    with patch("app.services.trading_engine.async_session_maker",
               return_value=_Ctx(session)), \
         patch.object(config_service, "get", side_effect=cfg_get), \
         patch("app.services.trading_engine.asyncio.sleep", new=AsyncMock()), \
         patch("app.services.trading_engine.RiskManagementService") as MockRisk, \
         patch("app.services.trading_engine.VirtualWalletService"), \
         patch("app.services.trading_engine.decision_status_store") as mock_store:
        MockRisk.return_value.full_risk_check = AsyncMock(return_value=risk_ok)
        mock_store.update_from_signal = Mock(side_effect=RuntimeError("boom in status formatter"))
        await engine._run_bot_loop(bot_id)

    # The presentation error was swallowed: the breaker never tripped.
    pause.assert_not_awaited()
