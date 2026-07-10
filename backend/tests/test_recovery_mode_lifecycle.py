"""Integration tests for RECOVERY_MODE lifecycle.

Validates the full recovery flow end-to-end:
  1. 3 consecutive losses trigger ENTER_RECOVERY_MODE (not PAUSE).
  2. Bot enters RECOVERY_MODE: state persisted, diagnostics updated.
  3. Paper trades are generated instead of real orders.
  4. Paper wins/losses are recorded in state and diagnostics.
  5. When exit criteria are met, bot returns to RUNNING.
  6. No real orders are placed during recovery.
  7. Virtual position is returned by _get_bot_positions during recovery.
  8. Recovery state restored after a simulated restart.
"""

from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from sqlalchemy import select

from app.models import Bot, BotStatus, Order, OrderStatus, OrderType
from app.services.diagnostics import DiagnosticsStore
from app.services.decision_status import DecisionState, DecisionStatusStore
from app.services.risk_management import RiskAction, RiskAssessment, RiskManagementService
from app.services.trading_engine import TradingEngine, TradeSignal, _VirtualPosition


# ============================================================================
# Helpers
# ============================================================================


def _bot(strategy="dca_accumulator", status=BotStatus.RUNNING, bot_id=99) -> Bot:
    b = Bot(
        name="test-bot",
        trading_pair="BTC/USDT",
        strategy=strategy,
        strategy_params={},
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
        status=status,
        strategy_state=None,
    )
    b.id = bot_id
    return b


def _signal(action: str, amount: float = 100.0) -> TradeSignal:
    return TradeSignal(action=action, amount=amount, reason=f"{action} test signal")


async def _make_real_bot(test_db, strategy="dca_accumulator") -> Bot:
    b = Bot(
        name="recovery-test",
        trading_pair="BTC/USDT",
        strategy=strategy,
        strategy_params={},
        budget=500.0,
        current_balance=500.0,
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    async with test_db as session:
        session.add(b)
        await session.commit()
        await session.refresh(b)
    return b


# ============================================================================
# 1. Risk check: consecutive losses → ENTER_RECOVERY_MODE (not PAUSE)
# ============================================================================


class TestConsecutiveLossesEnterRecovery:
    """Risk-management layer returns ENTER_RECOVERY_MODE, never PAUSE_BOT,
    for consecutive-loss threshold violations on fixed-strategy bots."""

    @pytest.mark.asyncio
    async def test_three_losses_return_enter_recovery_mode(self, test_db):
        """3 consecutive losses from the DB trigger ENTER_RECOVERY_MODE."""
        from tests.test_risk_management import create_mock_bot, create_mock_realized_gain
        from unittest.mock import Mock, AsyncMock

        session = AsyncMock()
        bot = create_mock_bot(strategy="dca_accumulator")
        bot_result = AsyncMock()
        bot_result.scalar_one_or_none = Mock(return_value=bot)

        gains = [
            create_mock_realized_gain(gain_loss=-30.0),
            create_mock_realized_gain(gain_loss=-20.0),
            create_mock_realized_gain(gain_loss=-10.0),
        ]
        gains_result = AsyncMock()
        gains_result.scalars = Mock(return_value=Mock(all=Mock(return_value=gains)))
        session.execute.side_effect = [bot_result, gains_result]

        svc = RiskManagementService(session)
        count, result = await svc.check_consecutive_losses(bot_id=1, threshold=3)

        assert count == 3
        assert result.action == RiskAction.ENTER_RECOVERY_MODE
        assert "recovery mode" in result.reason.lower()
        assert result.action != RiskAction.PAUSE_BOT

    @pytest.mark.asyncio
    async def test_auto_mode_continues_on_losses(self, test_db):
        """auto_mode self-manages; it must still return CONTINUE."""
        from tests.test_risk_management import create_mock_bot, create_mock_realized_gain
        from unittest.mock import Mock, AsyncMock

        session = AsyncMock()
        bot = create_mock_bot(strategy="auto_mode")
        bot_result = AsyncMock()
        bot_result.scalar_one_or_none = Mock(return_value=bot)
        gains_result = AsyncMock()
        gains_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[
            create_mock_realized_gain(gain_loss=-10.0),
            create_mock_realized_gain(gain_loss=-10.0),
            create_mock_realized_gain(gain_loss=-10.0),
        ])))
        session.execute.side_effect = [bot_result, gains_result]

        svc = RiskManagementService(session)
        _, result = await svc.check_consecutive_losses(bot_id=1, threshold=3)
        assert result.action == RiskAction.CONTINUE


# ============================================================================
# 2. _enter_recovery_mode: state and diagnostics
# ============================================================================


class TestEnterRecoveryMode:
    @pytest.mark.asyncio
    async def test_enter_sets_status_and_state(self, test_db):
        """_enter_recovery_mode changes bot.status and persists recovery_state."""
        engine = TradingEngine()
        bot = _bot()
        session = AsyncMock()
        session.commit = AsyncMock()

        diag_store = DiagnosticsStore()
        ds_store = DecisionStatusStore()

        with (
            patch("app.services.trading_engine.diagnostics_store", diag_store),
            patch("app.services.trading_engine.decision_status_store", ds_store),
            patch("app.services.trading_engine.email_service") as mock_email,
        ):
            await engine._enter_recovery_mode(bot, bot.id, "3 losses", session)

        assert bot.status == BotStatus.RECOVERY_MODE
        assert bot.id in engine._recovery_states
        state = engine._recovery_states[bot.id]
        assert state["active"] is True
        assert state["trigger_reason"] == "3 losses"
        assert state["paper_position"] is None
        assert state["paper_trades"] == []
        assert state["consecutive_paper_wins"] == 0

        d = diag_store.get(bot.id)
        assert d.recovery_is_active is True
        assert d.recovery_reason == "3 losses"

        status = ds_store.get(bot.id)
        assert status is not None
        assert status.state == DecisionState.RECOVERY_MODE_PAPER_TRADING

        mock_email.send_bot_paused_alert.assert_called_once()


# ============================================================================
# 3. _process_paper_trade: no real orders, state tracking
# ============================================================================


class TestPaperTradeProcessing:
    def _engine_with_recovery(self, bot_id: int, entry_price: Optional[float] = None):
        engine = TradingEngine()
        recovery = {
            "active": True,
            "entered_at": datetime.utcnow().isoformat(),
            "trigger_reason": "test losses",
            "paper_position": (
                {
                    "entry_price": entry_price,
                    "amount_usd": 100.0,
                    "trading_pair": "BTC/USDT",
                    "entered_at": datetime.utcnow().isoformat(),
                }
                if entry_price is not None
                else None
            ),
            "paper_trades": [],
            "consecutive_paper_wins": 0,
        }
        engine._recovery_states[bot_id] = recovery
        return engine, recovery

    @pytest.mark.asyncio
    async def test_buy_signal_opens_paper_position(self):
        """A BUY signal during recovery opens a paper position, no real order."""
        engine, recovery = self._engine_with_recovery(1)
        bot = _bot(bot_id=1)
        session = AsyncMock()
        session.commit = AsyncMock()

        diag_store = DiagnosticsStore()
        ds_store = DecisionStatusStore()

        with (
            patch("app.services.trading_engine.diagnostics_store", diag_store),
            patch("app.services.trading_engine.decision_status_store", ds_store),
        ):
            await engine._process_paper_trade(bot, 1, _signal("buy", 100.0), 50000.0, session)

        assert recovery["paper_position"] is not None
        assert recovery["paper_position"]["entry_price"] == 50000.0
        assert recovery["paper_position"]["amount_usd"] == 100.0

    @pytest.mark.asyncio
    async def test_sell_signal_closes_paper_position_win(self):
        """A SELL at a higher price records a winning paper trade."""
        engine, recovery = self._engine_with_recovery(1, entry_price=50000.0)
        bot = _bot(bot_id=1)
        session = AsyncMock()
        session.commit = AsyncMock()

        diag_store = DiagnosticsStore()
        ds_store = DecisionStatusStore()

        with (
            patch("app.services.trading_engine.diagnostics_store", diag_store),
            patch("app.services.trading_engine.decision_status_store", ds_store),
        ):
            # Price went up 5% → should be a win after fees
            await engine._process_paper_trade(bot, 1, _signal("sell"), 52500.0, session)

        assert recovery["paper_position"] is None  # closed
        assert len(recovery["paper_trades"]) == 1
        trade = recovery["paper_trades"][0]
        assert trade["win"] is True
        assert trade["gain_loss_usd"] > 0
        assert recovery["consecutive_paper_wins"] == 1

        d = diag_store.get(1)
        assert d.recovery_win_count == 1
        assert d.recovery_loss_count == 0

    @pytest.mark.asyncio
    async def test_sell_signal_records_loss(self):
        """A SELL at a lower price records a losing paper trade."""
        engine, recovery = self._engine_with_recovery(1, entry_price=50000.0)
        bot = _bot(bot_id=1)
        session = AsyncMock()
        session.commit = AsyncMock()

        diag_store = DiagnosticsStore()
        ds_store = DecisionStatusStore()

        with (
            patch("app.services.trading_engine.diagnostics_store", diag_store),
            patch("app.services.trading_engine.decision_status_store", ds_store),
        ):
            await engine._process_paper_trade(bot, 1, _signal("sell"), 47000.0, session)

        trade = recovery["paper_trades"][0]
        assert trade["win"] is False
        assert trade["gain_loss_usd"] < 0
        assert recovery["consecutive_paper_wins"] == 0

        d = diag_store.get(1)
        assert d.recovery_loss_count == 1
        assert d.recovery_win_count == 0

    @pytest.mark.asyncio
    async def test_two_consecutive_wins_exit_recovery(self):
        """Two consecutive paper wins trigger _exit_recovery_mode."""
        engine, recovery = self._engine_with_recovery(1)
        bot = _bot(bot_id=1)

        exited = []

        async def fake_exit(b, bid, reason, sess):
            exited.append(reason)
            b.status = BotStatus.RUNNING

        engine._exit_recovery_mode = fake_exit

        diag_store = DiagnosticsStore()
        ds_store = DecisionStatusStore()

        with (
            patch("app.services.trading_engine.diagnostics_store", diag_store),
            patch("app.services.trading_engine.decision_status_store", ds_store),
        ):
            session = AsyncMock()
            session.commit = AsyncMock()

            # Paper BUY at 50000
            await engine._process_paper_trade(bot, 1, _signal("buy", 100.0), 50000.0, session)
            # Paper SELL at 52000 (win #1)
            await engine._process_paper_trade(bot, 1, _signal("sell"), 52000.0, session)
            # Paper BUY again
            await engine._process_paper_trade(bot, 1, _signal("buy", 100.0), 52000.0, session)
            # Paper SELL at 54000 (win #2 → exit criteria met)
            await engine._process_paper_trade(bot, 1, _signal("sell"), 54000.0, session)

        assert len(exited) == 1, "Recovery should have exited exactly once"
        assert "consecutive" in exited[0].lower()


# ============================================================================
# 4. _exit_recovery_mode: restores live trading
# ============================================================================


class TestExitRecoveryMode:
    @pytest.mark.asyncio
    async def test_exit_restores_running_status(self):
        engine = TradingEngine()
        bot = _bot(status=BotStatus.RECOVERY_MODE)
        engine._recovery_states[bot.id] = {"active": True, "paper_trades": []}

        ss = {"recovery_mode": {"active": True}}
        bot.strategy_state = ss

        session = AsyncMock()
        session.commit = AsyncMock()

        diag_store = DiagnosticsStore()
        ds_store = DecisionStatusStore()

        with (
            patch("app.services.trading_engine.diagnostics_store", diag_store),
            patch("app.services.trading_engine.decision_status_store", ds_store),
        ):
            await engine._exit_recovery_mode(bot, bot.id, "2 consecutive wins", session)

        assert bot.status == BotStatus.RUNNING
        assert bot.id not in engine._recovery_states
        assert "recovery_mode" not in (bot.strategy_state or {})

        d = diag_store.get(bot.id)
        assert d.recovery_is_active is False

        status = ds_store.get(bot.id)
        assert status is not None
        assert status.state == DecisionState.EVALUATING


# ============================================================================
# 5. Virtual position: _get_bot_positions returns virtual during recovery
# ============================================================================


class TestVirtualPosition:
    @pytest.mark.asyncio
    async def test_virtual_position_returned_during_recovery(self):
        """When recovery has an open paper position, _get_bot_positions
        prepends a _VirtualPosition so strategies produce SELL signals."""
        engine = TradingEngine()
        bot_id = 7

        engine._recovery_states[bot_id] = {
            "active": True,
            "paper_position": {
                "entry_price": 50000.0,
                "amount_usd": 200.0,
                "trading_pair": "BTC/USDT",
                "entered_at": datetime.utcnow().isoformat(),
            },
            "paper_trades": [],
            "consecutive_paper_wins": 0,
        }

        session = AsyncMock()
        # DB has no real positions
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
        session.execute = AsyncMock(return_value=mock_result)

        positions = await engine._get_bot_positions(bot_id, session)
        assert len(positions) == 1
        virt = positions[0]
        assert isinstance(virt, _VirtualPosition)
        assert virt.entry_price == 50000.0
        assert virt.trading_pair == "BTC/USDT"
        # amount should be base units (USD / price)
        assert abs(virt.amount - 200.0 / 50000.0) < 1e-8

    @pytest.mark.asyncio
    async def test_no_virtual_position_when_no_open_paper_pos(self):
        """If recovery is active but no paper position open, no virtual row injected."""
        engine = TradingEngine()
        bot_id = 8
        engine._recovery_states[bot_id] = {
            "active": True,
            "paper_position": None,
            "paper_trades": [],
            "consecutive_paper_wins": 0,
        }

        session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
        session.execute = AsyncMock(return_value=mock_result)

        positions = await engine._get_bot_positions(bot_id, session)
        assert positions == []

    @pytest.mark.asyncio
    async def test_no_virtual_position_when_not_in_recovery(self):
        """Without a recovery state entry, result is just the real DB positions."""
        engine = TradingEngine()
        bot_id = 9

        session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
        session.execute = AsyncMock(return_value=mock_result)

        positions = await engine._get_bot_positions(bot_id, session)
        assert positions == []


# ============================================================================
# 6. State restoration after restart
# ============================================================================


class TestRecoveryStateRestoration:
    def test_recovery_state_restored_from_strategy_state(self):
        """On restart the in-memory _recovery_states is empty; the main loop
        must re-populate it from bot.strategy_state on first tick.
        This test simulates that logic directly."""
        engine = TradingEngine()
        bot_id = 10

        # Simulate bot loaded from DB with persisted recovery state
        bot = _bot(status=BotStatus.RECOVERY_MODE, bot_id=bot_id)
        persisted_rm = {
            "active": True,
            "entered_at": "2026-06-24T10:00:00",
            "trigger_reason": "3 consecutive losses",
            "paper_position": None,
            "paper_trades": [
                {"gain_loss_usd": 5.0, "win": True, "timestamp": "2026-06-24T10:05:00"},
            ],
            "consecutive_paper_wins": 1,
        }
        bot.strategy_state = {"recovery_mode": persisted_rm}

        # Simulate what the main loop does on first tick
        if bot.id not in engine._recovery_states and bot.status == BotStatus.RECOVERY_MODE:
            ss = bot.strategy_state or {}
            rm = ss.get("recovery_mode")
            if rm and rm.get("active"):
                engine._recovery_states[bot.id] = rm

        assert bot_id in engine._recovery_states
        state = engine._recovery_states[bot_id]
        assert state["consecutive_paper_wins"] == 1
        assert len(state["paper_trades"]) == 1


# ============================================================================
# 7. Recovery exit criteria
# ============================================================================


class TestRecoveryExitCriteriaEdgeCases:
    """Supplement the TestRecoveryExitCriteria tests in test_risk_management.py
    with a few more edge cases specifically relevant to the engine flow."""

    def _state(self, trades, consecutive_wins=0):
        return {
            "paper_trades": [{"gain_loss_usd": g, "win": g > 0} for g in trades],
            "consecutive_paper_wins": consecutive_wins,
        }

    def test_exit_check_wraps_risk_management_service(self):
        engine = TradingEngine()
        # Two consecutive wins: should exit
        state = self._state([-5, 3, 4], consecutive_wins=2)
        should_exit, reason = engine._check_recovery_exit(state)
        assert should_exit
        assert "consecutive" in reason

    def test_no_exit_below_thresholds(self):
        engine = TradingEngine()
        # 2 trades, 1 win, no consecutive wins — criteria not met
        state = self._state([-5, 3], consecutive_wins=1)
        should_exit, _ = engine._check_recovery_exit(state)
        assert not should_exit


# ============================================================================
# 8. Loop-level: a RECOVERY_MODE bot keeps evaluating the market every cycle
# ============================================================================


class _LoopCtx:
    """Async context manager yielding a fixed session (does not close it)."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        return False


class TestRecoveryLoopKeepsEvaluating:
    """Drive the REAL ``_run_bot_loop`` for a RECOVERY_MODE bot.

    Defends the core defect end-to-end at the loop level: once a recovery bot
    has a running task, it must call ``_execute_strategy`` and increment the
    evaluation heartbeat on EVERY cycle — never silently stop. (The companion
    guard that the task is even *created* after a restart lives in
    ``test_startup_resume.py::test_resume_starts_persisted_recovery_mode_bot``.)
    """

    @pytest.mark.asyncio
    async def test_loop_executes_strategy_and_counts_every_cycle(self, test_db):
        import asyncio as _asyncio

        from app.services.diagnostics import DiagnosticsStore

        bot = Bot(
            name="recovery-loop",
            trading_pair="BTC/USDT",
            strategy="dca_accumulator",
            strategy_params={},
            strategy_state={
                "recovery_mode": {
                    "active": True,
                    "entered_at": "2026-06-24T10:00:00",
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

        engine = TradingEngine()
        engine._stop_flags[bot_id] = False

        # Live ticker every tick.
        ticker = Mock(last=50000.0)
        fake_exchange = AsyncMock()
        fake_exchange.get_ticker = AsyncMock(return_value=ticker)
        engine._exchange_services[bot_id] = fake_exchange

        # Count strategy executions; stop the loop after the 3rd evaluation.
        calls = {"n": 0}

        async def fake_execute_strategy(_bot, _price, _session):
            calls["n"] += 1
            if calls["n"] >= 3:
                engine._stop_flags[bot_id] = True
            return _signal("hold")

        engine._execute_strategy = fake_execute_strategy

        # CONTINUE: bot is already recovering; no entry/exit transition this tick.
        cont = RiskAssessment(action=RiskAction.CONTINUE, reason="ok", details={})

        diag = DiagnosticsStore()
        ds = DecisionStatusStore()

        async def _noop(*a, **k):
            return None

        with (
            patch(
                "app.services.trading_engine.async_session_maker",
                return_value=_LoopCtx(test_db),
            ),
            patch("app.services.trading_engine.diagnostics_store", diag),
            patch("app.services.trading_engine.decision_status_store", ds),
            patch.object(
                RiskManagementService, "full_risk_check",
                AsyncMock(return_value=cont),
            ),
            patch.object(TradingEngine, "_check_positions_stop_loss", new=_noop),
            patch.object(TradingEngine, "_take_pnl_snapshot", new=_noop),
            patch.object(TradingEngine, "_resolve_pending_orders", new=_noop),
            patch.object(TradingEngine, "_save_bot_state", new=_noop),
            patch.object(TradingEngine, "_reconcile_live_account", new=_noop),
            patch("app.services.trading_engine.asyncio.sleep", new=AsyncMock()),
        ):
            await _asyncio.wait_for(engine._run_bot_loop(bot_id), timeout=5.0)

        # The strategy pipeline ran on every cycle...
        assert calls["n"] == 3
        # ...and each HOLD evaluation incremented the heartbeat the UI reads.
        assert diag.get(bot_id).total_evaluations == 3
        # Recovery state was restored from strategy_state on the first tick.
        assert bot_id in engine._recovery_states


# ============================================================================
# 9. Regression: recovery_mode must survive a normal strategy-state checkpoint
#
# Defect: bot.strategy_state["recovery_mode"] was written correctly by
# _enter_recovery_mode / _process_paper_trade, but _save_bot_state (the H-1
# periodic checkpoint that runs every tick, and graceful_shutdown) persisted
# `bot.strategy_state = self._collect_bot_state(bot_id)` — a WHOLESALE
# replacement built only from _PERSISTED_STATE_ATTRS, which never included
# recovery data. The very next checkpoint after entering RECOVERY_MODE erased
# recovery_mode from the DB entirely, so paper trades, entered_at and the
# recovery counters all read back NULL even though the bot kept evaluating
# and status stayed RECOVERY_MODE.
# ============================================================================


class TestRecoveryModeSurvivesNormalCheckpoint:
    """Required test (A): enter RECOVERY_MODE, persist, run a normal strategy
    save, assert strategy_state.recovery_mode still exists."""

    @pytest.mark.asyncio
    async def test_recovery_mode_survives_a_normal_strategy_state_checkpoint(self, test_db):
        bot = Bot(
            name="checkpoint-bot",
            trading_pair="BTC/USDT",
            strategy="dca_accumulator",
            strategy_params={},
            budget=1000.0,
            current_balance=1000.0,
            is_dry_run=True,
            status=BotStatus.RUNNING,
        )
        test_db.add(bot)
        await test_db.flush()
        bot_id = bot.id

        engine = TradingEngine()
        try:
            # Enter recovery: writes to both the in-memory dict (fast path) and
            # bot.strategy_state (durable path).
            await engine._enter_recovery_mode(
                bot, bot_id, "3 consecutive losses", test_db
            )
            await test_db.commit()

            refreshed = await test_db.get(Bot, bot_id)
            assert refreshed.strategy_state["recovery_mode"]["active"] is True

            # Unrelated strategy runtime state a normal tick would also
            # checkpoint (trailing stop, cooldowns, ...).
            engine._trend_states = {bot_id: {"trailing_stop": 95.0}}

            # This is exactly what the H-1 periodic checkpoint (and
            # graceful_shutdown) calls on every tick for a RUNNING/RECOVERY_MODE
            # bot — it must not clobber recovery_mode.
            await engine._save_bot_state(bot_id, test_db)
            await test_db.commit()

            refreshed = await test_db.get(Bot, bot_id)
            rm = refreshed.strategy_state.get("recovery_mode")
            assert rm is not None, (
                "recovery_mode was erased by a normal strategy-state checkpoint"
            )
            assert rm["active"] is True
            assert rm["trigger_reason"] == "3 consecutive losses"
            # Unrelated runtime state was still persisted normally alongside it.
            assert refreshed.strategy_state["_trend_states"]["trailing_stop"] == 95.0
        finally:
            engine._recovery_states.pop(bot_id, None)
            engine.cleanup_bot_state(bot_id)


class TestRecoveryCountersSurviveRestart:
    """Required test (B): a RECOVERY_MODE bot loaded from DB after a restart
    has its recovery state (paper trades, consecutive wins) restored by
    resume_bots_on_startup, and a post-restore checkpoint does not reset those
    counters back to empty."""

    @pytest.mark.asyncio
    async def test_resume_restores_and_preserves_recovery_counters(self, test_db):
        import asyncio

        prior_trade = {
            "gain_loss_usd": 4.2,
            "win": True,
            "entry_price": 100.0,
            "exit_price": 104.2,
            "timestamp": "2026-06-24T10:05:00",
        }
        bot = Bot(
            name="restart-recovery-bot",
            trading_pair="BTC/USDT",
            strategy="trend_following",
            strategy_params={},
            strategy_state={
                "recovery_mode": {
                    "active": True,
                    "entered_at": "2026-06-24T10:00:00",
                    "trigger_reason": "3 consecutive losses",
                    "paper_position": None,
                    "paper_trades": [prior_trade],
                    "consecutive_paper_wins": 1,
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

        engine = TradingEngine()

        fake_exchange = AsyncMock()
        fake_exchange.connect = AsyncMock(return_value=True)
        fake_exchange.disconnect = AsyncMock(return_value=None)
        fake_exchange.get_ticker = AsyncMock(return_value=Mock(last=50000.0))
        # export_state must return a JSON-serializable dict like the real
        # SimulatedExchangeService — a bare AsyncMock return value would make
        # the checkpoint's _to_jsonable(...) raise, and that exception is
        # swallowed by _run_bot_loop's outer handler, silently skipping the
        # very checkpoint this test exists to exercise.
        fake_exchange.export_state = Mock(return_value={"balances": {}, "order_counter": 0})

        calls = {"n": 0}

        async def fake_execute_strategy(_bot, _price, _session):
            calls["n"] += 1
            # Stop after one full tick: enough to exercise restore + checkpoint.
            engine._stop_flags[bot_id] = True
            return _signal("hold")

        cont = RiskAssessment(action=RiskAction.CONTINUE, reason="ok", details={})

        async def _noop(*a, **k):
            return None

        class _Ctx:
            def __init__(self, session):
                self._session = session

            async def __aenter__(self):
                return self._session

            async def __aexit__(self, *args):
                return False

        try:
            with (
                patch(
                    "app.services.trading_engine.async_session_maker",
                    return_value=_Ctx(test_db),
                ),
                patch.object(
                    TradingEngine, "_make_simulated_exchange",
                    return_value=fake_exchange,
                ),
                patch.object(TradingEngine, "_recover_bot_orders", new=AsyncMock()),
                patch("app.services.trading_engine.BotLoggingService"),
                patch("app.services.trading_engine.ensure_bot_log_directory"),
                patch.object(
                    RiskManagementService, "full_risk_check",
                    AsyncMock(return_value=cont),
                ),
                patch.object(TradingEngine, "_check_positions_stop_loss", new=_noop),
                patch.object(TradingEngine, "_take_pnl_snapshot", new=_noop),
                patch.object(TradingEngine, "_resolve_pending_orders", new=_noop),
                patch.object(TradingEngine, "_reconcile_live_account", new=_noop),
                patch("app.services.trading_engine.asyncio.sleep", new=AsyncMock()),
            ):
                # Instance attribute (not patch.object on the class): a plain
                # function stored on the instance is called as-is, with no
                # descriptor binding self as an extra argument.
                engine._execute_strategy = fake_execute_strategy

                resumed = await engine.resume_bots_on_startup()
                assert resumed == 1, "RECOVERY_MODE bot must be resumed on startup"

                task = engine._running_bots[bot_id]
                await asyncio.wait_for(task, timeout=5.0)

            # Restore populated the in-memory fast path from the persisted JSON.
            assert bot_id in engine._recovery_states
            restored = engine._recovery_states[bot_id]
            assert restored["consecutive_paper_wins"] == 1
            assert len(restored["paper_trades"]) == 1

            # The checkpoint that fires on this first tick (_save_bot_state) must
            # not reset the counters it just restored back to empty/zero.
            refreshed = await test_db.get(Bot, bot_id)
            rm = refreshed.strategy_state.get("recovery_mode")
            assert rm is not None, (
                "post-restore checkpoint erased recovery_mode from the DB"
            )
            assert rm["consecutive_paper_wins"] == 1, (
                "recovery counters were reset instead of continuing from the "
                "persisted value"
            )
            assert len(rm["paper_trades"]) == 1
        finally:
            task = engine._running_bots.pop(bot_id, None)
            if task is not None:
                task.cancel()
            engine._exchange_services.pop(bot_id, None)
            engine._stop_flags.pop(bot_id, None)
            engine._bot_loggers.pop(bot_id, None)
            engine._recovery_states.pop(bot_id, None)
            engine.cleanup_bot_state(bot_id)


class TestNoStrategyStateAssignmentErasesRecoveryMode:
    """Required test (C): no assignment to strategy_state can erase
    recovery_mode — including the edge case where a checkpoint runs before
    this process has restored self._recovery_states into memory (e.g. a save
    sandwiched between resume_bots_on_startup and the loop's first tick)."""

    @pytest.mark.asyncio
    async def test_save_bot_state_preserves_recovery_mode_absent_from_memory(
        self, test_db
    ):
        persisted_rm = {
            "active": True,
            "entered_at": "2026-06-24T10:00:00",
            "trigger_reason": "3 consecutive losses",
            "paper_position": None,
            "paper_trades": [],
            "consecutive_paper_wins": 0,
        }
        bot = Bot(
            name="edge-case-bot",
            trading_pair="BTC/USDT",
            strategy="dca_accumulator",
            strategy_params={},
            strategy_state={"recovery_mode": persisted_rm},
            budget=1000.0,
            current_balance=1000.0,
            is_dry_run=True,
            status=BotStatus.RECOVERY_MODE,
        )
        test_db.add(bot)
        await test_db.flush()
        bot_id = bot.id

        # Fresh engine: self._recovery_states has NO entry for this bot, exactly
        # as it would be immediately after a restart before the loop's first
        # tick has run its restore step.
        engine = TradingEngine()
        assert bot_id not in engine._recovery_states

        # Also give it unrelated runtime state to persist, to prove the merge
        # doesn't just skip the whole save.
        engine._twap_states = {bot_id: {"slices_remaining": 3}}

        await engine._save_bot_state(bot_id, test_db)
        await test_db.commit()

        refreshed = await test_db.get(Bot, bot_id)
        assert refreshed.strategy_state.get("recovery_mode") == persisted_rm, (
            "a strategy-state save with no in-memory recovery state must fall "
            "back to preserving whatever was already persisted, not erase it"
        )
        assert refreshed.strategy_state["_twap_states"]["slices_remaining"] == 3
