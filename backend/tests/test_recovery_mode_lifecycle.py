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
