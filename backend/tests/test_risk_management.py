"""Unit tests for risk management service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from app.services.risk_management import (
    RiskManagementService,
    RiskAction,
    RiskAssessment,
    PositionRisk,
)
from app.models import Bot, BotStatus, Order, OrderStatus


@pytest.fixture
def mock_session():
    """Create mock async database session."""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def risk_service(mock_session):
    """Create risk management service with mock session."""
    return RiskManagementService(mock_session)


def create_mock_bot(
    bot_id=1,
    budget=10000.0,
    current_balance=9000.0,
    stop_loss_percent=None,
    stop_loss_absolute=None,
    drawdown_limit_percent=None,
    drawdown_limit_absolute=None,
    daily_loss_limit=None,
    weekly_loss_limit=None,
    max_strategy_rotations=3,
    running_time_hours=None,
    started_at=None,
    strategy="momentum",
):
    """Helper to create mock bot."""
    bot = Mock(spec=Bot)
    bot.id = bot_id
    bot.budget = budget
    bot.current_balance = current_balance
    bot.stop_loss_percent = stop_loss_percent
    bot.stop_loss_absolute = stop_loss_absolute
    bot.drawdown_limit_percent = drawdown_limit_percent
    bot.drawdown_limit_absolute = drawdown_limit_absolute
    bot.daily_loss_limit = daily_loss_limit
    bot.weekly_loss_limit = weekly_loss_limit
    bot.max_strategy_rotations = max_strategy_rotations
    bot.running_time_hours = running_time_hours
    bot.started_at = started_at
    bot.strategy = strategy
    bot.updated_at = datetime.utcnow()
    return bot


def create_mock_order(
    bot_id=1,
    fees=1.0,
    modeled_total_cost=0.5,
    running_balance_after=9000.0,
    status=OrderStatus.FILLED,
    filled_at=None,
    created_at=None,
):
    """Helper to create mock order."""
    order = Mock(spec=Order)
    order.bot_id = bot_id
    order.fees = fees
    order.modeled_total_cost = modeled_total_cost
    order.running_balance_after = running_balance_after
    order.status = status
    order.filled_at = filled_at or datetime.utcnow()
    order.created_at = created_at or datetime.utcnow()
    return order


# ============================================================================
# Test Class: Stop Loss
# ============================================================================


class TestStopLoss:
    """Tests for stop-loss trigger logic."""

    @pytest.mark.asyncio
    async def test_percentage_stop_loss_triggered(self, risk_service, mock_session):
        """Test stop loss triggered when percentage loss exceeds limit."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=5.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 47000.0  # -6% loss
        position_amount = 0.1

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is True
        assert "Stop loss triggered" in result.reason
        assert result.pnl_percent < -5.0
        assert result.unrealized_pnl < 0

    @pytest.mark.asyncio
    async def test_percentage_stop_loss_not_triggered(self, risk_service, mock_session):
        """Test stop loss not triggered when within limit."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=5.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 48000.0  # -4% loss
        position_amount = 0.1

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is False
        assert "Within risk limits" in result.reason

    @pytest.mark.asyncio
    async def test_percentage_stop_loss_exactly_at_threshold(
        self, risk_service, mock_session
    ):
        """Test stop loss triggered when exactly at threshold."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=5.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 47500.0  # Exactly -5% loss
        position_amount = 0.1

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is True
        assert "Stop loss triggered" in result.reason

    @pytest.mark.asyncio
    async def test_absolute_stop_loss_triggered(self, risk_service, mock_session):
        """Test stop loss triggered when absolute loss exceeds limit."""
        # Arrange
        bot = create_mock_bot(stop_loss_absolute=200.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 48000.0
        position_amount = 0.15  # Loss = 2000 * 0.15 = 300

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is True
        assert "Stop loss triggered" in result.reason
        assert abs(result.unrealized_pnl) > 200.0

    @pytest.mark.asyncio
    async def test_absolute_stop_loss_not_triggered(self, risk_service, mock_session):
        """Test stop loss not triggered when absolute loss within limit."""
        # Arrange
        bot = create_mock_bot(stop_loss_absolute=200.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 49000.0
        position_amount = 0.1  # Loss = 1000 * 0.1 = 100

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is False
        assert "Within risk limits" in result.reason

    @pytest.mark.asyncio
    async def test_stop_loss_short_position(self, risk_service, mock_session):
        """Test stop loss for short positions."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=5.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 53000.0  # Price went up = loss for short
        position_amount = 0.1

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=False,
        )

        # Assert
        assert result.should_close is True
        assert result.unrealized_pnl < 0

    @pytest.mark.asyncio
    async def test_stop_loss_bot_not_found(self, risk_service, mock_session):
        """Test stop loss check when bot doesn't exist."""
        # Arrange
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=999,
            entry_price=50000.0,
            current_price=48000.0,
            position_amount=0.1,
        )

        # Assert
        assert result.should_close is False
        assert "Bot not found" in result.reason
        assert result.unrealized_pnl == 0
        assert result.pnl_percent == 0

    @pytest.mark.asyncio
    async def test_stop_loss_zero_entry_price(self, risk_service, mock_session):
        """Test stop loss doesn't divide by zero with zero entry price."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=5.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=0.0,
            current_price=50000.0,
            position_amount=0.1,
        )

        # Assert
        assert result.pnl_percent == 0
        assert result.should_close is False

    @pytest.mark.asyncio
    async def test_stop_loss_no_limits_set(self, risk_service, mock_session):
        """Test stop loss when no limits are configured."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=None, stop_loss_absolute=None)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=50000.0,
            current_price=40000.0,  # Large loss
            position_amount=0.1,
        )

        # Assert
        assert result.should_close is False
        assert "Within risk limits" in result.reason


# ============================================================================
# Test Class: Drawdown
# ============================================================================


class TestDrawdown:
    """Tests for drawdown limit detection."""

    @pytest.mark.asyncio
    async def test_percentage_drawdown_triggered(self, risk_service, mock_session):
        """Test drawdown triggered when percentage exceeds limit."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0, current_balance=7000.0, drawdown_limit_percent=20.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.PAUSE_BOT
        assert "Drawdown limit reached" in result.reason
        assert result.details["drawdown_percent"] >= 20.0

    @pytest.mark.asyncio
    async def test_percentage_drawdown_not_triggered(self, risk_service, mock_session):
        """Test drawdown not triggered when within limit."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0, current_balance=8500.0, drawdown_limit_percent=20.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Within drawdown limits" in result.reason

    @pytest.mark.asyncio
    async def test_percentage_drawdown_exactly_at_threshold(
        self, risk_service, mock_session
    ):
        """Test drawdown triggered when exactly at threshold."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0, current_balance=8000.0, drawdown_limit_percent=20.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.PAUSE_BOT
        assert result.details["drawdown_percent"] == 20.0

    @pytest.mark.asyncio
    async def test_absolute_drawdown_triggered(self, risk_service, mock_session):
        """Test drawdown triggered when absolute loss exceeds limit."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0, current_balance=7000.0, drawdown_limit_absolute=2500.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.PAUSE_BOT
        assert "Drawdown limit reached" in result.reason
        assert result.details["drawdown"] >= 2500.0

    @pytest.mark.asyncio
    async def test_absolute_drawdown_not_triggered(self, risk_service, mock_session):
        """Test drawdown not triggered when absolute loss within limit."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0, current_balance=8000.0, drawdown_limit_absolute=2500.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Within drawdown limits" in result.reason

    @pytest.mark.asyncio
    async def test_drawdown_bot_not_found(self, risk_service, mock_session):
        """Test drawdown check when bot doesn't exist."""
        # Arrange
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=999)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Bot not found" in result.reason

    @pytest.mark.asyncio
    async def test_drawdown_zero_budget(self, risk_service, mock_session):
        """Test drawdown calculation with zero budget."""
        # Arrange
        bot = create_mock_bot(
            budget=0.0, current_balance=0.0, drawdown_limit_percent=20.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.details["drawdown_percent"] == 0
        assert result.action == RiskAction.CONTINUE

    @pytest.mark.asyncio
    async def test_drawdown_no_limits_set(self, risk_service, mock_session):
        """Test drawdown when no limits are configured."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0,
            current_balance=5000.0,
            drawdown_limit_percent=None,
            drawdown_limit_absolute=None,
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Within drawdown limits" in result.reason


# ============================================================================
# Test Class: Kill Switch (Daily/Weekly Limits)
# ============================================================================


class TestKillSwitch:
    """Tests for kill switch / daily and weekly loss limits."""

    @pytest.mark.asyncio
    async def test_daily_loss_limit_triggered(self, risk_service, mock_session):
        """Test daily loss limit triggered."""
        # Arrange
        bot = create_mock_bot(daily_loss_limit=100.0)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Mock orders with losses totaling > 100
        orders = [
            create_mock_order(fees=50.0, modeled_total_cost=30.0),
            create_mock_order(fees=30.0, modeled_total_cost=20.0),
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        result = await risk_service.check_daily_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.PAUSE_BOT
        assert "Daily loss limit reached" in result.reason
        assert result.details["daily_loss"] >= 100.0

    @pytest.mark.asyncio
    async def test_daily_loss_limit_not_triggered(self, risk_service, mock_session):
        """Test daily loss limit not triggered."""
        # Arrange
        bot = create_mock_bot(daily_loss_limit=100.0)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        orders = [create_mock_order(fees=20.0, modeled_total_cost=10.0)]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        result = await risk_service.check_daily_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Within daily loss limit" in result.reason

    @pytest.mark.asyncio
    async def test_daily_loss_limit_exactly_at_threshold(
        self, risk_service, mock_session
    ):
        """Test daily loss limit triggered when exactly at threshold."""
        # Arrange
        bot = create_mock_bot(daily_loss_limit=100.0)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        orders = [create_mock_order(fees=60.0, modeled_total_cost=40.0)]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        result = await risk_service.check_daily_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.PAUSE_BOT
        assert result.details["daily_loss"] >= 100.0

    @pytest.mark.asyncio
    async def test_daily_loss_limit_not_set(self, risk_service, mock_session):
        """Test daily loss limit when not configured."""
        # Arrange
        bot = create_mock_bot(daily_loss_limit=None)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_daily_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "No daily loss limit set" in result.reason

    @pytest.mark.asyncio
    async def test_daily_loss_bot_not_found(self, risk_service, mock_session):
        """Test daily loss check when bot doesn't exist."""
        # Arrange
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_daily_loss_limit(bot_id=999)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "No daily loss limit set" in result.reason

    @pytest.mark.asyncio
    async def test_weekly_loss_limit_triggered(self, risk_service, mock_session):
        """Test weekly loss limit triggered."""
        # Arrange
        bot = create_mock_bot(weekly_loss_limit=500.0)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        orders = [
            create_mock_order(fees=200.0, modeled_total_cost=100.0),
            create_mock_order(fees=150.0, modeled_total_cost=80.0),
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        result = await risk_service.check_weekly_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.PAUSE_BOT
        assert "Weekly loss limit reached" in result.reason
        assert result.details["weekly_loss"] >= 500.0

    @pytest.mark.asyncio
    async def test_weekly_loss_limit_not_triggered(self, risk_service, mock_session):
        """Test weekly loss limit not triggered."""
        # Arrange
        bot = create_mock_bot(weekly_loss_limit=500.0)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        orders = [create_mock_order(fees=100.0, modeled_total_cost=50.0)]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        result = await risk_service.check_weekly_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Within weekly loss limit" in result.reason

    @pytest.mark.asyncio
    async def test_weekly_loss_limit_not_set(self, risk_service, mock_session):
        """Test weekly loss limit when not configured."""
        # Arrange
        bot = create_mock_bot(weekly_loss_limit=None)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_weekly_loss_limit(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "No weekly loss limit set" in result.reason


# ============================================================================
# Test Class: Consecutive Losses
# ============================================================================


class TestConsecutiveLosses:
    """Tests for consecutive loss detection."""

    @pytest.mark.asyncio
    async def test_consecutive_losses_triggers_rotation(
        self, risk_service, mock_session
    ):
        """Test consecutive losses trigger strategy rotation."""
        # Arrange
        bot = create_mock_bot(max_strategy_rotations=3)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Mock 3 consecutive losing orders (balance decreasing from 10000 -> 9700)
        # Need 4 orders to get 3 consecutive losses (comparisons between adjacent pairs)
        orders = [
            create_mock_order(running_balance_after=9700.0),  # Most recent (lowest)
            create_mock_order(running_balance_after=9800.0),
            create_mock_order(running_balance_after=9900.0),
            create_mock_order(running_balance_after=10000.0),  # Oldest (highest)
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        # Mock rotation count = 0
        mock_rotation_result = AsyncMock()
        mock_rotation_result.scalar = Mock(return_value=0)

        mock_session.execute.side_effect = [
            mock_bot_result,
            mock_orders_result,
            mock_rotation_result,
        ]

        # Act
        count, result = await risk_service.check_consecutive_losses(
            bot_id=1, threshold=3
        )

        # Assert
        assert count == 3
        assert result.action == RiskAction.ROTATE_STRATEGY
        assert "consecutive losses" in result.reason

    @pytest.mark.asyncio
    async def test_consecutive_losses_below_threshold(
        self, risk_service, mock_session
    ):
        """Test consecutive losses below threshold continues trading."""
        # Arrange
        bot = create_mock_bot()
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Mock 2 consecutive losses (below threshold of 3)
        # Need 3 orders to get 2 consecutive losses
        orders = [
            create_mock_order(running_balance_after=9800.0),  # Most recent
            create_mock_order(running_balance_after=9900.0),
            create_mock_order(running_balance_after=10000.0),  # Oldest
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        count, result = await risk_service.check_consecutive_losses(
            bot_id=1, threshold=3
        )

        # Assert
        assert count == 2
        assert result.action == RiskAction.CONTINUE
        assert "Within consecutive loss threshold" in result.reason

    @pytest.mark.asyncio
    async def test_consecutive_losses_max_rotations_reached(
        self, risk_service, mock_session
    ):
        """Test consecutive losses pause bot when max rotations reached."""
        # Arrange
        bot = create_mock_bot(max_strategy_rotations=3)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Mock 3 consecutive losses (4 orders for 3 comparisons)
        orders = [
            create_mock_order(running_balance_after=9700.0),
            create_mock_order(running_balance_after=9800.0),
            create_mock_order(running_balance_after=9900.0),
            create_mock_order(running_balance_after=10000.0),
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        # Mock rotation count = 3 (at max)
        mock_rotation_result = AsyncMock()
        mock_rotation_result.scalar = Mock(return_value=3)

        mock_session.execute.side_effect = [
            mock_bot_result,
            mock_orders_result,
            mock_rotation_result,
        ]

        # Act
        count, result = await risk_service.check_consecutive_losses(
            bot_id=1, threshold=3
        )

        # Assert
        assert count == 3
        assert result.action == RiskAction.PAUSE_BOT
        assert "Max strategy rotations reached" in result.reason

    @pytest.mark.asyncio
    async def test_consecutive_losses_bot_not_found(self, risk_service, mock_session):
        """Test consecutive losses check when bot doesn't exist."""
        # Arrange
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_result

        # Act
        count, result = await risk_service.check_consecutive_losses(bot_id=999)

        # Assert
        assert count == 0
        assert result.action == RiskAction.CONTINUE
        assert "Bot not found" in result.reason

    @pytest.mark.asyncio
    async def test_consecutive_losses_win_resets_counter(
        self, risk_service, mock_session
    ):
        """Test winning trade resets consecutive loss counter."""
        # Arrange
        bot = create_mock_bot()
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Orders DESC (most recent first):
        # Most recent: 10100 (win compared to 10000)
        # Then: 10000
        # Then: 9900 (loss compared to 10000)
        # Oldest: 10000
        # Should count 0 losses because most recent trade was a win
        orders = [
            create_mock_order(running_balance_after=10100.0),  # Most recent - WIN
            create_mock_order(running_balance_after=10000.0),
            create_mock_order(running_balance_after=9900.0),
            create_mock_order(running_balance_after=10000.0),
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        count, result = await risk_service.check_consecutive_losses(bot_id=1)

        # Assert
        assert count == 0  # Win breaks the streak immediately
        assert result.action == RiskAction.CONTINUE


# ============================================================================
# Test Class: Strategy Rotation
# ============================================================================


class TestStrategyRotation:
    """Tests for strategy rotation logic."""

    @pytest.mark.asyncio
    async def test_rotate_strategy_success(self, risk_service, mock_session):
        """Test successful strategy rotation."""
        # Arrange
        bot = create_mock_bot(strategy="momentum")
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        success, message = await risk_service.rotate_strategy(
            bot_id=1, new_strategy="mean_reversion", reason="Test rotation"
        )

        # Assert
        assert success is True
        assert "momentum" in message
        assert "mean_reversion" in message
        assert bot.strategy == "mean_reversion"
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotate_strategy_bot_not_found(self, risk_service, mock_session):
        """Test strategy rotation when bot doesn't exist."""
        # Arrange
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_result

        # Act
        success, message = await risk_service.rotate_strategy(
            bot_id=999, new_strategy="mean_reversion"
        )

        # Assert
        assert success is False
        assert "Bot not found" in message

    @pytest.mark.asyncio
    async def test_rotate_strategy_blocked_for_auto_mode(
        self, risk_service, mock_session
    ):
        """Test strategy rotation blocked for auto_mode bots."""
        # Arrange
        bot = create_mock_bot(strategy="auto_mode")
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        success, message = await risk_service.rotate_strategy(
            bot_id=1, new_strategy="momentum", reason="Consecutive losses"
        )

        # Assert
        assert success is False
        assert "Cannot rotate auto_mode" in message
        assert bot.strategy == "auto_mode"  # Strategy unchanged

    @pytest.mark.asyncio
    async def test_rotate_strategy_records_rotation(self, risk_service, mock_session):
        """Test strategy rotation records the rotation in database."""
        # Arrange
        bot = create_mock_bot(strategy="momentum")
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        await risk_service.rotate_strategy(
            bot_id=1, new_strategy="mean_reversion", reason="Test"
        )

        # Assert
        mock_session.add.assert_called_once()
        rotation = mock_session.add.call_args[0][0]
        assert rotation.bot_id == 1
        assert rotation.from_strategy == "momentum"
        assert rotation.to_strategy == "mean_reversion"
        assert rotation.reason == "Test"


# ============================================================================
# Test Class: Running Time Limits
# ============================================================================


class TestRunningTimeLimits:
    """Tests for running time limit checks."""

    @pytest.mark.asyncio
    async def test_running_time_limit_exceeded(self, risk_service, mock_session):
        """Test running time limit exceeded."""
        # Arrange
        started_at = datetime.utcnow() - timedelta(hours=25)
        bot = create_mock_bot(running_time_hours=24.0, started_at=started_at)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_running_time(bot_id=1)

        # Assert
        assert result.action == RiskAction.STOP_BOT
        assert "Running time limit reached" in result.reason
        assert result.details["running_hours"] >= 24.0

    @pytest.mark.asyncio
    async def test_running_time_limit_not_exceeded(self, risk_service, mock_session):
        """Test running time limit not exceeded."""
        # Arrange
        started_at = datetime.utcnow() - timedelta(hours=10)
        bot = create_mock_bot(running_time_hours=24.0, started_at=started_at)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_running_time(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Within running time limit" in result.reason
        assert result.details["remaining_hours"] > 0

    @pytest.mark.asyncio
    async def test_running_time_limit_not_set(self, risk_service, mock_session):
        """Test running time when no limit is set (run forever)."""
        # Arrange
        bot = create_mock_bot(running_time_hours=None, started_at=datetime.utcnow())
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_running_time(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "No running time limit" in result.reason

    @pytest.mark.asyncio
    async def test_running_time_bot_not_started(self, risk_service, mock_session):
        """Test running time when bot hasn't started yet."""
        # Arrange
        bot = create_mock_bot(running_time_hours=24.0, started_at=None)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_running_time(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "Bot not started yet" in result.reason

    @pytest.mark.asyncio
    async def test_running_time_exactly_at_limit(self, risk_service, mock_session):
        """Test running time exactly at limit."""
        # Arrange
        started_at = datetime.utcnow() - timedelta(hours=24)
        bot = create_mock_bot(running_time_hours=24.0, started_at=started_at)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_running_time(bot_id=1)

        # Assert
        assert result.action == RiskAction.STOP_BOT
        assert result.details["running_hours"] >= 24.0


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_full_risk_check_stops_on_first_failure(
        self, risk_service, mock_session
    ):
        """Test full risk check stops on first critical failure."""
        # Arrange
        started_at = datetime.utcnow() - timedelta(hours=25)
        bot = create_mock_bot(running_time_hours=24.0, started_at=started_at)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.full_risk_check(bot_id=1)

        # Assert
        assert result.action == RiskAction.STOP_BOT
        assert "Running time limit" in result.reason

    @pytest.mark.asyncio
    async def test_full_risk_check_all_pass(self, risk_service, mock_session):
        """Test full risk check when all checks pass."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0,
            current_balance=9500.0,
            drawdown_limit_percent=10.0,
            running_time_hours=None,
            daily_loss_limit=None,
            weekly_loss_limit=None,
        )

        empty_orders = []

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = bot

        mock_scalars = Mock()
        mock_scalars.all.return_value = empty_orders
        mock_result.scalars.return_value = mock_scalars

        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.full_risk_check(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert "All risk checks passed" in result.reason

    @pytest.mark.asyncio
    async def test_stop_loss_with_profit_position(self, risk_service, mock_session):
        """Test stop loss doesn't trigger on profitable positions."""
        # Arrange
        bot = create_mock_bot(stop_loss_percent=5.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        entry_price = 50000.0
        current_price = 55000.0  # +10% profit
        position_amount = 0.1

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is False
        assert result.unrealized_pnl > 0
        assert result.pnl_percent > 0

    @pytest.mark.asyncio
    async def test_consecutive_losses_with_no_orders(
        self, risk_service, mock_session
    ):
        """Test consecutive losses with no order history."""
        # Arrange
        bot = create_mock_bot()
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        empty_orders = []
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=empty_orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        count, result = await risk_service.check_consecutive_losses(bot_id=1)

        # Assert
        assert count == 0
        assert result.action == RiskAction.CONTINUE

    @pytest.mark.asyncio
    async def test_drawdown_with_increased_balance(self, risk_service, mock_session):
        """Test drawdown when balance increased (profit)."""
        # Arrange
        bot = create_mock_bot(
            budget=10000.0, current_balance=11000.0, drawdown_limit_percent=20.0
        )
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Act
        result = await risk_service.check_drawdown(bot_id=1)

        # Assert
        assert result.action == RiskAction.CONTINUE
        assert result.details["drawdown"] < 0  # Negative drawdown = profit

    @pytest.mark.asyncio
    async def test_loss_calculation_includes_all_costs(
        self, risk_service, mock_session
    ):
        """Test loss calculation includes fees and modeled costs."""
        # Arrange
        bot = create_mock_bot(daily_loss_limit=100.0)
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Order with fees and modeled costs
        orders = [create_mock_order(fees=30.0, modeled_total_cost=20.0)]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        result = await risk_service.check_daily_loss_limit(bot_id=1)

        # Assert
        assert result.details["daily_loss"] == 50.0  # 30 + 20

    @pytest.mark.asyncio
    async def test_multiple_stop_loss_conditions_checked(
        self, risk_service, mock_session
    ):
        """Test both percentage and absolute stop loss are checked."""
        # Arrange - Set both limits
        bot = create_mock_bot(stop_loss_percent=5.0, stop_loss_absolute=100.0)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result

        # Trigger percentage but not absolute
        entry_price = 50000.0
        current_price = 47000.0  # -6% (triggers percentage)
        position_amount = 0.001  # Small position, absolute loss < 100

        # Act
        result = await risk_service.check_stop_loss(
            bot_id=1,
            entry_price=entry_price,
            current_price=current_price,
            position_amount=position_amount,
            is_long=True,
        )

        # Assert
        assert result.should_close is True
        assert "percent" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_consecutive_losses_ignores_orders_without_balance(
        self, risk_service, mock_session
    ):
        """Test consecutive losses ignores orders without running_balance_after."""
        # Arrange
        bot = create_mock_bot()
        mock_bot_result = AsyncMock()
        mock_bot_result.scalar_one_or_none = Mock(return_value=bot)

        # Orders without balance tracking
        orders = [
            create_mock_order(running_balance_after=None),
            create_mock_order(running_balance_after=None),
        ]
        mock_orders_result = AsyncMock()
        mock_orders_result.scalars = Mock(return_value=Mock(all=Mock(return_value=orders)))

        mock_session.execute.side_effect = [mock_bot_result, mock_orders_result]

        # Act
        count, result = await risk_service.check_consecutive_losses(bot_id=1)

        # Assert
        assert count == 0
        assert result.action == RiskAction.CONTINUE
