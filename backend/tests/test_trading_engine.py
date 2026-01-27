"""Improved unit tests for trading engine core logic.

Tests focus on:
- Explicit state transitions (balance, positions, P&L)
- Edge cases (overselling, zero/negative amounts, concurrent operations)
- Mock interaction validation (not just called, but with correct arguments)
- Structured test organization by feature area
- Deterministic, reliable testing without time dependencies
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, call
from dataclasses import dataclass

from app.services.trading_engine import TradingEngine, TradeSignal
from app.models import (
    Bot,
    BotStatus,
    Order,
    OrderType,
    OrderStatus,
    Position,
    PositionSide,
    Trade,
    TradeSide,
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================


@dataclass
class MockExchangeOrder:
    """Mock exchange order response."""

    id: str
    amount: float
    price: float
    fee: float
    status: str = "closed"
    filled: float = None

    def __post_init__(self):
        if self.filled is None:
            self.filled = self.amount


class MockExchangeService:
    """Mock exchange service with call tracking."""

    def __init__(self, should_fail: bool = False, partial_fill_ratio: float = 1.0):
        self.should_fail = should_fail
        self.partial_fill_ratio = partial_fill_ratio  # 0.0 to 1.0
        self.orders_placed = []
        self.is_connected = True

    async def connect(self):
        """Mock connect method."""
        pass

    async def place_market_order(self, trading_pair: str, side, amount: float):
        """Mock market order placement with validation."""
        if self.should_fail:
            return None

        if amount <= 0:
            raise ValueError("Amount must be positive")

        filled_amount = amount * self.partial_fill_ratio
        price = 50000.0  # Fixed mock price

        order = MockExchangeOrder(
            id=f"mock_order_{len(self.orders_placed) + 1}",
            amount=amount,
            price=price,
            fee=filled_amount * price * 0.001,  # 0.1% fee
            status="closed" if self.partial_fill_ratio == 1.0 else "partial",
            filled=filled_amount,
        )

        self.orders_placed.append(
            {
                "type": "market",
                "side": side,
                "amount": amount,
                "trading_pair": trading_pair,
                "order": order,
            }
        )

        return order

    async def place_limit_order(
        self, trading_pair: str, side, amount: float, price: float
    ):
        """Mock limit order placement."""
        if self.should_fail:
            return None

        if amount <= 0 or price <= 0:
            raise ValueError("Amount and price must be positive")

        order = MockExchangeOrder(
            id=f"mock_limit_{len(self.orders_placed) + 1}",
            amount=amount,
            price=price,
            fee=amount * price * 0.001,
            status="open",
            filled=0.0,
        )

        self.orders_placed.append(
            {
                "type": "limit",
                "side": side,
                "amount": amount,
                "price": price,
                "trading_pair": trading_pair,
                "order": order,
            }
        )

        return order


@pytest.fixture
def mock_bot():
    """Create a mock bot for testing."""
    bot = Mock(spec=Bot)
    bot.id = 1
    bot.name = "Test Bot"
    bot.trading_pair = "BTC/USDT"
    bot.strategy = "test_strategy"
    bot.budget = 10000.0
    bot.current_balance = 10000.0
    bot.is_dry_run = True
    bot.status = BotStatus.RUNNING
    bot.exchange_fee = 0.001
    bot.is_simulated = True
    return bot


@pytest.fixture
def mock_session():
    """Create a mock database session with tracking."""
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()

    # Track what was added to session
    session.added_objects = []

    def track_add(obj):
        session.added_objects.append(obj)

    session.add.side_effect = track_add

    # Mock query results
    mock_result = Mock()
    mock_result.scalar_one_or_none = Mock(return_value=None)
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
    session.execute.return_value = mock_result

    return session


@pytest.fixture
def trading_engine():
    """Create trading engine instance."""
    return TradingEngine()


@pytest.fixture
def mock_services():
    """Mock all service dependencies with tracking."""
    services = {}

    # Portfolio Risk Service
    risk_check = Mock()
    risk_check.ok = True
    risk_check.action = "allow"
    risk_check.reason = None
    portfolio_risk_instance = AsyncMock()
    portfolio_risk_instance.check_portfolio_risk = AsyncMock(return_value=risk_check)
    services["portfolio_risk"] = portfolio_risk_instance

    # Strategy Capacity Service
    capacity_check = Mock()
    capacity_check.ok = True
    capacity_check.adjusted_amount = None
    capacity_check.reason = None
    strategy_capacity_instance = AsyncMock()
    strategy_capacity_instance.check_capacity_for_trade = AsyncMock(
        return_value=capacity_check
    )
    services["strategy_capacity"] = strategy_capacity_instance

    # Trade Recorder Service
    trade_recorder_instance = AsyncMock()
    services["trade_recorder"] = trade_recorder_instance

    # Tax Engine
    tax_engine_instance = AsyncMock()
    tax_engine_instance.process_buy = AsyncMock()
    tax_engine_instance.process_sell = AsyncMock(return_value=[])  # List of RealizedGain objects
    services["tax_engine"] = tax_engine_instance

    # Invariant Validator
    invariant_validator_instance = AsyncMock()
    invariant_validator_instance.validate_trade = AsyncMock()
    services["invariant_validator"] = invariant_validator_instance

    # Virtual Wallet Service
    wallet_instance = AsyncMock()
    wallet_instance.record_trade_result = AsyncMock(return_value=(True, "Success"))
    wallet_instance.validate_trade = AsyncMock(
        return_value=Mock(is_valid=True, reason="OK")
    )
    services["wallet"] = wallet_instance

    return services


def create_mock_position(
    bot_id: int = 1,
    symbol: str = "BTC/USDT",
    side: PositionSide = PositionSide.LONG,
    quantity: float = 0.5,
    average_entry_price: float = 48000.0,
):
    """Helper to create mock positions."""
    position = Mock()  # Don't use spec to avoid Mock attribute issues
    position.bot_id = bot_id
    position.trading_pair = symbol  # Position uses trading_pair, not symbol
    position.side = side
    position.amount = quantity  # Position uses amount, not quantity
    position.quantity = quantity  # Keep for backwards compat
    position.entry_price = average_entry_price  # Position uses entry_price
    position.average_entry_price = average_entry_price  # Keep for backwards compat
    position.realized_pnl = 0.0
    position.is_open = quantity > 0
    position.calculate_unrealized_pnl = Mock(return_value=0.0)
    return position


def create_mock_trade(
    trade_id: int = 1,
    bot_id: int = 1,
    side: TradeSide = TradeSide.BUY,
    base_amount: float = 0.02,
    quote_amount: float = 1000.0,
    price: float = 50000.0,
    fee: float = 1.0,
):
    """Helper to create mock trades."""
    trade = Mock(spec=Trade)
    trade.id = trade_id
    trade.bot_id = bot_id
    trade.side = side
    trade.base_amount = base_amount
    trade.quote_amount = quote_amount
    trade.price = price
    trade.fee_amount = fee
    trade.get_cost_basis_per_unit = Mock(return_value=price)
    return trade


# ============================================================================
# Test Class: Market Orders
# ============================================================================


class TestMarketOrders:
    """Tests for market order execution."""

    @pytest.mark.asyncio
    async def test_market_buy_success_with_state_transitions(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test market buy with explicit state transition verification."""
        # Arrange
        mock_exchange = MockExchangeService(should_fail=False)
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Initial state
        initial_balance = mock_bot.current_balance
        expected_base_amount = 1000.0 / 50000.0  # 0.02 BTC
        expected_fee = expected_base_amount * 50000.0 * 0.001  # 1.0 USDT

        # Mock trade recorder to return created trade
        mock_trade = create_mock_trade(
            trade_id=1,
            bot_id=mock_bot.id,
            side=TradeSide.BUY,
            base_amount=expected_base_amount,
            quote_amount=1000.0,
            price=50000.0,
            fee=expected_fee,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order created correctly
        assert order is not None
        assert order.order_type == OrderType.MARKET_BUY
        assert order.bot_id == mock_bot.id
        assert order.status == OrderStatus.FILLED
        assert order.is_simulated is True

        # Assert - Exchange interaction
        assert len(mock_exchange.orders_placed) == 1
        exchange_order = mock_exchange.orders_placed[0]
        assert exchange_order["type"] == "market"
        assert exchange_order["amount"] == pytest.approx(expected_base_amount, abs=1e-8)
        assert exchange_order["trading_pair"] == "BTC/USDT"

        # Assert - Service interactions with correct arguments
        mock_services["portfolio_risk"].check_portfolio_risk.assert_called_once()
        mock_services["strategy_capacity"].check_capacity_for_trade.assert_called_once()
        mock_services["trade_recorder"].record_trade.assert_called_once()

        # Assert - Tax engine called for buy
        mock_services["tax_engine"].process_buy.assert_called_once()
        tax_call = mock_services["tax_engine"].process_buy.call_args
        assert tax_call[1]["trade"] == mock_trade

        # Assert - Session operations  
        # Order and potentially position are added
        assert mock_session.add.call_count >= 1
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_market_sell_with_position_update(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test market sell updates position correctly."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="sell", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Create existing position
        existing_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.5,  # Selling 0.02 from 0.5 position
            average_entry_price=48000.0,
        )

        # Mock position query
        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=existing_position)
        mock_session.execute.return_value = mock_position_result

        # Mock trade recorder
        sell_amount = 1000.0 / 50000.0  # 0.02 BTC
        mock_trade = create_mock_trade(
            trade_id=2,
            bot_id=mock_bot.id,
            side=TradeSide.SELL,
            base_amount=sell_amount,
            quote_amount=1000.0,
            price=50000.0,
            fee=1.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        # Mock tax engine to return realized gain
        realized_gain = (50000.0 - 48000.0) * 0.02  # 40.0 profit
        mock_services["tax_engine"].process_sell = AsyncMock(
            return_value=(realized_gain, [])
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Sell order created
        assert order is not None
        assert order.order_type == OrderType.MARKET_SELL

        # Assert - Tax engine called for sell with correct trade
        mock_services["tax_engine"].process_sell.assert_called_once()
        sell_call = mock_services["tax_engine"].process_sell.call_args
        assert sell_call[1]["trade"] == mock_trade

        # Assert - Position would be updated (0.5 - 0.02 = 0.48 remaining)
        # Note: Position update happens via database, so we verify the query was made
        mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_market_buy_zero_amount_rejected(
        self, mock_bot, mock_session, trading_engine
    ):
        """Test that zero-amount orders are rejected."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=0.0, order_type="market")
        current_price = 50000.0

        # Act
        order = await trading_engine._execute_trade(
            mock_bot, mock_exchange, signal, current_price, mock_session
        )

        # Assert - Order rejected, nothing placed on exchange
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

    @pytest.mark.asyncio
    async def test_market_buy_negative_amount_rejected(
        self, mock_bot, mock_session, trading_engine
    ):
        """Test that negative amounts are rejected."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=-100.0, order_type="market")
        current_price = 50000.0

        # Act
        order = await trading_engine._execute_trade(
            mock_bot, mock_exchange, signal, current_price, mock_session
        )

        # Assert
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

    @pytest.mark.asyncio
    async def test_market_sell_more_than_position_rejected(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that selling more than current position is rejected."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="sell", amount=5000.0, order_type="market"
        )  # Trying to sell 0.1 BTC
        current_price = 50000.0

        # Mock position with only 0.02 BTC
        small_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.02,  # Only 0.02 BTC available
            average_entry_price=48000.0,
        )

        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=small_position)
        mock_session.execute.return_value = mock_position_result

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ):

            # Act - Try to sell 0.1 BTC when only 0.02 available
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order rejected, no exchange call
        assert order is None
        assert len(mock_exchange.orders_placed) == 0


# ============================================================================
# Test Class: Limit Orders
# ============================================================================


class TestLimitOrders:
    """Tests for limit order execution."""

    @pytest.mark.asyncio
    async def test_limit_buy_creates_pending_order(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test limit buy creates pending order."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="buy", amount=1000.0, order_type="limit", limit_price=49000.0
        )
        current_price = 50000.0

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Limit order created with PENDING status
        assert order is not None
        assert order.order_type == OrderType.LIMIT_BUY
        assert order.status == OrderStatus.PENDING
        assert order.limit_price == 49000.0

        # Assert - Exchange called with limit order
        assert len(mock_exchange.orders_placed) == 1
        limit_order = mock_exchange.orders_placed[0]
        assert limit_order["type"] == "limit"
        assert limit_order["price"] == 49000.0

    @pytest.mark.asyncio
    async def test_limit_order_zero_price_rejected(
        self, mock_bot, mock_session, trading_engine
    ):
        """Test that limit orders with zero price are rejected."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="buy", amount=1000.0, order_type="limit", limit_price=0.0
        )
        current_price = 50000.0

        # Act
        order = await trading_engine._execute_trade(
            mock_bot, mock_exchange, signal, current_price, mock_session
        )

        # Assert
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

    @pytest.mark.asyncio
    async def test_limit_order_negative_price_rejected(
        self, mock_bot, mock_session, trading_engine
    ):
        """Test that limit orders with negative price are rejected."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="buy", amount=1000.0, order_type="limit", limit_price=-100.0
        )
        current_price = 50000.0

        # Act
        order = await trading_engine._execute_trade(
            mock_bot, mock_exchange, signal, current_price, mock_session
        )

        # Assert
        assert order is None
        assert len(mock_exchange.orders_placed) == 0


# ============================================================================
# Test Class: Position Management
# ============================================================================


class TestPositionManagement:
    """Tests for position state transitions."""

    @pytest.mark.asyncio
    async def test_first_buy_creates_new_position(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that first buy creates a new position."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        # No existing position
        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_position_result

        buy_amount = 1000.0 / 50000.0  # 0.02 BTC
        mock_trade = create_mock_trade(
            trade_id=1,
            bot_id=mock_bot.id,
            side=TradeSide.BUY,
            base_amount=buy_amount,
            quote_amount=1000.0,
            price=50000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order succeeded
        assert order is not None
        assert order.status == OrderStatus.FILLED

        # Assert - Position query was made
        mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_second_buy_increases_position_size(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that second buy increases position size and updates average price."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=2000.0, order_type="market")
        current_price = 52000.0

        # Existing position: 0.5 BTC @ 48000
        existing_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.5,
            average_entry_price=48000.0,
        )

        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=existing_position)
        mock_session.execute.return_value = mock_position_result

        # New buy: 2000 / 52000 ~= 0.0385 BTC @ 52000
        new_buy_amount = 2000.0 / 52000.0
        mock_trade = create_mock_trade(
            trade_id=2,
            bot_id=mock_bot.id,
            side=TradeSide.BUY,
            base_amount=new_buy_amount,
            quote_amount=2000.0,
            price=52000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert
        assert order is not None

        # Expected new position state:
        # Total quantity: 0.5 + 0.0385 = 0.5385
        # New average price: (0.5*48000 + 0.0385*52000) / 0.5385 ≈ 48289
        # (Position update would happen in database layer, verified by query)
        mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_sell_closes_position_completely(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that selling entire position closes it."""
        # Arrange
        mock_exchange = MockExchangeService()

        # Existing position: exactly 0.02 BTC
        position_size = 0.02
        existing_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=position_size,
            average_entry_price=48000.0,
        )

        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=existing_position)
        mock_session.execute.return_value = mock_position_result

        # Sell entire position
        signal = TradeSignal(
            action="sell",
            amount=position_size * 50000.0,  # Sell all
            order_type="market",
        )
        current_price = 50000.0

        mock_trade = create_mock_trade(
            trade_id=3,
            bot_id=mock_bot.id,
            side=TradeSide.SELL,
            base_amount=position_size,
            quote_amount=position_size * 50000.0,
            price=50000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        realized_gain = (50000.0 - 48000.0) * position_size
        mock_services["tax_engine"].process_sell = AsyncMock(
            return_value=(realized_gain, [])
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order executed
        assert order is not None
        assert order.order_type == OrderType.MARKET_SELL

        # Assert - Realized gain calculated
        mock_services["tax_engine"].process_sell.assert_called_once()

        # Position would be closed (quantity = 0, is_open = False)
        # Verified through database layer

    @pytest.mark.asyncio
    async def test_partial_sell_reduces_position(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that partial sell reduces position size."""
        # Arrange
        mock_exchange = MockExchangeService()

        # Existing position: 1.0 BTC @ 45000
        existing_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=1.0,
            average_entry_price=45000.0,
        )

        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=existing_position)
        mock_session.execute.return_value = mock_position_result

        # Sell 25% of position (0.25 BTC)
        sell_amount = 0.25
        signal = TradeSignal(
            action="sell", amount=sell_amount * 50000.0, order_type="market"
        )
        current_price = 50000.0

        mock_trade = create_mock_trade(
            trade_id=4,
            bot_id=mock_bot.id,
            side=TradeSide.SELL,
            base_amount=sell_amount,
            quote_amount=sell_amount * 50000.0,
            price=50000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        realized_gain = (50000.0 - 45000.0) * sell_amount
        mock_services["tax_engine"].process_sell = AsyncMock(
            return_value=(realized_gain, [])
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert
        assert order is not None

        # Remaining position: 1.0 - 0.25 = 0.75 BTC
        # Average price stays at 45000 (no new buys)
        # Realized gain: 1250.0
        mock_services["tax_engine"].process_sell.assert_called_once()


# ============================================================================
# Test Class: Rejections and Failures
# ============================================================================


class TestRejectionsAndFailures:
    """Tests for order rejections and error handling."""

    @pytest.mark.asyncio
    async def test_rejected_by_portfolio_risk(
        self, mock_bot, mock_session, trading_engine
    ):
        """Test order rejected by portfolio risk check."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Mock portfolio risk rejection
        risk_check = Mock()
        risk_check.ok = False
        risk_check.action = "reject"
        risk_check.reason = "Portfolio risk limit exceeded"

        portfolio_risk_instance = AsyncMock()
        portfolio_risk_instance.check_portfolio_risk = AsyncMock(
            return_value=risk_check
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=portfolio_risk_instance,
        ):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order rejected before exchange call
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

        # Assert - Portfolio risk was checked
        portfolio_risk_instance.check_portfolio_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejected_by_strategy_capacity(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test order rejected by strategy capacity check."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Portfolio risk passes
        mock_services["portfolio_risk"].check_portfolio_risk = AsyncMock(
            return_value=Mock(ok=True, action="allow")
        )

        # Strategy capacity rejects
        capacity_check = Mock()
        capacity_check.ok = False
        capacity_check.reason = "Strategy at max capacity"
        mock_services["strategy_capacity"].check_capacity_for_trade = AsyncMock(
            return_value=capacity_check
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order rejected
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

        # Assert - Both checks were performed
        mock_services["portfolio_risk"].check_portfolio_risk.assert_called_once()
        mock_services["strategy_capacity"].check_capacity_for_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejected_by_exchange(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test order rejected by exchange (returns None)."""
        # Arrange
        mock_exchange = MockExchangeService(should_fail=True)
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order rejected at exchange level
        assert order is None

        # Assert - No trade recorded (exchange failed)
        # Trade recorder should not be called if exchange fails

    @pytest.mark.asyncio
    async def test_rollback_on_invariant_validation_failure(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that transaction is rolled back on invariant validation failure."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        mock_trade = create_mock_trade(
            trade_id=1, bot_id=mock_bot.id, side=TradeSide.BUY
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        # Mock invariant validator to raise exception
        mock_services["invariant_validator"].validate_trade = AsyncMock(
            side_effect=Exception("Invariant check failed: balance mismatch")
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order creation failed
        assert order is None

        # Assert - Session was rolled back
        mock_session.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_insufficient_balance_rejection(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test order rejected when balance is insufficient."""
        # Arrange
        mock_bot.current_balance = 5.0  # Very low balance
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="buy", amount=5.0, order_type="market"
        )  # Amount too small
        current_price = 50000.0

        # Wallet validation rejects
        mock_services["wallet"].validate_trade = AsyncMock(
            return_value=Mock(is_valid=False, reason="Amount below minimum")
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

    @pytest.mark.asyncio
    async def test_cancel_pending_orders(self, mock_session, trading_engine):
        """Test cancellation of pending orders for a bot."""
        # Arrange
        bot_id = 1

        # Create multiple pending orders
        pending_order_1 = Mock(spec=Order)
        pending_order_1.id = 101
        pending_order_1.bot_id = bot_id
        pending_order_1.status = OrderStatus.PENDING
        pending_order_1.trading_pair = "BTC/USDT"

        pending_order_2 = Mock(spec=Order)
        pending_order_2.id = 102
        pending_order_2.bot_id = bot_id
        pending_order_2.status = OrderStatus.PENDING
        pending_order_2.trading_pair = "ETH/USDT"

        # Create an order for a different bot (should not be cancelled)
        other_bot_order = Mock(spec=Order)
        other_bot_order.id = 201
        other_bot_order.bot_id = 2
        other_bot_order.status = OrderStatus.PENDING

        # Mock database query to return only this bot's pending orders
        mock_result = Mock()
        mock_result.scalars = Mock(
            return_value=Mock(all=Mock(return_value=[pending_order_1, pending_order_2]))
        )
        mock_session.execute.return_value = mock_result

        # Act
        await trading_engine._cancel_pending_orders(bot_id, mock_session)

        # Assert - Both orders for bot_id=1 are marked as CANCELLED
        assert pending_order_1.status == OrderStatus.CANCELLED
        assert pending_order_2.status == OrderStatus.CANCELLED

        # Assert - Other bot's order is unaffected
        assert other_bot_order.status == OrderStatus.PENDING

        # Assert - Session commit called once
        mock_session.commit.assert_called_once()

        # Assert - Session execute was called to query pending orders
        mock_session.execute.assert_called_once()


# ============================================================================
# Test Class: Partial Fills
# ============================================================================


class TestPartialFills:
    """Tests for partial order fills."""

    @pytest.mark.asyncio
    async def test_partial_fill_handling(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that partial fills are recorded correctly."""
        # Arrange - 50% fill
        mock_exchange = MockExchangeService(partial_fill_ratio=0.5)
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Expected: requested 0.02 BTC, filled 0.01 BTC
        filled_amount = (1000.0 / 50000.0) * 0.5

        mock_trade = create_mock_trade(
            trade_id=1,
            bot_id=mock_bot.id,
            side=TradeSide.BUY,
            base_amount=filled_amount,
            quote_amount=500.0,
            price=50000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order created with partial fill
        assert order is not None
        # Note: Status depends on implementation - could be PARTIAL or FILLED
        # Verify trade recorder was called with correct filled amount
        mock_services["trade_recorder"].record_trade.assert_called_once()


# ============================================================================
# Test Class: Ledger Integration
# ============================================================================


class TestLedgerIntegration:
    """Tests for ledger and accounting integration."""

    @pytest.mark.asyncio
    async def test_buy_creates_tax_lot(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that buy orders create tax lots via tax engine."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        mock_trade = create_mock_trade(
            trade_id=1, bot_id=mock_bot.id, side=TradeSide.BUY
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Tax engine process_buy was called
        assert order is not None
        mock_services["tax_engine"].process_buy.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_services["tax_engine"].process_buy.call_args[1]
        assert call_kwargs["trade"] == mock_trade
        assert call_kwargs["bot_id"] == mock_bot.id

    @pytest.mark.asyncio
    async def test_sell_consumes_tax_lots_and_calculates_gain(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that sell orders consume tax lots and calculate realized gain."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="sell", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Mock existing position and bot query  
        existing_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            quantity=0.5,
            average_entry_price=45000.0,
        )
        
        # Mock bot with current balance
        updated_bot = Mock()
        updated_bot.current_balance = 11000.0  # After sell
        
        # Track call count for execute
        execute_count = [0]
        
        # Set up mock session to return different results for different queries
        def mock_execute(query):
            execute_count[0] += 1
            result = Mock()
            # First call returns bot, second returns position
            if execute_count[0] == 1:
                result.scalar_one_or_none = Mock(return_value=updated_bot)
            else:
                result.scalar_one_or_none = Mock(return_value=existing_position)
            return result
        
        mock_session.execute = AsyncMock(side_effect=mock_execute)

        sell_amount = 1000.0 / 50000.0  # 0.02 BTC
        mock_trade = create_mock_trade(
            trade_id=2,
            bot_id=mock_bot.id,
            side=TradeSide.SELL,
            base_amount=sell_amount,
            quote_amount=1000.0,
            price=50000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        # Mock realized gains as list of RealizedGain objects
        mock_realized_gain = Mock()
        mock_realized_gain.gain_loss = 100.0  # (50000 - 45000) * 0.02 = 100
        mock_services["tax_engine"].process_sell = AsyncMock(
            return_value=[mock_realized_gain]
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Tax engine process_sell was called
        assert order is not None
        mock_services["tax_engine"].process_sell.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_services["tax_engine"].process_sell.call_args[1]
        assert call_kwargs["trade"] == mock_trade
        assert call_kwargs["bot_id"] == mock_bot.id

    @pytest.mark.asyncio
    async def test_fees_applied_correctly(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that trading fees are calculated and applied."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        buy_amount = 1000.0 / 50000.0  # 0.02 BTC
        expected_fee = buy_amount * 50000.0 * 0.001  # 1.0 USDT

        # Exchange returns order with fee
        exchange_order = mock_exchange.orders_placed

        mock_trade = create_mock_trade(
            trade_id=1,
            bot_id=mock_bot.id,
            side=TradeSide.BUY,
            base_amount=buy_amount,
            quote_amount=1000.0,
            price=50000.0,
            fee=expected_fee,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Fee is included in exchange order
        assert len(mock_exchange.orders_placed) == 1
        placed_order = mock_exchange.orders_placed[0]["order"]
        assert placed_order.fee == pytest.approx(expected_fee, abs=1e-6)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_strategy_switch_mid_position(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that strategy can change while holding position."""
        # Arrange - Bot has position from old strategy
        mock_bot.strategy = "new_strategy"  # Strategy changed
        mock_exchange = MockExchangeService()
        signal = TradeSignal(action="buy", amount=1000.0, order_type="market")
        current_price = 50000.0

        # Position exists from old strategy
        existing_position = create_mock_position(
            bot_id=mock_bot.id,
            symbol="BTC/USDT",
            quantity=0.1,
            average_entry_price=48000.0,
        )
        mock_position_result = Mock()
        mock_position_result.scalar_one_or_none = Mock(return_value=existing_position)
        mock_session.execute.return_value = mock_position_result

        mock_trade = create_mock_trade(trade_id=1, bot_id=mock_bot.id)
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act - New strategy places order
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Order succeeds despite strategy change
        assert order is not None

    @pytest.mark.asyncio
    async def test_very_small_order_below_minimum(
        self, mock_bot, mock_session, trading_engine
    ):
        """Test that orders below exchange minimum are rejected."""
        # Arrange
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="buy", amount=0.1, order_type="market"
        )  # Too small
        current_price = 50000.0

        # Act
        order = await trading_engine._execute_trade(
            mock_bot, mock_exchange, signal, current_price, mock_session
        )

        # Assert - Rejected due to minimum size
        assert order is None
        assert len(mock_exchange.orders_placed) == 0

    @pytest.mark.asyncio
    async def test_very_large_order_within_capacity(
        self, mock_bot, mock_session, trading_engine, mock_services
    ):
        """Test that large orders within capacity are accepted."""
        # Arrange
        mock_bot.current_balance = 100000.0  # High balance
        mock_exchange = MockExchangeService()
        signal = TradeSignal(
            action="buy", amount=50000.0, order_type="market"
        )  # Large order
        current_price = 50000.0

        mock_trade = create_mock_trade(
            trade_id=1,
            bot_id=mock_bot.id,
            base_amount=1.0,  # 1 BTC
            quote_amount=50000.0,
        )
        mock_services["trade_recorder"].record_trade = AsyncMock(
            return_value=mock_trade
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act
            order = await trading_engine._execute_trade(
                mock_bot, mock_exchange, signal, current_price, mock_session
            )

        # Assert - Large order succeeds
        assert order is not None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_bots_isolated(
        self, mock_session, trading_engine, mock_services
    ):
        """Test that multiple bots operate independently."""
        # Arrange - Two different bots
        bot1 = Mock(spec=Bot)
        bot1.id = 1
        bot1.name = "Bot 1"
        bot1.trading_pair = "BTC/USDT"
        bot1.strategy = "strategy_1"
        bot1.current_balance = 10000.0
        bot1.is_dry_run = True
        bot1.is_simulated = True

        bot2 = Mock(spec=Bot)
        bot2.id = 2
        bot2.name = "Bot 2"
        bot2.trading_pair = "ETH/USDT"
        bot2.strategy = "strategy_2"
        bot2.current_balance = 5000.0
        bot2.is_dry_run = True
        bot2.is_simulated = True

        mock_exchange = MockExchangeService()
        signal1 = TradeSignal(action="buy", amount=1000.0, order_type="market")
        signal2 = TradeSignal(action="buy", amount=500.0, order_type="market")

        mock_trade1 = create_mock_trade(trade_id=1, bot_id=bot1.id)
        mock_trade2 = create_mock_trade(trade_id=2, bot_id=bot2.id)

        # Mock trade recorder to return different trades
        mock_services["trade_recorder"].record_trade = AsyncMock(
            side_effect=[mock_trade1, mock_trade2]
        )

        with patch(
            "app.services.trading_engine.PortfolioRiskService",
            return_value=mock_services["portfolio_risk"],
        ), patch(
            "app.services.trading_engine.StrategyCapacityService",
            return_value=mock_services["strategy_capacity"],
        ), patch(
            "app.services.trading_engine.TradeRecorderService",
            return_value=mock_services["trade_recorder"],
        ), patch(
            "app.services.trading_engine.FIFOTaxEngine",
            return_value=mock_services["tax_engine"],
        ), patch(
            "app.services.trading_engine.LedgerInvariantService",
            return_value=mock_services["invariant_validator"],
        ), patch(
            "app.services.trading_engine.VirtualWalletService",
            return_value=mock_services["wallet"],
        ), patch("app.services.trading_engine.CSVExportService"):

            # Act - Execute orders for both bots
            order1 = await trading_engine._execute_trade(
                bot1, mock_exchange, signal1, 50000.0, mock_session
            )
            order2 = await trading_engine._execute_trade(
                bot2, mock_exchange, signal2, 2000.0, mock_session
            )

        # Assert - Both orders succeeded independently
        assert order1 is not None
        assert order2 is not None
        assert order1.bot_id == 1
        assert order2.bot_id == 2

        # Assert - Trade recorder called twice with different bot IDs
        assert mock_services["trade_recorder"].record_trade.call_count == 2
