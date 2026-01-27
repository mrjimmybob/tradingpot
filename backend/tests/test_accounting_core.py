"""Unit tests for accounting core logic.

Tests focus on FIFO tax lot assignment and P&L calculation.
All tests use in-memory objects without database dependencies.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import List

from app.services.accounting import FIFOTaxEngine
from app.models import Trade, TradeSide, TaxLot, RealizedGain


def create_trade(
    trade_id: int,
    owner_id: str,
    side: TradeSide,
    base_asset: str,
    base_amount: float,
    quote_amount: float,
    fee_amount: float = 0.0,
    modeled_cost: float = 0.0,
    executed_at: datetime = None,
) -> Mock:
    """Create a mock trade object."""
    trade = Mock(spec=Trade)
    trade.id = trade_id
    trade.owner_id = owner_id
    trade.side = side
    trade.base_asset = base_asset
    trade.base_amount = base_amount
    trade.quote_amount = quote_amount
    trade.fee_amount = fee_amount
    trade.modeled_cost = modeled_cost
    trade.executed_at = executed_at or datetime(2025, 1, 1, 12, 0, 0)
    
    total_cost = quote_amount + fee_amount + modeled_cost
    trade.get_total_cost = Mock(return_value=total_cost)
    trade.get_cost_basis_per_unit = Mock(
        return_value=total_cost / base_amount if base_amount > 0 else 0.0
    )
    
    return trade


def create_tax_lot(
    lot_id: int,
    owner_id: str,
    asset: str,
    quantity_acquired: float,
    quantity_remaining: float,
    unit_cost: float,
    total_cost: float,
    purchase_trade_id: int,
    purchase_date: datetime,
) -> Mock:
    """Create a mock tax lot that implements consume()."""
    lot = Mock(spec=TaxLot)
    lot.id = lot_id
    lot.owner_id = owner_id
    lot.asset = asset
    lot.quantity_acquired = quantity_acquired
    lot.quantity_remaining = quantity_remaining
    lot.unit_cost = unit_cost
    lot.total_cost = total_cost
    lot.purchase_trade_id = purchase_trade_id
    lot.purchase_date = purchase_date
    lot.is_fully_consumed = False
    lot.consumed_at = None
    
    def consume(quantity: float, consumed_at: datetime) -> float:
        consumed = min(quantity, lot.quantity_remaining)
        lot.quantity_remaining -= consumed
        
        if lot.quantity_remaining <= 1e-8:
            lot.quantity_remaining = 0.0
            lot.is_fully_consumed = True
            lot.consumed_at = consumed_at
        
        return consumed
    
    lot.consume = consume
    
    return lot


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def tax_engine(mock_session):
    """Create a FIFOTaxEngine instance."""
    return FIFOTaxEngine(mock_session)


class TestFIFOTaxLotAssignment:
    """Test FIFO tax lot assignment logic."""
    
    @pytest.mark.asyncio
    async def test_buy_two_lots_sell_one_consumes_oldest(self, tax_engine, mock_session):
        """Buy 2 lots, sell 1 → consumes oldest lot."""
        owner_id = "user1"
        asset = "BTC"
        
        lot1_date = datetime(2025, 1, 1)
        lot2_date = datetime(2025, 1, 15)
        sell_date = datetime(2025, 2, 1)
        
        lot1 = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot1_date)
        lot2 = create_tax_lot(2, owner_id, asset, 1.0, 1.0, 41000.0, 41000.0, 2, lot2_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot1, lot2])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(3, owner_id, TradeSide.SELL, asset, 0.5, 21000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        assert lot1.quantity_remaining == 0.5
        assert lot2.quantity_remaining == 1.0
        assert lot1.is_fully_consumed is False
        assert lot2.is_fully_consumed is False
    
    @pytest.mark.asyncio
    async def test_buy_two_lots_sell_across_both(self, tax_engine, mock_session):
        """Buy 2 lots, sell across both lots."""
        owner_id = "user1"
        asset = "BTC"
        
        lot1_date = datetime(2025, 1, 1)
        lot2_date = datetime(2025, 1, 15)
        sell_date = datetime(2025, 2, 1)
        
        lot1 = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot1_date)
        lot2 = create_tax_lot(2, owner_id, asset, 1.0, 1.0, 41000.0, 41000.0, 2, lot2_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot1, lot2])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(3, owner_id, TradeSide.SELL, asset, 1.5, 63000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 2
        assert lot1.quantity_remaining == 0.0
        assert lot1.is_fully_consumed is True
        assert lot2.quantity_remaining == 0.5
        assert lot2.is_fully_consumed is False
    
    @pytest.mark.asyncio
    async def test_sell_more_than_available_consumes_all(self, tax_engine, mock_session):
        """Sell more than available → consumes all available lots."""
        owner_id = "user1"
        asset = "BTC"
        
        lot1_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot1 = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot1_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot1])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 2.0, 84000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        assert lot1.quantity_remaining == 0.0
        assert lot1.is_fully_consumed is True
    
    @pytest.mark.asyncio
    async def test_sell_with_no_lots_returns_empty(self, tax_engine, mock_session):
        """Sell with no available lots → returns empty list."""
        owner_id = "user1"
        asset = "BTC"
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(1, owner_id, TradeSide.SELL, asset, 1.0, 42000.0)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 0


class TestRealizedPnLCalculation:
    """Test realized P&L calculation."""
    
    @pytest.mark.asyncio
    async def test_sell_at_profit(self, tax_engine, mock_session):
        """Sell at profit → positive gain."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 45000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.quantity == 1.0
        assert gain.proceeds == 45000.0
        assert gain.cost_basis == 40000.0
        assert gain.gain_loss == 5000.0
    
    @pytest.mark.asyncio
    async def test_sell_at_loss(self, tax_engine, mock_session):
        """Sell at loss → negative gain."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 35000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.quantity == 1.0
        assert gain.proceeds == 35000.0
        assert gain.cost_basis == 40000.0
        assert gain.gain_loss == -5000.0
    
    @pytest.mark.asyncio
    async def test_partial_close_profit(self, tax_engine, mock_session):
        """Partial close with profit."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 2.0, 2.0, 40000.0, 80000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 45000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.quantity == 1.0
        assert gain.proceeds == 45000.0
        assert gain.cost_basis == 40000.0
        assert gain.gain_loss == 5000.0
        assert lot.quantity_remaining == 1.0
    
    @pytest.mark.asyncio
    async def test_full_close_profit(self, tax_engine, mock_session):
        """Full close with profit."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 50000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.quantity == 1.0
        assert gain.proceeds == 50000.0
        assert gain.cost_basis == 40000.0
        assert gain.gain_loss == 10000.0
        assert lot.quantity_remaining == 0.0
        assert lot.is_fully_consumed is True
    
    @pytest.mark.asyncio
    async def test_floating_point_precision_safety(self, tax_engine, mock_session):
        """Ensure floating-point precision is handled correctly."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 0.12345678, 0.12345678, 40123.456, 4953.512680231680, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 0.12345678, 5234.567, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert abs(gain.quantity - 0.12345678) < 1e-8
        assert abs(gain.proceeds - 5234.567) < 1e-6
        assert abs(gain.cost_basis - 4953.512680231680) < 1e-6
        assert lot.quantity_remaining == 0.0
        assert lot.is_fully_consumed is True


class TestFeeHandling:
    """Test fee handling in cost basis."""
    
    @pytest.mark.asyncio
    async def test_buy_fee_increases_cost_basis(self, tax_engine, mock_session):
        """Buy fee increases cost basis per unit."""
        owner_id = "user1"
        
        buy_trade = create_trade(
            1, owner_id, TradeSide.BUY, "BTC",
            base_amount=1.0,
            quote_amount=40000.0,
            fee_amount=100.0,
            modeled_cost=0.0,
            executed_at=datetime(2025, 1, 1)
        )
        
        lot = await tax_engine.process_buy(buy_trade)
        
        assert lot.quantity_acquired == 1.0
        assert lot.unit_cost == 40100.0
        assert lot.total_cost == 40100.0
    
    @pytest.mark.asyncio
    async def test_buy_with_modeled_cost_increases_basis(self, tax_engine, mock_session):
        """Buy with modeled cost increases cost basis."""
        owner_id = "user1"
        
        buy_trade = create_trade(
            1, owner_id, TradeSide.BUY, "BTC",
            base_amount=1.0,
            quote_amount=40000.0,
            fee_amount=100.0,
            modeled_cost=50.0,
            executed_at=datetime(2025, 1, 1)
        )
        
        lot = await tax_engine.process_buy(buy_trade)
        
        assert lot.quantity_acquired == 1.0
        assert lot.unit_cost == 40150.0
        assert lot.total_cost == 40150.0
    
    @pytest.mark.asyncio
    async def test_sell_fee_reduces_proceeds(self, tax_engine, mock_session):
        """Sell fee is NOT deducted from proceeds in gain calculation."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(
            2, owner_id, TradeSide.SELL, asset,
            base_amount=1.0,
            quote_amount=45000.0,
            fee_amount=100.0,
            executed_at=sell_date
        )
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.proceeds == 45000.0
        assert gain.cost_basis == 40000.0
        assert gain.gain_loss == 5000.0
    
    @pytest.mark.asyncio
    async def test_fee_included_in_realized_gain_loss(self, tax_engine, mock_session):
        """Fee is reflected through cost basis in gain/loss."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40100.0, 40100.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 45000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.cost_basis == 40100.0
        assert gain.gain_loss == 4900.0


class TestPositionClosing:
    """Test position closing scenarios."""
    
    @pytest.mark.asyncio
    async def test_partial_position_close(self, tax_engine, mock_session):
        """Partial position close leaves remaining quantity."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 2.0, 2.0, 40000.0, 80000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 0.5, 21000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        assert lot.quantity_remaining == 1.5
        assert lot.is_fully_consumed is False
    
    @pytest.mark.asyncio
    async def test_full_position_close_removes_lot(self, tax_engine, mock_session):
        """Full position close marks lot as consumed."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 42000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        assert lot.quantity_remaining == 0.0
        assert lot.is_fully_consumed is True
        assert lot.consumed_at == sell_date
    
    @pytest.mark.asyncio
    async def test_multiple_partial_closes_across_lots(self, tax_engine, mock_session):
        """Multiple partial closes consume lots in FIFO order."""
        owner_id = "user1"
        asset = "BTC"
        
        lot1_date = datetime(2025, 1, 1)
        lot2_date = datetime(2025, 1, 15)
        lot3_date = datetime(2025, 2, 1)
        sell_date = datetime(2025, 3, 1)
        
        lot1 = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot1_date)
        lot2 = create_tax_lot(2, owner_id, asset, 1.0, 1.0, 41000.0, 41000.0, 2, lot2_date)
        lot3 = create_tax_lot(3, owner_id, asset, 1.0, 1.0, 42000.0, 42000.0, 3, lot3_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot1, lot2, lot3])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(4, owner_id, TradeSide.SELL, asset, 2.3, 98000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 3
        assert lot1.quantity_remaining == 0.0
        assert lot1.is_fully_consumed is True
        assert lot2.quantity_remaining == 0.0
        assert lot2.is_fully_consumed is True
        assert abs(lot3.quantity_remaining - 0.7) < 1e-8
        assert lot3.is_fully_consumed is False


class TestYearBoundaryBehavior:
    """Test holding period and long-term vs short-term classification."""
    
    @pytest.mark.asyncio
    async def test_buy_dec_sell_jan_short_term(self, tax_engine, mock_session):
        """Buy in Dec, sell in Jan → short-term (< 365 days)."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2024, 12, 15)
        sell_date = datetime(2025, 1, 20)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 45000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.holding_period_days == 36
        assert gain.is_long_term is False
    
    @pytest.mark.asyncio
    async def test_buy_over_365_days_ago_long_term(self, tax_engine, mock_session):
        """Buy >365 days ago → long-term."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2024, 1, 1)
        sell_date = datetime(2025, 2, 1)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 50000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.holding_period_days == 397
        assert gain.is_long_term is True
    
    @pytest.mark.asyncio
    async def test_sell_exactly_at_365_days_short_term(self, tax_engine, mock_session):
        """Sell exactly at 365 days → short-term (> 365 required for long-term)."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2024, 1, 1, 12, 0, 0)
        sell_date = datetime(2024, 12, 31, 12, 0, 0)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 45000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.holding_period_days == 365
        assert gain.is_long_term is False
    
    @pytest.mark.asyncio
    async def test_sell_at_366_days_long_term(self, tax_engine, mock_session):
        """Sell at 366 days → long-term."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2024, 1, 1)
        sell_date = datetime(2025, 1, 3)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 45000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        gain = gains[0]
        assert gain.holding_period_days > 365
        assert gain.is_long_term is True


class TestComplexScenarios:
    """Test complex multi-trade scenarios."""
    
    @pytest.mark.asyncio
    async def test_three_buys_two_sells_fifo_order(self, tax_engine, mock_session):
        """Three buys, two sells → FIFO consumption order."""
        owner_id = "user1"
        asset = "BTC"
        
        lot1_date = datetime(2025, 1, 1)
        lot2_date = datetime(2025, 1, 10)
        lot3_date = datetime(2025, 1, 20)
        sell_date = datetime(2025, 2, 1)
        
        lot1 = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot1_date)
        lot2 = create_tax_lot(2, owner_id, asset, 1.0, 1.0, 41000.0, 41000.0, 2, lot2_date)
        lot3 = create_tax_lot(3, owner_id, asset, 1.0, 1.0, 42000.0, 42000.0, 3, lot3_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot1, lot2, lot3])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(4, owner_id, TradeSide.SELL, asset, 1.8, 77000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 2
        
        assert gains[0].quantity == 1.0
        assert gains[0].cost_basis == 40000.0
        
        assert abs(gains[1].quantity - 0.8) < 1e-8
        assert abs(gains[1].cost_basis - 32800.0) < 1e-6
        
        assert lot1.is_fully_consumed is True
        assert lot2.is_fully_consumed is False
        assert abs(lot2.quantity_remaining - 0.2) < 1e-8
        assert lot3.quantity_remaining == 1.0
    
    @pytest.mark.asyncio
    async def test_proceeds_split_proportionally_across_lots(self, tax_engine, mock_session):
        """Proceeds are split proportionally when consuming multiple lots."""
        owner_id = "user1"
        asset = "BTC"
        
        lot1_date = datetime(2025, 1, 1)
        lot2_date = datetime(2025, 1, 15)
        sell_date = datetime(2025, 2, 1)
        
        lot1 = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot1_date)
        lot2 = create_tax_lot(2, owner_id, asset, 1.0, 1.0, 41000.0, 41000.0, 2, lot2_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot1, lot2])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(3, owner_id, TradeSide.SELL, asset, 2.0, 90000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 2
        
        assert gains[0].quantity == 1.0
        assert gains[0].proceeds == 45000.0
        assert gains[0].cost_basis == 40000.0
        assert gains[0].gain_loss == 5000.0
        
        assert gains[1].quantity == 1.0
        assert gains[1].proceeds == 45000.0
        assert gains[1].cost_basis == 41000.0
        assert gains[1].gain_loss == 4000.0
    
    @pytest.mark.asyncio
    async def test_wash_sale_not_implemented(self, tax_engine, mock_session):
        """Wash sale rules are NOT implemented (future enhancement)."""
        owner_id = "user1"
        asset = "BTC"
        
        lot_date = datetime(2025, 1, 1)
        sell_date = datetime(2025, 1, 10)
        
        lot = create_tax_lot(1, owner_id, asset, 1.0, 1.0, 40000.0, 40000.0, 1, lot_date)
        
        mock_result = AsyncMock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[lot])))
        mock_session.execute.return_value = mock_result
        
        sell_trade = create_trade(2, owner_id, TradeSide.SELL, asset, 1.0, 35000.0, executed_at=sell_date)
        
        gains = await tax_engine.process_sell(sell_trade)
        
        assert len(gains) == 1
        assert gains[0].gain_loss == -5000.0
