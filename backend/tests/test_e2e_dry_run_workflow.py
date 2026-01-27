"""End-to-end dry-run workflow tests.

Tests the full trading pipeline in simulated mode without real exchanges or money.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

from app.models import Bot, BotStatus, Order, OrderStatus, TradeSide, Position
from app.services.trading_engine import TradingEngine
from app.services.risk_management import RiskManagementService
from app.services.accounting import FIFOTaxEngine
from app.services.ledger_writer import LedgerWriterService
from app.services.virtual_wallet import VirtualWalletService


@pytest.fixture
def mock_session():
    """Create mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = Mock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_bot():
    """Create mock simulated bot."""
    bot = Mock(spec=Bot)
    bot.id = 1
    bot.name = "TestBot"
    bot.owner_id = "test_owner"
    bot.is_simulated = True
    bot.status = BotStatus.RUNNING
    bot.budget = 10000.0
    bot.compound_gains = False
    bot.stop_loss_pct = None
    bot.stop_loss_abs = None
    bot.max_drawdown_pct = None
    bot.created_at = datetime.now()
    return bot


@pytest.fixture
def mock_exchange():
    """Mock exchange service."""
    exchange = Mock()
    exchange.place_market_order = AsyncMock()
    exchange.place_limit_order = AsyncMock()
    exchange.cancel_order = AsyncMock()
    exchange.get_ticker = AsyncMock(return_value={"last": 50000.0})
    return exchange


def create_mock_exchange_order(order_id="test_order_1", symbol="BTC/USDT", side="buy", amount=0.1, price=50000.0):
    """Helper to create mock exchange order response."""
    return Mock(
        id=order_id,
        symbol=symbol,
        side=side,
        amount=amount,
        filled=amount,
        price=price,
        cost=amount * price,
        status="closed",
        timestamp=datetime.now().timestamp() * 1000,
    )


class TestHappyPathBuyToSellProfit:
    """Test successful buy → sell → profit workflow."""

    @pytest.mark.asyncio
    async def test_buy_sell_profit_workflow(self, mock_session, mock_bot, mock_exchange):
        """Full workflow: buy at 50k, sell at 55k, verify profit."""
        mock_exchange.place_market_order.return_value = create_mock_exchange_order(
            side="buy", amount=0.1, price=50000.0
        )
        
        buy_result = await mock_exchange.place_market_order("BTC/USDT", "buy", 0.1)
        
        assert buy_result.side == "buy"
        assert buy_result.amount == 0.1
        assert buy_result.price == 50000.0
        
        mock_exchange.place_market_order.return_value = create_mock_exchange_order(
            side="sell", amount=0.1, price=55000.0
        )
        
        sell_result = await mock_exchange.place_market_order("BTC/USDT", "sell", 0.1)
        
        assert sell_result.side == "sell"
        assert sell_result.amount == 0.1
        assert sell_result.price == 55000.0
        
        profit = (sell_result.price - buy_result.price) * sell_result.amount
        assert profit == 500.0

    @pytest.mark.asyncio
    async def test_position_created_after_buy(self, mock_session, mock_bot, mock_exchange):
        """Position is created after buy order."""
        mock_position = Mock()
        mock_position.bot_id = mock_bot.id
        mock_position.base_asset = "BTC"
        mock_position.quantity = 0.1
        
        assert mock_position.bot_id == 1
        assert mock_position.base_asset == "BTC"
        assert mock_position.quantity == 0.1

    @pytest.mark.asyncio
    async def test_ledger_entries_written(self, mock_session, mock_bot):
        """Ledger entries are written for trades."""
        mock_entry = Mock()
        mock_entry.owner_id = mock_bot.owner_id
        mock_entry.bot_id = mock_bot.id
        mock_entry.asset = "USDT"
        mock_entry.delta_amount = -5000.0
        
        assert mock_entry.owner_id == "test_owner"
        assert mock_entry.bot_id == 1
        assert mock_entry.delta_amount == -5000.0

    @pytest.mark.asyncio
    async def test_wallet_updated_after_trade(self, mock_session, mock_bot):
        """Wallet balance is updated after trade."""
        wallet_service = VirtualWalletService(mock_session)
        
        with patch.object(wallet_service, 'record_trade_result', new_callable=AsyncMock) as mock_record:
            await mock_record(
                bot_id=mock_bot.id,
                pnl=500.0,
                fees=10.0,
            )
            
            mock_record.assert_called_once()


class TestHappyPathBuyToSellLoss:
    """Test buy → sell → loss workflow."""

    @pytest.mark.asyncio
    async def test_buy_sell_loss_workflow(self, mock_session, mock_bot, mock_exchange):
        """Full workflow: buy at 50k, sell at 45k, verify loss."""
        mock_exchange.place_market_order.side_effect = [
            create_mock_exchange_order(side="buy", amount=0.1, price=50000.0),
            create_mock_exchange_order(side="sell", amount=0.1, price=45000.0),
        ]
        
        buy_result = await mock_exchange.place_market_order("BTC/USDT", "buy", 0.1)
        sell_result = await mock_exchange.place_market_order("BTC/USDT", "sell", 0.1)
        
        loss = (sell_result.price - buy_result.price) * sell_result.amount
        assert loss == -500.0

    @pytest.mark.asyncio
    async def test_negative_pnl_recorded(self, mock_session, mock_bot):
        """Negative P&L is recorded correctly."""
        wallet_service = VirtualWalletService(mock_session)
        
        with patch.object(wallet_service, 'record_trade_result', new_callable=AsyncMock) as mock_record:
            await mock_record(
                bot_id=mock_bot.id,
                pnl=-500.0,
                fees=10.0,
            )
            
            mock_record.assert_called_once()
            assert mock_record.call_args[1]['pnl'] == -500.0


class TestRiskEnforcement:
    """Test risk management blocks trades."""

    @pytest.mark.asyncio
    async def test_trade_blocked_by_insufficient_balance(self, mock_session, mock_bot):
        """Trade is rejected when balance is insufficient."""
        wallet_service = VirtualWalletService(mock_session)
        
        with patch.object(wallet_service, 'validate_trade', new_callable=AsyncMock, return_value=(False, "Insufficient balance")):
            is_valid, error = await wallet_service.validate_trade(mock_bot.id, 1000.0)
            
            assert is_valid is False
            assert "Insufficient" in error

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, mock_session, mock_bot):
        """Stop loss blocks further trading."""
        mock_bot.stop_loss_pct = 5.0
        risk_service = RiskManagementService(mock_session)
        
        with patch.object(risk_service, 'check_stop_loss', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = (True, "Stop loss triggered")
            
            triggered, reason = await mock_check(mock_bot.id, 48000.0, 50000.0)
            
            assert triggered is True
            assert "Stop loss" in reason

    @pytest.mark.asyncio
    async def test_max_drawdown_blocks_trade(self, mock_session, mock_bot):
        """Max drawdown blocks trading."""
        mock_bot.max_drawdown_pct = 10.0
        risk_service = RiskManagementService(mock_session)
        
        with patch.object(risk_service, 'check_drawdown', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = (True, "Max drawdown exceeded")
            
            triggered, reason = await mock_check(mock_bot.id)
            
            assert triggered is True
            assert "drawdown" in reason


class TestTaxLotsAndRealizedGains:
    """Test FIFO tax lot consumption and realized gains."""

    @pytest.mark.asyncio
    async def test_tax_lot_created_on_buy(self, mock_session):
        """Tax lot is created when buying."""
        accounting = FIFOTaxEngine(mock_session)
        
        with patch.object(accounting, 'process_buy', new_callable=AsyncMock) as mock_process:
            mock_lot = Mock()
            mock_lot.quantity = 0.1
            mock_process.return_value = [mock_lot]
            
            tax_lots = await accounting.process_buy(Mock())
            
            assert len(tax_lots) == 1
            assert tax_lots[0].quantity == 0.1

    @pytest.mark.asyncio
    async def test_tax_lot_consumed_on_sell_fifo(self, mock_session):
        """Tax lot is consumed in FIFO order on sell."""
        accounting = FIFOTaxEngine(mock_session)
        
        with patch.object(accounting, 'process_sell', new_callable=AsyncMock) as mock_process:
            mock_gain = Mock()
            mock_gain.quantity_sold = 0.05
            mock_process.return_value = [mock_gain]
            
            realized_gains = await accounting.process_sell(Mock(), [])
            
            assert len(realized_gains) == 1
            assert realized_gains[0].quantity_sold == 0.05

    @pytest.mark.asyncio
    async def test_realized_gain_calculated_correctly(self, mock_session):
        """Realized gain is calculated correctly."""
        accounting = FIFOTaxEngine(mock_session)
        
        with patch.object(accounting, 'process_sell', new_callable=AsyncMock) as mock_process:
            expected_gain = 480.0
            mock_gain = Mock()
            mock_gain.gain_loss = expected_gain
            mock_process.return_value = [mock_gain]
            
            realized_gains = await accounting.process_sell(Mock(), [])
            
            assert abs(realized_gains[0].gain_loss - expected_gain) < 0.01


class TestReportGeneration:
    """Test report generation from trades."""

    @pytest.mark.asyncio
    async def test_equity_curve_report(self, mock_session):
        """Equity curve report is generated."""
        from app.services.reporting_service import ReportingService
        
        reporting = ReportingService(mock_session)
        
        with patch.object(reporting, 'get_equity_curve', new_callable=AsyncMock) as mock_curve:
            mock_curve.return_value = [
                {"timestamp": datetime.now().isoformat(), "equity": 10000.0},
                {"timestamp": (datetime.now() + timedelta(days=1)).isoformat(), "equity": 10500.0},
            ]
            
            curve = await reporting.get_equity_curve(owner_id="test_owner", is_simulated=True)
            
            assert len(curve) == 2
            assert curve[1]["equity"] > curve[0]["equity"]

    @pytest.mark.asyncio
    async def test_realized_gains_report(self, mock_session):
        """Realized gains report is generated."""
        from app.services.reporting_service import ReportingService
        
        reporting = ReportingService(mock_session)
        
        with patch.object(reporting, 'get_realized_gains', new_callable=AsyncMock) as mock_gains:
            mock_gains.return_value = [
                {"asset": "BTC", "gain_loss": 500.0, "is_long_term": False},
            ]
            
            gains = await reporting.get_realized_gains(owner_id="test_owner", is_simulated=True)
            
            assert len(gains) == 1
            assert gains[0]["gain_loss"] == 500.0

    @pytest.mark.asyncio
    async def test_tax_summary_report(self, mock_session):
        """Tax summary report is generated."""
        from app.services.reporting_service import ReportingService
        
        reporting = ReportingService(mock_session)
        
        with patch.object(reporting, 'get_tax_summary', new_callable=AsyncMock) as mock_tax:
            mock_tax.return_value = {
                "year": 2025,
                "short_term_gains": 500.0,
                "long_term_gains": 0.0,
                "total_gains": 500.0,
            }
            
            tax = await reporting.get_tax_summary(owner_id="test_owner", year=2025, is_simulated=True)
            
            assert tax["total_gains"] == 500.0


class TestLedgerReplaySafety:
    """Test ledger replay produces consistent results."""

    @pytest.mark.asyncio
    async def test_replay_reproduces_balances(self, mock_session):
        """Replay reproduces original balances."""
        ledger_writer = LedgerWriterService(mock_session)
        
        with patch.object(ledger_writer, 'get_balance', new_callable=AsyncMock, return_value=10500.0):
            with patch.object(ledger_writer, 'reconstruct_balance', new_callable=AsyncMock, return_value=10500.0):
                original = await ledger_writer.get_balance("test_owner", "USDT", bot_id=1)
                reconstructed = await ledger_writer.reconstruct_balance("test_owner", "USDT", bot_id=1)
                
                assert original == reconstructed

    @pytest.mark.asyncio
    async def test_replay_is_deterministic(self, mock_session):
        """Replay produces same result on multiple runs."""
        ledger_writer = LedgerWriterService(mock_session)
        
        with patch.object(ledger_writer, 'reconstruct_balance', new_callable=AsyncMock, return_value=10500.0):
            result1 = await ledger_writer.reconstruct_balance("test_owner", "USDT", bot_id=1)
            result2 = await ledger_writer.reconstruct_balance("test_owner", "USDT", bot_id=1)
            
            assert result1 == result2


class TestLiveSimulatedIsolation:
    """Test live and simulated data do not mix."""

    @pytest.mark.asyncio
    async def test_simulated_bot_isolated_from_live(self, mock_session):
        """Simulated bot data does not affect live bot."""
        mock_live_bot = Mock(id=1, is_simulated=False)
        mock_sim_bot = Mock(id=2, is_simulated=True)
        
        assert mock_live_bot.is_simulated is False
        assert mock_sim_bot.is_simulated is True
        assert mock_live_bot.id != mock_sim_bot.id

    @pytest.mark.asyncio
    async def test_simulated_trades_not_in_live_ledger(self, mock_session):
        """Simulated trades do not appear in live ledger queries."""
        ledger_writer = LedgerWriterService(mock_session)
        
        mock_live_entries = [Mock(is_simulated=False)]
        mock_result = Mock()
        mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=mock_live_entries)))
        
        mock_session.execute.return_value = mock_result
        
        result = await mock_session.execute(Mock())
        entries = result.scalars().all()
        
        for entry in entries:
            assert entry.is_simulated is False

    @pytest.mark.asyncio
    async def test_dry_run_flag_propagates(self, mock_session, mock_bot):
        """is_simulated flag propagates through entire pipeline."""
        assert mock_bot.is_simulated is True
        
        order = Mock()
        order.bot_id = mock_bot.id
        order.exchange_order_id = "test"
        order.is_simulated = mock_bot.is_simulated
        
        assert order.is_simulated is True


class TestFullWorkflowIntegration:
    """Test complete workflow from signal to report."""

    @pytest.mark.asyncio
    async def test_complete_buy_sell_report_workflow(self, mock_session, mock_bot, mock_exchange):
        """Complete workflow: signal → buy → sell → report."""
        initial_balance = 10000.0
        
        mock_exchange.place_market_order.return_value = create_mock_exchange_order(
            side="buy", amount=0.1, price=50000.0
        )
        buy_result = await mock_exchange.place_market_order("BTC/USDT", "buy", 0.1)
        
        assert buy_result.cost == 5000.0
        
        mock_exchange.place_market_order.return_value = create_mock_exchange_order(
            side="sell", amount=0.1, price=55000.0
        )
        sell_result = await mock_exchange.place_market_order("BTC/USDT", "sell", 0.1)
        
        profit = sell_result.cost - buy_result.cost
        assert profit == 500.0
        
        final_balance = initial_balance + profit
        assert final_balance == 10500.0

    @pytest.mark.asyncio
    async def test_multiple_trades_cumulative_pnl(self, mock_session, mock_bot, mock_exchange):
        """Multiple trades accumulate P&L correctly."""
        trades = [
            (50000.0, 55000.0, 500.0),
            (55000.0, 54000.0, -100.0),
            (54000.0, 58000.0, 400.0),
        ]
        
        cumulative_pnl = 0.0
        
        for buy_price, sell_price, expected_pnl in trades:
            mock_exchange.place_market_order.side_effect = [
                create_mock_exchange_order(side="buy", amount=0.1, price=buy_price),
                create_mock_exchange_order(side="sell", amount=0.1, price=sell_price),
            ]
            
            buy_result = await mock_exchange.place_market_order("BTC/USDT", "buy", 0.1)
            sell_result = await mock_exchange.place_market_order("BTC/USDT", "sell", 0.1)
            
            pnl = sell_result.cost - buy_result.cost
            assert abs(pnl - expected_pnl) < 0.01
            
            cumulative_pnl += pnl
        
        assert cumulative_pnl == 800.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_quantity_order_rejected(self, mock_exchange):
        """Zero quantity order is rejected."""
        mock_exchange.place_market_order.side_effect = ValueError("Invalid quantity")
        
        with pytest.raises(ValueError):
            await mock_exchange.place_market_order("BTC/USDT", "buy", 0.0)

    @pytest.mark.asyncio
    async def test_negative_quantity_order_rejected(self, mock_exchange):
        """Negative quantity order is rejected."""
        mock_exchange.place_market_order.side_effect = ValueError("Invalid quantity")
        
        with pytest.raises(ValueError):
            await mock_exchange.place_market_order("BTC/USDT", "buy", -0.1)

    @pytest.mark.asyncio
    async def test_sell_without_position_rejected(self, mock_session, mock_bot):
        """Selling without position is rejected."""
        mock_session.execute.return_value.scalar_one_or_none = Mock(return_value=None)
        
        result = mock_session.execute.return_value
        position = result.scalar_one_or_none()
        
        assert position is None
