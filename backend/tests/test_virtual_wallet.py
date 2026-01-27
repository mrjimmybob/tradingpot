"""Unit tests for virtual wallet service.

Tests budget tracking, balance calculations, and trade validation logic.
Pure business logic tests - no database, API, or external dependencies.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import Optional

from app.services.virtual_wallet import VirtualWalletService, WalletStatus, TradeValidation
from app.models import Bot


def create_mock_bot(
    bot_id: int = 1,
    budget: float = 10000.0,
    current_balance: float = None,
    total_pnl: float = 0.0,
    compound_enabled: bool = False,
) -> Mock:
    """Create a mock bot object."""
    bot = Mock(spec=Bot)
    bot.id = bot_id
    bot.budget = budget
    bot.current_balance = current_balance if current_balance is not None else budget
    bot.total_pnl = total_pnl
    bot.compound_enabled = compound_enabled
    bot.updated_at = datetime(2025, 1, 1, 12, 0, 0)
    return bot


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def wallet_service(mock_session):
    """Create a VirtualWalletService instance."""
    return VirtualWalletService(mock_session)


class TestBalanceCalculation:
    """Test balance calculation logic."""
    
    @pytest.mark.asyncio
    async def test_initial_balance_state(self, wallet_service, mock_session):
        """Initial balance state."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, total_pnl=0.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet is not None
        assert wallet.budget == 10000.0
        assert wallet.current_balance == 10000.0
        assert wallet.total_pnl == 0.0
        assert wallet.available_for_trade == 10000.0
        assert wallet.is_budget_exceeded is False
    
    @pytest.mark.asyncio
    async def test_balance_after_profit(self, wallet_service, mock_session):
        """Balance after profit."""
        bot = create_mock_bot(budget=10000.0, current_balance=10500.0, total_pnl=500.0, compound_enabled=True)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.current_balance == 10500.0
        assert wallet.total_pnl == 500.0
        assert wallet.available_for_trade == 10500.0
    
    @pytest.mark.asyncio
    async def test_balance_after_loss(self, wallet_service, mock_session):
        """Balance after loss."""
        bot = create_mock_bot(budget=10000.0, current_balance=9500.0, total_pnl=-500.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.current_balance == 9500.0
        assert wallet.total_pnl == -500.0
        assert wallet.available_for_trade == 9500.0
    
    @pytest.mark.asyncio
    async def test_balance_after_fees(self, wallet_service, mock_session):
        """Balance after fees."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, total_pnl=0.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.record_trade_result(bot_id=1, pnl=100.0, fees=10.0)
        
        assert success is True
        assert bot.total_pnl == 90.0
    
    @pytest.mark.asyncio
    async def test_rounding_floating_point_safety(self, wallet_service, mock_session):
        """Rounding / floating point safety."""
        bot = create_mock_bot(budget=10000.12345678, current_balance=10000.12345678)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.record_trade_result(bot_id=1, pnl=0.00000001, fees=0.00000001)
        
        assert success is True
        assert bot.current_balance == 10000.12345678


class TestDebitCreditOperations:
    """Test debit and credit operations."""
    
    @pytest.mark.asyncio
    async def test_credit_increases_balance(self, wallet_service, mock_session):
        """Credit increases balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, compound_enabled=True)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.record_win(bot_id=1, win_amount=500.0, fees=0)
        
        assert success is True
        assert bot.current_balance == 10500.0
        assert bot.total_pnl == 500.0
    
    @pytest.mark.asyncio
    async def test_debit_reduces_balance(self, wallet_service, mock_session):
        """Debit reduces balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.record_loss(bot_id=1, loss_amount=500.0, fees=0)
        
        assert success is True
        assert bot.current_balance == 9500.0
        assert bot.total_pnl == -500.0
    
    @pytest.mark.asyncio
    async def test_debit_exact_balance_allowed(self, wallet_service, mock_session):
        """Debit exact balance → allowed."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=10000.0)
        
        assert validation.is_valid is True
    
    @pytest.mark.asyncio
    async def test_debit_exceeding_balance_rejected(self, wallet_service, mock_session):
        """Debit exceeding balance → rejected."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=15000.0)
        
        assert validation.is_valid is False
        assert "exceeds available balance" in validation.reason
        assert validation.max_trade_amount == 10000.0
    
    @pytest.mark.asyncio
    async def test_debit_zero_amount_rejected(self, wallet_service, mock_session):
        """Debit zero amount → rejected."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=0.0)
        
        assert validation.is_valid is False
        assert "must be positive" in validation.reason
    
    @pytest.mark.asyncio
    async def test_debit_negative_amount_rejected(self, wallet_service, mock_session):
        """Debit negative amount → rejected."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=-500.0)
        
        assert validation.is_valid is False
        assert "must be positive" in validation.reason


class TestBudgetIsolation:
    """Test budget isolation between bots."""
    
    @pytest.mark.asyncio
    async def test_two_bots_with_separate_budgets(self, wallet_service, mock_session):
        """Two bots with separate budgets."""
        bot1 = create_mock_bot(bot_id=1, budget=10000.0, current_balance=10000.0)
        bot2 = create_mock_bot(bot_id=2, budget=5000.0, current_balance=5000.0)
        
        mock_result1 = AsyncMock()
        mock_result1.scalar_one_or_none = Mock(return_value=bot1)
        
        mock_result2 = AsyncMock()
        mock_result2.scalar_one_or_none = Mock(return_value=bot2)
        
        mock_session.execute = AsyncMock(side_effect=[mock_result1, mock_result2])
        
        wallet1 = await wallet_service.get_wallet_status(bot_id=1)
        wallet2 = await wallet_service.get_wallet_status(bot_id=2)
        
        assert wallet1.budget == 10000.0
        assert wallet2.budget == 5000.0
    
    @pytest.mark.asyncio
    async def test_operations_on_bot_a_do_not_affect_bot_b(self, wallet_service, mock_session):
        """Operations on bot A do NOT affect bot B."""
        bot1 = create_mock_bot(bot_id=1, budget=10000.0, current_balance=10000.0)
        bot2 = create_mock_bot(bot_id=2, budget=5000.0, current_balance=5000.0)
        
        bot1_initial_balance = bot1.current_balance
        bot2_initial_balance = bot2.current_balance
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot1)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_loss(bot_id=1, loss_amount=1000.0, fees=0)
        
        assert bot1.current_balance == bot1_initial_balance - 1000.0
        assert bot2.current_balance == bot2_initial_balance
    
    @pytest.mark.asyncio
    async def test_cross_bot_leakage_must_fail(self, wallet_service, mock_session):
        """Cross-bot leakage must fail tests."""
        bot1 = create_mock_bot(bot_id=1, budget=10000.0, current_balance=10000.0)
        bot2 = create_mock_bot(bot_id=2, budget=5000.0, current_balance=5000.0)
        
        mock_result1 = AsyncMock()
        mock_result1.scalar_one_or_none = Mock(return_value=bot1)
        
        mock_result2 = AsyncMock()
        mock_result2.scalar_one_or_none = Mock(return_value=bot2)
        
        mock_session.execute = AsyncMock(side_effect=[mock_result1, mock_result2])
        
        validation1 = await wallet_service.validate_trade(bot_id=1, trade_amount=5000.0)
        validation2 = await wallet_service.validate_trade(bot_id=2, trade_amount=5000.0)
        
        assert validation1.is_valid is True
        assert validation2.is_valid is True
        assert validation1.max_trade_amount != validation2.max_trade_amount


class TestInsufficientFundsRejection:
    """Test insufficient funds rejection."""
    
    @pytest.mark.asyncio
    async def test_reject_trade_larger_than_available_balance(self, wallet_service, mock_session):
        """Reject trade larger than available balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=5000.0, total_pnl=-5000.0, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=8000.0)
        
        assert validation.is_valid is False
        assert "exceeds available balance" in validation.reason
    
    @pytest.mark.asyncio
    async def test_reject_after_multiple_losses(self, wallet_service, mock_session):
        """Reject after multiple losses."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_loss(bot_id=1, loss_amount=4000.0, fees=0)
        await wallet_service.record_loss(bot_id=1, loss_amount=4000.0, fees=0)
        await wallet_service.record_loss(bot_id=1, loss_amount=1500.0, fees=0)
        
        assert bot.current_balance == 500.0
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=1000.0)
        
        assert validation.is_valid is False
    
    @pytest.mark.asyncio
    async def test_reject_if_budget_exhausted(self, wallet_service, mock_session):
        """Reject if budget exhausted."""
        bot = create_mock_bot(budget=10000.0, current_balance=0.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=100.0)
        
        assert validation.is_valid is False
        assert "Budget exhausted" in validation.reason
    
    @pytest.mark.asyncio
    async def test_correct_error_message_returned(self, wallet_service, mock_session):
        """Correct error message / result returned."""
        bot = create_mock_bot(budget=10000.0, current_balance=1000.0, total_pnl=-9000.0, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=2000.0)
        
        assert validation.is_valid is False
        assert "$2000.00" in validation.reason
        assert "$1000.00" in validation.reason
        assert validation.max_trade_amount == 1000.0


class TestNegativeBalancePrevention:
    """Test negative balance prevention."""
    
    @pytest.mark.asyncio
    async def test_balance_never_goes_below_zero(self, wallet_service, mock_session):
        """Balance must never go below zero."""
        bot = create_mock_bot(budget=10000.0, current_balance=100.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.available_for_trade >= 0
    
    @pytest.mark.asyncio
    async def test_multiple_losses_in_sequence_clamp_correctly(self, wallet_service, mock_session):
        """Multiple losses in sequence still clamp correctly."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_loss(bot_id=1, loss_amount=5000.0, fees=0)
        await wallet_service.record_loss(bot_id=1, loss_amount=5000.0, fees=0)
        await wallet_service.record_loss(bot_id=1, loss_amount=5000.0, fees=0)
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.available_for_trade >= 0
    
    @pytest.mark.asyncio
    async def test_precision_edge_cases(self, wallet_service, mock_session):
        """Precision edge cases (e.g. 0.0000001)."""
        bot = create_mock_bot(budget=10000.0, current_balance=0.0000001, total_pnl=-9999.9999999, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.available_for_trade >= 0
        assert wallet.available_for_trade < 0.01


class TestEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.asyncio
    async def test_very_small_trade_amounts(self, wallet_service, mock_session):
        """Very small trade amounts."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=0.01)
        
        assert validation.is_valid is True
    
    @pytest.mark.asyncio
    async def test_very_large_trade_amounts(self, wallet_service, mock_session):
        """Very large trade amounts."""
        bot = create_mock_bot(budget=1000000.0, current_balance=1000000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        validation = await wallet_service.validate_trade(bot_id=1, trade_amount=999999.99)
        
        assert validation.is_valid is True
    
    @pytest.mark.asyncio
    async def test_compound_vs_non_compound_mode(self, wallet_service, mock_session):
        """Compound vs non-compound mode."""
        bot_compound = create_mock_bot(budget=10000.0, current_balance=10000.0, compound_enabled=True)
        bot_non_compound = create_mock_bot(budget=10000.0, current_balance=10000.0, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot_compound)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_win(bot_id=1, win_amount=1000.0, fees=0)
        compound_balance = bot_compound.current_balance
        
        mock_result.scalar_one_or_none = Mock(return_value=bot_non_compound)
        await wallet_service.record_win(bot_id=1, win_amount=1000.0, fees=0)
        non_compound_balance = bot_non_compound.current_balance
        
        assert compound_balance == 11000.0
        assert non_compound_balance == 10000.0
    
    @pytest.mark.asyncio
    async def test_multiple_sequential_trades(self, wallet_service, mock_session):
        """Multiple sequential trades."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_win(bot_id=1, win_amount=500.0, fees=10.0)
        await wallet_service.record_loss(bot_id=1, loss_amount=200.0, fees=5.0)
        await wallet_service.record_win(bot_id=1, win_amount=300.0, fees=8.0)
        
        expected_pnl = 500.0 - 10.0 - 200.0 - 5.0 + 300.0 - 8.0
        assert bot.total_pnl == expected_pnl
    
    @pytest.mark.asyncio
    async def test_reset_wallet_behavior(self, wallet_service, mock_session):
        """Reset wallet behavior."""
        bot = create_mock_bot(budget=10000.0, current_balance=8000.0, total_pnl=-2000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.reset_wallet(bot_id=1)
        
        assert success is True
        assert bot.current_balance == 10000.0
        assert bot.total_pnl == 0.0
    
    @pytest.mark.asyncio
    async def test_updating_budget_mid_run(self, wallet_service, mock_session):
        """Updating budget mid-run."""
        bot = create_mock_bot(budget=10000.0, current_balance=8000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.update_budget(bot_id=1, new_budget=15000.0)
        
        assert success is True
        assert bot.budget == 15000.0
        assert bot.current_balance == 12000.0
    
    @pytest.mark.asyncio
    async def test_bot_not_found(self, wallet_service, mock_session):
        """Bot not found."""
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=None)
        mock_session.execute.return_value = mock_result
        
        wallet = await wallet_service.get_wallet_status(bot_id=999)
        
        assert wallet is None


class TestCompoundModeLogic:
    """Test compound mode specific logic."""
    
    @pytest.mark.asyncio
    async def test_compound_mode_profit_increases_available_balance(self, wallet_service, mock_session):
        """Compound mode: profit increases available balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, total_pnl=0.0, compound_enabled=True)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_win(bot_id=1, win_amount=1000.0, fees=0)
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.available_for_trade == 11000.0
    
    @pytest.mark.asyncio
    async def test_non_compound_mode_profit_does_not_increase_available(self, wallet_service, mock_session):
        """Non-compound mode: profit does NOT increase available balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, total_pnl=0.0, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_win(bot_id=1, win_amount=1000.0, fees=0)
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.available_for_trade == 10000.0
    
    @pytest.mark.asyncio
    async def test_non_compound_mode_loss_reduces_available(self, wallet_service, mock_session):
        """Non-compound mode: loss reduces available balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, total_pnl=0.0, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        await wallet_service.record_loss(bot_id=1, loss_amount=2000.0, fees=0)
        
        wallet = await wallet_service.get_wallet_status(bot_id=1)
        
        assert wallet.available_for_trade == 8000.0
    
    @pytest.mark.asyncio
    async def test_switching_compound_mode(self, wallet_service, mock_session):
        """Switching compound mode."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0, compound_enabled=False)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.set_compound_mode(bot_id=1, enabled=True)
        
        assert success is True
        assert bot.compound_enabled is True


class TestPnLSummary:
    """Test P&L summary functionality."""
    
    @pytest.mark.asyncio
    async def test_pnl_summary_calculation(self, wallet_service, mock_session):
        """P&L summary calculation."""
        bot = create_mock_bot(budget=10000.0, current_balance=10500.0, total_pnl=500.0, compound_enabled=True)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        summary = await wallet_service.get_pnl_summary(bot_id=1)
        
        assert summary is not None
        assert summary["budget"] == 10000.0
        assert summary["current_balance"] == 10500.0
        assert summary["total_pnl"] == 500.0
        assert summary["pnl_percent"] == 5.0
    
    @pytest.mark.asyncio
    async def test_pnl_summary_with_loss(self, wallet_service, mock_session):
        """P&L summary with loss."""
        bot = create_mock_bot(budget=10000.0, current_balance=9000.0, total_pnl=-1000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        summary = await wallet_service.get_pnl_summary(bot_id=1)
        
        assert summary is not None
        assert summary["pnl_percent"] == -10.0


class TestBudgetUpdateLogic:
    """Test budget update logic."""
    
    @pytest.mark.asyncio
    async def test_budget_update_with_zero_rejected(self, wallet_service, mock_session):
        """Budget update with zero rejected."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.update_budget(bot_id=1, new_budget=0.0)
        
        assert success is False
        assert "must be positive" in msg
    
    @pytest.mark.asyncio
    async def test_budget_update_with_negative_rejected(self, wallet_service, mock_session):
        """Budget update with negative rejected."""
        bot = create_mock_bot(budget=10000.0, current_balance=10000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.update_budget(bot_id=1, new_budget=-5000.0)
        
        assert success is False
        assert "must be positive" in msg
    
    @pytest.mark.asyncio
    async def test_budget_update_scales_current_balance(self, wallet_service, mock_session):
        """Budget update scales current balance."""
        bot = create_mock_bot(budget=10000.0, current_balance=5000.0)
        
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = Mock(return_value=bot)
        mock_session.execute.return_value = mock_result
        
        success, msg = await wallet_service.update_budget(bot_id=1, new_budget=20000.0)
        
        assert success is True
        assert bot.budget == 20000.0
        assert bot.current_balance == 10000.0
