"""Virtual wallet service for budget tracking and enforcement."""

import logging
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import Bot

logger = logging.getLogger(__name__)


@dataclass
class WalletStatus:
    """Virtual wallet status."""
    budget: float
    current_balance: float
    total_pnl: float
    available_for_trade: float
    compound_enabled: bool
    is_budget_exceeded: bool


@dataclass
class TradeValidation:
    """Trade validation result."""
    is_valid: bool
    reason: str
    max_trade_amount: float


class VirtualWalletService:
    """Service for managing virtual wallet and budget enforcement."""

    def __init__(self, session: AsyncSession):
        """Initialize virtual wallet service.

        Args:
            session: Database session
        """
        self.session = session

    async def get_wallet_status(self, bot_id: int) -> Optional[WalletStatus]:
        """Get current wallet status for a bot.

        Args:
            bot_id: The bot ID

        Returns:
            WalletStatus or None if bot not found
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return None

        available = self._calculate_available_balance(bot)

        return WalletStatus(
            budget=bot.budget,
            current_balance=bot.current_balance,
            total_pnl=bot.total_pnl,
            available_for_trade=available,
            compound_enabled=bot.compound_enabled,
            is_budget_exceeded=bot.current_balance <= 0,
        )

    def _calculate_available_balance(self, bot: Bot) -> float:
        """Calculate available balance for trading.

        If compound mode is disabled, wins don't increase available balance.
        If compound mode is enabled, wins are added to available balance.

        Args:
            bot: The bot model

        Returns:
            Available balance for trading
        """
        if bot.compound_enabled:
            # Compound mode: use current balance (budget + cumulative P&L)
            return max(0, bot.current_balance)
        else:
            # Non-compound mode: use budget minus losses only
            # If we've had net gains, still cap at original budget
            if bot.total_pnl >= 0:
                return bot.budget
            else:
                # Losses reduce available balance
                return max(0, bot.budget + bot.total_pnl)

    async def validate_trade(
        self,
        bot_id: int,
        trade_amount: float,
    ) -> TradeValidation:
        """Validate if a trade is within budget limits.

        Args:
            bot_id: The bot ID
            trade_amount: Amount to trade in quote currency (e.g., USDT)

        Returns:
            TradeValidation result
        """
        wallet = await self.get_wallet_status(bot_id)

        if not wallet:
            return TradeValidation(
                is_valid=False,
                reason="Bot not found",
                max_trade_amount=0,
            )

        if wallet.is_budget_exceeded:
            return TradeValidation(
                is_valid=False,
                reason="Budget exhausted - current balance is zero or negative",
                max_trade_amount=0,
            )

        if trade_amount > wallet.available_for_trade:
            return TradeValidation(
                is_valid=False,
                reason=f"Trade amount ${trade_amount:.2f} exceeds available balance ${wallet.available_for_trade:.2f}",
                max_trade_amount=wallet.available_for_trade,
            )

        if trade_amount <= 0:
            return TradeValidation(
                is_valid=False,
                reason="Trade amount must be positive",
                max_trade_amount=wallet.available_for_trade,
            )

        return TradeValidation(
            is_valid=True,
            reason="Trade validated successfully",
            max_trade_amount=wallet.available_for_trade,
        )

    async def record_trade_result(
        self,
        bot_id: int,
        pnl: float,
        fees: float = 0,
    ) -> Tuple[bool, str]:
        """Record trade result and update wallet balance.

        Args:
            bot_id: The bot ID
            pnl: Profit/loss from the trade (positive = profit, negative = loss)
            fees: Trading fees (always positive, will be subtracted)

        Returns:
            Tuple of (success, message)
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return False, "Bot not found"

        # Calculate net P&L after fees
        net_pnl = pnl - fees

        # Update total P&L
        bot.total_pnl += net_pnl

        # Update current balance based on compound mode
        if bot.compound_enabled:
            # Compound mode: both wins and losses affect balance
            bot.current_balance += net_pnl
        else:
            # Non-compound mode: only losses affect balance
            if net_pnl < 0:
                bot.current_balance += net_pnl

        bot.updated_at = datetime.utcnow()
        await self.session.commit()

        logger.info(
            f"Bot {bot_id}: Recorded trade P&L={pnl:.2f}, fees={fees:.2f}, "
            f"net={net_pnl:.2f}, new_balance={bot.current_balance:.2f}"
        )

        return True, f"Trade recorded: P&L={net_pnl:.2f}"

    async def record_loss(
        self,
        bot_id: int,
        loss_amount: float,
        fees: float = 0,
    ) -> Tuple[bool, str]:
        """Record a loss and update wallet balance.

        Args:
            bot_id: The bot ID
            loss_amount: The loss amount (positive number representing loss)
            fees: Trading fees

        Returns:
            Tuple of (success, message)
        """
        # Losses are recorded as negative P&L
        return await self.record_trade_result(bot_id, -abs(loss_amount), fees)

    async def record_win(
        self,
        bot_id: int,
        win_amount: float,
        fees: float = 0,
    ) -> Tuple[bool, str]:
        """Record a win and update wallet balance.

        Args:
            bot_id: The bot ID
            win_amount: The win amount (positive number representing profit)
            fees: Trading fees

        Returns:
            Tuple of (success, message)
        """
        # Wins are recorded as positive P&L
        return await self.record_trade_result(bot_id, abs(win_amount), fees)

    async def reset_wallet(self, bot_id: int) -> Tuple[bool, str]:
        """Reset wallet to initial budget.

        Args:
            bot_id: The bot ID

        Returns:
            Tuple of (success, message)
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return False, "Bot not found"

        bot.current_balance = bot.budget
        bot.total_pnl = 0.0
        bot.updated_at = datetime.utcnow()
        await self.session.commit()

        logger.info(f"Bot {bot_id}: Wallet reset to budget ${bot.budget:.2f}")
        return True, f"Wallet reset to ${bot.budget:.2f}"

    async def update_budget(
        self,
        bot_id: int,
        new_budget: float,
    ) -> Tuple[bool, str]:
        """Update bot budget.

        Args:
            bot_id: The bot ID
            new_budget: New budget amount

        Returns:
            Tuple of (success, message)
        """
        if new_budget <= 0:
            return False, "Budget must be positive"

        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return False, "Bot not found"

        old_budget = bot.budget
        bot.budget = new_budget

        # Adjust current balance proportionally
        if old_budget > 0:
            ratio = new_budget / old_budget
            bot.current_balance = bot.current_balance * ratio

        bot.updated_at = datetime.utcnow()
        await self.session.commit()

        logger.info(f"Bot {bot_id}: Budget updated from ${old_budget:.2f} to ${new_budget:.2f}")
        return True, f"Budget updated to ${new_budget:.2f}"

    async def set_compound_mode(
        self,
        bot_id: int,
        enabled: bool,
    ) -> Tuple[bool, str]:
        """Set compound mode for a bot.

        Args:
            bot_id: The bot ID
            enabled: Whether to enable compound mode

        Returns:
            Tuple of (success, message)
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return False, "Bot not found"

        bot.compound_enabled = enabled
        bot.updated_at = datetime.utcnow()
        await self.session.commit()

        mode = "enabled" if enabled else "disabled"
        logger.info(f"Bot {bot_id}: Compound mode {mode}")
        return True, f"Compound mode {mode}"

    async def get_pnl_summary(self, bot_id: int) -> Optional[dict]:
        """Get P&L summary for a bot.

        Args:
            bot_id: The bot ID

        Returns:
            Dictionary with P&L summary or None
        """
        wallet = await self.get_wallet_status(bot_id)
        if not wallet:
            return None

        return {
            "budget": wallet.budget,
            "current_balance": wallet.current_balance,
            "total_pnl": wallet.total_pnl,
            "pnl_percent": (wallet.total_pnl / wallet.budget * 100) if wallet.budget > 0 else 0,
            "available_for_trade": wallet.available_for_trade,
            "compound_enabled": wallet.compound_enabled,
        }
