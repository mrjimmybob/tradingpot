"""Risk management service for stop losses, drawdowns, and loss limits."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models import Bot, BotStatus, Order, OrderStatus, Alert, StrategyRotation

logger = logging.getLogger(__name__)


class RiskAction(str, Enum):
    """Actions to take based on risk assessment."""
    CONTINUE = "continue"
    CLOSE_POSITION = "close_position"
    ROTATE_STRATEGY = "rotate_strategy"
    PAUSE_BOT = "pause_bot"
    STOP_BOT = "stop_bot"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    action: RiskAction
    reason: str
    details: dict


@dataclass
class PositionRisk:
    """Risk assessment for a single position."""
    should_close: bool
    reason: str
    unrealized_pnl: float
    pnl_percent: float


class RiskManagementService:
    """Service for managing trading risk."""

    def __init__(self, session: AsyncSession):
        """Initialize risk management service.

        Args:
            session: Database session
        """
        self.session = session

    async def check_stop_loss(
        self,
        bot_id: int,
        entry_price: float,
        current_price: float,
        position_amount: float,
        is_long: bool = True,
    ) -> PositionRisk:
        """Check if stop loss should be triggered for a position.

        Args:
            bot_id: The bot ID
            entry_price: Position entry price
            current_price: Current market price
            position_amount: Position size
            is_long: Whether it's a long position

        Returns:
            PositionRisk assessment
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return PositionRisk(
                should_close=False,
                reason="Bot not found",
                unrealized_pnl=0,
                pnl_percent=0,
            )

        # Calculate unrealized P&L
        if is_long:
            unrealized_pnl = (current_price - entry_price) * position_amount
        else:
            unrealized_pnl = (entry_price - current_price) * position_amount

        pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        if not is_long:
            pnl_percent = -pnl_percent

        # Check percentage-based stop loss
        if bot.stop_loss_percent and pnl_percent <= -bot.stop_loss_percent:
            return PositionRisk(
                should_close=True,
                reason=f"Stop loss triggered: {pnl_percent:.2f}% loss exceeds {bot.stop_loss_percent}% limit",
                unrealized_pnl=unrealized_pnl,
                pnl_percent=pnl_percent,
            )

        # Check absolute stop loss
        if bot.stop_loss_absolute and unrealized_pnl <= -bot.stop_loss_absolute:
            return PositionRisk(
                should_close=True,
                reason=f"Stop loss triggered: ${abs(unrealized_pnl):.2f} loss exceeds ${bot.stop_loss_absolute:.2f} limit",
                unrealized_pnl=unrealized_pnl,
                pnl_percent=pnl_percent,
            )

        return PositionRisk(
            should_close=False,
            reason="Within risk limits",
            unrealized_pnl=unrealized_pnl,
            pnl_percent=pnl_percent,
        )

    async def check_drawdown(self, bot_id: int) -> RiskAssessment:
        """Check if bot has exceeded drawdown limits.

        Args:
            bot_id: The bot ID

        Returns:
            RiskAssessment with recommended action
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="Bot not found",
                details={},
            )

        # Calculate drawdown
        drawdown = bot.budget - bot.current_balance
        drawdown_percent = (drawdown / bot.budget * 100) if bot.budget > 0 else 0

        # Check percentage-based drawdown limit
        if bot.drawdown_limit_percent and drawdown_percent >= bot.drawdown_limit_percent:
            await self._log_alert(
                bot_id,
                "drawdown_limit",
                f"Drawdown limit reached: {drawdown_percent:.2f}% exceeds {bot.drawdown_limit_percent}% limit"
            )
            return RiskAssessment(
                action=RiskAction.PAUSE_BOT,
                reason=f"Drawdown limit reached: {drawdown_percent:.2f}%",
                details={
                    "drawdown": drawdown,
                    "drawdown_percent": drawdown_percent,
                    "limit_percent": bot.drawdown_limit_percent,
                },
            )

        # Check absolute drawdown limit
        if bot.drawdown_limit_absolute and drawdown >= bot.drawdown_limit_absolute:
            await self._log_alert(
                bot_id,
                "drawdown_limit",
                f"Drawdown limit reached: ${drawdown:.2f} exceeds ${bot.drawdown_limit_absolute:.2f} limit"
            )
            return RiskAssessment(
                action=RiskAction.PAUSE_BOT,
                reason=f"Drawdown limit reached: ${drawdown:.2f}",
                details={
                    "drawdown": drawdown,
                    "drawdown_percent": drawdown_percent,
                    "limit_absolute": bot.drawdown_limit_absolute,
                },
            )

        return RiskAssessment(
            action=RiskAction.CONTINUE,
            reason="Within drawdown limits",
            details={
                "drawdown": drawdown,
                "drawdown_percent": drawdown_percent,
            },
        )

    async def check_daily_loss_limit(self, bot_id: int) -> RiskAssessment:
        """Check if daily loss limit has been reached.

        Args:
            bot_id: The bot ID

        Returns:
            RiskAssessment with recommended action
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot or not bot.daily_loss_limit:
            return RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="No daily loss limit set",
                details={},
            )

        # Calculate today's losses from orders
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        daily_loss = await self._calculate_period_loss(bot_id, today_start)

        if daily_loss >= bot.daily_loss_limit:
            await self._log_alert(
                bot_id,
                "daily_loss_limit",
                f"Daily loss limit reached: ${daily_loss:.2f} exceeds ${bot.daily_loss_limit:.2f} limit"
            )
            return RiskAssessment(
                action=RiskAction.PAUSE_BOT,
                reason=f"Daily loss limit reached: ${daily_loss:.2f}",
                details={
                    "daily_loss": daily_loss,
                    "daily_limit": bot.daily_loss_limit,
                },
            )

        return RiskAssessment(
            action=RiskAction.CONTINUE,
            reason="Within daily loss limit",
            details={
                "daily_loss": daily_loss,
                "daily_limit": bot.daily_loss_limit,
            },
        )

    async def check_weekly_loss_limit(self, bot_id: int) -> RiskAssessment:
        """Check if weekly loss limit has been reached.

        Args:
            bot_id: The bot ID

        Returns:
            RiskAssessment with recommended action
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot or not bot.weekly_loss_limit:
            return RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="No weekly loss limit set",
                details={},
            )

        # Calculate this week's losses (Monday = 0)
        today = datetime.utcnow()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        weekly_loss = await self._calculate_period_loss(bot_id, week_start)

        if weekly_loss >= bot.weekly_loss_limit:
            await self._log_alert(
                bot_id,
                "weekly_loss_limit",
                f"Weekly loss limit reached: ${weekly_loss:.2f} exceeds ${bot.weekly_loss_limit:.2f} limit"
            )
            return RiskAssessment(
                action=RiskAction.PAUSE_BOT,
                reason=f"Weekly loss limit reached: ${weekly_loss:.2f}",
                details={
                    "weekly_loss": weekly_loss,
                    "weekly_limit": bot.weekly_loss_limit,
                },
            )

        return RiskAssessment(
            action=RiskAction.CONTINUE,
            reason="Within weekly loss limit",
            details={
                "weekly_loss": weekly_loss,
                "weekly_limit": bot.weekly_loss_limit,
            },
        )

    async def check_consecutive_losses(
        self,
        bot_id: int,
        threshold: int = 3,
    ) -> Tuple[int, RiskAssessment]:
        """Check for consecutive losses and determine action.

        Args:
            bot_id: The bot ID
            threshold: Number of consecutive losses to trigger action

        Returns:
            Tuple of (consecutive_loss_count, RiskAssessment)
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return 0, RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="Bot not found",
                details={},
            )

        # Get recent orders to count consecutive losses
        consecutive_losses = await self._count_consecutive_losses(bot_id)

        if consecutive_losses >= threshold:
            # Check if we can rotate strategy
            rotation_count = await self._count_strategy_rotations(bot_id)

            if rotation_count >= bot.max_strategy_rotations:
                await self._log_alert(
                    bot_id,
                    "max_rotations",
                    f"Max strategy rotations ({bot.max_strategy_rotations}) reached after consecutive losses"
                )
                return consecutive_losses, RiskAssessment(
                    action=RiskAction.PAUSE_BOT,
                    reason=f"Max strategy rotations reached ({rotation_count})",
                    details={
                        "consecutive_losses": consecutive_losses,
                        "rotation_count": rotation_count,
                        "max_rotations": bot.max_strategy_rotations,
                    },
                )
            else:
                return consecutive_losses, RiskAssessment(
                    action=RiskAction.ROTATE_STRATEGY,
                    reason=f"{consecutive_losses} consecutive losses - strategy rotation recommended",
                    details={
                        "consecutive_losses": consecutive_losses,
                        "rotation_count": rotation_count,
                        "max_rotations": bot.max_strategy_rotations,
                    },
                )

        return consecutive_losses, RiskAssessment(
            action=RiskAction.CONTINUE,
            reason="Within consecutive loss threshold",
            details={
                "consecutive_losses": consecutive_losses,
                "threshold": threshold,
            },
        )

    async def rotate_strategy(
        self,
        bot_id: int,
        new_strategy: str,
        reason: str = "Consecutive losses",
    ) -> Tuple[bool, str]:
        """Rotate bot to a different strategy.

        IMPORTANT: This method NEVER modifies Bot.strategy for auto_mode bots.
        Auto_mode is a meta-strategy (policy engine) that governs itself.
        Risk events can only pause trading or reduce activity, but cannot
        override auto_mode's strategy selection policy.

        Args:
            bot_id: The bot ID
            new_strategy: The new strategy name
            reason: Reason for rotation

        Returns:
            Tuple of (success, message)
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return False, "Bot not found"

        old_strategy = bot.strategy

        # CRITICAL: Do NOT rotate auto_mode bots
        # Auto_mode is a policy engine that manages its own strategy selection
        if old_strategy == "auto_mode":
            logger.warning(
                f"Bot {bot_id}: Risk-based strategy rotation blocked - "
                f"auto_mode cannot be disabled by risk events. "
                f"Reason: {reason}. "
                f"Consider pausing bot or adjusting auto_mode parameters instead."
            )
            await self._log_alert(
                bot_id,
                "rotation_blocked",
                f"Strategy rotation blocked for auto_mode bot. Reason: {reason}"
            )
            return False, "Cannot rotate auto_mode - it is a policy engine that governs itself"

        # Record rotation
        rotation = StrategyRotation(
            bot_id=bot_id,
            from_strategy=old_strategy,
            to_strategy=new_strategy,
            reason=reason,
        )
        self.session.add(rotation)

        # Update bot strategy (only for non-auto_mode bots)
        bot.strategy = new_strategy
        bot.updated_at = datetime.utcnow()

        await self.session.commit()

        logger.info(f"Bot {bot_id}: Rotated strategy from {old_strategy} to {new_strategy}")
        return True, f"Strategy rotated from {old_strategy} to {new_strategy}"

    async def check_running_time(self, bot_id: int) -> RiskAssessment:
        """Check if bot has exceeded its running time limit.

        Args:
            bot_id: The bot ID

        Returns:
            RiskAssessment with recommended action
        """
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="Bot not found",
                details={},
            )

        # No time limit = run forever
        if not bot.running_time_hours:
            return RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="No running time limit (runs forever)",
                details={"running_time_hours": None},
            )

        # Check if started
        if not bot.started_at:
            return RiskAssessment(
                action=RiskAction.CONTINUE,
                reason="Bot not started yet",
                details={},
            )

        # Calculate running time
        running_time = datetime.utcnow() - bot.started_at
        running_hours = running_time.total_seconds() / 3600

        if running_hours >= bot.running_time_hours:
            return RiskAssessment(
                action=RiskAction.STOP_BOT,
                reason=f"Running time limit reached: {running_hours:.2f} hours",
                details={
                    "running_hours": running_hours,
                    "limit_hours": bot.running_time_hours,
                },
            )

        return RiskAssessment(
            action=RiskAction.CONTINUE,
            reason="Within running time limit",
            details={
                "running_hours": running_hours,
                "limit_hours": bot.running_time_hours,
                "remaining_hours": bot.running_time_hours - running_hours,
            },
        )

    async def full_risk_check(self, bot_id: int) -> RiskAssessment:
        """Perform full risk assessment for a bot.

        Args:
            bot_id: The bot ID

        Returns:
            RiskAssessment with the most severe action needed
        """
        # Check running time first
        time_check = await self.check_running_time(bot_id)
        if time_check.action != RiskAction.CONTINUE:
            return time_check

        # Check drawdown
        drawdown_check = await self.check_drawdown(bot_id)
        if drawdown_check.action != RiskAction.CONTINUE:
            return drawdown_check

        # Check daily loss limit
        daily_check = await self.check_daily_loss_limit(bot_id)
        if daily_check.action != RiskAction.CONTINUE:
            return daily_check

        # Check weekly loss limit
        weekly_check = await self.check_weekly_loss_limit(bot_id)
        if weekly_check.action != RiskAction.CONTINUE:
            return weekly_check

        # Check consecutive losses
        _, consecutive_check = await self.check_consecutive_losses(bot_id)
        if consecutive_check.action != RiskAction.CONTINUE:
            return consecutive_check

        return RiskAssessment(
            action=RiskAction.CONTINUE,
            reason="All risk checks passed",
            details={},
        )

    async def _calculate_period_loss(
        self,
        bot_id: int,
        start_time: datetime,
    ) -> float:
        """Calculate total loss for a period WITH EXECUTION COSTS.

        IMPORTANT: This now includes modeled execution costs in loss calculation.
        This provides more accurate risk assessment by including all trading costs.

        Args:
            bot_id: The bot ID
            start_time: Start of the period

        Returns:
            Total loss amount (positive number)
        """
        # Get filled orders to calculate realized losses
        query = select(Order).where(
            and_(
                Order.bot_id == bot_id,
                Order.status == OrderStatus.FILLED,
                Order.created_at >= start_time,
            )
        )
        result = await self.session.execute(query)
        orders = result.scalars().all()

        # Calculate total P&L from orders (cost-aware)
        total_loss = 0.0
        for order in orders:
            # Include exchange fees
            total_loss += order.fees

            # NEW: Include modeled execution costs
            if order.modeled_total_cost is not None:
                total_loss += order.modeled_total_cost

        return total_loss

    async def _count_consecutive_losses(self, bot_id: int) -> int:
        """Count consecutive losing trades.

        Args:
            bot_id: The bot ID

        Returns:
            Number of consecutive losses
        """
        # Get recent filled orders ordered by time
        query = select(Order).where(
            and_(
                Order.bot_id == bot_id,
                Order.status == OrderStatus.FILLED,
            )
        ).order_by(Order.filled_at.desc()).limit(20)

        result = await self.session.execute(query)
        orders = result.scalars().all()

        # Count consecutive losses
        # For simplicity, we check if running_balance decreased
        consecutive_losses = 0
        prev_balance = None

        for order in orders:
            if order.running_balance_after is None:
                continue

            if prev_balance is not None:
                if order.running_balance_after > prev_balance:
                    # This was a loss (balance decreased after this trade)
                    consecutive_losses += 1
                else:
                    # Win - reset counter
                    break

            prev_balance = order.running_balance_after

        return consecutive_losses

    async def _count_strategy_rotations(self, bot_id: int) -> int:
        """Count strategy rotations for a bot.

        Args:
            bot_id: The bot ID

        Returns:
            Number of rotations
        """
        query = select(func.count(StrategyRotation.id)).where(
            StrategyRotation.bot_id == bot_id
        )
        result = await self.session.execute(query)
        return result.scalar() or 0

    async def _log_alert(
        self,
        bot_id: Optional[int],
        alert_type: str,
        message: str,
    ) -> None:
        """Log an alert to the database.

        Args:
            bot_id: The bot ID (or None for global alerts)
            alert_type: Type of alert
            message: Alert message
        """
        alert = Alert(
            bot_id=bot_id,
            alert_type=alert_type,
            message=message,
            email_sent=False,
        )
        self.session.add(alert)
        await self.session.commit()

        logger.warning(f"[ALERT] {alert_type}: {message}")
