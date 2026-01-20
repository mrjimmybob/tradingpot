"""Portfolio-level risk management service - cross-bot aggregation and enforcement.

IMPORTANT: Portfolio risk operates ACROSS all bots for the same owner.
This is the final safety boundary before trade execution.

Design constraints:
- All caps default to disabled (preserves current behavior)
- Loss caps = hard block (reject trade)
- Exposure caps = resize order to fit remaining capacity
- Must not change strategy logic
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models import Bot, BotStatus, Order, OrderStatus, Position, PortfolioRisk

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskCheck:
    """Result of portfolio risk check."""
    ok: bool  # True if trade can proceed
    violated_cap: Optional[str]  # Type of cap violated: "daily_loss", "weekly_loss", "drawdown", "exposure"
    details: dict  # Numeric details
    action: str  # "allow", "block", "resize"
    adjusted_amount: Optional[float]  # Adjusted order size if resized


class PortfolioRiskService:
    """Service for portfolio-level risk management across bots.

    Responsibilities:
    - Aggregate P&L across all bots for same owner
    - Aggregate open exposure across all bots
    - Enforce portfolio-level caps
    - Resize or block trades based on caps
    """

    def __init__(self, session: AsyncSession):
        """Initialize portfolio risk service.

        Args:
            session: Database session
        """
        self.session = session

    async def check_portfolio_risk(
        self,
        bot_id: int,
        order_amount_usd: float,
        order_side: str,  # "buy" or "sell"
    ) -> PortfolioRiskCheck:
        """Check if trade violates portfolio-level risk caps.

        This is called BEFORE trade execution to validate against portfolio caps.

        Trade lifecycle enforcement order:
        1. Strategy generates signal
        2. Auto_mode approves strategy
        3. **Portfolio risk caps checked** ← THIS METHOD
        4. Strategy capacity checked
        5. Execution cost estimated
        6. Per-bot risk checks
        7. Execute trade

        Args:
            bot_id: Bot attempting to trade
            order_amount_usd: Order size in USD
            order_side: "buy" or "sell"

        Returns:
            PortfolioRiskCheck with decision
        """
        # Get bot to find owner
        result = await self.session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return PortfolioRiskCheck(
                ok=False,
                violated_cap="bot_not_found",
                details={},
                action="block",
                adjusted_amount=None,
            )

        # TODO: Bot model needs owner_id field - for now use bot_id as owner_id
        owner_id = str(bot.id)  # FIXME: Replace with bot.owner_id when added

        # Get portfolio risk config
        result = await self.session.execute(
            select(PortfolioRisk).where(PortfolioRisk.owner_id == owner_id)
        )
        portfolio_risk = result.scalar_one_or_none()

        # If no config or disabled, allow trade
        if not portfolio_risk or not portfolio_risk.enabled:
            return PortfolioRiskCheck(
                ok=True,
                violated_cap=None,
                details={"portfolio_risk_enabled": False},
                action="allow",
                adjusted_amount=order_amount_usd,
            )

        # Get all bots for this owner
        result = await self.session.execute(
            select(Bot)
            # TODO: Add .where(Bot.owner_id == owner_id) when owner_id field exists
        )
        owner_bots = result.scalars().all()
        bot_ids = [b.id for b in owner_bots if b.id == bot_id]  # FIXME: Get all owner bots

        # Calculate portfolio metrics
        portfolio_balance = sum(b.current_balance for b in owner_bots if b.id in bot_ids)
        portfolio_initial = sum(b.budget for b in owner_bots if b.id in bot_ids)

        # === CHECK 1: Daily Loss Cap ===
        if portfolio_risk.daily_loss_cap_pct:
            daily_loss = await self._calculate_portfolio_period_loss(
                bot_ids,
                datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            )
            daily_loss_pct = (daily_loss / portfolio_initial * 100) if portfolio_initial > 0 else 0

            if daily_loss_pct >= portfolio_risk.daily_loss_cap_pct:
                logger.warning(
                    f"Portfolio daily loss cap exceeded for owner {owner_id}: "
                    f"{daily_loss_pct:.2f}% >= {portfolio_risk.daily_loss_cap_pct}%"
                )
                return PortfolioRiskCheck(
                    ok=False,
                    violated_cap="daily_loss",
                    details={
                        "daily_loss_usd": daily_loss,
                        "daily_loss_pct": daily_loss_pct,
                        "cap_pct": portfolio_risk.daily_loss_cap_pct,
                    },
                    action="block",
                    adjusted_amount=None,
                )

        # === CHECK 2: Weekly Loss Cap ===
        if portfolio_risk.weekly_loss_cap_pct:
            today = datetime.utcnow()
            week_start = today - timedelta(days=today.weekday())
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

            weekly_loss = await self._calculate_portfolio_period_loss(bot_ids, week_start)
            weekly_loss_pct = (weekly_loss / portfolio_initial * 100) if portfolio_initial > 0 else 0

            if weekly_loss_pct >= portfolio_risk.weekly_loss_cap_pct:
                logger.warning(
                    f"Portfolio weekly loss cap exceeded for owner {owner_id}: "
                    f"{weekly_loss_pct:.2f}% >= {portfolio_risk.weekly_loss_cap_pct}%"
                )
                return PortfolioRiskCheck(
                    ok=False,
                    violated_cap="weekly_loss",
                    details={
                        "weekly_loss_usd": weekly_loss,
                        "weekly_loss_pct": weekly_loss_pct,
                        "cap_pct": portfolio_risk.weekly_loss_cap_pct,
                    },
                    action="block",
                    adjusted_amount=None,
                )

        # === CHECK 3: Drawdown Cap ===
        if portfolio_risk.max_drawdown_pct:
            drawdown_usd = portfolio_initial - portfolio_balance
            drawdown_pct = (drawdown_usd / portfolio_initial * 100) if portfolio_initial > 0 else 0

            if drawdown_pct >= portfolio_risk.max_drawdown_pct:
                logger.warning(
                    f"Portfolio drawdown cap exceeded for owner {owner_id}: "
                    f"{drawdown_pct:.2f}% >= {portfolio_risk.max_drawdown_pct}%"
                )
                return PortfolioRiskCheck(
                    ok=False,
                    violated_cap="drawdown",
                    details={
                        "drawdown_usd": drawdown_usd,
                        "drawdown_pct": drawdown_pct,
                        "cap_pct": portfolio_risk.max_drawdown_pct,
                    },
                    action="block",
                    adjusted_amount=None,
                )

        # === CHECK 4: Exposure Cap ===
        if portfolio_risk.max_total_exposure_pct and order_side == "buy":
            # Calculate current open exposure
            result = await self.session.execute(
                select(Position).where(
                    and_(
                        Position.bot_id.in_(bot_ids),
                        Position.amount > 0
                    )
                )
            )
            positions = result.scalars().all()
            current_exposure = sum(p.amount * p.entry_price for p in positions)

            # Calculate exposure after this order
            new_exposure = current_exposure + order_amount_usd
            new_exposure_pct = (new_exposure / portfolio_balance * 100) if portfolio_balance > 0 else 0

            if new_exposure_pct > portfolio_risk.max_total_exposure_pct:
                # Calculate remaining capacity
                max_exposure_usd = portfolio_balance * (portfolio_risk.max_total_exposure_pct / 100)
                remaining_capacity = max_exposure_usd - current_exposure

                if remaining_capacity <= 0:
                    logger.warning(
                        f"Portfolio exposure cap exceeded for owner {owner_id}: "
                        f"{new_exposure_pct:.2f}% > {portfolio_risk.max_total_exposure_pct}%"
                    )
                    return PortfolioRiskCheck(
                        ok=False,
                        violated_cap="exposure",
                        details={
                            "current_exposure_usd": current_exposure,
                            "new_exposure_usd": new_exposure,
                            "new_exposure_pct": new_exposure_pct,
                            "cap_pct": portfolio_risk.max_total_exposure_pct,
                            "remaining_capacity": 0,
                        },
                        action="block",
                        adjusted_amount=None,
                    )
                else:
                    # Resize order to fit remaining capacity
                    adjusted_amount = min(order_amount_usd, remaining_capacity)
                    logger.info(
                        f"Portfolio exposure cap enforced for owner {owner_id}: "
                        f"Order resized from ${order_amount_usd:.2f} to ${adjusted_amount:.2f} "
                        f"(remaining capacity: ${remaining_capacity:.2f})"
                    )
                    return PortfolioRiskCheck(
                        ok=True,
                        violated_cap="exposure",  # Cap hit but allowed via resize
                        details={
                            "current_exposure_usd": current_exposure,
                            "requested_amount": order_amount_usd,
                            "adjusted_amount": adjusted_amount,
                            "remaining_capacity": remaining_capacity,
                            "cap_pct": portfolio_risk.max_total_exposure_pct,
                        },
                        action="resize",
                        adjusted_amount=adjusted_amount,
                    )

        # All checks passed
        return PortfolioRiskCheck(
            ok=True,
            violated_cap=None,
            details={
                "portfolio_balance": portfolio_balance,
                "portfolio_initial": portfolio_initial,
            },
            action="allow",
            adjusted_amount=order_amount_usd,
        )

    async def _calculate_portfolio_period_loss(
        self,
        bot_ids: list,
        start_time: datetime,
    ) -> float:
        """Calculate total loss across portfolio for a period.

        Args:
            bot_ids: List of bot IDs in portfolio
            start_time: Start of period

        Returns:
            Total loss amount (positive number)
        """
        if not bot_ids:
            return 0.0

        # Get all filled orders in period
        query = select(Order).where(
            and_(
                Order.bot_id.in_(bot_ids),
                Order.status == OrderStatus.FILLED,
                Order.created_at >= start_time,
            )
        )
        result = await self.session.execute(query)
        orders = result.scalars().all()

        # Calculate total loss (simplified - sum fees and modeled costs)
        total_loss = 0.0
        for order in orders:
            # Include exchange fees
            total_loss += order.fees

            # Include modeled execution costs if available
            if order.modeled_total_cost:
                total_loss += order.modeled_total_cost

        return total_loss

    async def get_portfolio_metrics(self, owner_id: str) -> dict:
        """Get current portfolio metrics for an owner.

        Args:
            owner_id: Owner identifier

        Returns:
            Dictionary with portfolio metrics
        """
        # Get all bots for this owner
        result = await self.session.execute(
            select(Bot)
            # TODO: Add .where(Bot.owner_id == owner_id) when owner_id field exists
        )
        owner_bots = result.scalars().all()

        if not owner_bots:
            return {
                "total_balance": 0.0,
                "total_initial": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "open_exposure": 0.0,
                "bot_count": 0,
            }

        bot_ids = [b.id for b in owner_bots]

        # Calculate aggregated metrics
        total_balance = sum(b.current_balance for b in owner_bots)
        total_initial = sum(b.budget for b in owner_bots)
        total_pnl = total_balance - total_initial
        total_pnl_pct = (total_pnl / total_initial * 100) if total_initial > 0 else 0

        # Calculate open exposure
        result = await self.session.execute(
            select(Position).where(
                and_(
                    Position.bot_id.in_(bot_ids),
                    Position.amount > 0
                )
            )
        )
        positions = result.scalars().all()
        open_exposure = sum(p.amount * p.entry_price for p in positions)

        return {
            "total_balance": total_balance,
            "total_initial": total_initial,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "open_exposure": open_exposure,
            "open_exposure_pct": (open_exposure / total_balance * 100) if total_balance > 0 else 0,
            "bot_count": len(owner_bots),
        }
