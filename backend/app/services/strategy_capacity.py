"""Strategy capacity tracking and enforcement service.

IMPORTANT: Strategy capacity limits prevent over-concentration in any single strategy.
This is enforced in BOTH auto_mode selection AND final execution.

Design constraints:
- All limits default to unlimited (preserves current behavior)
- Capacity limits are soft (reject trade, don't pause bot)
- Must not change strategy logic
- Enforced at two layers: auto_mode eligibility + execution safety
"""

import logging
from typing import Optional, Dict
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from ..models import Bot, BotStatus, Position

logger = logging.getLogger(__name__)


@dataclass
class StrategyCapacity:
    """Strategy capacity information."""
    strategy_name: str
    max_allocation_pct: Optional[float]  # Max % of portfolio, None = unlimited
    max_concurrent_bots: Optional[int]   # Max # of bots, None = unlimited
    current_allocation_usd: float        # Current allocated capital
    current_allocation_pct: float        # Current % of portfolio
    current_bot_count: int               # Current # of active bots
    remaining_capacity_pct: Optional[float]  # Remaining % capacity, None = unlimited
    remaining_bot_slots: Optional[int]   # Remaining bot slots, None = unlimited
    is_at_capacity: bool                 # True if any limit reached


@dataclass
class CapacityCheck:
    """Result of capacity check."""
    ok: bool  # True if trade can proceed
    reason: str  # Explanation
    adjusted_amount: Optional[float]  # Adjusted order size if resized


# Global strategy capacity configuration
# TODO: Move to database or config file for per-user customization
STRATEGY_CAPACITY_CONFIG = {
    "mean_reversion": {
        "max_portfolio_allocation_pct": None,  # Unlimited by default
        "max_concurrent_bots": None,
    },
    "trend_following": {
        "max_portfolio_allocation_pct": None,
        "max_concurrent_bots": None,
    },
    "volatility_breakout": {
        "max_portfolio_allocation_pct": None,
        "max_concurrent_bots": None,
    },
    "adaptive_grid": {
        "max_portfolio_allocation_pct": None,
        "max_concurrent_bots": None,
    },
    "dca_accumulator": {
        "max_portfolio_allocation_pct": None,  # DCA is fallback, never limited
        "max_concurrent_bots": None,
    },
}


class StrategyCapacityService:
    """Service for tracking and enforcing strategy capacity limits.

    Responsibilities:
    - Track per-strategy allocated capital
    - Track per-strategy active bot count
    - Enforce capacity limits in auto_mode eligibility
    - Enforce capacity limits at execution (last defense)
    """

    def __init__(self, session: AsyncSession):
        """Initialize strategy capacity service.

        Args:
            session: Database session
        """
        self.session = session

    async def get_strategy_capacity(
        self,
        strategy_name: str,
        owner_id: Optional[str] = None,
    ) -> StrategyCapacity:
        """Get current capacity information for a strategy.

        Args:
            strategy_name: Name of strategy
            owner_id: Owner identifier (None = all users)

        Returns:
            StrategyCapacity with current usage
        """
        # Get capacity config for this strategy
        config = STRATEGY_CAPACITY_CONFIG.get(strategy_name, {})
        max_allocation_pct = config.get("max_portfolio_allocation_pct")
        max_concurrent_bots = config.get("max_concurrent_bots")

        # Get all active bots using this strategy
        query = select(Bot).where(
            and_(
                Bot.strategy == strategy_name,
                Bot.status == BotStatus.RUNNING,
            )
        )

        # TODO: Add owner filter when owner_id field exists
        # if owner_id:
        #     query = query.where(Bot.owner_id == owner_id)

        result = await self.session.execute(query)
        strategy_bots = result.scalars().all()

        # Calculate current allocation
        current_allocation_usd = sum(b.current_balance for b in strategy_bots)
        current_bot_count = len(strategy_bots)

        # Calculate portfolio total (for percentage)
        all_bots_result = await self.session.execute(
            select(Bot).where(Bot.status == BotStatus.RUNNING)
        )
        all_bots = all_bots_result.scalars().all()
        portfolio_total = sum(b.current_balance for b in all_bots)

        current_allocation_pct = (
            (current_allocation_usd / portfolio_total * 100)
            if portfolio_total > 0
            else 0.0
        )

        # Calculate remaining capacity
        if max_allocation_pct is not None:
            remaining_capacity_pct = max_allocation_pct - current_allocation_pct
        else:
            remaining_capacity_pct = None  # Unlimited

        if max_concurrent_bots is not None:
            remaining_bot_slots = max_concurrent_bots - current_bot_count
        else:
            remaining_bot_slots = None  # Unlimited

        # Check if at capacity
        is_at_capacity = False
        if max_allocation_pct is not None and current_allocation_pct >= max_allocation_pct:
            is_at_capacity = True
        if max_concurrent_bots is not None and current_bot_count >= max_concurrent_bots:
            is_at_capacity = True

        return StrategyCapacity(
            strategy_name=strategy_name,
            max_allocation_pct=max_allocation_pct,
            max_concurrent_bots=max_concurrent_bots,
            current_allocation_usd=current_allocation_usd,
            current_allocation_pct=current_allocation_pct,
            current_bot_count=current_bot_count,
            remaining_capacity_pct=remaining_capacity_pct,
            remaining_bot_slots=remaining_bot_slots,
            is_at_capacity=is_at_capacity,
        )

    async def check_capacity_for_trade(
        self,
        bot_id: int,
        strategy_name: str,
        order_amount_usd: float,
    ) -> CapacityCheck:
        """Check if a trade would exceed strategy capacity limits.

        This is called at EXECUTION time (last defense layer).

        Args:
            bot_id: Bot attempting to trade
            strategy_name: Strategy being used
            order_amount_usd: Order size in USD

        Returns:
            CapacityCheck with decision
        """
        capacity = await self.get_strategy_capacity(strategy_name)

        # Check allocation limit
        if capacity.max_allocation_pct is not None:
            # Calculate what allocation would be after this trade
            new_allocation_usd = capacity.current_allocation_usd + order_amount_usd

            # Get portfolio total
            all_bots_result = await self.session.execute(
                select(Bot).where(Bot.status == BotStatus.RUNNING)
            )
            all_bots = all_bots_result.scalars().all()
            portfolio_total = sum(b.current_balance for b in all_bots)

            new_allocation_pct = (
                (new_allocation_usd / portfolio_total * 100)
                if portfolio_total > 0
                else 0.0
            )

            if new_allocation_pct > capacity.max_allocation_pct:
                # Calculate remaining capacity
                max_allocation_usd = portfolio_total * (capacity.max_allocation_pct / 100)
                remaining_capacity_usd = max_allocation_usd - capacity.current_allocation_usd

                if remaining_capacity_usd <= 0:
                    logger.warning(
                        f"Strategy capacity exceeded for {strategy_name}: "
                        f"{new_allocation_pct:.2f}% > {capacity.max_allocation_pct}%"
                    )
                    return CapacityCheck(
                        ok=False,
                        reason=(
                            f"Strategy capacity exceeded: {strategy_name} allocation "
                            f"{new_allocation_pct:.2f}% > {capacity.max_allocation_pct}%"
                        ),
                        adjusted_amount=None,
                    )
                else:
                    # Resize order to fit remaining capacity
                    adjusted_amount = min(order_amount_usd, remaining_capacity_usd)
                    logger.info(
                        f"Strategy capacity enforced for {strategy_name}: "
                        f"Order resized from ${order_amount_usd:.2f} to ${adjusted_amount:.2f} "
                        f"(remaining capacity: ${remaining_capacity_usd:.2f})"
                    )
                    return CapacityCheck(
                        ok=True,
                        reason=(
                            f"Order resized due to strategy capacity: "
                            f"${order_amount_usd:.2f} → ${adjusted_amount:.2f}"
                        ),
                        adjusted_amount=adjusted_amount,
                    )

        # Check bot count limit (only when bot is not already using this strategy)
        # This check is more relevant for auto_mode switching, not per-trade
        # So we don't enforce it here at execution time

        # All checks passed
        return CapacityCheck(
            ok=True,
            reason="Within strategy capacity limits",
            adjusted_amount=order_amount_usd,
        )

    async def is_strategy_at_capacity(
        self,
        strategy_name: str,
        owner_id: Optional[str] = None,
    ) -> tuple:
        """Check if strategy is at capacity (for auto_mode eligibility).

        Args:
            strategy_name: Name of strategy
            owner_id: Owner identifier

        Returns:
            (is_at_capacity: bool, reason: str)
        """
        capacity = await self.get_strategy_capacity(strategy_name, owner_id)

        if capacity.is_at_capacity:
            reasons = []

            if (
                capacity.max_allocation_pct is not None
                and capacity.current_allocation_pct >= capacity.max_allocation_pct
            ):
                reasons.append(
                    f"allocation {capacity.current_allocation_pct:.1f}% >= "
                    f"{capacity.max_allocation_pct}%"
                )

            if (
                capacity.max_concurrent_bots is not None
                and capacity.current_bot_count >= capacity.max_concurrent_bots
            ):
                reasons.append(
                    f"bot count {capacity.current_bot_count} >= "
                    f"{capacity.max_concurrent_bots}"
                )

            return True, ", ".join(reasons)

        return False, "not at capacity"

    def update_capacity_config(
        self,
        strategy_name: str,
        max_allocation_pct: Optional[float] = None,
        max_concurrent_bots: Optional[int] = None,
    ) -> None:
        """Update capacity configuration for a strategy.

        TODO: This should be moved to database for persistence.

        Args:
            strategy_name: Name of strategy
            max_allocation_pct: Max % of portfolio (None = unlimited)
            max_concurrent_bots: Max # of bots (None = unlimited)
        """
        if strategy_name not in STRATEGY_CAPACITY_CONFIG:
            STRATEGY_CAPACITY_CONFIG[strategy_name] = {}

        if max_allocation_pct is not None:
            STRATEGY_CAPACITY_CONFIG[strategy_name]["max_portfolio_allocation_pct"] = max_allocation_pct

        if max_concurrent_bots is not None:
            STRATEGY_CAPACITY_CONFIG[strategy_name]["max_concurrent_bots"] = max_concurrent_bots

        logger.info(
            f"Updated capacity config for {strategy_name}: "
            f"max_allocation={max_allocation_pct}%, max_bots={max_concurrent_bots}"
        )
