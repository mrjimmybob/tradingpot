"""Statistics and global operations router."""

from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_session, Bot, BotStatus, PnLSnapshot, Alert, Order, OrderStatus

router = APIRouter()


class GlobalStatsResponse(BaseModel):
    """Schema for global statistics response."""
    total_bots: int
    running_bots: int
    paused_bots: int
    stopped_bots: int
    total_pnl: float
    active_trades: int
    dry_run_bots: int


class PnLDataPoint(BaseModel):
    """Schema for P&L chart data point."""
    timestamp: datetime
    pnl: float


class KillAllResponse(BaseModel):
    """Schema for kill-all response."""
    killed_bots: int
    message: str


@router.get("/stats", response_model=GlobalStatsResponse)
async def get_global_stats(
    session: AsyncSession = Depends(get_session),
):
    """Get global statistics across all bots."""
    # Total bots
    total_result = await session.execute(select(func.count(Bot.id)))
    total_bots = total_result.scalar() or 0

    # Running bots
    running_result = await session.execute(
        select(func.count(Bot.id)).where(Bot.status == BotStatus.RUNNING)
    )
    running_bots = running_result.scalar() or 0

    # Paused bots
    paused_result = await session.execute(
        select(func.count(Bot.id)).where(Bot.status == BotStatus.PAUSED)
    )
    paused_bots = paused_result.scalar() or 0

    # Stopped bots
    stopped_result = await session.execute(
        select(func.count(Bot.id)).where(Bot.status == BotStatus.STOPPED)
    )
    stopped_bots = stopped_result.scalar() or 0

    # Total P&L
    pnl_result = await session.execute(select(func.sum(Bot.total_pnl)))
    total_pnl = pnl_result.scalar() or 0.0

    # Dry run bots
    dry_run_result = await session.execute(
        select(func.count(Bot.id)).where(Bot.is_dry_run == True)
    )
    dry_run_bots = dry_run_result.scalar() or 0

    # TODO: Active trades count from positions table
    active_trades = 0

    return GlobalStatsResponse(
        total_bots=total_bots,
        running_bots=running_bots,
        paused_bots=paused_bots,
        stopped_bots=stopped_bots,
        total_pnl=total_pnl,
        active_trades=active_trades,
        dry_run_bots=dry_run_bots,
    )


@router.get("/pnl", response_model=List[PnLDataPoint])
async def get_pnl_history(
    session: AsyncSession = Depends(get_session),
    limit: int = 100,
):
    """Get global P&L history for chart."""
    result = await session.execute(
        select(PnLSnapshot)
        .where(PnLSnapshot.bot_id == None)  # Global snapshots
        .order_by(PnLSnapshot.snapshot_at.desc())
        .limit(limit)
    )
    snapshots = result.scalars().all()

    return [
        PnLDataPoint(timestamp=s.snapshot_at, pnl=s.total_pnl)
        for s in reversed(snapshots)
    ]


@router.post("/kill-all", response_model=KillAllResponse)
async def kill_all_bots(
    session: AsyncSession = Depends(get_session),
):
    """Global kill switch - stops all running bots and cancels all pending orders."""
    result = await session.execute(
        select(Bot).where(Bot.status == BotStatus.RUNNING)
    )
    running_bots = result.scalars().all()

    killed_count = 0
    cancelled_orders_count = 0

    for bot in running_bots:
        bot.status = BotStatus.STOPPED
        bot.updated_at = datetime.utcnow()
        killed_count += 1

        # Cancel all pending orders for this bot
        orders_result = await session.execute(
            select(Order).where(
                Order.bot_id == bot.id,
                Order.status == OrderStatus.PENDING
            )
        )
        pending_orders = orders_result.scalars().all()
        for order in pending_orders:
            order.status = OrderStatus.CANCELLED
            cancelled_orders_count += 1

        # Log alert for this bot
        alert = Alert(
            bot_id=bot.id,
            alert_type="kill_switch",
            message=f"Global kill switch activated for bot '{bot.name}' (ID: {bot.id})",
            email_sent=False  # Email sending would be handled separately
        )
        session.add(alert)

    # Log a global alert
    if killed_count > 0:
        global_alert = Alert(
            bot_id=None,  # Global alert
            alert_type="global_kill_switch",
            message=f"Global kill switch activated. Stopped {killed_count} bots and cancelled {cancelled_orders_count} pending orders.",
            email_sent=False
        )
        session.add(global_alert)

        # Log to console (email would go here in production)
        print(f"[ALERT] Global kill switch activated. Stopped {killed_count} bots.")

    await session.commit()

    return KillAllResponse(
        killed_bots=killed_count,
        message=f"Successfully killed {killed_count} bots and cancelled {cancelled_orders_count} pending orders"
    )
