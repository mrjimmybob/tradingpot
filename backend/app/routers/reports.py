"""Reports router for P&L, tax, and fee reports."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import get_session, Bot, Order, OrderStatus, OrderType

router = APIRouter()


class PnLReportEntry(BaseModel):
    """Schema for P&L report entry."""
    bot_id: int
    bot_name: str
    trading_pair: str
    strategy: str
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    total_fees: float


class TaxReportEntry(BaseModel):
    """Schema for tax report entry (gains only)."""
    date: datetime
    trading_pair: str
    purchase_price: float
    sale_price: float
    gain: float
    token: str


class FeeReportEntry(BaseModel):
    """Schema for fee report entry."""
    bot_id: int
    bot_name: str
    total_fees: float
    order_count: int


class PnLReportResponse(BaseModel):
    """Schema for P&L report response."""
    entries: List[PnLReportEntry]
    total_pnl: float
    overall_win_rate: float


class TaxReportResponse(BaseModel):
    """Schema for tax report response."""
    entries: List[TaxReportEntry]
    total_gains: float
    year: int


class FeeReportResponse(BaseModel):
    """Schema for fee report response."""
    entries: List[FeeReportEntry]
    total_fees: float


@router.get("/pnl", response_model=PnLReportResponse)
async def get_pnl_report(
    session: AsyncSession = Depends(get_session),
    bot_id: Optional[int] = None,
    strategy: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Get P&L report with optional filters."""
    # Base query for bots
    query = select(Bot)

    if bot_id:
        query = query.where(Bot.id == bot_id)
    if strategy:
        query = query.where(Bot.strategy == strategy)

    result = await session.execute(query)
    bots = result.scalars().all()

    entries = []
    total_pnl = 0.0
    total_wins = 0
    total_losses = 0

    for bot in bots:
        # Build order query with date filters
        order_query = select(Order).where(Order.bot_id == bot.id)
        if start_date:
            order_query = order_query.where(Order.created_at >= start_date)
        if end_date:
            order_query = order_query.where(Order.created_at <= end_date)

        order_result = await session.execute(order_query)
        orders = order_result.scalars().all()

        # Calculate metrics
        fees = sum(o.fees for o in orders)
        # TODO: Calculate actual win/loss from matched trades
        win_count = 0
        loss_count = 0

        win_rate = 0.0
        if win_count + loss_count > 0:
            win_rate = win_count / (win_count + loss_count) * 100

        entries.append(PnLReportEntry(
            bot_id=bot.id,
            bot_name=bot.name,
            trading_pair=bot.trading_pair,
            strategy=bot.strategy,
            total_pnl=bot.total_pnl,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            total_fees=fees,
        ))

        total_pnl += bot.total_pnl
        total_wins += win_count
        total_losses += loss_count

    overall_win_rate = 0.0
    if total_wins + total_losses > 0:
        overall_win_rate = total_wins / (total_wins + total_losses) * 100

    return PnLReportResponse(
        entries=entries,
        total_pnl=total_pnl,
        overall_win_rate=overall_win_rate,
    )


@router.get("/tax", response_model=TaxReportResponse)
async def get_tax_report(
    session: AsyncSession = Depends(get_session),
    year: int = Query(default=datetime.now().year),
    bot_id: Optional[int] = None,
):
    """Get tax report (gains only) for a specific year."""
    # TODO: Implement proper tax gain calculation from matched buy/sell orders
    # For now, return empty report

    entries = []
    total_gains = 0.0

    return TaxReportResponse(
        entries=entries,
        total_gains=total_gains,
        year=year,
    )


@router.get("/fees", response_model=FeeReportResponse)
async def get_fee_report(
    session: AsyncSession = Depends(get_session),
    bot_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Get fee report with optional filters."""
    # Base query for bots
    query = select(Bot)
    if bot_id:
        query = query.where(Bot.id == bot_id)

    result = await session.execute(query)
    bots = result.scalars().all()

    entries = []
    total_fees = 0.0

    for bot in bots:
        # Build order query with date filters
        order_query = select(Order).where(Order.bot_id == bot.id)
        if start_date:
            order_query = order_query.where(Order.created_at >= start_date)
        if end_date:
            order_query = order_query.where(Order.created_at <= end_date)

        order_result = await session.execute(order_query)
        orders = order_result.scalars().all()

        fees = sum(o.fees for o in orders)
        order_count = len(orders)

        if order_count > 0 or fees > 0:
            entries.append(FeeReportEntry(
                bot_id=bot.id,
                bot_name=bot.name,
                total_fees=fees,
                order_count=order_count,
            ))

        total_fees += fees

    return FeeReportResponse(
        entries=entries,
        total_fees=total_fees,
    )
