"""Ledger and accounting API endpoints.

These endpoints provide read-only access to the accounting-grade ledger system.
All data comes from the authoritative SQLite database.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from ..models import (
    get_session,
    WalletLedger, LedgerReason,
    Trade, TradeSide,
    TaxLot,
    RealizedGain,
    Bot,
)
from ..services.ledger_writer import LedgerWriterService
from ..services.accounting import CSVExportService

router = APIRouter(prefix="/ledger", tags=["ledger"])


# =====================================================================
# WALLET LEDGER ENDPOINTS
# =====================================================================

@router.get("/entries")
async def get_ledger_entries(
    owner_id: Optional[str] = None,
    bot_id: Optional[int] = None,
    asset: Optional[str] = None,
    reason: Optional[LedgerReason] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """Get wallet ledger entries with filters.

    Args:
        owner_id: Filter by owner
        bot_id: Filter by bot
        asset: Filter by asset
        reason: Filter by transaction reason
        limit: Maximum entries to return
        offset: Pagination offset

    Returns:
        List of ledger entries
    """
    query = select(WalletLedger)

    # Apply filters
    if owner_id:
        query = query.where(WalletLedger.owner_id == owner_id)
    if bot_id:
        query = query.where(WalletLedger.bot_id == bot_id)
    if asset:
        query = query.where(WalletLedger.asset == asset)
    if reason:
        query = query.where(WalletLedger.reason == reason)

    # Order by most recent first
    query = query.order_by(desc(WalletLedger.created_at))
    query = query.limit(limit).offset(offset)

    result = await session.execute(query)
    entries = result.scalars().all()

    return [entry.to_dict() for entry in entries]


@router.get("/balance/{owner_id}/{asset}")
async def get_balance(
    owner_id: str,
    asset: str,
    bot_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get current balance for an asset.

    Args:
        owner_id: Owner identifier
        asset: Asset symbol
        bot_id: Bot ID (optional, None = owner-level)

    Returns:
        Current balance
    """
    ledger_writer = LedgerWriterService(session)
    balance = await ledger_writer.get_balance(owner_id, asset, bot_id)

    return {
        "owner_id": owner_id,
        "bot_id": bot_id,
        "asset": asset,
        "balance": balance,
    }


@router.get("/reconstruct/{owner_id}/{asset}")
async def reconstruct_balance(
    owner_id: str,
    asset: str,
    bot_id: Optional[int] = None,
    up_to_entry_id: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Reconstruct balance from ledger entries (validation).

    This endpoint verifies that balance_after values are correct by
    recalculating balance from all ledger entries.

    Args:
        owner_id: Owner identifier
        asset: Asset symbol
        bot_id: Bot ID (optional)
        up_to_entry_id: Stop at this entry (optional)

    Returns:
        Reconstructed balance and validation result
    """
    ledger_writer = LedgerWriterService(session)

    # Get current balance from latest entry
    current_balance = await ledger_writer.get_balance(owner_id, asset, bot_id)

    # Reconstruct balance from entries
    reconstructed_balance = await ledger_writer.reconstruct_balance(
        owner_id, asset, bot_id, up_to_entry_id
    )

    # Check if they match
    tolerance = 1e-8  # Floating point tolerance
    is_valid = abs(current_balance - reconstructed_balance) < tolerance

    return {
        "owner_id": owner_id,
        "bot_id": bot_id,
        "asset": asset,
        "current_balance": current_balance,
        "reconstructed_balance": reconstructed_balance,
        "is_valid": is_valid,
        "difference": current_balance - reconstructed_balance,
    }


# =====================================================================
# TRADE ENDPOINTS
# =====================================================================

@router.get("/trades")
async def get_trades(
    bot_id: Optional[int] = None,
    owner_id: Optional[str] = None,
    side: Optional[TradeSide] = None,
    asset: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """Get trade execution records.

    Args:
        bot_id: Filter by bot
        owner_id: Filter by owner
        side: Filter by side (buy/sell)
        asset: Filter by base asset
        limit: Maximum trades to return
        offset: Pagination offset

    Returns:
        List of trades
    """
    query = select(Trade)

    if bot_id:
        query = query.where(Trade.bot_id == bot_id)
    if owner_id:
        query = query.where(Trade.owner_id == owner_id)
    if side:
        query = query.where(Trade.side == side)
    if asset:
        query = query.where(Trade.base_asset == asset)

    query = query.order_by(desc(Trade.executed_at))
    query = query.limit(limit).offset(offset)

    result = await session.execute(query)
    trades = result.scalars().all()

    return [trade.to_dict() for trade in trades]


@router.get("/trades/{trade_id}")
async def get_trade(
    trade_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a specific trade by ID.

    Args:
        trade_id: Trade ID

    Returns:
        Trade details
    """
    result = await session.execute(
        select(Trade).where(Trade.id == trade_id)
    )
    trade = result.scalar_one_or_none()

    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return trade.to_dict()


# =====================================================================
# TAX LOT ENDPOINTS
# =====================================================================

@router.get("/tax-lots")
async def get_tax_lots(
    owner_id: Optional[str] = None,
    asset: Optional[str] = None,
    include_consumed: bool = False,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """Get tax lots (FIFO cost basis).

    Args:
        owner_id: Filter by owner
        asset: Filter by asset
        include_consumed: Include fully consumed lots
        limit: Maximum lots to return
        offset: Pagination offset

    Returns:
        List of tax lots
    """
    query = select(TaxLot)

    if owner_id:
        query = query.where(TaxLot.owner_id == owner_id)
    if asset:
        query = query.where(TaxLot.asset == asset)
    if not include_consumed:
        query = query.where(TaxLot.is_fully_consumed == False)

    query = query.order_by(TaxLot.purchase_date)
    query = query.limit(limit).offset(offset)

    result = await session.execute(query)
    lots = result.scalars().all()

    return [lot.to_dict() for lot in lots]


@router.get("/tax-lots/summary/{owner_id}")
async def get_tax_lot_summary(
    owner_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get summary of tax lots by asset.

    Args:
        owner_id: Owner identifier

    Returns:
        Summary by asset
    """
    result = await session.execute(
        select(
            TaxLot.asset,
            func.sum(TaxLot.quantity_remaining).label('total_quantity'),
            func.sum(TaxLot.quantity_remaining * TaxLot.unit_cost).label('total_cost'),
            func.avg(TaxLot.unit_cost).label('avg_cost'),
            func.count(TaxLot.id).label('lot_count'),
        )
        .where(
            and_(
                TaxLot.owner_id == owner_id,
                TaxLot.quantity_remaining > 0,
            )
        )
        .group_by(TaxLot.asset)
    )

    summary = []
    for row in result:
        summary.append({
            "asset": row.asset,
            "total_quantity": row.total_quantity,
            "total_cost": row.total_cost,
            "avg_cost_per_unit": row.avg_cost,
            "lot_count": row.lot_count,
        })

    return summary


# =====================================================================
# REALIZED GAINS ENDPOINTS
# =====================================================================

@router.get("/realized-gains")
async def get_realized_gains(
    owner_id: Optional[str] = None,
    asset: Optional[str] = None,
    year: Optional[int] = None,
    is_long_term: Optional[bool] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """Get realized gain/loss records.

    Args:
        owner_id: Filter by owner
        asset: Filter by asset
        year: Filter by tax year
        is_long_term: Filter by long-term/short-term
        limit: Maximum records to return
        offset: Pagination offset

    Returns:
        List of realized gains
    """
    query = select(RealizedGain)

    if owner_id:
        query = query.where(RealizedGain.owner_id == owner_id)
    if asset:
        query = query.where(RealizedGain.asset == asset)
    if year:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        query = query.where(
            and_(
                RealizedGain.sell_date >= start_date,
                RealizedGain.sell_date < end_date,
            )
        )
    if is_long_term is not None:
        query = query.where(RealizedGain.is_long_term == is_long_term)

    query = query.order_by(desc(RealizedGain.sell_date))
    query = query.limit(limit).offset(offset)

    result = await session.execute(query)
    gains = result.scalars().all()

    return [gain.to_dict() for gain in gains]


@router.get("/realized-gains/summary/{owner_id}")
async def get_realized_gains_summary(
    owner_id: str,
    year: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """Get summary of realized gains/losses.

    Args:
        owner_id: Owner identifier
        year: Tax year (optional)

    Returns:
        Summary by term (short/long)
    """
    query = select(
        RealizedGain.is_long_term,
        func.sum(RealizedGain.gain_loss).label('total_gain_loss'),
        func.sum(RealizedGain.proceeds).label('total_proceeds'),
        func.sum(RealizedGain.cost_basis).label('total_cost'),
        func.count(RealizedGain.id).label('trade_count'),
    ).where(RealizedGain.owner_id == owner_id)

    if year:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        query = query.where(
            and_(
                RealizedGain.sell_date >= start_date,
                RealizedGain.sell_date < end_date,
            )
        )

    query = query.group_by(RealizedGain.is_long_term)

    result = await session.execute(query)

    summary = {
        "short_term": {
            "total_gain_loss": 0.0,
            "total_proceeds": 0.0,
            "total_cost": 0.0,
            "trade_count": 0,
        },
        "long_term": {
            "total_gain_loss": 0.0,
            "total_proceeds": 0.0,
            "total_cost": 0.0,
            "trade_count": 0,
        },
    }

    for row in result:
        term = "long_term" if row.is_long_term else "short_term"
        summary[term] = {
            "total_gain_loss": row.total_gain_loss or 0.0,
            "total_proceeds": row.total_proceeds or 0.0,
            "total_cost": row.total_cost or 0.0,
            "trade_count": row.trade_count,
        }

    # Add totals
    summary["total"] = {
        "total_gain_loss": summary["short_term"]["total_gain_loss"] + summary["long_term"]["total_gain_loss"],
        "total_proceeds": summary["short_term"]["total_proceeds"] + summary["long_term"]["total_proceeds"],
        "total_cost": summary["short_term"]["total_cost"] + summary["long_term"]["total_cost"],
        "trade_count": summary["short_term"]["trade_count"] + summary["long_term"]["trade_count"],
    }

    return summary


# =====================================================================
# CSV EXPORT ENDPOINTS
# =====================================================================

@router.post("/export/trades/{bot_id}")
async def export_trades_csv(
    bot_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Export trades to CSV file.

    Args:
        bot_id: Bot ID

    Returns:
        Export status
    """
    # Verify bot exists
    result = await session.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()

    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    # Export to CSV
    csv_exporter = CSVExportService(session)
    output_path = Path(f"backend/logs/{bot_id}/trades.csv")

    try:
        await csv_exporter.export_trades_csv(bot_id, output_path)
        return {
            "success": True,
            "message": f"Trades exported to {output_path}",
            "path": str(output_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/export/fiscal/{owner_id}/{year}")
async def export_fiscal_csv(
    owner_id: str,
    year: int,
    session: AsyncSession = Depends(get_session),
):
    """Export fiscal/tax report to CSV.

    Args:
        owner_id: Owner identifier
        year: Tax year

    Returns:
        Export status
    """
    # Export to CSV
    csv_exporter = CSVExportService(session)
    output_path = Path(f"backend/logs/fiscal/{owner_id}/fiscal_{year}.csv")

    try:
        await csv_exporter.export_fiscal_csv(owner_id, year, output_path)
        return {
            "success": True,
            "message": f"Fiscal report exported to {output_path}",
            "path": str(output_path),
            "year": year,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
