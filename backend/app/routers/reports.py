"""Reports Router

API endpoints for read-only reports derived from authoritative data sources.
All endpoints require is_simulated parameter to enforce data separation.
"""

import csv
import io
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..models import get_session
from ..services.reporting_service import (
    ReportingService,
    TradeHistoryRecord,
    OrderLifecycleRecord,
    RealizedGainRecord,
    UnrealizedPnLRecord,
    BalanceHistoryRecord,
    DrawdownRecord,
    ExposureRecord,
    StrategyPerformanceRecord,
    AssetPerformanceRecord,
    LedgerAuditRecord,
    CostBasisRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models (Pydantic)
# ============================================================================

class TradeHistoryResponse(BaseModel):
    """Trade history response model."""
    trade_id: int
    order_id: int
    bot_id: int
    strategy: str
    symbol: str
    side: str
    price: float
    quantity: float
    total_cost: float
    realized_pnl: Optional[float]
    timestamp: datetime
    is_simulated: bool

    class Config:
        from_attributes = True


class OrderLifecycleResponse(BaseModel):
    """Order lifecycle response model."""
    order_id: int
    bot_id: int
    strategy: str
    symbol: str
    type: str
    status: str
    amount: float
    price: float
    reason: Optional[str]
    created_at: datetime
    filled_at: Optional[datetime]
    is_simulated: bool

    class Config:
        from_attributes = True


class RealizedGainResponse(BaseModel):
    """Realized gain response model."""
    asset: str
    quantity: float
    buy_date: datetime
    sell_date: datetime
    cost_basis: float
    proceeds: float
    gain_loss: float
    holding_period_days: int
    bot_id: Optional[int]
    is_simulated: bool

    class Config:
        from_attributes = True


class UnrealizedPnLResponse(BaseModel):
    """Unrealized P&L response model."""
    asset: str
    entry_price: float
    current_price: float
    quantity: float
    unrealized_pnl: float
    bot_id: int
    is_simulated: bool

    class Config:
        from_attributes = True


class BalanceHistoryResponse(BaseModel):
    """Balance history response model."""
    timestamp: datetime
    asset: str
    balance: float

    class Config:
        from_attributes = True


class DrawdownResponse(BaseModel):
    """Drawdown response model."""
    timestamp: datetime
    equity: float
    drawdown_pct: float
    max_drawdown_pct: float

    class Config:
        from_attributes = True


class ExposureResponse(BaseModel):
    """Exposure response model."""
    asset: str
    exposure_value: float
    exposure_pct_of_portfolio: float
    strategy: str

    class Config:
        from_attributes = True


class StrategyPerformanceResponse(BaseModel):
    """Strategy performance response model."""
    strategy: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float

    class Config:
        from_attributes = True


class AssetPerformanceResponse(BaseModel):
    """Asset performance response model."""
    symbol: str
    total_trades: int
    total_pnl: float
    avg_trade_pnl: float

    class Config:
        from_attributes = True


class LedgerAuditResponse(BaseModel):
    """Ledger audit response model."""
    ledger_entry_id: int
    timestamp: datetime
    asset: str
    debit: Optional[float]
    credit: Optional[float]
    balance_after: float
    related_trade_id: Optional[int]
    related_order_id: Optional[int]
    reason: str

    class Config:
        from_attributes = True


class CostBasisResponse(BaseModel):
    """Cost basis response model."""
    asset: str
    acquisition_date: datetime
    quantity_remaining: float
    unit_cost: float
    unrealized_gain: float

    class Config:
        from_attributes = True


# ============================================================================
# CORE FINANCIAL REPORTS
# ============================================================================

@router.get("/trades", response_model=List[TradeHistoryResponse])
async def get_trade_history(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    asset: Optional[str] = Query(None, description="Filter by base asset"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get trade history report.

    Source: trades + orders

    Returns all executed trades with realized P&L for SELL trades.
    """
    service = ReportingService(session)
    records = await service.get_trade_history(
        is_simulated=is_simulated,
        bot_id=bot_id,
        asset=asset,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
    )
    return [TradeHistoryResponse.model_validate(r) for r in records]


@router.get("/orders", response_model=List[OrderLifecycleResponse])
async def get_order_lifecycle(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    status: Optional[str] = Query(None, description="Filter by order status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get order lifecycle report.

    Source: orders

    Returns all orders with their status and lifecycle information.
    """
    service = ReportingService(session)
    records = await service.get_order_lifecycle(
        is_simulated=is_simulated,
        bot_id=bot_id,
        strategy=strategy,
        status=status,
        start_date=start_date,
        end_date=end_date,
    )
    return [OrderLifecycleResponse.model_validate(r) for r in records]


@router.get("/realized-gains", response_model=List[RealizedGainResponse])
async def get_realized_gains(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get realized gains report.

    Source: realized_gains

    Returns all realized gains/losses from SELL trades with FIFO cost basis.
    """
    service = ReportingService(session)
    records = await service.get_realized_gains(
        is_simulated=is_simulated,
        bot_id=bot_id,
        asset=asset,
        start_date=start_date,
        end_date=end_date,
    )
    return [RealizedGainResponse.model_validate(r) for r in records]


@router.get("/unrealized-pnl", response_model=List[UnrealizedPnLResponse])
async def get_unrealized_pnl(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    session: AsyncSession = Depends(get_session),
):
    """Get unrealized P&L report.

    Source: positions + latest price

    Returns unrealized gains/losses for open positions.
    Note: Positions are cache only.
    """
    service = ReportingService(session)
    records = await service.get_unrealized_pnl(
        is_simulated=is_simulated,
        bot_id=bot_id,
        asset=asset,
    )
    return [UnrealizedPnLResponse.model_validate(r) for r in records]


@router.get("/balance-history", response_model=List[BalanceHistoryResponse])
async def get_balance_history(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    asset: str = Query(..., description="Asset symbol (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    time_bucket: str = Query('hour', description="Time bucket ('hour' or 'day')"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get balance history report.

    Source: wallet_ledger

    Returns balance over time aggregated by time bucket (hour/day).
    """
    if time_bucket not in ['hour', 'day']:
        raise HTTPException(status_code=400, detail="time_bucket must be 'hour' or 'day'")

    service = ReportingService(session)
    records = await service.get_balance_history(
        is_simulated=is_simulated,
        asset=asset,
        bot_id=bot_id,
        time_bucket=time_bucket,
        start_date=start_date,
        end_date=end_date,
    )
    return [BalanceHistoryResponse.model_validate(r) for r in records]


# ============================================================================
# RISK REPORTS
# ============================================================================

@router.get("/drawdown", response_model=List[DrawdownResponse])
async def get_drawdown(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    asset: str = Query('USDT', description="Quote asset for equity calculation"),
    session: AsyncSession = Depends(get_session),
):
    """Get drawdown report.

    Source: reconstructed equity curve from wallet_ledger

    Returns drawdown percentage and max drawdown over time.
    """
    service = ReportingService(session)
    records = await service.get_drawdown(
        is_simulated=is_simulated,
        bot_id=bot_id,
        asset=asset,
    )
    return [DrawdownResponse.model_validate(r) for r in records]


@router.get("/exposure", response_model=List[ExposureResponse])
async def get_exposure(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    session: AsyncSession = Depends(get_session),
):
    """Get exposure report.

    Source: positions

    Returns current exposure by asset and strategy.
    """
    service = ReportingService(session)
    records = await service.get_exposure(
        is_simulated=is_simulated,
        bot_id=bot_id,
    )
    return [ExposureResponse.model_validate(r) for r in records]


# ============================================================================
# PERFORMANCE REPORTS
# ============================================================================

@router.get("/strategy-performance", response_model=List[StrategyPerformanceResponse])
async def get_strategy_performance(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get strategy performance report.

    Source: trades (aggregated by strategy)

    Returns performance metrics aggregated by strategy.
    """
    service = ReportingService(session)
    records = await service.get_strategy_performance(
        is_simulated=is_simulated,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
    )
    return [StrategyPerformanceResponse.model_validate(r) for r in records]


@router.get("/asset-performance", response_model=List[AssetPerformanceResponse])
async def get_asset_performance(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get asset performance report.

    Source: trades (aggregated by symbol)

    Returns performance metrics aggregated by trading pair.
    """
    service = ReportingService(session)
    records = await service.get_asset_performance(
        is_simulated=is_simulated,
        asset=asset,
        start_date=start_date,
        end_date=end_date,
    )
    return [AssetPerformanceResponse.model_validate(r) for r in records]


# ============================================================================
# ACCOUNTING REPORTS
# ============================================================================

@router.get("/ledger-audit", response_model=List[LedgerAuditResponse])
async def get_ledger_audit(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get ledger audit report.

    Source: wallet_ledger

    Returns all ledger entries with debit/credit classification for audit trail.
    """
    service = ReportingService(session)
    records = await service.get_ledger_audit(
        is_simulated=is_simulated,
        bot_id=bot_id,
        asset=asset,
        start_date=start_date,
        end_date=end_date,
    )
    return [LedgerAuditResponse.model_validate(r) for r in records]


@router.get("/cost-basis", response_model=List[CostBasisResponse])
async def get_cost_basis(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    asset: Optional[str] = Query(None, description="Filter by asset"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    session: AsyncSession = Depends(get_session),
):
    """Get cost basis (open lots) report.

    Source: tax_lots (remaining only)

    Returns all open tax lots with their cost basis.
    """
    service = ReportingService(session)
    records = await service.get_cost_basis(
        is_simulated=is_simulated,
        asset=asset,
        bot_id=bot_id,
    )
    return [CostBasisResponse.model_validate(r) for r in records]


# ============================================================================
# TAX REPORT
# ============================================================================

@router.post("/tax-export/{year}")
async def export_fiscal_year_gains(
    year: int,
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    session: AsyncSession = Depends(get_session),
):
    """Export fiscal year gains to CSV.

    Source: realized_gains (filtered by year)

    Returns CSV file with all realized gains for the specified fiscal year.
    """
    if year < 2000 or year > 2100:
        raise HTTPException(status_code=400, detail="Invalid year")

    service = ReportingService(session)
    records = await service.get_fiscal_year_gains(
        year=year,
        is_simulated=is_simulated,
        owner_id=owner_id,
    )

    # Generate CSV
    output = io.StringIO()
    if records:
        fieldnames = records[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    output.seek(0)

    # Determine filename prefix
    prefix = "simulated" if is_simulated else "live"
    filename = f"{prefix}_fiscal_{year}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
