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
    TradeDetailRecord,
    BalanceDrilldownRecord,
    BalanceDrilldownEntry,
    RiskStatusRecord,
    BotRiskInfo,
    EquityCurveRecord,
    EquityEvent,
    StrategyReasonRecord,
    BlockedStrategyInfo,
    TaxSummaryRecord,
    AuditLogRecord,
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


class TradeDetailResponse(BaseModel):
    """Trade detail response model."""
    trade: dict
    order: dict
    ledger_entries: List[dict]
    tax_lots_consumed: List[dict]
    realized_gain_loss: Optional[float]
    modeled_cost: float
    realized_cost: float

    class Config:
        from_attributes = True


class BalanceDrilldownEntryResponse(BaseModel):
    """Balance drilldown entry response model."""
    ledger_entry_id: int
    timestamp: datetime
    delta_amount: float
    balance_after: float
    reason: str
    source_classification: str
    related_trade_id: Optional[int]
    related_order_id: Optional[int]

    class Config:
        from_attributes = True


class BalanceDrilldownResponse(BaseModel):
    """Balance drilldown response model."""
    current_balance: float
    ledger_entries: List[BalanceDrilldownEntryResponse]
    cumulative_total: float

    class Config:
        from_attributes = True


class BotRiskInfoResponse(BaseModel):
    """Bot risk info response model."""
    bot_id: int
    bot_name: str
    drawdown_pct: float
    daily_loss_pct: float
    strategy_capacity_pct: float
    kill_switch_state: str
    last_risk_event: Optional[dict]

    class Config:
        from_attributes = True


class RiskStatusResponse(BaseModel):
    """Risk status response model."""
    bots: List[BotRiskInfoResponse]
    portfolio: dict

    class Config:
        from_attributes = True


class EquityCurvePointResponse(BaseModel):
    """Equity curve point response model."""
    timestamp: datetime
    equity: float

    class Config:
        from_attributes = True


class EquityEventResponse(BaseModel):
    """Equity event response model."""
    timestamp: datetime
    event_type: str
    description: str
    bot_id: Optional[int]

    class Config:
        from_attributes = True


class EquityCurveResponse(BaseModel):
    """Equity curve with events response model."""
    curve: List[EquityCurvePointResponse]
    events: List[EquityEventResponse]

    class Config:
        from_attributes = True


class BlockedStrategyInfoResponse(BaseModel):
    """Blocked strategy info response model."""
    strategy_name: str
    blocked_reason: str

    class Config:
        from_attributes = True


class StrategyReasonResponse(BaseModel):
    """Strategy reason response model."""
    current_strategy: str
    current_regime: Optional[str]
    eligible_strategies: List[str]
    blocked_strategies: List[BlockedStrategyInfoResponse]

    class Config:
        from_attributes = True


class TaxSummaryResponse(BaseModel):
    """Tax summary response model."""
    total_realized_gain: float
    short_term_gain: float
    long_term_gain: float
    lot_count: int
    trade_count: int

    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    id: int
    timestamp: datetime
    severity: str
    source: str
    bot_id: Optional[int]
    message: str
    details: Optional[dict]

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


# ============================================================================
# MONEY FORENSICS REPORTS
# ============================================================================

@router.get("/trade-detail/{trade_id}", response_model=TradeDetailResponse)
async def get_trade_detail(
    trade_id: int,
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    session: AsyncSession = Depends(get_session),
):
    """Get full forensic detail for a trade.

    Source: trades + orders + wallet_ledger + tax_lots + realized_gains

    Returns:
    - Trade record
    - Linked order
    - Linked ledger entries (debit/credit)
    - Linked tax lots consumed
    - Realized gain/loss
    - Modeled + realized costs
    """
    service = ReportingService(session)
    record = await service.get_trade_detail(
        trade_id=trade_id,
        is_simulated=is_simulated,
    )

    if not record:
        raise HTTPException(status_code=404, detail="Trade not found")

    return TradeDetailResponse.model_validate(record)


@router.get("/balance-drilldown", response_model=BalanceDrilldownResponse)
async def get_balance_drilldown(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    asset: str = Query(..., description="Asset symbol (required)"),
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    limit: int = Query(20, description="Number of recent entries"),
    session: AsyncSession = Depends(get_session),
):
    """Get balance drilldown report.

    Source: wallet_ledger

    Returns:
    - Current balance
    - Last N wallet_ledger entries affecting this asset
    - Cumulative total
    - Entry source classification (trade, fee, funding, correction)
    """
    service = ReportingService(session)
    record = await service.get_balance_drilldown(
        is_simulated=is_simulated,
        asset=asset,
        owner_id=owner_id,
        limit=limit,
    )

    # Convert nested dataclasses
    response = BalanceDrilldownResponse(
        current_balance=record.current_balance,
        ledger_entries=[
            BalanceDrilldownEntryResponse.model_validate(entry)
            for entry in record.ledger_entries
        ],
        cumulative_total=record.cumulative_total,
    )

    return response


# ============================================================================
# RISK STATUS REPORTS
# ============================================================================

@router.get("/risk-status", response_model=RiskStatusResponse)
async def get_risk_status(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    session: AsyncSession = Depends(get_session),
):
    """Get risk status report.

    Source: bots + wallet_ledger + alerts_log + positions

    Returns:
    - Per-bot:
      - drawdown_pct
      - daily_loss_pct
      - strategy_capacity_pct
      - kill_switch_state
      - last_risk_event
    - Portfolio:
      - total_exposure_pct
      - loss_caps_remaining
    """
    service = ReportingService(session)
    record = await service.get_risk_status(
        is_simulated=is_simulated,
        owner_id=owner_id,
    )

    # Convert nested dataclasses
    response = RiskStatusResponse(
        bots=[
            BotRiskInfoResponse.model_validate(bot)
            for bot in record.bots
        ],
        portfolio=record.portfolio,
    )

    return response


# ============================================================================
# EQUITY CURVE REPORTS
# ============================================================================

@router.get("/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    asset: str = Query('USDT', description="Quote asset for equity calculation"),
    session: AsyncSession = Depends(get_session),
):
    """Get equity curve with event overlays.

    Source: wallet_ledger + strategy_rotations + alerts_log

    Returns:
    - Time series: timestamp, equity
    - Event overlays:
      - strategy_switch
      - kill_switch
      - regime_change
      - grid_recenter
      - large_loss
    """
    service = ReportingService(session)
    curve, events = await service.get_equity_curve(
        is_simulated=is_simulated,
        owner_id=owner_id,
        asset=asset,
    )

    response = EquityCurveResponse(
        curve=[
            EquityCurvePointResponse.model_validate(point)
            for point in curve
        ],
        events=[
            EquityEventResponse.model_validate(event)
            for event in events
        ],
    )

    return response


# ============================================================================
# STRATEGY INTROSPECTION REPORTS
# ============================================================================

@router.get("/strategy-reason/{bot_id}", response_model=StrategyReasonResponse)
async def get_strategy_reason(
    bot_id: int,
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    session: AsyncSession = Depends(get_session),
):
    """Get strategy reasoning for a bot.

    Source: bots + strategy_rotations

    Returns:
    - current_strategy
    - current_regime
    - eligible_strategies
    - blocked_strategies with reasons:
      - cooldown
      - capacity
      - regime_mismatch
      - risk
    """
    service = ReportingService(session)
    record = await service.get_strategy_reason(
        bot_id=bot_id,
        is_simulated=is_simulated,
    )

    if not record:
        raise HTTPException(status_code=404, detail="Bot not found")

    # Convert nested dataclasses
    response = StrategyReasonResponse(
        current_strategy=record.current_strategy,
        current_regime=record.current_regime,
        eligible_strategies=record.eligible_strategies,
        blocked_strategies=[
            BlockedStrategyInfoResponse.model_validate(blocked)
            for blocked in record.blocked_strategies
        ],
    )

    return response


# ============================================================================
# TAX SUMMARY REPORTS
# ============================================================================

@router.get("/tax-summary/{year}", response_model=TaxSummaryResponse)
async def get_tax_summary(
    year: int,
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
    session: AsyncSession = Depends(get_session),
):
    """Get tax summary for a fiscal year.

    Source: realized_gains

    Returns:
    - total_realized_gain
    - short_term_gain
    - long_term_gain
    - lot_count
    - trade_count
    """
    if year < 2000 or year > 2100:
        raise HTTPException(status_code=400, detail="Invalid year")

    service = ReportingService(session)
    record = await service.get_tax_summary(
        year=year,
        is_simulated=is_simulated,
        owner_id=owner_id,
    )

    return TaxSummaryResponse.model_validate(record)


# ============================================================================
# AUDIT & COMPLIANCE REPORTS
# ============================================================================

@router.get("/audit-log", response_model=List[AuditLogResponse])
async def get_audit_log(
    is_simulated: bool = Query(..., description="Filter by simulated flag (required)"),
    bot_id: Optional[int] = Query(None, description="Filter by bot ID"),
    severity: Optional[str] = Query(None, description="Filter by severity (info, warning, error)"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    session: AsyncSession = Depends(get_session),
):
    """Get audit log report.

    Source: alerts_log + strategy_rotations + ledger_invariant failures

    Returns union of:
    - alerts_log
    - strategy_rotations
    - ledger_invariant failures
    """
    if severity and severity not in ['info', 'warning', 'error']:
        raise HTTPException(status_code=400, detail="severity must be 'info', 'warning', or 'error'")

    service = ReportingService(session)
    records = await service.get_audit_log(
        is_simulated=is_simulated,
        bot_id=bot_id,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
    )

    return [AuditLogResponse.model_validate(r) for r in records]
