"""Reporting Service

This service provides read-only reports derived from authoritative tables:
- wallet_ledger (ledger entries)
- trades (trade execution)
- orders (order lifecycle)
- tax_lots (cost basis)
- realized_gains (realized P&L)
- positions (cache only - not authoritative)

Design principles:
- Read-only (no mutations)
- Strict is_simulated separation
- All P&L from ledger/trades/gains (not positions)
- Full traceability
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from decimal import Decimal
from sqlalchemy import select, func, and_, or_, case, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    WalletLedger,
    Trade,
    TradeSide,
    Order,
    OrderStatus,
    RealizedGain,
    Position,
    TaxLot,
    Bot,
    BotStatus,
    LedgerReason,
    Alert,
    StrategyRotation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TradeHistoryRecord:
    """Trade history report record."""
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


@dataclass
class OrderLifecycleRecord:
    """Order lifecycle report record."""
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


@dataclass
class RealizedGainRecord:
    """Realized gains report record."""
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


@dataclass
class UnrealizedPnLRecord:
    """Unrealized P&L report record."""
    asset: str
    entry_price: float
    current_price: float
    quantity: float
    unrealized_pnl: float
    bot_id: int
    is_simulated: bool


@dataclass
class BalanceHistoryRecord:
    """Balance history report record."""
    timestamp: datetime
    asset: str
    balance: float


@dataclass
class DrawdownRecord:
    """Drawdown report record."""
    timestamp: datetime
    equity: float
    drawdown_pct: float
    max_drawdown_pct: float


@dataclass
class ExposureRecord:
    """Exposure report record."""
    asset: str
    exposure_value: float
    exposure_pct_of_portfolio: float
    strategy: str


@dataclass
class StrategyPerformanceRecord:
    """Strategy performance report record."""
    strategy: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float


@dataclass
class AssetPerformanceRecord:
    """Asset performance report record."""
    symbol: str
    total_trades: int
    total_pnl: float
    avg_trade_pnl: float


@dataclass
class LedgerAuditRecord:
    """Ledger audit report record."""
    ledger_entry_id: int
    timestamp: datetime
    asset: str
    debit: Optional[float]
    credit: Optional[float]
    balance_after: float
    related_trade_id: Optional[int]
    related_order_id: Optional[int]
    reason: str


@dataclass
class CostBasisRecord:
    """Cost basis (open lots) report record."""
    asset: str
    acquisition_date: datetime
    quantity_remaining: float
    unit_cost: float
    unrealized_gain: float


@dataclass
class TradeDetailRecord:
    """Trade detail with full forensic trail."""
    trade: Dict[str, Any]
    order: Dict[str, Any]
    ledger_entries: List[Dict[str, Any]]
    tax_lots_consumed: List[Dict[str, Any]]
    realized_gain_loss: Optional[float]
    modeled_cost: float
    realized_cost: float


@dataclass
class BalanceDrilldownEntry:
    """Single ledger entry in balance drilldown."""
    ledger_entry_id: int
    timestamp: datetime
    delta_amount: float
    balance_after: float
    reason: str
    source_classification: str
    related_trade_id: Optional[int]
    related_order_id: Optional[int]


@dataclass
class BalanceDrilldownRecord:
    """Balance drilldown report."""
    current_balance: float
    ledger_entries: List[BalanceDrilldownEntry]
    cumulative_total: float


@dataclass
class BotRiskInfo:
    """Risk information for a single bot."""
    bot_id: int
    bot_name: str
    drawdown_pct: float
    daily_loss_pct: float
    strategy_capacity_pct: float
    kill_switch_state: str
    last_risk_event: Optional[Dict[str, Any]]


@dataclass
class RiskStatusRecord:
    """Risk status report."""
    bots: List[BotRiskInfo]
    portfolio: Dict[str, Any]


@dataclass
class EquityEvent:
    """Equity curve event overlay."""
    timestamp: datetime
    event_type: str
    description: str
    bot_id: Optional[int]


@dataclass
class EquityCurveRecord:
    """Equity curve with events."""
    timestamp: datetime
    equity: float


@dataclass
class BlockedStrategyInfo:
    """Information about a blocked strategy."""
    strategy_name: str
    blocked_reason: str


@dataclass
class StrategyReasonRecord:
    """Strategy reasoning report."""
    current_strategy: str
    current_regime: Optional[str]
    eligible_strategies: List[str]
    blocked_strategies: List[BlockedStrategyInfo]


@dataclass
class TaxSummaryRecord:
    """Tax summary report."""
    total_realized_gain: float
    short_term_gain: float
    long_term_gain: float
    lot_count: int
    trade_count: int


@dataclass
class AuditLogRecord:
    """Audit log entry."""
    id: int
    timestamp: datetime
    severity: str
    source: str
    bot_id: Optional[int]
    message: str
    details: Optional[Dict[str, Any]]


# ============================================================================
# Reporting Service
# ============================================================================

class ReportingService:
    """Service for generating read-only reports from authoritative data sources.

    All reports:
    - Are read-only (no mutations)
    - Enforce is_simulated separation
    - Support optional filters (bot_id, asset, strategy, date range)
    - Derive P&L from ledger/trades/gains (not positions)
    """

    def __init__(self, session: AsyncSession):
        """Initialize the reporting service.

        Args:
            session: Async database session
        """
        self.session = session

    # ========================================================================
    # CORE FINANCIAL REPORTS
    # ========================================================================

    async def get_trade_history(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        asset: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[TradeHistoryRecord]:
        """Get trade history report.

        Source: trades + orders

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            asset: Filter by base asset (optional)
            strategy: Filter by strategy (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of trade history records
        """
        query = select(Trade, Order).join(Order).where(
            Order.is_simulated == is_simulated
        )

        if bot_id is not None:
            query = query.where(Trade.bot_id == bot_id)

        if asset is not None:
            query = query.where(Trade.base_asset == asset)

        if strategy is not None:
            query = query.where(Trade.strategy_used == strategy)

        if start_date is not None:
            query = query.where(Trade.executed_at >= start_date)

        if end_date is not None:
            query = query.where(Trade.executed_at <= end_date)

        query = query.order_by(Trade.executed_at.desc())

        result = await self.session.execute(query)
        rows = result.all()

        records = []
        for trade, order in rows:
            # Calculate realized P&L for SELL trades
            realized_pnl = None
            if trade.side == TradeSide.SELL:
                # Query realized gains for this trade
                gains_query = select(func.sum(RealizedGain.gain_loss)).where(
                    RealizedGain.sell_trade_id == trade.id
                )
                gains_result = await self.session.execute(gains_query)
                realized_pnl = gains_result.scalar() or 0.0

            records.append(TradeHistoryRecord(
                trade_id=trade.id,
                order_id=trade.order_id,
                bot_id=trade.bot_id,
                strategy=trade.strategy_used or '',
                symbol=trade.trading_pair,
                side=trade.side.value.upper(),
                price=trade.price,
                quantity=trade.base_amount,
                total_cost=trade.get_total_cost(),
                realized_pnl=realized_pnl,
                timestamp=trade.executed_at,
                is_simulated=order.is_simulated,
            ))

        return records

    async def get_order_lifecycle(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        strategy: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[OrderLifecycleRecord]:
        """Get order lifecycle report.

        Source: orders

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            strategy: Filter by strategy (optional)
            status: Filter by order status (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of order lifecycle records
        """
        query = select(Order).where(
            Order.is_simulated == is_simulated
        )

        if bot_id is not None:
            query = query.where(Order.bot_id == bot_id)

        if strategy is not None:
            query = query.where(Order.strategy_used == strategy)

        if status is not None:
            query = query.where(Order.status == status)

        if start_date is not None:
            query = query.where(Order.created_at >= start_date)

        if end_date is not None:
            query = query.where(Order.created_at <= end_date)

        query = query.order_by(Order.created_at.desc())

        result = await self.session.execute(query)
        orders = result.scalars().all()

        records = []
        for order in orders:
            records.append(OrderLifecycleRecord(
                order_id=order.id,
                bot_id=order.bot_id,
                strategy=order.strategy_used,
                symbol=order.trading_pair,
                type=order.order_type.value,
                status=order.status.value.upper(),
                amount=order.amount,
                price=order.price,
                reason=order.reason,
                created_at=order.created_at,
                filled_at=order.filled_at,
                is_simulated=order.is_simulated,
            ))

        return records

    async def get_realized_gains(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        asset: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[RealizedGainRecord]:
        """Get realized gains report.

        Source: realized_gains

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            asset: Filter by asset (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of realized gain records
        """
        # Join with Trade to get bot_id and filter by is_simulated
        query = select(RealizedGain, Trade).join(
            Trade, RealizedGain.sell_trade_id == Trade.id
        ).join(
            Order, Trade.order_id == Order.id
        ).where(
            Order.is_simulated == is_simulated
        )

        if bot_id is not None:
            query = query.where(Trade.bot_id == bot_id)

        if asset is not None:
            query = query.where(RealizedGain.asset == asset)

        if start_date is not None:
            query = query.where(RealizedGain.sell_date >= start_date)

        if end_date is not None:
            query = query.where(RealizedGain.sell_date <= end_date)

        query = query.order_by(RealizedGain.sell_date.desc())

        result = await self.session.execute(query)
        rows = result.all()

        records = []
        for gain, trade in rows:
            records.append(RealizedGainRecord(
                asset=gain.asset,
                quantity=gain.quantity,
                buy_date=gain.purchase_date,
                sell_date=gain.sell_date,
                cost_basis=gain.cost_basis,
                proceeds=gain.proceeds,
                gain_loss=gain.gain_loss,
                holding_period_days=gain.holding_period_days,
                bot_id=trade.bot_id,
                is_simulated=is_simulated,
            ))

        return records

    async def get_unrealized_pnl(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        asset: Optional[str] = None,
    ) -> List[UnrealizedPnLRecord]:
        """Get unrealized P&L report.

        Source: positions + latest price
        Note: Positions are cache only. P&L is derived from entry vs current price.

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            asset: Filter by asset (optional)

        Returns:
            List of unrealized P&L records
        """
        query = select(Position, Bot).join(Bot).where(
            Bot.is_dry_run == is_simulated
        )

        if bot_id is not None:
            query = query.where(Position.bot_id == bot_id)

        if asset is not None:
            # Extract base asset from trading_pair (e.g., "BTC/USDT" -> "BTC")
            query = query.where(Position.trading_pair.like(f"{asset}/%"))

        result = await self.session.execute(query)
        rows = result.all()

        records = []
        for position, bot in rows:
            # Extract base asset from trading_pair
            base_asset = position.trading_pair.split('/')[0] if '/' in position.trading_pair else position.trading_pair

            # Calculate unrealized P&L
            # For LONG: (current_price - entry_price) * quantity
            # For SHORT: (entry_price - current_price) * quantity
            if position.side.value == 'LONG':
                unrealized_pnl = (position.current_price - position.entry_price) * position.amount
            else:
                unrealized_pnl = (position.entry_price - position.current_price) * position.amount

            records.append(UnrealizedPnLRecord(
                asset=base_asset,
                entry_price=position.entry_price,
                current_price=position.current_price,
                quantity=position.amount,
                unrealized_pnl=unrealized_pnl,
                bot_id=position.bot_id,
                is_simulated=is_simulated,
            ))

        return records

    async def get_balance_history(
        self,
        is_simulated: bool,
        asset: str,
        bot_id: Optional[int] = None,
        time_bucket: str = 'hour',  # 'hour' or 'day'
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[BalanceHistoryRecord]:
        """Get balance history report.

        Source: wallet_ledger
        Aggregates by time bucket (hour/day)

        Args:
            is_simulated: Filter by simulated flag (required)
            asset: Asset symbol (required)
            bot_id: Filter by bot ID (optional)
            time_bucket: Time bucket ('hour' or 'day')
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of balance history records
        """
        query = select(WalletLedger).join(Bot).where(
            and_(
                WalletLedger.asset == asset,
                Bot.is_dry_run == is_simulated
            )
        )

        if bot_id is not None:
            query = query.where(WalletLedger.bot_id == bot_id)

        if start_date is not None:
            query = query.where(WalletLedger.created_at >= start_date)

        if end_date is not None:
            query = query.where(WalletLedger.created_at <= end_date)

        query = query.order_by(WalletLedger.created_at)

        result = await self.session.execute(query)
        entries = result.scalars().all()

        # Aggregate by time bucket
        records = []
        if not entries:
            return records

        # Group entries by time bucket
        buckets: Dict[datetime, float] = {}
        for entry in entries:
            if time_bucket == 'hour':
                bucket_time = entry.created_at.replace(minute=0, second=0, microsecond=0)
            else:  # day
                bucket_time = entry.created_at.replace(hour=0, minute=0, second=0, microsecond=0)

            # Use the last balance_after in each bucket
            if bucket_time not in buckets or entry.created_at > bucket_time:
                buckets[bucket_time] = entry.balance_after or 0.0

        # Convert to records
        for timestamp in sorted(buckets.keys()):
            records.append(BalanceHistoryRecord(
                timestamp=timestamp,
                asset=asset,
                balance=buckets[timestamp],
            ))

        return records

    # ========================================================================
    # RISK REPORTS
    # ========================================================================

    async def get_drawdown(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        asset: str = 'USDT',
    ) -> List[DrawdownRecord]:
        """Get drawdown report.

        Source: reconstructed equity curve from wallet_ledger

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            asset: Quote asset for equity calculation (default: USDT)

        Returns:
            List of drawdown records
        """
        # Get balance history
        query = select(WalletLedger).join(Bot).where(
            and_(
                WalletLedger.asset == asset,
                Bot.is_dry_run == is_simulated
            )
        )

        if bot_id is not None:
            query = query.where(WalletLedger.bot_id == bot_id)

        query = query.order_by(WalletLedger.created_at)

        result = await self.session.execute(query)
        entries = result.scalars().all()

        if not entries:
            return []

        records = []
        peak_equity = 0.0
        max_drawdown_pct = 0.0

        for entry in entries:
            equity = entry.balance_after or 0.0

            # Update peak
            if equity > peak_equity:
                peak_equity = equity

            # Calculate drawdown
            if peak_equity > 0:
                drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
            else:
                drawdown_pct = 0.0

            # Update max drawdown
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

            records.append(DrawdownRecord(
                timestamp=entry.created_at,
                equity=equity,
                drawdown_pct=drawdown_pct,
                max_drawdown_pct=max_drawdown_pct,
            ))

        return records

    async def get_exposure(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
    ) -> List[ExposureRecord]:
        """Get exposure report.

        Source: positions

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)

        Returns:
            List of exposure records
        """
        query = select(Position, Bot).join(Bot).where(
            Bot.is_dry_run == is_simulated
        )

        if bot_id is not None:
            query = query.where(Position.bot_id == bot_id)

        result = await self.session.execute(query)
        rows = result.all()

        # Calculate total portfolio value
        total_portfolio_value = 0.0
        position_data = []

        for position, bot in rows:
            exposure_value = position.amount * position.current_price
            total_portfolio_value += exposure_value

            base_asset = position.trading_pair.split('/')[0] if '/' in position.trading_pair else position.trading_pair

            position_data.append({
                'asset': base_asset,
                'exposure_value': exposure_value,
                'strategy': bot.strategy,
            })

        # Calculate exposure percentages
        records = []
        for data in position_data:
            exposure_pct = (data['exposure_value'] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0

            records.append(ExposureRecord(
                asset=data['asset'],
                exposure_value=data['exposure_value'],
                exposure_pct_of_portfolio=exposure_pct,
                strategy=data['strategy'],
            ))

        return records

    # ========================================================================
    # PERFORMANCE REPORTS
    # ========================================================================

    async def get_strategy_performance(
        self,
        is_simulated: bool,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[StrategyPerformanceRecord]:
        """Get strategy performance report.

        Source: trades
        Aggregates by strategy

        Args:
            is_simulated: Filter by simulated flag (required)
            strategy: Filter by strategy (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of strategy performance records
        """
        # Query all trades for strategies
        query = select(Trade, Order).join(Order).where(
            Order.is_simulated == is_simulated
        )

        if strategy is not None:
            query = query.where(Trade.strategy_used == strategy)

        if start_date is not None:
            query = query.where(Trade.executed_at >= start_date)

        if end_date is not None:
            query = query.where(Trade.executed_at <= end_date)

        result = await self.session.execute(query)
        rows = result.all()

        # Group by strategy
        strategy_stats: Dict[str, Dict[str, Any]] = {}

        for trade, order in rows:
            strat = trade.strategy_used or order.strategy_used or 'unknown'

            if strat not in strategy_stats:
                strategy_stats[strat] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0.0,
                    'pnls': [],
                }

            strategy_stats[strat]['total_trades'] += 1

            # Calculate P&L for this trade
            if trade.side == TradeSide.SELL:
                # Get realized gains
                gains_query = select(func.sum(RealizedGain.gain_loss)).where(
                    RealizedGain.sell_trade_id == trade.id
                )
                gains_result = await self.session.execute(gains_query)
                pnl = gains_result.scalar() or 0.0

                strategy_stats[strat]['total_pnl'] += pnl
                strategy_stats[strat]['pnls'].append(pnl)

                if pnl > 0:
                    strategy_stats[strat]['winning_trades'] += 1

        # Calculate metrics
        records = []
        for strat, stats in strategy_stats.items():
            total_trades = stats['total_trades']
            win_rate = (stats['winning_trades'] / total_trades * 100) if total_trades > 0 else 0.0
            avg_pnl = stats['total_pnl'] / total_trades if total_trades > 0 else 0.0

            # Calculate max drawdown (simplified - use cumulative P&L)
            max_drawdown = 0.0
            if stats['pnls']:
                cumulative = 0.0
                peak = 0.0
                for pnl in stats['pnls']:
                    cumulative += pnl
                    if cumulative > peak:
                        peak = cumulative
                    drawdown = peak - cumulative
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            records.append(StrategyPerformanceRecord(
                strategy=strat,
                total_trades=total_trades,
                win_rate=win_rate,
                total_pnl=stats['total_pnl'],
                avg_pnl=avg_pnl,
                max_drawdown=max_drawdown,
            ))

        return records

    async def get_asset_performance(
        self,
        is_simulated: bool,
        asset: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[AssetPerformanceRecord]:
        """Get asset performance report.

        Source: trades
        Aggregates by symbol

        Args:
            is_simulated: Filter by simulated flag (required)
            asset: Filter by asset (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of asset performance records
        """
        query = select(Trade).join(Order).where(
            Order.is_simulated == is_simulated
        )

        if asset is not None:
            query = query.where(Trade.base_asset == asset)

        if start_date is not None:
            query = query.where(Trade.executed_at >= start_date)

        if end_date is not None:
            query = query.where(Trade.executed_at <= end_date)

        result = await self.session.execute(query)
        trades = result.scalars().all()

        # Group by trading_pair
        asset_stats: Dict[str, Dict[str, Any]] = {}

        for trade in trades:
            symbol = trade.trading_pair

            if symbol not in asset_stats:
                asset_stats[symbol] = {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                }

            asset_stats[symbol]['total_trades'] += 1

            # Calculate P&L for SELL trades
            if trade.side == TradeSide.SELL:
                gains_query = select(func.sum(RealizedGain.gain_loss)).where(
                    RealizedGain.sell_trade_id == trade.id
                )
                gains_result = await self.session.execute(gains_query)
                pnl = gains_result.scalar() or 0.0
                asset_stats[symbol]['total_pnl'] += pnl

        # Calculate metrics
        records = []
        for symbol, stats in asset_stats.items():
            avg_trade_pnl = stats['total_pnl'] / stats['total_trades'] if stats['total_trades'] > 0 else 0.0

            records.append(AssetPerformanceRecord(
                symbol=symbol,
                total_trades=stats['total_trades'],
                total_pnl=stats['total_pnl'],
                avg_trade_pnl=avg_trade_pnl,
            ))

        return records

    # ========================================================================
    # ACCOUNTING REPORTS
    # ========================================================================

    async def get_ledger_audit(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        asset: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[LedgerAuditRecord]:
        """Get ledger audit report.

        Source: wallet_ledger

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            asset: Filter by asset (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of ledger audit records
        """
        query = select(WalletLedger).join(Bot).where(
            Bot.is_dry_run == is_simulated
        )

        if bot_id is not None:
            query = query.where(WalletLedger.bot_id == bot_id)

        if asset is not None:
            query = query.where(WalletLedger.asset == asset)

        if start_date is not None:
            query = query.where(WalletLedger.created_at >= start_date)

        if end_date is not None:
            query = query.where(WalletLedger.created_at <= end_date)

        query = query.order_by(WalletLedger.created_at.desc())

        result = await self.session.execute(query)
        entries = result.scalars().all()

        records = []
        for entry in entries:
            # Classify as debit or credit
            debit = abs(entry.delta_amount) if entry.delta_amount < 0 else None
            credit = entry.delta_amount if entry.delta_amount > 0 else None

            records.append(LedgerAuditRecord(
                ledger_entry_id=entry.id,
                timestamp=entry.created_at,
                asset=entry.asset,
                debit=debit,
                credit=credit,
                balance_after=entry.balance_after or 0.0,
                related_trade_id=entry.related_trade_id,
                related_order_id=entry.related_order_id,
                reason=entry.reason.value,
            ))

        return records

    async def get_cost_basis(
        self,
        is_simulated: bool,
        asset: Optional[str] = None,
        bot_id: Optional[int] = None,
    ) -> List[CostBasisRecord]:
        """Get cost basis (open lots) report.

        Source: tax_lots (remaining only)

        Args:
            is_simulated: Filter by simulated flag (required)
            asset: Filter by asset (optional)
            bot_id: Filter by bot ID (optional - requires joining with trades)

        Returns:
            List of cost basis records
        """
        # Tax lots don't have is_simulated or bot_id directly
        # We need to join with Trade via purchase_trade_id
        query = select(TaxLot).join(
            Trade, TaxLot.purchase_trade_id == Trade.id
        ).join(
            Order, Trade.order_id == Order.id
        ).where(
            and_(
                TaxLot.is_fully_consumed == False,
                Order.is_simulated == is_simulated
            )
        )

        if asset is not None:
            query = query.where(TaxLot.asset == asset)

        if bot_id is not None:
            query = query.where(Trade.bot_id == bot_id)

        query = query.order_by(TaxLot.purchase_date)

        result = await self.session.execute(query)
        lots = result.scalars().all()

        records = []
        for lot in lots:
            # Unrealized gain = (current_price - unit_cost) * quantity_remaining
            # Note: We don't have current_price here, so this is 0 for now
            # In a real implementation, you'd fetch current price
            unrealized_gain = 0.0

            records.append(CostBasisRecord(
                asset=lot.asset,
                acquisition_date=lot.purchase_date,
                quantity_remaining=lot.quantity_remaining,
                unit_cost=lot.unit_cost,
                unrealized_gain=unrealized_gain,
            ))

        return records

    # ========================================================================
    # TAX REPORT
    # ========================================================================

    async def get_fiscal_year_gains(
        self,
        year: int,
        is_simulated: bool,
        owner_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get fiscal year gains for tax export.

        Source: realized_gains
        Filter by year

        Args:
            year: Fiscal year (required)
            is_simulated: Filter by simulated flag (required)
            owner_id: Filter by owner ID (optional)

        Returns:
            List of gain records for CSV export
        """
        # Filter by year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        query = select(RealizedGain).join(
            Trade, RealizedGain.sell_trade_id == Trade.id
        ).join(
            Order, Trade.order_id == Order.id
        ).where(
            and_(
                RealizedGain.sell_date >= start_date,
                RealizedGain.sell_date <= end_date,
                Order.is_simulated == is_simulated
            )
        )

        if owner_id is not None:
            query = query.where(RealizedGain.owner_id == owner_id)

        query = query.order_by(RealizedGain.sell_date)

        result = await self.session.execute(query)
        gains = result.scalars().all()

        records = []
        for gain in gains:
            records.append({
                'asset': gain.asset,
                'quantity': gain.quantity,
                'buy_date': gain.purchase_date.isoformat(),
                'sell_date': gain.sell_date.isoformat(),
                'cost_basis': gain.cost_basis,
                'proceeds': gain.proceeds,
                'gain_loss': gain.gain_loss,
                'holding_period_days': gain.holding_period_days,
                'is_long_term': gain.is_long_term,
            })

        return records

    # ========================================================================
    # MONEY FORENSICS REPORTS
    # ========================================================================

    async def get_trade_detail(
        self,
        trade_id: int,
        is_simulated: bool,
    ) -> Optional[TradeDetailRecord]:
        """Get full forensic detail for a trade.

        Source: trades + orders + wallet_ledger + tax_lots + realized_gains

        Args:
            trade_id: Trade ID (required)
            is_simulated: Filter by simulated flag (required)

        Returns:
            Trade detail record or None if not found
        """
        # Get trade with order
        query = select(Trade, Order).join(Order).where(
            and_(
                Trade.id == trade_id,
                Order.is_simulated == is_simulated
            )
        )
        result = await self.session.execute(query)
        row = result.one_or_none()

        if not row:
            return None

        trade, order = row

        # Get ledger entries for this trade
        ledger_query = select(WalletLedger).where(
            WalletLedger.related_trade_id == trade_id
        ).order_by(WalletLedger.created_at)
        ledger_result = await self.session.execute(ledger_query)
        ledger_entries = ledger_result.scalars().all()

        # Get tax lots consumed (for SELL trades)
        tax_lots_consumed = []
        realized_gain_loss = None
        if trade.side == TradeSide.SELL:
            # Get realized gains linked to this sell trade
            gains_query = select(RealizedGain).where(
                RealizedGain.sell_trade_id == trade_id
            )
            gains_result = await self.session.execute(gains_query)
            gains = gains_result.scalars().all()

            realized_gain_loss = sum(g.gain_loss for g in gains)

            # Get tax lot details
            for gain in gains:
                if gain.tax_lot_id:
                    lot_query = select(TaxLot).where(TaxLot.id == gain.tax_lot_id)
                    lot_result = await self.session.execute(lot_query)
                    lot = lot_result.scalar_one_or_none()
                    if lot:
                        tax_lots_consumed.append({
                            'tax_lot_id': lot.id,
                            'quantity_consumed': gain.quantity,
                            'unit_cost': lot.unit_cost,
                            'purchase_date': lot.purchase_date.isoformat(),
                        })

        return TradeDetailRecord(
            trade=trade.to_dict(),
            order={
                'order_id': order.id,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'amount': order.amount,
                'price': order.price,
                'created_at': order.created_at.isoformat(),
                'filled_at': order.filled_at.isoformat() if order.filled_at else None,
            },
            ledger_entries=[
                {
                    'ledger_entry_id': entry.id,
                    'asset': entry.asset,
                    'delta_amount': entry.delta_amount,
                    'balance_after': entry.balance_after,
                    'reason': entry.reason.value,
                    'timestamp': entry.created_at.isoformat(),
                }
                for entry in ledger_entries
            ],
            tax_lots_consumed=tax_lots_consumed,
            realized_gain_loss=realized_gain_loss,
            modeled_cost=trade.modeled_cost,
            realized_cost=trade.get_total_cost(),
        )

    async def get_balance_drilldown(
        self,
        is_simulated: bool,
        asset: str,
        owner_id: Optional[str] = None,
        limit: int = 20,
    ) -> BalanceDrilldownRecord:
        """Get balance drilldown with last N ledger entries.

        Source: wallet_ledger

        Args:
            is_simulated: Filter by simulated flag (required)
            asset: Asset symbol (required)
            owner_id: Filter by owner ID (optional)
            limit: Number of recent entries (default: 20)

        Returns:
            Balance drilldown record
        """
        # Get last N ledger entries for this asset
        query = select(WalletLedger).join(
            Bot, WalletLedger.bot_id == Bot.id
        ).where(
            and_(
                WalletLedger.asset == asset,
                Bot.is_dry_run == is_simulated
            )
        )

        if owner_id is not None:
            query = query.where(WalletLedger.owner_id == owner_id)

        query = query.order_by(WalletLedger.created_at.desc()).limit(limit)

        result = await self.session.execute(query)
        entries = result.scalars().all()

        # Get current balance (most recent entry)
        current_balance = entries[0].balance_after if entries else 0.0

        # Classify entries by source
        def classify_source(reason_value: str) -> str:
            """Classify ledger entry source."""
            if reason_value in ['buy', 'sell']:
                return 'trade'
            elif reason_value == 'fee':
                return 'fee'
            elif reason_value in ['allocation', 'deallocation', 'transfer']:
                return 'funding'
            elif reason_value == 'correction':
                return 'correction'
            else:
                return 'other'

        # Build drilldown entries (reverse to show chronological order)
        drilldown_entries = []
        for entry in reversed(entries):
            drilldown_entries.append(BalanceDrilldownEntry(
                ledger_entry_id=entry.id,
                timestamp=entry.created_at,
                delta_amount=entry.delta_amount,
                balance_after=entry.balance_after or 0.0,
                reason=entry.reason.value,
                source_classification=classify_source(entry.reason.value),
                related_trade_id=entry.related_trade_id,
                related_order_id=entry.related_order_id,
            ))

        # Calculate cumulative total
        cumulative_total = current_balance

        return BalanceDrilldownRecord(
            current_balance=current_balance,
            ledger_entries=drilldown_entries,
            cumulative_total=cumulative_total,
        )

    # ========================================================================
    # RISK STATUS REPORTS
    # ========================================================================

    async def get_risk_status(
        self,
        is_simulated: bool,
        owner_id: Optional[str] = None,
    ) -> RiskStatusRecord:
        """Get risk status report for all bots.

        Source: bots + wallet_ledger + alerts_log + positions

        Args:
            is_simulated: Filter by simulated flag (required)
            owner_id: Filter by owner ID (optional)

        Returns:
            Risk status record
        """
        # Get all active bots
        query = select(Bot).where(Bot.is_dry_run == is_simulated)

        # Note: owner_id filtering would require adding owner_id to Bot model
        # For now, we'll skip this filter

        result = await self.session.execute(query)
        bots = result.scalars().all()

        bot_risk_info = []
        total_exposure_usd = 0.0
        total_portfolio_value = 0.0

        for bot in bots:
            # Calculate drawdown
            drawdown_pct = 0.0
            if bot.budget > 0:
                drawdown_pct = ((bot.budget - bot.current_balance) / bot.budget) * 100

            # Calculate daily loss
            daily_loss_pct = 0.0
            if bot.started_at:
                one_day_ago = datetime.utcnow() - timedelta(days=1)
                if bot.started_at >= one_day_ago:
                    # Get balance from 24h ago
                    balance_query = select(WalletLedger).where(
                        and_(
                            WalletLedger.bot_id == bot.id,
                            WalletLedger.created_at >= one_day_ago
                        )
                    ).order_by(WalletLedger.created_at).limit(1)
                    balance_result = await self.session.execute(balance_query)
                    first_entry = balance_result.scalar_one_or_none()

                    if first_entry and first_entry.balance_after:
                        daily_loss_pct = ((first_entry.balance_after - bot.current_balance) / first_entry.balance_after) * 100

            # Get strategy capacity (simplified)
            strategy_capacity_pct = 0.0
            # In a real implementation, this would call StrategyCapacityService

            # Determine kill switch state
            kill_switch_state = "active"
            if bot.status == BotStatus.STOPPED:
                kill_switch_state = "stopped"
            elif bot.status == BotStatus.PAUSED:
                kill_switch_state = "paused"

            # Get last risk event (last alert)
            alert_query = select(Alert).where(
                Alert.bot_id == bot.id
            ).order_by(Alert.created_at.desc()).limit(1)
            alert_result = await self.session.execute(alert_query)
            last_alert = alert_result.scalar_one_or_none()

            last_risk_event = None
            if last_alert:
                last_risk_event = {
                    'timestamp': last_alert.created_at.isoformat(),
                    'type': last_alert.alert_type,
                    'message': last_alert.message,
                }

            bot_risk_info.append(BotRiskInfo(
                bot_id=bot.id,
                bot_name=bot.name,
                drawdown_pct=drawdown_pct,
                daily_loss_pct=daily_loss_pct,
                strategy_capacity_pct=strategy_capacity_pct,
                kill_switch_state=kill_switch_state,
                last_risk_event=last_risk_event,
            ))

            # Calculate exposure
            positions_query = select(Position).where(Position.bot_id == bot.id)
            positions_result = await self.session.execute(positions_query)
            positions = positions_result.scalars().all()

            for position in positions:
                exposure_value = position.amount * position.current_price
                total_exposure_usd += exposure_value

            total_portfolio_value += bot.current_balance

        # Calculate portfolio-level metrics
        total_exposure_pct = (total_exposure_usd / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0

        portfolio_info = {
            'total_exposure_pct': total_exposure_pct,
            'total_exposure_usd': total_exposure_usd,
            'total_portfolio_value': total_portfolio_value,
            'loss_caps_remaining': {
                # Placeholder for loss cap tracking
                'daily': None,
                'weekly': None,
            },
        }

        return RiskStatusRecord(
            bots=bot_risk_info,
            portfolio=portfolio_info,
        )

    # ========================================================================
    # EQUITY CURVE REPORTS
    # ========================================================================

    async def get_equity_curve(
        self,
        is_simulated: bool,
        owner_id: Optional[str] = None,
        asset: str = 'USDT',
    ) -> Tuple[List[EquityCurveRecord], List[EquityEvent]]:
        """Get equity curve with event overlays.

        Source: wallet_ledger + strategy_rotations + alerts_log

        Args:
            is_simulated: Filter by simulated flag (required)
            owner_id: Filter by owner ID (optional)
            asset: Quote asset for equity calculation (default: USDT)

        Returns:
            Tuple of (equity curve records, event records)
        """
        # Get balance history for equity curve
        query = select(WalletLedger).join(Bot).where(
            and_(
                WalletLedger.asset == asset,
                Bot.is_dry_run == is_simulated
            )
        ).order_by(WalletLedger.created_at)

        result = await self.session.execute(query)
        entries = result.scalars().all()

        equity_curve = [
            EquityCurveRecord(
                timestamp=entry.created_at,
                equity=entry.balance_after or 0.0,
            )
            for entry in entries
        ]

        # Get events
        events = []

        # Strategy rotations
        rotations_query = select(StrategyRotation).join(Bot).where(
            Bot.is_dry_run == is_simulated
        ).order_by(StrategyRotation.created_at)
        rotations_result = await self.session.execute(rotations_query)
        rotations = rotations_result.scalars().all()

        for rotation in rotations:
            events.append(EquityEvent(
                timestamp=rotation.created_at,
                event_type='strategy_switch',
                description=f"Strategy switched from {rotation.from_strategy} to {rotation.to_strategy}",
                bot_id=rotation.bot_id,
            ))

        # Alerts (kill switch, large loss, etc.)
        alerts_query = select(Alert).join(Bot).where(
            Bot.is_dry_run == is_simulated
        ).order_by(Alert.created_at)
        alerts_result = await self.session.execute(alerts_query)
        alerts = alerts_result.scalars().all()

        for alert in alerts:
            # Classify alert type
            event_type = 'alert'
            if 'stop' in alert.alert_type.lower() or 'kill' in alert.alert_type.lower():
                event_type = 'kill_switch'
            elif 'loss' in alert.alert_type.lower():
                event_type = 'large_loss'
            elif 'drawdown' in alert.alert_type.lower():
                event_type = 'drawdown'

            events.append(EquityEvent(
                timestamp=alert.created_at,
                event_type=event_type,
                description=alert.message,
                bot_id=alert.bot_id,
            ))

        return equity_curve, events

    # ========================================================================
    # STRATEGY INTROSPECTION REPORTS
    # ========================================================================

    async def get_strategy_reason(
        self,
        bot_id: int,
        is_simulated: bool,
    ) -> Optional[StrategyReasonRecord]:
        """Get strategy reasoning for a bot.

        Source: bots + strategy_rotations + (strategy eligibility logic)

        Args:
            bot_id: Bot ID (required)
            is_simulated: Filter by simulated flag (required)

        Returns:
            Strategy reason record or None if not found
        """
        # Get bot
        query = select(Bot).where(
            and_(
                Bot.id == bot_id,
                Bot.is_dry_run == is_simulated
            )
        )
        result = await self.session.execute(query)
        bot = result.scalar_one_or_none()

        if not bot:
            return None

        current_strategy = bot.strategy
        current_regime = None  # Placeholder - would need regime detection logic

        # Get all available strategies (hardcoded for now)
        all_strategies = [
            'mean_reversion',
            'trend_following',
            'volatility_breakout',
            'adaptive_grid',
            'dca_accumulator',
        ]

        # Determine eligible and blocked strategies
        eligible_strategies = []
        blocked_strategies = []

        # Check strategy rotation limit
        rotation_count_query = select(func.count(StrategyRotation.id)).where(
            StrategyRotation.bot_id == bot_id
        )
        rotation_count_result = await self.session.execute(rotation_count_query)
        rotation_count = rotation_count_result.scalar() or 0

        for strategy in all_strategies:
            if strategy == current_strategy:
                eligible_strategies.append(strategy)
                continue

            # Check cooldown (last rotation within 1 hour)
            recent_rotation_query = select(StrategyRotation).where(
                and_(
                    StrategyRotation.bot_id == bot_id,
                    StrategyRotation.to_strategy == strategy,
                    StrategyRotation.created_at >= datetime.utcnow() - timedelta(hours=1)
                )
            )
            recent_rotation_result = await self.session.execute(recent_rotation_query)
            recent_rotation = recent_rotation_result.scalar_one_or_none()

            if recent_rotation:
                blocked_strategies.append(BlockedStrategyInfo(
                    strategy_name=strategy,
                    blocked_reason='cooldown (1 hour since last rotation)',
                ))
                continue

            # Check capacity
            if bot.max_strategy_rotations and rotation_count >= bot.max_strategy_rotations:
                blocked_strategies.append(BlockedStrategyInfo(
                    strategy_name=strategy,
                    blocked_reason=f'capacity (max {bot.max_strategy_rotations} rotations reached)',
                ))
                continue

            # Check risk (simplified)
            if bot.status == BotStatus.PAUSED or bot.status == BotStatus.STOPPED:
                blocked_strategies.append(BlockedStrategyInfo(
                    strategy_name=strategy,
                    blocked_reason=f'risk (bot status: {bot.status.value})',
                ))
                continue

            eligible_strategies.append(strategy)

        return StrategyReasonRecord(
            current_strategy=current_strategy,
            current_regime=current_regime,
            eligible_strategies=eligible_strategies,
            blocked_strategies=blocked_strategies,
        )

    # ========================================================================
    # TAX SUMMARY REPORTS
    # ========================================================================

    async def get_tax_summary(
        self,
        year: int,
        is_simulated: bool,
        owner_id: Optional[str] = None,
    ) -> TaxSummaryRecord:
        """Get tax summary for a fiscal year.

        Source: realized_gains

        Args:
            year: Fiscal year (required)
            is_simulated: Filter by simulated flag (required)
            owner_id: Filter by owner ID (optional)

        Returns:
            Tax summary record
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        # Get all realized gains for the year
        query = select(RealizedGain).join(
            Trade, RealizedGain.sell_trade_id == Trade.id
        ).join(
            Order, Trade.order_id == Order.id
        ).where(
            and_(
                RealizedGain.sell_date >= start_date,
                RealizedGain.sell_date <= end_date,
                Order.is_simulated == is_simulated
            )
        )

        if owner_id is not None:
            query = query.where(RealizedGain.owner_id == owner_id)

        result = await self.session.execute(query)
        gains = result.scalars().all()

        # Calculate summary
        total_realized_gain = 0.0
        short_term_gain = 0.0
        long_term_gain = 0.0
        lot_count = len(gains)

        # Count unique trades
        trade_ids = set()
        for gain in gains:
            total_realized_gain += gain.gain_loss
            if gain.is_long_term:
                long_term_gain += gain.gain_loss
            else:
                short_term_gain += gain.gain_loss
            trade_ids.add(gain.sell_trade_id)

        trade_count = len(trade_ids)

        return TaxSummaryRecord(
            total_realized_gain=total_realized_gain,
            short_term_gain=short_term_gain,
            long_term_gain=long_term_gain,
            lot_count=lot_count,
            trade_count=trade_count,
        )

    # ========================================================================
    # AUDIT & COMPLIANCE REPORTS
    # ========================================================================

    async def get_audit_log(
        self,
        is_simulated: bool,
        bot_id: Optional[int] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[AuditLogRecord]:
        """Get audit log combining alerts and strategy rotations.

        Source: alerts_log + strategy_rotations + (ledger_invariant failures)

        Args:
            is_simulated: Filter by simulated flag (required)
            bot_id: Filter by bot ID (optional)
            severity: Filter by severity (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of audit log records
        """
        records = []

        # Get alerts
        alerts_query = select(Alert).join(Bot).where(
            Bot.is_dry_run == is_simulated
        )

        if bot_id is not None:
            alerts_query = alerts_query.where(Alert.bot_id == bot_id)

        if start_date is not None:
            alerts_query = alerts_query.where(Alert.created_at >= start_date)

        if end_date is not None:
            alerts_query = alerts_query.where(Alert.created_at <= end_date)

        alerts_query = alerts_query.order_by(Alert.created_at.desc())

        alerts_result = await self.session.execute(alerts_query)
        alerts = alerts_result.scalars().all()

        for alert in alerts:
            # Map alert_type to severity
            alert_severity = 'info'
            if 'error' in alert.alert_type.lower() or 'stop' in alert.alert_type.lower():
                alert_severity = 'error'
            elif 'warning' in alert.alert_type.lower() or 'loss' in alert.alert_type.lower():
                alert_severity = 'warning'

            if severity is None or alert_severity == severity:
                records.append(AuditLogRecord(
                    id=alert.id,
                    timestamp=alert.created_at,
                    severity=alert_severity,
                    source='alerts_log',
                    bot_id=alert.bot_id,
                    message=alert.message,
                    details={'alert_type': alert.alert_type, 'email_sent': alert.email_sent},
                ))

        # Get strategy rotations
        rotations_query = select(StrategyRotation).join(Bot).where(
            Bot.is_dry_run == is_simulated
        )

        if bot_id is not None:
            rotations_query = rotations_query.where(StrategyRotation.bot_id == bot_id)

        if start_date is not None:
            rotations_query = rotations_query.where(StrategyRotation.created_at >= start_date)

        if end_date is not None:
            rotations_query = rotations_query.where(StrategyRotation.created_at <= end_date)

        rotations_query = rotations_query.order_by(StrategyRotation.created_at.desc())

        rotations_result = await self.session.execute(rotations_query)
        rotations = rotations_result.scalars().all()

        for rotation in rotations:
            rotation_severity = 'info'
            if severity is None or rotation_severity == severity:
                records.append(AuditLogRecord(
                    id=rotation.id,
                    timestamp=rotation.created_at,
                    severity=rotation_severity,
                    source='strategy_rotations',
                    bot_id=rotation.bot_id,
                    message=f"Strategy rotated from {rotation.from_strategy} to {rotation.to_strategy}",
                    details={'from_strategy': rotation.from_strategy, 'to_strategy': rotation.to_strategy, 'reason': rotation.reason},
                ))

        # Sort by timestamp descending
        records.sort(key=lambda r: r.timestamp, reverse=True)

        return records
