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
from typing import List, Optional, Dict, Any
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
    LedgerReason,
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
                side=trade.side.value,
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
                status=order.status.value,
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
        query = select(Trade).join(Order).where(
            Order.is_simulated == is_simulated
        )

        if strategy is not None:
            query = query.where(Trade.strategy_used == strategy)

        if start_date is not None:
            query = query.where(Trade.executed_at >= start_date)

        if end_date is not None:
            query = query.where(Trade.executed_at <= end_date)

        result = await self.session.execute(query)
        trades = result.scalars().all()

        # Group by strategy
        strategy_stats: Dict[str, Dict[str, Any]] = {}

        for trade in trades:
            strat = trade.strategy_used or 'unknown'

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
