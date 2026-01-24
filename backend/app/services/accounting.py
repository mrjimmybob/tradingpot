"""Complete accounting services: Trade recording, FIFO tax engine, CSV exports.

CRITICAL: This module provides accounting-grade ledger management:
- Trade recording with full audit trail
- FIFO tax lot matching (deterministic)
- Realized gain/loss calculation
- CSV export from database (not primary storage)

Design constraints:
- SQLite is single source of truth
- CSV files are derived exports only
- FIFO matching is deterministic and persisted
- Never compute tax on the fly
"""

import logging
import csv
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models import (
    Trade, TradeSide, TaxLot, RealizedGain,
    Order, Bot, WalletLedger
)
from .ledger_writer import LedgerWriterService

logger = logging.getLogger(__name__)


class TradeRecorderService:
    """Service for recording trade executions.

    CRITICAL: Trades are the authoritative record of what was executed.
    Orders are intent, trades are reality.

    Responsibilities:
    - Record trade execution details
    - Create corresponding ledger entries
    - Link trades to orders
    """

    def __init__(self, session: AsyncSession):
        """Initialize trade recorder service.

        Args:
            session: Database session
        """
        self.session = session
        self.ledger_writer = LedgerWriterService(session)

    async def record_trade(
        self,
        order_id: int,
        owner_id: str,
        bot_id: int,
        exchange: str,
        trading_pair: str,
        side: TradeSide,
        base_asset: str,
        quote_asset: str,
        base_amount: float,
        quote_amount: float,
        price: float,
        fee_amount: float,
        fee_asset: str,
        modeled_cost: float,
        exchange_trade_id: Optional[str] = None,
        executed_at: Optional[datetime] = None,
        strategy_used: Optional[str] = None,
    ) -> Trade:
        """Record a trade execution.

        This creates:
        1. Trade record
        2. Ledger entries (double-entry accounting)
        3. Tax lots (for BUY) or consumes lots (for SELL)
        4. Realized gains (for SELL)

        Args:
            order_id: Order ID
            owner_id: Owner identifier
            bot_id: Bot ID
            exchange: Exchange name
            trading_pair: Trading pair
            side: Buy or sell
            base_asset: Base asset
            quote_asset: Quote asset
            base_amount: Base amount
            quote_amount: Quote amount
            price: Execution price
            fee_amount: Fee amount
            fee_asset: Fee asset
            modeled_cost: Modeled execution cost
            exchange_trade_id: Exchange's trade ID
            executed_at: Execution timestamp
            strategy_used: Strategy name

        Returns:
            Created trade record

        Note:
            Caller must commit the session.
        """
        # Create trade record
        trade = Trade(
            order_id=order_id,
            owner_id=owner_id,
            bot_id=bot_id,
            exchange=exchange,
            trading_pair=trading_pair,
            side=side,
            base_asset=base_asset,
            quote_asset=quote_asset,
            base_amount=base_amount,
            quote_amount=quote_amount,
            price=price,
            fee_amount=fee_amount,
            fee_asset=fee_asset,
            modeled_cost=modeled_cost,
            exchange_trade_id=exchange_trade_id,
            executed_at=executed_at or datetime.utcnow(),
            strategy_used=strategy_used,
        )

        self.session.add(trade)
        await self.session.flush()  # Get trade ID

        # Write ledger entries (double-entry accounting)
        await self.ledger_writer.write_trade_entries(
            owner_id=owner_id,
            bot_id=bot_id,
            trade_id=trade.id,
            order_id=order_id,
            side=side.value,
            base_asset=base_asset,
            quote_asset=quote_asset,
            base_amount=base_amount,
            quote_amount=quote_amount,
            fee_amount=fee_amount,
            modeled_cost=modeled_cost,
            description=f"{side.value.upper()} {base_amount:.8f} {base_asset} @ ${price:.2f}",
        )

        logger.info(
            f"Recorded trade {trade.id}: {side.value} {base_amount:.8f} {base_asset} "
            f"@ ${price:.2f} (order {order_id}, bot {bot_id})"
        )

        return trade


class FIFOTaxEngine:
    """FIFO tax engine for cost basis tracking and realized gain/loss calculation.

    CRITICAL: This engine provides deterministic, persisted tax lot matching.
    - BUY trades create tax lots
    - SELL trades consume lots in FIFO order
    - Lot consumption is persisted (not recomputed)
    - Realized gains are recorded immediately

    Design constraints:
    - FIFO matching is deterministic
    - All lot changes are persisted
    - Never compute tax on the fly
    """

    def __init__(self, session: AsyncSession):
        """Initialize FIFO tax engine.

        Args:
            session: Database session
        """
        self.session = session

    async def process_buy(
        self,
        trade: Trade,
    ) -> TaxLot:
        """Process a BUY trade - create a new tax lot.

        Args:
            trade: Buy trade

        Returns:
            Created tax lot

        Note:
            Caller must commit the session.
        """
        # Calculate unit cost (includes all costs)
        unit_cost = trade.get_cost_basis_per_unit()

        # Create tax lot
        lot = TaxLot(
            owner_id=trade.owner_id,
            asset=trade.base_asset,
            quantity_acquired=trade.base_amount,
            quantity_remaining=trade.base_amount,
            unit_cost=unit_cost,
            total_cost=trade.get_total_cost(),
            purchase_trade_id=trade.id,
            purchase_date=trade.executed_at,
            is_fully_consumed=False,
            created_at=datetime.utcnow(),
        )

        self.session.add(lot)

        logger.info(
            f"Created tax lot {lot.id}: {lot.quantity_acquired:.8f} {lot.asset} "
            f"@ ${lot.unit_cost:.2f}/unit (trade {trade.id})"
        )

        return lot

    async def process_sell(
        self,
        trade: Trade,
    ) -> List[RealizedGain]:
        """Process a SELL trade - consume tax lots in FIFO order.

        Args:
            trade: Sell trade

        Returns:
            List of realized gain records

        Note:
            Caller must commit the session.
        """
        # Get available tax lots (FIFO order)
        query = select(TaxLot).where(
            and_(
                TaxLot.owner_id == trade.owner_id,
                TaxLot.asset == trade.base_asset,
                TaxLot.quantity_remaining > 0,
            )
        ).order_by(TaxLot.purchase_date)

        result = await self.session.execute(query)
        lots = result.scalars().all()

        if not lots:
            logger.warning(
                f"No tax lots available for SELL trade {trade.id} "
                f"({trade.base_amount} {trade.base_asset}). "
                f"Creating zero-cost-basis lot."
            )
            # Create a zero-cost lot for tracking
            return []

        # Consume lots in FIFO order
        remaining_to_sell = trade.base_amount
        realized_gains = []

        for lot in lots:
            if remaining_to_sell <= 0:
                break

            # Calculate how much to consume from this lot
            consumed = lot.consume(remaining_to_sell, trade.executed_at)

            if consumed > 0:
                # Calculate proceeds for this portion
                proceeds = (consumed / trade.base_amount) * trade.quote_amount
                cost_basis = consumed * lot.unit_cost
                gain_loss = proceeds - cost_basis

                # Calculate holding period
                holding_period = (trade.executed_at - lot.purchase_date).days
                is_long_term = holding_period > 365

                # Record realized gain/loss
                gain = RealizedGain(
                    owner_id=trade.owner_id,
                    asset=trade.base_asset,
                    quantity=consumed,
                    proceeds=proceeds,
                    cost_basis=cost_basis,
                    gain_loss=gain_loss,
                    holding_period_days=holding_period,
                    is_long_term=is_long_term,
                    purchase_trade_id=lot.purchase_trade_id,
                    sell_trade_id=trade.id,
                    tax_lot_id=lot.id,
                    purchase_date=lot.purchase_date,
                    sell_date=trade.executed_at,
                    created_at=datetime.utcnow(),
                )

                self.session.add(gain)
                realized_gains.append(gain)

                logger.info(
                    f"Realized gain: {consumed:.8f} {trade.base_asset} "
                    f"gain/loss=${gain_loss:+.2f} "
                    f"({holding_period} days, {'LT' if is_long_term else 'ST'})"
                )

                remaining_to_sell -= consumed

        if remaining_to_sell > 1e-8:  # Floating point tolerance
            logger.warning(
                f"Insufficient tax lots for SELL trade {trade.id}: "
                f"{remaining_to_sell:.8f} {trade.base_asset} remaining"
            )

        return realized_gains


class CSVExportService:
    """Service for exporting ledger data to CSV files.

    CRITICAL: CSV files are EXPORTS, not primary storage.
    - SQLite is the authoritative source
    - CSV files can be regenerated at any time
    - Never write business logic directly to CSV

    Design constraints:
    - All CSV data comes from database queries
    - CSVs are best-effort (async, can fail)
    - Deleting CSVs does NOT lose data
    """

    def __init__(self, session: AsyncSession):
        """Initialize CSV export service.

        Args:
            session: Database session
        """
        self.session = session

    async def export_trades_csv(
        self,
        bot_id: int,
        output_path: Path,
        is_simulated: bool,
    ) -> None:
        """Export trades to CSV.

        Args:
            bot_id: Bot ID
            output_path: Output file path
            is_simulated: Filter by simulated flag (required to prevent mixing data)

        Raises:
            ValueError: If bot.is_dry_run doesn't match is_simulated
        """
        # Import here to avoid circular dependency
        from ..models import Bot, Order

        # Get bot to verify is_simulated matches
        bot_query = select(Bot).where(Bot.id == bot_id)
        bot_result = await self.session.execute(bot_query)
        bot = bot_result.scalar_one_or_none()

        if not bot:
            raise ValueError(f"Bot {bot_id} not found")

        if bot.is_dry_run != is_simulated:
            raise ValueError(
                f"Bot {bot_id} has is_dry_run={bot.is_dry_run}, "
                f"but requested is_simulated={is_simulated}"
            )

        # Query trades with is_simulated filter via Order join
        query = select(Trade).join(Order).where(
            Trade.bot_id == bot_id
        ).where(
            Order.is_simulated == is_simulated
        ).order_by(Trade.executed_at)
        result = await self.session.execute(query)
        trades = result.scalars().all()

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'trade_id', 'executed_at', 'side', 'trading_pair',
                'base_amount', 'quote_amount', 'price',
                'fee_amount', 'modeled_cost', 'total_cost',
                'strategy_used'
            ])

            for trade in trades:
                writer.writerow([
                    trade.id,
                    trade.executed_at.isoformat() if trade.executed_at else '',
                    trade.side.value,
                    trade.trading_pair,
                    trade.base_amount,
                    trade.quote_amount,
                    trade.price,
                    trade.fee_amount,
                    trade.modeled_cost,
                    trade.get_total_cost(),
                    trade.strategy_used or '',
                ])

        logger.info(f"Exported {len(trades)} trades to {output_path}")

    async def export_fiscal_csv(
        self,
        owner_id: str,
        year: int,
        output_path: Path,
    ) -> None:
        """Export fiscal/tax report to CSV.

        Args:
            owner_id: Owner identifier
            year: Tax year
            output_path: Output file path
        """
        # Query realized gains for year
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)

        query = select(RealizedGain).where(
            and_(
                RealizedGain.owner_id == owner_id,
                RealizedGain.sell_date >= start_date,
                RealizedGain.sell_date < end_date,
            )
        ).order_by(RealizedGain.sell_date)

        result = await self.session.execute(query)
        gains = result.scalars().all()

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'asset', 'quantity', 'purchase_date', 'purchase_price',
                'sell_date', 'sell_price', 'proceeds', 'cost_basis',
                'gain_loss', 'holding_period_days', 'term'
            ])

            for gain in gains:
                purchase_price = gain.cost_basis / gain.quantity if gain.quantity > 0 else 0
                sell_price = gain.proceeds / gain.quantity if gain.quantity > 0 else 0

                writer.writerow([
                    gain.asset,
                    gain.quantity,
                    gain.purchase_date.isoformat() if gain.purchase_date else '',
                    purchase_price,
                    gain.sell_date.isoformat() if gain.sell_date else '',
                    sell_price,
                    gain.proceeds,
                    gain.cost_basis,
                    gain.gain_loss,
                    gain.holding_period_days,
                    'Long-term' if gain.is_long_term else 'Short-term',
                ])

        logger.info(f"Exported {len(gains)} realized gains to {output_path}")
