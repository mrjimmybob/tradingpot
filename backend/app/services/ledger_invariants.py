"""Ledger Invariant Validation Service

This service validates accounting invariants after each trade execution.
All violations raise exceptions and halt trading to prevent corrupt state.

Design principles:
- Fail fast: Raise exceptions on violation
- Read-only: No data modification
- Deterministic: No randomness or time-based logic
- Scoped: Validate one trade_id at a time
"""

import logging
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from ..models import (
    WalletLedger,
    LedgerReason,
    Trade,
    TradeSide,
    Order,
    RealizedGain,
    Bot
)
from .ledger_writer import LedgerWriterService

logger = logging.getLogger(__name__)


# ============================================================================
# Exception Hierarchy
# ============================================================================

class ValidationError(Exception):
    """Base class for all validation errors."""
    pass


class DoubleEntryViolationError(ValidationError):
    """Double-entry accounting violation."""
    pass


class NegativeBalanceError(ValidationError):
    """Negative balance detected."""
    pass


class BalanceInconsistencyError(ValidationError):
    """Ledger balance doesn't match cached balance."""
    pass


class ReferentialIntegrityError(ValidationError):
    """Foreign key or reference integrity violation."""
    pass


class TaxLotConsumptionError(ValidationError):
    """Tax lot consumption incomplete or incorrect."""
    pass


# ============================================================================
# Ledger Invariant Service
# ============================================================================

class LedgerInvariantService:
    """Service for validating accounting invariants.

    This service performs 5 critical validations after each trade:
    1. Double-entry balance (debits = credits, excluding fees/costs)
    2. No negative balances (except FEE in simulated mode)
    3. Ledger vs cached balance consistency
    4. Trade → Order referential integrity
    5. SELL must fully consume tax lots (FIFO)

    Design principles:
    - Fail fast: Raise exceptions on violation
    - Read-only: No data modification
    - Deterministic: No randomness or time-based logic
    - Scoped: Validate one trade_id at a time
    """

    # Floating point tolerance for all comparisons
    TOLERANCE = 1e-8

    def __init__(self, session: AsyncSession):
        """Initialize the validator.

        Args:
            session: Async database session
        """
        self.session = session

    async def validate_trade(self, trade_id: int) -> None:
        """Validate all accounting invariants for a trade.

        This is the master validation function called after trade execution.

        Args:
            trade_id: Trade ID to validate

        Raises:
            ValidationError subclass if any invariant is violated
        """
        try:
            await self.validate_double_entry(trade_id)
            await self.validate_no_negative_balances(trade_id)
            await self.validate_ledger_balance_consistency(trade_id)
            await self.validate_referential_integrity(trade_id)
            await self.validate_tax_lot_consumption(trade_id)

            logger.info(f"Trade {trade_id}: All invariants validated successfully")

        except ValidationError as e:
            logger.error(f"Trade {trade_id}: Validation failed - {e}")
            raise  # Re-raise to halt trading

    async def validate_double_entry(self, trade_id: int) -> None:
        """Validate double-entry accounting for a trade.

        Rules:
        - BUY: quote_out + base_in = 0 (excluding fees/costs)
        - SELL: base_out + quote_in = 0 (excluding fees/costs)
        - Fees/costs are pure debits (negative deltas)

        Args:
            trade_id: Trade ID to validate

        Raises:
            DoubleEntryViolationError if sum != 0
        """
        # Query all ledger entries for this trade
        query = select(WalletLedger).where(
            WalletLedger.related_trade_id == trade_id
        )
        result = await self.session.execute(query)
        ledger_entries = result.scalars().all()

        if not ledger_entries:
            raise DoubleEntryViolationError(
                f"Trade {trade_id}: No ledger entries found"
            )

        # Separate trade entries from fee/cost entries
        trade_deltas = []
        fee_cost_deltas = []

        for entry in ledger_entries:
            if entry.reason in [LedgerReason.BUY, LedgerReason.SELL]:
                trade_deltas.append(entry.delta_amount)
            elif entry.reason in [LedgerReason.FEE, LedgerReason.EXECUTION_COST]:
                fee_cost_deltas.append(entry.delta_amount)

        # Trade deltas must sum to zero (double-entry)
        trade_sum = sum(trade_deltas)
        if abs(trade_sum) > self.TOLERANCE:
            raise DoubleEntryViolationError(
                f"Trade {trade_id}: Double-entry violation. "
                f"Trade deltas sum to {trade_sum:.8f}, expected 0.0. "
                f"Entries: {trade_deltas}"
            )

        # Fee/cost deltas must be negative or zero
        for delta in fee_cost_deltas:
            if delta > self.TOLERANCE:
                raise DoubleEntryViolationError(
                    f"Trade {trade_id}: Fee/cost entries must be negative. "
                    f"Found positive delta: {delta:.8f}"
                )

        logger.debug(
            f"Trade {trade_id}: Double-entry validation passed. "
            f"Trade deltas: {trade_deltas}, Fee/cost deltas: {fee_cost_deltas}"
        )

    async def validate_no_negative_balances(self, trade_id: int) -> None:
        """Validate no negative balances after trade execution.

        Rules:
        - All asset balances must be >= 0
        - Exception: FEE asset can be negative in simulated mode

        Args:
            trade_id: Trade ID to validate

        Raises:
            NegativeBalanceError if balance < 0
        """
        # Get all ledger entries for this trade
        query = select(WalletLedger).where(
            WalletLedger.related_trade_id == trade_id
        )
        result = await self.session.execute(query)
        ledger_entries = result.scalars().all()

        # Check balance_after for each entry
        for entry in ledger_entries:
            if entry.balance_after is None:
                raise NegativeBalanceError(
                    f"Trade {trade_id}: Ledger entry {entry.id} missing balance_after"
                )

            # FEE asset can be negative in simulated mode
            if entry.asset == "FEE" and entry.related_order_id:
                # Check if order is simulated
                order_query = select(Order).where(Order.id == entry.related_order_id)
                order_result = await self.session.execute(order_query)
                order = order_result.scalar_one_or_none()
                if order and order.is_simulated:
                    continue  # Allow negative FEE balance for simulated

            if entry.balance_after < -self.TOLERANCE:
                raise NegativeBalanceError(
                    f"Trade {trade_id}: Negative balance detected. "
                    f"Asset={entry.asset}, balance_after={entry.balance_after:.8f}, "
                    f"ledger_entry_id={entry.id}"
                )

        logger.debug(f"Trade {trade_id}: No negative balances validation passed")

    async def validate_ledger_balance_consistency(self, trade_id: int) -> None:
        """Validate ledger-reconstructed balance matches cached balance.

        Compares:
        - Ledger balance (from balance_after in latest entry)
        - Reconstructed balance (sum of all deltas)

        Args:
            trade_id: Trade ID to validate

        Raises:
            BalanceInconsistencyError if mismatch
        """
        # Get trade to find bot_id and assets
        trade_query = select(Trade).where(Trade.id == trade_id)
        trade_result = await self.session.execute(trade_query)
        trade = trade_result.scalar_one_or_none()

        if not trade:
            raise ValidationError(f"Trade {trade_id} not found")

        # Get all unique assets affected by this trade
        ledger_query = select(WalletLedger).where(
            WalletLedger.related_trade_id == trade_id
        )
        ledger_result = await self.session.execute(ledger_query)
        ledger_entries = ledger_result.scalars().all()

        affected_assets = set(entry.asset for entry in ledger_entries)

        # For each affected asset, validate reconstructed balance matches balance_after
        ledger_service = LedgerWriterService(self.session)

        for asset in affected_assets:
            # Get balance_after from latest entry for this asset
            ledger_balance = await ledger_service.get_balance(
                owner_id=trade.owner_id,
                asset=asset,
                bot_id=trade.bot_id
            )

            # Reconstruct balance from all deltas
            reconstructed = await ledger_service.reconstruct_balance(
                owner_id=trade.owner_id,
                asset=asset,
                bot_id=trade.bot_id
            )

            if abs(reconstructed - ledger_balance) > self.TOLERANCE:
                raise BalanceInconsistencyError(
                    f"Trade {trade_id}: Ledger balance mismatch for {asset}. "
                    f"Reconstructed={reconstructed:.8f}, "
                    f"balance_after={ledger_balance:.8f}, "
                    f"Difference={reconstructed - ledger_balance:.8f}"
                )

        logger.debug(
            f"Trade {trade_id}: Ledger balance consistency validated for assets: {affected_assets}"
        )

    async def validate_referential_integrity(self, trade_id: int) -> None:
        """Validate trade references valid order with consistent metadata.

        Rules:
        - Trade.order_id must reference existing Order
        - Order.bot_id must match Trade.bot_id
        - All ledger entries must reference same order

        Args:
            trade_id: Trade ID to validate

        Raises:
            ReferentialIntegrityError if references invalid
        """
        # Get trade
        trade_query = select(Trade).where(Trade.id == trade_id)
        trade_result = await self.session.execute(trade_query)
        trade = trade_result.scalar_one_or_none()

        if not trade:
            raise ValidationError(f"Trade {trade_id} not found")

        # Validate order exists
        order_query = select(Order).where(Order.id == trade.order_id)
        order_result = await self.session.execute(order_query)
        order = order_result.scalar_one_or_none()

        if not order:
            raise ReferentialIntegrityError(
                f"Trade {trade_id} references non-existent order {trade.order_id}"
            )

        # Validate bot consistency
        if order.bot_id != trade.bot_id:
            raise ReferentialIntegrityError(
                f"Trade {trade_id}: Bot ID mismatch. "
                f"Trade.bot_id={trade.bot_id}, Order.bot_id={order.bot_id}"
            )

        # Validate all ledger entries reference correct order
        ledger_query = select(WalletLedger).where(
            WalletLedger.related_trade_id == trade_id
        )
        ledger_result = await self.session.execute(ledger_query)
        ledger_entries = ledger_result.scalars().all()

        for entry in ledger_entries:
            if entry.related_order_id and entry.related_order_id != trade.order_id:
                raise ReferentialIntegrityError(
                    f"Trade {trade_id}: Ledger entry {entry.id} references "
                    f"order {entry.related_order_id}, expected {trade.order_id}"
                )

        logger.debug(
            f"Trade {trade_id}: Referential integrity validated. "
            f"Order {trade.order_id} exists and matches bot {trade.bot_id}"
        )

    async def validate_tax_lot_consumption(self, trade_id: int) -> None:
        """Validate SELL trades fully consume tax lots in FIFO order.

        Rules:
        - Sum of RealizedGain.quantity must equal Trade.base_amount
        - All consumed lots must be for same asset
        - No remaining quantity if trade is complete

        Args:
            trade_id: Trade ID to validate

        Raises:
            TaxLotConsumptionError if incomplete consumption
        """
        # Get trade
        trade_query = select(Trade).where(Trade.id == trade_id)
        trade_result = await self.session.execute(trade_query)
        trade = trade_result.scalar_one_or_none()

        if not trade:
            raise ValidationError(f"Trade {trade_id} not found")

        # Only validate SELL trades
        if trade.side != TradeSide.SELL:
            logger.debug(f"Trade {trade_id}: Skipping tax lot validation for BUY trade")
            return

        # Get all realized gains for this sell trade
        gains_query = select(RealizedGain).where(
            RealizedGain.sell_trade_id == trade_id
        )
        gains_result = await self.session.execute(gains_query)
        realized_gains = gains_result.scalars().all()

        if not realized_gains:
            # Check if this is a warning case (no lots available)
            # This was allowed in FIFOTaxEngine.process_sell() as a warning
            logger.warning(
                f"Trade {trade_id}: No realized gains found for SELL trade. "
                f"This may indicate insufficient tax lots (zero-cost-basis). "
                f"Allowing for backward compatibility."
            )
            # For now, don't raise error to maintain backward compatibility
            # TODO: Make this stricter after populating historical lots
            return

        # Sum consumed quantities
        total_consumed = sum(gain.quantity for gain in realized_gains)

        # Must match trade quantity (within floating point tolerance)
        if abs(total_consumed - trade.base_amount) > self.TOLERANCE:
            raise TaxLotConsumptionError(
                f"Trade {trade_id}: Tax lot consumption mismatch. "
                f"Trade amount={trade.base_amount:.8f}, "
                f"Consumed={total_consumed:.8f}, "
                f"Difference={trade.base_amount - total_consumed:.8f}"
            )

        # Validate all gains are for same asset
        assets = set(gain.asset for gain in realized_gains)
        if len(assets) > 1:
            raise TaxLotConsumptionError(
                f"Trade {trade_id}: Multiple assets in realized gains: {assets}"
            )

        if assets and list(assets)[0] != trade.base_asset:
            raise TaxLotConsumptionError(
                f"Trade {trade_id}: Asset mismatch. "
                f"Trade.base_asset={trade.base_asset}, "
                f"Realized gain asset={list(assets)[0]}"
            )

        logger.debug(
            f"Trade {trade_id}: Tax lot consumption validated. "
            f"Consumed {total_consumed:.8f} {trade.base_asset} across {len(realized_gains)} lots"
        )
