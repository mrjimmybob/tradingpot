"""Ledger writer service - records all balance changes to wallet_ledger.

CRITICAL: This is the ONLY way to modify balances in the system.
- All balance changes MUST go through this service
- Direct balance modifications are FORBIDDEN
- Append-only: never mutate historical records
- Corrections are new entries, not edits

Design constraints:
- Deterministic (no randomness)
- Double-entry accounting for trades
- Every entry has a reason and audit trail
"""

import logging
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from ..models import WalletLedger, LedgerReason, Bot

logger = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    """Single ledger entry to be written."""
    owner_id: str
    bot_id: Optional[int]
    asset: str
    delta_amount: float
    reason: LedgerReason
    description: Optional[str] = None
    related_order_id: Optional[int] = None
    related_trade_id: Optional[int] = None


class LedgerWriterService:
    """Service for writing to the wallet ledger.

    CRITICAL: This is the ONLY authorized way to modify balances.
    All financial transactions must go through this service.

    Responsibilities:
    - Record all balance changes
    - Maintain balance_after for validation
    - Log all entries at INFO level
    - Ensure deterministic writes
    """

    def __init__(self, session: AsyncSession):
        """Initialize ledger writer service.

        Args:
            session: Database session
        """
        self.session = session

    async def write_entry(
        self,
        owner_id: str,
        asset: str,
        delta_amount: float,
        reason: LedgerReason,
        bot_id: Optional[int] = None,
        description: Optional[str] = None,
        related_order_id: Optional[int] = None,
        related_trade_id: Optional[int] = None,
    ) -> WalletLedger:
        """Write a single ledger entry.

        Args:
            owner_id: Owner identifier
            asset: Asset symbol (e.g., "USDT", "BTC")
            delta_amount: Change in balance (signed)
            reason: Reason for change
            bot_id: Bot ID (optional)
            description: Human-readable description
            related_order_id: Related order ID
            related_trade_id: Related trade ID

        Returns:
            Created ledger entry

        Note:
            Caller must commit the session.
        """
        # Calculate balance after this entry
        balance_after = await self._calculate_balance_after(
            owner_id, bot_id, asset, delta_amount
        )

        # Create ledger entry
        entry = WalletLedger(
            owner_id=owner_id,
            bot_id=bot_id,
            asset=asset,
            delta_amount=delta_amount,
            balance_after=balance_after,
            reason=reason,
            description=description,
            related_order_id=related_order_id,
            related_trade_id=related_trade_id,
            created_at=datetime.utcnow(),
        )

        self.session.add(entry)

        # Log at INFO level (financial event)
        logger.info(
            f"Ledger entry: owner={owner_id}, bot={bot_id}, "
            f"asset={asset}, delta={delta_amount:+.8f}, "
            f"balance_after={balance_after:.8f}, reason={reason.value}, "
            f"description={description}"
        )

        return entry

    async def write_trade_entries(
        self,
        owner_id: str,
        bot_id: int,
        trade_id: int,
        order_id: int,
        side: str,  # "buy" or "sell"
        base_asset: str,
        quote_asset: str,
        base_amount: float,
        quote_amount: float,
        fee_amount: float,
        modeled_cost: float,
        description: str,
    ) -> List[WalletLedger]:
        """Write ledger entries for a trade (double-entry accounting).

        BUY creates:
        - Quote currency debit (-)
        - Base currency credit (+)
        - Fee debit (-)
        - Cost debit (-)

        SELL creates:
        - Base currency debit (-)
        - Quote currency credit (+)
        - Fee debit (-)
        - Cost debit (-)

        Args:
            owner_id: Owner identifier
            bot_id: Bot ID
            trade_id: Trade ID
            order_id: Order ID
            side: "buy" or "sell"
            base_asset: Base asset symbol
            quote_asset: Quote asset symbol
            base_amount: Base asset amount
            quote_amount: Quote asset amount
            fee_amount: Fee amount (in quote currency)
            modeled_cost: Modeled execution cost
            description: Description

        Returns:
            List of created ledger entries

        Note:
            Caller must commit the session.
        """
        entries = []

        if side == "buy":
            # BUY: Pay quote currency, receive base currency
            # 1. Quote currency out
            entry1 = await self.write_entry(
                owner_id=owner_id,
                bot_id=bot_id,
                asset=quote_asset,
                delta_amount=-quote_amount,
                reason=LedgerReason.BUY,
                description=f"{description} - quote payment",
                related_order_id=order_id,
                related_trade_id=trade_id,
            )
            entries.append(entry1)

            # 2. Base currency in
            entry2 = await self.write_entry(
                owner_id=owner_id,
                bot_id=bot_id,
                asset=base_asset,
                delta_amount=base_amount,
                reason=LedgerReason.BUY,
                description=f"{description} - base received",
                related_order_id=order_id,
                related_trade_id=trade_id,
            )
            entries.append(entry2)

        else:  # sell
            # SELL: Pay base currency, receive quote currency
            # 1. Base currency out
            entry1 = await self.write_entry(
                owner_id=owner_id,
                bot_id=bot_id,
                asset=base_asset,
                delta_amount=-base_amount,
                reason=LedgerReason.SELL,
                description=f"{description} - base sold",
                related_order_id=order_id,
                related_trade_id=trade_id,
            )
            entries.append(entry1)

            # 2. Quote currency in
            entry2 = await self.write_entry(
                owner_id=owner_id,
                bot_id=bot_id,
                asset=quote_asset,
                delta_amount=quote_amount,
                reason=LedgerReason.SELL,
                description=f"{description} - quote received",
                related_order_id=order_id,
                related_trade_id=trade_id,
            )
            entries.append(entry2)

        # 3. Fee (always in quote currency)
        if fee_amount > 0:
            entry3 = await self.write_entry(
                owner_id=owner_id,
                bot_id=bot_id,
                asset=quote_asset,
                delta_amount=-fee_amount,
                reason=LedgerReason.FEE,
                description=f"{description} - exchange fee",
                related_order_id=order_id,
                related_trade_id=trade_id,
            )
            entries.append(entry3)

        # 4. Modeled execution cost
        if modeled_cost > 0:
            entry4 = await self.write_entry(
                owner_id=owner_id,
                bot_id=bot_id,
                asset=quote_asset,
                delta_amount=-modeled_cost,
                reason=LedgerReason.EXECUTION_COST,
                description=f"{description} - execution cost",
                related_order_id=order_id,
                related_trade_id=trade_id,
            )
            entries.append(entry4)

        logger.info(
            f"Recorded {len(entries)} ledger entries for trade {trade_id} "
            f"({side} {base_amount} {base_asset} @ {quote_amount / base_amount if base_amount > 0 else 0:.2f})"
        )

        return entries

    async def _calculate_balance_after(
        self,
        owner_id: str,
        bot_id: Optional[int],
        asset: str,
        delta_amount: float,
    ) -> float:
        """Calculate balance after applying delta.

        Args:
            owner_id: Owner identifier
            bot_id: Bot ID (None for owner-level)
            asset: Asset symbol
            delta_amount: Change amount

        Returns:
            New balance
        """
        # Get current balance from last ledger entry
        query = select(WalletLedger).where(
            and_(
                WalletLedger.owner_id == owner_id,
                WalletLedger.asset == asset,
            )
        )

        if bot_id is not None:
            query = query.where(WalletLedger.bot_id == bot_id)

        query = query.order_by(WalletLedger.id.desc()).limit(1)

        result = await self.session.execute(query)
        last_entry = result.scalar_one_or_none()

        if last_entry and last_entry.balance_after is not None:
            current_balance = last_entry.balance_after
        else:
            current_balance = 0.0

        return current_balance + delta_amount

    async def get_balance(
        self,
        owner_id: str,
        asset: str,
        bot_id: Optional[int] = None,
    ) -> float:
        """Get current balance for an asset.

        Args:
            owner_id: Owner identifier
            asset: Asset symbol
            bot_id: Bot ID (optional)

        Returns:
            Current balance
        """
        query = select(WalletLedger).where(
            and_(
                WalletLedger.owner_id == owner_id,
                WalletLedger.asset == asset,
            )
        )

        if bot_id is not None:
            query = query.where(WalletLedger.bot_id == bot_id)

        query = query.order_by(WalletLedger.id.desc()).limit(1)

        result = await self.session.execute(query)
        last_entry = result.scalar_one_or_none()

        if last_entry and last_entry.balance_after is not None:
            return last_entry.balance_after

        return 0.0

    async def reconstruct_balance(
        self,
        owner_id: str,
        asset: str,
        bot_id: Optional[int] = None,
        up_to_entry_id: Optional[int] = None,
    ) -> float:
        """Reconstruct balance from ledger entries.

        This validates that balance_after values are correct.

        Args:
            owner_id: Owner identifier
            asset: Asset symbol
            bot_id: Bot ID (optional)
            up_to_entry_id: Stop at this entry ID (optional)

        Returns:
            Reconstructed balance
        """
        query = select(WalletLedger).where(
            and_(
                WalletLedger.owner_id == owner_id,
                WalletLedger.asset == asset,
            )
        )

        if bot_id is not None:
            query = query.where(WalletLedger.bot_id == bot_id)

        if up_to_entry_id is not None:
            query = query.where(WalletLedger.id <= up_to_entry_id)

        query = query.order_by(WalletLedger.id)

        result = await self.session.execute(query)
        entries = result.scalars().all()

        balance = 0.0
        for entry in entries:
            balance += entry.delta_amount

        return balance
