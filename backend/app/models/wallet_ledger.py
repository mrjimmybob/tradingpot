"""Wallet ledger model - AUTHORITATIVE source for all balance changes.

CRITICAL: This is the single source of truth for all financial transactions.
- Every balance change MUST create ledger entries
- APPEND-ONLY: Never mutate historical records
- Corrections are new entries, not edits
- All balances must be reconstructable from this ledger

Design constraints:
- Event-based, not state-based
- Double-entry accounting (buys/sells create two rows)
- Deterministic and auditable
- Supports multi-asset portfolios
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship

from .database import Base


class LedgerReason(str, Enum):
    """Reason for ledger entry - tracks transaction type."""
    ALLOCATION = "allocation"           # Initial bot funding
    DEALLOCATION = "deallocation"       # Bot shutdown/withdrawal
    BUY = "buy"                         # Asset purchase
    SELL = "sell"                       # Asset sale
    FEE = "fee"                         # Exchange/transaction fee
    EXECUTION_COST = "execution_cost"   # Modeled execution costs
    TRANSFER = "transfer"               # Inter-bot transfer
    CORRECTION = "correction"           # Manual correction (append-only)
    REALIZED_GAIN = "realized_gain"     # P&L from sale
    REALIZED_LOSS = "realized_loss"     # P&L from sale


class WalletLedger(Base):
    """Append-only ledger of all balance changes.

    AUTHORITATIVE SOURCE: All balances, P&L, and positions must be
    reconstructable from this ledger.

    Every financial event creates one or more ledger entries:
    - BUY: Two entries (quote currency -, base currency +)
    - SELL: Two entries (base currency -, quote currency +)
    - FEE: One entry (quote currency -)
    - ALLOCATION: One entry per asset (asset +)

    Example BUY BTC with USDT:
        Entry 1: asset="USDT", delta=-1000, reason=BUY
        Entry 2: asset="BTC", delta=+0.05, reason=BUY
        Entry 3: asset="USDT", delta=-1.5, reason=FEE
    """
    __tablename__ = "wallet_ledger"

    id = Column(Integer, primary_key=True, index=True)

    # Owner and bot identification
    owner_id = Column(String(100), nullable=False, index=True)  # TODO: Add FK when User model exists
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=True, index=True)

    # Asset and amount
    asset = Column(String(10), nullable=False, index=True)  # e.g., "USDT", "BTC", "ETH"
    delta_amount = Column(Float, nullable=False)  # Signed: + for credit, - for debit
    balance_after = Column(Float, nullable=True)  # Balance after this entry (for validation)

    # Transaction metadata
    reason = Column(SQLEnum(LedgerReason), nullable=False, index=True)
    description = Column(String(500), nullable=True)  # Human-readable description

    # Foreign keys for traceability
    related_order_id = Column(Integer, ForeignKey("orders.id"), nullable=True, index=True)
    related_trade_id = Column(Integer, nullable=True, index=True)  # FK to trades table

    # Timestamp (append-only, immutable)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    bot = relationship("Bot", back_populates="ledger_entries")
    order = relationship("Order", back_populates="ledger_entries")

    def __repr__(self):
        return (
            f"<WalletLedger(id={self.id}, "
            f"asset={self.asset}, "
            f"delta={self.delta_amount:+.8f}, "
            f"reason={self.reason.value})>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "bot_id": self.bot_id,
            "asset": self.asset,
            "delta_amount": self.delta_amount,
            "balance_after": self.balance_after,
            "reason": self.reason.value,
            "description": self.description,
            "related_order_id": self.related_order_id,
            "related_trade_id": self.related_trade_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# IMPORTANT: Ledger validation rules
# 1. Every entry must have a reason
# 2. Delta amounts must be non-zero
# 3. Related IDs should be set when applicable
# 4. Timestamps are immutable
# 5. Balance reconstruction should match balance_after
