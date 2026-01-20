"""Tax lot model - FIFO cost basis tracking for tax reporting.

CRITICAL: Tax lots track the cost basis of acquired assets using FIFO (First-In-First-Out).
- BUY trades create tax lots
- SELL trades consume lots in FIFO order
- Lot consumption is deterministic and persisted
- NEVER compute tax on the fly - always use persisted lots

Design constraints:
- FIFO matching is deterministic
- Partial lot consumption is tracked
- Lot consumption history is immutable
- Supports multi-asset portfolios
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from .database import Base


class TaxLot(Base):
    """Tax lot for FIFO cost basis tracking.

    Tax lots represent acquired assets with their cost basis.
    When assets are sold, lots are consumed in FIFO order.

    Example workflow:
        BUY 1: 0.5 BTC @ $40,000 → Lot A (0.5 BTC remaining)
        BUY 2: 0.3 BTC @ $41,000 → Lot B (0.3 BTC remaining)
        SELL 1: 0.6 BTC @ $42,000 →
            Lot A: 0.5 BTC consumed (0 remaining)
            Lot B: 0.1 BTC consumed (0.2 remaining)

    This enables accurate tax reporting with proper cost basis matching.
    """
    __tablename__ = "tax_lots"

    id = Column(Integer, primary_key=True, index=True)

    # Owner and asset
    owner_id = Column(String(100), nullable=False, index=True)
    asset = Column(String(10), nullable=False, index=True)  # e.g., "BTC", "ETH"

    # Lot details
    quantity_acquired = Column(Float, nullable=False)       # Original quantity
    quantity_remaining = Column(Float, nullable=False)      # Remaining after sales
    unit_cost = Column(Float, nullable=False)               # Cost per unit (including fees)
    total_cost = Column(Float, nullable=False)              # Total acquisition cost

    # Purchase trade reference
    purchase_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    purchase_date = Column(DateTime, nullable=False, index=True)

    # Lot status
    is_fully_consumed = Column(Boolean, default=False, index=True)
    consumed_at = Column(DateTime, nullable=True)  # When lot was fully consumed

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    purchase_trade = relationship("Trade", foreign_keys=[purchase_trade_id])

    def __repr__(self):
        return (
            f"<TaxLot(id={self.id}, "
            f"asset={self.asset}, "
            f"remaining={self.quantity_remaining:.8f}, "
            f"cost=${self.unit_cost:.2f})>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "asset": self.asset,
            "quantity_acquired": self.quantity_acquired,
            "quantity_remaining": self.quantity_remaining,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "purchase_trade_id": self.purchase_trade_id,
            "purchase_date": self.purchase_date.isoformat() if self.purchase_date else None,
            "is_fully_consumed": self.is_fully_consumed,
            "consumed_at": self.consumed_at.isoformat() if self.consumed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def consume(self, quantity: float, consumed_at: datetime) -> float:
        """Consume quantity from this lot (FIFO).

        Args:
            quantity: Amount to consume
            consumed_at: Timestamp of consumption

        Returns:
            Amount actually consumed (may be less if lot doesn't have enough)

        Note:
            This method updates quantity_remaining but does NOT commit.
            Caller must commit the session.
        """
        consumed = min(quantity, self.quantity_remaining)
        self.quantity_remaining -= consumed

        if self.quantity_remaining <= 1e-8:  # Floating point tolerance
            self.quantity_remaining = 0.0
            self.is_fully_consumed = True
            self.consumed_at = consumed_at

        return consumed


class RealizedGain(Base):
    """Realized gain/loss record for tax reporting.

    CRITICAL: This is the AUTHORITATIVE source for tax reporting.
    Never compute tax on the fly - always use these persisted records.

    Each row represents a matched buy-sell pair with gain/loss calculation.
    """
    __tablename__ = "realized_gains"

    id = Column(Integer, primary_key=True, index=True)

    # Owner and asset
    owner_id = Column(String(100), nullable=False, index=True)
    asset = Column(String(10), nullable=False, index=True)

    # Transaction details
    quantity = Column(Float, nullable=False)                # Quantity sold
    proceeds = Column(Float, nullable=False)                # Sale proceeds (before fees)
    cost_basis = Column(Float, nullable=False)              # Original cost basis
    gain_loss = Column(Float, nullable=False, index=True)   # Net gain/loss

    # Holding period
    holding_period_days = Column(Integer, nullable=False)   # Days held
    is_long_term = Column(Boolean, nullable=False)          # True if > 365 days

    # Trade references
    purchase_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False)
    sell_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    tax_lot_id = Column(Integer, ForeignKey("tax_lots.id"), nullable=False)

    # Dates
    purchase_date = Column(DateTime, nullable=False)
    sell_date = Column(DateTime, nullable=False, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    purchase_trade = relationship("Trade", foreign_keys=[purchase_trade_id])
    sell_trade = relationship("Trade", foreign_keys=[sell_trade_id])
    tax_lot = relationship("TaxLot", foreign_keys=[tax_lot_id])

    def __repr__(self):
        return (
            f"<RealizedGain(id={self.id}, "
            f"asset={self.asset}, "
            f"quantity={self.quantity:.8f}, "
            f"gain_loss=${self.gain_loss:+.2f})>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "asset": self.asset,
            "quantity": self.quantity,
            "proceeds": self.proceeds,
            "cost_basis": self.cost_basis,
            "gain_loss": self.gain_loss,
            "holding_period_days": self.holding_period_days,
            "is_long_term": self.is_long_term,
            "purchase_trade_id": self.purchase_trade_id,
            "sell_trade_id": self.sell_trade_id,
            "tax_lot_id": self.tax_lot_id,
            "purchase_date": self.purchase_date.isoformat() if self.purchase_date else None,
            "sell_date": self.sell_date.isoformat() if self.sell_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
