"""Position model for tracking open positions."""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship

from .database import Base


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class Position(Base):
    """Open position model."""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)

    trading_pair = Column(String(50), nullable=False)
    side = Column(SQLEnum(PositionSide), nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)

    # Ownership: which strategy opened this position. Until this position
    # closes, Auto Mode MUST dispatch only to this strategy - see
    # auto-mode-position-ownership. NULL means "unowned" (e.g. a position that
    # predates this column), which is treated as "do not force-switch away
    # from whatever strategy is currently selected", never as an error.
    owning_strategy = Column(String(50), nullable=True)
    entry_reason = Column(String(500), nullable=True)
    # Minimal strategy state needed to make exit decisions for this specific
    # position (e.g. locked entry ATR, trailing-stop anchor) - NOT the full
    # per-bot strategy state blob (that stays in Bot.strategy_state).
    entry_strategy_state = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    bot = relationship("Bot", back_populates="positions")

    def __repr__(self):
        return f"<Position(id={self.id}, pair={self.trading_pair}, side={self.side.value})>"

    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L based on current price."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.amount
        else:  # SHORT
            return (self.entry_price - self.current_price) * self.amount
