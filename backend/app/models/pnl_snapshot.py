"""P&L snapshot model for historical tracking."""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class PnLSnapshot(Base):
    """P&L snapshot model for historical P&L tracking."""
    __tablename__ = "pnl_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=True)  # null for global

    total_pnl = Column(Float, nullable=False)

    # Timestamp
    snapshot_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bot = relationship("Bot", back_populates="pnl_snapshots")

    def __repr__(self):
        return f"<PnLSnapshot(id={self.id}, pnl={self.total_pnl})>"
