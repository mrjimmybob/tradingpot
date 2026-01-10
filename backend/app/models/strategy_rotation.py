"""Strategy rotation model for tracking strategy changes."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from .database import Base


class StrategyRotation(Base):
    """Strategy rotation log model."""
    __tablename__ = "strategy_rotations"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)

    from_strategy = Column(String(50), nullable=False)
    to_strategy = Column(String(50), nullable=False)
    reason = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bot = relationship("Bot", back_populates="strategy_rotations")

    def __repr__(self):
        return f"<StrategyRotation(id={self.id}, from={self.from_strategy}, to={self.to_strategy})>"
