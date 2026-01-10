"""Alert model for logging alerts."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from .database import Base


class Alert(Base):
    """Alert log model."""
    __tablename__ = "alerts_log"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=True)  # null for global alerts

    alert_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    email_sent = Column(Boolean, default=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    bot = relationship("Bot", back_populates="alerts")

    def __repr__(self):
        return f"<Alert(id={self.id}, type={self.alert_type})>"
