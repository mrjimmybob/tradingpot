"""Order model for tracking all orders."""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship

from .database import Base


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Order(Base):
    """Order model for tracking trades."""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    exchange_order_id = Column(String(100), nullable=True)

    order_type = Column(SQLEnum(OrderType), nullable=False)
    trading_pair = Column(String(50), nullable=False)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)

    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING)
    strategy_used = Column(String(50), nullable=False)
    running_balance_after = Column(Float, nullable=True)

    # Dry run flag
    is_simulated = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)

    # Relationships
    bot = relationship("Bot", back_populates="orders")

    def __repr__(self):
        return f"<Order(id={self.id}, type={self.order_type.value}, status={self.status.value})>"
