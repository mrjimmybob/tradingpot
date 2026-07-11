"""Order model for tracking all orders."""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Enum as SQLEnum
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
    limit_price = Column(Float, nullable=True)  # For limit orders
    fees = Column(Float, default=0.0)

    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING)
    strategy_used = Column(String(50), nullable=False)
    running_balance_after = Column(Float, nullable=True)
    reason = Column(String(500), nullable=True)  # Trade/rejection reason

    # Execution cost modeling (nullable, backward compatible)
    modeled_exchange_fee = Column(Float, nullable=True)
    modeled_spread_cost = Column(Float, nullable=True)
    modeled_slippage_cost = Column(Float, nullable=True)
    modeled_total_cost = Column(Float, nullable=True)
    realized_total_cost = Column(Float, nullable=True)  # Future-proof for actual cost tracking

    # Dry run flag
    is_simulated = Column(Boolean, default=False)

    # Structured decision explanation (add-strategy-decision-framework,
    # Phase 0.6): a bounded summary of the ExplanationBuilder/
    # DecisionExplanation that produced this order, persisted once at
    # execution time so a historical trade is explainable from the DB, not
    # only from the in-memory, current-state-only DiagnosticsStore. See
    # app/services/strategy_framework/explanation_persistence.py. Nullable -
    # absent for orders with no strategy decision behind them (e.g. a
    # recovered/imported untracked exchange fill).
    decision_explanation = Column(JSON, nullable=True)
    # Active Strategy Edge Management category (A/B/C) at execution time,
    # if the strategy that produced this order has been migrated to
    # StrategyEdgeManager (Phase 1-6). Null for every strategy prior to its
    # own migration, and for orders with no active degradation classified.
    edge_management_category = Column(String(20), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)

    # Relationships
    bot = relationship("Bot", back_populates="orders")
    trades = relationship("Trade", back_populates="order", cascade="all, delete-orphan")
    ledger_entries = relationship("WalletLedger", back_populates="order")

    def __repr__(self):
        return f"<Order(id={self.id}, type={self.order_type.value}, status={self.status.value})>"
