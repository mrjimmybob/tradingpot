"""Trade model - execution events for accounting.

CRITICAL: Trades represent actual executed fills.
- Orders are intent/execution wrappers
- Trades are what actually happened
- Ledger entries reference trades, not orders
- One order may produce 0..N trades (partial fills, TWAP slices, etc.)

Design constraints:
- Immutable once recorded
- Every trade creates ledger entries
- Trades are used for tax lot creation and FIFO matching
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship

from .database import Base


class TradeSide(str, Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"


class Trade(Base):
    """Trade execution record - actual fill events.

    This is the authoritative record of what was executed.
    Orders track intent and routing, trades track execution.

    Example:
        Order: "Buy $1000 of BTC"
        Trades:
            - Fill 1: 0.025 BTC @ $40,000 = $1000 (complete fill)

        or:
            - Fill 1: 0.01 BTC @ $40,000 = $400 (partial)
            - Fill 2: 0.015 BTC @ $40,100 = $601.50 (complete)

    Each trade creates multiple ledger entries:
        BUY: quote currency -, base currency +, fee -
        SELL: base currency -, quote currency +, fee -
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign keys
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    owner_id = Column(String(100), nullable=False, index=True)  # TODO: Add FK when User model exists
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False, index=True)

    # Exchange and trading pair
    exchange = Column(String(50), nullable=False)  # e.g., "binance", "simulated"
    trading_pair = Column(String(50), nullable=False, index=True)  # e.g., "BTC/USDT"

    # Trade details
    side = Column(SQLEnum(TradeSide), nullable=False, index=True)
    base_asset = Column(String(10), nullable=False)    # e.g., "BTC"
    quote_asset = Column(String(10), nullable=False)   # e.g., "USDT"
    base_amount = Column(Float, nullable=False)        # Amount of base asset
    quote_amount = Column(Float, nullable=False)       # Amount of quote asset
    price = Column(Float, nullable=False)              # Execution price

    # Costs
    fee_amount = Column(Float, default=0.0)                 # Exchange fee (in quote currency)
    fee_asset = Column(String(10), nullable=True)           # Fee currency
    modeled_cost = Column(Float, default=0.0)               # Modeled execution costs

    # Execution metadata
    exchange_trade_id = Column(String(100), nullable=True)  # Exchange's trade ID
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Strategy context
    strategy_used = Column(String(50), nullable=True)

    # Relationships
    order = relationship("Order", back_populates="trades")
    bot = relationship("Bot", back_populates="trades")
    ledger_entries = relationship("WalletLedger", foreign_keys="WalletLedger.related_trade_id", viewonly=True)

    def __repr__(self):
        return (
            f"<Trade(id={self.id}, "
            f"{self.side.value} {self.base_amount:.8f} {self.base_asset} "
            f"@ ${self.price:.2f})>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "owner_id": self.owner_id,
            "bot_id": self.bot_id,
            "exchange": self.exchange,
            "trading_pair": self.trading_pair,
            "side": self.side.value,
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "base_amount": self.base_amount,
            "quote_amount": self.quote_amount,
            "price": self.price,
            "fee_amount": self.fee_amount,
            "fee_asset": self.fee_asset,
            "modeled_cost": self.modeled_cost,
            "exchange_trade_id": self.exchange_trade_id,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "strategy_used": self.strategy_used,
        }

    def get_total_cost(self) -> float:
        """Get total cost including fees and modeled costs."""
        return self.quote_amount + self.fee_amount + self.modeled_cost

    def get_cost_basis_per_unit(self) -> float:
        """Get cost basis per unit of base asset (for tax lots)."""
        if self.base_amount <= 0:
            return 0.0
        return self.get_total_cost() / self.base_amount
