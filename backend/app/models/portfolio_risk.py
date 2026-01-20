"""Portfolio-level risk caps model for cross-bot risk management.

IMPORTANT: Portfolio risk caps operate ACROSS all bots for the same owner.
This is the highest level of risk control, above per-bot limits.

Design constraints:
- All caps default to NULL/disabled
- System behavior unchanged unless explicitly configured
- Caps are hard stops (block trades)
- Exposure caps can resize orders to fit remaining capacity
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, Boolean, DateTime, String
from sqlalchemy.orm import relationship

from .database import Base


class PortfolioRisk(Base):
    """Portfolio-level risk configuration for an owner.

    This model defines risk caps that apply across ALL bots owned by
    the same owner. These are global limits above per-bot limits.

    All caps are optional (NULL = disabled). When enabled, they act as
    hard stops that can block or resize trades.
    """
    __tablename__ = "portfolio_risks"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(String(100), nullable=False, unique=True, index=True)

    # Loss caps (percentage of total portfolio initial balance)
    daily_loss_cap_pct = Column(Float, nullable=True)  # e.g., 5.0 = 5% daily loss limit
    weekly_loss_cap_pct = Column(Float, nullable=True)  # e.g., 10.0 = 10% weekly loss limit

    # Drawdown cap (from portfolio high water mark)
    max_drawdown_pct = Column(Float, nullable=True)  # e.g., 15.0 = 15% max drawdown

    # Exposure cap (total open notional as % of portfolio)
    max_total_exposure_pct = Column(Float, nullable=True)  # e.g., 80.0 = 80% max exposure

    # Enable/disable flag
    enabled = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return (
            f"<PortfolioRisk(owner={self.owner_id}, enabled={self.enabled}, "
            f"daily_loss={self.daily_loss_cap_pct}%, "
            f"max_dd={self.max_drawdown_pct}%)>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "daily_loss_cap_pct": self.daily_loss_cap_pct,
            "weekly_loss_cap_pct": self.weekly_loss_cap_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
