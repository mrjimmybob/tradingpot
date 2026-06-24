"""Strategy performance metrics model for persistent auto-mode learning."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from .database import Base


class StrategyPerformanceMetrics(Base):
    """Persistent storage for per-bot, per-strategy performance tracking.
    
    Used by auto-mode to remember:
    - Which strategies are performing well/poorly
    - Which strategies are in cooldown
    - Which strategies are blacklisted
    
    This ensures auto-mode decisions survive bot restarts.
    """
    __tablename__ = "strategy_performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, ForeignKey("bots.id", ondelete="CASCADE"), nullable=False)
    strategy_name = Column(String(50), nullable=False)

    # Performance metrics
    recent_pnl_pct = Column(Float, nullable=False, default=0.0)
    max_drawdown_pct = Column(Float, nullable=False, default=0.0)

    # Rolling trade statistics (last 20 completed trades)
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    realized_pnl_usd = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=False, default=0.0)
    last_trade_time = Column(DateTime, nullable=True)

    # Failure tracking
    failure_count = Column(Integer, nullable=False, default=0)

    # Timing
    last_exit_time = Column(DateTime, nullable=True)
    cooldown_until = Column(DateTime, nullable=True)

    # Metadata
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('bot_id', 'strategy_name', name='uq_bot_strategy'),
    )
    
    # Relationships
    # Use back_populates against an explicit Bot-side relationship that declares
    # cascade="all, delete-orphan" (see Bot.strategy_metrics). A bare backref
    # would create a Bot-side collection with SQLAlchemy's DEFAULT cascade
    # ("save-update, merge"), which on bot deletion DISASSOCIATES children by
    # emitting `UPDATE strategy_performance_metrics SET bot_id=NULL` — a NOT NULL
    # violation, since bot_id is non-nullable. Matching the other child models
    # makes the ORM DELETE these rows with the parent instead.
    bot = relationship("Bot", back_populates="strategy_metrics")
    
    def __repr__(self):
        return (
            f"<StrategyPerformanceMetrics("
            f"bot_id={self.bot_id}, "
            f"strategy={self.strategy_name}, "
            f"pnl={self.recent_pnl_pct:.2f}%, "
            f"failures={self.failure_count})>"
        )
    
    def to_dict(self) -> dict:
        """Convert to in-memory format used by TradingEngine."""
        return {
            "recent_pnl_pct": self.recent_pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "realized_pnl_usd": self.realized_pnl_usd,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "failure_count": self.failure_count,
            "last_exit_time": self.last_exit_time.isoformat() if self.last_exit_time else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    @classmethod
    def from_dict(cls, bot_id: int, strategy_name: str, data: dict):
        """Create from in-memory format."""
        def _parse_dt(val):
            if val is None:
                return None
            return datetime.fromisoformat(val) if isinstance(val, str) else val

        return cls(
            bot_id=bot_id,
            strategy_name=strategy_name,
            recent_pnl_pct=data.get("recent_pnl_pct", 0.0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            realized_pnl_usd=data.get("realized_pnl_usd", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            win_rate=data.get("win_rate", 0.0),
            last_trade_time=_parse_dt(data.get("last_trade_time")),
            failure_count=data.get("failure_count", 0),
            last_exit_time=_parse_dt(data.get("last_exit_time")),
            cooldown_until=_parse_dt(data.get("cooldown_until")),
        )
