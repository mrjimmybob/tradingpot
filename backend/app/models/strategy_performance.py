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
    bot = relationship("Bot", backref="strategy_metrics")
    
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
            "failure_count": self.failure_count,
            "last_exit_time": self.last_exit_time.isoformat() if self.last_exit_time else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }
    
    @classmethod
    def from_dict(cls, bot_id: int, strategy_name: str, data: dict):
        """Create from in-memory format."""
        return cls(
            bot_id=bot_id,
            strategy_name=strategy_name,
            recent_pnl_pct=data.get("recent_pnl_pct", 0.0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            failure_count=data.get("failure_count", 0),
            last_exit_time=datetime.fromisoformat(data["last_exit_time"]) if data.get("last_exit_time") else None,
            cooldown_until=datetime.fromisoformat(data["cooldown_until"]) if data.get("cooldown_until") else None,
        )
