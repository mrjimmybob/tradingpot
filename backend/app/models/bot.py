"""Bot model for trading bot instances."""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship

from .database import Base


class BotStatus(str, Enum):
    """Bot status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class Bot(Base):
    """Trading bot model."""
    __tablename__ = "bots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    trading_pair = Column(String(50), nullable=False)
    strategy = Column(String(50), nullable=False)
    strategy_params = Column(JSON, default=dict)

    # Virtual wallet
    budget = Column(Float, nullable=False)
    compound_enabled = Column(Boolean, default=False)
    current_balance = Column(Float, nullable=False)

    # Running time
    running_time_hours = Column(Float, nullable=True)  # null = forever

    # Risk parameters
    stop_loss_percent = Column(Float, nullable=True)
    stop_loss_absolute = Column(Float, nullable=True)
    drawdown_limit_percent = Column(Float, nullable=True)
    drawdown_limit_absolute = Column(Float, nullable=True)
    daily_loss_limit = Column(Float, nullable=True)
    weekly_loss_limit = Column(Float, nullable=True)
    max_strategy_rotations = Column(Integer, default=3)

    # Dry run mode
    is_dry_run = Column(Boolean, default=False)

    # Status
    status = Column(SQLEnum(BotStatus), default=BotStatus.CREATED)

    # P&L tracking
    total_pnl = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)

    # Relationships
    orders = relationship("Order", back_populates="bot", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="bot", cascade="all, delete-orphan")
    strategy_rotations = relationship("StrategyRotation", back_populates="bot", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="bot", cascade="all, delete-orphan")
    pnl_snapshots = relationship("PnLSnapshot", back_populates="bot", cascade="all, delete-orphan")
    ledger_entries = relationship("WalletLedger", back_populates="bot", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="bot", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Bot(id={self.id}, name='{self.name}', status={self.status.value})>"
