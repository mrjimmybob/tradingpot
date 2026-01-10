"""Market data cache model."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime

from .database import Base


class MarketDataCache(Base):
    """Market data cache model for external data sources."""
    __tablename__ = "market_data_cache"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False)  # coinmarketcap, coingecko, etc.
    data_type = Column(String(50), nullable=False)  # price, volume, market_cap, fear_greed
    trading_pair = Column(String(50), nullable=True)  # null for global data like fear/greed
    value = Column(Float, nullable=False)

    # Timestamp
    fetched_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MarketDataCache(source={self.source}, type={self.data_type})>"
