"""Configuration router for strategies and trading pairs."""

from typing import List, Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class StrategyInfo(BaseModel):
    """Schema for strategy information."""
    name: str
    display_name: str
    description: str
    parameters: Dict[str, Any]


class TradingPair(BaseModel):
    """Schema for trading pair."""
    symbol: str
    base: str
    quote: str


# Available strategies
STRATEGIES = [
    StrategyInfo(
        name="dca_accumulator",
        display_name="DCA Accumulator",
        description="Dollar-cost averaging with configurable intervals and amounts",
        parameters={
            "interval_minutes": {"type": "number", "default": 60, "min": 1, "description": "Buy interval in minutes"},
            "amount_percent": {"type": "number", "default": 10, "min": 1, "max": 100, "description": "Percent of budget per buy"},
        }
    ),
    StrategyInfo(
        name="adaptive_grid",
        display_name="Adaptive Grid",
        description="Dynamic grid trading with configurable grid count, spacing, and range",
        parameters={
            "grid_count": {"type": "number", "default": 10, "min": 2, "max": 100, "description": "Number of grid levels"},
            "grid_spacing_percent": {"type": "number", "default": 1.0, "min": 0.1, "max": 10, "description": "Spacing between levels (%)"},
            "range_percent": {"type": "number", "default": 10, "min": 1, "max": 50, "description": "Total grid range (%)"},
        }
    ),
    StrategyInfo(
        name="mean_reversion",
        display_name="Mean Reversion",
        description="Trade reversions to mean with Bollinger bands",
        parameters={
            "bollinger_period": {"type": "number", "default": 20, "min": 5, "max": 100, "description": "Bollinger band period"},
            "bollinger_std": {"type": "number", "default": 2.0, "min": 0.5, "max": 4, "description": "Standard deviation multiplier"},
        }
    ),
    StrategyInfo(
        name="breakdown_momentum",
        display_name="Breakdown Momentum",
        description="Trend-following on breakouts with volume confirmation",
        parameters={
            "breakout_threshold_percent": {"type": "number", "default": 2.0, "min": 0.5, "max": 10, "description": "Breakout threshold (%)"},
            "volume_threshold_multiplier": {"type": "number", "default": 1.5, "min": 1, "max": 5, "description": "Volume threshold multiplier"},
        }
    ),
    StrategyInfo(
        name="twap",
        display_name="TWAP",
        description="Time-weighted average price execution over configurable period",
        parameters={
            "execution_period_minutes": {"type": "number", "default": 60, "min": 1, "description": "Execution period in minutes"},
            "slice_count": {"type": "number", "default": 10, "min": 2, "max": 100, "description": "Number of order slices"},
        }
    ),
    StrategyInfo(
        name="vwap",
        display_name="VWAP",
        description="Volume-weighted average price targeting",
        parameters={
            "lookback_period_minutes": {"type": "number", "default": 30, "min": 5, "description": "VWAP lookback period in minutes"},
        }
    ),
    StrategyInfo(
        name="scalping",
        display_name="Scalping",
        description="High-frequency small profit captures",
        parameters={
            "take_profit_percent": {"type": "number", "default": 0.5, "min": 0.1, "max": 5, "description": "Take profit (%)"},
            "max_position_time_seconds": {"type": "number", "default": 300, "min": 10, "description": "Max position hold time (seconds)"},
        }
    ),
    StrategyInfo(
        name="arbitrage",
        display_name="Arbitrage",
        description="Cross-exchange or cross-pair price difference exploitation",
        parameters={
            "min_spread_percent": {"type": "number", "default": 0.3, "min": 0.1, "max": 5, "description": "Minimum spread to trade (%)"},
        }
    ),
    StrategyInfo(
        name="event_filler",
        display_name="Event Filler",
        description="React to market events/news with rapid execution",
        parameters={
            "event_sources": {"type": "array", "default": [], "description": "List of event sources to monitor"},
        }
    ),
    StrategyInfo(
        name="auto_mode",
        display_name="Auto Mode",
        description="Factor-based strategy selection (trend/volume/volatility)",
        parameters={
            "factor_precedence": {"type": "array", "default": ["trend", "volume", "volatility"], "description": "Order of factor importance"},
            "disabled_factors": {"type": "array", "default": [], "description": "Factors to disable"},
            "switch_threshold": {"type": "number", "default": 0.7, "min": 0.1, "max": 1, "description": "Threshold to trigger strategy switch"},
        }
    ),
]


@router.get("/strategies", response_model=List[StrategyInfo])
async def get_strategies():
    """Get available trading strategies with their parameters."""
    return STRATEGIES


@router.get("/pairs", response_model=List[TradingPair])
async def get_trading_pairs():
    """Get available trading pairs from exchange."""
    # TODO: Fetch from exchange via ccxt
    # For now, return common pairs
    common_pairs = [
        TradingPair(symbol="BTC/USDT", base="BTC", quote="USDT"),
        TradingPair(symbol="ETH/USDT", base="ETH", quote="USDT"),
        TradingPair(symbol="SOL/USDT", base="SOL", quote="USDT"),
        TradingPair(symbol="XRP/USDT", base="XRP", quote="USDT"),
        TradingPair(symbol="ADA/USDT", base="ADA", quote="USDT"),
        TradingPair(symbol="DOGE/USDT", base="DOGE", quote="USDT"),
        TradingPair(symbol="DOT/USDT", base="DOT", quote="USDT"),
        TradingPair(symbol="LINK/USDT", base="LINK", quote="USDT"),
        TradingPair(symbol="AVAX/USDT", base="AVAX", quote="USDT"),
        TradingPair(symbol="MATIC/USDT", base="MATIC", quote="USDT"),
    ]
    return common_pairs
