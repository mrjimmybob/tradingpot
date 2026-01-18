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
            "amount_percent": {"type": "number", "default": 10, "min": 1, "max": 100, "description": "Percent of balance per buy"},
            "amount_usd": {"type": "number", "default": None, "min": 1, "description": "Fixed USD amount per buy (overrides percent)"},
            "immediate_first_buy": {"type": "boolean", "default": True, "description": "Execute first buy immediately on start"},
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
            "order_size_percent": {"type": "number", "default": 10, "min": 1, "max": 50, "description": "Order size per level (% of budget)"},
        }
    ),
    StrategyInfo(
        name="mean_reversion",
        display_name="Mean Reversion",
        description="Trade reversions to mean with Bollinger bands",
        parameters={
            "bollinger_period": {"type": "number", "default": 20, "min": 5, "max": 100, "description": "Bollinger band period"},
            "bollinger_std": {"type": "number", "default": 2.0, "min": 0.5, "max": 4, "description": "Standard deviation multiplier"},
            "order_size_percent": {"type": "number", "default": 20, "min": 5, "max": 100, "description": "Order size (% of budget)"},
            "exit_at_mean": {"type": "boolean", "default": True, "description": "Exit at mean instead of upper band"},
        }
    ),
    # Note: breakdown_momentum strategy was removed (stub implementation, overlaps with other strategies)
    StrategyInfo(
        name="trend_following",
        display_name="Trend Following",
        description="Conservative long-only momentum strategy using EMA crossover and ATR-based stops",
        parameters={
            "short_period": {"type": "number", "default": 50, "min": 10, "max": 100, "description": "EMA short period"},
            "long_period": {"type": "number", "default": 200, "min": 50, "max": 500, "description": "EMA long period"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period for volatility"},
            "atr_multiplier": {"type": "number", "default": 2.0, "min": 1, "max": 5, "description": "ATR multiplier for stop loss"},
            "risk_percent": {"type": "number", "default": 1.0, "min": 0.5, "max": 5, "description": "Percent of capital to risk per trade"},
        }
    ),
    StrategyInfo(
        name="cross_sectional_momentum",
        display_name="Cross-Sectional Momentum",
        description="Relative strength strategy that ranks assets by performance and holds top performers only",
        parameters={
            "universe": {"type": "array", "default": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT"], "description": "List of symbols to compare"},
            "lookback_days": {"type": "number", "default": 60, "min": 7, "max": 180, "description": "Days to calculate momentum"},
            "top_n": {"type": "number", "default": 3, "min": 1, "max": 10, "description": "Number of top assets to hold"},
            "rebalance_hours": {"type": "number", "default": 168, "min": 24, "max": 720, "description": "Hours between rebalances (168 = weekly)"},
            "allocation_percent": {"type": "number", "default": 100, "min": 10, "max": 100, "description": "Percent of capital to allocate"},
            "trend_filter_enabled": {"type": "boolean", "default": False, "description": "Enable global trend filter"},
            "trend_filter_symbol": {"type": "string", "default": "BTC/USDT", "description": "Symbol for trend filter"},
            "trend_filter_ema": {"type": "number", "default": 200, "min": 50, "max": 500, "description": "EMA period for trend filter"},
        }
    ),
    StrategyInfo(
        name="volatility_breakout",
        display_name="Volatility Breakout",
        description="Enters on price breakouts following low-volatility compression using Bollinger Bands and ATR",
        parameters={
            "bb_period": {"type": "number", "default": 20, "min": 10, "max": 50, "description": "Bollinger Band period"},
            "bb_std": {"type": "number", "default": 2.0, "min": 1, "max": 3, "description": "Bollinger Band standard deviation"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period"},
            "compression_method": {"type": "string", "default": "bb_width", "options": ["bb_width", "atr_average"], "description": "Compression detection method"},
            "compression_percentile": {"type": "number", "default": 20, "min": 5, "max": 50, "description": "BB width percentile threshold (%)"},
            "atr_threshold_multiplier": {"type": "number", "default": 0.8, "min": 0.5, "max": 1.5, "description": "ATR threshold vs average"},
            "min_compression_bars": {"type": "number", "default": 5, "min": 2, "max": 20, "description": "Minimum compression duration (bars)"},
            "atr_stop_multiplier": {"type": "number", "default": 2.0, "min": 1, "max": 5, "description": "ATR stop loss multiplier"},
            "risk_percent": {"type": "number", "default": 1.0, "min": 0.5, "max": 5, "description": "Percent of capital to risk per trade"},
            "cooldown_hours": {"type": "number", "default": 24, "min": 1, "max": 168, "description": "Hours between breakout attempts"},
            "failed_breakout_bars": {"type": "number", "default": 3, "min": 1, "max": 10, "description": "Bars to check for failed breakout"},
        }
    ),
    StrategyInfo(
        name="twap",
        display_name="TWAP",
        description="Time-weighted average price execution over configurable period",
        parameters={
            "execution_period_minutes": {"type": "number", "default": 60, "min": 1, "description": "Execution period in minutes"},
            "slice_count": {"type": "number", "default": 10, "min": 2, "max": 100, "description": "Number of order slices"},
            "total_amount_usd": {"type": "number", "default": None, "min": 1, "description": "Total amount to execute (USD)"},
            "side": {"type": "string", "default": "buy", "options": ["buy", "sell"], "description": "Order side"},
        }
    ),
    StrategyInfo(
        name="vwap",
        display_name="VWAP",
        description="Volume-weighted average price targeting - buy below VWAP, sell above",
        parameters={
            "lookback_period_minutes": {"type": "number", "default": 30, "min": 5, "description": "VWAP lookback period in minutes"},
            "deviation_threshold_percent": {"type": "number", "default": 0.5, "min": 0.1, "max": 5, "description": "Min deviation to trigger trade (%)"},
            "order_size_percent": {"type": "number", "default": 20, "min": 5, "max": 100, "description": "Order size (% of budget)"},
        }
    ),
    StrategyInfo(
        name="scalping",
        display_name="Scalping",
        description="Conservative tactical strategy for quick profits from micro-breakouts with strict risk controls",
        parameters={
            "short_ema": {"type": "number", "default": 5, "min": 3, "max": 20, "description": "Short EMA period for momentum"},
            "long_ema": {"type": "number", "default": 15, "min": 10, "max": 50, "description": "Long EMA period for momentum"},
            "take_profit_percent": {"type": "number", "default": 0.5, "min": 0.1, "max": 2, "description": "Profit target (%)"},
            "stop_loss_percent": {"type": "number", "default": 0.5, "min": 0.1, "max": 2, "description": "Stop loss (%)"},
            "max_position_time_seconds": {"type": "number", "default": 300, "min": 60, "max": 1800, "description": "Max position hold time (seconds)"},
            "position_size_percent": {"type": "number", "default": 5, "min": 1, "max": 20, "description": "Position size (% of balance)"},
            "cooldown_minutes": {"type": "number", "default": 10, "min": 1, "max": 60, "description": "Cooldown between trades (minutes)"},
            "max_trades_per_hour": {"type": "number", "default": 3, "min": 1, "max": 10, "description": "Maximum trades per hour"},
            "max_trades_per_day": {"type": "number", "default": 20, "min": 1, "max": 100, "description": "Maximum trades per day"},
            "trend_filter_ema": {"type": "number", "default": 50, "min": 0, "max": 200, "description": "Global trend filter EMA (0 = disabled)"},
        }
    ),
    # Note: arbitrage and event_filler strategies were removed (placeholders without implementation)
    StrategyInfo(
        name="auto_mode",
        display_name="Auto Mode",
        description="Factor-based strategy selection - automatically adapts to market conditions",
        parameters={
            "factor_precedence": {"type": "array", "default": ["trend", "volatility", "volume"], "description": "Order of factor importance"},
            "disabled_factors": {"type": "array", "default": [], "description": "Factors to disable"},
            "switch_threshold": {"type": "number", "default": 0.7, "min": 0.1, "max": 1, "description": "Confidence threshold to switch strategy"},
            "min_switch_interval_minutes": {"type": "number", "default": 15, "min": 1, "description": "Minimum time between strategy switches"},
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
