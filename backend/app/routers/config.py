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
        display_name="DCA Accumulator (Institutional Grade)",
        description="Infinite accumulation: clock-driven buys at regular intervals with regime-aware pausing - continues until balance exhausted or manually stopped",
        parameters={
            "interval_minutes": {"type": "number", "default": 60, "min": 1, "description": "Buy interval in minutes"},
            "amount_percent": {"type": "number", "default": 10, "min": 1, "max": 100, "description": "Percent of balance per buy"},
            "amount_usd": {"type": "number", "default": None, "min": 1, "description": "Fixed USD amount per buy (overrides percent)"},
            "immediate_first_buy": {"type": "boolean", "default": True, "description": "Execute first buy immediately on start"},
            "regime_filter_enabled": {"type": "boolean", "default": True, "description": "Pause accumulation during unfavorable market regimes"},
            "allowed_regimes": {"type": "array", "default": ["trend_up", "trend_flat"], "description": "Allowed trend regimes for buying (trend_up, trend_down, trend_flat)"},
        }
    ),
    StrategyInfo(
        name="adaptive_grid",
        display_name="Adaptive Grid (Institutional Grade)",
        description="CAPITAL-BOUNDED, REGIME-AWARE, LONG-BIASED grid - bar-aggregated manufacturing process with kill switches, depth-aware sizing, and convex payoff profile",
        parameters={
            "bar_interval_seconds": {"type": "number", "default": 60, "min": 30, "max": 300, "description": "Seconds per bar for aggregation (60 = 1min bars)"},
            "grid_count": {"type": "number", "default": 10, "min": 2, "max": 50, "description": "Total grid levels (70% buy below, 30% sell above)"},
            "grid_spacing_percent": {"type": "number", "default": 1.0, "min": 0.1, "max": 5, "description": "Spacing between adjacent levels (%)"},
            "range_percent": {"type": "number", "default": 10, "min": 1, "max": 30, "description": "Total grid range % from center (deprecated, use spacing)"},
            "base_order_size_percent": {"type": "number", "default": 5, "min": 1, "max": 20, "description": "Base order size (% of budget)"},
            "depth_multiplier": {"type": "number", "default": 1.5, "min": 1.0, "max": 2.5, "description": "Multiplier for deeper levels (convex sizing)"},
            "max_drawdown_percent": {"type": "number", "default": 15, "min": 5, "max": 50, "description": "Max drawdown % before kill switch"},
            "kill_atr_multiplier": {"type": "number", "default": 3.0, "min": 1, "max": 10, "description": "ATR distance for range escape kill switch"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period for kill switch (bars)"},
            "regime_filter_enabled": {"type": "boolean", "default": True, "description": "Enable regime-aware gating (flat/normal markets only)"},
            "allowed_regimes": {"type": "array", "default": ["trend_flat", "volatility_normal"], "description": "Allowed regimes (trend_flat, volatility_normal recommended)"},
            "cooldown_after_kill_hours": {"type": "number", "default": 2, "min": 0, "max": 24, "description": "Hours to wait after kill switch (prevents re-entry)"},
        }
    ),
    StrategyInfo(
        name="mean_reversion",
        display_name="Mean Reversion (Institutional Grade)",
        description="ANTI-TREND, BOUNDED-RISK, REGIME-AWARE: Bar-aggregated Bollinger Bands with hard stop + time stop - force-exits on trend flips",
        parameters={
            "bar_interval_seconds": {"type": "number", "default": 60, "min": 30, "max": 300, "description": "Seconds per bar for aggregation (60 = 1min bars)"},
            "bollinger_period": {"type": "number", "default": 20, "min": 5, "max": 100, "description": "Bollinger band period (bars)"},
            "bollinger_std": {"type": "number", "default": 2.0, "min": 0.5, "max": 4, "description": "Standard deviation multiplier"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period for hard stop (bars)"},
            "atr_stop_multiplier": {"type": "number", "default": 2.0, "min": 1, "max": 5, "description": "ATR stop multiplier (locked at entry)"},
            "max_hold_bars": {"type": "number", "default": 10, "min": 3, "max": 50, "description": "Maximum bars to hold (time stop)"},
            "order_size_percent": {"type": "number", "default": 20, "min": 5, "max": 100, "description": "Order size (% of balance)"},
            "exit_at_mean": {"type": "boolean", "default": True, "description": "Exit at mean vs upper band"},
            "regime_filter_enabled": {"type": "boolean", "default": True, "description": "Enable regime gating (force-exits on trends)"},
            "cooldown_seconds": {"type": "number", "default": 300, "min": 0, "max": 3600, "description": "Seconds between trades"},
        }
    ),
    # Note: breakdown_momentum strategy was removed (stub implementation, overlaps with other strategies)
    StrategyInfo(
        name="trend_following",
        display_name="Trend Following (Institutional Grade)",
        description="Hardened long-only momentum with locked entry ATR, noise-resistant confirmation, and re-entry cooldown",
        parameters={
            "short_period": {"type": "number", "default": 50, "min": 10, "max": 100, "description": "EMA short period"},
            "long_period": {"type": "number", "default": 200, "min": 50, "max": 500, "description": "EMA long period"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period for volatility"},
            "atr_multiplier": {"type": "number", "default": 2.0, "min": 1, "max": 5, "description": "ATR multiplier for stop loss"},
            "risk_percent": {"type": "number", "default": 1.0, "min": 0.5, "max": 5, "description": "Percent of capital to risk per trade"},
            "entry_confirmation_loops": {"type": "number", "default": 3, "min": 1, "max": 10, "description": "Consecutive loops required for entry (noise defense)"},
            "exit_confirmation_loops": {"type": "number", "default": 2, "min": 1, "max": 10, "description": "Consecutive loops required for exit (anti-whipsaw)"},
            "cooldown_seconds": {"type": "number", "default": 300, "min": 0, "max": 3600, "description": "Seconds to wait after exit before re-entry (anti-churn)"},
        }
    ),
    # Note: cross_sectional_momentum strategy removed (stub implementation, requires multi-asset framework)
    StrategyInfo(
        name="volatility_breakout",
        display_name="Volatility Breakout (Institutional Grade)",
        description="RARE, CONVEX, REGIME-AWARE: Bar-aggregated compression + breakout with locked entry ATR and monotonic trailing stop - long-only upper-band breakouts",
        parameters={
            "bar_interval_seconds": {"type": "number", "default": 60, "min": 30, "max": 300, "description": "Seconds per bar for aggregation (60 = 1min bars)"},
            "bb_period": {"type": "number", "default": 20, "min": 10, "max": 50, "description": "Bollinger Band period (bars)"},
            "bb_std": {"type": "number", "default": 2.0, "min": 1, "max": 3, "description": "Bollinger Band standard deviation"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period (bars)"},
            "compression_method": {"type": "string", "default": "bb_width", "options": ["bb_width", "atr_average"], "description": "Compression detection method"},
            "compression_percentile": {"type": "number", "default": 20, "min": 5, "max": 50, "description": "BB width percentile threshold (%)"},
            "atr_threshold_multiplier": {"type": "number", "default": 0.8, "min": 0.5, "max": 1.5, "description": "ATR threshold vs average"},
            "min_compression_bars": {"type": "number", "default": 20, "min": 2, "max": 50, "description": "Minimum compression bars (SPARSE: 20 = rare trades)"},
            "atr_stop_multiplier": {"type": "number", "default": 2.0, "min": 1, "max": 5, "description": "ATR stop loss multiplier (locked at entry)"},
            "risk_percent": {"type": "number", "default": 1.0, "min": 0.5, "max": 5, "description": "Percent of capital to risk per trade"},
            "cooldown_hours": {"type": "number", "default": 72, "min": 1, "max": 720, "description": "Hours between attempts (SPARSE: 72 = 3 days)"},
            "failed_breakout_bars": {"type": "number", "default": 3, "min": 1, "max": 10, "description": "Bars to detect failed breakout"},
            "regime_filter_enabled": {"type": "boolean", "default": True, "description": "Enable regime-aware gating"},
            "allowed_regimes": {"type": "array", "default": ["volatility_expanding"], "description": "Allowed volatility regimes (contracting, normal, expanding)"},
        }
    ),
    # Note: TWAP and VWAP are execution algorithms, not alpha strategies.
    # They are intentionally excluded from strategy selection.
    # TWAP/VWAP exist only in the execution layer for order execution methods.
    # Note: arbitrage and event_filler strategies were removed (placeholders without implementation)
    StrategyInfo(
        name="auto_mode",
        display_name="Auto Mode",
        description="Regime-based strategy selection policy - detects market regimes (trend, volatility, liquidity) and selects optimal strategy with inertia to prevent overtrading",
        parameters={
            "min_switch_interval_minutes": {"type": "number", "default": 15, "min": 1, "description": "Minimum time between strategy switches"},
            # DEPRECATED parameters (kept for backward compatibility, ignored by engine):
            "factor_precedence": {"type": "array", "default": ["trend", "volatility", "volume"], "description": "[DEPRECATED] Legacy parameter - ignored"},
            "disabled_factors": {"type": "array", "default": [], "description": "[DEPRECATED] Legacy parameter - ignored"},
            "switch_threshold": {"type": "number", "default": 0.7, "min": 0.1, "max": 1, "description": "[DEPRECATED] Legacy parameter - ignored"},
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
