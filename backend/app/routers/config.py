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
            "atr_range_multiplier": {"type": "number", "default": 8.0, "min": 1.0, "max": 30.0, "description": "Total grid span = ATR × this value. Higher = wider grid in volatile markets."},
            "atr_warmup_spacing_pct": {"type": "number", "default": 0.5, "min": 0.1, "max": 2.0, "description": "Fixed % spacing used before ATR warms up (first 14 bars)"},
            "base_order_size_percent": {"type": "number", "default": 5, "min": 1, "max": 50, "description": "Base order size (% of budget). Must produce ≥ $10 order."},
            "depth_multiplier": {"type": "number", "default": 1.5, "min": 1.0, "max": 2.5, "description": "Multiplier for deeper levels (convex sizing)"},
            "max_drawdown_percent": {"type": "number", "default": 15, "min": 5, "max": 50, "description": "Max drawdown % before kill switch"},
            "kill_atr_multiplier": {"type": "number", "default": 3.0, "min": 1, "max": 10, "description": "ATR distance for range escape kill switch"},
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "ATR period for spacing and kill switch (bars)"},
            "regime_filter_enabled": {"type": "boolean", "default": True, "description": "Enable regime-aware gating (flat/normal markets only)"},
            "allowed_regimes": {"type": "array", "default": ["trend_flat", "volatility_medium"], "description": "Allowed regimes (trend_flat, volatility_medium recommended)"},
            "cooldown_after_kill_hours": {"type": "number", "default": 2, "min": 0, "max": 24, "description": "Hours to wait after kill switch (prevents re-entry)"},
            "taker_fee_pct": {"type": "number", "default": 0.1, "min": 0.0, "max": 1.0, "description": "Taker fee rate (%). Used to compute minimum profitable spacing."},
            "spread_pct": {"type": "number", "default": 0.05, "min": 0.0, "max": 0.5, "description": "Estimated bid-ask spread (%). Part of minimum profitable spacing."},
            "min_profit_buffer_pct": {"type": "number", "default": 0.1, "min": 0.0, "max": 1.0, "description": "Minimum profit buffer above fees (%). Ensures each cycle is economically viable."},
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
    StrategyInfo(
        name="dip_recovery",
        display_name="Dip Recovery (Reversal Momentum)",
        description="Pullback reversal strategy: monitors for a significant decline (relative to current volatility) but only buys after price has confirmed a reversal off a tracked local low. Never buys into a still-falling market - accepts a later entry for reduced downside risk.",
        parameters={
            "atr_period": {"type": "number", "default": 14, "min": 5, "max": 50, "description": "Ticks in the ATR proxy window used to adapt every threshold to current volatility"},
            "reference_high_lookback_ticks": {"type": "number", "default": 60, "min": 10, "max": 500, "description": "Rolling window used to derive the recent high a decline is measured against"},
            "min_drop_percent": {"type": "number", "default": 1.5, "min": 0.1, "max": 20, "description": "Floor for the adaptive drop threshold (%) - a decline must clear at least this even in dead-calm markets"},
            "drop_atr_multiplier": {"type": "number", "default": 2.5, "min": 0.5, "max": 10, "description": "Adaptive drop threshold = max(min_drop_percent, ATR% x this) - how many ATRs deep a decline must be to count as significant"},
            "min_recovery_percent": {"type": "number", "default": 0.5, "min": 0.05, "max": 10, "description": "Floor for the adaptive recovery confirmation (%)"},
            "recovery_atr_multiplier": {"type": "number", "default": 0.8, "min": 0.1, "max": 5, "description": "Adaptive recovery confirmation = max(min_recovery_percent, ATR% x this) - how much of a bounce off the low is required before buying"},
            "require_ema_slope_confirmation": {"type": "boolean", "default": True, "description": "Require the short EMA to be sloping upward before entry"},
            "ema_slope_period": {"type": "number", "default": 5, "min": 2, "max": 50, "description": "Period of the short EMA used for the slope confirmation filter"},
            "require_no_new_low_confirmation": {"type": "boolean", "default": True, "description": "Require N ticks with no new low before entry"},
            "min_ticks_without_new_low": {"type": "number", "default": 2, "min": 0, "max": 50, "description": "Consecutive ticks without a new low required for the no-new-low confirmation filter"},
            "take_profit_atr_multiplier": {"type": "number", "default": 3.0, "min": 0.5, "max": 10, "description": "Take-profit target = entry price + ATR (locked at entry) x this"},
            "trailing_stop_atr_multiplier": {"type": "number", "default": 1.5, "min": 0.5, "max": 10, "description": "Trailing stop distance = highest price since entry - ATR (locked at entry) x this; only tightens, never expands"},
            "emergency_stop_atr_multiplier": {"type": "number", "default": 5.0, "min": 1, "max": 20, "description": "Last-resort hard stop distance from entry price - must be wider than trailing_stop_atr_multiplier"},
            "max_position_duration_minutes": {"type": "number", "default": 720, "min": 5, "max": 20160, "description": "Force-exit a position after this long regardless of P&L"},
            "setup_expiry_minutes": {"type": "number", "default": 240, "min": 5, "max": 20160, "description": "Abandon an unresolved decline/reversal setup after this long with no confirmed entry, returning to IDLE"},
            "cooldown_seconds": {"type": "number", "default": 300, "min": 0, "max": 86400, "description": "Pause before monitoring resumes after any exit"},
            "loss_cooldown_seconds": {"type": "number", "default": 1800, "min": 0, "max": 86400, "description": "Extended pause after a losing exit (must be >= cooldown_seconds)"},
            "risk_percent": {"type": "number", "default": 1.0, "min": 0.1, "max": 100, "description": "Percent of balance risked per trade, sized off the trailing-stop distance"},
            "spike_guard_atr_multiplier": {"type": "number", "default": 6.0, "min": 1, "max": 50, "description": "A single-tick move bigger than this many ATRs is treated as noise and ignored when updating the tracked high/low"},
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
