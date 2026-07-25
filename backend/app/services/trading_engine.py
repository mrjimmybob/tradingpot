"""Trading engine for bot execution and management."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models import (
    Bot, BotStatus, Order, OrderType, OrderStatus,
    Position, PositionSide, PnLSnapshot,
    Trade, TradeSide, Alert,
    async_session_maker,
)
from .clock import Clock, SystemClock
from .exchange import ExchangeService, SimulatedExchangeService, OrderSide
from .virtual_wallet import VirtualWalletService
from .risk_management import RiskManagementService, RiskAction
from .email import email_service
from .logging_service import (
    BotLoggingService,
    TradeLogEntry,
    FiscalLogEntry,
    ensure_bot_log_directory,
)
from .execution_cost_model import ExecutionCostModel, get_cost_model
from .portfolio_risk import PortfolioRiskService
from .strategy_capacity import StrategyCapacityService
from .ledger_writer import LedgerWriterService
from .accounting import TradeRecorderService, FIFOTaxEngine, CSVExportService
from .ledger_invariants import LedgerInvariantService, ValidationError
from .decision_status import decision_status_store, DecisionState
from .strategy_explain import ExplanationBuilder
from .diagnostics import (
    diagnostics_store,
    BLOCK_RISK_MANAGER,
    BLOCK_MIN_ORDER_SIZE,
    BLOCK_INSUFFICIENT_BALANCE,
    BLOCK_POSITION_LIMITS,
    BLOCK_OTHER,
    DATA_UNAVAILABLE,
)
from .strategy_framework.explanation_persistence import (
    extract_edge_management_category,
    summarize_decision_explanation,
)
from .strategy_framework.market_suitability import MarketSuitabilityGate, MarketSuitabilityResult
from .strategy_framework.decision_score import DecisionScoreEngine, EvidenceItem
from .strategy_framework.adaptive_params import AdaptiveParameterResolver
from .strategy_framework.trade_management import TradeManagementMonitor
from .strategy_framework.edge_management import DcaEdgeManager, EdgeCategory, StrategyEdgeManager
from .strategy_framework.proposal import (
    Direction,
    ExecutionIntent,
    ProposalValidity,
    StrategyProposal,
    derive_reasons,
)
from .strategy_framework.standalone_adapter import StandaloneAdapter

logger = logging.getLogger(__name__)


# Minimum executable order notional (USD). A buy below this is rejected at the
# execution layer, so strategies must size at or above it (or HOLD with a clear
# reason) rather than emit a doomed sub-minimum order that gets rejected every
# loop. Single source of truth shared by strategy sizing and _execute_trade.
MIN_ORDER_USD = 10.0

# Maximum fraction of current_balance a strategy may commit to a single BUY.
# Reserves 0.2 % for the simulated exchange fee (0.1 %) plus a bid/ask spread
# buffer so that cost + fee never exceeds the available balance.  Constraint:
#   balance * _BUY_BALANCE_FRACTION * (1 + sim_fee_rate) <= balance
#   → fraction <= 1 / 1.001 ≈ 0.999001
# 0.998 comfortably covers 0.1 % fee + up to ~0.1 % spread.
_BUY_BALANCE_FRACTION = 0.998

# Minimum safety buffer (fraction of notional) added on top of round-trip fees
# when the central trade viability gate evaluates a BUY signal.  Ensures the
# trade must beat the fee hurdle by a small margin rather than just break even
# at the theoretical minimum.  0.05 % is intentionally conservative — the gate
# is a catastrophe filter, not a profitability predictor.
_VIABILITY_SAFETY_MARGIN_PCT = 0.0005

# Minimum acceptable reward:risk ratio for directional BUY signals that supply
# expected_risk_pct (distance to their own locked stop). Forensic review of
# closed trades found average losses running ~3x average wins (stops sized
# independently of targets); this ties them together at the gate so no
# directional strategy can risk materially more than it targets to gain.
# Strategies without a fixed price target (no locked stop to measure risk
# against) leave expected_risk_pct=None and are unaffected by this check.
_MIN_REWARD_RISK_RATIO = 1.5


# Repeated-rejection circuit breaker. When the SAME executable trade is rejected
# or fails for the SAME reason this many consecutive times, the bot is paused and
# the reason surfaced in Decision Status, instead of retrying the doomed action
# every tick forever (e.g. a sub-minimum order, or an un-settleable exit). This
# is independent of the loop-level failure breaker (which counts raised
# exceptions); a rejection is a clean None return, not an exception.
MAX_CONSECUTIVE_REJECTIONS = 5

# Accumulation strategies (DCA, grid-like) have no per-trade directional target.
# Gate applies a fee sanity check only: one-way fee must not exceed this fraction.
_MAX_ACCUMULATION_FEE_PCT = 0.05


def _normalize_trend_state(state: dict) -> dict:
    """Fill any keys missing from a restored trend-following state dict.

    Called on every access so state persisted before a key was added continues
    to work.  This is the single canonical source of valid trend state shape.
    """
    defaults: Dict[str, Any] = {
        "trailing_stop": None,
        "highest_price": None,
        "entry_atr": None,
        "entry_time": None,
        "last_exit_time": None,
        "entry_confirmation_count": 0,
        "exit_confirmation_count": 0,
        "tf_bars": [],
        "tf_current_bar": None,
        # Added by the Strategy Decision Framework migration (Phase 2). All
        # backfilled here so a state dict persisted before this migration
        # (no framework keys) restores cleanly with safe defaults.
        "tf_atr_history": [],          # rolling bar-ATR history for the adaptive resolver
        "entry_price": None,           # LOCKED at entry - edge-management P&L attribution
        "entry_stop_multiplier": None, # LOCKED adaptive ATR multiplier at entry
        "regime_state": None,          # persisted bar-based regime (MarketSuitabilityGate)
    }
    for key, val in defaults.items():
        if key not in state:
            state[key] = list(val) if isinstance(val, list) else val
    return state


# === STRATEGY STATE PERSISTENCE (C3/H1, M5) ===
# Per-bot, in-memory strategy state attributes that must survive a restart so
# resumed bots keep their risk state (trailing stops, locked entry ATR,
# cooldowns) and price history. Stored in Bot.strategy_state (a dedicated JSON
# column), NEVER in strategy_params (which is user config). Transient caches
# are intentionally excluded - they are safe to rebuild.
_PERSISTED_STATE_ATTRS = (
    "_grid_states",
    "_mean_reversion_states",
    "_trend_states",
    "_volatility_breakout_states",
    "_twap_states",
    "_vwap_states",
    "_auto_states",
    "_dip_recovery_states",
)

# Largest price-history window any strategy needs (trend_following long EMA).
# Persisting at least this many points means a resumed bot does not sit in a
# "collecting data" warmup while holding an unmanaged open position (H1).
_PERSISTED_PRICE_HISTORY_LEN = 250

# Legacy singular keys older builds stored inside strategy_params. Mapped to the
# current state attributes for backward-compatible restore.
_LEGACY_STATE_KEYS = {
    "_grid_state": "_grid_states",
    "_twap_state": "_twap_states",
    "_vwap_state": "_vwap_states",
    "_auto_state": "_auto_states",
}

# Valid alpha strategies for rotation. Execution algorithms (twap/vwap) and the
# auto_mode meta-policy are intentionally excluded: rotating into twap/vwap would
# silently disable the bot (no executor), and auto_mode would nest selectors.
_ALPHA_STRATEGIES = (
    "dca_accumulator",
    "adaptive_grid",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
)

_DT_TAG = "__dt__"


def _to_jsonable(obj):
    """Convert strategy state to JSON-safe primitives, tagging datetimes.

    Strategy state holds datetime objects (entry/exit times) that a plain JSON
    column cannot serialize. Datetimes become {"__dt__": isoformat}; dicts and
    lists are converted recursively.
    """
    if isinstance(obj, datetime):
        return {_DT_TAG: obj.isoformat()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj):
    """Inverse of _to_jsonable: restore tagged datetimes to datetime objects."""
    if isinstance(obj, dict):
        if set(obj.keys()) == {_DT_TAG}:
            try:
                return datetime.fromisoformat(obj[_DT_TAG])
            except (ValueError, TypeError):
                return None
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


class BotStartError(Exception):
    """Raised when a bot cannot be started safely (credentials, connectivity)."""


@dataclass
class TradeSignal:
    """Trading signal from strategy.

    Separates alpha (strategy decision) from execution (how to execute).

    Strategy Layer (Alpha):
        - action: "buy", "sell", "hold" (WHAT to do)
        - amount: How much to trade
        - reason: WHY this decision was made

    Execution Layer (How):
        - execution: "market", "twap", "vwap" (HOW to execute)
        - execution_params: Execution-specific parameters
        - order_type: "market" or "limit" (legacy, prefer execution)
    """
    action: str  # "buy", "sell", "hold"
    amount: float  # Amount in quote currency (e.g., USDT)
    price: Optional[float] = None  # For limit orders (legacy)
    limit_price: Optional[float] = None  # For limit orders (preferred)
    order_type: str = "market"  # "market" or "limit" (legacy)
    reason: str = ""

    # Execution layer fields (new)
    execution: Optional[str] = None  # "market", "twap", "vwap" (None defaults to "market")
    execution_params: Optional[dict] = None  # Execution-specific parameters

    # Observability only (not used for trading): optional strategy conviction
    # score and the threshold it was compared against, surfaced in the bot
    # detail "Decision Status" panel when a strategy chooses to populate them.
    score: Optional[float] = None
    threshold: Optional[float] = None

    # Expected price move as a fraction of current price (e.g. 0.005 = 0.5%).
    # Set by directional strategies that can estimate the distance to their
    # profit target. The viability gate checks expected_move > round_trip_fees +
    # safety_margin before executing a BUY on directional signals.
    # Leave as None when is_accumulation=True — the gate uses a fee sanity check
    # instead of requiring a price-target estimate.
    expected_move_pct: Optional[float] = None

    # Expected adverse move to the strategy's own stop-loss, as a fraction of
    # current price (same units as expected_move_pct — the "reward" side).
    # Set by directional strategies that already compute a locked stop at
    # entry (mean_reversion, dip_recovery). When present, the viability gate
    # additionally enforces a minimum reward:risk ratio
    # (expected_move_pct / expected_risk_pct >= _MIN_REWARD_RISK_RATIO).
    # Leave as None for strategies with no fixed price target (trend_following,
    # volatility_breakout ride a trend/breakout with no take-profit level, so a
    # reward:risk ratio is not well-defined) or accumulation strategies — those
    # keep the pre-existing fee-only viability check, unchanged.
    expected_risk_pct: Optional[float] = None

    # True for accumulation strategies (DCA, grid-like) that build position
    # without a per-trade directional price target. The viability gate applies
    # a fee sanity check (fee % < _MAX_ACCUMULATION_FEE_PCT) instead of the
    # directional edge check used for is_accumulation=False signals.
    is_accumulation: bool = False


def evaluate_reward_risk(
    expected_move_pct: Optional[float],
    expected_risk_pct: Optional[float],
    min_ratio: float = _MIN_REWARD_RISK_RATIO,
) -> tuple:
    """Evaluate a directional BUY signal's reward:risk ratio for the viability gate.

    Single implementation of the reward:risk check, used by the central gate in
    _execute_trade and unit-tested independently of a running bot/engine.

    Args:
        expected_move_pct: Reward side — distance to the strategy's profit
            target, as a fraction of price (same convention as the existing
            fee-viability check).
        expected_risk_pct: Risk side — distance to the strategy's own locked
            stop, as a fraction of price. None means the strategy has no fixed
            stop to measure risk against (e.g. a trend-following exit with no
            price target) — the check is a no-op in that case, preserving the
            pre-existing fee-only viability behavior.
        min_ratio: Minimum acceptable reward:risk ratio.

    Returns:
        (ok, reason) — reason is "" when ok is True.
    """
    if expected_risk_pct is None:
        return True, ""
    if not isinstance(expected_risk_pct, (int, float)) or expected_risk_pct <= 0:
        return False, (
            f"invalid risk estimate ({expected_risk_pct}) — stop must be strictly "
            "below entry for a long"
        )
    reward = expected_move_pct if isinstance(expected_move_pct, (int, float)) else 0.0
    if reward <= 0:
        return False, (
            f"invalid reward estimate ({reward * 100:.3f}%) — target must be "
            "strictly above entry for a long"
        )
    ratio = reward / expected_risk_pct
    if ratio < min_ratio:
        return False, (
            f"reward:risk {ratio:.2f} (reward {reward * 100:.3f}% / "
            f"risk {expected_risk_pct * 100:.3f}%) < minimum {min_ratio:.2f}"
        )
    return True, ""


def validate_dip_recovery_params(params: dict) -> list:
    """Validate dip_recovery strategy parameters.

    Returns a list of human-readable error strings (empty when valid). Kept as a
    pure function so it can be reused by configuration checks and tests without
    constructing an engine.

    Args:
        params: The bot's strategy_params dict.
    """
    errors = []

    def _positive(name: str, allow_zero: bool = False) -> None:
        value = params.get(name)
        if value is None:
            return
        if not isinstance(value, (int, float)):
            errors.append(f"{name} must be a number")
            return
        if allow_zero and value < 0:
            errors.append(f"{name} must be >= 0")
        elif not allow_zero and value <= 0:
            errors.append(f"{name} must be > 0")

    for field in (
        "min_drop_percent", "drop_atr_multiplier",
        "min_recovery_percent", "recovery_atr_multiplier",
        "take_profit_atr_multiplier", "trailing_stop_atr_multiplier",
        "emergency_stop_atr_multiplier", "max_position_duration_minutes",
        "setup_expiry_minutes", "risk_percent", "spike_guard_atr_multiplier",
        "reference_high_lookback_ticks", "atr_period", "ema_slope_period",
    ):
        _positive(field)

    for field in ("cooldown_seconds", "loss_cooldown_seconds", "min_ticks_without_new_low"):
        _positive(field, allow_zero=True)

    trailing_mult = params.get("trailing_stop_atr_multiplier")
    emergency_mult = params.get("emergency_stop_atr_multiplier")
    if (
        isinstance(trailing_mult, (int, float))
        and isinstance(emergency_mult, (int, float))
        and emergency_mult <= trailing_mult
    ):
        errors.append(
            f"emergency_stop_atr_multiplier ({emergency_mult}) must be greater than "
            f"trailing_stop_atr_multiplier ({trailing_mult}) - it is meant to be a wider, "
            "last-resort safety net beyond the trailing stop"
        )

    cooldown = params.get("cooldown_seconds")
    loss_cooldown = params.get("loss_cooldown_seconds")
    if (
        isinstance(cooldown, (int, float))
        and isinstance(loss_cooldown, (int, float))
        and loss_cooldown < cooldown
    ):
        errors.append(
            f"loss_cooldown_seconds ({loss_cooldown}) must be >= cooldown_seconds ({cooldown})"
        )

    risk_percent = params.get("risk_percent")
    if isinstance(risk_percent, (int, float)) and not (0 < risk_percent <= 100):
        errors.append("risk_percent must be in (0, 100]")

    return errors


class _DipRecoveryState:
    """String constants for Dip Recovery's lifecycle state machine.

    Persisted as a plain string under state["state"] (JSON-compatible, matching
    the dict-based persistence every other strategy uses - no separate typed
    state system). ENTRY_ARMED is transient: it is only ever reported via
    ExplanationBuilder.state() on the tick a BUY is emitted, and is never itself
    written to the persisted dict (the next persisted value is LONG_OPEN).
    """
    IDLE = "IDLE"
    TRACKING_DROP = "TRACKING_DROP"
    WAITING_REVERSAL = "WAITING_REVERSAL"
    ENTRY_ARMED = "ENTRY_ARMED"
    LONG_OPEN = "LONG_OPEN"
    COOLDOWN = "COOLDOWN"


class _VirtualPosition:
    """Lightweight duck-type substitute for a real Position row.

    Used by RECOVERY_MODE so strategies see an open position and produce
    sensible exit signals instead of triggering infinite BUY loops.
    """

    def __init__(
        self,
        bot_id: int,
        trading_pair: str,
        entry_price: float,
        amount_usd: float,
    ) -> None:
        from ..models.position import PositionSide
        self.id = None
        self.bot_id = bot_id
        self.trading_pair = trading_pair
        self.side = PositionSide.LONG
        self.entry_price = entry_price
        # amount in base asset (e.g. BTC), not USD
        self.amount = amount_usd / entry_price if entry_price > 0 else 0.0
        self.current_price = entry_price
        self.unrealized_pnl = 0.0

    def calculate_unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.amount


class TradingEngine:
    """Engine for executing trading bots."""

    def __init__(self, clock: Optional[Clock] = None):
        """Initialize trading engine.

        Args:
            clock: Time source for every wall-clock read in
                this engine (bar-close detection, cooldown timers, entry/exit
                timestamps). Defaults to SystemClock (real wall-clock time),
                which is what live trading and dry-run both use. The backtest
                engine injects a BacktestClock instead, so each TradingEngine
                instance owns its own clock - no shared/global time state, so
                a backtest cannot affect a live or dry-run engine's clock even
                when run in the same process.
        """
        self.clock: Clock = clock or SystemClock()
        self._running_bots: Dict[int, asyncio.Task] = {}
        self._exchange_services: Dict[int, ExchangeService] = {}
        self._stop_flags: Dict[int, bool] = {}
        self._bot_loggers: Dict[int, BotLoggingService] = {}
        self._last_reconciliation: Optional[datetime] = None
        # Throttle for in-loop pending-order resolution (per bot).
        self._last_pending_resolve: Dict[int, datetime] = {}
        # Throttle for in-loop strategy/simulator state checkpointing (per bot).
        self._last_checkpoint: Dict[int, datetime] = {}
        # Shared ticker cache for dry-run bots: dedupes identical-symbol public
        # API polls across bots in this process (L-C).
        self._shared_ticker_cache: Dict[str, tuple] = {}
        # Recovery mode state per bot (paper trades, consecutive wins, etc.).
        # Persisted to bot.strategy_state["recovery_mode"] so it survives restarts.
        self._recovery_states: Dict[int, dict] = {}
        # Per-bot structured decision-explanation builder for the CURRENT
        # evaluation. Created fresh each tick by _execute_strategy; strategies
        # populate it via self._explain(bot_id). Observe-only.
        self._explanations: Dict[int, ExplanationBuilder] = {}

    def _make_simulated_exchange(self, budget: float) -> SimulatedExchangeService:
        """Construct a dry-run exchange wired to the shared ticker cache (L-C)."""
        from .config import config_service
        ttl = float(config_service.get("trading.ticker_cache_ttl_seconds") or 2.0)
        return SimulatedExchangeService(
            initial_balance=budget,
            ticker_cache_ttl=ttl,
            ticker_cache=self._shared_ticker_cache,
        )

    async def _seed_sim_exchange(self, bot, exchange, session) -> None:
        """Restore dry-run simulator balances and reconcile them with positions.

        Runs whenever a simulated exchange is (re)created for a bot - on both
        start and resume. It (1) re-imports the persisted ``_sim_state`` snapshot
        so balances survive a restart, then (2) guarantees the simulator holds at
        least as much of each base asset as the bot's OPEN positions, which are
        the source of truth for dry-run holdings.

        Without (2), a dry-run bot with an open position but a fresh/lagging
        simulator could not sell: every stop-loss or exit hit "Insufficient
        simulated balance", the position stayed open, and the exit retried every
        tick forever (the Bot 1 stop-loss loop).
        """
        if not hasattr(exchange, "ensure_base_balance"):
            return  # live exchange: real wallet, nothing to reconcile

        sim_state = (getattr(bot, "strategy_state", None) or {}).get("_sim_state")
        if sim_state:
            exchange.import_state(sim_state)

        positions = await self._get_bot_positions(bot.id, session)
        held: Dict[str, float] = {}
        for pos in positions:
            try:
                base = pos.trading_pair.split("/")[0]
            except (AttributeError, IndexError):
                continue
            amount = getattr(pos, "amount", 0.0) or 0.0
            if isinstance(amount, (int, float)) and amount > 0:
                held[base] = held.get(base, 0.0) + float(amount)
        for base, amount in held.items():
            exchange.ensure_base_balance(base, amount)

    async def start_bot(self, bot_id: int) -> bool:
        """Start a trading bot.

        Args:
            bot_id: The bot ID to start

        Returns:
            True if started successfully
        """
        if bot_id in self._running_bots:
            logger.warning(f"Bot {bot_id} is already running")
            return False

        async with async_session_maker() as session:
            result = await session.execute(select(Bot).where(Bot.id == bot_id))
            bot = result.scalar_one_or_none()

            if not bot:
                logger.error(f"Bot {bot_id} not found")
                return False

            if bot.status == BotStatus.RUNNING:
                logger.warning(f"Bot {bot_id} is already in RUNNING status")
                return False

            # Create exchange service
            if bot.is_dry_run:
                exchange = self._make_simulated_exchange(bot.budget)
                # Restore persisted dry-run balances and reconcile them with any
                # open positions so a bot started with an existing holding can
                # actually sell it (otherwise sells fail "Insufficient simulated
                # balance" and the exit loops forever).
                await self._seed_sim_exchange(bot, exchange, session)
            else:
                exchange = ExchangeService()
                if not exchange.has_credentials():
                    raise BotStartError(
                        "Exchange API credentials are required for live trading. "
                        "Set MEXC_API_KEY/MEXC_API_SECRET or config/exchanges.yaml."
                    )

            # The bot must not run if the exchange (or, for dry-run, the
            # public market data API) is unreachable.
            if not await exchange.connect():
                await exchange.disconnect()
                raise BotStartError(
                    "Could not connect to the exchange"
                    + (" public market data API" if bot.is_dry_run else "")
                    + ". Bot not started."
                )

            # Update bot status only after the exchange connection succeeded
            bot.status = BotStatus.RUNNING
            bot.started_at = self.clock.now()
            bot.paused_at = None
            bot.updated_at = self.clock.now()
            await session.commit()

            self._exchange_services[bot_id] = exchange

            # Initialize per-bot file logger
            ensure_bot_log_directory(bot_id)
            self._bot_loggers[bot_id] = BotLoggingService(
                bot_id, bot.name, bot.is_dry_run
            )
            self._bot_loggers[bot_id].log_activity(
                f"Bot started with strategy '{bot.strategy}' on {bot.trading_pair}"
            )

            # Seed an immediate decision status so the UI shows activity at once.
            decision_status_store.update(
                bot_id, DecisionState.EVALUATING,
                reason="Bot started", symbol=bot.trading_pair,
            )

            # Start bot task
            self._stop_flags[bot_id] = False
            task = asyncio.create_task(self._run_bot_loop(bot_id))
            self._running_bots[bot_id] = task

            logger.info(f"Started bot {bot_id} ({bot.name})")
            return True

    async def pause_bot(self, bot_id: int) -> bool:
        """Pause a running bot.

        Args:
            bot_id: The bot ID to pause

        Returns:
            True if paused successfully
        """
        if bot_id not in self._running_bots:
            logger.warning(f"Bot {bot_id} is not running")
            return False

        # Signal stop
        self._stop_flags[bot_id] = True

        # Wait for task to complete
        task = self._running_bots.pop(bot_id)
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except asyncio.TimeoutError:
            task.cancel()

        # Update bot status
        async with async_session_maker() as session:
            result = await session.execute(select(Bot).where(Bot.id == bot_id))
            bot = result.scalar_one_or_none()
            if bot:
                bot.status = BotStatus.PAUSED
                bot.paused_at = self.clock.now()
                bot.updated_at = self.clock.now()
                await session.commit()
                diagnostics_store.record_pause(bot_id, "Manual pause by operator")
                decision_status_store.update(
                    bot_id, DecisionState.PAUSED,
                    reason="Bot paused by operator", symbol=bot.trading_pair,
                )

        logger.info(f"Paused bot {bot_id}")
        return True

    async def stop_bot(self, bot_id: int, cancel_orders: bool = True) -> bool:
        """Stop a bot completely.

        Args:
            bot_id: The bot ID to stop
            cancel_orders: Whether to cancel pending orders

        Returns:
            True if stopped successfully
        """
        # First pause the bot if running
        if bot_id in self._running_bots:
            await self.pause_bot(bot_id)

        async with async_session_maker() as session:
            result = await session.execute(select(Bot).where(Bot.id == bot_id))
            bot = result.scalar_one_or_none()

            if not bot:
                return False

            # Cancel pending orders if requested
            if cancel_orders:
                await self._cancel_pending_orders(bot_id, session)

            # Update status
            bot.status = BotStatus.STOPPED
            bot.updated_at = self.clock.now()
            await session.commit()

        # Disconnect exchange
        if bot_id in self._exchange_services:
            await self._exchange_services[bot_id].disconnect()
            del self._exchange_services[bot_id]

        # L4: a stopped bot does not auto-resume; release its in-memory state.
        self._cleanup_bot_state(bot_id)

        logger.info(f"Stopped bot {bot_id}")
        return True

    async def kill_bot(self, bot_id: int) -> bool:
        """Kill switch for a bot - immediately stop and cancel all orders.

        Args:
            bot_id: The bot ID to kill

        Returns:
            True if killed successfully
        """
        logger.warning(f"Kill switch activated for bot {bot_id}")
        return await self.stop_bot(bot_id, cancel_orders=True)

    async def kill_all_bots(self) -> int:
        """Global kill switch - stop all running bots.

        Returns:
            Number of bots killed
        """
        logger.warning("Global kill switch activated")
        killed = 0
        for bot_id in list(self._running_bots.keys()):
            if await self.kill_bot(bot_id):
                killed += 1
        return killed

    async def _run_bot_loop(self, bot_id: int) -> None:
        """Main bot execution loop.

        Args:
            bot_id: The bot ID to run
        """
        logger.info(f"Bot {bot_id}: Starting execution loop")

        # Observe-only: mark a fresh runtime so the diagnostics panel can show
        # "evaluations during current runtime" and clear any stale pause reason.
        diagnostics_store.start_runtime(bot_id)

        # M1/M2: consecutive-failure circuit breaker with exponential backoff.
        # Reset on every successful iteration; on the threshold, pause + alert
        # instead of spinning at 1 Hz forever.
        from .config import config_service
        max_failures = int(config_service.get("trading.max_consecutive_failures") or 10)
        max_backoff = float(config_service.get("trading.failure_backoff_max_seconds") or 60)
        checkpoint_interval = float(config_service.get("trading.state_checkpoint_seconds") or 60)
        consecutive_failures = 0
        last_error = ""

        def _backoff(n: int) -> float:
            return min(1.0 * (2 ** max(0, n - 1)), max_backoff)

        while not self._stop_flags.get(bot_id, True):
            try:
                async with async_session_maker() as session:
                    # Get bot
                    result = await session.execute(select(Bot).where(Bot.id == bot_id))
                    bot = result.scalar_one_or_none()

                    if not bot or bot.status not in (
                        BotStatus.RUNNING, BotStatus.RECOVERY_MODE
                    ):
                        logger.info(f"Bot {bot_id}: No longer running, stopping loop")
                        break

                    # Restore recovery state from persisted storage on first loop
                    # iteration after a restart (in-memory dict is empty at boot).
                    if bot.id not in self._recovery_states and bot.status == BotStatus.RECOVERY_MODE:
                        ss = bot.strategy_state or {}
                        rm = ss.get("recovery_mode")
                        if rm and rm.get("active"):
                            self._recovery_states[bot.id] = rm
                            logger.info(
                                f"Bot {bot_id}: Restored RECOVERY_MODE from persisted state "
                                f"(entered {rm.get('entered_at', 'unknown')})"
                            )

                    # Initialize services
                    wallet = VirtualWalletService(session)
                    risk_mgr = RiskManagementService(session)

                    # Perform risk checks
                    risk_assessment = await risk_mgr.full_risk_check(bot_id)

                    if risk_assessment.action == RiskAction.PAUSE_BOT:
                        logger.warning(f"Bot {bot_id}: Pausing due to {risk_assessment.reason}")
                        diagnostics_store.record_pause(bot_id, risk_assessment.reason)
                        decision_status_store.update(
                            bot_id, DecisionState.RISK_LIMIT,
                            reason=risk_assessment.reason, symbol=bot.trading_pair,
                        )
                        bot.status = BotStatus.PAUSED
                        bot.paused_at = self.clock.now()
                        await session.commit()
                        self._stop_flags[bot_id] = True

                        # Send email alert
                        email_service.send_bot_paused_alert(
                            bot_id=bot.id,
                            bot_name=bot.name,
                            reason=risk_assessment.reason,
                            pnl=bot.total_pnl,
                            trading_pair=bot.trading_pair,
                        )
                        break

                    if risk_assessment.action == RiskAction.STOP_BOT:
                        logger.warning(f"Bot {bot_id}: Stopping due to {risk_assessment.reason}")
                        decision_status_store.update(
                            bot_id, DecisionState.RISK_LIMIT,
                            reason=risk_assessment.reason, symbol=bot.trading_pair,
                        )
                        bot.status = BotStatus.STOPPED
                        await session.commit()
                        self._stop_flags[bot_id] = True
                        break

                    if risk_assessment.action == RiskAction.ROTATE_STRATEGY:
                        # Get next available strategy
                        new_strategy = await self._get_next_strategy(bot.strategy)
                        await risk_mgr.rotate_strategy(bot_id, new_strategy, risk_assessment.reason)

                    if risk_assessment.action == RiskAction.ENTER_RECOVERY_MODE:
                        if bot.status != BotStatus.RECOVERY_MODE:
                            await self._enter_recovery_mode(bot, bot_id, risk_assessment.reason, session)
                        # Don't break — fall through to paper-trade the strategy signal.

                    # Keep decision status current when already in recovery mode
                    # (entering recovery updates it; this keeps it live on every tick).
                    if bot.status == BotStatus.RECOVERY_MODE and risk_assessment.action != RiskAction.ENTER_RECOVERY_MODE:
                        recovery = self._recovery_states.get(bot.id)
                        if recovery:
                            trades = recovery.get("paper_trades", [])
                            wins = sum(1 for t in trades if t.get("win"))
                            cons = recovery.get("consecutive_paper_wins", 0)
                            decision_status_store.update(
                                bot_id,
                                DecisionState.RECOVERY_MODE_PAPER_TRADING,
                                reason=(
                                    f"{len(trades)} paper trades, {wins} wins, "
                                    f"{cons} consecutive — evaluating {bot.strategy}"
                                ),
                                symbol=bot.trading_pair,
                            )

                    # 7-day stale recovery warning (periodic, not every tick)
                    if bot.status == BotStatus.RECOVERY_MODE:
                        recovery = self._recovery_states.get(bot.id)
                        if recovery:
                            try:
                                entered_at = datetime.fromisoformat(recovery["entered_at"])
                                days_stuck = (self.clock.now() - entered_at).days
                                if days_stuck >= 7:
                                    # Warn roughly once per hour (3600 ticks ≈ 1 hour at 1 Hz)
                                    ticks_since_entry = int(
                                        (self.clock.now() - entered_at).total_seconds()
                                    )
                                    if ticks_since_entry % 3600 < 2:
                                        logger.warning(
                                            f"Bot {bot_id}: RECOVERY_MODE has persisted "
                                            f"for {days_stuck} days — market conditions "
                                            "may not be recovering. Monitoring continues."
                                        )
                            except (KeyError, ValueError):
                                pass

                    # Get exchange service
                    exchange = self._exchange_services.get(bot_id)
                    if not exchange:
                        logger.error(f"Bot {bot_id}: No exchange service")
                        break

                    # Get current market data
                    ticker = await exchange.get_ticker(bot.trading_pair)
                    if not ticker:
                        # M2: a persistent missing ticker (often an exchange
                        # disconnect) feeds the failure breaker so a sustained
                        # outage pauses + alerts instead of looping silently.
                        consecutive_failures += 1
                        last_error = f"market data unavailable for {bot.trading_pair}"
                        diagnostics_store.record_data_failure(
                            bot_id, DATA_UNAVAILABLE, last_error
                        )
                        decision_status_store.update(
                            bot_id, DecisionState.WAITING_FOR_DATA,
                            reason=last_error, symbol=bot.trading_pair,
                        )
                        logger.warning(
                            f"Bot {bot_id}: {last_error} "
                            f"(failure {consecutive_failures}/{max_failures})"
                        )
                        if consecutive_failures >= max_failures:
                            await self._pause_bot_for_failures(
                                bot_id, consecutive_failures, last_error
                            )
                            break
                        await asyncio.sleep(_backoff(consecutive_failures))
                        continue

                    # Generate trading signal from strategy
                    signal = await self._execute_strategy(bot, ticker.last, session)

                    # Publish the engine's current "thinking" to the in-memory
                    # decision-status store (read by the bot detail UI). State
                    # transitions are logged at INFO so evaluations and decision
                    # changes are visible without spamming a line every second.
                    #
                    # This is presentation/observability only and must never affect
                    # trading: a formatting or logging error here is isolated so it
                    # cannot bubble to the failure circuit breaker below and pause an
                    # otherwise-healthy bot.
                    try:
                        # Observe-only diagnostics: count this evaluation and the
                        # signal/decision-reason it produced (store methods are
                        # internally exception-safe; this can never pause a bot).
                        diagnostics_store.record_evaluation(bot_id)
                        diagnostics_store.record_signal(bot_id, signal)
                        changed = decision_status_store.update_from_signal(
                            bot_id, signal, symbol=bot.trading_pair
                        )
                        status = decision_status_store.get(bot_id)
                        log = logger.info if changed else logger.debug
                        log(
                            "Bot %s decision: %s @ %s — %s",
                            bot_id,
                            status.state if status else "?",
                            ticker.last,
                            status.reason if status else "",
                        )
                    except Exception as status_err:  # noqa: BLE001 - presentation only
                        logger.warning(
                            "Bot %s: failed to publish decision status (non-fatal): %s",
                            bot_id, status_err,
                        )

                    if signal and signal.action != "hold":
                        # CR-1: budget validation gates BUYS only (capital
                        # deployment). Sells reduce exposure and must never be
                        # blocked by the budget; they are validated against the
                        # open position inside _execute_trade.
                        # In RECOVERY_MODE no real capital is spent, so validation
                        # is bypassed — the paper trade sets its own paper amount.
                        proceed = True
                        if signal.action == "buy" and bot.status != BotStatus.RECOVERY_MODE:
                            validation = await wallet.validate_trade(bot_id, signal.amount)
                            proceed = validation.is_valid
                            if not proceed:
                                diagnostics_store.record_blocked(
                                    bot_id, BLOCK_INSUFFICIENT_BALANCE, validation.reason
                                )
                                logger.warning(
                                    f"Bot {bot_id}: Trade rejected - {validation.reason}"
                                )

                        if proceed:
                            # Reflect the actual order intent in the status panel.
                            decision_status_store.update(
                                bot_id,
                                DecisionState.ENTERING_POSITION
                                if signal.action == "buy"
                                else DecisionState.EXITING_POSITION,
                                reason=signal.reason,
                                symbol=bot.trading_pair,
                                score=signal.score,
                                threshold=signal.threshold,
                            )
                            if bot.status == BotStatus.RECOVERY_MODE:
                                # Paper-trade: simulate the order without touching
                                # the real exchange or committing real capital.
                                await self._process_paper_trade(
                                    bot, bot_id, signal, ticker.last, session
                                )
                            else:
                                # Execute trade
                                await self._execute_trade(
                                    bot, exchange, signal, ticker.last, session
                                )

                    # C1: Enforce per-position stop-loss on EVERY iteration, not
                    # only after a trade. While the strategy holds, the price-based
                    # stop-loss is the only safety net that catches an open
                    # position's unrealized loss (drawdown/daily-loss checks are
                    # realized-only). Running it unconditionally makes stop-loss
                    # enforcement deterministic.
                    await self._check_positions_stop_loss(
                        bot_id, exchange, risk_mgr, session
                    )

                    # Take P&L snapshot periodically
                    await self._take_pnl_snapshot(bot_id, session)

                    # H2: resolve any orders left pending (orphaned resting limit
                    # orders, or market orders the exchange confirms late) against
                    # the exchange. Throttled so we do not poll every second.
                    now = self.clock.now()
                    last_resolve = self._last_pending_resolve.get(bot_id)
                    if last_resolve is None or (now - last_resolve).total_seconds() >= 30:
                        self._last_pending_resolve[bot_id] = now
                        await self._resolve_pending_orders(bot_id, exchange, session)

                    # H-1: periodically checkpoint strategy + simulator state so a
                    # crash (not just a graceful shutdown) loses minimal state on
                    # resume - trailing stops, cooldowns, and dry-run balances.
                    last_ckpt = self._last_checkpoint.get(bot_id)
                    if last_ckpt is None or (now - last_ckpt).total_seconds() >= checkpoint_interval:
                        self._last_checkpoint[bot_id] = now
                        await self._save_bot_state(bot_id, session)
                        await session.commit()

                    # Reconcile exchange balances for live trading (throttled,
                    # alert-only). Dry-run bots have nothing to reconcile.
                    if not bot.is_dry_run:
                        await self._reconcile_live_account(exchange, session)

                # Iteration completed successfully - reset the failure breaker.
                consecutive_failures = 0

            except ValidationError as e:
                # Accounting validation failure - MUST stop bot
                logger.critical(
                    f"Bot {bot_id}: Accounting validation failed. STOPPING BOT. Error: {e}",
                    exc_info=True
                )
                async with async_session_maker() as session:
                    result = await session.execute(select(Bot).where(Bot.id == bot_id))
                    bot = result.scalar_one_or_none()
                    if bot:
                        bot.status = BotStatus.STOPPED
                        await session.commit()
                        # M2: alerting - this halts trading and needs attention.
                        await self._emit_alert(
                            session, bot_id, "accounting_validation_failure",
                            f"Bot {bot_id} STOPPED: accounting validation failed: {e}",
                            email_subject=f"TradingBot: accounting failure on bot {bot_id}",
                        )
                self._stop_flags[bot_id] = True
                break  # Exit loop immediately

            except Exception as e:
                # M1: count consecutive failures, back off exponentially, and trip
                # the circuit breaker (pause + alert) rather than spinning forever.
                consecutive_failures += 1
                last_error = str(e)
                logger.error(
                    f"Bot {bot_id}: Error in execution loop "
                    f"(failure {consecutive_failures}/{max_failures}): {e}"
                )
                if consecutive_failures >= max_failures:
                    await self._pause_bot_for_failures(bot_id, consecutive_failures, last_error)
                    break
                await asyncio.sleep(_backoff(consecutive_failures))
                continue

            # Sleep before next iteration
            await asyncio.sleep(1)  # 1 second between iterations

        logger.info(f"Bot {bot_id}: Execution loop ended")

    async def _execute_strategy(
        self,
        bot: Bot,
        current_price: float,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Execute trading strategy to generate signal.

        Args:
            bot: The bot model
            current_price: Current market price
            session: Database session

        Returns:
            TradeSignal or None
        """
        strategy_name = bot.strategy
        params = bot.strategy_params or {}

        # Get strategy executor
        executor = self._get_strategy_executor(strategy_name)
        if not executor:
            return None

        # Fresh structured-explanation builder for THIS evaluation. Strategies
        # populate it (self._explain(bot.id)) at the points where the numbers are
        # computed; we finalize decision+reason from the returned signal and
        # record it for the diagnostics API. Wrapped so explanation handling can
        # never affect the trading signal.
        builder = ExplanationBuilder(strategy_name)
        self._explanations[bot.id] = builder

        signal = await executor(bot, current_price, params, session)

        try:
            builder.finalize(signal)
            diagnostics_store.record_explanation(bot.id, builder.to_dict())
        except Exception as exc:  # noqa: BLE001 - observability must never throw
            logger.debug("Bot %s: explanation record failed (non-fatal): %s", bot.id, exc)

        return signal

    def _explain(self, bot_id: int) -> ExplanationBuilder:
        """Return the current evaluation's explanation builder for a bot.

        Always returns a builder: if none is set (e.g. a strategy invoked
        directly in a test rather than via _execute_strategy), a detached
        throwaway is created so strategy call sites never need a None check.
        """
        builder = self._explanations.get(bot_id)
        if builder is None:
            builder = ExplanationBuilder("unknown")
            self._explanations[bot_id] = builder
        return builder

    def _decision_explanation_for_order(self, bot_id: int) -> tuple:
        """Summarized (decision_explanation, edge_management_category) for
        the order about to be persisted, from THIS cycle's explanation
        builder (add-strategy-decision-framework, Phase 0.6 - closes the
        Pillar 8 persistence gap: a historical trade should be explainable
        from the DB, not only from the in-memory DiagnosticsStore).

        Never raises: observability must never affect a trading decision or
        block an order from being recorded.
        """
        try:
            builder = self._explanations.get(bot_id)
            if builder is None:
                return None, None
            explanation = builder.to_dict()
            summary = summarize_decision_explanation(explanation)
            edge_category = extract_edge_management_category(explanation)
            return summary, edge_category
        except Exception as exc:  # noqa: BLE001 - observability must never throw
            logger.debug("Bot %s: decision explanation summarization failed (non-fatal): %s", bot_id, exc)
            return None, None

    def _get_strategy_executor(
        self,
        strategy_name: str,
    ) -> Optional[Callable[[Bot, float, dict, AsyncSession], Awaitable[Optional[TradeSignal]]]]:
        """Get strategy executor function.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy executor function or None

        Note:
            VWAP and TWAP are execution algorithms, not strategies.
            They are intentionally excluded from strategy selection.
        """
        strategies = {
            "dca_accumulator": self._strategy_dca,
            "adaptive_grid": self._strategy_grid,
            "mean_reversion": self._strategy_mean_reversion,
            "trend_following": self._strategy_trend_following,
            "volatility_breakout": self._strategy_volatility_breakout,
            "dip_recovery": self._strategy_dip_recovery,
                    "auto_mode": self._strategy_auto,
        }

        # Defensive: Block execution-only algorithms from being used as strategies
        if strategy_name in ["vwap", "twap"]:
            logger.error(
                f"Attempted to use execution algorithm '{strategy_name}' as a strategy. "
                "VWAP/TWAP are execution methods, not alpha strategies. "
                "This is a configuration error."
            )
            return None

        return strategies.get(strategy_name)

    async def _strategy_dca(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """DCA (Dollar Cost Averaging) strategy - Institutional Grade.

        CLASSIC, DIRECTION-AGNOSTIC accumulation: buys fixed-size chunks at
        regular clock-driven intervals *no matter the market conditions*. It
        takes NO view on short- or medium-term price direction and buys the dip
        and the rip alike. Continues indefinitely until balance is exhausted,
        the long-term thesis is invalidated, or the bot is manually stopped.

        This is the project's REFERENCE accumulation strategy and long-term
        benchmark (see audits/dca_accumulator.md, Strategy Decision Framework
        Phase 6). Its value is deterministic: fixed dollar chunks make the
        average entry cost the harmonic mean of purchase prices (<= arithmetic
        mean, for every price path, no forecast required). Keep it simple and
        deterministic on purpose - do NOT make it "smarter".

        IMPORTANT: This strategy is INFINITE by design. It will keep buying until:
        - Balance falls below minimum order size (accumulation complete)
        - The long-term investment thesis is invalidated (thesis_invalidated)
        - Bot is manually stopped by operator
        This is intentional, not a bug.

        DCA NEVER SELLS. Exits are handled by other strategies or manual intervention.

        SUITABILITY (Pillar 2, Phase 6): DCA does NOT gate on market direction.
        The only things that may stop/defer/limit a scheduled buy are:
          (a) execution feasibility - enforced downstream by the shared
              MIN_ORDER_USD floor and the fee/spread-adjusted balance cap below;
          (b) portfolio constraints - enforced downstream by the execution
              pipeline's PortfolioRiskService.check_portfolio_risk and by the
              budget-exhaustion floor below;
          (c) long-term-thesis invalidation - the thesis_invalidated structural
              stop below (a non-price condition; NEVER a direction forecast).
        A trend-regime pause is deliberately NOT part of classic DCA - it is
        market timing. It survives only as an explicit, off-by-default,
        non-classic operator overlay (regime_filter_enabled), see below.

        SIZING (Pillar 5, Phase 6.5): FLAT and SCHEDULE-DRIVEN by design. Each
        buy is a fixed chunk - a fixed % of balance (amount_percent) or a fixed
        USD amount (amount_usd) - and takes NO Decision Score, no price, no
        regime, and no expected-future-move input. Classic DCA does not try to
        maximise return; it deploys capital consistently, so deterministic flat
        sizing IS the correct implementation, not a gap (it is also what makes
        DCA a clean benchmark). The ONLY legitimate adjustments are objective
        portfolio-governance constraints, and those are enforced DOWNSTREAM by
        the execution pipeline (PortfolioRiskService resize/block at STEP 3,
        StrategyCapacityService at STEP 4) plus the fee-adjusted balance cap and
        MIN_ORDER_USD floor below - never re-implemented here, and never an
        increase or a directional adjustment.

        PROPOSAL (Pillar 10, Phase 6.7): builds a StrategyProposal per tick and
        routes it through the StandaloneAdapter (still returns TradeSignal). A
        scheduled buy is BUY + OPEN_POSITION (first ever) or ADD_TO_POSITION
        (subsequent); every non-buying tick is NO_TRADE + NO_ACTION (a pure
        accumulator has no managed position, so HOLD_POSITION would misrepresent
        it). DCA never emits SELL. validity is tied to the buy interval;
        assumptions are objective execution/portfolio/thesis conditions, never a
        direction forecast.

        Auto Mode may select this strategy (it is the designated fallback -
        see _get_strategy_capabilities): that is a SUPERVISED call, routed
        through _strategy_auto's own eligibility/regime/scoring gate, and is
        allowed. What remains blocked is an UNSUPERVISED direct call on an
        auto_mode bot (bypassing that gate entirely) - DCA's clock-driven,
        never-sell behaviour is still a poor fit run that way.

        Parameters:
            interval_minutes: Time between buys (default: 60)
            amount_percent: Percent of budget per buy (default: 10)
            amount_usd: Fixed USD amount per buy (overrides amount_percent if set)
            immediate_first_buy: Execute first buy immediately (default: True)
            thesis_invalidated: Structural stop - halt all accumulation when the
                long-term thesis is no longer valid (default: False). Not a price signal.
            regime_filter_enabled: NON-CLASSIC market-timing overlay (default: False).
                When True, pauses buying in disallowed trend regimes - departs from
                classic DCA. Off by default; classic DCA buys through downtrends.
            allowed_regimes: Only used when regime_filter_enabled=True
                (default: ["trend_up", "trend_flat"])
        """
        # Defensive check: block DIRECT/unsupervised use of DCA on an
        # auto_mode bot. bot.strategy alone can't tell us that - it reads
        # "auto_mode" for every Auto-managed bot regardless of which
        # sub-strategy Auto has actually selected, since Auto dispatches
        # sub-strategies using the same bot object. _strategy_auto marks its
        # own dispatch calls explicitly via params["_invoked_by_auto"], so a
        # supervised call (Auto selected DCA itself) is let through while a
        # call that bypasses Auto's gate entirely is still blocked.
        if bot.strategy == "auto_mode" and not params.get("_invoked_by_auto"):
            logger.warning(
                f"Bot {bot.id}: DCA strategy invoked directly on an auto_mode bot, "
                f"bypassing Auto Mode's own supervision. This is NOT recommended - "
                f"DCA conflicts with unsupervised regime-based rotation."
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason="DCA: Not intended for direct/unsupervised use inside auto_mode"
            )

        interval_minutes = params.get("interval_minutes", 60)
        amount_percent = params.get("amount_percent", 10) / 100
        amount_usd = params.get("amount_usd")  # Fixed amount in USD
        immediate_first_buy = params.get("immediate_first_buy", True)
        thesis_invalidated = params.get("thesis_invalidated", False)
        # NON-CLASSIC market-timing overlay - OFF by default (see docstring
        # SUITABILITY note and audits/dca_accumulator.md Phase 6.1). A classic
        # DCA is direction-agnostic; this only exists as an explicit operator
        # opt-in.
        regime_filter_enabled = params.get("regime_filter_enabled", False)
        allowed_regimes = params.get("allowed_regimes", ["trend_up", "trend_flat"])

        now = self.clock.now()
        # validity.valid_until is tied to the BUY INTERVAL (not a candle
        # timeframe) - a proposal is meaningful until the next scheduled buy
        # (Pillar 10, Phase 6.7). The validity window is floored at 1s so a
        # zero-interval config (interval_minutes=0, used by backtests to buy
        # every tick) still yields a strictly-positive validity the frozen
        # ProposalValidity contract requires.
        interval_seconds = interval_minutes * 60
        _validity_interval_seconds = max(interval_seconds, 1)

        # === STRATEGY EDGE MANAGEMENT + SUITABILITY GATE (Pillars 7 & 2) ===
        # A classic DCA does NOT fail because the market falls - it fails ONLY
        # when its accumulation thesis is objectively invalidated. DcaEdgeManager
        # therefore monitors the health of the ACCUMULATION PROCESS (thesis +
        # operational state), never mark-to-market profitability: it has no
        # price/pnl/trend input at all, so an ordinary drawdown can never be
        # reinterpreted as loss of edge (Phase 6.1/6.4; design.md "Pure-
        # accumulator exception"). Category C = objective thesis invalidation
        # (the ONLY halt on suitability grounds; a non-price, structural stop).
        # Execution-feasibility and portfolio-constraint suitability are enforced
        # downstream (the MIN_ORDER_USD floor + fee-adjusted cap below, and the
        # execution pipeline's PortfolioRiskService) - not duplicated here.
        if not hasattr(self, "_dca_edge_manager"):
            self._dca_edge_manager = DcaEdgeManager()
        thesis_conditions = {
            "operator_invalidated": bool(thesis_invalidated or params.get("operator_invalidated", False)),
            "asset_delisted": bool(params.get("asset_delisted", False)),
            "fundamental_failure": bool(params.get("fundamental_failure", False)),
            "regulatory_impossibility": bool(params.get("regulatory_impossibility", False)),
            "portfolio_withdrawn": bool(params.get("portfolio_withdrawn", False)),
        }
        edge_status = self._dca_edge_manager.evaluate(
            thesis_invalidation=thesis_conditions, now=now,
        )
        _edge_exp = self._explain(bot.id)
        _edge_exp.metric("edge_status_category", edge_status.category.value)
        _edge_exp.check(
            "Accumulation thesis intact", edge_status.category.value,
            f"!= {EdgeCategory.C.value}", edge_status.category != EdgeCategory.C,
            detail=edge_status.reason,
        )

        # DCA is direction-agnostic (Pillar 2 re-scoped in 6.3): regime is NOT a
        # suitability input, so the default MarketSuitabilityResult is permissive
        # (allowed=["all"]). The non-classic overlay below can replace it when an
        # operator opts in. Every proposal still carries a real
        # MarketSuitabilityResult (mirrors adaptive_grid's regime_filter_enabled
        # =False -> allowed=["all"] convention).
        suitability = MarketSuitabilityGate().evaluate({}, ["all"])

        # === Proposal helpers (Pillar 10, Phase 6.7) ===
        # DCA has no conviction/Decision Score (6.3/6.5). The frozen
        # StrategyProposal contract nonetheless requires a decision_score field,
        # so - exactly as adaptive_grid does for its non-scored branches - each
        # proposal carries a single DESCRIPTIVE, deterministic evidence item that
        # restates the schedule/thesis/operational decision DCA already made. It
        # is NOT a price/conviction factor and introduces no scoring intelligence.
        def _dca_proposal(
            *, direction: "Direction", execution_intent: "ExecutionIntent",
            evidence_name: str, evidence_value: float, evidence_reason: str,
            suggested_position_size: Optional[float] = None, assumptions: tuple = (),
            edge: Optional["EdgeStatus"] = None,
            market_suitability: Optional["MarketSuitabilityResult"] = None,
            expected_holding_horizon: Optional[str] = None,
        ) -> StrategyProposal:
            item = EvidenceItem(
                name=evidence_name, measurement=lambda d: evidence_value,
                normalization=lambda r: r, weight=100.0, reason=evidence_reason,
            )
            score = DecisionScoreEngine().score("dca_accumulator", [item], {}, threshold=0.0)
            reasons_for, reasons_against = derive_reasons(score)
            return StrategyProposal(
                strategy_id="dca_accumulator", bot_id=bot.id, generated_at=now,
                direction=direction, execution_intent=execution_intent,
                validity=ProposalValidity(
                    generated_at=now, valid_until=now + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=score,
                market_suitability=(market_suitability if market_suitability is not None else suitability),
                edge_status=(edge if edge is not None else edge_status),
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=suggested_position_size,
                suggested_risk_budget_pct=(None if (amount_usd and amount_usd > 0) else amount_percent * 100.0),
                expected_holding_horizon=expected_holding_horizon,
                adaptive_parameters_used={},  # flat by design - DCA has no adaptive params
                explanation=self._explain(bot.id).to_dict(),
            )

        def _no_trade(
            evidence_name: str, evidence_reason: str, reason: str, *,
            assumptions: tuple = (), edge: Optional["EdgeStatus"] = None,
            market_suitability: Optional["MarketSuitabilityResult"] = None,
        ) -> TradeSignal:
            # A non-buying tick makes no ENTRY decision this interval -> NO_TRADE
            # + NO_ACTION (a pure accumulator has no actively-managed position, so
            # HOLD/HOLD_POSITION would misrepresent it - tasks.md 6.7). The adapter
            # returns None for a no-order intent, so the explicit hold reason is
            # preserved unchanged.
            proposal = _dca_proposal(
                direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                evidence_name=evidence_name, evidence_value=0.0, evidence_reason=evidence_reason,
                assumptions=assumptions, edge=edge, market_suitability=market_suitability,
            )
            sig = StandaloneAdapter.to_trade_signal(proposal)
            return sig if sig is not None else TradeSignal(action="hold", amount=0, reason=reason)

        if edge_status.should_stop:  # Category C - objective thesis invalidation
            logger.info(
                f"Bot {bot.id}: DCA HALTED (Pillar 7 Category C) - {edge_status.reason}. "
                f"Structural stop, not a price-direction signal; requires re-certification."
            )
            _edge_exp.state("THESIS_INVALIDATED").update({"current_price": current_price})
            return _no_trade(
                "Accumulation thesis intact", f"thesis invalidated: {edge_status.reason}",
                f"DCA: Halted - {edge_status.reason} (thesis invalidated, structural stop)",
                assumptions=("Long-term investment thesis remains valid",),
                edge=edge_status,
            )

        # === REGIME FILTER (NON-CLASSIC market-timing overlay, off by default) ===
        # This is market timing - it pauses buying on expected short-term
        # weakness - and departs from classic DCA. Retained ONLY as an explicit
        # operator opt-in (regime_filter_enabled=True). A classic accumulator
        # buys through downtrends; do not enable this to "improve" DCA.
        if regime_filter_enabled:
            logger.info(
                f"Bot {bot.id}: DCA non-classic market-timing overlay is ENABLED "
                f"(regime_filter_enabled=True) - this departs from classic DCA."
            )
            # Get price history for regime detection
            price_history = self._get_price_history(bot.id)

            # Add current price to history for regime detection
            price_history_with_current = price_history + [{
                "timestamp": self.clock.now().isoformat(),
                "price": current_price
            }]

            # Detect current market regime
            current_regime = self._detect_market_regime(price_history_with_current, None)
            trend_state = current_regime.get("trend_state", "flat")

            # Map trend_state to regime names (for user-friendly config)
            # trend_state values: "up", "down", "flat"
            # user config values: "trend_up", "trend_down", "trend_flat"
            trend_regime_name = f"trend_{trend_state}"

            # Check if current regime is allowed
            if trend_regime_name not in allowed_regimes:
                logger.info(
                    f"Bot {bot.id}: DCA PAUSED by regime filter - "
                    f"Current regime: {trend_regime_name}, Allowed: {allowed_regimes}"
                )
                self._explain(bot.id).state("WAITING_REGIME").update({
                    "current_price": current_price,
                    "regime": trend_regime_name,
                }).check(
                    "Regime allows DCA", trend_regime_name,
                    f"in [{', '.join(allowed_regimes)}]", False,
                )
                # The overlay's regime block carries a real (non-permissive)
                # MarketSuitabilityResult on this proposal so the opt-in pause is
                # explainable; still NO_TRADE + NO_ACTION.
                overlay_suitability = MarketSuitabilityResult(
                    is_suitable=False, regime_tags=[trend_regime_name],
                    allowed_regimes=list(allowed_regimes), matched_tags=[],
                    reason=f"non-classic overlay: {trend_regime_name} not in {allowed_regimes}",
                )
                return _no_trade(
                    "Regime allows DCA (non-classic overlay)",
                    f"overlay paused buying in {trend_regime_name}",
                    f"DCA: Paused (regime={trend_regime_name}, waiting for {allowed_regimes})",
                    market_suitability=overlay_suitability,
                )

        # Get order history for this bot
        last_order = await self._get_last_order(bot.id, session)
        order_count = await self._get_order_count(bot.id, session)

        # === TIME-BASED INTERVAL LOGIC (hardened) ===
        # Ensures clock-stable behavior: one buy max per interval, no catch-up

        if last_order:
            # Defensive: If last order timestamp is in the future, treat as no last order
            # This handles clock skew or bad data gracefully
            if last_order.created_at > now:
                logger.warning(
                    f"Bot {bot.id}: DCA last order timestamp is in future "
                    f"({last_order.created_at} > {now}). Treating as no previous order."
                )
                last_order = None

        # --- Structured decision explanation (observe-only) -----------------
        # DCA here is CLOCK-driven (buy every interval regardless of price), not
        # a price/drawdown ladder — so it exposes the interval timer and capital
        # state, not an average-entry/drawdown trigger this version does not use.
        _dca_exp = self._explain(bot.id)
        _secs_since = (now - last_order.created_at).total_seconds() if last_order else None
        _interval_ok = (_secs_since is None) or (_secs_since >= interval_seconds)
        _remaining = max(0.0, interval_seconds - _secs_since) if _secs_since is not None else 0.0
        _dca_exp.update({
            "current_price": current_price,
            "order_count": order_count,
            "interval_minutes": interval_minutes,
            "interval_seconds": interval_seconds,
            "seconds_since_last_order": round(_secs_since, 1) if _secs_since is not None else None,
            "time_until_next_buy_s": round(_remaining, 1),
            "current_balance": bot.current_balance,
            "budget": bot.budget,
            "allocated_capital": round((bot.budget or 0.0) - (bot.current_balance or 0.0), 2),
            "remaining_capital": bot.current_balance,
        })
        _dca_exp.check(
            "Interval elapsed",
            (f"{_secs_since:.0f}s since last buy" if _secs_since is not None else "no prior order"),
            f">= {interval_seconds}s", _interval_ok,
            detail=(f"next buy in {_remaining / 60:.1f} min" if not _interval_ok else "buy due"),
        )
        if _interval_ok:
            _dca_exp.state("INTERVAL_DUE")
        else:
            _dca_exp.state("WAITING_INTERVAL").next_trade(
                current=round(_secs_since, 1) if _secs_since is not None else 0,
                current_label="Elapsed since last buy (s)",
                target=interval_seconds, target_label="Interval (s)",
                distance=round(_remaining, 1),
                status=f"next buy in {_remaining / 60:.1f} min",
            )
        # ---------------------------------------------------------------------

        if last_order:
            time_since_last = now - last_order.created_at
            seconds_since_last = time_since_last.total_seconds()

            if seconds_since_last < interval_seconds:
                remaining_seconds = interval_seconds - seconds_since_last
                next_buy_time = last_order.created_at + timedelta(seconds=interval_seconds)
                return _no_trade(
                    "Interval elapsed", "buy interval not yet elapsed - no entry decision this tick",
                    f"DCA: Next buy in {remaining_seconds/60:.1f} min (at {next_buy_time.strftime('%H:%M:%S')})",
                )
        elif not immediate_first_buy:
            # No orders yet but not immediate - check time since bot started
            # Defensive: If bot.started_at is None, allow immediate buy (treat as edge case)
            if bot.started_at:
                time_since_start = now - bot.started_at
                if time_since_start.total_seconds() < interval_seconds:
                    remaining_seconds = interval_seconds - time_since_start.total_seconds()
                    first_buy_time = bot.started_at + timedelta(seconds=interval_seconds)
                    return _no_trade(
                        "Interval elapsed", "first-buy interval not yet elapsed - no entry decision this tick",
                        f"DCA: First buy in {remaining_seconds/60:.1f} min (at {first_buy_time.strftime('%H:%M:%S')})",
                    )
            else:
                # Edge case: bot.started_at is missing - allow immediate buy
                logger.info(
                    f"Bot {bot.id}: DCA bot.started_at is None. "
                    f"Allowing immediate first buy (edge case handling)."
                )

        # === AMOUNT CALCULATION (Pillar 5: flat, schedule-driven, deterministic) ===
        # DELIBERATELY flat: a fixed USD chunk or a fixed % of balance, with NO
        # Decision Score, price, regime, or expected-move input. This is not a
        # missing feature - a classic DCA deploys capital consistently rather
        # than trying to time or size by conviction (see docstring SIZING note,
        # audit Pillar 5). Objective portfolio governance is the only legitimate
        # adjuster and is enforced downstream (PortfolioRiskService / capacity),
        # not here. The cap below is _BUY_BALANCE_FRACTION of balance (not the
        # full balance) so the simulated fee (cost * 0.1 %) + bid/ask spread
        # cannot push the total deduction over available funds.
        max_buy = bot.current_balance * _BUY_BALANCE_FRACTION
        # Track the sizing branch taken so Pillar 8 can explain HOW the amount
        # was reached, not just the final number (Phase 6.6).
        _sizing_basis = "fixed_usd" if (amount_usd and amount_usd > 0) else "percent"
        if amount_usd and amount_usd > 0:
            # Use fixed USD amount
            buy_amount = min(amount_usd, max_buy)
        else:
            # Use percentage of current balance
            buy_amount = bot.current_balance * amount_percent
        _chunk_raw = buy_amount  # pre-floor/cap chunk, for diagnostics
        _sizing_floored = False
        _sizing_capped = False

        # === MINIMUM ORDER FLOOR (executable, shared $10 minimum) ===
        # A DCA buy must clear the same MIN_ORDER_USD the execution layer enforces
        # (_execute_trade STEP 6). Emitting a sub-minimum buy here would be
        # rejected downstream every tick without ever recording an order, so the
        # interval gate never advances and the same "$9.95 REJECTED" repeats
        # forever. Floor up to the minimum when affordable; otherwise HOLD - the
        # infinite accumulation has reached its natural end (budget exhausted).
        if buy_amount < MIN_ORDER_USD:
            if bot.current_balance >= MIN_ORDER_USD:
                # Operational parameter adaptation (Pillar 7 Category B): the
                # configured chunk is below the executable minimum, so adapt it
                # up to MIN_ORDER_USD. Operational, not directional.
                buy_amount = MIN_ORDER_USD
                _sizing_floored = True
                _b_edge = self._dca_edge_manager.evaluate(
                    operational_adaptation=(
                        f"chunk floored to ${MIN_ORDER_USD:.0f} executable minimum"
                    ),
                    now=now,
                )
                self._explain(bot.id).metric("edge_status_category", _b_edge.category.value)
            else:
                # Temporary operational pause (Pillar 7 Category A): capital is
                # unavailable this cycle (below the minimum order). Not a thesis
                # failure - accumulation resumes if the balance is topped up.
                _a_edge = self._dca_edge_manager.evaluate(
                    operational_pause=(
                        f"balance ${bot.current_balance:.2f} below "
                        f"${MIN_ORDER_USD:.0f} minimum order (capital unavailable)"
                    ),
                    now=now,
                )
                logger.info(
                    f"Bot {bot.id}: DCA infinite accumulation complete - "
                    f"Balance ${bot.current_balance:.2f} < ${MIN_ORDER_USD:.0f} minimum order"
                )
                self._explain(bot.id).state("ACCUMULATION_PAUSED").update({
                    "edge_status_category": _a_edge.category.value,
                    "sizing_basis": _sizing_basis,
                    "chunk_raw": round(_chunk_raw, 2),
                }).check(
                    "Buy clears minimum order size",
                    f"${bot.current_balance:.2f} balance", f">= ${MIN_ORDER_USD:.0f}", False,
                    detail="capital unavailable - accumulation paused (Category A, operational)",
                )
                return _no_trade(
                    "Buy clears minimum order size",
                    "capital unavailable this cycle - accumulation paused (operational)",
                    (
                        f"DCA: balance ${bot.current_balance:.2f} below "
                        f"${MIN_ORDER_USD:.0f} minimum order (accumulation complete)"
                    ),
                    edge=_a_edge,
                )

        # Defensive: Cap at fee-adjusted balance ceiling so cost + fee cannot
        # exceed available funds regardless of the amount_percent path above.
        if buy_amount > max_buy:
            buy_amount = max_buy
            _sizing_capped = True

        # === EXECUTE BUY (infinite accumulation continues) ===
        logger.info(
            f"Bot {bot.id}: DCA buy #{order_count + 1} - "
            f"${buy_amount:.2f} ({"fixed" if amount_usd else f"{amount_percent*100:.1f}%"}) "
            f"at ${current_price:.2f} | "
            f"Balance: ${bot.current_balance:.2f} → ${bot.current_balance - buy_amount:.2f}"
        )

        # Pillar 8 (Phase 6.6): explain the buy-amount branching, not just the
        # final number - which sizing basis, the raw chunk, whether it was
        # floored to the minimum or capped to the fee-adjusted budget, and the
        # two execution-feasibility gates the buy clears.
        self._explain(bot.id).state("BUYING").update({
            "sizing_basis": _sizing_basis,
            "chunk_raw": round(_chunk_raw, 2),
            "sizing_floored": _sizing_floored,
            "sizing_capped": _sizing_capped,
            "buy_amount": round(buy_amount, 2),
        }).check(
            "Buy clears minimum order size", round(buy_amount, 2),
            f">= ${MIN_ORDER_USD:.0f}", buy_amount >= MIN_ORDER_USD,
            detail=("floored up to minimum" if _sizing_floored else "chunk above minimum"),
        ).check(
            "Buy within fee-adjusted budget", round(buy_amount, 2),
            f"<= ${max_buy:.2f}", buy_amount <= max_buy + 1e-9,
            detail=("capped to fee-adjusted budget" if _sizing_capped
                    else f"flat {_sizing_basis} chunk within budget"),
        )

        # Pillar 10 (Phase 6.7): a due, ungated buy is an accumulation deployment.
        # The FIRST buy opens the position (OPEN_POSITION); every subsequent buy
        # adds to it (ADD_TO_POSITION) - DCA never emits SELL (never-sell). The
        # assumptions are objective/falsifiable and execution/portfolio/thesis-
        # based (never a direction forecast). Routed through the Standalone
        # Adapter with is_accumulation=True so the execution outcome (action,
        # amount, accumulation flag, market order type) is identical to the
        # pre-migration TradeSignal path.
        _buy_intent = (
            ExecutionIntent.OPEN_POSITION if order_count == 0
            else ExecutionIntent.ADD_TO_POSITION
        )
        buy_proposal = _dca_proposal(
            direction=Direction.BUY, execution_intent=_buy_intent,
            evidence_name="Scheduled accumulation due", evidence_value=1.0,
            evidence_reason=(
                "buy interval elapsed and long-term thesis valid - deploy the "
                "scheduled fixed chunk (direction-agnostic)"
            ),
            suggested_position_size=buy_amount,
            assumptions=(
                "Long-term investment thesis remains valid",
                f"Buy clears the ${MIN_ORDER_USD:.0f} minimum order size",
                "Buy fits within available balance after fees",
            ),
            expected_holding_horizon="indefinite",
        )
        sig = StandaloneAdapter.to_trade_signal(buy_proposal, is_accumulation=True)
        return sig if sig is not None else TradeSignal(
            action="hold", amount=0, reason="DCA: adapter produced no order",
        )

    async def _strategy_grid(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Adaptive Grid trading strategy - Institutional Grade.

        CAPITAL-BOUNDED, REGIME-AWARE, LONG-BIASED grid for crypto spot markets.

        Migrated to the Strategy Decision Framework
        (add-strategy-decision-framework, Phase 5 - see this change's
        design.md/tasks.md and audits/adaptive_grid.md). This is the most
        mechanically complex of the six (virtual grid inventory, not a single
        position); Phase 5 adds the Evidence-Based Decision Score (Pillar 3),
        Decision-Score-weighted sizing on top of the existing depth multiplier
        (Pillar 5), real Strategy Edge Management fed by the previously-unused
        kill-switch telemetry (Pillar 7), self-diagnostics on both kill
        switches (Pillar 8), and the StrategyProposal/Standalone-Adapter
        interface (Pillar 10), WITHOUT changing the grid's mechanical
        buy-low/sell-high behaviour, its regime gate, or its kill switches.

        THEORY (Pillar 1): the grid manufactures profit from MEAN REVERSION
        inside a bounded range. In a range-bound (non-trending) regime, price
        oscillates around a center; the grid pre-positions resting buy levels
        below and sell levels above and harvests each oscillation as a
        completed buy->sell cycle whose gross profit is one grid spacing. The
        edge exists ONLY while the market is genuinely ranging: a sustained
        trend turns "buy the dip" into "catch a falling knife", so the regime
        gate (Pillar 2) and the two kill switches (drawdown, range-escape) are
        the core risk control, not add-ons. Long-biased for crypto's secular
        drift (7 buy / 3 sell of 10 levels). Failure modes: (a) a trend the
        regime gate misses -> range-escape kill (a spacing/multiplier
        mismatch, Pillar 7 Category B); (b) a sustained bleed -> drawdown kill
        (a genuine loss of edge, Pillar 7 Category C candidate).

        DESIGN PHILOSOPHY:
        - Grid is a MANUFACTURING PROCESS: converts cash to crypto at favorable prices
        - Long-biased for crypto: more buy levels below, fewer sell levels above
        - Capital-bounded: hard kill switches prevent runaway losses
        - Regime-aware: pauses in trends/high volatility, operates in flat/normal markets
        - Depth-aware sizing: larger orders at deeper discounts (convex payoff)
        - Bar-based: one order max per bar (no cascading, no tick noise) - this
          per-bar single-order invariant is exactly WHY one StrategyProposal
          per evaluation remains sufficient: the grid never needs multiple
          concurrent proposals or staged execution intents (see tasks.md 5.6).
        - Fast exits: admits failure quickly via kill switches

        RISK CONTROLS:
        1. Max drawdown kill switch (% of initial capital)
        2. ATR distance kill switch (price escapes grid range — hard stop)
        3. Regime gating (only operates in trend_flat + volatility_normal)
        4. Soft re-centering when price drifts >50% of grid half-range (no cooldown)

        CRITICAL: All logic operates on AGGREGATED PSEUDO-BARS, not tick data.
        Bar interval defines time granularity (default: 60 seconds per bar).

        Parameters:
            bar_interval_seconds: Seconds per bar for aggregation (default: 60)
            grid_count: Total grid levels (default: 10, long-biased split: 7 buy / 3 sell)
            atr_range_multiplier: Total grid span = ATR × this value (default: 8.0).
                Spacing is derived as (ATR × multiplier) / grid_count so the grid
                automatically widens in high volatility and narrows when calm.
            atr_warmup_spacing_pct: Fixed % spacing used before ATR warms up (default: 0.5)
            base_order_size_percent: Base order size % of budget (default: 5)
            depth_multiplier: Multiplier for deeper levels (default: 1.5, convex sizing)
            max_drawdown_percent: Max drawdown % before kill (default: 15)
            kill_atr_multiplier: ATR distance multiplier for hard kill switch (default: 3.0)
            atr_period: ATR period for spacing and kill switch (default: 14 bars)
            regime_filter_enabled: Enable regime gating (default: True)
            allowed_regimes: Allowed regimes (default: ["trend_flat", "volatility_medium"])
            cooldown_after_kill_hours: Hours to wait after hard kill switch (default: 2)
            decision_score_threshold: minimum Evidence-Based Decision Score a
                crossed level must clear to fill (default: 0.0). Default 0.0 is
                deliberate and preserves the grid's mechanical nature: the
                three Evidence Items (range-bound conviction, post-fee spacing
                margin, volatility adequacy) are all non-negative in normal
                range-bound operation, so at 0.0 the score gate NEVER suppresses
                a fill in the conditions the grid is designed for - it only
                declines a crossing whose net evidence has gone negative (a
                trend the gate let slip, or a dead market with no cycles to
                harvest). The score's primary job here is Pillar 5 sizing, not
                gating. An operator can raise this to demand higher-quality
                fills. The evidence is side-independent (identical for a buy or
                sell crossing on the same bar), so the gate never introduces
                buy/sell inventory asymmetry.
        """
        # === PARAMETER EXTRACTION ===
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        grid_count = params.get("grid_count", 10)
        atr_range_multiplier = params.get("atr_range_multiplier", 8.0)
        atr_warmup_spacing_pct = params.get("atr_warmup_spacing_pct", 0.5) / 100
        base_order_size_pct = params.get("base_order_size_percent", 5) / 100
        depth_multiplier = params.get("depth_multiplier", 1.5)
        max_drawdown_pct = params.get("max_drawdown_percent", 15) / 100
        kill_atr_mult = params.get("kill_atr_multiplier", 3.0)
        atr_period = params.get("atr_period", 14)
        regime_filter_enabled = params.get("regime_filter_enabled", True)
        # NOTE: the regime detector emits volatility in {low, medium, high}; it
        # never emits 'normal'. The old default ['trend_flat','volatility_normal']
        # had a DEAD second tag, so the grid effectively only ran when trend was
        # flat. 'volatility_medium' is the detector's word for normal volatility,
        # which is the range-bound condition the grid is designed for.
        allowed_regimes = params.get("allowed_regimes", ["trend_flat", "volatility_medium"])
        cooldown_after_kill_hours = params.get("cooldown_after_kill_hours", 2)
        decision_score_threshold = params.get("decision_score_threshold", 0.0)
        _validity_interval_seconds = max(bar_interval_seconds, 1)

        # Pillar 7: per-engine, per-bot Strategy Edge Manager. Fed the grid's
        # own realised outcomes (each realised SELL fill = a harvested cycle
        # win; each kill switch = a loss episode) plus every fill's Decision
        # Score. Range-escape kills are classified Category B (spacing/multiplier
        # mismatch - parameter, adaptable, non-blocking); a drawdown kill in a
        # suitable regime is the genuine Category C "edge gone" signal. See the
        # audit's Pillar 7 section for the modelling rationale.
        if not hasattr(self, "_grid_edge_manager"):
            self._grid_edge_manager = StrategyEdgeManager()
        edge_manager = self._grid_edge_manager

        # Long-biased split for crypto: more buy levels below, fewer sell above
        buy_levels = int(grid_count * 0.7)  # 70% below center
        sell_levels = grid_count - buy_levels  # 30% above center

        # === STATE INITIALIZATION ===
        state = self._get_grid_state(bot.id, params)

        # Initialize state structure (comprehensive institutional state)
        if "initialized" not in state:
            state.update({
                "initialized": True,
                "center_price": current_price,
                "initial_capital": bot.budget,
                "virtual_cash": bot.budget,  # Virtual wallet for depth-aware sizing
                "virtual_crypto": 0.0,  # Virtual crypto holdings (in base currency units)
                "grid_levels": {},  # {level: {"price": float, "filled": bool, "side": "buy"/"sell"}}
                "last_bar_close_time": None,
                "current_bar": None,  # {"open": float, "high": float, "low": float, "close": float, "start_ts": datetime}
                "completed_bars": [],  # List of completed bars for ATR calculation
                "last_order_bar": None,  # Timestamp of last order bar (one order per bar)
                "peak_portfolio_value": bot.budget,  # Track peak for drawdown
                "last_recenter_time": self.clock.now(),
                "total_trades": 0,
                # TELEMETRY METRICS (for dashboards and auto-mode learning)
                "lifetime_return_pct": 0.0,  # Total return since inception (%)
                "lifetime_max_drawdown_pct": 0.0,  # Worst drawdown ever experienced (%)
                # COOLDOWN TRACKING (prevents immediate re-entry after kill)
                "last_kill_switch_time": None,  # Timestamp of last kill switch activation
                "kill_switch_count": 0,  # Number of times kill switch has fired
                "last_kill_reason": None,  # "range_escape" | "drawdown" - drives Pillar 7 classification
                # ATR LOCKING (refreshed on recenter)
                "atr_at_recenter": None,  # ATR value captured at last recenter
                # DYNAMIC SPACING DIAGNOSTICS
                "current_atr": None,
                "current_grid_range": None,
                "current_grid_spacing": None,
                "atr_spacing": None,  # spacing value used to build current grid levels
            })
            logger.info(
                f"Bot {bot.id}: Adaptive Grid initialized - "
                f"Center: ${current_price:.2f}, Capital: ${bot.budget:.2f}, "
                f"Long-biased: {buy_levels} buy / {sell_levels} sell levels"
            )

        # === BAR AGGREGATION (60-second bars) ===
        now = self.clock.now()
        current_bar = state.get("current_bar")

        if current_bar is None:
            # Start new bar
            state["current_bar"] = {
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "start_ts": now,
            }
            return TradeSignal(action="hold", amount=0, reason="Grid: Starting new bar")

        # Update current bar
        current_bar["high"] = max(current_bar["high"], current_price)
        current_bar["low"] = min(current_bar["low"], current_price)
        current_bar["close"] = current_price

        # Check if bar is complete
        bar_duration = (now - current_bar["start_ts"]).total_seconds()
        if bar_duration < bar_interval_seconds:
            _rem = max(0.0, bar_interval_seconds - bar_duration)
            self._explain(bot.id).state("WAITING_BAR_CLOSE").update({
                "current_price": current_price,
                "grid_center": state.get("center_price"),
                "bar_progress_s": round(bar_duration, 1),
                "bar_interval_s": bar_interval_seconds,
                "bar_time_remaining_s": round(_rem, 1),
            }).next_trade(
                current=round(bar_duration, 1), current_label="Bar elapsed (s)",
                target=bar_interval_seconds, target_label="Bar interval (s)",
                distance=round(_rem, 1), status=f"{_rem:.0f}s until bar closes",
            ).check(
                "Bar complete", f"{bar_duration:.0f}s", f">= {bar_interval_seconds}s", False,
                detail="grid evaluates once per completed bar",
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Grid: Bar in progress ({bar_duration:.0f}/{bar_interval_seconds}s)"
            )

        # === BAR COMPLETED - Process on completed bar ===
        completed_bars = state.get("completed_bars", [])
        completed_bars.append(current_bar)

        # Retain last 100 bars for ATR calculation
        if len(completed_bars) > 100:
            completed_bars = completed_bars[-100:]
        state["completed_bars"] = completed_bars

        # Start new bar for next iteration
        state["current_bar"] = {
            "open": current_price,
            "high": current_price,
            "low": current_price,
            "close": current_price,
            "start_ts": now,
        }

        bar_close_price = completed_bars[-1]["close"]

        logger.info(
            f"Bot {bot.id}: Grid bar completed #{len(completed_bars)} - "
            f"OHLC: ${completed_bars[-1]['open']:.2f} / ${completed_bars[-1]['high']:.2f} / "
            f"${completed_bars[-1]['low']:.2f} / ${completed_bars[-1]['close']:.2f}"
        )

        # === MARKET SUITABILITY (Pillar 2, shared MarketSuitabilityGate) ===
        # Computed every completed bar (the price-only _detect_market_regime is
        # a pure function, no side effects), so every proposal below carries a
        # real MarketSuitabilityResult and the hard regime gate still returns at
        # its original position — the order of gates is unchanged. Uses the same
        # dict-shaped bar-close price history the pre-migration grid built, so
        # the regime verdict is bit-for-bit identical. Only LEVEL/trend tags are
        # referenced by allowed_regimes, so volatility_direction is not needed.
        price_history_from_bars = [
            {"price": b["close"], "timestamp": b["start_ts"]} for b in completed_bars
        ]
        price_history_with_current = price_history_from_bars + [
            {"price": current_price, "timestamp": now}
        ]
        current_regime = self._detect_market_regime(price_history_with_current, None)
        trend_state = current_regime.get("trend_state", "flat")
        volatility_state = current_regime.get("volatility_state", "medium")
        regime_names = [f"trend_{trend_state}", f"volatility_{volatility_state}"]
        # regime_filter_enabled (kept as a test-isolation affordance) maps to
        # allowed=["all"] when disabled, so `suitability` is always a real
        # result for the proposal while Pillar 2 stays enforced by default. The
        # gate's OR-across-tags semantics reproduce the pre-migration
        # `any(r in allowed_regimes ...)` check exactly for the default tags.
        suitability = MarketSuitabilityGate().evaluate(
            current_regime, allowed_regimes if regime_filter_enabled else ["all"],
        )
        regime_allowed = suitability.is_suitable

        # === STRATEGY EDGE MANAGEMENT (Pillar 7) — baseline for this bar ===
        # A range-escape kill is a spacing/multiplier (parameter) mismatch ->
        # Category B (adaptable, non-blocking). A drawdown kill cites no
        # parameter, so in a suitable regime it surfaces as Category C (edge
        # gone -> stop). The kill-switch branches below record the fresh outcome
        # and re-evaluate; this baseline is what the pre-fill proposals and the
        # fill gate consult.
        _recent_kill_range_escape = state.get("last_kill_reason") == "range_escape"
        _baseline_param_evidence = (
            f"range-escape kills dominate recent history: the grid span "
            f"(atr_range_multiplier={atr_range_multiplier}x ATR) keeps failing to "
            f"contain realised volatility — a spacing/multiplier mismatch"
            if _recent_kill_range_escape else None
        )
        edge_status = edge_manager.evaluate(
            bot.id, "adaptive_grid",
            regime_outside_suitable_range=not suitability.is_suitable,
            parameter_mismatch_evidence=_baseline_param_evidence,
            now=now,
        )

        # === Proposal helpers (Pillar 10) ===
        def _single_evidence_proposal(
            *, direction: "Direction", execution_intent: "ExecutionIntent",
            evidence_name: str, evidence_value: float, evidence_reason: str,
            threshold: float = 1.0, suggested_position_size: Optional[float] = None,
            assumptions: tuple = (), edge: Optional["EdgeStatus"] = None,
        ) -> StrategyProposal:
            """Single-evidence proposal for every already-decided branch
            (cooldown/regime/kill/insufficient-funds no-trades and the
            no-level-crossed hold)."""
            item = EvidenceItem(
                name=evidence_name, measurement=lambda d: evidence_value,
                normalization=lambda r: r, weight=100.0, reason=evidence_reason,
            )
            score = DecisionScoreEngine().score("adaptive_grid", [item], {}, threshold=threshold)
            reasons_for, reasons_against = derive_reasons(score)
            return StrategyProposal(
                strategy_id="adaptive_grid", bot_id=bot.id, generated_at=now,
                direction=direction, execution_intent=execution_intent,
                validity=ProposalValidity(
                    generated_at=now,
                    valid_until=now + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=score, market_suitability=suitability,
                edge_status=(edge if edge is not None else edge_status),
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=suggested_position_size,
                suggested_risk_budget_pct=base_order_size_pct,
                explanation=self._explain(bot.id).to_dict(),
            )

        def _emit_hold(proposal: StrategyProposal, reason: str) -> TradeSignal:
            sig = StandaloneAdapter.to_trade_signal(proposal)
            return sig if sig is not None else TradeSignal(action="hold", amount=0, reason=reason)

        def _no_trade(
            evidence_name: str, evidence_value: float, evidence_reason: str, reason: str,
            *, assumptions: tuple = (), edge: Optional["EdgeStatus"] = None,
        ) -> TradeSignal:
            return _emit_hold(
                _single_evidence_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name=evidence_name, evidence_value=evidence_value,
                    evidence_reason=evidence_reason, assumptions=assumptions, edge=edge,
                ),
                reason,
            )

        def _hold_position(
            evidence_name: str, evidence_reason: str, reason: str, *, assumptions: tuple = (),
        ) -> TradeSignal:
            # "no grid level crossed this bar": the grid is live and holding its
            # virtual inventory, so HOLD + HOLD_POSITION is the correct pairing
            # (not NO_TRADE/NO_ACTION) — see tasks.md 5.6.
            return _emit_hold(
                _single_evidence_proposal(
                    direction=Direction.HOLD, execution_intent=ExecutionIntent.HOLD_POSITION,
                    evidence_name=evidence_name, evidence_value=1.0,
                    evidence_reason=evidence_reason, assumptions=assumptions,
                ),
                reason,
            )

        # === COOLDOWN CHECK (after kill switch) ===
        last_kill_time = state.get("last_kill_switch_time")
        if last_kill_time is not None:
            cooldown_seconds = cooldown_after_kill_hours * 3600
            time_since_kill = (now - last_kill_time).total_seconds()

            if time_since_kill < cooldown_seconds:
                remaining_minutes = (cooldown_seconds - time_since_kill) / 60
                logger.info(
                    f"Bot {bot.id}: Grid in COOLDOWN after kill switch - "
                    f"{remaining_minutes:.1f} minutes remaining (prevents re-entry into bad conditions)"
                )
                self._explain(bot.id).state("COOLDOWN").update({
                    "current_price": current_price,
                    "cooldown_remaining_min": round(remaining_minutes, 1),
                }).next_trade(
                    current=round(remaining_minutes, 1), current_label="Cooldown remaining (min)",
                    target=0, target_label="Ready", distance=round(remaining_minutes, 1),
                    status=f"{remaining_minutes:.0f} min until grid resumes",
                ).check(
                    "Kill-switch cooldown", f"{remaining_minutes:.1f} min remaining",
                    "elapsed", False, detail="paused after a kill switch",
                )
                return _no_trade(
                    "Kill-switch cooldown", 0.0,
                    f"{remaining_minutes:.0f} min of post-kill cooldown remain before the grid may resume",
                    f"Grid: Cooldown after kill ({remaining_minutes:.0f}min remaining)",
                )
            else:
                # Cooldown expired, clear the timestamp
                logger.info(
                    f"Bot {bot.id}: Grid cooldown expired - "
                    f"Resuming normal operation after {cooldown_after_kill_hours}h pause"
                )
                state["last_kill_switch_time"] = None

        # === REGIME GATING (operate only in flat/normal markets) ===
        if regime_filter_enabled and not regime_allowed:
            logger.info(
                f"Bot {bot.id}: Grid PAUSED (regime unsuitable) - "
                f"Current: {regime_names}, Allowed: {allowed_regimes}. "
                f"Grid operates in flat/normal markets only (range-bound conditions)."
            )
            self._explain(bot.id).state("WAITING_REGIME").update({
                "current_price": current_price,
                "regime": ", ".join(regime_names),
            }).check(
                "Regime suitable", ", ".join(regime_names),
                f"in [{', '.join(allowed_regimes)}]", False,
            )
            return _no_trade(
                "Market suitability", 0.0, suitability.reason,
                f"Grid: Paused (regime={regime_names}, need flat/normal markets)",
            )

        # === CALCULATE ATR (drives both spacing and kill switch) ===
        atr = None
        if len(completed_bars) >= atr_period:
            atr = self.calculate_atr_proxy(completed_bars, atr_period)

        # === DYNAMIC GRID SPACING (ATR-based, recomputed every bar) ===
        center_price = state["center_price"]
        if atr is not None and atr > 0:
            grid_range = atr * atr_range_multiplier      # total span ($)
            grid_spacing = grid_range / grid_count        # $ between adjacent levels
        else:
            # Warmup: not enough bars for ATR yet — use fixed % fallback
            grid_spacing = center_price * atr_warmup_spacing_pct
            grid_range = grid_spacing * grid_count

        # Minimum economically viable spacing: grid profit per cycle must exceed
        # round-trip transaction costs (two taker fills + bid-ask spread) plus a
        # configurable profit buffer.  This prevents the grid from targeting
        # profit smaller than fees, which would make every trade a net loser.
        taker_fee_pct = params.get("taker_fee_pct", 0.1) / 100.0
        spread_pct = params.get("spread_pct", 0.05) / 100.0
        min_profit_buffer_pct = params.get("min_profit_buffer_pct", 0.1) / 100.0
        min_spacing_usd = center_price * (2.0 * taker_fee_pct + spread_pct + min_profit_buffer_pct)

        if grid_spacing < min_spacing_usd:
            grid_spacing = min_spacing_usd
            grid_range = grid_spacing * grid_count

        grid_half_range = grid_range / 2.0  # each side extends this far from center

        # Update diagnostic state every bar
        state["current_atr"] = atr
        state["current_grid_range"] = grid_range
        state["current_grid_spacing"] = grid_spacing

        # === SOFT RECENTER (no cooldown — gentle drift correction) ===
        # When price moves more than 50% of the half-range from center, shift
        # the grid to the new price. This fires before the hard kill switch so
        # normal trending sessions don't trigger unnecessary cooldowns.
        distance_from_center = abs(bar_close_price - center_price)
        if distance_from_center > grid_half_range * 0.5:
            old_center = center_price
            state["center_price"] = bar_close_price
            center_price = bar_close_price
            state["grid_levels"] = {}  # cleared so grid is rebuilt below
            state["last_recenter_time"] = now
            logger.info(
                f"Bot {bot.id}: Grid SOFT RECENTER - price ${bar_close_price:.2f} "
                f"drifted {distance_from_center:.2f} from center ${old_center:.2f} "
                f"(threshold {grid_half_range * 0.5:.2f} = 50% half-range). "
                f"ATR={f'{atr:.2f}' if atr else 'n/a'}, spacing=${grid_spacing:.2f}"
            )

        # === TELEMETRY METRICS (track portfolio performance) ===
        # Calculate current portfolio value (virtual wallet concept)
        current_portfolio_value = state["virtual_cash"] + (state["virtual_crypto"] * bar_close_price)
        initial_capital = state["initial_capital"]

        # Update lifetime return
        lifetime_return_pct = ((current_portfolio_value - initial_capital) / initial_capital) * 100
        state["lifetime_return_pct"] = lifetime_return_pct

        # Update peak portfolio value
        if current_portfolio_value > state["peak_portfolio_value"]:
            state["peak_portfolio_value"] = current_portfolio_value

        # Calculate drawdown from peak
        drawdown = (state["peak_portfolio_value"] - current_portfolio_value) / state["peak_portfolio_value"]
        drawdown_pct = drawdown * 100

        # Update lifetime max drawdown (worst ever experienced)
        if drawdown_pct > state["lifetime_max_drawdown_pct"]:
            state["lifetime_max_drawdown_pct"] = drawdown_pct

        # === KILL SWITCH 1: MAX DRAWDOWN ===

        if drawdown > max_drawdown_pct:
            # Activate cooldown and track kill switch event
            state["last_kill_switch_time"] = now
            state["kill_switch_count"] = state.get("kill_switch_count", 0) + 1
            state["last_kill_reason"] = "drawdown"

            logger.warning(
                f"Bot {bot.id}: Grid KILL SWITCH #{state['kill_switch_count']} ACTIVATED (drawdown) - "
                f"Drawdown: {drawdown*100:.1f}% > Max: {max_drawdown_pct*100:.1f}%. "
                f"Grid admits failure. Liquidating all virtual positions. "
                f"Cooldown: {cooldown_after_kill_hours}h"
            )

            # Pillar 7: a drawdown kill is a realised loss episode with NO
            # parameter citation (the grid was correctly spaced but still bled),
            # so in a suitable regime it is the genuine Category C "edge gone"
            # signal. Record it, then re-evaluate so the proposal carries the
            # escalated status.
            edge_manager.record_trade_outcome(
                bot.id, "adaptive_grid", pnl=-float(drawdown), win=False, at=now,
            )
            dd_edge = edge_manager.evaluate(
                bot.id, "adaptive_grid",
                regime_outside_suitable_range=not suitability.is_suitable,
                parameter_mismatch_evidence=None,
                now=now,
            )

            # Pillar 8: the drawdown kill switch was previously invisible to the
            # diagnostics UI (log-only). Surface the full decision math.
            self._explain(bot.id).state("KILL_SWITCH_DRAWDOWN").update({
                "current_price": current_price,
                "grid_center": center_price,
                "drawdown_pct": drawdown_pct,
                "max_drawdown_pct": max_drawdown_pct * 100,
                "peak_portfolio_value": state["peak_portfolio_value"],
                "portfolio_value": current_portfolio_value,
                "kill_switch_count": state["kill_switch_count"],
                "edge_status_category": dd_edge.category.value,
            }).check(
                "Drawdown kill switch", f"{drawdown_pct:.2f}%",
                f"<= {max_drawdown_pct * 100:.1f}%", False,
                detail="grid liquidates virtual inventory and enters cooldown",
            )

            # Reset grid (liquidate virtual positions, restart)
            state["virtual_cash"] = current_portfolio_value
            state["virtual_crypto"] = 0.0
            state["center_price"] = bar_close_price
            state["grid_levels"] = {}
            state["peak_portfolio_value"] = current_portfolio_value
            state["last_recenter_time"] = now
            self._save_grid_state(bot.id, state)

            return _no_trade(
                "Drawdown kill switch", 0.0,
                (
                    f"portfolio drawdown {drawdown * 100:.1f}% exceeded the "
                    f"{max_drawdown_pct * 100:.1f}% max: grid liquidated virtual inventory, "
                    f"{cooldown_after_kill_hours}h cooldown"
                ),
                f"Grid: Kill switch (drawdown {drawdown*100:.1f}%), {cooldown_after_kill_hours}h cooldown",
                edge=dd_edge,
            )

        # === KILL SWITCH 2: ATR DISTANCE (hard stop — soft recenter fires first) ===
        # After a soft recenter, distance_from_center resets to 0, so this kill
        # only fires when price moves faster than the soft recenter can react.
        if atr is not None:
            atr_distance = abs(bar_close_price - center_price)
            kill_distance = atr * kill_atr_mult

            if atr_distance > kill_distance:
                # Activate cooldown and track kill switch event
                state["last_kill_switch_time"] = now
                state["kill_switch_count"] = state.get("kill_switch_count", 0) + 1
                state["last_kill_reason"] = "range_escape"

                # Lock ATR at recenter (ensures fresh baseline for new grid)
                state["atr_at_recenter"] = atr

                logger.warning(
                    f"Bot {bot.id}: Grid KILL SWITCH #{state['kill_switch_count']} ACTIVATED (range escape) - "
                    f"Price ${bar_close_price:.2f} is {atr_distance:.2f} from center ${center_price:.2f} "
                    f"(> {kill_atr_mult}x ATR = {kill_distance:.2f}). Re-centering grid. "
                    f"ATR locked at {atr:.2f}. Cooldown: {cooldown_after_kill_hours}h"
                )

                # Pillar 7: a range-escape kill is a spacing/multiplier
                # (parameter) mismatch — the grid span didn't contain realised
                # volatility. Record the loss episode and classify with that
                # parameter citation so it lands as Category B (adaptable,
                # non-blocking), NOT Category C.
                param_evidence = (
                    f"price escaped {atr_distance:.2f} from center (> {kill_atr_mult}x ATR "
                    f"= {kill_distance:.2f}): the grid span (atr_range_multiplier="
                    f"{atr_range_multiplier}x ATR) did not contain realised volatility"
                )
                edge_manager.record_trade_outcome(
                    bot.id, "adaptive_grid",
                    pnl=-(atr_distance / center_price if center_price > 0 else 0.0),
                    win=False, at=now,
                )
                atr_edge = edge_manager.evaluate(
                    bot.id, "adaptive_grid",
                    regime_outside_suitable_range=not suitability.is_suitable,
                    parameter_mismatch_evidence=param_evidence,
                    now=now,
                )

                # Pillar 8: the range-escape kill switch was previously invisible
                # to the diagnostics UI (log-only). Surface the full math.
                self._explain(bot.id).state("KILL_SWITCH_RANGE_ESCAPE").update({
                    "current_price": current_price,
                    "grid_center": center_price,
                    "atr": atr,
                    "atr_distance": atr_distance,
                    "kill_distance": kill_distance,
                    "kill_atr_multiplier": kill_atr_mult,
                    "kill_switch_count": state["kill_switch_count"],
                    "edge_status_category": atr_edge.category.value,
                }).check(
                    "Range-escape kill switch", f"{atr_distance:.2f} from center",
                    f"<= {kill_distance:.2f} ({kill_atr_mult}x ATR)", False,
                    detail="grid re-centers on the escaped price and enters cooldown",
                )

                # Re-center grid
                state["center_price"] = bar_close_price
                state["grid_levels"] = {}
                state["last_recenter_time"] = now
                self._save_grid_state(bot.id, state)

                return _no_trade(
                    "Range-escape kill switch", 0.0, param_evidence,
                    (
                        f"Grid: Re-centered at ${bar_close_price:.2f} (range escape), "
                        f"{cooldown_after_kill_hours}h cooldown"
                    ),
                    assumptions=(
                        f"price stays within the grid range (|price - center| <= "
                        f"{kill_atr_mult}x ATR)",
                    ),
                    edge=atr_edge,
                )

        # === ONE ORDER PER BAR CHECK ===
        last_order_bar = state.get("last_order_bar")
        if last_order_bar is not None and last_order_bar == completed_bars[-1]["start_ts"]:
            return _hold_position(
                "One order per bar",
                "already filled a level this bar — the grid caps itself at one order per bar",
                "Grid: Already traded this bar (one order per bar limit)",
            )

        # === CALCULATE GRID LEVELS (ATR-based; rebuild when spacing changes >10%) ===
        stored_spacing = state.get("atr_spacing")
        spacing_changed = (
            stored_spacing is not None
            and stored_spacing > 0
            and abs(grid_spacing - stored_spacing) / stored_spacing > 0.10
        )

        if not state["grid_levels"] or spacing_changed:
            grid_levels = {}

            # Create buy levels (below center, long-biased)
            for i in range(1, buy_levels + 1):
                level_price = center_price - grid_spacing * i
                grid_levels[-i] = {
                    "price": level_price,
                    "side": "buy",
                    "filled": False,
                    "depth": i,
                }

            # Create sell levels (above center)
            for i in range(1, sell_levels + 1):
                level_price = center_price + grid_spacing * i
                grid_levels[i] = {
                    "price": level_price,
                    "side": "sell",
                    "filled": False,
                    "depth": i,
                }

            state["grid_levels"] = grid_levels
            state["atr_spacing"] = grid_spacing

            if spacing_changed:
                logger.info(
                    f"Bot {bot.id}: Grid RECOMPUTED — ATR spacing changed "
                    f"${stored_spacing:.2f} → ${grid_spacing:.2f} "
                    f"({(grid_spacing - stored_spacing) / stored_spacing * 100:+.1f}%). "
                    f"New: {buy_levels} buy / {sell_levels} sell levels at "
                    f"center ${center_price:.2f}, ATR ${f'{atr:.2f}' if atr else 'warmup'}"
                )
            else:
                logger.info(
                    f"Bot {bot.id}: Grid levels created (ATR-based) — "
                    f"center=${center_price:.2f}, ATR=${f'{atr:.2f}' if atr else 'warmup'}, "
                    f"range=${grid_range:.2f}, spacing=${grid_spacing:.2f}, "
                    f"{buy_levels} buy / {sell_levels} sell levels"
                )

        # === FIND NEAREST UNFILLED LEVEL ===
        grid_levels = state["grid_levels"]
        nearest_level = None
        nearest_distance = float("inf")

        for level_num, level_data in grid_levels.items():
            if level_data["filled"]:
                continue

            distance = abs(bar_close_price - level_data["price"])

            # Check if price crossed this level
            if level_data["side"] == "buy" and bar_close_price <= level_data["price"]:
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = level_num
            elif level_data["side"] == "sell" and bar_close_price >= level_data["price"]:
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = level_num

        # --- Structured decision explanation (observe-only): the exact spacing
        # and nearest-level math that determines whether an order is created ---
        exp = self._explain(bot.id)
        if atr is not None and atr > 0:
            calc_spacing = atr * atr_range_multiplier / grid_count
        else:
            calc_spacing = center_price * atr_warmup_spacing_pct
        _unfilled_buys = [lv["price"] for lv in grid_levels.values()
                          if lv["side"] == "buy" and not lv["filled"]]
        _unfilled_sells = [lv["price"] for lv in grid_levels.values()
                           if lv["side"] == "sell" and not lv["filled"]]
        nearest_buy_price = (
            min(_unfilled_buys, key=lambda p: abs(p - bar_close_price)) if _unfilled_buys else None)
        nearest_sell_price = (
            min(_unfilled_sells, key=lambda p: abs(p - bar_close_price)) if _unfilled_sells else None)
        dist_buy = (bar_close_price - nearest_buy_price) if nearest_buy_price is not None else None
        dist_sell = (nearest_sell_price - bar_close_price) if nearest_sell_price is not None else None
        exp.update({
            "current_price": bar_close_price,
            "grid_center": center_price,
            "atr": atr if atr is not None else "warmup",
            "atr_range_multiplier": atr_range_multiplier,
            "calculated_spacing": calc_spacing,
            "min_profitable_spacing": min_spacing_usd,
            "effective_spacing": grid_spacing,
            "grid_range": grid_range,
            "buy_levels": buy_levels,
            "sell_levels": sell_levels,
            "nearest_buy": nearest_buy_price,
            "nearest_sell": nearest_sell_price,
            "distance_to_buy": dist_buy,
            "distance_to_sell": dist_sell,
            "virtual_cash": state["virtual_cash"],
            "virtual_crypto": state["virtual_crypto"],
            "total_trades": state.get("total_trades", 0),
            "drawdown_pct": drawdown_pct,
        })
        if nearest_buy_price is not None:
            exp.check(
                "Nearest BUY", f"{dist_buy:+.2f} from level",
                f"price <= {nearest_buy_price:.2f}", bar_close_price <= nearest_buy_price,
                detail=f"trigger spacing {grid_spacing:.2f}",
            )
        if nearest_sell_price is not None:
            exp.check(
                "Nearest SELL", f"{dist_sell:+.2f} from level",
                f"price >= {nearest_sell_price:.2f}", bar_close_price >= nearest_sell_price,
                detail=f"trigger spacing {grid_spacing:.2f}",
            )
        exp.check(
            "Grid level triggered",
            "yes" if nearest_level is not None else "no",
            "price crossed an unfilled level", nearest_level is not None,
        )
        # Current state + next-trade preview (from values already computed above).
        if not _unfilled_buys and not _unfilled_sells:
            exp.state("GRID_COMPLETE")
        else:
            _cand = []
            if nearest_buy_price is not None:
                _cand.append(("WAITING_BUY_LEVEL", "Nearest BUY", nearest_buy_price, dist_buy, "BUY"))
            if nearest_sell_price is not None:
                _cand.append(("WAITING_SELL_LEVEL", "Nearest SELL", nearest_sell_price, dist_sell, "SELL"))
            # Wait on whichever unfilled level the price is closest to crossing.
            _cand.sort(key=lambda c: abs(c[3]))
            _st, _lbl, _tgt, _d, _side = _cand[0]
            exp.state(_st).next_trade(
                current=bar_close_price, current_label="Current price",
                target=_tgt, target_label=_lbl, distance=_d, trigger=grid_spacing,
                status=(f"{abs(_d):.2f} away from {_side} level"
                        if nearest_level is None else f"{_side} level reached — order armed"),
            )
        # ---------------------------------------------------------------------

        if nearest_level is None:
            # PRODUCTION DIAGNOSIS: how far price is from the closest unfilled
            # buy/sell level, so an operator can see whether the grid is simply
            # waiting for a move or is mis-centered. Runs once per completed bar.
            nearest_buy = min(
                (lv["price"] for lv in grid_levels.values()
                 if lv["side"] == "buy" and not lv["filled"]), default=None)
            nearest_sell = min(
                (lv["price"] for lv in grid_levels.values()
                 if lv["side"] == "sell" and not lv["filled"]),
                key=lambda pr: abs(pr - bar_close_price), default=None)
            buy_gap = (
                f"{(bar_close_price - nearest_buy)/bar_close_price*100:+.2f}%"
                if nearest_buy else "n/a")
            atr_str = f"${atr:.2f}" if atr else "warmup"
            logger.info(
                f"Bot {bot.id}: Grid no-trade diag - close=${bar_close_price:.2f} "
                f"center=${center_price:.2f} spacing=${grid_spacing:.2f} "
                f"ATR={atr_str} range=${grid_range:.2f} "
                f"nearest_buy=${nearest_buy or 0:.2f} (price {buy_gap} vs nearest buy) "
                f"virtual_cash=${state['virtual_cash']:.2f} "
                f"virtual_crypto={state['virtual_crypto']:.6f}"
            )
            nearest_buy_str = f"${nearest_buy:.2f}" if nearest_buy else "none"
            return _hold_position(
                "No level crossed",
                (
                    f"bar close ${bar_close_price:.2f} did not cross any unfilled grid level "
                    f"(spacing ${grid_spacing:.2f}, ATR={atr_str}, nearest buy {nearest_buy_str}) — "
                    "grid holds its virtual inventory"
                ),
                (
                    f"Grid: No levels triggered "
                    f"(spacing=${grid_spacing:.2f}, ATR={atr_str}, "
                    f"nearest_buy={nearest_buy_str})"
                ),
                assumptions=(
                    f"regime remains range-bound ({', '.join(regime_names)})",
                    f"price stays within the grid range (|price - center| <= {grid_half_range:.2f})",
                ),
            )

        # === EXECUTE ORDER AT NEAREST LEVEL ===
        level_data = grid_levels[nearest_level]
        depth = level_data["depth"]

        # === PILLAR 3: EVIDENCE-BASED DECISION SCORE (per crossed level) ===
        # The level crossing is the grid's hard precondition (analogous to mean
        # reversion's band touch). These three side-independent Evidence Items
        # then GRADE the crossing's quality — feeding Pillar 5 sizing and, when
        # decision_score_threshold > 0, an optional quality gate. All three are
        # non-negative in the range-bound conditions the grid is designed for,
        # so the default 0.0 threshold never suppresses a normal fill; they turn
        # negative (declining the fill, leaving virtual state untouched) only
        # when the grid's own edge premise has broken. Crucially, NONE of them
        # references level depth — depth is the grid's intended convex payoff,
        # rewarded separately by the depth multiplier, not penalised here.
        _fee_threshold_move = 2.0 * taker_fee_pct + spread_pct + min_profit_buffer_pct
        _spacing_move = grid_spacing / bar_close_price if bar_close_price > 0 else 0.0
        _spacing_margin = (
            (_spacing_move - _fee_threshold_move) / _fee_threshold_move
            if _fee_threshold_move > 0 else 0.0
        )
        # Trend flatness: EMA(20) slope over recent bar closes, as a fraction.
        _recent_closes = [b["close"] for b in completed_bars]
        _ema_slope_pct = self._ema_slope_pct(_recent_closes, 20)
        _flat_threshold = 0.005  # 0.5% EMA slope over the window = "no longer flat"
        _vol_ratio = (atr / bar_close_price) if (atr and bar_close_price > 0) else 0.0
        _vol_floor, _vol_target = 0.0005, 0.004  # ATR/price sweet spot for cycling

        evidence_items = [
            EvidenceItem(
                name="Range-bound conviction",
                measurement=lambda d: _ema_slope_pct,
                normalization=lambda r: max(
                    -1.0, min(1.0, (_flat_threshold - abs(r)) / _flat_threshold)
                ),
                weight=40.0,
                reason=(
                    "Theory (mean reversion in a range): the grid only has an edge while the "
                    "market is genuinely ranging. A flat EMA(20) is direct evidence the range "
                    "holds; a steepening slope is evidence AGAINST — a trend turns buy-the-dip "
                    "into catch-the-knife. This is independent of which level filled."
                ),
            ),
            EvidenceItem(
                name="Post-fee spacing margin",
                measurement=lambda d: _spacing_margin,
                normalization=lambda r: max(-1.0, min(1.0, r)),
                weight=35.0,
                reason=(
                    "Theory: a completed buy->sell cycle's gross profit is one grid spacing; the "
                    "net edge is that spacing minus round-trip fees. The more the spacing exceeds "
                    "the fee floor, the more each harvested cycle actually earns. Barely covering "
                    "fees is evidence the edge is marginal."
                ),
            ),
            EvidenceItem(
                name="Volatility adequacy",
                measurement=lambda d: _vol_ratio,
                normalization=lambda r: max(
                    -1.0, min(1.0, (r - _vol_floor) / (_vol_target - _vol_floor))
                ),
                weight=25.0,
                reason=(
                    "Theory: the grid harvests oscillation. Too little volatility (ATR/price near "
                    "zero) means price never traverses levels and no cycles complete — evidence "
                    "against acting; adequate volatility means the range is alive and cycling."
                ),
            ),
        ]
        decision_score = DecisionScoreEngine().score(
            "adaptive_grid", evidence_items, {}, threshold=decision_score_threshold,
        )
        edge_manager.record_decision_score(bot.id, "adaptive_grid", decision_score.total)
        reasons_for, reasons_against = derive_reasons(decision_score)
        _fill_assumptions = (
            f"regime remains range-bound ({', '.join(regime_names)})",
            f"price stays within the grid range (|price - center| <= {grid_half_range:.2f})",
            "grid spacing continues to cover round-trip fees",
        )
        exp.metric("decision_score_total", decision_score.total)
        exp.metric("decision_score_threshold", decision_score_threshold)
        exp.metric("edge_status_category", edge_status.category.value)
        exp.check(
            "Decision Score clears threshold", round(decision_score.total, 2),
            f">= {decision_score_threshold:.1f}", decision_score.approved,
        )
        exp.check(
            "Strategy edge not disqualified", edge_status.category.value,
            f"!= {EdgeCategory.C.value}", edge_status.category != EdgeCategory.C,
            detail=edge_status.reason,
        )

        # Framework gates (both no-ops in the grid's normal range-bound regime):
        # the Decision Score threshold (default 0.0) and a Category-C edge stop.
        # A blocked crossing leaves the level UNFILLED and the virtual wallet
        # untouched — no buy/sell inventory desync — and the grid retries next bar.
        blocking_reasons = []
        if not decision_score.approved:
            blocking_reasons.append(
                f"Decision Score {decision_score.total:.1f} < threshold {decision_score_threshold:.1f}"
            )
        if edge_status.category == EdgeCategory.C:
            blocking_reasons.append(f"Strategy Edge Management: Category C — {edge_status.reason}")
        if blocking_reasons:
            reason_text = "; ".join(blocking_reasons)
            self._save_grid_state(bot.id, state)
            return _no_trade(
                "Framework gates", 0.0, reason_text,
                f"Grid: level {nearest_level} crossed but {reason_text}",
                assumptions=_fill_assumptions,
            )

        # === PILLAR 5: DECISION-SCORE-WEIGHTED SIZING (preserves depth mult) ===
        # order = initial_capital * base% * depth_multiplier * score_multiplier.
        # The depth multiplier (convex payoff) is unchanged from the pre-migration
        # formula; the score multiplier (0.5..1.5) is the new, orthogonal factor.
        size_multiplier = depth_multiplier ** (depth - 1)
        _score_range = max(100.0 - decision_score_threshold, 1e-9)
        _score_margin = max(0.0, min(1.0, (decision_score.total - decision_score_threshold) / _score_range))
        score_size_multiplier = 0.5 + _score_margin
        exp.metric("decision_score_size_multiplier", round(score_size_multiplier, 4))
        order_size_usd = min(
            initial_capital * base_order_size_pct * size_multiplier * score_size_multiplier,
            bot.current_balance * _BUY_BALANCE_FRACTION,  # never exceed real funds
        )

        if order_size_usd < MIN_ORDER_USD:
            self._save_grid_state(bot.id, state)
            return _no_trade(
                "Order size", 0.0,
                (
                    f"order ${order_size_usd:.2f} below the ${MIN_ORDER_USD:.0f} minimum "
                    "(increase budget or base_order_size_percent)"
                ),
                (
                    f"Grid: Order ${order_size_usd:.2f} below minimum ${MIN_ORDER_USD} "
                    f"(increase budget or base_order_size_percent)"
                ),
                assumptions=_fill_assumptions,
            )

        if level_data["side"] == "buy":
            # === BUY ORDER (accumulate crypto) ===
            # Check virtual wallet has cash
            if state["virtual_cash"] < order_size_usd:
                logger.info(
                    f"Bot {bot.id}: Grid BUY skipped at level {nearest_level} - "
                    f"Insufficient virtual cash: ${state['virtual_cash']:.2f} < ${order_size_usd:.2f}"
                )
                self._save_grid_state(bot.id, state)
                return _no_trade(
                    "Virtual cash sufficiency", 0.0,
                    (
                        f"virtual cash ${state['virtual_cash']:.2f} < order ${order_size_usd:.2f} "
                        f"at buy level {nearest_level}"
                    ),
                    f"Grid: Insufficient virtual cash for buy at level {nearest_level}",
                    assumptions=_fill_assumptions,
                )

            # Viability pre-check: verify grid_spacing covers round-trip fees before
            # mutating the virtual wallet. spacing / price = minimum profitable move
            # from the buy level to the nearest sell level. This must exceed the
            # configured fee threshold so the central gate does not later reject an
            # order whose virtual state was already updated.
            _grid_expected_move = grid_spacing / bar_close_price if bar_close_price > 0 else 0.0
            _fee_raw_grid = getattr(bot, 'exchange_fee', 0.1)
            _grid_fee_pct = (
                float(_fee_raw_grid) if isinstance(_fee_raw_grid, (int, float)) else 0.1
            ) / 100.0
            _grid_min_move = 2.0 * _grid_fee_pct + _VIABILITY_SAFETY_MARGIN_PCT
            exp.check(
                "Fee viability", round(_grid_expected_move, 6),
                f">= {_grid_min_move:.5f}", _grid_expected_move >= _grid_min_move,
                detail="grid spacing must cover round-trip fees",
            )
            if _grid_expected_move < _grid_min_move:
                self._save_grid_state(bot.id, state)
                return _no_trade(
                    "Fee viability", 0.0,
                    (
                        f"grid spacing {_grid_expected_move * 100:.3f}% < fee threshold "
                        f"{_grid_min_move * 100:.3f}% (exchange_fee={_grid_fee_pct * 100:.2f}%)"
                    ),
                    (
                        f"Grid: spacing {_grid_expected_move * 100:.3f}% < "
                        f"fee threshold {_grid_min_move * 100:.3f}% "
                        f"(exchange_fee={_grid_fee_pct * 100:.2f}%)"
                    ),
                    assumptions=_fill_assumptions,
                )

            # Execute virtual buy
            crypto_amount = order_size_usd / bar_close_price
            state["virtual_cash"] -= order_size_usd
            state["virtual_crypto"] += crypto_amount
            state["grid_levels"][nearest_level]["filled"] = True
            state["last_order_bar"] = completed_bars[-1]["start_ts"]
            state["total_trades"] += 1

            self._save_grid_state(bot.id, state)

            logger.info(
                f"Bot {bot.id}: Grid BUY at level {nearest_level} (depth={depth}) - "
                f"Price: ${bar_close_price:.2f}, Amount: ${order_size_usd:.2f} "
                f"(size_mult={size_multiplier:.2f}x, score_mult={score_size_multiplier:.2f}x), "
                f"Virtual: ${state['virtual_cash']:.2f} cash + "
                f"{state['virtual_crypto']:.4f} crypto | "
                f"Decision Score {decision_score.total:.1f}/{decision_score_threshold:.1f} | "
                f"Lifetime: {state['lifetime_return_pct']:+.2f}% return, "
                f"{state['lifetime_max_drawdown_pct']:.2f}% max DD"
            )

            # Pillar 10: grid entries are INCREMENTAL virtual-inventory adds, not
            # fresh single positions (contrast Phases 1-4's OPEN_POSITION), so a
            # buy fill maps to BUY + ADD_TO_POSITION (tasks.md 5.6).
            gen = now
            proposal = StrategyProposal(
                strategy_id="adaptive_grid", bot_id=bot.id, generated_at=gen,
                direction=Direction.BUY, execution_intent=ExecutionIntent.ADD_TO_POSITION,
                validity=ProposalValidity(
                    generated_at=gen,
                    valid_until=gen + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=decision_score, market_suitability=suitability,
                edge_status=edge_status, assumptions=_fill_assumptions,
                reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=order_size_usd,
                suggested_risk_budget_pct=base_order_size_pct * size_multiplier * score_size_multiplier,
                expected_holding_horizon="short",
                adaptive_parameters_used={
                    "depth_size_multiplier": size_multiplier,
                    "decision_score_size_multiplier": score_size_multiplier,
                    "grid_spacing": grid_spacing,
                },
                explanation=exp.to_dict(),
            )
            return StandaloneAdapter.to_trade_signal(
                proposal, expected_move_pct=_grid_expected_move,
            )

        else:
            # === SELL ORDER (realize gains) ===
            # Check virtual wallet has crypto
            crypto_to_sell = order_size_usd / bar_close_price
            if state["virtual_crypto"] < crypto_to_sell:
                logger.info(
                    f"Bot {bot.id}: Grid SELL skipped at level {nearest_level} - "
                    f"Insufficient virtual crypto: {state['virtual_crypto']:.4f} < {crypto_to_sell:.4f}"
                )
                self._save_grid_state(bot.id, state)
                return _no_trade(
                    "Virtual crypto sufficiency", 0.0,
                    (
                        f"virtual crypto {state['virtual_crypto']:.6f} < {crypto_to_sell:.6f} "
                        f"needed at sell level {nearest_level}"
                    ),
                    f"Grid: Insufficient virtual crypto for sell at level {nearest_level}",
                    assumptions=_fill_assumptions,
                )

            # Execute virtual sell
            state["virtual_crypto"] -= crypto_to_sell
            state["virtual_cash"] += order_size_usd
            state["grid_levels"][nearest_level]["filled"] = True
            state["last_order_bar"] = completed_bars[-1]["start_ts"]
            state["total_trades"] += 1

            # Pillar 7: a realised sell fill IS a harvested buy->sell cycle — the
            # grid's profit unit — so record it as a win of roughly one grid
            # spacing (gross), the modelling the audit documents.
            edge_manager.record_trade_outcome(
                bot.id, "adaptive_grid",
                pnl=float(grid_spacing) * crypto_to_sell, win=True, at=now,
            )

            self._save_grid_state(bot.id, state)

            # Remaining virtual inventory decides CLOSE vs REDUCE: a sell that
            # empties the virtual crypto pool closes the grid's net position;
            # otherwise it reduces it (tasks.md 5.6, "remaining depth").
            _remaining_crypto = state["virtual_crypto"]
            _dust = crypto_to_sell * 1e-6
            if _remaining_crypto <= _dust:
                sell_intent = ExecutionIntent.CLOSE_POSITION
            else:
                sell_intent = ExecutionIntent.REDUCE_POSITION

            logger.info(
                f"Bot {bot.id}: Grid SELL at level {nearest_level} (depth={depth}) - "
                f"Price: ${bar_close_price:.2f}, Amount: ${order_size_usd:.2f} "
                f"(size_mult={size_multiplier:.2f}x, score_mult={score_size_multiplier:.2f}x), "
                f"intent={sell_intent.value}, Virtual: ${state['virtual_cash']:.2f} cash + "
                f"{state['virtual_crypto']:.4f} crypto | "
                f"Decision Score {decision_score.total:.1f}/{decision_score_threshold:.1f} | "
                f"Lifetime: {state['lifetime_return_pct']:+.2f}% return, "
                f"{state['lifetime_max_drawdown_pct']:.2f}% max DD"
            )

            gen = now
            proposal = StrategyProposal(
                strategy_id="adaptive_grid", bot_id=bot.id, generated_at=gen,
                direction=Direction.SELL, execution_intent=sell_intent,
                validity=ProposalValidity(
                    generated_at=gen,
                    valid_until=gen + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=decision_score, market_suitability=suitability,
                edge_status=edge_status, assumptions=_fill_assumptions,
                reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=order_size_usd,
                suggested_risk_budget_pct=base_order_size_pct * size_multiplier * score_size_multiplier,
                expected_holding_horizon="short",
                adaptive_parameters_used={
                    "depth_size_multiplier": size_multiplier,
                    "decision_score_size_multiplier": score_size_multiplier,
                    "grid_spacing": grid_spacing,
                },
                explanation=exp.to_dict(),
            )
            return StandaloneAdapter.to_trade_signal(proposal)

    def _get_grid_state(self, bot_id: int, params: dict) -> dict:
        """Get grid state for a bot, initializing if needed."""
        if not hasattr(self, "_grid_states"):
            self._grid_states = {}
        if bot_id not in self._grid_states:
            self._grid_states[bot_id] = {}
        return self._grid_states[bot_id]

    def _save_grid_state(self, bot_id: int, state: dict) -> None:
        """Save grid state for a bot."""
        if not hasattr(self, "_grid_states"):
            self._grid_states = {}
        self._grid_states[bot_id] = state

    def calculate_atr_proxy(self, bars: list, period: int) -> float:
        """ATR proxy from completed OHLC pseudo-bars.

        Average true range approximated as the mean (high - low) over the last
        ``period`` bars. Used by the Adaptive Grid range-escape kill switch.
        Returns 0.0 when there is insufficient data.

        NOTE: this is the missing companion to the bar-based grid. Without it
        the grid raised AttributeError on the first bar past ``atr_period`` -
        i.e. immediately after warmup, every run.
        """
        if not bars or len(bars) < period:
            return 0.0
        recent = bars[-period:]
        true_ranges = []
        for bar in recent:
            high = bar.get("high")
            low = bar.get("low")
            if isinstance(high, (int, float)) and isinstance(low, (int, float)):
                true_ranges.append(high - low)
        if not true_ranges:
            return 0.0
        return sum(true_ranges) / len(true_ranges)

    def _ema_slope_pct(self, prices: list, period: int) -> float:
        """Fractional slope of the EMA(period) over the last ``period`` bars.

        Deterministic Measurement for the Adaptive Grid's "range-bound
        conviction" Evidence Item (Pillar 3): computes the EMA series via the
        shared ``_calculate_ema`` and returns
        ``(ema[-1] - ema[-1-period]) / ema[-1-period]`` — the fractional change
        in the trend line across the window. ~0 means a flat range (grid edge
        intact); a larger magnitude means a developing trend (edge against).
        Returns 0.0 (neutral/flat) when there is insufficient data, so a
        warming-up grid is treated as range-bound rather than penalised.
        """
        if not prices or len(prices) < period * 2:
            return 0.0
        ema = self._calculate_ema(prices, period)
        if len(ema) < period + 1:
            return 0.0
        past = ema[-1 - period]
        if past == 0:
            return 0.0
        return (ema[-1] - past) / past

    async def _strategy_mean_reversion(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Mean Reversion strategy.

        Migrated to the Strategy Decision Framework
        (add-strategy-decision-framework, Phase 4 - see this change's
        design.md/tasks.md and audits/mean_reversion.md). ANTI-TREND,
        BOUNDED-RISK, REGIME-AWARE strategy for range-bound markets. Its
        Pillar 2 (regime suitability) and Pillar 6 (four-way exit management)
        were already the reference implementation among the six; Phase 4 adds
        the Evidence-Based Decision Score (Pillar 3), Decision-Score-weighted
        sizing (Pillar 5), Strategy Edge Management (Pillar 7), the missing
        Pillar 8 checks, and the StrategyProposal/Standalone-Adapter interface
        (Pillar 10), without changing the regime or exit logic.

        THEORY (Pillar 1): exploits short-horizon overreaction in a
        range-bound market. When price is pushed to a statistical extreme (a
        lower Bollinger Band = ~2 std below a rolling mean) by transient
        order-flow imbalance rather than new information, liquidity providers
        and value buyers step in and price tends to revert toward the mean.
        The edge exists only while the market is genuinely ranging (no trend);
        a trend turns "cheap" into "cheaper", so the regime gate and the
        trend-flip force-exit are the core risk control, not an add-on.

        NOT designed to hold through trends. Regime gating force-exits on trend flips.

        CRITICAL: All logic operates on AGGREGATED PSEUDO-BARS, not tick data.
        Bar interval defines time granularity (default: 60 seconds per bar).

        This is a DEFENSIVE STATISTICAL STRATEGY, not a conviction trade.

        Parameters:
            bar_interval_seconds: Time per bar for aggregation (default: 60)
            bollinger_period: Bollinger Band period in bars (default: 20)
            bollinger_std: Standard deviation multiplier (default: 1.8) -
                FIXED (Pillar 4): this DEFINES what "a statistical extreme"
                means for the strategy; changing it changes the strategy's
                identity, not how it adapts to conditions. (Docstring/code
                mismatch fixed - both now read 1.8.)
            atr_period: ATR period for hard stop (default: 14)
            atr_stop_multiplier: ATR stop multiplier (default: 2.0)
            max_hold_bars: Maximum bars to hold position (time stop, default: 10)
            order_size_percent: Percent of balance per order (default: 20)
            exit_at_mean: Exit at mean vs upper band (default: True)
            regime_filter_enabled: Enable regime gating (default: True). Kept
                (unlike volatility_breakout's removal) as a TEST-isolation
                affordance; Pillar 2 is enforced by default and remains this
                strategy's reference-implementation hard gate.
            cooldown_seconds: Seconds between trades (default: 300)
            decision_score_threshold: minimum Evidence-Based Decision Score
                (0-100) to enter (default: 40.0) - certification-pending.
            allowed_regimes: regimes suitable for entry (default:
                ["trend_flat", "volatility_high"]).
        """
        # Get parameters
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        period = params.get("bollinger_period", 20)
        std_mult = params.get("bollinger_std", 1.8)
        atr_period = params.get("atr_period", 14)
        atr_stop_mult = params.get("atr_stop_multiplier", 2.0)
        max_hold_bars = params.get("max_hold_bars", 10)
        order_size_percent = params.get("order_size_percent", 20) / 100
        exit_at_mean = params.get("exit_at_mean", True)
        regime_filter_enabled = params.get("regime_filter_enabled", True)
        cooldown_seconds = params.get("cooldown_seconds", 300)
        decision_score_threshold = params.get("decision_score_threshold", 40.0)
        _validity_interval_seconds = max(bar_interval_seconds, 1)
        if not hasattr(self, "_mean_reversion_edge_manager"):
            self._mean_reversion_edge_manager = StrategyEdgeManager()
        edge_manager = self._mean_reversion_edge_manager

        # === BAR AGGREGATION SYSTEM ===
        # Aggregate tick prices into fixed-time bars (OHLC)
        # Identical system to volatility_breakout

        # Initialize state tracking (INSTITUTIONAL STRUCTURE)
        if not hasattr(self, "_mean_reversion_states"):
            self._mean_reversion_states = {}

        state = self._mean_reversion_states.get(bot.id, {
            "bars": [],  # List of {"open", "high", "low", "close", "start_ts"}
            "current_bar": None,  # Bar being built
            "entry_price": None,
            "entry_atr": None,  # LOCKED at entry - risk never expands
            "target_price": None,  # LOCKED at entry - profit target never moves
            "hard_stop": None,  # ATR-based hard stop
            "bars_since_entry": 0,  # Time stop counter
            "last_exit_time": None,  # For cooldown
        })

        now = self.clock.now()

        # Initialize current bar if needed
        if state["current_bar"] is None:
            state["current_bar"] = {
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "start_ts": now,
            }

        # Update current bar with new tick
        current_bar = state["current_bar"]
        current_bar["high"] = max(current_bar["high"], current_price)
        current_bar["low"] = min(current_bar["low"], current_price)
        current_bar["close"] = current_price

        # Check if bar is complete (time-based)
        bar_duration = (now - current_bar["start_ts"]).total_seconds()
        bar_completed = bar_duration >= bar_interval_seconds

        if bar_completed:
            # Close current bar and add to history
            state["bars"].append(current_bar)
            # Keep sufficient bar history
            max_bars = max(period + 50, 100)
            state["bars"] = state["bars"][-max_bars:]

            # Start new bar
            state["current_bar"] = {
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "start_ts": now,
            }

            logger.debug(
                f"Bot {bot.id}: Mean Reversion - Bar completed: "
                f"O:{current_bar['open']:.2f} H:{current_bar['high']:.2f} "
                f"L:{current_bar['low']:.2f} C:{current_bar['close']:.2f}"
            )

        # Need enough bars for calculations
        if len(state["bars"]) < period:
            self._mean_reversion_states[bot.id] = state
            self._explain(bot.id).state("WARMING_UP").metric(
                "current_price", current_price
            ).check(
                "Bars collected", len(state["bars"]), f">= {period}", False,
                detail="warming up Bollinger window",
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Collecting bars ({len(state['bars'])}/{period})"
            )

        # === BAR-BASED INDICATOR CALCULATIONS ===
        # All indicators operate on bar close prices

        def calculate_bollinger_bands_from_bars(bars: list, period: int, std_mult: float):
            """Calculate Bollinger Bands from bar closes."""
            closes = [bar["close"] for bar in bars[-period:]]
            sma = sum(closes) / len(closes)

            # Calculate standard deviation
            variance = sum((c - sma) ** 2 for c in closes) / len(closes)
            std_dev = variance ** 0.5

            upper_band = sma + (std_mult * std_dev)
            lower_band = sma - (std_mult * std_dev)

            return sma, upper_band, lower_band

        def calculate_atr_from_bars(bars: list, period: int) -> float:
            """Calculate ATR proxy from bar data."""
            if len(bars) < period:
                return 0.0

            true_ranges = []
            for i in range(len(bars) - period, len(bars)):
                if i < 0:
                    continue
                # TR approximation: high - low of each bar
                tr = bars[i]["high"] - bars[i]["low"]
                true_ranges.append(tr)

            if not true_ranges:
                return 0.0

            return sum(true_ranges) / len(true_ranges)

        # Calculate indicators from completed bars
        sma, upper_band, lower_band = calculate_bollinger_bands_from_bars(
            state["bars"], period, std_mult
        )
        atr = calculate_atr_from_bars(state["bars"], atr_period)

        # === PILLAR 2: MARKET SUITABILITY (the reference implementation, now
        # routed through the shared MarketSuitabilityGate) ===
        # Mean reversion is suitable only in trend_flat / volatility_high;
        # a trending market force-exits (Pillar 6). Detect from THIS strategy's
        # own completed bar closes (the shared tick buffer is trend_following's;
        # a standalone MR bot would otherwise see the neutral default forever).
        # By here we already have >= period bars, so the detector has data. Uses
        # the price-only _detect_market_regime (bar CLOSES, no volatility_
        # direction) - which is sufficient here because this strategy's
        # allowed_regimes reference only LEVEL/trend tags (trend_flat,
        # volatility_high), never a *direction* tag, so nothing is lost.
        allowed_regimes = params.get("allowed_regimes", ["trend_flat", "volatility_high"])
        bar_closes = [b["close"] for b in state["bars"]] + [current_price]
        current_regime = self._detect_market_regime(bar_closes, None)
        trend_state = current_regime.get("trend_state", "flat")
        volatility_state = current_regime.get("volatility_state", "medium")
        regime_name = f"trend_{trend_state}, vol={volatility_state}"
        # regime_filter_enabled (kept for test isolation) maps to allowed=["all"]
        # when disabled, so `suitability` is always a real MarketSuitabilityResult
        # for the proposal, and Pillar 2 is enforced by default.
        suitability = MarketSuitabilityGate().evaluate(
            current_regime, allowed_regimes if regime_filter_enabled else ["all"],
        )
        regime_allows_entry = suitability.is_suitable
        force_exit_regime = regime_filter_enabled and trend_state in ["up", "down"]

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # === PILLAR 7: STRATEGY EDGE MANAGEMENT ===
        # Category A = regime unsuitable. Category B ("parameter mismatch") is
        # never cited for this strategy: this phase makes no parameter adaptive
        # (bollinger_std/period and the ATR stop multiplier are FIXED by design
        # - Pillar 4), so there is no adaptive base to point at as miscalibrated;
        # degradation classifies as A (regime) or C (edge gone) only. Documented
        # in the audit's Pillar 7 section.
        edge_status = edge_manager.evaluate(
            bot.id, "mean_reversion",
            regime_outside_suitable_range=not suitability.is_suitable,
            parameter_mismatch_evidence=None,
            now=now,
        )

        def _single_evidence_proposal(
            *, direction: "Direction", execution_intent: "ExecutionIntent",
            evidence_name: str, evidence_value: float, evidence_reason: str,
            threshold: float = 1.0, suggested_position_size: Optional[float] = None,
            assumptions: tuple = (),
        ) -> StrategyProposal:
            """Single-evidence proposal for every already-decided branch
            (exits, holds, no-trades)."""
            item = EvidenceItem(
                name=evidence_name, measurement=lambda d: evidence_value,
                normalization=lambda r: r, weight=100.0, reason=evidence_reason,
            )
            score = DecisionScoreEngine().score("mean_reversion", [item], {}, threshold=threshold)
            reasons_for, reasons_against = derive_reasons(score)
            gen = self.clock.now()
            return StrategyProposal(
                strategy_id="mean_reversion", bot_id=bot.id, generated_at=gen,
                direction=direction, execution_intent=execution_intent,
                validity=ProposalValidity(
                    generated_at=gen,
                    valid_until=gen + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=score, market_suitability=suitability, edge_status=edge_status,
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=suggested_position_size, suggested_risk_budget_pct=order_size_percent,
                explanation=self._explain(bot.id).to_dict(),
            )

        def _emit_hold(proposal: StrategyProposal, reason: str) -> TradeSignal:
            sig = StandaloneAdapter.to_trade_signal(proposal)
            return sig if sig is not None else TradeSignal(action="hold", amount=0, reason=reason)

        # Get last completed bar close for logic
        last_bar_close = state["bars"][-1]["close"] if state["bars"] else current_price

        logger.debug(
            f"Bot {bot.id}: Mean Reversion - Bar close: ${last_bar_close:.2f}, "
            f"SMA: ${sma:.2f}, Upper: ${upper_band:.2f}, Lower: ${lower_band:.2f}, "
            f"ATR: ${atr:.2f}, Regime: {regime_name}"
        )

        # PRODUCTION DIAGNOSIS (once per completed bar, ~1/min): exactly why this
        # bot is/ isn't trading - entry distance to the lower band, regime gate,
        # and position state. Answers "why no trade?" without DEBUG logging.
        if bar_completed:
            entry_gap_pct = (
                (last_bar_close - lower_band) / lower_band * 100 if lower_band > 0 else 0.0
            )
            cd_remaining = 0
            if state.get("last_exit_time") is not None:
                cd_remaining = max(0, int(cooldown_seconds - (
                    self.clock.now() - state["last_exit_time"]).total_seconds()))
            logger.info(
                f"Bot {bot.id}: MR no-trade diag - close=${last_bar_close:.2f} "
                f"lower_band=${lower_band:.2f} gap_to_entry={entry_gap_pct:+.3f}% "
                f"regime={regime_name} entry_allowed={regime_allows_entry} "
                f"cooldown={cd_remaining}s has_position={has_position} bars={len(state['bars'])}"
            )

        # --- Structured decision explanation (observe-only; never affects logic) ---
        # Records the EXACT numbers behind this evaluation: every band, the ATR,
        # the distances, cooldown, regime gate, and the entry/exit gates with
        # current-vs-required-vs-pass. This MR implementation is Bollinger+ATR
        # based and computes NO RSI, so none is reported (we never fabricate a
        # check that did not participate).
        exp = self._explain(bot.id)
        cd_remaining = 0
        if state.get("last_exit_time") is not None:
            cd_remaining = max(
                0,
                int(cooldown_seconds - (now - state["last_exit_time"]).total_seconds()),
            )
        dist_lower_pct = ((last_bar_close - lower_band) / lower_band * 100) if lower_band > 0 else 0.0
        dist_upper_pct = ((upper_band - last_bar_close) / upper_band * 100) if upper_band > 0 else 0.0
        exit_target = sma if exit_at_mean else upper_band
        exp.update({
            "current_price": current_price,
            "bar_close": last_bar_close,
            "middle_bb_sma": sma,
            "upper_bb": upper_band,
            "lower_bb": lower_band,
            "atr": atr,
            "distance_to_lower_pct": dist_lower_pct,
            "distance_to_upper_pct": dist_upper_pct,
            "exit_target": exit_target,
            "regime": regime_name,
            "bars_collected": len(state["bars"]),
            "cooldown_seconds": cooldown_seconds,
            "cooldown_remaining_s": cd_remaining,
            "hard_stop": state.get("hard_stop"),
            "bars_since_entry": state.get("bars_since_entry", 0),
            "max_hold_bars": max_hold_bars,
            "has_position": has_position,
        })
        if has_position:
            exp.check("Open position", "Yes", "held → evaluating exits", True)
            exp.check(
                f"Exit target ({'mean' if exit_at_mean else 'upper band'})",
                last_bar_close, f">= {exit_target:.2f}", last_bar_close >= exit_target,
                detail=f"{dist_upper_pct:+.3f}% from target",
            )
            _hs = state.get("hard_stop")
            if _hs is not None:
                exp.check("Hard stop", current_price, f"<= {_hs:.2f}", current_price <= _hs)
            _bse = state.get("bars_since_entry", 0)
            exp.check("Time stop", _bse, f">= {max_hold_bars} bars", _bse >= max_hold_bars)
            exp.check("Regime flip", regime_name, "trend_up/down forces exit", force_exit_regime)
        else:
            exp.check("Regime allows entry", regime_name, "trend_flat or volatility_high", regime_allows_entry)
            exp.check("Cooldown", f"{cd_remaining} sec remaining", f">= {cooldown_seconds} sec elapsed", cd_remaining == 0)
            exp.check("Open position", "No", "must be flat to enter", not has_position)
            exp.check(
                "Lower band touch", last_bar_close, f"<= {lower_band:.2f}",
                last_bar_close <= lower_band, detail=f"{dist_lower_pct:+.3f}% from band",
            )
        # Current state + next-trade preview (from values already computed above).
        _dist_to_band = last_bar_close - lower_band
        if has_position:
            exp.state("WAITING_EXIT").next_trade(
                current=last_bar_close, current_label="Bar close",
                target=exit_target, target_label="Exit target",
                distance=last_bar_close - exit_target,
                status=(f"{exit_target - last_bar_close:.2f} below target"
                        if last_bar_close < exit_target else "Target reached — exit armed"),
            )
        elif not regime_allows_entry:
            exp.state("WAITING_REGIME")
        elif cd_remaining > 0:
            exp.state("COOLDOWN").next_trade(
                current=cd_remaining, current_label="Cooldown remaining (s)",
                target=0, target_label="Ready", distance=cd_remaining,
                status=f"{cd_remaining}s until cooldown clears",
            )
        else:
            exp.state("ENTRY_ARMED" if last_bar_close <= lower_band else "WAITING_LOWER_BAND").next_trade(
                current=last_bar_close, current_label="Current price",
                target=lower_band, target_label="Lower band", distance=_dist_to_band,
                status=(f"Needs another {_dist_to_band:.2f} points lower"
                        if _dist_to_band > 0 else "At/below band — entry armed"),
            )
        # ---------------------------------------------------------------------

        # === POSITION EXIT LOGIC (BOUNDED RISK) - Pillar 6, unchanged logic;
        # now emits SELL/CLOSE_POSITION proposals and records edge outcomes ===
        # Exits: Regime flip | Mean reached | Hard stop | Time stop
        if has_position:
            pos = positions[0]
            # Increment bars_since_entry only when bar completes
            if bar_completed and state["entry_price"] is not None:
                state["bars_since_entry"] += 1

            # CRITICAL: LOCKED entry_atr (risk never expands); fallback for legacy.
            entry_atr_locked = state.get("entry_atr") or atr
            hard_stop = state.get("hard_stop", None)
            exit_label = "mean" if exit_at_mean else "upper band"
            # LOCKED target (never recomputed live - see the prior implementation's
            # note on why a live sma silently decays the take-profit into a
            # breakeven-or-worse exit). Legacy fallback once.
            target_price = state.get("target_price")
            if target_price is None:
                target_price = sma if exit_at_mean else upper_band

            def _mr_exit(*, evidence_name: str, evidence_reason: str) -> TradeSignal:
                # Pillar 7: record outcome BEFORE clearing state (only with a real
                # entry price - a fabricated value would corrupt the statistics).
                ep = state.get("entry_price")
                if ep is not None:
                    initial_risk = entry_atr_locked * atr_stop_mult
                    pnl_per_unit = current_price - ep
                    rr = (pnl_per_unit / initial_risk) if initial_risk > 0 else None
                    edge_manager.record_trade_outcome(
                        bot.id, "mean_reversion",
                        pnl=pnl_per_unit, win=(pnl_per_unit > 0), reward_risk_realized=rr,
                        holding_seconds=state.get("bars_since_entry", 0) * bar_interval_seconds,
                        at=self.clock.now(),
                    )
                state["entry_price"] = None
                state["entry_atr"] = None
                state["target_price"] = None
                state["hard_stop"] = None
                state["bars_since_entry"] = 0
                state["last_exit_time"] = self.clock.now()
                self._mean_reversion_states[bot.id] = state
                proposal = _single_evidence_proposal(
                    direction=Direction.SELL, execution_intent=ExecutionIntent.CLOSE_POSITION,
                    evidence_name=evidence_name, evidence_value=1.0,
                    evidence_reason=evidence_reason,
                    suggested_position_size=pos.amount * current_price,
                )
                return StandaloneAdapter.to_trade_signal(proposal)

            # EXIT 1: Regime flip (force exit) - a trending market makes MR wrong.
            if force_exit_regime:
                logger.info(f"Bot {bot.id}: Mean Reversion EXIT (regime flip) - {regime_name}")
                return _mr_exit(
                    evidence_name="Regime flip exit",
                    evidence_reason=f"regime flipped to {regime_name} - mean reversion not suitable for trends",
                )
            # EXIT 2: Mean reached (locked target).
            if last_bar_close >= target_price:
                logger.info(
                    f"Bot {bot.id}: Mean Reversion EXIT (target) - close ${last_bar_close:.2f} >= ${target_price:.2f}"
                )
                return _mr_exit(
                    evidence_name=f"{exit_label} reached",
                    evidence_reason=f"bar close ${last_bar_close:.2f} reached the locked {exit_label} target ${target_price:.2f}",
                )
            # EXIT 3: Hard stop (locked ATR-based).
            if hard_stop is not None and current_price <= hard_stop:
                logger.info(f"Bot {bot.id}: Mean Reversion EXIT (hard stop) - ${current_price:.2f} <= ${hard_stop:.2f}")
                return _mr_exit(
                    evidence_name="Hard stop hit",
                    evidence_reason=f"price ${current_price:.2f} <= locked hard stop ${hard_stop:.2f}",
                )
            # EXIT 4: Time stop (max hold bars).
            if state["bars_since_entry"] >= max_hold_bars:
                logger.info(f"Bot {bot.id}: Mean Reversion EXIT (time stop) - {state['bars_since_entry']} bars")
                return _mr_exit(
                    evidence_name="Time stop",
                    evidence_reason=f"held {state['bars_since_entry']} bars >= max {max_hold_bars} without reaching the mean",
                )

            # No exit - hold.
            self._mean_reversion_states[bot.id] = state
            stop_str = f"${hard_stop:.2f}" if hard_stop is not None else "N/A"
            hold_proposal = _single_evidence_proposal(
                direction=Direction.HOLD, execution_intent=ExecutionIntent.HOLD_POSITION,
                evidence_name="Position intact",
                evidence_value=1.0,
                evidence_reason="no exit condition (regime flip, mean target, hard stop, time stop) triggered this cycle",
                assumptions=("range/band structure not broken and the regime is still range-bound",),
            )
            return _emit_hold(
                hold_proposal,
                f"Mean Reversion: Holding, target ${target_price:.2f}, stop {stop_str}, "
                f"bars {state['bars_since_entry']}/{max_hold_bars}",
            )

        # === ENTRY LOGIC (BAR-BASED) ===
        # Entry: bar close <= lower BB (precondition), regime suitable (Pillar 2),
        # Decision Score clears threshold (Pillar 3), edge not Category C
        # (Pillar 7), cooldown elapsed.

        # PILLAR 2 hard gate: refuse unsuitable regimes.
        if not regime_allows_entry:
            self._mean_reversion_states[bot.id] = state
            return _emit_hold(
                _single_evidence_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Market suitability", evidence_value=0.0,
                    evidence_reason=suitability.reason,
                ),
                f"Mean Reversion: Waiting for suitable regime (current: {regime_name})",
            )

        # Cooldown check
        if state["last_exit_time"] is not None:
            time_since_exit = (self.clock.now() - state["last_exit_time"]).total_seconds()
            if time_since_exit < cooldown_seconds:
                remaining = int(cooldown_seconds - time_since_exit)
                self._mean_reversion_states[bot.id] = state
                return _emit_hold(
                    _single_evidence_proposal(
                        direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                        evidence_name="Cooldown elapsed", evidence_value=0.0,
                        evidence_reason=f"{remaining}s of post-exit cooldown remain",
                    ),
                    f"Mean Reversion: Cooldown ({remaining}s remaining)",
                )

        # PRECONDITION: bar close at/below the lower Bollinger Band. Mean
        # reversion only ever buys a statistical extreme - kept as a hard gate;
        # the Decision Score below grades the QUALITY of a genuine touch.
        if last_bar_close <= lower_band:
            # Expected profit: current execution price -> the SMA (MR's target).
            reversion_frac = (sma - current_price) / current_price if current_price > 0 else 0.0
            expected_move_pct = max(0.0, reversion_frac)
            penetration_frac = (lower_band - last_bar_close) / lower_band if lower_band > 0 else 0.0
            band_width_frac = (upper_band - lower_band) / sma if sma > 0 else 0.0

            # Pillar 8 + pre-flight viability: the band must cover round-trip
            # fees BEFORE state mutation (kept ahead of the score - a trade that
            # cannot clear fees is rejected regardless of conviction).
            _fee_raw = getattr(bot, 'exchange_fee', 0.1)
            _mr_fee_pct = (
                float(_fee_raw) if isinstance(_fee_raw, (int, float)) else 0.1
            ) / 100.0
            _mr_min_move = 2.0 * _mr_fee_pct + _VIABILITY_SAFETY_MARGIN_PCT
            exp.check("Fee viability", expected_move_pct, f">= {_mr_min_move:.5f}", expected_move_pct >= _mr_min_move)
            if expected_move_pct < _mr_min_move:
                self._mean_reversion_states[bot.id] = state
                return _emit_hold(
                    _single_evidence_proposal(
                        direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                        evidence_name="Fee viability", evidence_value=0.0,
                        evidence_reason=(
                            f"reversion target {expected_move_pct * 100:.3f}% < fee threshold "
                            f"{_mr_min_move * 100:.3f}% - cannot clear round-trip costs"
                        ),
                    ),
                    (
                        f"Mean Reversion: Band too narrow for fees "
                        f"({expected_move_pct * 100:.3f}% < {_mr_min_move * 100:.3f}%)"
                    ),
                )

            # === PILLAR 3: EVIDENCE-BASED DECISION SCORE ===
            evidence_items = [
                EvidenceItem(
                    name="Reversion target distance",
                    measurement=lambda d: reversion_frac,
                    normalization=lambda r: max(-1.0, min(1.0, r / 0.02)),
                    weight=35.0,
                    reason=(
                        "Theory (overreaction reversion): the reward is the distance from the "
                        "oversold price back to the mean; a larger gap to the SMA is a larger "
                        "expected reversion move, the core of the edge."
                    ),
                ),
                EvidenceItem(
                    name="Oversold penetration",
                    measurement=lambda d: penetration_frac,
                    normalization=lambda r: max(-1.0, min(1.0, r / 0.005)),
                    weight=35.0,
                    reason=(
                        "Theory: the further price has pushed BELOW the lower band (a ~2-std "
                        "extreme), the more likely the move is transient order-flow overreaction "
                        "rather than information, and the stronger the snap-back tends to be."
                    ),
                ),
                EvidenceItem(
                    name="Band width adequacy",
                    measurement=lambda d: band_width_frac,
                    normalization=lambda r: max(-1.0, min(1.0, (r - 0.01) / 0.03)),
                    weight=30.0,
                    reason=(
                        "Theory: reversion is only worth trading if the range is wide enough to "
                        "profit after fees; a wider band (more realized volatility) means more "
                        "room between the extreme and the mean, a collapsed band means none."
                    ),
                ),
            ]
            decision_score = DecisionScoreEngine().score(
                "mean_reversion", evidence_items, {}, threshold=decision_score_threshold,
            )
            edge_manager.record_decision_score(bot.id, "mean_reversion", decision_score.total)
            reasons_for, reasons_against = derive_reasons(decision_score)
            assumptions = (
                "range/band structure not broken: price stays within a mean-reverting range, "
                "not the start of a trend",
                "regime remains range-bound (trend_flat / high-volatility, not trending)",
            )
            exp.metric("decision_score_total", decision_score.total)
            exp.metric("decision_score_threshold", decision_score_threshold)
            exp.metric("edge_status_category", edge_status.category.value)
            exp.check(
                "Decision Score clears threshold", decision_score.total,
                f">= {decision_score_threshold:.1f}", decision_score.approved,
            )
            exp.check(
                "Strategy edge not disqualified", edge_status.category.value,
                f"!= {EdgeCategory.C.value}", edge_status.category != EdgeCategory.C,
                detail=edge_status.reason,
            )

            blocking_reasons = []
            if not decision_score.approved:
                blocking_reasons.append(
                    f"Decision Score {decision_score.total:.1f} < threshold {decision_score_threshold:.1f}"
                )
            if edge_status.category == EdgeCategory.C:
                blocking_reasons.append(f"Strategy Edge Management: Category C - {edge_status.reason}")
            if blocking_reasons:
                self._mean_reversion_states[bot.id] = state
                reason_text = "; ".join(blocking_reasons)
                return _emit_hold(
                    _single_evidence_proposal(
                        direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                        evidence_name="Framework gates", evidence_value=0.0,
                        evidence_reason=reason_text, assumptions=assumptions,
                    ),
                    f"Mean Reversion: band touched but {reason_text}",
                )

            # === PILLAR 5: DECISION-SCORE-WEIGHTED POSITION SIZING ===
            score_range = max(100.0 - decision_score_threshold, 1e-9)
            score_margin = max(0.0, min(1.0, (decision_score.total - decision_score_threshold) / score_range))
            size_multiplier = 0.5 + score_margin
            exp.metric("decision_score_size_multiplier", size_multiplier)

            buy_amount = min(
                bot.current_balance * order_size_percent * size_multiplier,
                bot.current_balance * _BUY_BALANCE_FRACTION,
            )
            if buy_amount < MIN_ORDER_USD:
                if bot.current_balance >= MIN_ORDER_USD:
                    buy_amount = MIN_ORDER_USD
                else:
                    exp.check("Position size >= min order", buy_amount, f">= {MIN_ORDER_USD:.0f}", False)
                    self._mean_reversion_states[bot.id] = state
                    return _emit_hold(
                        _single_evidence_proposal(
                            direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                            evidence_name="Order size", evidence_value=0.0,
                            evidence_reason=(
                                f"balance ${bot.current_balance:.2f} below ${MIN_ORDER_USD:.0f} minimum order"
                            ),
                        ),
                        (
                            f"Mean Reversion: balance ${bot.current_balance:.2f} "
                            f"below ${MIN_ORDER_USD:.0f} minimum order"
                        ),
                    )
            exp.check("Position size >= min order", buy_amount, f">= {MIN_ORDER_USD:.0f}", True)

            # Lock entry state (LOCKED entry_atr / hard stop / target).
            locked_target = sma if exit_at_mean else upper_band
            locked_stop = current_price - (atr * atr_stop_mult)
            exp.check(
                "Initial hard stop below entry", locked_stop, f"< {current_price:.2f}",
                locked_stop < current_price,
            )
            state["entry_price"] = current_price
            state["entry_atr"] = atr
            state["target_price"] = locked_target
            state["hard_stop"] = locked_stop
            state["bars_since_entry"] = 0
            self._mean_reversion_states[bot.id] = state

            expected_risk_pct = (
                (current_price - locked_stop) / current_price if current_price > 0 else 0.0
            )

            logger.info(
                f"Bot {bot.id}: Mean Reversion ENTRY - "
                f"Decision Score {decision_score.total:.1f}/{decision_score_threshold:.1f}, "
                f"Bar close ${last_bar_close:.2f} <= lower BB ${lower_band:.2f}, "
                f"Entry ATR ${atr:.4f}, Position: ${buy_amount:.2f}"
            )

            # === PILLAR 10: STRATEGYPROPOSAL -> STANDALONE ADAPTER ===
            gen = self.clock.now()
            proposal = StrategyProposal(
                strategy_id="mean_reversion", bot_id=bot.id, generated_at=gen,
                direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
                validity=ProposalValidity(
                    generated_at=gen,
                    valid_until=gen + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=decision_score, market_suitability=suitability, edge_status=edge_status,
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=buy_amount,
                suggested_risk_budget_pct=order_size_percent * size_multiplier,
                expected_holding_horizon="short",
                adaptive_parameters_used={"decision_score_size_multiplier": size_multiplier},
                explanation=exp.to_dict(),
            )
            return StandaloneAdapter.to_trade_signal(
                proposal, expected_move_pct=expected_move_pct, expected_risk_pct=expected_risk_pct,
            )

        # No band touch - wait.
        self._mean_reversion_states[bot.id] = state
        return _emit_hold(
            _single_evidence_proposal(
                direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                evidence_name="Lower band touch", evidence_value=0.0,
                evidence_reason=(
                    f"bar close ${last_bar_close:.2f} above lower band ${lower_band:.2f} - "
                    "no statistical extreme to fade"
                ),
            ),
            f"Mean Reversion: Waiting for lower band (current: ${last_bar_close:.2f}, target: ${lower_band:.2f})",
        )

    async def _reconcile_live_account(
        self,
        exchange: ExchangeService,
        session: AsyncSession,
    ) -> None:
        """Compare exchange balances against running live bots' expectations.

        Sufficiency check, alert-only (never stops bots): the exchange account
        must hold at least the aggregate virtual cash (quote currency) and the
        aggregate open position amounts (base assets) of all running live
        bots. Shortfalls beyond the tolerance produce a warning log and an
        Alert record (alert_type "balance_reconciliation").

        Throttled internally; safe to call from every bot loop iteration.

        ARCHITECTURAL ASSUMPTION (M-4): all live bots share a SINGLE exchange
        account. Expectations are aggregated across every live bot and compared
        against the one ``exchange`` passed in, and the reconciliation clock
        (``_last_reconciliation``) is global on purpose. If multi-account or
        multi-exchange support is ever added, this must become per-account.
        """
        from .config import config_service

        interval = config_service.get("trading.reconciliation_interval_seconds") or 300
        now = self.clock.now()
        if self._last_reconciliation and (now - self._last_reconciliation).total_seconds() < interval:
            return
        self._last_reconciliation = now

        # Aggregate expectations across ALL running live bots (single account)
        result = await session.execute(
            select(Bot).where(
                Bot.status == BotStatus.RUNNING,
                Bot.is_dry_run == False,  # noqa: E712
            )
        )
        live_bots = result.scalars().all()
        if not live_bots:
            return

        tolerance = 0.01  # 1%
        expected: Dict[str, float] = {}

        for live_bot in live_bots:
            quote_asset = live_bot.trading_pair.split("/")[1]
            expected[quote_asset] = expected.get(quote_asset, 0.0) + (live_bot.current_balance or 0.0)

        live_bot_ids = [b.id for b in live_bots]
        pos_result = await session.execute(
            select(Position).where(Position.bot_id.in_(live_bot_ids))
        )
        for position in pos_result.scalars().all():
            base_asset = position.trading_pair.split("/")[0]
            expected[base_asset] = expected.get(base_asset, 0.0) + (position.amount or 0.0)

        shortfalls = []
        for asset, expected_amount in expected.items():
            if expected_amount <= 0:
                continue
            balance = await exchange.get_balance(asset)
            if balance is None:
                logger.warning(f"Reconciliation: could not fetch {asset} balance, skipping")
                continue
            if balance.total < expected_amount * (1 - tolerance):
                shortfalls.append((asset, expected_amount, balance.total))

        if not shortfalls:
            logger.info(
                f"Reconciliation OK: exchange covers {len(live_bots)} live bot(s) "
                f"across {len(expected)} asset(s)"
            )
            return

        for asset, expected_amount, actual in shortfalls:
            message = (
                f"Balance reconciliation shortfall: exchange holds {actual:.8f} {asset} "
                f"but running live bots expect {expected_amount:.8f} {asset}. "
                f"Check for withdrawals, fee drift, or accounting errors."
            )
            logger.warning(message)
            session.add(Alert(
                bot_id=None,
                alert_type="balance_reconciliation",
                message=message,
            ))
        await session.commit()

    def _get_price_history(self, bot_id: int) -> list:
        """Get price history for a bot."""
        if not hasattr(self, "_price_histories"):
            self._price_histories = {}
        return self._price_histories.get(bot_id, [])

    def _save_price_history(self, bot_id: int, history: list, max_len: int = 100) -> None:
        """Save price history for a bot, keeping last max_len entries."""
        if not hasattr(self, "_price_histories"):
            self._price_histories = {}
        # Keep only the last max_len prices
        self._price_histories[bot_id] = history[-max_len:]

    def _calc_price_atr_proxy(self, prices: list, period: int) -> float:
        """Calculate an ATR proxy from a plain (tick/close) price series.

        Same formula as trend_following's inline calculate_atr_proxy: True Range
        is approximated as the absolute difference between consecutive prices
        (no OHLC available for tick data), averaged over the last `period`
        differences. Extracted here as a shared, reusable method - rather than
        duplicating trend_following's private closure or modifying it to call
        out to this method - specifically so dip_recovery (and any future
        tick-based strategy) does not reimplement the math, while leaving
        trend_following's own pinned implementation completely untouched.

        Returns 0.0 if insufficient data.
        """
        if len(prices) < period + 1:
            return 0.0

        true_ranges = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        if len(true_ranges) < period:
            return 0.0

        recent_trs = true_ranges[-period:]
        return sum(recent_trs) / len(recent_trs)

    # Note: _strategy_momentum (breakdown_momentum) was removed (stub implementation, overlaps with other strategies)

    async def _strategy_trend_following(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Trend Following (time-series momentum) strategy.

        Migrated to the Strategy Decision Framework
        (add-strategy-decision-framework, Phase 2 - see this change's
        design.md/tasks.md, and audits/trend_following.md for the full
        Strategy Audit Document). Complete decision flow:

            Market Data
                -> Market Suitability Gate (shared MarketSuitabilityGate -
                   a HARD gate; refuses new entries in an unsuitable regime.
                   The audit's most acute finding was that a STANDALONE
                   trend_following bot had ZERO internal regime awareness -
                   this gate is built from scratch here; Pillar 2)
                -> Adaptive Parameter Resolver (shared
                   AdaptiveParameterResolver - the ATR stop multiplier is
                   resolved from the current ATR percentile every cycle,
                   then LOCKED at entry; Pillar 4)
                -> Evidence Collection (trend strength, price participation,
                   confirmation persistence, volatility-normalized trend -
                   all deterministic, all traced to the Theory section of
                   the Strategy Audit Document; Pillar 3)
                -> Evidence-Based Decision Score (shared DecisionScoreEngine),
                   replacing the pre-migration boolean EMA-cross +
                   confirmation-count gate
                -> Strategy Edge Management (shared StrategyEdgeManager -
                   refuses new entries once classified Category C; never
                   force-closes an existing position; Pillar 7)
                -> StrategyProposal (shared, immutable; Pillar 10)
                -> Standalone Adapter (shared StandaloneAdapter - the ONLY
                   place a StrategyProposal becomes the TradeSignal this
                   function returns)
                -> existing execution pipeline (UNCHANGED)

        THEORY (Pillar 1 - see the Strategy Audit Document for the full
        write-up): exploits the documented time-series-momentum anomaly.
        Established trends persist longer than an efficient market would
        allow because information diffuses gradually, trend-following flows
        (CTAs, momentum funds) reinforce the move, and behavioral
        herding/disposition effects delay full repricing. Long-only; enters
        only WITH a confirmed up-trend (fast EMA above slow EMA AND price
        above slow EMA), never counter-trend.

        Entry: an Evidence-Based Decision Score over the up-trend evidence
        clears its threshold, the regime is suitable, edge health is not
        Category C, and re-entry cooldown has elapsed.
        Exit: price < EMA(long) (confirmed over N loops) OR trailing stop hit
        (immediate). Trailing stop distance is LOCKED at entry ATR x the
        LOCKED adaptive multiplier, so risk never increases mid-trade.

        Parameters:
            short_period: EMA short period (default: 50)
            long_period: EMA long period (default: 100)
            atr_period: ATR period in bars (default: 14)
            atr_multiplier: BASE ATR stop-loss multiplier (default: 2.0) -
                ADAPTIVE (Pillar 4): resolved every cycle via
                AdaptiveParameterResolver.atr_percentile_scaled_multiplier
                against current-bar-ATR-vs-its-own-20-bar-average, then
                LOCKED at entry (risk never expands mid-trade).
            min_atr_multiplier / max_atr_multiplier: bounds for the adaptive
                resolution above (default: 1.0 / 4.0).
            risk_percent: Percent of capital to risk per trade (default: 1.0)
            entry_confirmation_loops: Consecutive loops the base up-trend
                condition must persist to reach full "confirmation
                persistence" Evidence (default: 3) - FIXED (Pillar 4): a
                debounce count against noise, not a volatility-derived value.
            exit_confirmation_loops: Consecutive loops required for a
                trend-break exit (default: 2) - FIXED (Pillar 4).
            cooldown_seconds: Seconds to wait after exit before re-entry
                (default: 300) - FIXED (Pillar 4): an anti-churn debounce.
            bar_interval_seconds: Time per bar for regime/ATR aggregation
                (default: 60).
            allowed_regimes: Regimes this strategy is suitable in (default:
                ["trend_up", "volatility_expanding"]) - now a HARD gate
                (Pillar 2) via the shared MarketSuitabilityGate, matching
                _get_strategy_capabilities()'s own declaration for Auto
                dispatch (both route through the SAME regime-tag convention).
            decision_score_threshold: minimum Evidence-Based Decision Score
                (0-100) required to enter (default: 40.0) - an initial,
                certification-pending value; see the Strategy Audit
                Document's before/after backtest comparison.
        """
        short_period = params.get("short_period", 50)
        long_period = params.get("long_period", 100)
        atr_period = params.get("atr_period", 14)
        atr_mult_base = params.get("atr_multiplier", 2.0)
        min_atr_mult = params.get("min_atr_multiplier", 1.0)
        max_atr_mult = params.get("max_atr_multiplier", 4.0)
        risk_percent = params.get("risk_percent", 1.0) / 100
        entry_confirmation_loops = params.get("entry_confirmation_loops", 3)
        exit_confirmation_loops = params.get("exit_confirmation_loops", 2)
        cooldown_seconds = params.get("cooldown_seconds", 300)  # 5 minutes default
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        allowed_regimes = params.get("allowed_regimes", ["trend_up", "volatility_expanding"])
        decision_score_threshold = params.get("decision_score_threshold", 40.0)
        # ProposalValidity requires a strictly positive window; some test
        # harnesses set bar_interval_seconds=0 ("close a bar every call").
        _validity_interval_seconds = max(bar_interval_seconds, 1)

        # Normalize persisted state before any early-return so missing keys added
        # after state was first saved (e.g. tf_bars) are backfilled on every tick,
        # including the warmup ticks that return before reaching the trading logic.
        if not hasattr(self, "_trend_states"):
            self._trend_states = {}
        if not hasattr(self, "_trend_following_edge_manager"):
            # Pillar 7: one shared, continuous tracker for this strategy,
            # keyed internally by (bot_id, strategy) - see edge_management.py.
            self._trend_following_edge_manager = StrategyEdgeManager()
        state = _normalize_trend_state(self._trend_states.get(bot.id, {}))
        edge_manager = self._trend_following_edge_manager

        # Get price history
        price_history = self._get_price_history(bot.id)

        # Add current price to history
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history, max_len=max(long_period + 50, 250))

        # Need enough data for EMA calculation
        if len(price_history) < long_period:
            self._trend_states[bot.id] = state  # Persist normalization even on early return
            self._explain(bot.id).state("WARMING_UP").metric("current_price", current_price).check(
                "Data collected", len(price_history), f">= {long_period}", False,
                detail="warming up EMA window",
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Trend Following: Collecting data ({len(price_history)}/{long_period})"
            )

        # Calculate EMA (Exponential Moving Average)
        def calculate_ema(prices: list, period: int) -> float:
            """Calculate EMA using standard formula."""
            if len(prices) < period:
                return sum(prices) / len(prices)  # Fall back to SMA

            # Use SMA for the first value
            k = 2 / (period + 1)  # Smoothing factor
            ema = sum(prices[:period]) / period

            # Calculate EMA for remaining values
            for price in prices[period:]:
                ema = (price * k) + (ema * (1 - k))

            return ema

        # Calculate indicators
        ema_short = calculate_ema(price_history, short_period)
        ema_long = calculate_ema(price_history, long_period)

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # === BAR ATR ACCUMULATION ===
        # Replace tick-level ATR with bar-based ATR.  The original code computed
        # |price[i] − price[i-1]| at 1 Hz → ATR ≈ $1-3 on BTC, placing stops
        # well inside the fee hurdle (guaranteed loss on every trade).
        # 60-second bar H-L ranges (~$50-200 on BTC) reflect actual volatility.
        _now = self.clock.now()
        if state.get("tf_current_bar") is None:
            state["tf_current_bar"] = {
                "high": current_price, "low": current_price,
                "close": current_price, "start_ts": _now,
            }
        _cb = state["tf_current_bar"]
        _cb["high"] = max(_cb["high"], current_price)
        _cb["low"] = min(_cb["low"], current_price)
        _cb["close"] = current_price
        bar_completed = (_now - _cb["start_ts"]).total_seconds() >= bar_interval_seconds
        if bar_completed:
            state["tf_bars"].append(dict(_cb))
            state["tf_bars"] = state["tf_bars"][-100:]
            state["tf_current_bar"] = {
                "high": current_price, "low": current_price,
                "close": current_price, "start_ts": _now,
            }
        self._trend_states[bot.id] = state

        def _calc_bar_atr(bars: list, period: int) -> float:
            if len(bars) < period:
                return 0.0
            return sum(b["high"] - b["low"] for b in bars[-period:]) / period

        tf_bars = state.get("tf_bars", [])
        bar_atr = _calc_bar_atr(tf_bars, atr_period)

        # Track per-bar ATR history (only when a bar completes) so the
        # adaptive stop-multiplier resolver (Pillar 4) has a current-vs-
        # recent-average ATR ratio to scale by.
        if bar_completed and bar_atr > 0:
            state["tf_atr_history"].append(bar_atr)
            state["tf_atr_history"] = state["tf_atr_history"][-100:]

        # Fee-coverage floor: when bar ATR is zero (not enough bars collected yet),
        # use a stop distance that at minimum covers the round-trip fee + safety.
        # This prevents microscopic stops during the bar warmup window AND fixes
        # the original tick-ATR bug (1-second tick diffs ~$1-3 on BTC, well inside
        # the fee hurdle).  Once bars are ready, the larger bar ATR takes over.
        _fee_raw_tf = getattr(bot, 'exchange_fee', 0.1)
        _fee_pct_tf = (
            float(_fee_raw_tf) if isinstance(_fee_raw_tf, (int, float)) else 0.1
        ) / 100.0
        _min_stop_pct = 2.0 * _fee_pct_tf + _VIABILITY_SAFETY_MARGIN_PCT
        atr = max(bar_atr, current_price * _min_stop_pct)

        logger.debug(
            f"Bot {bot.id}: Trend Following - Price: ${current_price:.2f}, "
            f"EMA({short_period}): ${ema_short:.2f}, EMA({long_period}): ${ema_long:.2f}, "
            f"ATR: ${atr:.2f}"
        )

        # === PILLAR 2: MARKET SUITABILITY GATE (shared, HARD-enforced) ===
        # Built from scratch here - the audit's most acute finding was that a
        # STANDALONE trend_following bot had ZERO internal regime awareness.
        # Uses the same bar-based direction detector and regime-tag convention
        # _strategy_auto / _is_strategy_eligible use, so this gate and Auto
        # dispatch can never disagree about what "trend_up" means. Enforced as
        # a hard NO_TRADE on entries below (never on exits).
        current_regime = self._detect_market_regime_bar_based(
            state.get("tf_bars", []), state.get("regime_state")
        )
        state["regime_state"] = current_regime
        suitability = MarketSuitabilityGate().evaluate(current_regime, allowed_regimes)

        # === PILLAR 4: ADAPTIVE PARAMETER RESOLVER ===
        # Stop multiplier scales with current bar-ATR relative to its own
        # 20-bar average - wider when volatility is currently elevated,
        # tighter when calm. Resolved every cycle; LOCKED at entry so an open
        # position's risk never expands mid-trade.
        recent_atr_window = state.get("tf_atr_history", [])[-20:]
        avg_atr = (sum(recent_atr_window) / len(recent_atr_window)) if recent_atr_window else bar_atr
        atr_percentile = (bar_atr / avg_atr) if avg_atr > 0 else 1.0
        resolved_stop = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
            name="atr_multiplier",
            atr_percentile=atr_percentile,
            base_multiplier=atr_mult_base,
            min_multiplier=min_atr_mult,
            max_multiplier=max_atr_mult,
        )
        atr_stop_mult = resolved_stop.value

        # === PILLAR 7: STRATEGY EDGE MANAGEMENT ===
        # regime_outside_suitable_range comes from the Pillar 2 gate above.
        # parameter_mismatch_evidence is cited only when bar-ATR has drifted
        # far from the window the resolver's BASE multiplier implicitly
        # assumes "normal" - a citation for a future Category B response,
        # never a free-text guess.
        parameter_mismatch_evidence = None
        if atr_percentile > 1.5 or atr_percentile < 0.6:
            parameter_mismatch_evidence = (
                f"bar-ATR percentile {atr_percentile:.2f}x its 20-bar average - "
                f"atr_multiplier base {atr_mult_base} may be miscalibrated for "
                "sustained current volatility (the live value is already "
                "adaptively resolved; sustained drift outside [0.6, 1.5] is "
                "evidence the BASE itself may need certification-phase "
                "recalibration)"
            )
        edge_status = edge_manager.evaluate(
            bot.id, "trend_following",
            regime_outside_suitable_range=not suitability.is_suitable,
            parameter_mismatch_evidence=parameter_mismatch_evidence,
            now=self.clock.now(),
        )

        logger.debug(
            f"Bot {bot.id}: Trend Following - Price: ${current_price:.2f}, "
            f"EMA({short_period}): ${ema_short:.2f}, EMA({long_period}): ${ema_long:.2f}, "
            f"ATR: ${atr:.2f}, Suitable: {suitability.is_suitable} "
            f"({suitability.reason}), Edge: {edge_status.category.value}"
        )

        def _single_evidence_proposal(
            *, direction: "Direction", execution_intent: "ExecutionIntent",
            evidence_name: str, evidence_value: float, evidence_reason: str,
            threshold: float, suggested_position_size: Optional[float] = None,
            assumptions: tuple = (),
        ) -> StrategyProposal:
            """Shared helper for every branch that isn't the full multi-
            factor entry evaluation: exits, holds, and suitability/cooldown/
            edge-blocked no-trades. Each is a single, deterministic,
            already-decided condition - a single Evidence Item correctly
            represents that, per Pillar 3 (nothing subjective, it just isn't
            MULTI-factor)."""
            item = EvidenceItem(
                name=evidence_name,
                measurement=lambda d: evidence_value,
                normalization=lambda r: r,
                weight=100.0,
                reason=evidence_reason,
            )
            score = DecisionScoreEngine().score(
                "trend_following", [item], {}, threshold=threshold,
            )
            reasons_for, reasons_against = derive_reasons(score)
            generated_at = self.clock.now()
            return StrategyProposal(
                strategy_id="trend_following",
                bot_id=bot.id,
                generated_at=generated_at,
                direction=direction,
                execution_intent=execution_intent,
                validity=ProposalValidity(
                    generated_at=generated_at,
                    valid_until=generated_at + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=score,
                market_suitability=suitability,
                edge_status=edge_status,
                assumptions=assumptions,
                reasons_for=reasons_for,
                reasons_against=reasons_against,
                suggested_position_size=suggested_position_size,
                suggested_risk_budget_pct=risk_percent,
                explanation=self._explain(bot.id).to_dict(),
            )

        # === PILLAR 6: CONTINUOUS TRADE MANAGEMENT (position open) ===
        if has_position:
            pos = positions[0]
            _exp = self._explain(bot.id)
            # CRITICAL: use LOCKED entry values (risk never expands mid-trade).
            # Fallbacks cover legacy positions from before this migration.
            entry_atr_locked = state.get("entry_atr") or atr
            entry_stop_mult_locked = state.get("entry_stop_multiplier") or atr_stop_mult
            entry_price_known = state.get("entry_price")
            entry_time = state.get("entry_time")
            holding_seconds = (
                (self.clock.now() - entry_time).total_seconds()
                if isinstance(entry_time, datetime) else 0.0
            )
            initial_risk_per_unit = entry_atr_locked * entry_stop_mult_locked

            # Maintain the monotonic trailing stop using the LOCKED entry ATR x
            # LOCKED adaptive multiplier (risk never expands).
            if state["highest_price"] is None or current_price > state["highest_price"]:
                state["highest_price"] = current_price
                state["trailing_stop"] = current_price - (entry_atr_locked * entry_stop_mult_locked)
                state["exit_confirmation_count"] = 0  # reset trend-break confirm on new high

            _ts = state.get("trailing_stop")
            trailing_stop_hit = _ts is not None and current_price <= _ts
            trend_break = current_price < ema_long
            _proj_exit = (state.get("exit_confirmation_count", 0) + 1) if trend_break else 0

            _exp.update({
                "current_price": current_price,
                "ema_fast": ema_short,
                "ema_slow": ema_long,
                "ema_distance": ema_short - ema_long,
                "atr": atr,
                "atr_percentile": atr_percentile,
                "trailing_stop": _ts,
                "highest_price": state.get("highest_price"),
                "holding_seconds": holding_seconds,
                "exit_confirmation_count": state.get("exit_confirmation_count", 0),
                "exit_threshold_loops": exit_confirmation_loops,
                "regime_tags": suitability.regime_tags,
                "market_suitable": suitability.is_suitable,
                "has_position": True,
            })
            _exp.metric("edge_status_category", edge_status.category.value)
            if _ts is not None:
                _exp.check("Trailing stop hit", current_price, f"<= {_ts:.2f}", trailing_stop_hit)
            _exp.check(
                "Trend break (price < slow EMA)", current_price, f"< {ema_long:.2f}",
                trend_break, detail=f"confirm {_proj_exit}/{exit_confirmation_loops}",
            )
            _exp.state("LONG_OPEN").next_trade(
                current=current_price, current_label="Current price",
                target=(_ts if _ts is not None else ema_long),
                target_label=("Trailing stop" if _ts is not None else "Slow EMA (exit)"),
                distance=(current_price - _ts) if _ts is not None else (current_price - ema_long),
                status="holding — exits on trailing stop or confirmed trend break",
            )

            def _record_exit_outcome() -> None:
                # Pillar 7: record the outcome BEFORE state is cleared. Only
                # when entry_price is actually known - a position with no
                # locally-tracked entry (imported, or opened pre-migration)
                # has no real P&L; a fabricated value would corrupt the
                # StrategyEdgeManager's statistics.
                if entry_price_known is None:
                    return
                pnl_per_unit = current_price - entry_price_known
                reward_risk_realized = (
                    (pnl_per_unit / initial_risk_per_unit) if initial_risk_per_unit > 0 else None
                )
                edge_manager.record_trade_outcome(
                    bot.id, "trend_following",
                    pnl=pnl_per_unit, win=(pnl_per_unit > 0),
                    reward_risk_realized=reward_risk_realized,
                    holding_seconds=holding_seconds, at=self.clock.now(),
                )

            def _reset_position_state() -> None:
                # Clear position keys but PRESERVE bar/regime history
                # (tf_bars, tf_atr_history, tf_current_bar, regime_state) so
                # the Pillar 2 regime gate does not have to re-warm from zero
                # after every exit. Mirrors volatility_breakout's exit reset.
                state["trailing_stop"] = None
                state["highest_price"] = None
                state["entry_atr"] = None
                state["entry_stop_multiplier"] = None
                state["entry_price"] = None
                state["entry_time"] = None
                state["last_exit_time"] = self.clock.now()
                state["entry_confirmation_count"] = 0
                state["exit_confirmation_count"] = 0
                self._trend_states[bot.id] = state

            # Exit 1: trailing stop hit (hard stop - no confirmation needed).
            if trailing_stop_hit:
                logger.info(
                    f"Bot {bot.id}: Trend Following EXIT (trailing stop) - "
                    f"Price ${current_price:.2f} <= Stop ${_ts:.2f}, "
                    f"Entry ATR: ${entry_atr_locked:.4f} x {entry_stop_mult_locked:.2f}"
                )
                _record_exit_outcome()
                proposal = _single_evidence_proposal(
                    direction=Direction.SELL,
                    execution_intent=ExecutionIntent.CLOSE_POSITION,
                    evidence_name="Trailing stop hit",
                    evidence_value=1.0,
                    evidence_reason=(
                        f"Price ${current_price:.2f} <= locked trailing stop ${_ts:.2f} "
                        "— risk control exit"
                    ),
                    threshold=1.0,
                    suggested_position_size=pos.amount * current_price,
                )
                _reset_position_state()
                return StandaloneAdapter.to_trade_signal(proposal)

            # Exit 2: price below EMA(long) - trend break (requires confirmation).
            if trend_break:
                state["exit_confirmation_count"] = state.get("exit_confirmation_count", 0) + 1
                self._trend_states[bot.id] = state
                if state["exit_confirmation_count"] < exit_confirmation_loops:
                    hold_proposal = _single_evidence_proposal(
                        direction=Direction.HOLD,
                        execution_intent=ExecutionIntent.HOLD_POSITION,
                        evidence_name="Trend-break confirmation building",
                        evidence_value=1.0,
                        evidence_reason=(
                            f"price < slow EMA for "
                            f"{state['exit_confirmation_count']}/{exit_confirmation_loops} "
                            "loops — not yet a confirmed trend break"
                        ),
                        threshold=1.0,
                    )
                    adapter_signal = StandaloneAdapter.to_trade_signal(hold_proposal)
                    if adapter_signal is not None:
                        return adapter_signal
                    return TradeSignal(
                        action="hold", amount=0,
                        reason=(
                            f"Trend Following: Exit confirmation "
                            f"{state['exit_confirmation_count']}/{exit_confirmation_loops} (price < EMA)"
                        ),
                    )

                logger.info(
                    f"Bot {bot.id}: Trend Following EXIT (trend break confirmed) - "
                    f"Price ${current_price:.2f} < EMA({long_period}) ${ema_long:.2f}, "
                    f"Confirmed over {exit_confirmation_loops} loops"
                )
                _record_exit_outcome()
                proposal = _single_evidence_proposal(
                    direction=Direction.SELL,
                    execution_intent=ExecutionIntent.CLOSE_POSITION,
                    evidence_name="Confirmed trend break",
                    evidence_value=1.0,
                    evidence_reason=(
                        f"Price ${current_price:.2f} < slow EMA ${ema_long:.2f}, confirmed "
                        f"over {exit_confirmation_loops} loops — the trend thesis has broken"
                    ),
                    threshold=1.0,
                    suggested_position_size=pos.amount * current_price,
                )
                _reset_position_state()
                return StandaloneAdapter.to_trade_signal(proposal)

            # No exit this tick - price still above slow EMA: reset the
            # trend-break confirmation counter and hold.
            state["exit_confirmation_count"] = 0
            self._trend_states[bot.id] = state

            hold_proposal = _single_evidence_proposal(
                direction=Direction.HOLD,
                execution_intent=ExecutionIntent.HOLD_POSITION,
                evidence_name="Trend intact",
                evidence_value=1.0,
                evidence_reason=(
                    "price above the slow EMA and trailing stop not hit — the "
                    "up-trend thesis still holds"
                ),
                threshold=1.0,
                assumptions=(
                    "trend direction unchanged: price remains above the slow "
                    "EMA and the locked trailing stop is not hit",
                ),
            )
            adapter_signal = StandaloneAdapter.to_trade_signal(hold_proposal)
            if adapter_signal is not None:
                return adapter_signal
            # A conditional inside an f-string format spec is a ValueError
            # ("Invalid format specifier"); format the optional stop outside it.
            stop_str = (
                f"${state['trailing_stop']:.2f}"
                if state["trailing_stop"] is not None else "N/A"
            )
            reason_text = "; ".join(hold_proposal.reasons_for) or "trend still valid"
            return TradeSignal(
                action="hold", amount=0,
                reason=f"Trend Following: Holding position, stop at {stop_str} ({reason_text})",
            )

        # === ENTRY SIDE (no position) ===
        exp = self._explain(bot.id)

        # Re-entry cooldown (anti-churn) - computed as a blocking reason below,
        # not a separate early return, so the Evidence/edge diagnostics are
        # still produced every cycle.
        cooldown_remaining = 0
        if state.get("last_exit_time") is not None:
            elapsed = (self.clock.now() - state["last_exit_time"]).total_seconds()
            if elapsed < cooldown_seconds:
                cooldown_remaining = int(cooldown_seconds - elapsed)

        # Base up-trend condition drives the confirmation-persistence counter
        # (folded into the Decision Score below as an Evidence Item, replacing
        # the pre-migration hard N-loop gate).
        base_trend_ok = current_price > ema_long and ema_short > ema_long
        if base_trend_ok:
            state["entry_confirmation_count"] = state.get("entry_confirmation_count", 0) + 1
        else:
            state["entry_confirmation_count"] = 0
        confirmation_count = state["entry_confirmation_count"]

        # === PILLAR 3: EVIDENCE COLLECTION ===
        trend_strength_frac = (ema_short - ema_long) / ema_long if ema_long else 0.0
        price_participation_frac = (current_price - ema_long) / ema_long if ema_long else 0.0
        confirmation_ratio = (
            confirmation_count / entry_confirmation_loops if entry_confirmation_loops > 0 else 0.0
        )
        trend_atr_units = (ema_short - ema_long) / atr if atr > 0 else 0.0

        evidence_items = [
            EvidenceItem(
                name="Trend strength",
                measurement=lambda d: trend_strength_frac,
                normalization=lambda r: max(-1.0, min(1.0, r / 0.02)),
                weight=35.0,
                reason=(
                    "Theory (time-series momentum): the core signal is the fast "
                    "EMA leading the slow EMA. A larger separation (as a fraction "
                    "of the slow EMA) is a more established trend, which the "
                    "momentum anomaly says is more likely to persist; a negative "
                    "separation is counter-trend and correctly scores against entry."
                ),
            ),
            EvidenceItem(
                name="Price participation",
                measurement=lambda d: price_participation_frac,
                normalization=lambda r: max(-1.0, min(1.0, r / 0.02)),
                weight=25.0,
                reason=(
                    "Theory: price itself must participate in the trend, not just "
                    "the smoothed EMAs. Price above the slow EMA confirms the "
                    "up-trend is current; price below it (even with EMAs still "
                    "ordered up) is early evidence the move is fading."
                ),
            ),
            EvidenceItem(
                name="Confirmation persistence",
                measurement=lambda d: confirmation_ratio,
                normalization=lambda r: max(-1.0, min(1.0, r)),
                weight=20.0,
                reason=(
                    "Theory: a momentum signal that persists across consecutive "
                    "bars is less likely to be noise than a single-bar flicker. "
                    "This is the noise defense the pre-migration confirmation-loop "
                    "gate provided, now expressed as measurable evidence rather "
                    "than a hard boolean count."
                ),
            ),
            EvidenceItem(
                name="Volatility-normalized trend",
                measurement=lambda d: trend_atr_units,
                normalization=lambda r: max(-1.0, min(1.0, r / 1.0)),
                weight=20.0,
                reason=(
                    "Theory: the EMA separation must be meaningful relative to "
                    "current noise, not just relative to price level. Measuring "
                    "separation in ATR units confirms the trend stands out above "
                    "the bar-to-bar volatility, independent of the percentage "
                    "measure above."
                ),
            ),
        ]

        # === PILLAR 3: EVIDENCE-BASED DECISION SCORE ===
        decision_score = DecisionScoreEngine().score(
            "trend_following", evidence_items, {}, threshold=decision_score_threshold,
        )
        edge_manager.record_decision_score(bot.id, "trend_following", decision_score.total)
        reasons_for, reasons_against = derive_reasons(decision_score)
        assumptions = (
            "trend direction unchanged: fast EMA remains above slow EMA "
            "(EMA ordering not reversed)",
            "price remains above the slow EMA (participation in the trend holds)",
            "current market regime remains within the strategy's allowed set",
        )

        # --- Structured decision explanation (observe-only) ---
        exp.update({
            "current_price": current_price,
            "ema_fast": ema_short,
            "ema_slow": ema_long,
            "ema_distance": ema_short - ema_long,
            "ema_distance_pct": trend_strength_frac * 100,
            "price_vs_slow_ema_pct": price_participation_frac * 100,
            "atr": atr,
            "atr_percentile": atr_percentile,
            "effective_stop_multiplier": atr_stop_mult,
            "entry_confirmation_count": confirmation_count,
            "entry_threshold_loops": entry_confirmation_loops,
            "cooldown_seconds": cooldown_seconds,
            "cooldown_remaining_s": cooldown_remaining,
            "regime_tags": suitability.regime_tags,
            "market_suitable": suitability.is_suitable,
            "has_position": False,
        })
        exp.metric("edge_status_category", edge_status.category.value)
        exp.metric("decision_score_total", decision_score.total)
        exp.metric("decision_score_threshold", decision_score_threshold)
        exp.check(
            "Cooldown", f"{cooldown_remaining} sec remaining",
            f">= {cooldown_seconds} sec elapsed", cooldown_remaining == 0,
        )
        exp.check(
            "Market suitability", suitability.is_suitable, "must be True",
            suitability.is_suitable, detail=suitability.reason,
        )
        exp.check(
            "Decision Score clears threshold", decision_score.total,
            f">= {decision_score_threshold:.1f}", decision_score.approved,
        )
        exp.check(
            "Strategy edge not disqualified", edge_status.category.value,
            f"!= {EdgeCategory.C.value}", edge_status.category != EdgeCategory.C,
            detail=edge_status.reason,
        )

        # === PILLAR 2/3/7 GATES: any blocking reason -> NO_TRADE ===
        blocking_reasons = []
        if not suitability.is_suitable:
            blocking_reasons.append(f"regime unsuitable ({suitability.reason})")
        if not decision_score.approved:
            blocking_reasons.append(
                f"Decision Score {decision_score.total:.1f} < threshold {decision_score_threshold:.1f}"
            )
        if edge_status.category == EdgeCategory.C:
            blocking_reasons.append(f"Strategy Edge Management: Category C - {edge_status.reason}")
        if cooldown_remaining > 0:
            blocking_reasons.append(f"re-entry cooldown active ({cooldown_remaining}s remaining)")

        if blocking_reasons:
            if base_trend_ok and decision_score.approved and suitability.is_suitable:
                exp.state("WAITING_COOLDOWN")
            elif base_trend_ok:
                exp.state("WAITING_CONFIRMATION").next_trade(
                    current=confirmation_count, current_label="Confirmations",
                    target=entry_confirmation_loops, target_label="Required",
                    distance=max(0, entry_confirmation_loops - confirmation_count),
                    status=f"score {decision_score.total:.1f}/{decision_score_threshold:.1f}",
                )
            else:
                exp.state("WAITING_TREND").next_trade(
                    current=ema_short, current_label="EMA Fast", target=ema_long,
                    target_label="EMA Slow", distance=ema_short - ema_long,
                    status="waiting for an up-trend (fast EMA above slow EMA, price above slow EMA)",
                )
            self._trend_states[bot.id] = state
            generated_at = self.clock.now()
            proposal = StrategyProposal(
                strategy_id="trend_following", bot_id=bot.id, generated_at=generated_at,
                direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                validity=ProposalValidity(
                    generated_at=generated_at,
                    valid_until=generated_at + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=decision_score, market_suitability=suitability, edge_status=edge_status,
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_risk_budget_pct=risk_percent,
                adaptive_parameters_used={"atr_multiplier": atr_stop_mult},
                explanation=exp.to_dict(),
            )
            adapter_signal = StandaloneAdapter.to_trade_signal(proposal)
            if adapter_signal is not None:
                return adapter_signal
            reason_text = "; ".join(blocking_reasons)
            return TradeSignal(
                action="hold", amount=0,
                reason=f"Trend Following: {reason_text} (score {decision_score.total:.1f}/{decision_score_threshold:.1f})",
            )

        # === PILLAR 5: DECISION-SCORE-WEIGHTED POSITION SIZING ===
        # A marginal-score trade (just above threshold) sizes toward 0.5x; a
        # maximal-score trade toward 1.5x. Deterministic and reproducible -
        # not a separate, unaccountable scaling knob. (add-unified-position-
        # sizing will consolidate this into a shared cross-strategy function;
        # this is this strategy's certification-phase implementation until then.)
        score_range = max(100.0 - decision_score_threshold, 1e-9)
        score_margin = max(0.0, min(1.0, (decision_score.total - decision_score_threshold) / score_range))
        size_multiplier = 0.5 + score_margin
        exp.metric("decision_score_size_multiplier", size_multiplier)

        risk_amount = bot.current_balance * risk_percent * size_multiplier
        if atr > 0:
            stop_distance = atr * atr_stop_mult             # USD per coin (LOCKED-at-entry multiplier)
            position_coins = risk_amount / stop_distance    # base coins
            position_size = position_coins * current_price  # quote USD notional
        else:
            position_size = risk_amount

        buy_amount = min(position_size, bot.current_balance * _BUY_BALANCE_FRACTION)

        # Pillar 8: sizing is a decision point - surface it as a check.
        if buy_amount < MIN_ORDER_USD:
            if bot.current_balance >= MIN_ORDER_USD:
                buy_amount = MIN_ORDER_USD
            else:
                exp.check(
                    "Position size >= min order", buy_amount, f">= {MIN_ORDER_USD:.0f}", False,
                )
                self._trend_states[bot.id] = state
                return TradeSignal(
                    action="hold", amount=0,
                    reason=(
                        f"Trend Following: balance ${bot.current_balance:.2f} "
                        f"below ${MIN_ORDER_USD:.0f} minimum order"
                    ),
                )
        exp.check("Position size >= min order", buy_amount, f">= {MIN_ORDER_USD:.0f}", True)

        # Pillar 8: fee-viability gate as an explicit check. Expected move is
        # the locked stop distance as a fraction of price (the pre-migration
        # expected_move_pct convention); by construction of the ATR fee-floor
        # this clears the fee hurdle, but it is now surfaced, not implicit.
        _tf_expected_move = (atr * atr_stop_mult) / current_price if current_price > 0 else 0.0
        _fee_raw_tf2 = getattr(bot, 'exchange_fee', 0.1)
        _tf_fee_pct = (
            float(_fee_raw_tf2) if isinstance(_fee_raw_tf2, (int, float)) else 0.1
        ) / 100.0
        _tf_min_move = 2.0 * _tf_fee_pct + _VIABILITY_SAFETY_MARGIN_PCT
        exp.check(
            "Fee viability", _tf_expected_move, f">= {_tf_min_move:.5f}",
            _tf_expected_move >= _tf_min_move,
        )
        if _tf_expected_move < _tf_min_move:
            self._trend_states[bot.id] = state
            return TradeSignal(
                action="hold", amount=0,
                reason=(
                    f"Trend Following: stop distance {_tf_expected_move * 100:.3f}% < "
                    f"fee threshold {_tf_min_move * 100:.3f}% (exchange_fee={_tf_fee_pct * 100:.2f}%)"
                ),
            )

        # Pillar 8: initial stop placement is a decision point - surface it.
        entry_stop_distance = atr * atr_stop_mult
        trailing_stop_price = current_price - entry_stop_distance
        exp.check(
            "Initial stop below entry", trailing_stop_price, f"< {current_price:.2f}",
            trailing_stop_price < current_price,
        )
        exp.state("ENTRY_ARMED")

        # === PILLAR 10: STRATEGYPROPOSAL ===
        generated_at = self.clock.now()
        proposal = StrategyProposal(
            strategy_id="trend_following", bot_id=bot.id, generated_at=generated_at,
            direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
            validity=ProposalValidity(
                generated_at=generated_at,
                valid_until=generated_at + timedelta(seconds=_validity_interval_seconds),
            ),
            decision_score=decision_score, market_suitability=suitability, edge_status=edge_status,
            assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
            suggested_position_size=buy_amount, suggested_risk_budget_pct=risk_percent * size_multiplier,
            expected_holding_horizon="long",
            adaptive_parameters_used={
                "atr_multiplier": atr_stop_mult,
                "decision_score_size_multiplier": size_multiplier,
            },
            explanation=exp.to_dict(),
        )

        logger.info(
            f"Bot {bot.id}: Trend Following ENTRY - "
            f"Decision Score {decision_score.total:.1f}/{decision_score_threshold:.1f}, "
            f"Price ${current_price:.2f} > EMA({long_period}) ${ema_long:.2f}, "
            f"EMA({short_period}) ${ema_short:.2f} > EMA({long_period}), "
            f"Entry ATR locked ${atr:.4f} x {atr_stop_mult:.2f}, Position: ${buy_amount:.2f}"
        )

        # Lock entry state (risk never expands mid-trade) - including the
        # LIVE-resolved adaptive stop multiplier. entry_time stays a datetime
        # (JSON-roundtrip-safe via _to_jsonable, and the state-persistence
        # regression suite asserts datetime fidelity for this key).
        state["trailing_stop"] = trailing_stop_price
        state["highest_price"] = current_price
        state["entry_price"] = current_price
        state["entry_atr"] = atr
        state["entry_stop_multiplier"] = atr_stop_mult
        state["entry_time"] = self.clock.now()
        state["last_exit_time"] = None
        state["entry_confirmation_count"] = 0
        state["exit_confirmation_count"] = 0
        self._trend_states[bot.id] = state

        return StandaloneAdapter.to_trade_signal(
            proposal, expected_move_pct=_tf_expected_move,
        )

    async def _strategy_volatility_breakout(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Volatility Breakout (volatility expansion) strategy.

        Migrated to the Strategy Decision Framework
        (add-strategy-decision-framework, Phase 1 - see this change's
        design.md/tasks.md, and audits/volatility_breakout.md for the full
        Strategy Audit Document). Complete decision flow:

            Market Data
                -> Market Suitability Gate (shared MarketSuitabilityGate -
                   a HARD gate; refuses new entries in an unsuitable
                   regime, closing the audit's "computed but never
                   enforced" Pillar 2 finding)
                -> Adaptive Parameter Resolver (shared
                   AdaptiveParameterResolver - the ATR stop multiplier is
                   resolved from the current ATR percentile every cycle,
                   then LOCKED at entry; Pillar 4)
                -> Evidence Collection (breakout magnitude, compression
                   maturity, compression tightness, volatility expansion
                   strength - all deterministic, all traced to the Theory
                   section of the Strategy Audit Document; Pillar 3)
                -> Evidence-Based Decision Score (shared DecisionScoreEngine)
                -> Strategy Edge Management (shared StrategyEdgeManager -
                   refuses new entries once classified Category C; never
                   force-closes an existing position; Pillar 7)
                -> StrategyProposal (shared, immutable; Pillar 10)
                -> Standalone Adapter (shared StandaloneAdapter - the ONLY
                   place a StrategyProposal is translated into the
                   TradeSignal this function returns)
                -> existing execution pipeline (UNCHANGED, downstream of
                   this function's return value)

        THEORY (Pillar 1 - see the Strategy Audit Document for the full
        write-up): exploits a liquidity-vacuum / stop-run inefficiency.
        Price coiled inside a multi-bar compression (a thin order book and
        suppressed realized volatility) tends to move fast and far once it
        breaks the range, because the same tight range that compressed
        volatility also concentrated resting stop and breakout orders just
        beyond it - the initial move is often mechanically self-reinforcing
        (stops trigger further stops), not merely a random continuation.
        RARE, CONVEX, REGIME-AWARE, long-only, upper-band breakouts only.

        CRITICAL: All logic operates on AGGREGATED PSEUDO-BARS, not tick
        data. Bar interval defines time granularity (default: 60 seconds).

        Parameters:
            bar_interval_seconds: Time per bar for aggregation (default: 60)
            bb_period: Bollinger Band period in bars (default: 20)
            bb_std: Bollinger Band standard deviation (default: 2.0)
            atr_period: ATR period in bars (default: 14)
            compression_method: "bb_width" or "atr_average" (default: "bb_width")
            compression_percentile: BB width percentile threshold % (default: 20)
            atr_threshold_multiplier: ATR threshold vs average, used only by
                the "atr_average" compression_method (default: 0.8)
            min_compression_bars: Minimum bars of compression before arming
                (default: 5) - FIXED (Pillar 4): a debounce count against
                noise, not a quantity that should scale with volatility.
            atr_stop_multiplier: BASE ATR stop-loss multiplier (default:
                2.0) - ADAPTIVE (Pillar 4): resolved every cycle via
                AdaptiveParameterResolver.atr_percentile_scaled_multiplier
                against current-ATR-vs-its-own-20-bar-average, then LOCKED
                at entry (risk never expands mid-trade).
            min_atr_stop_multiplier / max_atr_stop_multiplier: bounds for
                the adaptive resolution above (default: 1.0 / 4.0).
            take_profit_rr_multiple: reward target as a multiple of the
                locked stop distance (default: 3.0) - FIXED (Pillar 4): a
                convexity target set by this strategy's own "rare, convex"
                thesis (a few winners must fund many small, bounded
                losses), not a quantity that should scale with volatility.
            max_holding_bars: force a full exit if a breakout has hit
                neither its stop nor its target within this many bars
                (default: 48) - FIXED (Pillar 4): a debounce/timeout count,
                not volatility-scaled. Closes the audit's Pillar 6 gap
                ("no take-profit, no time-based exit").
            risk_percent: Percent of capital to risk (default: 1.0)
            cooldown_hours: Hours between breakout attempts (default: 24)
            failed_breakout_bars: Bars to detect failed breakout (default: 3)
            allowed_regimes: Volatility regimes this strategy is suitable
                in (default: ["volatility_expanding"]) - now a HARD gate
                (Pillar 2) via the shared MarketSuitabilityGate. DIRECTION
                tags, derived from the bar-based regime detector's rate-
                of-change classification, the same convention
                _strategy_auto/_is_strategy_eligible use. `regime_filter_
                enabled` (a disable switch for this gate) has been REMOVED
                - a strategy in this framework cannot opt out of Pillar 2.
            decision_score_threshold: minimum Evidence-Based Decision Score
                (0-100) required to enter (default: 40.0) - an initial,
                certification-pending value; see the Strategy Audit
                Document's before/after backtest comparison.
        """
        # Get parameters with SPARSE defaults
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        bb_period = params.get("bb_period", 20)
        bb_std = params.get("bb_std", 2.0)
        atr_period = params.get("atr_period", 14)
        compression_method = params.get("compression_method", "bb_width")
        compression_percentile = params.get("compression_percentile", 20)
        atr_threshold_mult = params.get("atr_threshold_multiplier", 0.8)
        min_compression_bars = params.get("min_compression_bars", 5)
        atr_stop_mult_base = params.get("atr_stop_multiplier", 2.0)
        min_atr_stop_mult = params.get("min_atr_stop_multiplier", 1.0)
        max_atr_stop_mult = params.get("max_atr_stop_multiplier", 4.0)
        take_profit_rr_multiple = params.get("take_profit_rr_multiple", 3.0)
        max_holding_bars = params.get("max_holding_bars", 48)
        risk_percent = params.get("risk_percent", 1.0) / 100
        cooldown_hours = params.get("cooldown_hours", 24)
        failed_breakout_bars = params.get("failed_breakout_bars", 3)
        allowed_regimes = params.get("allowed_regimes", ["volatility_expanding"])
        decision_score_threshold = params.get("decision_score_threshold", 40.0)
        # ProposalValidity requires a strictly positive window; some test
        # harnesses set bar_interval_seconds=0 ("close a bar every call") -
        # a real proposal must still carry a non-zero validity window.
        _validity_interval_seconds = max(bar_interval_seconds, 1)

        # === BAR AGGREGATION SYSTEM ===
        # Aggregate tick prices into fixed-time bars (OHLC)
        # This converts 1-second ticks into meaningful time-based bars

        # Initialize state tracking (INSTITUTIONAL STRUCTURE)
        if not hasattr(self, "_volatility_breakout_states"):
            self._volatility_breakout_states = {}
        if not hasattr(self, "_volatility_breakout_edge_manager"):
            # Pillar 7: one shared, continuous tracker for this strategy,
            # keyed internally by (bot_id, strategy) - see edge_management.py.
            self._volatility_breakout_edge_manager = StrategyEdgeManager()

        state = self._volatility_breakout_states.get(bot.id, {
            "bars": [],  # List of {"open", "high", "low", "close", "start_ts"}
            "current_bar": None,  # Bar being built
            "bb_width_history": [],
            "atr_history": [],
            "compression_active": False,
            "compression_bars": 0,
            "compression_start": None,
            "compression_min_width": None,  # running min bb_width this compression episode
            "breakout_armed": False,  # latched after compression, survives breakout bar
            "armed_compression_bars": None,  # bars compressed, captured at the moment of arming
            "armed_compression_min_width": None,  # tightest bb_width seen, captured at arming
            "entry_price": None,
            "entry_atr": None,  # LOCKED at entry - risk never expands
            "entry_stop_multiplier": None,  # LOCKED adaptive multiplier at entry
            "entry_time": None,  # LOCKED entry timestamp (isoformat) - holding time / time-stop
            "highest_price": None,  # For monotonic trailing stop
            "trailing_stop": None,
            "take_profit_price": None,  # LOCKED at entry - Pillar 6 gap closure
            "bars_since_entry": 0,
            "last_breakout_attempt": None,
            "regime_state": None,  # persisted bar-based regime (see MARKET SUITABILITY below)
        })

        now = self.clock.now()

        # Initialize current bar if needed
        if state["current_bar"] is None:
            state["current_bar"] = {
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "start_ts": now,
            }

        # Update current bar with new tick
        current_bar = state["current_bar"]
        current_bar["high"] = max(current_bar["high"], current_price)
        current_bar["low"] = min(current_bar["low"], current_price)
        current_bar["close"] = current_price

        # Check if bar is complete (time-based)
        bar_duration = (now - current_bar["start_ts"]).total_seconds()
        bar_completed = bar_duration >= bar_interval_seconds

        if bar_completed:
            # Close current bar and add to history
            state["bars"].append(current_bar)
            # Keep sufficient bar history: max(bb_period + 100, 150) bars
            max_bars = max(bb_period + 100, 150)
            state["bars"] = state["bars"][-max_bars:]

            # Start new bar
            state["current_bar"] = {
                "open": current_price,
                "high": current_price,
                "low": current_price,
                "close": current_price,
                "start_ts": now,
            }

            logger.debug(
                f"Bot {bot.id}: Volatility Breakout - Bar completed: "
                f"O:{current_bar['open']:.2f} H:{current_bar['high']:.2f} "
                f"L:{current_bar['low']:.2f} C:{current_bar['close']:.2f}"
            )

        # Need enough bars for calculations
        if len(state["bars"]) < bb_period:
            self._volatility_breakout_states[bot.id] = state
            self._explain(bot.id).state("WARMING_UP").metric(
                "current_price", current_price
            ).check(
                "Bars collected", len(state["bars"]), f">= {bb_period}", False,
                detail="warming up Bollinger window",
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Collecting bars ({len(state['bars'])}/{bb_period})"
            )

        # === BAR-BASED INDICATOR CALCULATIONS ===
        # All indicators operate on bar close prices, not ticks

        def calculate_bollinger_bands_from_bars(bars: list, period: int, std_mult: float):
            """Calculate Bollinger Bands from bar closes."""
            closes = [bar["close"] for bar in bars[-period:]]
            sma = sum(closes) / len(closes)

            # Calculate standard deviation
            variance = sum((c - sma) ** 2 for c in closes) / len(closes)
            std_dev = variance ** 0.5

            upper_band = sma + (std_mult * std_dev)
            lower_band = sma - (std_mult * std_dev)
            bandwidth = (upper_band - lower_band) / sma if sma > 0 else 0

            return sma, upper_band, lower_band, bandwidth

        def calculate_atr_from_bars(bars: list, period: int) -> float:
            """Calculate ATR proxy from bar data.

            Uses bar high-low range as True Range approximation since we're
            aggregating ticks. This is a proxy, not true OHLC ATR.
            """
            if len(bars) < period:
                return 0.0

            true_ranges = []
            for i in range(len(bars) - period, len(bars)):
                if i < 0:
                    continue
                # TR approximation: high - low of each bar
                tr = bars[i]["high"] - bars[i]["low"]
                true_ranges.append(tr)

            if not true_ranges:
                return 0.0

            return sum(true_ranges) / len(true_ranges)

        # Calculate indicators from completed bars
        sma, upper_band, lower_band, bb_width = calculate_bollinger_bands_from_bars(
            state["bars"], bb_period, bb_std
        )
        atr = calculate_atr_from_bars(state["bars"], atr_period)

        # Track historical Bollinger width and ATR (only when bar completes)
        if bar_completed:
            state["bb_width_history"].append(bb_width)
            state["bb_width_history"] = state["bb_width_history"][-100:]  # Keep last 100

            state["atr_history"].append(atr)
            state["atr_history"] = state["atr_history"][-100:]  # Keep last 100

        # === PILLAR 2: MARKET SUITABILITY GATE (shared, HARD-enforced) ===
        # Uses the bar-based regime detector (the same one _strategy_auto /
        # _is_strategy_eligible use) so this gate tests real volatility
        # DIRECTION (rate of change), not a relabeled LEVEL - see
        # fix-regime-detection-consistency. state["bars"] already has
        # >= bb_period (>= 20 by the early-return above) real OHLC bars, so
        # it can be passed directly - no synthetic close-only point needs
        # to be appended. Unlike the pre-migration implementation, `suit-
        # ability.is_suitable` is now an ACTUAL hard gate on new entries
        # below (closes the audit's Pillar 2 "computed but never enforced"
        # finding) - there is no more disable switch for this check.
        current_regime = self._detect_market_regime_bar_based(
            state["bars"], state.get("regime_state")
        )
        state["regime_state"] = current_regime
        suitability = MarketSuitabilityGate().evaluate(current_regime, allowed_regimes)

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # Get last completed bar close for logic (current bar is incomplete)
        last_bar_close = state["bars"][-1]["close"] if state["bars"] else current_price

        # === PILLAR 4: ADAPTIVE PARAMETER RESOLVER ===
        # Stop multiplier scales with current ATR relative to its own
        # 20-bar average - wider when volatility is currently elevated
        # (needs more room to avoid a premature stop-out), tighter when
        # volatility is calm relative to its own recent history. Resolved
        # every cycle; LOCKED at entry (below) so an open position's risk
        # never expands mid-trade, matching the pre-migration "risk never
        # expands" guarantee.
        recent_atr_window = state["atr_history"][-20:]
        avg_atr = (sum(recent_atr_window) / len(recent_atr_window)) if recent_atr_window else atr
        atr_percentile = (atr / avg_atr) if avg_atr > 0 else 1.0
        resolved_stop = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
            name="atr_stop_multiplier",
            atr_percentile=atr_percentile,
            base_multiplier=atr_stop_mult_base,
            min_multiplier=min_atr_stop_mult,
            max_multiplier=max_atr_stop_mult,
        )
        atr_stop_mult = resolved_stop.value

        # === PILLAR 7: STRATEGY EDGE MANAGEMENT ===
        # regime_outside_suitable_range comes from the Pillar 2 gate above.
        # parameter_mismatch_evidence is cited only when ATR has drifted
        # far from the window the adaptive resolver's BASE multiplier
        # implicitly assumes "normal" - a citation for a future Category B
        # response (adapting atr_stop_multiplier's base via certification),
        # never a free-text guess.
        edge_manager = self._volatility_breakout_edge_manager
        parameter_mismatch_evidence = None
        if atr_percentile > 1.5 or atr_percentile < 0.6:
            parameter_mismatch_evidence = (
                f"ATR percentile {atr_percentile:.2f}x its 20-bar average - "
                f"stop multiplier base {atr_stop_mult_base} may be "
                "miscalibrated for sustained current volatility (the live "
                "value is already adaptively resolved above; sustained "
                "drift outside [0.6, 1.5] is evidence the BASE itself may "
                "need certification-phase recalibration)"
            )
        edge_status = edge_manager.evaluate(
            bot.id, "volatility_breakout",
            regime_outside_suitable_range=not suitability.is_suitable,
            parameter_mismatch_evidence=parameter_mismatch_evidence,
            now=self.clock.now(),
        )

        logger.debug(
            f"Bot {bot.id}: Volatility Breakout - Bar close: ${last_bar_close:.2f}, "
            f"BB: [${lower_band:.2f}, ${sma:.2f}, ${upper_band:.2f}], "
            f"Width: {bb_width:.4f}, ATR: ${atr:.2f}, "
            f"Suitable: {suitability.is_suitable} ({suitability.reason}), "
            f"Edge: {edge_status.category.value}"
        )

        def _single_evidence_proposal(
            *, direction: "Direction", execution_intent: "ExecutionIntent",
            evidence_name: str, evidence_value: float, evidence_reason: str,
            threshold: float, suggested_position_size: Optional[float] = None,
            assumptions: tuple = (),
        ) -> StrategyProposal:
            """Shared helper for every branch that isn't the full multi-
            factor entry evaluation below: exits, holds, and suitability/
            cooldown/edge-blocked no-trades. Each is a single, deterministic,
            already-decided condition (e.g. "stop was hit: yes/no"), not a
            discretionary multi-factor judgement - a single Evidence Item
            correctly and honestly represents that, per Pillar 3's
            determinism/reproducibility requirement (nothing here is
            subjective, it just isn't MULTI-factor)."""
            item = EvidenceItem(
                name=evidence_name,
                measurement=lambda d: evidence_value,
                normalization=lambda r: r,
                weight=100.0,
                reason=evidence_reason,
            )
            score = DecisionScoreEngine().score(
                "volatility_breakout", [item], {}, threshold=threshold,
            )
            reasons_for, reasons_against = derive_reasons(score)
            generated_at = self.clock.now()
            return StrategyProposal(
                strategy_id="volatility_breakout",
                bot_id=bot.id,
                generated_at=generated_at,
                direction=direction,
                execution_intent=execution_intent,
                validity=ProposalValidity(
                    generated_at=generated_at,
                    valid_until=generated_at + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=score,
                market_suitability=suitability,
                edge_status=edge_status,
                assumptions=assumptions,
                reasons_for=reasons_for,
                reasons_against=reasons_against,
                suggested_position_size=suggested_position_size,
                suggested_risk_budget_pct=risk_percent,
                explanation=self._explain(bot.id).to_dict(),
            )

        # === PILLAR 6: CONTINUOUS TRADE MANAGEMENT (position open) ===
        if has_position:
            _exp = self._explain(bot.id)
            _bse = state.get("bars_since_entry", 0)
            _ts = state.get("trailing_stop")
            take_profit_price = state.get("take_profit_price")
            # CRITICAL: use LOCKED entry values (risk never expands mid-trade).
            # Fallback covers legacy positions from before this migration.
            entry_atr_locked = state.get("entry_atr") or atr
            entry_stop_mult_locked = state.get("entry_stop_multiplier") or atr_stop_mult
            entry_time_iso = state.get("entry_time")
            holding_seconds = (
                (self.clock.now() - datetime.fromisoformat(entry_time_iso)).total_seconds()
                if entry_time_iso else 0.0
            )
            initial_risk_per_unit = entry_atr_locked * entry_stop_mult_locked

            _exp.update({
                "current_price": current_price,
                "bar_close": last_bar_close,
                "bb_upper": upper_band,
                "bb_middle": sma,
                "bb_lower": lower_band,
                "bb_width": bb_width,
                "atr": atr,
                "atr_percentile": atr_percentile,
                "trailing_stop": _ts,
                "take_profit_price": take_profit_price,
                "bars_since_entry": _bse,
                "holding_seconds": holding_seconds,
                "max_holding_bars": max_holding_bars,
                "highest_price": state.get("highest_price"),
                "failed_breakout_bars": failed_breakout_bars,
                "regime_tags": suitability.regime_tags,
                "market_suitable": suitability.is_suitable,
            })
            # Well-known metric key (explanation_persistence.py) so Order
            # persistence (Phase 0.6) surfaces the active edge category.
            _exp.metric("edge_status_category", edge_status.category.value)

            # Shared post-entry re-evaluation hook: (a) thesis/suitability
            # still passing is delegated to MarketSuitabilityGate internally;
            # (b)/(c)/(d) are this strategy's own pure predicates.
            trade_monitor = TradeManagementMonitor()
            tm_report = trade_monitor.evaluate(
                current_regime=current_regime,
                allowed_regimes=allowed_regimes,
                stop_tighten_check=lambda: (
                    atr_percentile < 0.7,
                    f"ATR percentile {atr_percentile:.2f} - volatility has "
                    "contracted since entry (informational: the locked "
                    "entry stop distance does not itself change, per "
                    "'risk never expands' - reflected in future entries' "
                    "adaptive resolution, not this open position's stop)",
                ),
                partial_profit_check=lambda: (
                    bool(take_profit_price is not None and current_price >= take_profit_price),
                    (f"price {current_price:.2f} reached the "
                     f"{take_profit_rr_multiple:.1f}x reward:risk target "
                     f"{take_profit_price:.2f}") if take_profit_price is not None else "",
                ),
            )
            _exp.metric("thesis_intact", tm_report.thesis_intact)

            failed_breakout_hit = _bse <= failed_breakout_bars and last_bar_close < upper_band
            take_profit_hit = bool(take_profit_price is not None and current_price >= take_profit_price)
            trailing_stop_hit = bool(_ts is not None and current_price <= _ts)
            time_stop_hit = _bse >= max_holding_bars

            _exp.check(
                "Failed-breakout exit", last_bar_close,
                f"< {upper_band:.2f} within {failed_breakout_bars} bars",
                failed_breakout_hit, detail=f"bar {_bse}/{failed_breakout_bars}",
            )
            _exp.check(
                "Take-profit hit", current_price,
                f">= {take_profit_price:.2f}" if take_profit_price is not None else "n/a",
                take_profit_hit,
            )
            _exp.check(
                "Trailing stop hit", current_price,
                f"<= {_ts:.2f}" if _ts is not None else "n/a",
                trailing_stop_hit,
            )
            _exp.check(
                "Time-stop", _bse, f"< {max_holding_bars} bars", not time_stop_hit,
            )
            _exp.state("LONG_OPEN").next_trade(
                current=current_price, current_label="Current price",
                target=(_ts if _ts is not None else upper_band),
                target_label=("Trailing stop" if _ts is not None else "BB upper"),
                distance=(current_price - _ts) if _ts is not None else (current_price - upper_band),
                status="holding — exits on stop, target, failed breakout, or time-stop",
            )

            exit_trigger = None
            if failed_breakout_hit:
                exit_trigger = (
                    "failed_breakout",
                    f"Bar close ${last_bar_close:.2f} < BB upper ${upper_band:.2f} "
                    f"within {failed_breakout_bars} bars of entry — the breakout did not hold",
                )
            elif take_profit_hit:
                exit_trigger = (
                    "take_profit",
                    f"Price ${current_price:.2f} reached the "
                    f"{take_profit_rr_multiple:.1f}x reward:risk target ${take_profit_price:.2f}",
                )
            elif trailing_stop_hit:
                exit_trigger = (
                    "trailing_stop",
                    f"Price ${current_price:.2f} <= locked trailing stop ${_ts:.2f}",
                )
            elif time_stop_hit:
                exit_trigger = (
                    "time_stop",
                    f"Held {_bse} bars without reaching stop or target (max "
                    f"{max_holding_bars}) — breakout failed to convert into a fast move",
                )

            for pos in positions:
                if bar_completed and state["entry_price"] is not None:
                    state["bars_since_entry"] += 1

                if exit_trigger is not None:
                    evidence_name, reason_text = exit_trigger
                    # Pillar 7: record this outcome BEFORE state is cleared,
                    # so the next evaluation's edge classification reflects
                    # it. Only when entry_price is actually known - a
                    # position with no locally-tracked entry (e.g. imported
                    # from the exchange, or opened before this migration)
                    # has no real pnl to report; recording a fabricated
                    # value would corrupt StrategyEdgeManager's statistics.
                    entry_price_known = state.get("entry_price")
                    if entry_price_known is not None:
                        pnl_per_unit = current_price - entry_price_known
                        reward_risk_realized = (
                            (pnl_per_unit / initial_risk_per_unit) if initial_risk_per_unit > 0 else None
                        )
                        edge_manager.record_trade_outcome(
                            bot.id, "volatility_breakout",
                            pnl=pnl_per_unit, win=(pnl_per_unit > 0),
                            reward_risk_realized=reward_risk_realized,
                            holding_seconds=holding_seconds, at=self.clock.now(),
                        )

                    proposal = _single_evidence_proposal(
                        direction=Direction.SELL,
                        execution_intent=ExecutionIntent.CLOSE_POSITION,
                        evidence_name=evidence_name,
                        evidence_value=1.0,
                        evidence_reason=reason_text,
                        threshold=1.0,
                        suggested_position_size=pos.amount * current_price,
                    )

                    logger.info(f"Bot {bot.id}: Volatility Breakout EXIT ({evidence_name}) - {reason_text}")

                    # Clear state
                    state["trailing_stop"] = None
                    state["highest_price"] = None
                    state["entry_price"] = None
                    state["entry_atr"] = None
                    state["entry_stop_multiplier"] = None
                    state["entry_time"] = None
                    state["take_profit_price"] = None
                    state["bars_since_entry"] = 0
                    state["compression_active"] = False
                    state["compression_bars"] = 0
                    self._volatility_breakout_states[bot.id] = state

                    return StandaloneAdapter.to_trade_signal(proposal)

                # No exit this tick — maintain the monotonic trailing stop
                # using the LOCKED entry ATR/multiplier (risk never expands).
                if state["highest_price"] is None:
                    state["highest_price"] = current_price
                if current_price > state["highest_price"]:
                    state["highest_price"] = current_price
                    state["trailing_stop"] = (
                        state["highest_price"] - (entry_atr_locked * entry_stop_mult_locked)
                    )
                    logger.debug(
                        f"Bot {bot.id}: Volatility Breakout trailing stop updated - "
                        f"New high ${state['highest_price']:.2f}, Stop ${state['trailing_stop']:.2f}"
                    )
                if state["trailing_stop"] is None:
                    state["trailing_stop"] = current_price - (entry_atr_locked * entry_stop_mult_locked)

            # Update state and hold.
            self._volatility_breakout_states[bot.id] = state

            hold_proposal = _single_evidence_proposal(
                direction=Direction.HOLD,
                execution_intent=ExecutionIntent.HOLD_POSITION,
                evidence_name="Position intact",
                evidence_value=1.0,
                evidence_reason=(
                    "no exit condition (failed breakout, take-profit, trailing "
                    "stop, time-stop) triggered this cycle"
                ),
                threshold=1.0,
                assumptions=(
                    "breakout level remains valid until price closes back "
                    "inside the range or the locked trailing stop is hit",
                ),
            )
            # Route through the Standalone Adapter like every other branch
            # ("must not bypass any shared component") - HOLD_POSITION
            # always translates to None (no order), but the ACTION decision
            # still comes from the adapter, never a shortcut around it. The
            # richer, human-readable reason below is purely a UI/diagnostics
            # enrichment layered on top, mechanically derived from the same
            # proposal, never a substitute for the adapter's decision.
            adapter_signal = StandaloneAdapter.to_trade_signal(hold_proposal)
            if adapter_signal is not None:
                return adapter_signal
            reason_text = "; ".join(hold_proposal.reasons_for) or "Volatility Breakout: holding position"
            stop_str = f"${state['trailing_stop']:.2f}" if state["trailing_stop"] is not None else "N/A"
            return TradeSignal(
                action="hold", amount=0,
                reason=f"Volatility Breakout: Holding position, stop at {stop_str} ({reason_text})",
            )

        # === COMPRESSION DETECTION (BAR-BASED) - ALWAYS TRACKED ===
        # Must run regardless of suitability/cooldown, or compression_bars
        # can never reach min_compression_bars.
        is_compressed = False
        percentile_value = None

        if compression_method == "bb_width":
            # Use Bollinger Band width percentile
            if len(state["bb_width_history"]) >= 20:
                sorted_widths = sorted(state["bb_width_history"])
                percentile_index = int(len(sorted_widths) * (compression_percentile / 100))
                percentile_value = sorted_widths[percentile_index]
                is_compressed = bb_width <= percentile_value
        elif compression_method == "atr_average":
            # Use ATR below its rolling average
            if len(state["atr_history"]) >= 20:
                atr_avg_20 = sum(state["atr_history"][-20:]) / 20
                is_compressed = atr <= (atr_avg_20 * atr_threshold_mult)

        # Track compression duration (BAR-BASED), and the tightest bb_width
        # seen this episode (Evidence: "Compression tightness" below).
        if bar_completed:
            if is_compressed:
                if not state["compression_active"]:
                    state["compression_active"] = True
                    state["compression_start"] = self.clock.now().isoformat()
                    state["compression_bars"] = 1
                    state["compression_min_width"] = bb_width
                    logger.info(
                        f"Bot {bot.id}: Volatility Breakout compression STARTED - "
                        f"Method: {compression_method}, BB width: {bb_width:.4f}"
                    )
                else:
                    state["compression_bars"] += 1
                    # .get(): a legacy/partially-seeded state dict (e.g. a
                    # pre-migration persisted state, or a test seeding only
                    # the original keys) may not have this key yet.
                    _cur_min_width = state.get("compression_min_width")
                    if _cur_min_width is None or bb_width < _cur_min_width:
                        state["compression_min_width"] = bb_width
            else:
                # Compression ended
                if state["compression_active"]:
                    logger.info(
                        f"Bot {bot.id}: Volatility Breakout compression ENDED - "
                        f"Lasted {state['compression_bars']} bars (threshold: {min_compression_bars})"
                    )
                    state["compression_active"] = False
                    state["compression_start"] = None
                    state["compression_bars"] = 0
                    state["compression_min_width"] = None

        # Latch a "breakout armed" flag once compression has persisted long
        # enough. It MUST survive the expansion that follows: a genuine breakout
        # bar has wide close-dispersion (high BB width) and therefore reads as
        # NOT compressed, which would otherwise reset compression on the very bar
        # we want to act on (so the strategy could never fire). Arming decouples
        # "we were compressed" from "now we break out". It disarms on entry or if
        # price falls back to the mean (the setup has gone stale). The bars/width
        # AT THE MOMENT OF ARMING are captured (LOCKED) here because
        # compression_bars/compression_min_width reset to 0/None on the very
        # breakout bar itself (it is, by definition, no longer "compressed") -
        # without capturing them, the Evidence Items below would read a
        # transient 0 instead of the real compression episode's statistics.
        if (
            state["compression_active"]
            and state["compression_bars"] >= min_compression_bars
            and not state.get("breakout_armed")
        ):
            state["breakout_armed"] = True
            state["armed_compression_bars"] = state["compression_bars"]
            state["armed_compression_min_width"] = state.get("compression_min_width")
        if state.get("breakout_armed") and last_bar_close < sma:
            state["breakout_armed"] = False
            state["armed_compression_bars"] = None
            state["armed_compression_min_width"] = None
        compression_satisfied = bool(state.get("breakout_armed"))

        upper_gap = last_bar_close - upper_band
        upper_gap_pct = (upper_gap / upper_band * 100) if upper_band > 0 else 0.0
        compression_ratio = (bb_width / percentile_value) if percentile_value else None

        # --- Structured decision explanation: entry-side state (observe-only) --
        exp = self._explain(bot.id)
        exp.update({
            "current_price": current_price,
            "bar_close": last_bar_close,
            "bb_upper": upper_band,
            "bb_middle": sma,
            "bb_lower": lower_band,
            "bb_width": bb_width,
            "historical_compression_width": percentile_value,
            "compression_ratio": compression_ratio,
            "is_compressed": is_compressed,
            "compression_bars": state.get("compression_bars", 0),
            "min_compression_bars": min_compression_bars,
            "breakout_armed": compression_satisfied,
            "atr": atr,
            "atr_percentile": atr_percentile,
            "effective_stop_multiplier": atr_stop_mult,
            "breakout_threshold": upper_band,
            "breakout_distance": upper_gap,
            "breakout_gap_pct": upper_gap_pct,
            "regime_tags": suitability.regime_tags,
            "market_suitable": suitability.is_suitable,
            "has_position": False,
        })
        exp.metric("edge_status_category", edge_status.category.value)
        exp.check(
            "Compression armed", state.get("compression_bars", 0),
            f">= {min_compression_bars} bars compressed", compression_satisfied,
        )
        exp.check(
            "Market suitability", suitability.is_suitable, "must be True",
            suitability.is_suitable, detail=suitability.reason,
        )
        exp.check(
            "Strategy edge not disqualified", edge_status.category.value,
            f"!= {EdgeCategory.C.value}", edge_status.category != EdgeCategory.C,
            detail=edge_status.reason,
        )
        if not compression_satisfied:
            _need_bars = max(0, min_compression_bars - state.get("compression_bars", 0))
            exp.state("WAITING_COMPRESSION").next_trade(
                current=state.get("compression_bars", 0), current_label="Compression bars",
                target=min_compression_bars, target_label="Required bars",
                distance=_need_bars,
                status=(f"{_need_bars} more compressed bar(s)" if _need_bars > 0
                        else "armed — awaiting breakout"),
            )
        else:
            exp.state("WAITING_BREAKOUT").next_trade(
                current=last_bar_close, current_label="Bar close",
                target=upper_band, target_label="BB upper (breakout)", distance=upper_gap,
                status=(f"needs +{abs(upper_gap):.2f} to break out" if upper_gap < 0
                        else "breakout level reached"),
            )
        # -------------------------------------------------------------------

        # === PRECONDITION 1: compression must be armed before evidence can
        # be meaningfully collected at all - there is no setup to score. ===
        if not compression_satisfied:
            self._volatility_breakout_states[bot.id] = state
            proposal = _single_evidence_proposal(
                direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                evidence_name="Compression armed", evidence_value=0.0,
                evidence_reason=(
                    f"only {state.get('compression_bars', 0)}/{min_compression_bars} "
                    "compressed bars accumulated - no mature setup to evaluate yet"
                ),
                threshold=1.0,
            )
            adapter_signal = StandaloneAdapter.to_trade_signal(proposal)
            if adapter_signal is not None:
                return adapter_signal
            reason_text = "; ".join(proposal.reasons_against) or "watching for compression"
            if state["compression_active"]:
                return TradeSignal(
                    action="hold", amount=0,
                    reason=(
                        f"Volatility Breakout: Compression building "
                        f"({state['compression_bars']}/{min_compression_bars} bars) ({reason_text})"
                    ),
                )
            return TradeSignal(
                action="hold", amount=0,
                reason=f"Volatility Breakout: Watching for compression (BB width: {bb_width:.4f}) ({reason_text})",
            )

        # === PRECONDITION 2: PILLAR 2 HARD GATE - refuse unsuitable markets,
        # even with a fully armed, mature compression setup. ===
        if not suitability.is_suitable:
            self._volatility_breakout_states[bot.id] = state
            proposal = _single_evidence_proposal(
                direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                evidence_name="Market suitability", evidence_value=0.0,
                evidence_reason=suitability.reason, threshold=1.0,
            )
            adapter_signal = StandaloneAdapter.to_trade_signal(proposal)
            if adapter_signal is not None:
                return adapter_signal
            reason_text = "; ".join(proposal.reasons_against) or suitability.reason
            return TradeSignal(
                action="hold", amount=0,
                reason=f"Volatility Breakout: Compression armed but regime unsuitable ({reason_text})",
            )

        # === PILLAR 3: EVIDENCE COLLECTION ===
        # Compression armed AND market suitable - a real setup exists.
        # Evidence is collected and scored regardless of whether price has
        # technically crossed the upper band yet; "Breakout magnitude" is
        # simply negative/small until it does, so the score naturally fails
        # to clear threshold rather than needing a separate boolean gate.
        armed_bars = state.get("armed_compression_bars") or state.get("compression_bars", 0)
        armed_min_width = state.get("armed_compression_min_width")

        def _volatility_expansion_ratio() -> float:
            hist = state["atr_history"]
            if len(hist) < 14:
                return 1.0
            recent7 = hist[-7:]
            older7 = hist[-14:-7]
            older_avg = sum(older7) / 7
            recent_avg = sum(recent7) / 7
            return (recent_avg / older_avg) if older_avg > 0 else 1.0

        expansion_ratio = _volatility_expansion_ratio()

        evidence_items = [
            EvidenceItem(
                name="Breakout magnitude",
                measurement=lambda d: (last_bar_close - upper_band) / atr if atr > 0 else 0.0,
                normalization=lambda r: max(-1.0, min(1.0, r / 1.0)),
                weight=30.0,
                reason=(
                    "Theory (liquidity-vacuum/stop-run): a genuine breakout is "
                    "already a meaningful fraction of the measured range (ATR) "
                    "beyond the compression boundary, not a marginal tick above "
                    "it - larger magnitude is stronger evidence the move has "
                    "actually started, not noise around the band."
                ),
            ),
            EvidenceItem(
                name="Compression maturity",
                measurement=lambda d: (armed_bars / min_compression_bars) if min_compression_bars > 0 else 0.0,
                normalization=lambda r: max(-1.0, min(1.0, (r - 1.0) * 2.0)),
                weight=25.0,
                reason=(
                    "Theory: the longer price coils before breaking out, the "
                    "more resting stop and breakout orders concentrate just "
                    "beyond the range, and the stronger the eventual move tends "
                    "to be once it releases."
                ),
            ),
            EvidenceItem(
                name="Compression tightness",
                measurement=lambda d: (
                    (1.0 - (armed_min_width / percentile_value))
                    if (armed_min_width is not None and percentile_value) else 0.0
                ),
                normalization=lambda r: max(-1.0, min(1.0, r / 0.5)),
                weight=25.0,
                reason=(
                    "Theory: a tighter compression (BB width well below the "
                    "historical compression threshold) implies a smaller, more "
                    "concentrated order-book vacuum and therefore a more convex "
                    "expected move once it releases."
                ),
            ),
            EvidenceItem(
                name="Volatility expansion strength",
                measurement=lambda d: expansion_ratio - 1.0,
                normalization=lambda r: max(-1.0, min(1.0, r / 0.3)),
                weight=20.0,
                reason=(
                    "Theory: the compression-then-breakout thesis requires "
                    "volatility to actually be expanding, not merely that price "
                    "crossed a static band level - a faster rate of ATR "
                    "expansion is stronger, independent confirmation beyond the "
                    "pass/fail Market Suitability Gate."
                ),
            ),
        ]

        # === PILLAR 3: EVIDENCE-BASED DECISION SCORE ===
        decision_score = DecisionScoreEngine().score(
            "volatility_breakout", evidence_items, {}, threshold=decision_score_threshold,
        )
        edge_manager.record_decision_score(bot.id, "volatility_breakout", decision_score.total)

        exp.metric("decision_score_total", decision_score.total)
        exp.metric("decision_score_threshold", decision_score_threshold)
        exp.check(
            "Decision Score clears threshold", decision_score.total,
            f">= {decision_score_threshold:.1f}", decision_score.approved,
        )

        reasons_for, reasons_against = derive_reasons(decision_score)
        assumptions = (
            f"breakout level (BB upper ${upper_band:.2f}) not invalidated by a "
            f"close back inside the range within {failed_breakout_bars} bars",
            "current volatility regime remains 'expanding' per the bar-based "
            "direction detector",
            f"compression episode ({armed_bars} bars) is not itself falsified "
            "by price falling back below the mean before a breakout confirms",
        )

        blocking_reasons = []
        if not decision_score.approved:
            blocking_reasons.append(
                f"Decision Score {decision_score.total:.1f} < threshold {decision_score_threshold:.1f}"
            )
        if edge_status.category == EdgeCategory.C:
            blocking_reasons.append(
                f"Strategy Edge Management: Category C - {edge_status.reason}"
            )
        cooldown_remaining_hours = 0.0
        if state["last_breakout_attempt"] is not None:
            last_attempt = datetime.fromisoformat(state["last_breakout_attempt"])
            hours_since = (self.clock.now() - last_attempt).total_seconds() / 3600
            if hours_since < cooldown_hours:
                cooldown_remaining_hours = cooldown_hours - hours_since
                blocking_reasons.append(f"cooldown active ({cooldown_remaining_hours:.1f}h remaining)")

        if blocking_reasons:
            self._volatility_breakout_states[bot.id] = state
            generated_at = self.clock.now()
            proposal = StrategyProposal(
                strategy_id="volatility_breakout", bot_id=bot.id, generated_at=generated_at,
                direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                validity=ProposalValidity(
                    generated_at=generated_at,
                    valid_until=generated_at + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=decision_score, market_suitability=suitability, edge_status=edge_status,
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_risk_budget_pct=risk_percent,
                adaptive_parameters_used={"atr_stop_multiplier": atr_stop_mult},
                explanation=exp.to_dict(),
            )
            adapter_signal = StandaloneAdapter.to_trade_signal(proposal)
            if adapter_signal is not None:
                return adapter_signal
            reason_text = "; ".join(blocking_reasons)
            logger.debug(f"Bot {bot.id}: VB no-trade - {reason_text} (score={decision_score.total:.1f})")
            return TradeSignal(
                action="hold", amount=0,
                reason=f"Volatility Breakout: {reason_text} (score {decision_score.total:.1f}/{decision_score_threshold:.1f})",
            )

        # === PILLAR 5: DECISION-SCORE-WEIGHTED POSITION SIZING ===
        # A marginal-score trade (just above threshold) sizes toward 0.5x;
        # a maximal-score trade sizes toward 1.5x. Deterministic and
        # reproducible per Pillar 5's own requirement - not a separate,
        # unaccountable scaling knob. (add-unified-position-sizing will
        # eventually consolidate this into a shared cross-strategy
        # function per that change's revised design.md; this is this
        # strategy's own certification-phase implementation until then.)
        score_range = max(100.0 - decision_score_threshold, 1e-9)
        score_margin = max(0.0, min(1.0, (decision_score.total - decision_score_threshold) / score_range))
        size_multiplier = 0.5 + score_margin
        exp.metric("decision_score_size_multiplier", size_multiplier)

        # Volatility-adjusted, Decision-Score-weighted position sizing: risk a
        # %-of-capital, scaled by size_multiplier; the ATR-based stop distance
        # (now the LIVE adaptively-resolved multiplier) gives the coin count.
        risk_amount = bot.current_balance * risk_percent * size_multiplier

        if atr > 0:
            stop_distance = atr * atr_stop_mult             # USD per coin
            position_coins = risk_amount / stop_distance    # base coins
            position_size = position_coins * current_price  # quote USD notional
        else:
            position_size = risk_amount

        # Cap at available balance less execution cost buffer (fee + spread)
        # so the simulated exchange cannot reject for insufficient funds.
        buy_amount = min(position_size, bot.current_balance * _BUY_BALANCE_FRACTION)

        # Floor to the executable minimum (a sub-minimum buy is rejected by
        # the engine every loop); HOLD if the balance cannot afford it.
        if buy_amount < MIN_ORDER_USD:
            if bot.current_balance >= MIN_ORDER_USD:
                buy_amount = MIN_ORDER_USD
            else:
                self._volatility_breakout_states[bot.id] = state
                return TradeSignal(
                    action="hold", amount=0,
                    reason=(
                        f"Volatility Breakout: balance ${bot.current_balance:.2f} "
                        f"below ${MIN_ORDER_USD:.0f} minimum order"
                    ),
                )

        # Viability pre-check: expected move = how far price has already
        # broken above the upper band (the measured breakout magnitude). A
        # breakout is expected to continue at least this distance, making it
        # the most conservative honest estimate available without inventing
        # a price target.
        _vb_expected_move = upper_gap / current_price if current_price > 0 else 0.0
        _fee_raw_vb = getattr(bot, 'exchange_fee', 0.1)
        _vb_fee_pct = (
            float(_fee_raw_vb) if isinstance(_fee_raw_vb, (int, float)) else 0.1
        ) / 100.0
        _vb_min_move = 2.0 * _vb_fee_pct + _VIABILITY_SAFETY_MARGIN_PCT
        exp.check(
            "Fee viability", _vb_expected_move, f">= {_vb_min_move:.5f}",
            _vb_expected_move >= _vb_min_move,
        )
        if _vb_expected_move < _vb_min_move:
            self._volatility_breakout_states[bot.id] = state
            return TradeSignal(
                action="hold", amount=0,
                reason=(
                    f"Volatility Breakout: breakout {_vb_expected_move * 100:.3f}% < "
                    f"fee threshold {_vb_min_move * 100:.3f}% "
                    f"(exchange_fee={_vb_fee_pct * 100:.2f}%)"
                ),
            )

        # === PILLAR 10: STRATEGYPROPOSAL ===
        generated_at = self.clock.now()
        proposal = StrategyProposal(
            strategy_id="volatility_breakout", bot_id=bot.id, generated_at=generated_at,
            direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
            validity=ProposalValidity(
                generated_at=generated_at,
                valid_until=generated_at + timedelta(seconds=_validity_interval_seconds),
            ),
            decision_score=decision_score, market_suitability=suitability, edge_status=edge_status,
            assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
            suggested_position_size=buy_amount, suggested_risk_budget_pct=risk_percent * size_multiplier,
            expected_holding_horizon="short",
            adaptive_parameters_used={
                "atr_stop_multiplier": atr_stop_mult,
                "decision_score_size_multiplier": size_multiplier,
            },
            explanation=exp.to_dict(),
        )

        logger.info(
            f"Bot {bot.id}: Volatility Breakout ENTRY - "
            f"Decision Score {decision_score.total:.1f}/{decision_score_threshold:.1f}, "
            f"{armed_bars} bars compression, Bar close ${last_bar_close:.2f} > BB upper ${upper_band:.2f}, "
            f"Entry ATR locked at ${atr:.4f} x {atr_stop_mult:.2f}, Position: ${buy_amount:.2f}"
        )

        # === PILLAR 10: STANDALONE ADAPTER -> existing execution pipeline ===
        # Lock entry state (risk never expands mid-trade) - including the
        # LIVE-resolved adaptive stop multiplier and a reward:risk take-
        # profit target (closes the audit's Pillar 6 "no take-profit" gap).
        entry_stop_distance = atr * atr_stop_mult
        state["trailing_stop"] = current_price - entry_stop_distance
        state["take_profit_price"] = current_price + (entry_stop_distance * take_profit_rr_multiple)
        state["highest_price"] = current_price
        state["entry_price"] = current_price
        state["entry_atr"] = atr
        state["entry_stop_multiplier"] = atr_stop_mult
        state["entry_time"] = self.clock.now().isoformat()
        state["bars_since_entry"] = 0
        state["breakout_armed"] = False  # consumed this setup
        state["armed_compression_bars"] = None
        state["armed_compression_min_width"] = None
        state["last_breakout_attempt"] = self.clock.now().isoformat()

        self._volatility_breakout_states[bot.id] = state

        return StandaloneAdapter.to_trade_signal(
            proposal, expected_move_pct=_vb_expected_move,
        )

    # Note: TWAP and VWAP strategy methods removed.
    # TWAP/VWAP are execution algorithms, not alpha strategies.
    # They now exist in the execution layer as _execute_twap() and _execute_vwap().


    # Note: _strategy_arbitrage and _strategy_event were removed (placeholders without implementation)

    def _dip_recovery_default_state(self) -> dict:
        """Fresh default runtime state for a Dip Recovery bot.

        Single source of truth for the shape of dip_recovery state - reused by
        the live strategy, restart-restore (generic _restore_bot_state), and
        tests, so there is exactly one place that defines what this strategy
        persists.
        """
        return {
            "state": _DipRecoveryState.IDLE,
            "reference_high": None,
            "reference_high_time": None,
            "lowest_price": None,
            "lowest_price_time": None,
            "tracking_started_at": None,
            "ticks_since_new_low": 0,
            "entry_price": None,
            "entry_time": None,
            "entry_atr": None,
            "highest_price_since_entry": None,
            "trailing_stop": None,
            "take_profit": None,
            "emergency_stop": None,
            "cooldown_until": None,
            "last_exit_was_loss": None,
        }

    async def _strategy_dip_recovery(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Dip Recovery / Reversal Momentum strategy.

        Captures the bounce AFTER a significant decline, not the decline
        itself: a drop only ARMS monitoring; a BUY additionally requires price
        to have already reversed off a confirmed local low by an adaptive
        margin, with optional momentum confirmation. This intentionally
        accepts a later entry than a pure dip-buyer in exchange for materially
        reduced downside risk - it never buys into a market that is still
        falling.

        Lifecycle (persisted in bot.strategy_state via _dip_recovery_states):
            IDLE -> TRACKING_DROP -> WAITING_REVERSAL -> LONG_OPEN -> COOLDOWN -> IDLE
        ENTRY_ARMED is reported only in the explanation on the single tick a
        BUY fires - see _DipRecoveryState for why it is never itself persisted.

        All thresholds adapt to current volatility via an ATR-percent proxy
        (_calc_price_atr_proxy) computed from the bot's own tick price
        history - the same shared source trend_following uses. No OHLC candles
        are available for tick-driven bots, so True Range is approximated as
        the absolute tick-to-tick price change, exactly like trend_following.
        This makes every threshold pair-and-volatility agnostic (no BTC-
        specific assumptions): a 2% move only matters relative to what THIS
        pair's recent ATR% says is normal for it.

        Parameters (sane crypto defaults; also documented in
        validate_dip_recovery_params and app/routers/config.py STRATEGIES):
            atr_period (14): ticks in the ATR proxy window.
            reference_high_lookback_ticks (60): rolling window used to derive
                the "recent high" a decline is measured against.
            min_drop_percent / drop_atr_multiplier (1.5 / 2.5): adaptive drop
                threshold = max(min_drop_percent, atr_percent * drop_atr_multiplier).
            min_recovery_percent / recovery_atr_multiplier (0.5 / 0.8): adaptive
                recovery confirmation = max(min_recovery_percent, atr_percent * recovery_atr_multiplier).
            require_ema_slope_confirmation / ema_slope_period (True / 5): the
                short EMA over price history must be rising.
            require_no_new_low_confirmation / min_ticks_without_new_low (True / 2):
                no new low for N ticks.
            take_profit_atr_multiplier / trailing_stop_atr_multiplier /
                emergency_stop_atr_multiplier (3.0 / 1.5 / 5.0): ATR-based
                exits, locked at entry (risk never expands, matching
                trend_following/volatility_breakout convention).
            max_position_duration_minutes (720): force-exit after this long
                regardless of P&L.
            setup_expiry_minutes (240): abandon an unresolved TRACKING_DROP /
                WAITING_REVERSAL setup after this long with no confirmed
                reversal, back to IDLE.
            cooldown_seconds / loss_cooldown_seconds (300 / 1800): pause after
                any exit, extended after a losing exit.
            risk_percent (1.0): % of balance risked per trade, sized off the
                trailing-stop distance (same sizing formula as trend_following).
            spike_guard_atr_multiplier (6.0): a single-tick move bigger than
                this many ATRs is treated as noise and ignored when updating
                the tracked high/low, so one bad tick cannot anchor an
                unrealistic reference point.
        """
        atr_period = int(params.get("atr_period", 14))
        ref_high_lookback = int(params.get("reference_high_lookback_ticks", 60))
        min_drop_pct = params.get("min_drop_percent", 1.5)
        drop_atr_mult = params.get("drop_atr_multiplier", 2.5)
        min_recovery_pct = params.get("min_recovery_percent", 0.5)
        recovery_atr_mult = params.get("recovery_atr_multiplier", 0.8)
        require_ema_slope = params.get("require_ema_slope_confirmation", True)
        ema_slope_period = int(params.get("ema_slope_period", 5))
        require_no_new_low = params.get("require_no_new_low_confirmation", True)
        min_ticks_no_new_low = int(params.get("min_ticks_without_new_low", 2))
        take_profit_atr_mult = params.get("take_profit_atr_multiplier", 3.0)
        trailing_atr_mult = params.get("trailing_stop_atr_multiplier", 1.5)
        emergency_atr_mult = params.get("emergency_stop_atr_multiplier", 5.0)
        max_duration_min = params.get("max_position_duration_minutes", 720)
        setup_expiry_min = params.get("setup_expiry_minutes", 240)
        cooldown_seconds = params.get("cooldown_seconds", 300)
        loss_cooldown_seconds = params.get("loss_cooldown_seconds", 1800)
        risk_percent = params.get("risk_percent", 1.0) / 100
        spike_guard_mult = params.get("spike_guard_atr_multiplier", 6.0)

        now = self.clock.now()

        # === Price history (shared tick history, same source as trend_following) ===
        price_history = self._get_price_history(bot.id)
        previous_price = price_history[-1] if price_history else current_price
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history, max_len=max(ref_high_lookback + 50, 250))

        if not hasattr(self, "_dip_recovery_states"):
            self._dip_recovery_states = {}
        state = self._dip_recovery_states.get(bot.id) or self._dip_recovery_default_state()

        exp = self._explain(bot.id)

        # Warm-up gate: need enough ticks for a meaningful ATR proxy.
        if len(price_history) < atr_period + 1:
            self._dip_recovery_states[bot.id] = state
            exp.state(state["state"]).metric("current_price", current_price).check(
                "History collected", len(price_history), f">= {atr_period + 1}", False,
                detail="warming up ATR window",
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Dip Recovery: Collecting data ({len(price_history)}/{atr_period + 1})"
            )

        atr = self._calc_price_atr_proxy(price_history, atr_period)
        atr_percent = (atr / current_price * 100.0) if current_price > 0 else 0.0
        is_spike = atr > 0 and abs(current_price - previous_price) > (atr * spike_guard_mult)

        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # === STRATEGY DECISION FRAMEWORK (Phase 3 migration) ===
        # Pillar 2 rationale (documented, not silently assumed - see the audit's
        # Pillar 2 section): dip_recovery is TICK-driven (it keeps a tick price
        # history and an ATR *proxy*, never OHLC bars), so it uses the price-only
        # `_detect_market_regime` - the documented variant for tick strategies -
        # NOT `_detect_market_regime_bar_based`. Consequence, recorded here and in
        # the audit: the price-only detector emits no `volatility_direction`, so
        # the `volatility_expanding` tag NEVER matches in this standalone path
        # (Auto Mode's bar-based path still evaluates it). `trend_down` and
        # `volatility_high` fully cover this strategy's setups (a sharp dip is a
        # high-volatility, declining regime), so nothing is silently lost.
        if not hasattr(self, "_dip_recovery_edge_manager"):
            self._dip_recovery_edge_manager = StrategyEdgeManager()
        edge_manager = self._dip_recovery_edge_manager
        allowed_regimes = params.get(
            "allowed_regimes", ["trend_down", "volatility_high", "volatility_expanding"]
        )
        decision_score_threshold = params.get("decision_score_threshold", 40.0)
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        _validity_interval_seconds = max(bar_interval_seconds, 1)
        current_regime = self._detect_market_regime(price_history, None)
        suitability = MarketSuitabilityGate().evaluate(current_regime, allowed_regimes)
        # Category B ("parameter mismatch") is never cited for this strategy:
        # its entry thresholds are ALREADY ATR-adaptive (Pillar 4, "best of the
        # six"), so there is no single fixed base-multiplier the Edge Manager
        # could point at as miscalibrated. Degradation therefore classifies as
        # Category A (regime unsuitable) or Category C (edge gone) only -
        # documented in the audit's Pillar 7 section.
        edge_status = edge_manager.evaluate(
            bot.id, "dip_recovery",
            regime_outside_suitable_range=not suitability.is_suitable,
            parameter_mismatch_evidence=None,
            now=now,
        )

        def mk_proposal(
            *, direction: "Direction", execution_intent: "ExecutionIntent",
            evidence_name: str, evidence_value: float, evidence_reason: str,
            threshold: float = 1.0, suggested_position_size: Optional[float] = None,
            assumptions: tuple = (),
        ) -> StrategyProposal:
            """Single-evidence proposal for every already-decided branch
            (tracking/waiting/cooldown holds, exits, no-trades). The full
            multi-factor entry proposal is built inline in the setup handler."""
            item = EvidenceItem(
                name=evidence_name, measurement=lambda d: evidence_value,
                normalization=lambda r: r, weight=100.0, reason=evidence_reason,
            )
            score = DecisionScoreEngine().score("dip_recovery", [item], {}, threshold=threshold)
            reasons_for, reasons_against = derive_reasons(score)
            gen = self.clock.now()
            return StrategyProposal(
                strategy_id="dip_recovery", bot_id=bot.id, generated_at=gen,
                direction=direction, execution_intent=execution_intent,
                validity=ProposalValidity(
                    generated_at=gen,
                    valid_until=gen + timedelta(seconds=_validity_interval_seconds),
                ),
                decision_score=score, market_suitability=suitability, edge_status=edge_status,
                assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
                suggested_position_size=suggested_position_size, suggested_risk_budget_pct=risk_percent,
                explanation=self._explain(bot.id).to_dict(),
            )

        def hold_via(proposal: StrategyProposal, reason: str) -> TradeSignal:
            """Route a HOLD/NO_TRADE proposal through the Standalone Adapter
            (which yields no order) but preserve the existing rich, human-
            readable hold reason for diagnostics/tests."""
            sig = StandaloneAdapter.to_trade_signal(proposal)
            return sig if sig is not None else TradeSignal(action="hold", amount=0, reason=reason)

        if not has_position and state["state"] == _DipRecoveryState.LONG_OPEN:
            # Defensive: persisted state says a position is open but none
            # exists (closed outside this strategy's knowledge, or a crash
            # between execution and the next state save). There is nothing to
            # manage and no valid TRACKING_DROP/WAITING_REVERSAL context either
            # (those fields were cleared on entry) - reset cleanly to IDLE
            # rather than falling into the setup-tracking branch below with
            # stale/absent low/high fields.
            state = self._dip_recovery_default_state()
            self._dip_recovery_states[bot.id] = state

        if has_position:
            return self._dip_recovery_manage_exit(
                bot, state, positions, current_price, atr, now,
                take_profit_atr_mult, trailing_atr_mult, emergency_atr_mult,
                max_duration_min, cooldown_seconds, loss_cooldown_seconds, exp,
                mk_proposal, edge_manager,
            )

        # === COOLDOWN ===
        if state["state"] == _DipRecoveryState.COOLDOWN:
            cooldown_until = state.get("cooldown_until")
            if cooldown_until and now < cooldown_until:
                remaining = (cooldown_until - now).total_seconds()
                exp.state(_DipRecoveryState.COOLDOWN).update({
                    "current_price": current_price, "atr": atr, "atr_percent": atr_percent,
                    "cooldown_remaining_s": remaining,
                    "last_exit_was_loss": state.get("last_exit_was_loss"),
                })
                exp.check("Cooldown elapsed", f"{remaining:.0f}s remaining", "0s remaining", False)
                exp.next_trade(
                    current=remaining, current_label="Cooldown remaining (s)",
                    target=0, target_label="Ready", distance=remaining,
                    status=f"{remaining:.0f}s until monitoring resumes",
                )
                self._dip_recovery_states[bot.id] = state
                return hold_via(
                    mk_proposal(
                        direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                        evidence_name="Cooldown elapsed", evidence_value=0.0,
                        evidence_reason=f"{remaining:.0f}s of post-exit cooldown remain before monitoring resumes",
                    ),
                    f"Dip Recovery: Cooldown active ({remaining:.0f}s remaining)",
                )
            # Cooldown elapsed - fall through to IDLE this same tick.
            state["state"] = _DipRecoveryState.IDLE
            state["cooldown_until"] = None

        drop_threshold = max(min_drop_pct, atr_percent * drop_atr_mult)
        recovery_threshold = max(min_recovery_pct, atr_percent * recovery_atr_mult)

        # === IDLE: watch for a significant decline ===
        if state["state"] == _DipRecoveryState.IDLE:
            window = price_history[-ref_high_lookback:]
            # A single-tick spike must not anchor the reference high.
            if is_spike and len(window) > 1:
                window = window[:-1]
            reference_high = max(window) if window else current_price
            drawdown_percent = (
                (current_price - reference_high) / reference_high * 100.0
                if reference_high > 0 else 0.0
            )
            decline_ratio = abs(drawdown_percent) / atr_percent if atr_percent > 0 else 0.0
            opportunity_score = self._dip_recovery_score_from_ratios(decline_ratio, 0.0)

            exp.state(_DipRecoveryState.IDLE).update({
                "current_price": current_price, "reference_high": reference_high,
                "atr": atr, "atr_percent": atr_percent,
                "drawdown_percent": drawdown_percent, "drop_threshold_percent": drop_threshold,
                "opportunity_score": opportunity_score * 10.0,
            })
            exp.check(
                "Decline vs adaptive threshold", f"{abs(drawdown_percent):.3f}%",
                f">= {drop_threshold:.3f}%", abs(drawdown_percent) >= drop_threshold,
            )

            if drawdown_percent < 0 and abs(drawdown_percent) >= drop_threshold:
                state.update({
                    "state": _DipRecoveryState.TRACKING_DROP,
                    "reference_high": reference_high,
                    "reference_high_time": now,
                    "lowest_price": current_price,
                    "lowest_price_time": now,
                    "tracking_started_at": now,
                    "ticks_since_new_low": 0,
                })
                self._dip_recovery_states[bot.id] = state
                exp.state(_DipRecoveryState.TRACKING_DROP).next_trade(
                    current=abs(drawdown_percent), current_label="Decline so far (%)",
                    target=drop_threshold, target_label="Drop threshold (%)",
                    distance=0.0, status="drop confirmed - tracking bottom for a reversal",
                )
                return hold_via(
                    mk_proposal(
                        direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                        evidence_name="Significant decline", evidence_value=1.0,
                        evidence_reason=(
                            f"decline {abs(drawdown_percent):.2f}% >= adaptive drop threshold "
                            f"{drop_threshold:.2f}% - arming reversal tracking (no entry yet)"
                        ),
                    ),
                    (
                        f"Dip Recovery: Significant decline detected "
                        f"({abs(drawdown_percent):.2f}% >= {drop_threshold:.2f}%) - tracking bottom"
                    ),
                )

            needed = max(0.0, drop_threshold - abs(drawdown_percent))
            exp.next_trade(
                current=abs(drawdown_percent), current_label="Current drawdown (%)",
                target=drop_threshold, target_label="Required (%)", distance=needed,
                status=f"{needed:.2f}% further decline needed to arm monitoring",
            )
            self._dip_recovery_states[bot.id] = state
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Significant decline", evidence_value=0.0,
                    evidence_reason=(
                        f"drawdown {abs(drawdown_percent):.2f}% < adaptive drop threshold "
                        f"{drop_threshold:.2f}% - no dip setup to evaluate"
                    ),
                ),
                (
                    f"Dip Recovery: No significant decline "
                    f"({abs(drawdown_percent):.2f}% < {drop_threshold:.2f}%)"
                ),
            )

        # === TRACKING_DROP / WAITING_REVERSAL ===
        return self._dip_recovery_manage_setup(
            bot, state, price_history, current_price, atr, atr_percent, is_spike, now,
            recovery_threshold, require_ema_slope, ema_slope_period,
            require_no_new_low, min_ticks_no_new_low, setup_expiry_min,
            risk_percent, take_profit_atr_mult, trailing_atr_mult, emergency_atr_mult,
            exp, mk_proposal, hold_via, suitability, edge_status, edge_manager,
            decision_score_threshold,
        )

    def _dip_recovery_manage_setup(
        self,
        bot: Bot,
        state: dict,
        price_history: list,
        current_price: float,
        atr: float,
        atr_percent: float,
        is_spike: bool,
        now: datetime,
        recovery_threshold: float,
        require_ema_slope: bool,
        ema_slope_period: int,
        require_no_new_low: bool,
        min_ticks_no_new_low: int,
        setup_expiry_min: float,
        risk_percent: float,
        take_profit_atr_mult: float,
        trailing_atr_mult: float,
        emergency_atr_mult: float,
        exp: ExplanationBuilder,
        mk_proposal,
        hold_via,
        suitability,
        edge_status,
        edge_manager,
        decision_score_threshold: float,
    ) -> TradeSignal:
        """Handle TRACKING_DROP / WAITING_REVERSAL: track the bottom, then
        confirm and arm a BUY once price has genuinely reversed off it.

        Migrated to the Strategy Decision Framework (Phase 3). The pre-migration
        "never buy on the way down" safety preconditions (a bounce has started,
        recovery cleared its adaptive threshold, and every enabled confirmation
        filter passed) are PRESERVED as hard gates; on top of them the migration
        adds the Pillar 2 suitability gate, a Pillar 3 Decision Score over the
        formerly-diagnostic-only opportunity signals, and the Pillar 7 edge gate.
        """
        reference_high = state.get("reference_high")
        tracking_started_at = state.get("tracking_started_at")

        # Invalidation: setup has gone stale without confirming a reversal.
        elapsed_minutes = (
            (now - tracking_started_at).total_seconds() / 60.0 if tracking_started_at else 0.0
        )
        if elapsed_minutes >= setup_expiry_min:
            self._dip_recovery_states[bot.id] = self._dip_recovery_default_state()
            exp.state(_DipRecoveryState.IDLE).check(
                "Setup expiry", f"{elapsed_minutes:.1f} min", f"< {setup_expiry_min} min", False,
            )
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Setup still valid", evidence_value=0.0,
                    evidence_reason=(
                        f"tracked dip setup expired after {elapsed_minutes:.1f} min without a "
                        "confirmed reversal - abandoned, back to IDLE"
                    ),
                ),
                (
                    f"Dip Recovery: Setup expired after {elapsed_minutes:.1f} min "
                    "without a confirmed reversal - back to IDLE"
                ),
            )

        # Invalidation: price fully round-tripped past the original high
        # without ever confirming entry - no longer "the dip"; reset and re-watch.
        if reference_high and current_price >= reference_high:
            self._dip_recovery_states[bot.id] = self._dip_recovery_default_state()
            exp.state(_DipRecoveryState.IDLE)
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Dip still valid", evidence_value=0.0,
                    evidence_reason=(
                        "price recovered past the original reference high without a confirmed "
                        "entry - no longer a dip, resetting"
                    ),
                ),
                (
                    "Dip Recovery: Price recovered past the original reference high "
                    "without confirmation - resetting"
                ),
            )

        lowest_price = state.get("lowest_price", current_price)
        is_new_low = (not is_spike) and current_price < lowest_price

        if is_new_low:
            state["lowest_price"] = current_price
            state["lowest_price_time"] = now
            state["ticks_since_new_low"] = 0
            lowest_price = current_price
        else:
            state["ticks_since_new_low"] = state.get("ticks_since_new_low", 0) + 1

        recovery_percent = (
            (current_price - lowest_price) / lowest_price * 100.0 if lowest_price > 0 else 0.0
        )

        if recovery_percent <= 0:
            # Still at/below the tracked low - no reversal yet, nothing to confirm.
            state["state"] = _DipRecoveryState.TRACKING_DROP
            self._dip_recovery_states[bot.id] = state
            exp.state(_DipRecoveryState.TRACKING_DROP).update({
                "current_price": current_price, "lowest_price": lowest_price,
                "atr": atr, "atr_percent": atr_percent, "recovery_percent": recovery_percent,
                "recovery_threshold_percent": recovery_threshold,
            })
            exp.check("New low tracking", current_price, f"<= {lowest_price:.4f}", True)
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Reversal started", evidence_value=0.0,
                    evidence_reason=(
                        f"price still at/below the tracked low ${lowest_price:.4f} - the "
                        "market is still falling, never buy on the way down"
                    ),
                ),
                f"Dip Recovery: Still declining, tracking bottom (${lowest_price:.4f})",
            )

        # Price has bounced off the low - evaluate reversal confirmation.
        ema_series = self._calculate_ema(price_history, ema_slope_period)
        ema_slope_positive = len(ema_series) >= 2 and ema_series[-1] > ema_series[-2]
        ema_ok = (not require_ema_slope) or ema_slope_positive

        no_new_low_ok = (
            (not require_no_new_low) or state["ticks_since_new_low"] >= min_ticks_no_new_low
        )

        recovery_ok = recovery_percent >= recovery_threshold
        # SAFETY PRECONDITIONS (preserved from pre-migration, unchanged): a real
        # reversal must be underway before ANY entry is considered. These are the
        # "never buy on the way down" guarantee and remain HARD gates - the
        # Decision Score below evaluates the QUALITY of a confirmed setup, it does
        # not override these safety filters.
        preconditions_met = recovery_ok and ema_ok and no_new_low_ok

        decline_ratio = (
            abs((reference_high - lowest_price) / reference_high * 100.0) / atr_percent
            if reference_high and atr_percent > 0 else 0.0
        )
        recovery_ratio = recovery_percent / atr_percent if atr_percent > 0 else 0.0
        opportunity_score = self._dip_recovery_score_from_ratios(decline_ratio, recovery_ratio)

        exp.update({
            "current_price": current_price, "lowest_price": lowest_price,
            "atr": atr, "atr_percent": atr_percent,
            "recovery_percent": recovery_percent, "recovery_threshold_percent": recovery_threshold,
            "ticks_since_new_low": state["ticks_since_new_low"],
            "ema_slope_positive": ema_slope_positive,
            "opportunity_score": opportunity_score * 10.0,
            "regime_tags": suitability.regime_tags,
            "market_suitable": suitability.is_suitable,
        })
        exp.metric("edge_status_category", edge_status.category.value)
        exp.check(
            "Recovery from bottom", f"{recovery_percent:.3f}%",
            f">= {recovery_threshold:.3f}%", recovery_ok,
        )
        if require_ema_slope:
            exp.check("Short EMA slope positive", ema_slope_positive, "True", ema_slope_positive)
        if require_no_new_low:
            exp.check(
                "No new low", state["ticks_since_new_low"],
                f">= {min_ticks_no_new_low} ticks", no_new_low_ok,
            )

        if not preconditions_met:
            state["state"] = _DipRecoveryState.WAITING_REVERSAL
            self._dip_recovery_states[bot.id] = state
            needed = max(0.0, recovery_threshold - recovery_percent)
            exp.state(_DipRecoveryState.WAITING_REVERSAL).next_trade(
                current=recovery_percent, current_label="Recovery from bottom (%)",
                target=recovery_threshold, target_label="Required (%)", distance=needed,
                status="waiting for full reversal confirmation",
            )
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Reversal confirmed", evidence_value=0.0,
                    evidence_reason=(
                        f"reversal not yet fully confirmed (recovery {recovery_percent:.2f}% / "
                        f"{recovery_threshold:.2f}% required, EMA slope / no-new-low filters)"
                    ),
                ),
                (
                    f"Dip Recovery: Reversal forming ({recovery_percent:.2f}%/"
                    f"{recovery_threshold:.2f}% required) - awaiting confirmation"
                ),
            )

        # === PILLAR 3: EVIDENCE-BASED DECISION SCORE ===
        # The formerly diagnostic-only opportunity signals (decline depth and
        # recovery strength, both ATR-normalized) plus the two confirmation
        # filters, formalized as documented Evidence Items feeding the actual
        # entry decision (Task 3.3).
        _ticks_since_low = float(state["ticks_since_new_low"])
        _no_low_ratio = (
            min(_ticks_since_low / min_ticks_no_new_low, 1.5)
            if min_ticks_no_new_low > 0 else 1.0
        )
        evidence_items = [
            EvidenceItem(
                name="Decline depth",
                measurement=lambda d: decline_ratio,
                normalization=lambda r: max(-1.0, min(1.0, r / 2.5)),
                weight=35.0,
                reason=(
                    "Theory (overreaction mean-reversion): the deeper the panic decline "
                    "relative to this pair's own ATR, the more likely it overshot fair value "
                    "and the larger the mean-reversion bounce being captured."
                ),
            ),
            EvidenceItem(
                name="Recovery strength",
                measurement=lambda d: recovery_ratio,
                normalization=lambda r: max(-1.0, min(1.0, r / 1.0)),
                weight=30.0,
                reason=(
                    "Theory: a reversal that has already retraced a meaningful fraction of an "
                    "ATR off the low is stronger evidence the bottom is in than a marginal tick "
                    "up - this is the 'confirmed reversal, not a falling knife' core of the thesis."
                ),
            ),
            EvidenceItem(
                name="Reversal momentum (EMA slope)",
                measurement=lambda d: 1.0 if ema_slope_positive else 0.0,
                normalization=lambda r: max(-1.0, min(1.0, r)),
                weight=20.0,
                reason=(
                    "Theory: a rising short EMA confirms the bounce has momentum behind it, "
                    "not just a single mean-reverting tick that will resume falling."
                ),
            ),
            EvidenceItem(
                name="Base stability (no new low)",
                measurement=lambda d: _no_low_ratio,
                normalization=lambda r: max(-1.0, min(1.0, r)),
                weight=15.0,
                reason=(
                    "Theory: the longer price holds without printing a new low, the more the "
                    "sellers are exhausted and the safer the reversal entry."
                ),
            ),
        ]
        decision_score = DecisionScoreEngine().score(
            "dip_recovery", evidence_items, {}, threshold=decision_score_threshold,
        )
        edge_manager.record_decision_score(bot.id, "dip_recovery", decision_score.total)
        reasons_for, reasons_against = derive_reasons(decision_score)
        assumptions = (
            "no new low since entry: the tracked bottom is not undercut",
            "the confirmed reversal is not reversed (price does not fall back below the "
            "recovery threshold off the low)",
            "current market regime remains within the strategy's allowed set",
        )
        exp.metric("decision_score_total", decision_score.total)
        exp.metric("decision_score_threshold", decision_score_threshold)
        exp.check(
            "Decision Score clears threshold", decision_score.total,
            f">= {decision_score_threshold:.1f}", decision_score.approved,
        )
        exp.check(
            "Market suitability", suitability.is_suitable, "must be True",
            suitability.is_suitable, detail=suitability.reason,
        )
        exp.check(
            "Strategy edge not disqualified", edge_status.category.value,
            f"!= {EdgeCategory.C.value}", edge_status.category != EdgeCategory.C,
            detail=edge_status.reason,
        )

        blocking_reasons = []
        if not suitability.is_suitable:
            blocking_reasons.append(f"regime unsuitable ({suitability.reason})")
        if not decision_score.approved:
            blocking_reasons.append(
                f"Decision Score {decision_score.total:.1f} < threshold {decision_score_threshold:.1f}"
            )
        if edge_status.category == EdgeCategory.C:
            blocking_reasons.append(f"Strategy Edge Management: Category C - {edge_status.reason}")

        if blocking_reasons:
            state["state"] = _DipRecoveryState.WAITING_REVERSAL
            self._dip_recovery_states[bot.id] = state
            reason_text = "; ".join(blocking_reasons)
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Framework gates", evidence_value=0.0,
                    evidence_reason=reason_text, assumptions=assumptions,
                ),
                f"Dip Recovery: reversal confirmed but {reason_text}",
            )

        # === ENTRY ARMED -> BUY ===
        exp.state(_DipRecoveryState.ENTRY_ARMED)

        # === PILLAR 5: DECISION-SCORE-WEIGHTED POSITION SIZING ===
        score_range = max(100.0 - decision_score_threshold, 1e-9)
        score_margin = max(0.0, min(1.0, (decision_score.total - decision_score_threshold) / score_range))
        size_multiplier = 0.5 + score_margin
        exp.metric("decision_score_size_multiplier", size_multiplier)

        risk_amount = bot.current_balance * risk_percent * size_multiplier
        trailing_distance = atr * trailing_atr_mult
        if trailing_distance > 0:
            position_coins = risk_amount / trailing_distance
            position_size = position_coins * current_price
        else:
            position_size = risk_amount

        buy_amount = min(position_size, bot.current_balance * _BUY_BALANCE_FRACTION)
        if buy_amount < MIN_ORDER_USD:
            if bot.current_balance >= MIN_ORDER_USD:
                buy_amount = MIN_ORDER_USD
            else:
                # Pillar 8: sizing is a decision point - surface it.
                exp.check("Position size >= min order", buy_amount, f">= {MIN_ORDER_USD:.0f}", False)
                self._dip_recovery_states[bot.id] = state
                return hold_via(
                    mk_proposal(
                        direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                        evidence_name="Order size", evidence_value=0.0,
                        evidence_reason=(
                            f"balance ${bot.current_balance:.2f} below ${MIN_ORDER_USD:.0f} "
                            "minimum order - cannot enter"
                        ),
                    ),
                    (
                        f"Dip Recovery: balance ${bot.current_balance:.2f} below "
                        f"${MIN_ORDER_USD:.0f} minimum order"
                    ),
                )
        exp.check("Position size >= min order", buy_amount, f">= {MIN_ORDER_USD:.0f}", True)

        # Viability pre-check before state mutation: expected move = distance from
        # entry to the strategy's take-profit target (atr × take_profit_atr_mult).
        # This is the genuine profit objective baked into the strategy — not
        # invented — and must cover round-trip fees before we commit state.
        _dr_expected_move = (atr * take_profit_atr_mult) / current_price if current_price > 0 else 0.0
        _fee_raw_dr = getattr(bot, 'exchange_fee', 0.1)
        _dr_fee_pct = (
            float(_fee_raw_dr) if isinstance(_fee_raw_dr, (int, float)) else 0.1
        ) / 100.0
        _dr_min_move = 2.0 * _dr_fee_pct + _VIABILITY_SAFETY_MARGIN_PCT
        exp.check(
            "Fee viability", _dr_expected_move, f">= {_dr_min_move:.5f}",
            _dr_expected_move >= _dr_min_move,
        )
        if _dr_expected_move < _dr_min_move:
            self._dip_recovery_states[bot.id] = state
            return hold_via(
                mk_proposal(
                    direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION,
                    evidence_name="Fee viability", evidence_value=0.0,
                    evidence_reason=(
                        f"take-profit target {_dr_expected_move * 100:.3f}% < fee threshold "
                        f"{_dr_min_move * 100:.3f}% - the trade cannot clear round-trip costs"
                    ),
                ),
                (
                    f"Dip Recovery: take-profit {_dr_expected_move * 100:.3f}% < "
                    f"fee threshold {_dr_min_move * 100:.3f}% "
                    f"(exchange_fee={_dr_fee_pct * 100:.2f}%)"
                ),
            )

        entry_price = current_price
        self._dip_recovery_states[bot.id] = {
            **self._dip_recovery_default_state(),
            "state": _DipRecoveryState.LONG_OPEN,
            "entry_price": entry_price,
            "entry_time": now,
            "entry_atr": atr,
            "highest_price_since_entry": entry_price,
            "trailing_stop": entry_price - (atr * trailing_atr_mult),
            "take_profit": entry_price + (atr * take_profit_atr_mult),
            "emergency_stop": entry_price - (atr * emergency_atr_mult),
        }

        logger.info(
            f"Bot {bot.id}: Dip Recovery ENTRY - Price ${entry_price:.4f}, "
            f"Decision Score {decision_score.total:.1f}/{decision_score_threshold:.1f}, "
            f"recovery {recovery_percent:.2f}% (required {recovery_threshold:.2f}%), "
            f"ATR ${atr:.4f}, Position ${buy_amount:.2f}"
        )

        # Risk side for the reward:risk viability check: distance to the
        # INITIAL trailing stop (atr × trailing_atr_mult) — the first exit that
        # would trigger on adverse movement, not the wider emergency_stop
        # (a last-resort net beyond the trailing stop, not the intended risk).
        _dr_expected_risk = (atr * trailing_atr_mult) / current_price if current_price > 0 else 0.0

        # === PILLAR 10: STRATEGYPROPOSAL -> STANDALONE ADAPTER ===
        gen = self.clock.now()
        proposal = StrategyProposal(
            strategy_id="dip_recovery", bot_id=bot.id, generated_at=gen,
            direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
            validity=ProposalValidity(
                generated_at=gen,
                valid_until=gen + timedelta(seconds=self._dip_recovery_validity_seconds(bot)),
            ),
            decision_score=decision_score, market_suitability=suitability, edge_status=edge_status,
            assumptions=assumptions, reasons_for=reasons_for, reasons_against=reasons_against,
            suggested_position_size=buy_amount,
            suggested_risk_budget_pct=risk_percent * size_multiplier,
            expected_holding_horizon="medium",
            adaptive_parameters_used={"decision_score_size_multiplier": size_multiplier},
            explanation=exp.to_dict(),
        )
        return StandaloneAdapter.to_trade_signal(
            proposal, expected_move_pct=_dr_expected_move, expected_risk_pct=_dr_expected_risk,
        )

    def _dip_recovery_validity_seconds(self, bot) -> int:
        """Proposal validity window for a Dip Recovery proposal - tied to this
        strategy's per-tick evaluation cadence (its `bar_interval_seconds`
        param, default 60), floored at 1s so ProposalValidity never gets a
        zero-duration window from a test harness's bar_interval_seconds=0."""
        params = getattr(bot, "strategy_params", None) or {}
        return max(int(params.get("bar_interval_seconds", 60)), 1)

    def _dip_recovery_exit_signal(
        self,
        bot: Bot,
        pos,
        current_price: float,
        entry_price: float,
        unrealized_pnl_pct: float,
        reason: str,
        cooldown_secs: float,
        now: datetime,
        mk_proposal,
    ) -> TradeSignal:
        """Build the SELL StrategyProposal for a Dip Recovery exit and reset
        state to COOLDOWN. Single source of truth for the cooldown-reset
        behaviour so every exit path (take-profit, trailing stop, emergency
        stop, max duration) resets state identically. Pillar 10: the proposal
        is routed through the Standalone Adapter, never returned directly.
        """
        is_loss = current_price < entry_price
        self._dip_recovery_states[bot.id] = {
            **self._dip_recovery_default_state(),
            "state": _DipRecoveryState.COOLDOWN,
            "cooldown_until": now + timedelta(seconds=cooldown_secs),
            "last_exit_was_loss": is_loss,
        }
        logger.info(
            f"Bot {bot.id}: Dip Recovery EXIT ({reason}) - Price ${current_price:.4f}, "
            f"Entry ${entry_price:.4f}, PnL {unrealized_pnl_pct:+.2f}%"
        )
        proposal = mk_proposal(
            direction=Direction.SELL, execution_intent=ExecutionIntent.CLOSE_POSITION,
            evidence_name=f"Exit: {reason}", evidence_value=1.0,
            evidence_reason=(
                f"Dip Recovery {reason} exit at ${current_price:.4f} "
                f"(entry ${entry_price:.4f}, PnL {unrealized_pnl_pct:+.2f}%)"
            ),
            suggested_position_size=pos.amount * current_price,
        )
        return StandaloneAdapter.to_trade_signal(proposal)

    def _dip_recovery_manage_exit(
        self,
        bot: Bot,
        state: dict,
        positions: list,
        current_price: float,
        atr: float,
        now: datetime,
        take_profit_atr_mult: float,
        trailing_atr_mult: float,
        emergency_atr_mult: float,
        max_duration_min: float,
        cooldown_seconds: float,
        loss_cooldown_seconds: float,
        exp: ExplanationBuilder,
        mk_proposal,
        edge_manager,
    ) -> TradeSignal:
        """Manage an open Dip Recovery position: take-profit, monotonic
        trailing stop, emergency stop, and max-duration exits, each routed
        through a loss-aware cooldown (see _dip_recovery_exit_signal).

        Migrated (Phase 3): exits record their outcome with the StrategyEdge
        Manager (Pillar 7) and are emitted as StrategyProposals via the
        Standalone Adapter (Pillar 10); the emergency-stop check is now surfaced
        to diagnostics (Pillar 8 gap the audit found). Trade-management logic
        (the four exits, monotonic trailing stop, loss-aware cooldown) is
        otherwise unchanged.
        """
        entry_price = state.get("entry_price")
        entry_price_known = entry_price is not None
        entry_atr_locked = state.get("entry_atr") or atr  # fallback for legacy/corrupted state
        entry_time = state.get("entry_time") or now
        if entry_price is None:
            # Defensive: a position exists but entry was never recorded (state
            # lost before restart-restore). Adopt current price as entry so
            # exits still function instead of crashing or buying again.
            entry_price = current_price
            entry_atr_locked = atr

        previous_highest = state.get("highest_price_since_entry") or entry_price
        highest = max(previous_highest, current_price)
        trailing_stop = state.get("trailing_stop")
        if trailing_stop is None or highest > previous_highest:
            # Monotonic tightening only - risk never expands (same convention
            # as trend_following / volatility_breakout).
            trailing_stop = highest - (entry_atr_locked * trailing_atr_mult)
        take_profit = state.get("take_profit") or (
            entry_price + entry_atr_locked * take_profit_atr_mult
        )
        emergency_stop = state.get("emergency_stop") or (
            entry_price - entry_atr_locked * emergency_atr_mult
        )

        duration_minutes = (now - entry_time).total_seconds() / 60.0
        unrealized_pnl_pct = (
            (current_price - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0
        )

        state.update({
            "state": _DipRecoveryState.LONG_OPEN, "entry_price": entry_price,
            "entry_time": entry_time, "entry_atr": entry_atr_locked,
            "highest_price_since_entry": highest, "trailing_stop": trailing_stop,
            "take_profit": take_profit, "emergency_stop": emergency_stop,
        })

        exp.state(_DipRecoveryState.LONG_OPEN).update({
            "current_price": current_price, "entry_price": entry_price, "highest_price": highest,
            "trailing_stop": trailing_stop, "take_profit": take_profit,
            "emergency_stop": emergency_stop, "unrealized_pnl_percent": unrealized_pnl_pct,
            "position_duration_minutes": duration_minutes,
        })
        exp.check("Take profit hit", current_price, f">= {take_profit:.4f}", current_price >= take_profit)
        exp.check("Trailing stop hit", current_price, f"<= {trailing_stop:.4f}", current_price <= trailing_stop)
        # Pillar 8 gap the audit found: the emergency stop was computed and used
        # to trigger sells but never surfaced as a diagnostic check.
        exp.check(
            "Emergency stop hit", current_price, f"<= {emergency_stop:.4f}",
            current_price <= emergency_stop,
        )
        exp.check(
            "Max duration", f"{duration_minutes:.1f} min",
            f"< {max_duration_min} min", duration_minutes < max_duration_min,
        )
        exp.next_trade(
            current=current_price, current_label="Current price",
            target=(take_profit if unrealized_pnl_pct >= 0 else trailing_stop),
            target_label=("Take profit" if unrealized_pnl_pct >= 0 else "Trailing stop"),
            distance=(take_profit - current_price),
            status="holding - exits on take-profit, trailing stop, emergency stop, or max duration",
        )

        def _record_exit_outcome() -> None:
            # Pillar 7: record the outcome BEFORE state is cleared. Only when a
            # real entry price is known - a defensively-adopted entry has no
            # true P&L and recording a fabricated value would corrupt the
            # StrategyEdgeManager's statistics.
            if not entry_price_known:
                return
            initial_risk_per_unit = entry_atr_locked * trailing_atr_mult
            pnl_per_unit = current_price - entry_price
            reward_risk_realized = (
                (pnl_per_unit / initial_risk_per_unit) if initial_risk_per_unit > 0 else None
            )
            edge_manager.record_trade_outcome(
                bot.id, "dip_recovery",
                pnl=pnl_per_unit, win=(pnl_per_unit > 0),
                reward_risk_realized=reward_risk_realized,
                holding_seconds=duration_minutes * 60.0, at=now,
            )

        for pos in positions:
            if current_price >= take_profit:
                _record_exit_outcome()
                return self._dip_recovery_exit_signal(
                    bot, pos, current_price, entry_price, unrealized_pnl_pct,
                    "take profit", cooldown_seconds, now, mk_proposal,
                )
            if current_price <= trailing_stop:
                cd = loss_cooldown_seconds if current_price < entry_price else cooldown_seconds
                _record_exit_outcome()
                return self._dip_recovery_exit_signal(
                    bot, pos, current_price, entry_price, unrealized_pnl_pct,
                    "trailing stop", cd, now, mk_proposal,
                )
            if current_price <= emergency_stop:
                _record_exit_outcome()
                return self._dip_recovery_exit_signal(
                    bot, pos, current_price, entry_price, unrealized_pnl_pct,
                    "emergency stop", loss_cooldown_seconds, now, mk_proposal,
                )
            if duration_minutes >= max_duration_min:
                cd = loss_cooldown_seconds if unrealized_pnl_pct < 0 else cooldown_seconds
                _record_exit_outcome()
                return self._dip_recovery_exit_signal(
                    bot, pos, current_price, entry_price, unrealized_pnl_pct,
                    "max duration", cd, now, mk_proposal,
                )

        self._dip_recovery_states[bot.id] = state
        hold_proposal = mk_proposal(
            direction=Direction.HOLD, execution_intent=ExecutionIntent.HOLD_POSITION,
            evidence_name="Position intact", evidence_value=1.0,
            evidence_reason=(
                "no exit condition (take-profit, trailing stop, emergency stop, max "
                "duration) triggered this cycle"
            ),
            assumptions=(
                "no new low since entry and the confirmed reversal is not reversed",
            ),
        )
        adapter_signal = StandaloneAdapter.to_trade_signal(hold_proposal)
        if adapter_signal is not None:
            return adapter_signal
        return TradeSignal(
            action="hold", amount=0,
            reason=(
                f"Dip Recovery: Holding (PnL {unrealized_pnl_pct:+.2f}%, "
                f"trailing stop ${trailing_stop:.4f}, take profit ${take_profit:.4f})"
            ),
        )

    async def _strategy_auto(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Auto Mode - Risk-first, regime-aware, performance-adaptive strategy allocator.

        This is a POLICY ENGINE, not a trading strategy. It selects, switches, and
        force-exits strategies based on:
        1. Market regime compatibility
        2. Per-strategy performance tracking
        3. Risk-adjusted dynamic priority scoring
        4. Capital preservation bias

        MANDATORY: Uses 60-second bar system only. All regime detection and performance
        metrics operate on completed bar closes only.

        Parameters:
            min_switch_interval_minutes: Minimum time between strategy switches (default: 15)
            bar_interval_seconds: Bar aggregation interval (default: 60)
            cooldown_hours_default: Default cooldown after failure (default: 6)
            max_failures_before_blacklist: Hard stop threshold (default: 3)
            performance_window_bars: Rolling window for PnL calculation (default: 20)

        DEPRECATED parameters (backward compatibility only, ignored):
            factor_precedence: Legacy parameter (ignored)
            disabled_factors: Legacy parameter (ignored)
            switch_threshold: Legacy parameter (ignored)
        """
        # === PARAMETER EXTRACTION ===
        min_switch_interval = params.get("min_switch_interval_minutes", 15)
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        cooldown_hours_default = params.get("cooldown_hours_default", 6)
        max_failures_before_blacklist = params.get("max_failures_before_blacklist", 3)
        performance_window_bars = params.get("performance_window_bars", 20)

        # === STATE INITIALIZATION ===
        auto_state = self._get_auto_state(bot.id)
        now = self.clock.now()

        if "current_strategy" not in auto_state:
            auto_state["current_strategy"] = "dca_accumulator"
            auto_state["last_switch_time"] = None
            auto_state["last_bar_close_time"] = None
            auto_state["current_bar"] = {"open": current_price, "high": current_price, "low": current_price, "close": current_price}
            auto_state["bar_history"] = []  # List of completed bar close prices
            auto_state["current_regime"] = None
            auto_state["regime_change_count"] = 0
            auto_state["strategy_metrics"] = {}  # Per-strategy performance tracking

            # Load existing strategy metrics from database
            db_metrics = await self._load_strategy_metrics_from_db(bot.id, session)
            if db_metrics:
                auto_state["strategy_metrics"] = db_metrics
                logger.info(f"Bot {bot.id}: Loaded {len(db_metrics)} strategy metrics from database")

            self._save_auto_state(bot.id, auto_state)

        # === BAR AGGREGATION (60-second bars) ===
        # Update current bar high/low
        current_bar = auto_state["current_bar"]
        current_bar["high"] = max(current_bar["high"], current_price)
        current_bar["low"] = min(current_bar["low"], current_price)
        current_bar["close"] = current_price

        # Check if bar should close
        last_bar_close = auto_state.get("last_bar_close_time")
        bar_closed = False

        if last_bar_close is None:
            # First bar, initialize
            auto_state["last_bar_close_time"] = now.isoformat()
            bar_closed = False
        else:
            last_bar_time = datetime.fromisoformat(last_bar_close)
            time_since_bar = (now - last_bar_time).total_seconds()

            if time_since_bar >= bar_interval_seconds:
                # Bar closed, append to history
                auto_state["bar_history"].append({
                    "timestamp": now.isoformat(),
                    "close": current_bar["close"],
                    "high": current_bar["high"],
                    "low": current_bar["low"],
                    "open": current_bar["open"]
                })

                # Keep last 100 bars
                auto_state["bar_history"] = auto_state["bar_history"][-100:]

                # Reset current bar
                auto_state["current_bar"] = {
                    "open": current_price,
                    "high": current_price,
                    "low": current_price,
                    "close": current_price
                }
                auto_state["last_bar_close_time"] = now.isoformat()
                bar_closed = True

        # === REGIME DETECTION (ON BAR CLOSE ONLY) ===
        if bar_closed and len(auto_state["bar_history"]) >= 20:
            previous_regime = auto_state.get("current_regime")
            current_regime = self._detect_market_regime_bar_based(
                auto_state["bar_history"],
                previous_regime
            )

            # Check if regime actually changed
            if previous_regime:
                prev_trend = previous_regime.get("trend_state")
                prev_vol = previous_regime.get("volatility_state")
                prev_liq = previous_regime.get("liquidity_state")

                curr_trend = current_regime.get("trend_state")
                curr_vol = current_regime.get("volatility_state")
                curr_liq = current_regime.get("liquidity_state")

                if prev_trend != curr_trend or prev_vol != curr_vol or prev_liq != curr_liq:
                    auto_state["regime_change_count"] = auto_state.get("regime_change_count", 0) + 1
                    logger.info(
                        f"Bot {bot.id}: Auto Mode regime change detected - "
                        f"trend:{prev_trend}→{curr_trend}, vol:{prev_vol}→{curr_vol}, liq:{prev_liq}→{curr_liq} "
                        f"(total changes: {auto_state['regime_change_count']})"
                    )

            auto_state["current_regime"] = current_regime
        elif not auto_state.get("current_regime"):
            # No regime yet, use neutral
            auto_state["current_regime"] = {
                "trend_state": "flat",
                "volatility_state": "medium",
                "liquidity_state": "normal"
            }

        current_regime = auto_state["current_regime"]

        # === UPDATE PERFORMANCE METRICS (ON BAR CLOSE ONLY) ===
        if bar_closed:
            await self._update_strategy_performance_metrics(
                bot.id,
                auto_state,
                session,
                performance_window_bars
            )

        # === STRATEGY ELIGIBILITY FILTERING (HARD GATE) ===
        capabilities = self._get_strategy_capabilities()
        strategy_metrics = auto_state.get("strategy_metrics", {})

        # Create strategy capacity service for capacity checks
        strategy_capacity = StrategyCapacityService(session)

        eligible_strategies = []
        ineligible_reasons = {}

        for strategy_name, caps in capabilities.items():
            # Check 1: Regime, cooldown, blacklist, kill switch
            is_eligible, reason = self._is_strategy_eligible(
                strategy_name,
                caps,
                current_regime,
                strategy_metrics.get(strategy_name, {}),
                now,
                max_failures_before_blacklist
            )

            if not is_eligible:
                ineligible_reasons[strategy_name] = reason
                continue

            # Check 2: Strategy capacity limits (NEW)
            is_at_capacity, capacity_reason = await strategy_capacity.is_strategy_at_capacity(
                strategy_name,
                owner_id=None,  # TODO: Add owner_id when Bot model has it
            )

            if is_at_capacity:
                ineligible_reasons[strategy_name] = f"strategy capacity: {capacity_reason}"
                continue

            # All checks passed
            eligible_strategies.append(strategy_name)

        # Fallback to dca_accumulator if no strategies eligible
        if not eligible_strategies:
            logger.warning(
                f"Bot {bot.id}: Auto Mode - no eligible strategies, forcing dca_accumulator. "
                f"Ineligible reasons: {ineligible_reasons}"
            )
            eligible_strategies = ["dca_accumulator"]

        # === DYNAMIC PRIORITY SCORING ===
        bar_history = auto_state.get("bar_history", [])
        scored_strategies = []
        for strategy_name in eligible_strategies:
            caps = capabilities[strategy_name]
            metrics = strategy_metrics.get(strategy_name, {})
            score_detail = self._score_strategy(
                strategy_name, caps, metrics,
                bar_history=bar_history,
                current_price=current_price,
                current_regime=current_regime,
            )
            scored_strategies.append((strategy_name, score_detail["final"], caps, metrics, score_detail))

        # Sort by final score (descending)
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        # === POSITION CHECK ===
        # Needed to separate entry eligibility (should Auto allocate NEW capital
        # to this strategy) from exit management (should an OPEN position be
        # closed). These are different questions - see FORCE-EXIT CHECK below.
        open_positions = await self._get_bot_positions(bot.id, session)
        has_open_position = len(open_positions) > 0

        # === OWNERSHIP SELF-HEAL ===
        # auto_state["current_strategy"] is a mutable, bot-level pointer. The
        # persisted Position.owning_strategy (set once, at the moment the
        # position was opened - see _resolve_owning_strategy /
        # _open_or_add_position) is the source of truth for which strategy is
        # actually managing an open position. If the two ever disagree (e.g. a
        # crash between opening the position and the next state checkpoint,
        # or the in-memory pointer drifting for any other reason), the
        # persisted fact wins: correct the pointer rather than let dispatch
        # follow a value that does not match what actually opened the trade.
        # NULL owning_strategy (a position opened before this field existed)
        # is treated as unowned, not as a mismatch.
        owning_strategy = None
        if has_open_position:
            owned_position = next(
                (
                    p for p in open_positions
                    if isinstance(getattr(p, "owning_strategy", None), str)
                    and getattr(p, "owning_strategy")
                ),
                None,
            )
            if owned_position is not None:
                owning_strategy = owned_position.owning_strategy
                if owning_strategy != auto_state["current_strategy"]:
                    logger.warning(
                        f"Bot {bot.id}: Auto Mode ownership mismatch - in-memory "
                        f"pointer {auto_state['current_strategy']!r} != persisted "
                        f"position owner {owning_strategy!r}; correcting to the "
                        f"persisted owner (self-heal)."
                    )
                    auto_state["current_strategy"] = owning_strategy

        # === FORCE-EXIT CHECK ===
        # _is_strategy_eligible() is an ENTRY ALLOCATION GATE ONLY (regime match,
        # cooldown, blacklist, kill switch) - it answers "should Auto put new
        # capital into this strategy right now?", not "should an existing
        # position be closed?" A strategy that entered correctly owns its own
        # exit rules (trailing stop, target reached, trend break, failed
        # breakout, time stop, strategy-specific invalidation) and must be
        # allowed to finish its trade even after the market moves out of its
        # entry regime. Auto Mode used to market-sell ("FORCE EXIT") the
        # instant the current strategy's entry regime rotated out, closing
        # positions before their own exit logic ever ran.
        current_strategy = auto_state["current_strategy"]
        current_ineligible = current_strategy not in eligible_strategies
        pinned_for_exit = current_ineligible and has_open_position

        if pinned_for_exit:
            # Keep running the current strategy's own executor below so it
            # manages the open position with its own exit rules. Losing entry
            # eligibility because the market regime moved on is normal and is
            # NOT a strategy failure - do not record a failure, cooldown, or
            # blacklist for it (see Fix 2 / _record_strategy_failure).
            ineligible_reason = ineligible_reasons.get(current_strategy, "unknown")
            logger.info(
                f"Bot {bot.id}: Auto Mode - {current_strategy} lost entry eligibility "
                f"({ineligible_reason}) but has an open position; delegating to its own "
                f"exit rules instead of force-selling."
            )
        elif current_ineligible:
            logger.info(
                f"Bot {bot.id}: Auto Mode - {current_strategy} lost entry eligibility "
                f"({ineligible_reasons.get(current_strategy, 'unknown')}); no open position "
                f"to protect, reallocating."
            )

        # === STRATEGY SELECTION WITH HYSTERESIS ===
        # Switch only when the new best score beats the current strategy by at
        # least the hysteresis margin.  This prevents oscillation when two
        # strategies have similar opportunity scores.
        _HYSTERESIS = params.get("switch_hysteresis", 1.0)

        best_strategy, best_score, best_caps, best_metrics, best_detail = scored_strategies[0]
        selected_strategy = current_strategy
        should_switch = False
        switch_reason = ""

        if pinned_for_exit:
            # Do not reallocate while an ineligible strategy still holds a
            # position - it keeps running (and executing below) until its own
            # exit rules close it out.
            switch_reason = (
                f"{current_strategy} not entry-eligible but holding an open position; "
                f"pinned until its own exit rules close it"
            )
        elif current_ineligible:
            # No open position to protect - free to reallocate immediately.
            selected_strategy = best_strategy
            should_switch = True
            switch_reason = f"current strategy ineligible, switching to {best_strategy}"
        elif has_open_position:
            # Bug fix: current_strategy is still entry-eligible (regime still
            # matches) but a competitor may score higher. Ownership of an open
            # position is not up for re-auction just because the market moved
            # - the strategy that opened it keeps it, and only its own exit
            # rules can close it, until it does. Without this branch the
            # hysteresis/score comparison below would happily switch
            # current_strategy away while a position was open, and the newly
            # selected strategy's executor would then run against a position
            # it never opened (this was the "Strategy A opens, Strategy B
            # closes" defect).
            switch_reason = (
                f"{current_strategy} owns an open position; holding regardless "
                f"of competing scores until it closes"
            )
        elif best_strategy != current_strategy:
            current_score = next((s[1] for s in scored_strategies if s[0] == current_strategy), 0.0)

            if best_score > current_score + _HYSTERESIS:
                # Check min switch interval (prevents flapping)
                last_switch = auto_state.get("last_switch_time")
                if last_switch:
                    time_since_switch = (now - datetime.fromisoformat(last_switch)).total_seconds() / 60
                    if time_since_switch < min_switch_interval:
                        switch_reason = (
                            f"{best_strategy} scores higher ({best_score:.2f} vs {current_score:.2f}) "
                            f"but too soon to switch ({time_since_switch:.1f}m since last)"
                        )
                    else:
                        selected_strategy = best_strategy
                        should_switch = True
                        switch_reason = (
                            f"{best_strategy} score {best_score:.2f} > {current_strategy} "
                            f"{current_score:.2f} + hysteresis {_HYSTERESIS}"
                        )
                else:
                    selected_strategy = best_strategy
                    should_switch = True
                    switch_reason = f"{best_strategy} score {best_score:.2f} dominates (first switch)"
            else:
                switch_reason = (
                    f"{current_strategy} retained (margin {best_score - current_score:.2f} "
                    f"< hysteresis {_HYSTERESIS})"
                )
        else:
            switch_reason = f"{current_strategy} is highest-scoring eligible strategy ({best_score:.2f})"

        # === STRATEGY SWITCHING ===
        if should_switch:
            logger.info(
                f"Bot {bot.id}: Auto Mode SWITCHING {current_strategy} → {selected_strategy} | {switch_reason}"
            )
            auto_state["current_strategy"] = selected_strategy
            auto_state["last_switch_time"] = now.isoformat()
            current_strategy = selected_strategy
            if current_strategy not in strategy_metrics:
                strategy_metrics[current_strategy] = {}
            auto_state["strategy_metrics"] = strategy_metrics
        else:
            logger.debug(f"Bot {bot.id}: Auto Mode HOLDING {current_strategy} | {switch_reason}")

        # Persist score breakdown for observability
        auto_state["last_scores"] = {
            name: detail for name, _, _, _, detail in scored_strategies
        }

        # === DECISION LOGGING ===
        self._log_auto_mode_decision(
            bot.id,
            current_regime,
            eligible_strategies,
            scored_strategies,
            current_strategy,
            should_switch,
            switch_reason,
            ineligible_reasons,
            strategy_metrics
        )

        # Save updated state
        self._save_auto_state(bot.id, auto_state)

        # --- Structured decision explanation: full Auto Mode scoring ---------
        # Exposes every strategy's opportunity / performance / risk / final
        # score and the selected one. The selected sub-strategy then appends its
        # OWN numeric gates to this same builder below, so the explanation shows
        # both WHY this strategy was chosen and WHY it did/didn't trade.
        exp = self._explain(bot.id)
        regime_str = (
            f"{current_regime.get('trend_state')}/"
            f"{current_regime.get('volatility_state')}/"
            f"{current_regime.get('liquidity_state')}"
        )
        for _name, _final, _caps2, _metrics2, _detail in scored_strategies:
            exp.candidate({
                "strategy": _name,
                "opportunity": _detail.get("opportunity", 0.0),
                "performance": _detail.get("performance", 0.0),
                "risk": _detail.get("risk_penalty", 0.0),
                "final": _final,
                "eligible": True,
                "selected": _name == current_strategy,
            })
        for _name, _reason in ineligible_reasons.items():
            exp.candidate({
                "strategy": _name,
                "opportunity": None, "performance": None, "risk": None, "final": None,
                "eligible": False, "selected": _name == current_strategy, "reason": _reason,
            })
        exp.select(current_strategy)
        exp.update({
            "regime": regime_str,
            "selected_strategy": current_strategy,
            "switch_reason": switch_reason,
            "eligible_count": len(eligible_strategies),
            "pinned_for_exit": pinned_for_exit,
        })
        exp.check(
            "Strategy selected", current_strategy,
            "highest final score among eligible", True, detail=switch_reason,
        )
        # ---------------------------------------------------------------------
        # No force-exit branch here anymore: the selected strategy's own
        # executor always runs below and sets its own state (including exit
        # states like WAITING_EXIT/LONG_OPEN when pinned_for_exit is true).

        # === STRATEGY EXECUTION ===
        strategy_executor = self._get_strategy_executor(current_strategy)

        if not strategy_executor:
            logger.warning(f"Bot {bot.id}: Auto Mode - unknown strategy {current_strategy}")
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Auto Mode: Invalid strategy {current_strategy}"
            )

        # Execute the selected strategy. Mark this call as coming through
        # Auto's own supervised dispatch - as opposed to some future direct/
        # unsupervised call - so a sub-strategy can tell the difference from
        # bot.strategy alone. bot.strategy reads "auto_mode" for every
        # Auto-managed bot no matter which sub-strategy is currently
        # selected, so it cannot answer "am I being run under supervision?"
        # by itself (see _strategy_dca's own guard). Copied, not mutated, so
        # the caller's params dict is never altered.
        dispatched_params = dict(params)
        dispatched_params["_invoked_by_auto"] = True
        signal = await strategy_executor(bot, current_price, dispatched_params, session)

        if signal:
            # Add auto mode context to reason
            regime_str = f"{current_regime['trend_state']}/{current_regime['volatility_state']}/{current_regime['liquidity_state']}"
            signal.reason = f"[Auto:{current_strategy}|{regime_str}] {signal.reason}"

            # Record last trade time for inactivity penalty tracking
            if signal.action in ("buy", "sell"):
                if current_strategy not in strategy_metrics:
                    strategy_metrics[current_strategy] = {}
                strategy_metrics[current_strategy]["last_trade_time"] = now.isoformat()
                auto_state["strategy_metrics"] = strategy_metrics
                self._save_auto_state(bot.id, auto_state)

        return signal

    def _classify_trend_state(self, ema_short: list, ema_long: list) -> str:
        """Classify trend_state ("up"/"down"/"flat") from EMA-20 vs EMA-50 using
        both a fast and a medium lookback slope of the short EMA.

        A fast-only slope (few bars back) only sees short-term momentum, so a
        slow but sustained move - a multi-hour grind that never produces a
        sharp short-term kink - is invisible to it and gets classified "flat"
        even though it is a real, persistent trend. The medium slope looks
        further back along the same EMA series so a sustained move is caught
        even when every short slice looks flat, while genuine chop still nets
        out near zero over the medium window and stays "flat" too. Both
        windows are expressed in bar counts, not fixed durations, so this
        scales with whatever bar interval the caller is aggregating at rather
        than being tuned to one asset's typical move size or a fixed calendar
        horizon like "24h".
        """
        n = len(ema_short)
        ema_now = ema_short[-1]
        ema_long_now = ema_long[-1]

        FAST_LOOKBACK = 5
        FAST_THRESHOLD_PCT = 0.5
        MEDIUM_LOOKBACK = min(n - 1, 30)
        MEDIUM_THRESHOLD_PCT = 1.0

        fast_prev = ema_short[-FAST_LOOKBACK] if n >= FAST_LOOKBACK else ema_short[0]
        fast_slope_pct = (
            (ema_now - fast_prev) / fast_prev * 100 if fast_prev > 0 else 0.0
        )

        if MEDIUM_LOOKBACK > FAST_LOOKBACK:
            medium_prev = ema_short[-MEDIUM_LOOKBACK]
            medium_slope_pct = (
                (ema_now - medium_prev) / medium_prev * 100 if medium_prev > 0 else 0.0
            )
        else:
            medium_slope_pct = fast_slope_pct

        is_up = (
            (fast_slope_pct > FAST_THRESHOLD_PCT or medium_slope_pct > MEDIUM_THRESHOLD_PCT)
            and ema_now > ema_long_now
        )
        is_down = (
            (fast_slope_pct < -FAST_THRESHOLD_PCT or medium_slope_pct < -MEDIUM_THRESHOLD_PCT)
            and ema_now < ema_long_now
        )

        if is_up:
            return "up"
        if is_down:
            return "down"
        return "flat"

    def _detect_market_regime(self, price_history: list, current_regime: Optional[dict]) -> dict:
        """Detect market regime from a plain price series (tick/close prices).

        Price-only variant of _detect_market_regime_bar_based for strategies
        that track tick price history instead of OHLC bars. Entries may be
        floats or dicts carrying a "price" (or "close") key; other entries
        are ignored.

        Returns the same shape: trend_state ("up"/"down"/"flat"),
        volatility_state ("low"/"medium"/"high"), liquidity_state (always
        "normal" - not measurable from prices alone), persistence_bars.
        """
        prices = []
        for entry in price_history:
            if isinstance(entry, dict):
                value = entry.get("price", entry.get("close"))
            else:
                value = entry
            if isinstance(value, (int, float)) and value > 0:
                prices.append(float(value))

        neutral = {
            "trend_state": "flat",
            "volatility_state": "medium",
            "liquidity_state": "normal",
            "persistence_bars": 0,
        }
        n = len(prices)
        if n < 20:
            return neutral

        # === TREND STATE (fast + medium EMA slope, same thresholds as bar-based variant) ===
        ema_20 = self._calculate_ema(prices, 20)
        ema_50 = self._calculate_ema(prices, 50) if n >= 50 else ema_20

        trend_state = self._classify_trend_state(ema_20, ema_50)

        # === VOLATILITY STATE (absolute price change as true-range proxy) ===
        changes = [abs(prices[i] - prices[i - 1]) for i in range(max(1, n - 30), n)]
        if changes:
            current_atr = sum(changes[-14:]) / min(14, len(changes[-14:]))
            avg_atr = sum(changes) / len(changes)
            atr_percentile = current_atr / avg_atr if avg_atr > 0 else 1.0
            if atr_percentile < 0.7:
                volatility_state = "low"
            elif atr_percentile > 1.3:
                volatility_state = "high"
            else:
                volatility_state = "medium"
        else:
            volatility_state = "medium"

        return {
            "trend_state": trend_state,
            "volatility_state": volatility_state,
            "liquidity_state": "normal",
            "persistence_bars": 0,
        }

    def _detect_market_regime_bar_based(self, bar_history: list, current_regime: dict) -> dict:
        """Detect current market regime from completed bar closes only.

        MANDATORY: Operates on 60-second bar closes only. No tick data.

        Returns a discrete regime state with:
        - trend_state: "up", "down", or "flat"
        - volatility_state: "low", "medium", or "high"
        - liquidity_state: "low", "normal", or "high"

        Regime changes require persistence over multiple bars to avoid noise.
        """
        if len(bar_history) < 20:
            # Not enough data, return neutral regime
            return {
                "trend_state": "flat",
                "volatility_state": "medium",
                "liquidity_state": "normal",
                "persistence_bars": 0
            }

        # Extract close prices from bars
        closes = [bar["close"] for bar in bar_history]
        n = len(closes)

        # === TREND STATE (fast + medium EMA slope on bar closes) ===
        ema_20 = self._calculate_ema(closes, 20)
        ema_50 = self._calculate_ema(closes, 50) if n >= 50 else ema_20

        new_trend = self._classify_trend_state(ema_20, ema_50)

        # === VOLATILITY STATE (using true range from bars) ===
        # Calculate ATR using actual bar high/low
        atr_values = []
        for i in range(max(1, n - 30), n):
            bar = bar_history[i]
            tr = bar["high"] - bar["low"]
            atr_values.append(tr)

        if atr_values:
            current_atr = sum(atr_values[-14:]) / min(14, len(atr_values[-14:]))
            avg_atr = sum(atr_values) / len(atr_values)
            atr_percentile = current_atr / avg_atr if avg_atr > 0 else 1.0

            if atr_percentile < 0.7:
                new_volatility = "low"
            elif atr_percentile > 1.3:
                new_volatility = "high"
            else:
                new_volatility = "medium"

            # Direction of ATR change (expanding vs contracting vs stable)
            # Compare recent 7-bar ATR against older 7-bar ATR
            if len(atr_values) >= 14:
                recent_half = sum(atr_values[-7:]) / 7
                older_half = sum(atr_values[-14:-7]) / 7
                if older_half > 0:
                    direction_ratio = recent_half / older_half
                    if direction_ratio > 1.15:
                        new_volatility_direction = "expanding"
                    elif direction_ratio < 0.85:
                        new_volatility_direction = "contracting"
                    else:
                        new_volatility_direction = "stable"
                else:
                    new_volatility_direction = "stable"
            else:
                new_volatility_direction = "stable"
        else:
            new_volatility = "medium"
            new_volatility_direction = "stable"

        # === LIQUIDITY STATE (proxy using bar range stability) ===
        recent_ranges = []
        for i in range(max(10, n - 20), n):
            bar = bar_history[i]
            range_val = (bar["high"] - bar["low"]) / bar["close"] if bar["close"] > 0 else 0
            recent_ranges.append(range_val)

        if recent_ranges:
            avg_range = sum(recent_ranges) / len(recent_ranges)
            range_std = (sum((r - avg_range) ** 2 for r in recent_ranges) / len(recent_ranges)) ** 0.5

            if range_std < 0.002:
                new_liquidity = "high"
            elif range_std > 0.005:
                new_liquidity = "low"
            else:
                new_liquidity = "normal"
        else:
            new_liquidity = "normal"

        # === REGIME PERSISTENCE ===
        persistence_required = 3

        if not current_regime:
            return {
                "trend_state": new_trend,
                "volatility_state": new_volatility,
                "volatility_direction": new_volatility_direction,
                "liquidity_state": new_liquidity,
                "persistence_bars": persistence_required
            }

        regime_changed = (
            new_trend != current_regime.get("trend_state") or
            new_volatility != current_regime.get("volatility_state") or
            new_liquidity != current_regime.get("liquidity_state")
        )

        if regime_changed:
            persistence_bars = current_regime.get("persistence_bars", 0) + 1
            if persistence_bars >= persistence_required:
                return {
                    "trend_state": new_trend,
                    "volatility_state": new_volatility,
                    "volatility_direction": new_volatility_direction,
                    "liquidity_state": new_liquidity,
                    "persistence_bars": 0
                }
            else:
                return {
                    "trend_state": current_regime.get("trend_state"),
                    "volatility_state": current_regime.get("volatility_state"),
                    "volatility_direction": current_regime.get("volatility_direction", "stable"),
                    "liquidity_state": current_regime.get("liquidity_state"),
                    "persistence_bars": persistence_bars,
                }
        else:
            return {
                "trend_state": new_trend,
                "volatility_state": new_volatility,
                "volatility_direction": new_volatility_direction,
                "liquidity_state": new_liquidity,
                "persistence_bars": 0
            }

    def _calculate_ema(self, prices: list, period: int) -> list:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices

        multiplier = 2 / (period + 1)
        ema_values = [sum(prices[:period]) / period]  # Start with SMA

        for price in prices[period:]:
            ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)

        return ema_values

    def _is_strategy_eligible(
        self,
        strategy_name: str,
        capabilities: dict,
        current_regime: dict,
        strategy_metrics: dict,
        now: datetime,
        max_failures: int
    ) -> tuple:
        """Check if strategy passes eligibility filter (HARD GATE).

        A strategy is eligible ONLY if ALL conditions hold:
        1. Current regime matches its allowed_regimes
        2. Strategy is not in cooldown
        3. Strategy is not blacklisted
        4. Strategy has not exceeded failure threshold

        Args:
            strategy_name: Name of strategy
            capabilities: Strategy capability dict
            current_regime: Current regime state
            strategy_metrics: Performance metrics for this strategy
            now: Current datetime
            max_failures: Maximum failures before blacklist

        Returns:
            (is_eligible: bool, reason: str)
        """
        # Check 1: Regime compatibility
        allowed = capabilities.get("allowed_regimes", [])
        trend = current_regime.get("trend_state", "flat")
        volatility = current_regime.get("volatility_state", "medium")
        vol_direction = current_regime.get("volatility_direction", "stable")
        liquidity = current_regime.get("liquidity_state", "normal")

        regime_tags = [
            f"trend_{trend}",
            f"volatility_{volatility}",
            f"volatility_{vol_direction}",
            f"liquidity_{liquidity}"
        ]

        if "all" not in allowed:
            matches = [tag for tag in regime_tags if tag in allowed]
            if not matches:
                return False, f"regime mismatch (need {allowed}, got {regime_tags})"

        # Check 2: Cooldown
        cooldown_until = strategy_metrics.get("cooldown_until")
        if cooldown_until:
            cooldown_time = datetime.fromisoformat(cooldown_until)
            if now < cooldown_time:
                remaining = (cooldown_time - now).total_seconds() / 3600
                return False, f"in cooldown ({remaining:.1f}h remaining)"

        # Check 3: Blacklist (exceeds failure threshold)
        failure_count = strategy_metrics.get("failure_count", 0)
        if failure_count >= max_failures:
            return False, f"blacklisted (failures: {failure_count} >= {max_failures})"

        # Check 4: Hard stop check (strategy metrics indicate kill switch)
        if strategy_metrics.get("kill_switch_active"):
            return False, "kill switch active"

        return True, "eligible"

    def _compute_opportunity_score(
        self,
        strategy_name: str,
        bar_history: list,
        current_price: float,
        current_regime: dict,
    ) -> float:
        """Live market opportunity score (0–10) for a strategy.

        Measures how well current market conditions match the strategy's ideal
        setup — independently of historical performance. This is the dominant
        selection factor.

        10 = exceptional setup present right now.
         5 = neutral / eligible but no strong setup.
         0 = setup actively absent.

        Returns 5.0 (neutral) when bar_history is too short to compute signals.
        """
        if len(bar_history) < 20:
            return 5.0

        closes = [b["close"] for b in bar_history]
        highs = [b["high"] for b in bar_history]
        lows = [b["low"] for b in bar_history]
        n = len(closes)

        def _clamp(v: float) -> float:
            return max(0.0, min(10.0, v))

        def _ema(prices: list, period: int) -> list:
            if len(prices) < period:
                return prices[:]
            mult = 2.0 / (period + 1)
            val = sum(prices[:period]) / period
            result = [val]
            for p in prices[period:]:
                val = (p - val) * mult + val
                result.append(val)
            return result

        def _atr_series(h: list, l: list, period: int) -> list:
            trs = [h[i] - l[i] for i in range(len(h))]
            if len(trs) < period:
                return trs
            result = [sum(trs[:period]) / period]
            mult = 2.0 / (period + 1)
            for tr in trs[period:]:
                result.append((tr - result[-1]) * mult + result[-1])
            return result

        def _rsi(prices: list, period: int = 14) -> float:
            if len(prices) < period + 1:
                return 50.0
            gains, losses = [], []
            for i in range(len(prices) - period, len(prices)):
                d = prices[i] - prices[i - 1]
                gains.append(max(d, 0.0))
                losses.append(max(-d, 0.0))
            ag = sum(gains) / period
            al = sum(losses) / period
            if al == 0:
                return 100.0
            return 100.0 - 100.0 / (1.0 + ag / al)

        def _bollinger(prices: list, period: int = 20, nstd: float = 2.0):
            w = prices[-period:]
            mean = sum(w) / len(w)
            var = sum((p - mean) ** 2 for p in w) / len(w)
            std = var ** 0.5
            return mean, mean + nstd * std, mean - nstd * std, std

        if strategy_name == "trend_following":
            ema20 = _ema(closes, 20)
            ema50 = _ema(closes, min(50, n))
            e20, e50 = ema20[-1], ema50[-1]
            # 5-bar slope of EMA-20
            e20_prev = ema20[-min(6, len(ema20))]
            separation_pct = (e20 - e50) / e50 * 100.0 if e50 > 0 else 0.0
            slope_pct = (e20 - e20_prev) / e20_prev * 100.0 if e20_prev > 0 else 0.0
            # separation > 0.5% + positive slope drives score to 10
            # each +0.5% separation ≈ +1.5 pts; each +0.5% slope ≈ +1 pt
            return _clamp(separation_pct * 3.0 + max(0.0, slope_pct) * 2.0 + 2.0)

        elif strategy_name == "mean_reversion":
            mean, upper, lower, std = _bollinger(closes, 20, 2.0)
            if std == 0:
                return 5.0
            z = (current_price - mean) / std        # negative = below mean = long opportunity
            rsi = _rsi(closes)
            # Opportunity when price is extended BELOW mean (fade the drop, buy)
            below_score = _clamp(-z * 2.5 + 5.0)   # z=-2 → 10, z=0 → 5, z=+1 → 2.5
            rsi_score = _clamp((50.0 - rsi) / 5.0 + 5.0)  # RSI=30 → 9, RSI=50 → 5, RSI=70 → 1
            return _clamp(below_score * 0.6 + rsi_score * 0.4)

        elif strategy_name == "adaptive_grid":
            # Grid works best when ATR is stable (low coefficient of variation)
            # and price oscillates moderately inside the range
            atrs = _atr_series(highs, lows, 14)
            if not atrs:
                return 5.0
            atr_mean = sum(atrs) / len(atrs)
            atr_std = (sum((a - atr_mean) ** 2 for a in atrs) / len(atrs)) ** 0.5
            atr_cv = atr_std / atr_mean if atr_mean > 0 else 1.0
            stability_score = _clamp((1.0 - min(atr_cv, 1.0)) * 8.0 + 2.0)

            # Moderate price oscillation = grid fill opportunities
            recent_c = closes[-20:]
            rc_mean = sum(recent_c) / len(recent_c)
            rc_std = (sum((p - rc_mean) ** 2 for p in recent_c) / len(recent_c)) ** 0.5
            oscillation_pct = rc_std / rc_mean * 100.0 if rc_mean > 0 else 0.0
            # 0.3–1.0% oscillation = ideal; above 2% grid may break
            if oscillation_pct <= 1.0:
                osc_score = _clamp(oscillation_pct * 5.0)
            else:
                osc_score = _clamp(10.0 - (oscillation_pct - 1.0) * 4.0)

            return _clamp(stability_score * 0.6 + osc_score * 0.4)

        elif strategy_name == "volatility_breakout":
            # Score = compression quality (tight BB relative to history)
            mean, upper, lower, std = _bollinger(closes, 20, 2.0)
            if mean == 0:
                return 5.0
            bb_width_pct = (upper - lower) / mean * 100.0

            # Rolling BB widths to compute compression percentile
            bb_widths = []
            for i in range(20, n):
                w = closes[i - 20:i]
                m = sum(w) / 20
                s = (sum((p - m) ** 2 for p in w) / 20) ** 0.5
                bb_widths.append(4.0 * s / m * 100.0 if m > 0 else 0.0)

            if not bb_widths:
                return 5.0
            avg_width = sum(bb_widths) / len(bb_widths)
            compression_ratio = bb_width_pct / avg_width if avg_width > 0 else 1.0

            # Tight compression (ratio < 0.5) → high score; already expanding (>1.5) → 0
            compression_score = _clamp((1.0 - compression_ratio) * 10.0 + 5.0)

            # Bonus if ATR just started expanding (breakout building)
            atrs = _atr_series(highs, lows, 7)
            if len(atrs) >= 2:
                atr_accel = atrs[-1] / atrs[-2] if atrs[-2] > 0 else 1.0
                expansion_bonus = _clamp((atr_accel - 1.0) * 20.0)
            else:
                expansion_bonus = 0.0

            return _clamp(compression_score * 0.75 + expansion_bonus * 0.25)

        elif strategy_name == "dca_accumulator":
            # Attractiveness = how far price has fallen from recent high (buy the dip)
            recent_c = closes[-min(30, n):]
            recent_high = max(recent_c)
            drawdown_pct = (recent_high - current_price) / recent_high * 100.0 if recent_high > 0 else 0.0
            # 0% drawdown → 3.0 (DCA is always mildly desirable as fallback)
            # 5% drawdown → ~10 (strong accumulation opportunity)
            return _clamp(drawdown_pct * 1.4 + 3.0)

        elif strategy_name == "dip_recovery":
            # Opportunity = a decline that is LARGE relative to normal volatility
            # AND already showing an early bounce off a local low. Mirrors the
            # live strategy's own adaptive-threshold logic (drop vs ATR%,
            # recovery-from-low vs ATR%, see _strategy_dip_recovery) but is
            # computed here from auto_mode's own bar history rather than the
            # strategy's tick-level state, since dip_recovery may not be the
            # currently-running sub-strategy yet. Delegates the actual score
            # curve to _dip_recovery_score_from_ratios so the math is defined
            # exactly once and shared with the live strategy's own diagnostics.
            atrs = _atr_series(highs, lows, 14)
            atr = atrs[-1] if atrs else 0.0
            atr_pct = (atr / current_price * 100.0) if current_price > 0 else 0.0
            if atr_pct <= 0:
                return 5.0

            recent_c = closes[-min(30, n):]
            recent_high = max(recent_c)
            high_idx = recent_c.index(recent_high)
            after_high = recent_c[high_idx:] or recent_c
            lowest = min(after_high)

            decline_pct = (recent_high - lowest) / recent_high * 100.0 if recent_high > 0 else 0.0
            recovery_pct = (current_price - lowest) / lowest * 100.0 if lowest > 0 else 0.0

            decline_ratio = decline_pct / atr_pct
            recovery_ratio = max(0.0, recovery_pct) / atr_pct
            return self._dip_recovery_score_from_ratios(decline_ratio, recovery_ratio)

        return 5.0  # neutral for unknown strategies

    def _dip_recovery_score_from_ratios(self, decline_ratio: float, recovery_ratio: float) -> float:
        """Dip Recovery opportunity score (0-10) from ATR-normalized ratios.

        decline_ratio: how many ATR%s deep the tracked decline is (depth of
            drop relative to normal volatility for this pair right now).
        recovery_ratio: how many ATR%s the price has already bounced off its
            local low.

        Shared by _compute_opportunity_score's "dip_recovery" branch (fed
        approximate ratios from auto_mode's bar history) AND
        _strategy_dip_recovery's own explanation (fed its exact, precisely
        tracked ratios) - one formula, two callers, so Auto Mode ranking and
        the strategy's own "Opportunity score" diagnostic never disagree on
        what a given setup is worth.

        Low score (no real decline, e.g. sideways market): decline_ratio < 1.
        Low score (already fully recovered / mature uptrend, nothing left to
        catch): recovery_ratio > 3. Otherwise scales with both a real decline
        having occurred AND an early reversal forming - the "sweet spot" this
        strategy targets. A market still actively falling (recovery_ratio ~ 0)
        scores low too: recovery_score is only ~2 until a bounce is underway.
        """
        def _clamp(v: float) -> float:
            return max(0.0, min(10.0, v))

        if decline_ratio < 1.0:
            return _clamp(2.0)
        if recovery_ratio > 3.0:
            return _clamp(4.0)

        decline_score = _clamp(decline_ratio * 2.0)          # ratio 2.5 -> 5, ratio 5 -> 10
        recovery_score = _clamp(recovery_ratio * 3.0 + 2.0)  # ratio 0 -> 2, ratio 1 -> 5, ratio 2.67 -> 10
        return _clamp(decline_score * 0.5 + recovery_score * 0.5)

    def _score_strategy(
        self,
        strategy_name: str,
        capabilities: dict,
        strategy_metrics: dict,
        bar_history: Optional[list] = None,
        current_price: float = 0.0,
        current_regime: Optional[dict] = None,
    ) -> dict:
        """Calculate market-first, performance-adjusted priority for a strategy.

        Formula:
            final_score = (opportunity_score × 3) + performance_score - risk_penalty

        Step 1 — Opportunity score (dominant, 0–10):
            Live market conditions matching the strategy's setup.
            Returns 5.0 when bar_history is unavailable.

        Step 2 — Performance score with confidence weighting:
            confidence = min(1.0, total_trades / 50)
            raw_performance = pnl + win_rate + profit_factor components
            performance_score = raw_performance × confidence

        Step 3 — Risk penalty (structural risks only):
            - Exponential drawdown penalty
            - Failure count penalty
            - Recent exit cooling period
            Inactivity alone is NOT penalised — the opportunity score handles
            whether there is a setup present.

        Returns:
            dict with keys: final, opportunity, performance, confidence, risk_penalty
        """
        bar_history = bar_history or []
        current_regime = current_regime or {"trend_state": "flat", "volatility_state": "medium",
                                            "volatility_direction": "stable", "liquidity_state": "normal"}

        # --- Step 1: Opportunity score ---
        opportunity = self._compute_opportunity_score(
            strategy_name, bar_history, current_price, current_regime
        )

        # --- Step 2: Performance score with confidence weighting ---
        total_trades = strategy_metrics.get("total_trades", 0)
        confidence = min(1.0, total_trades / 50.0)

        recent_pnl_pct = strategy_metrics.get("recent_pnl_pct", 0.0)
        win_rate = strategy_metrics.get("win_rate", 0.0)
        profit_factor = strategy_metrics.get("profit_factor", 0.0)

        raw_perf = recent_pnl_pct / 5.0   # ±20% PnL ≈ ±4 before confidence

        if total_trades >= 3:
            raw_perf += (win_rate - 0.5) * 2.5   # 50% WR = neutral; +10% WR = +0.25
            if profit_factor > 1.0:
                raw_perf += min(0.5, (profit_factor - 1.0) * 0.5)
            elif 0.0 < profit_factor < 1.0:
                raw_perf -= min(0.5, (1.0 - profit_factor) * 0.5)

        if total_trades > 0:
            raw_perf += min(0.5, total_trades / 40.0)   # activity evidence bonus

        raw_perf = min(4.0, max(-4.0, raw_perf))
        performance = raw_perf * confidence

        # --- Step 3: Risk penalty (structural only) ---
        risk_penalty = 0.0

        # Drawdown: exponential (5%→1, 10%→4, 15%→9, 20%→16)
        max_drawdown_pct = strategy_metrics.get("max_drawdown_pct", 0.0)
        if max_drawdown_pct > 0:
            risk_penalty += (max_drawdown_pct / 5.0) ** 2

        # Hard failure signals (kill switches, stop-loss trips)
        failure_count = strategy_metrics.get("failure_count", 0)
        risk_penalty += failure_count * 5.0

        # Recent exit cooling (prevents immediate re-entry after forced exit)
        last_exit_time = strategy_metrics.get("last_exit_time")
        if last_exit_time:
            try:
                exit_time = datetime.fromisoformat(last_exit_time)
                hours_since_exit = (self.clock.now() - exit_time).total_seconds() / 3600
                if hours_since_exit < 1.0:
                    risk_penalty += 3.0
                elif hours_since_exit < 6.0:
                    risk_penalty += 1.0
            except (ValueError, TypeError):
                pass

        # --- Final score ---
        final = (opportunity * 3.0) + performance - risk_penalty

        return {
            "final": final,
            "opportunity": opportunity,
            "performance": performance,
            "confidence": confidence,
            "risk_penalty": risk_penalty,
        }

    async def _update_strategy_performance_metrics(
        self,
        bot_id: int,
        auto_state: dict,
        session: AsyncSession,
        performance_window: int
    ) -> None:
        """Update per-strategy performance metrics on bar close.

        Source: realized_gains table — FIFO-matched P&L written by the
        accounting system on every SELL trade.  This is the only authoritative
        source for closed-trade profit/loss; running_balance_after snapshots are
        NOT used because they conflate cash-flow direction (buy = cash out) with
        actual gain/loss.

        For each sub-strategy computes from the last ``performance_window``
        closed sell cycles:
          - total_trades    count of closed sell events
          - winning_trades  count where gain_loss > 0
          - losing_trades   count where gain_loss < 0
          - realized_pnl_usd  sum of gain_loss USD
          - profit_factor   gross_profit / gross_loss  (0.0 if no losses yet)
          - win_rate        winning_trades / (winning_trades + losing_trades)
          - recent_pnl_pct  realized_pnl_usd / total_cost_basis × 100
          - max_drawdown_pct  worst rolling drawdown seen so far (monotonic)
          - last_trade_time   sell_date of most recent closed cycle

        Strategy attribution for auto_mode bots:
          1. Parse "[Auto:strategy_name|regime]" prefix in orders.reason.
          2. Fall back to Trade.strategy_used if no [Auto:] prefix.
        """
        from ..models import RealizedGain

        # Fetch the last N×3 closed sell cycles for this bot, oldest first.
        # The 3× headroom ensures enough rows per sub-strategy.
        fetch_limit = performance_window * 3

        query = (
            select(RealizedGain, Trade.strategy_used, Order.reason, RealizedGain.sell_date)
            .join(Trade, RealizedGain.sell_trade_id == Trade.id)
            .outerjoin(Order, Trade.order_id == Order.id)
            .where(Trade.bot_id == bot_id)
            .order_by(RealizedGain.sell_date.asc())
            .limit(fetch_limit)
        )

        result = await session.execute(query)
        rows = result.all()

        if not rows:
            return

        # Group closed-cycle records by sub-strategy
        strategy_cycles: dict = {}

        for rg, strategy_used, reason, sell_date in rows:
            # Attribute to sub-strategy
            sub_strategy = strategy_used or ""
            if reason and reason.startswith("[Auto:"):
                # "[Auto:mean_reversion|flat/medium/high] ..."
                try:
                    sub_strategy = reason.split("|")[0].replace("[Auto:", "").strip()
                except Exception:
                    pass

            if not sub_strategy:
                continue
            if sub_strategy not in strategy_cycles:
                strategy_cycles[sub_strategy] = []
            strategy_cycles[sub_strategy].append({
                "gain_loss": rg.gain_loss,
                "cost_basis": rg.cost_basis,
                "sell_date": sell_date,
            })

        strategy_metrics = auto_state.get("strategy_metrics", {})

        for strategy_name, cycles in strategy_cycles.items():
            if strategy_name not in strategy_metrics:
                strategy_metrics[strategy_name] = {
                    "recent_pnl_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "realized_pnl_usd": 0.0,
                    "profit_factor": 0.0,
                    "win_rate": 0.0,
                    "last_trade_time": None,
                    "failure_count": 0,
                    "last_exit_time": None,
                    "cooldown_until": None,
                }

            metrics = strategy_metrics[strategy_name]
            window = cycles[-performance_window:]

            # Last trade time
            last_sell = window[-1]["sell_date"]
            if last_sell:
                ts = last_sell.isoformat() if hasattr(last_sell, "isoformat") else str(last_sell)
                metrics["last_trade_time"] = ts

            metrics["total_trades"] = len(window)

            gross_profit = 0.0
            gross_loss = 0.0
            wins = 0
            losses = 0
            total_cost_basis = 0.0
            realized_pnl = 0.0

            for cycle in window:
                gl = cycle["gain_loss"]
                cb = cycle["cost_basis"] or 0.0
                realized_pnl += gl
                total_cost_basis += cb
                if gl > 0:
                    wins += 1
                    gross_profit += gl
                elif gl < 0:
                    losses += 1
                    gross_loss += abs(gl)

            metrics["winning_trades"] = wins
            metrics["losing_trades"] = losses
            metrics["realized_pnl_usd"] = realized_pnl
            metrics["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0
                else (1.0 if gross_profit > 0 else 0.0)
            )
            completed = wins + losses
            metrics["win_rate"] = (wins / completed) if completed > 0 else 0.0

            # PnL % = realized_pnl / total_cost_basis deployed (avoids balance snapshot bias)
            if total_cost_basis > 0:
                pnl_pct = (realized_pnl / total_cost_basis) * 100
                metrics["recent_pnl_pct"] = pnl_pct
                if pnl_pct < 0:
                    metrics["max_drawdown_pct"] = max(
                        metrics.get("max_drawdown_pct", 0.0), abs(pnl_pct)
                    )

        auto_state["strategy_metrics"] = strategy_metrics

        await self._save_all_strategy_metrics_to_db(
            bot_id=bot_id,
            all_metrics=strategy_metrics,
            session=session
        )

    async def _record_strategy_failure(
        self,
        bot_id: int,
        auto_state: dict,
        strategy_name: str,
        reason: str,
        now: datetime,
        cooldown_hours: float,
        session: AsyncSession
    ) -> None:
        """Record strategy failure and apply cooldown.

        Args:
            bot_id: Bot ID
            auto_state: Auto mode state
            strategy_name: Name of failed strategy
            reason: Failure reason
            now: Current datetime
            cooldown_hours: Hours to apply cooldown
            session: Database session
        """
        strategy_metrics = auto_state.get("strategy_metrics", {})

        if strategy_name not in strategy_metrics:
            strategy_metrics[strategy_name] = {
                "recent_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "failure_count": 0,
                "last_exit_time": None,
                "cooldown_until": None
            }

        metrics = strategy_metrics[strategy_name]
        metrics["failure_count"] = metrics.get("failure_count", 0) + 1
        metrics["last_exit_time"] = now.isoformat()
        metrics["cooldown_until"] = (now + timedelta(hours=cooldown_hours)).isoformat()

        logger.warning(
            f"Strategy {strategy_name} failure recorded: {reason}. "
            f"Total failures: {metrics['failure_count']}, cooldown until {metrics['cooldown_until']}"
        )

        auto_state["strategy_metrics"] = strategy_metrics

        # Persist to database
        await self._save_strategy_metrics_to_db(
            bot_id=bot_id,
            strategy_name=strategy_name,
            metrics=metrics,
            session=session
        )

    async def _load_strategy_metrics_from_db(
        self,
        bot_id: int,
        session: AsyncSession
    ) -> dict:
        """Load strategy performance metrics from database.

        Args:
            bot_id: Bot ID
            session: Database session

        Returns:
            Dictionary of strategy metrics in the format:
            {strategy_name: {recent_pnl_pct, max_drawdown_pct, failure_count, ...}}
        """
        from app.models.strategy_performance import StrategyPerformanceMetrics

        query = select(StrategyPerformanceMetrics).where(
            StrategyPerformanceMetrics.bot_id == bot_id
        )
        result = await session.execute(query)
        rows = result.scalars().all()

        metrics = {}
        for row in rows:
            metrics[row.strategy_name] = row.to_dict()

        logger.debug(f"Loaded {len(metrics)} strategy metrics from DB for bot {bot_id}")
        return metrics

    async def _save_strategy_metrics_to_db(
        self,
        bot_id: int,
        strategy_name: str,
        metrics: dict,
        session: AsyncSession
    ) -> None:
        """Save or update strategy performance metrics to database (UPSERT).

        Args:
            bot_id: Bot ID
            strategy_name: Name of strategy
            metrics: Metrics dictionary
            session: Database session
        """
        from app.models.strategy_performance import StrategyPerformanceMetrics

        # Check if record exists
        query = select(StrategyPerformanceMetrics).where(
            StrategyPerformanceMetrics.bot_id == bot_id,
            StrategyPerformanceMetrics.strategy_name == strategy_name
        )
        result = await session.execute(query)
        existing = result.scalar_one_or_none()

        def _parse_dt(val):
            if val is None:
                return None
            return datetime.fromisoformat(val) if isinstance(val, str) else val

        if existing:
            existing.recent_pnl_pct = metrics.get("recent_pnl_pct", 0.0)
            existing.max_drawdown_pct = metrics.get("max_drawdown_pct", 0.0)
            existing.total_trades = metrics.get("total_trades", 0)
            existing.winning_trades = metrics.get("winning_trades", 0)
            existing.losing_trades = metrics.get("losing_trades", 0)
            existing.realized_pnl_usd = metrics.get("realized_pnl_usd", 0.0)
            existing.profit_factor = metrics.get("profit_factor", 0.0)
            existing.win_rate = metrics.get("win_rate", 0.0)
            existing.last_trade_time = _parse_dt(metrics.get("last_trade_time"))
            existing.failure_count = metrics.get("failure_count", 0)
            existing.last_exit_time = _parse_dt(metrics.get("last_exit_time"))
            existing.cooldown_until = _parse_dt(metrics.get("cooldown_until"))
            existing.last_updated = self.clock.now()
        else:
            new_metrics = StrategyPerformanceMetrics.from_dict(
                bot_id=bot_id,
                strategy_name=strategy_name,
                data=metrics
            )
            session.add(new_metrics)

        await session.commit()
        logger.debug(f"Saved strategy metrics for bot {bot_id}, strategy {strategy_name}")

    async def _save_all_strategy_metrics_to_db(
        self,
        bot_id: int,
        all_metrics: dict,
        session: AsyncSession
    ) -> None:
        """Save all strategy metrics for a bot to database.

        Args:
            bot_id: Bot ID
            all_metrics: Dictionary of {strategy_name: metrics}
            session: Database session
        """
        for strategy_name, metrics in all_metrics.items():
            await self._save_strategy_metrics_to_db(
                bot_id=bot_id,
                strategy_name=strategy_name,
                metrics=metrics,
                session=session
            )

    def _log_auto_mode_decision(
        self,
        bot_id: int,
        current_regime: dict,
        eligible_strategies: list,
        scored_strategies: list,
        selected_strategy: str,
        should_switch: bool,
        switch_reason: str,
        ineligible_reasons: dict,
        strategy_metrics: dict
    ) -> None:
        """Log comprehensive auto mode decision for auditing.

        MANDATORY: Every decision must log:
        - Current regime
        - Eligible strategies
        - Scores per strategy
        - Switch / no-switch reason
        - Force-exit reason (if any)
        - Cooldown / blacklist status

        Args:
            bot_id: Bot ID
            current_regime: Current regime state
            eligible_strategies: List of eligible strategy names
            scored_strategies: List of (name, score, caps, metrics) tuples
            selected_strategy: Selected strategy
            should_switch: Whether switching occurred
            switch_reason: Reason for switch/no-switch
            ineligible_reasons: Map of ineligible strategies to reasons
            strategy_metrics: All strategy metrics
        """
        regime_str = f"{current_regime['trend_state']}/{current_regime['volatility_state']}/{current_regime['liquidity_state']}"

        log_msg = f"\n{'=' * 80}\n"
        log_msg += f"Bot {bot_id}: Auto Mode Decision\n"
        log_msg += f"{'-' * 80}\n"
        log_msg += f"Regime: {regime_str}\n"
        log_msg += f"Selected: {selected_strategy} (switched: {should_switch})\n"
        log_msg += f"Reason: {switch_reason}\n\n"

        log_msg += (
            f"{'Strategy':<22} {'Final':>6} {'Oppty':>6} {'Perf':>6} "
            f"{'Conf':>5} {'Risk':>5} {'Trades':>6}\n"
        )
        log_msg += f"{'-' * 60}\n"
        for entry in scored_strategies:
            name, score, caps, metrics, detail = entry
            trades = metrics.get("total_trades", 0)
            marker = " ◀" if name == selected_strategy else ""
            log_msg += (
                f"  {name:<20} {score:>6.2f} {detail['opportunity']:>6.2f} "
                f"{detail['performance']:>6.2f} {detail['confidence']:>5.2f} "
                f"{detail['risk_penalty']:>5.2f} {trades:>6}{marker}\n"
            )

        if ineligible_reasons:
            log_msg += f"\nIneligible ({len(ineligible_reasons)}):\n"
            for name, reason in ineligible_reasons.items():
                log_msg += f"  - {name:<22} {reason}\n"

        log_msg += f"{'=' * 80}\n"

        logger.info(log_msg)

    def _get_strategy_capabilities(self) -> dict:
        """Get strategy capability declarations for regime-based selection.

        Returns a map of strategy_name -> capability metadata.
        Each strategy declares:
        - allowed_regimes: List of regime patterns this strategy handles well
        - forbidden_regimes: Optional list of patterns to avoid (not used currently)
        - priority: Integer priority (higher = preferred when multiple strategies eligible)
        - typical_holding_time: "short", "medium", or "long"
        - description: Human-readable explanation
        """
        return {
            "trend_following": {
                # trend_up: classic momentum setup
                # volatility_expanding: early breakout phase, TF capitalises on
                #   expanding moves before "trend_up" is confirmed
                # NOTE: _strategy_trend_following has no internal regime gate of
                # its own - this table is its ONLY regime awareness under Auto,
                # so there is nothing for it to contradict.
                "allowed_regimes": ["trend_up", "volatility_expanding"],
                "priority": 4,
                "typical_holding_time": "long",
                "description": "Best for sustained uptrends with clear momentum"
            },
            "volatility_breakout": {
                # Must match _strategy_volatility_breakout's own regime_filter
                # default (allowed_regimes=["volatility_expanding"], see the
                # strategy's REGIME GATING section). The strategy itself only
                # ever enters once volatility has expanded (breakout confirmed)
                # - it does NOT enter during compression, it just watches for
                # one. The old ["volatility_contracting", "volatility_low"]
                # value here declared it Auto-eligible during compression and
                # then ineligible the instant it actually expanded and entered,
                # which caused force-exits right at (or just after) entry.
                # "volatility_expanding" here is a DIRECTION tag (rate of
                # change, from volatility_direction) - matched via
                # _is_strategy_eligible's `f"volatility_{vol_direction}"` tag,
                # NOT the separate `f"volatility_{volatility}"` LEVEL tag used
                # by e.g. mean_reversion/adaptive_grid below. The strategy's
                # own standalone regime_filter_enabled path (used outside
                # Auto) now uses the same direction-based detector for
                # consistency - see fix-regime-detection-consistency.
                "allowed_regimes": ["volatility_expanding"],
                "priority": 3,
                "typical_holding_time": "medium",
                "description": "Captures breakouts after volatility compression"
            },
            "mean_reversion": {
                # Must match _strategy_mean_reversion's own hardcoded regime gate
                # (allowed_regimes = ["trend_flat", "volatility_high"] in its
                # REGIME GATING section) - it also trades high-volatility fades,
                # not just flat/ranging markets.
                "allowed_regimes": ["trend_flat", "volatility_high"],
                "priority": 2,
                "typical_holding_time": "short",
                "description": "Profits from price mean reversion in choppy markets"
            },
            "adaptive_grid": {
                # Must match _strategy_grid's own regime gate default
                # (allowed_regimes = ["trend_flat", "volatility_medium"]), which
                # only ever checks trend_state/volatility_state tags. The old
                # "volatility_stable" entry here referred to volatility_direction,
                # a tag the grid's own gate never inspects, so it was eligible
                # here for a condition the strategy itself would never act on.
                "allowed_regimes": ["trend_flat", "volatility_medium"],
                "priority": 2,
                "typical_holding_time": "medium",
                "description": "Range-bound grid trading for sideways markets"
            },
            # Note: VWAP removed - it is an execution algorithm, not an alpha strategy
            "dca_accumulator": {
                # Always eligible as the default fallback accumulator
                "allowed_regimes": ["all"],
                "priority": 0,
                "typical_holding_time": "long",
                "description": "Safe default accumulator for all market conditions"
            },
            "dip_recovery": {
                # trend_down: the decline this strategy tracks a bottom during.
                # volatility_expanding / volatility_high: sharp, fast moves are
                #   exactly the setup it targets (pullback reversal, not a slow grind).
                # NOTE: _strategy_dip_recovery has no internal regime gate of its
                # own (its entry protection is its ATR-normalised decline/recovery
                # ratio logic) - this table is its ONLY regime awareness under
                # Auto, so there is nothing for it to contradict.
                "allowed_regimes": ["trend_down", "volatility_expanding", "volatility_high"],
                "priority": 3,
                "typical_holding_time": "medium",
                "description": "Buys confirmed reversals after significant declines - never buys a still-falling market"
            },
        }

    def _filter_eligible_strategies(self, regime: dict, capabilities: dict) -> list:
        """Filter strategies that are eligible for the current regime.

        Args:
            regime: Current market regime dict with trend_state, volatility_state, liquidity_state
            capabilities: Strategy capability map from _get_strategy_capabilities()

        Returns:
            List of (strategy_name, priority, reason) tuples for eligible strategies
        """
        trend = regime.get("trend_state", "flat")
        volatility = regime.get("volatility_state", "medium")
        liquidity = regime.get("liquidity_state", "normal")

        # Construct regime tags for matching
        regime_tags = [
            f"trend_{trend}",
            f"volatility_{volatility}",
            f"liquidity_{liquidity}"
        ]

        eligible = []

        for strategy_name, caps in capabilities.items():
            allowed = caps["allowed_regimes"]

            # Check if strategy is allowed in current regime
            if "all" in allowed:
                # Strategy allowed in all regimes
                eligible.append((
                    strategy_name,
                    caps["priority"],
                    f"fallback strategy (allowed in all regimes)"
                ))
            else:
                # Check if any regime tag matches allowed regimes
                matches = [tag for tag in regime_tags if tag in allowed]
                if matches:
                    eligible.append((
                        strategy_name,
                        caps["priority"],
                        f"matches regime: {', '.join(matches)}"
                    ))

        # If no strategies are eligible, fallback to dca_accumulator
        if not eligible:
            dca_caps = capabilities.get("dca_accumulator", {})
            eligible.append((
                "dca_accumulator",
                dca_caps.get("priority", 0),
                "fallback (no strategies matched regime)"
            ))

        return eligible

    def _select_strategy_from_eligible(
        self,
        eligible_strategies: list,
        current_strategy: str,
        auto_state: dict,
        min_switch_interval: int
    ) -> tuple:
        """Select best strategy from eligible strategies with inertia.

        Args:
            eligible_strategies: List of (strategy_name, priority, reason) tuples
            current_strategy: Currently active strategy
            auto_state: Auto mode state dict
            min_switch_interval: Minimum minutes between switches

        Returns:
            (selected_strategy, should_switch, reason) tuple
        """
        if not eligible_strategies:
            # Should never happen due to fallback, but handle gracefully
            return current_strategy, False, "no eligible strategies"

        # Sort by priority (descending)
        eligible_strategies.sort(key=lambda x: x[1], reverse=True)

        # Check if current strategy is still eligible
        current_eligible = any(s[0] == current_strategy for s in eligible_strategies)

        if current_eligible:
            # Current strategy is still eligible
            current_priority = next(s[1] for s in eligible_strategies if s[0] == current_strategy)
            best_strategy, best_priority, best_reason = eligible_strategies[0]

            # Only switch if new strategy has STRICTLY higher priority
            if best_priority > current_priority:
                # Check minimum switch interval
                last_switch = auto_state.get("last_switch_time")
                if last_switch:
                    time_since_switch = (
                        self.clock.now() - datetime.fromisoformat(last_switch)
                    ).total_seconds() / 60

                    if time_since_switch < min_switch_interval:
                        return current_strategy, False, f"too soon to switch (last switch {time_since_switch:.1f}m ago)"

                # Switch to higher priority strategy
                return best_strategy, True, f"higher priority strategy available ({best_reason})"
            else:
                # Keep current strategy (prefer inertia)
                return current_strategy, False, f"current strategy still optimal"
        else:
            # Current strategy no longer eligible, must switch
            best_strategy, best_priority, best_reason = eligible_strategies[0]
            return best_strategy, True, f"current strategy not eligible, switching to {best_strategy} ({best_reason})"

    def _get_auto_state(self, bot_id: int) -> dict:
        """Get Auto Mode state for a bot."""
        if not hasattr(self, "_auto_states"):
            self._auto_states = {}
        return self._auto_states.get(bot_id, {})

    def _save_auto_state(self, bot_id: int, state: dict) -> None:
        """Save Auto Mode state for a bot."""
        if not hasattr(self, "_auto_states"):
            self._auto_states = {}
        self._auto_states[bot_id] = state

    async def _execute_trade(
        self,
        bot: Bot,
        exchange: ExchangeService,
        signal: TradeSignal,
        current_price: float,
        session: AsyncSession,
    ) -> Optional[Order]:
        """Execute a trade based on signal with FULL SAFETY BOUNDARY LAYER.

        MANDATORY TRADE LIFECYCLE (enforced in order):
        1. Strategy generates TradeSignal
        2. Auto_mode approves strategy (if applicable)
        3. **Portfolio risk caps checked** ← Step 3
        4. **Strategy capacity checked** ← Step 4
        5. **Execution cost estimated** ← Step 5
        6. **Order size adjusted** ← Step 6
        7. Per-bot risk checks (existing)
        8. Execute trade ← Step 8

        This method implements steps 3-8.

        Separates alpha (strategy decision) from execution (how to execute).
        Strategy signals decide WHAT to trade and WHY.
        Execution layer decides HOW to execute the trade.

        Args:
            bot: The bot model
            exchange: Exchange service
            signal: Trade signal (includes execution method)
            current_price: Current market price
            session: Database session

        Returns:
            Order if executed, None otherwise
        """
        # === STEP 2.5: RESOLVE SELL SIZE AGAINST THE OPEN POSITION ===
        # Resolve how much base to sell BEFORE risk/cost sizing so every
        # downstream step (portfolio caps, cost model, order, position update)
        # sees the true notional. Two real, observed failure modes this closes:
        #   * "Sell all" exits pass amount<=0 as a sentinel (e.g. Auto Mode
        #     force-exit: TradeSignal(action="sell", amount=0)). Left
        #     unresolved that places a ZERO-size sell that never closes the
        #     position, so the strategy retries the exit every loop forever.
        #   * A full-position exit sizes amount=base*price; dividing back by
        #     price overshoots the held base by a float rounding error ~5% of
        #     the time. The old code REJECTED that, leaving the position stuck
        #     open. We clamp to the holding so a full exit always settles.
        sell_base = None
        if signal.action == "sell":
            result = await session.execute(
                select(Position).where(
                    Position.bot_id == bot.id,
                    Position.trading_pair == bot.trading_pair,
                )
            )
            position = result.scalar_one_or_none()
            if not position:
                reason = "cannot sell without open position"
                logger.warning(f"Bot {bot.id}: Trade REJECTED - {reason}")
                diagnostics_store.record_blocked(bot.id, BLOCK_OTHER, reason)
                await self._record_trade_outcome(bot, "sell_no_position", reason)
                return None

            position_amount = getattr(position, "amount", None)
            if position_amount is not None and isinstance(position_amount, (int, float)):
                if signal.amount is None or signal.amount <= 0:
                    sell_base = position_amount  # sentinel: close entire position
                else:
                    sell_base = min(signal.amount / current_price, position_amount)
                # Realign the signal's quote notional with the resolved base so
                # risk caps and the cost model price the trade that actually runs.
                signal.amount = sell_base * current_price
            else:
                # Test mock without a numeric position amount: trust the signal.
                sell_base = (signal.amount or 0.0) / current_price

        # === STEP 3: PORTFOLIO RISK CAPS CHECK ===
        portfolio_risk = PortfolioRiskService(session)
        portfolio_check = await portfolio_risk.check_portfolio_risk(
            bot.id,
            signal.amount,
            signal.action,
        )

        if not portfolio_check.ok:
            reason = f"portfolio risk cap {portfolio_check.violated_cap}: {portfolio_check.details}"
            logger.warning(f"Bot {bot.id}: Trade REJECTED by {reason}")
            diagnostics_store.record_blocked(bot.id, BLOCK_RISK_MANAGER, reason)
            await self._record_trade_outcome(
                bot, f"portfolio_cap:{portfolio_check.violated_cap}", reason
            )
            return None

        # Apply portfolio-level order resize if needed
        if portfolio_check.action == "resize" and portfolio_check.adjusted_amount:
            original_amount = signal.amount
            signal.amount = portfolio_check.adjusted_amount
            logger.info(
                f"Bot {bot.id}: Order resized by portfolio caps - "
                f"${original_amount:.2f} → ${signal.amount:.2f}"
            )

        # === STEP 4: STRATEGY CAPACITY CHECK ===
        if signal.action == "buy":  # Only check capacity for buys
            strategy_capacity = StrategyCapacityService(session)
            capacity_check = await strategy_capacity.check_capacity_for_trade(
                bot.id,
                bot.strategy,
                signal.amount,
            )

            if not capacity_check.ok:
                reason = f"strategy capacity limit: {capacity_check.reason}"
                logger.warning(f"Bot {bot.id}: Trade REJECTED by {reason}")
                diagnostics_store.record_blocked(bot.id, BLOCK_POSITION_LIMITS, reason)
                await self._record_trade_outcome(bot, "strategy_capacity", reason)
                return None

            # Apply strategy capacity resize if needed
            if capacity_check.adjusted_amount and capacity_check.adjusted_amount < signal.amount:
                original_amount = signal.amount
                signal.amount = capacity_check.adjusted_amount
                logger.info(
                    f"Bot {bot.id}: Order resized by strategy capacity - "
                    f"${original_amount:.2f} → ${signal.amount:.2f}"
                )

        # === STEP 5: EXECUTION COST ESTIMATION ===
        # bot.exchange_fee/market_spread_pct/slippage_pct are persisted
        # columns (default 0.1% / 0.0% / 0.0% - see add-trading-safety-
        # boundaries, which added the latter two so live can be configured
        # the same way the backtest CLI's --spread-pct/--slippage-pct
        # already were; previously these were hardcoded to 0.0 with no way
        # to configure them at all). The getattr fallback covers the window
        # between a deploy and migration run. The isinstance guard ensures
        # tests that use Mock bots (where attribute access returns a
        # non-numeric Mock) fall back to a safe default rather than
        # propagating a non-numeric value into the cost model arithmetic.
        _exec_fee_raw = getattr(bot, 'exchange_fee', 0.1)
        _exec_fee_pct = (
            float(_exec_fee_raw) if isinstance(_exec_fee_raw, (int, float)) else 0.1
        )
        _exec_spread_raw = getattr(bot, 'market_spread_pct', 0.0)
        _exec_spread_pct = (
            float(_exec_spread_raw) if isinstance(_exec_spread_raw, (int, float)) else 0.0
        )
        _exec_slippage_raw = getattr(bot, 'slippage_pct', 0.0)
        _exec_slippage_pct = (
            float(_exec_slippage_raw) if isinstance(_exec_slippage_raw, (int, float)) else 0.0
        )
        cost_model = get_cost_model(
            exchange_fee_pct=_exec_fee_pct,
            market_spread_pct=_exec_spread_pct,
            slippage_pct=_exec_slippage_pct,
            impact_pct=0.0,         # Not used for spot
        )

        cost_estimate = cost_model.estimate_cost(
            side=signal.action,
            notional_usd=signal.amount,
            price=current_price,
        )

        logger.debug(
            f"Bot {bot.id}: Estimated execution cost - "
            f"${cost_estimate.total_cost:.4f} "
            f"(fee=${cost_estimate.exchange_fee:.4f}, "
            f"spread=${cost_estimate.spread_cost:.4f}, "
            f"slip=${cost_estimate.slippage_cost:.4f})"
        )

        # === STEP 5.5: TRADE VIABILITY GATE (fail-closed) ===
        # Sells are never blocked — they close positions, not open risk.
        # BUYs are checked differently by strategy type:
        #   Accumulation (is_accumulation=True): fee sanity check only.
        #   Directional (is_accumulation=False): expected_move > round_trip + margin.
        # Viability rejections are VALID no-trade decisions, not failures.
        # They do NOT increment the repeated-rejection circuit breaker counter.
        if signal.action == "buy":
            _fee_raw_gate = getattr(bot, 'exchange_fee', 0.1)
            _fee_pct_gate = (
                float(_fee_raw_gate) if isinstance(_fee_raw_gate, (int, float)) else 0.1
            ) / 100.0

            if signal.is_accumulation:
                # Accumulation strategies have no per-trade price target.
                # Sanity-check only: reject if exchange fee is absurdly high.
                if _fee_pct_gate > _MAX_ACCUMULATION_FEE_PCT:
                    reason = (
                        f"exchange fee {_fee_pct_gate * 100:.2f}% exceeds "
                        f"accumulation maximum {_MAX_ACCUMULATION_FEE_PCT * 100:.0f}%"
                    )
                    logger.warning(f"Bot {bot.id}: Trade REJECTED (viability/accumulation) - {reason}")
                    diagnostics_store.record_blocked(bot.id, BLOCK_OTHER, reason)
                    return None
            else:
                # Directional strategies must prove edge clears round-trip fees.
                if not isinstance(signal.expected_move_pct, (int, float)):
                    reason = "missing expected move estimate — strategy did not provide expected_move_pct"
                    logger.warning(f"Bot {bot.id}: Trade REJECTED (viability) - {reason}")
                    diagnostics_store.record_blocked(bot.id, BLOCK_OTHER, reason)
                    return None

                rt_cost_usd = cost_model.estimate_roundtrip_cost(
                    notional_usd=signal.amount,
                    price=current_price,
                )
                rt_cost_pct = rt_cost_usd / signal.amount if signal.amount > 0 else 0.0
                min_viable_pct = rt_cost_pct + _VIABILITY_SAFETY_MARGIN_PCT
                if signal.expected_move_pct < min_viable_pct:
                    reason = (
                        f"expected move {signal.expected_move_pct * 100:.3f}% < "
                        f"round-trip {rt_cost_pct * 100:.3f}% + "
                        f"margin {_VIABILITY_SAFETY_MARGIN_PCT * 100:.3f}% "
                        f"= min {min_viable_pct * 100:.3f}%"
                    )
                    logger.warning(f"Bot {bot.id}: Trade REJECTED (viability) - {reason}")
                    diagnostics_store.record_blocked(bot.id, BLOCK_OTHER, reason)
                    return None

                # Reward:risk check — a no-op when the strategy has no fixed
                # stop (expected_risk_pct=None), so strategies not retrofitted
                # with a locked stop keep their pre-existing behavior exactly.
                rr_ok, rr_reason = evaluate_reward_risk(
                    signal.expected_move_pct, signal.expected_risk_pct
                )
                if not rr_ok:
                    reason = f"reward:risk check failed - {rr_reason}"
                    logger.warning(f"Bot {bot.id}: Trade REJECTED (viability) - {reason}")
                    diagnostics_store.record_blocked(bot.id, BLOCK_OTHER, reason)
                    return None

        # === STEP 6: ORDER SIZE VALIDATION ===
        # Ensure order size is still meaningful after adjustments.
        # H3: the minimum applies to BUYS only (opening/adding risk). Sells must
        # never be blocked by it, or a small/dust position could not be closed -
        # defeating stop-loss and trailing-stop exits. Live sells are still
        # validated against the exchange's own min-notional in _preflight_order.
        min_order_size = MIN_ORDER_USD  # shared $10 minimum
        if signal.action == "buy" and signal.amount < min_order_size:
            reason = (
                f"order size ${signal.amount:.2f} < ${min_order_size:.2f} minimum"
            )
            logger.warning(f"Bot {bot.id}: Trade REJECTED - {reason}")
            diagnostics_store.record_blocked(bot.id, BLOCK_MIN_ORDER_SIZE, reason)
            await self._record_trade_outcome(bot, "buy_below_min", reason)
            return None

        # === STEP 7: EXECUTION LAYER ROUTING ===
        # Determine execution mode from signal
        # Prefer signal.execution, but fall back to order_type for backward compatibility
        if signal.execution:
            execution_mode = signal.execution
        elif signal.order_type == "limit":
            execution_mode = "limit"
        else:
            execution_mode = "market"

        # Route to appropriate execution handler
        if execution_mode == "twap":
            logger.info(f"Bot {bot.id}: Executing {signal.action} using TWAP")
            return await self._execute_twap(bot, exchange, signal, current_price, session)
        elif execution_mode == "vwap":
            logger.info(f"Bot {bot.id}: Executing {signal.action} using VWAP")
            return await self._execute_vwap(bot, exchange, signal, current_price, session)
        elif execution_mode in ["market", "limit"]:
            pass
        else:
            logger.warning(
                f"Bot {bot.id}: Unknown execution mode '{execution_mode}', "
                f"falling back to market execution"
            )
            execution_mode = "market"

        # === STEP 8: EXECUTE TRADE ===
        # Amount in base currency. Buys size from the signal's quote amount;
        # sells use the position-resolved base from STEP 2.5 so a full exit
        # settles exactly (no float-rounding overshoot, no dust left, and the
        # "sell all" sentinel actually closes the position).
        if signal.action == "sell":
            amount_base = sell_base
        else:
            amount_base = signal.amount / current_price

        # Determine order side
        side = OrderSide.BUY if signal.action == "buy" else OrderSide.SELL

        # Place order
        if execution_mode == "market" or signal.order_type == "market":
            logger.debug(f"Bot {bot.id}: Placing market order: {side} {amount_base:.6f} {bot.trading_pair}")
            exchange_order = await exchange.place_market_order(
                bot.trading_pair, side, amount_base, reference_price=current_price
            )
        else:
            # Determine limit price: prefer limit_price, fallback to price, then current_price
            limit_price = signal.limit_price or signal.price or current_price
            logger.debug(f"Bot {bot.id}: Placing limit order: {side} {amount_base:.6f} {bot.trading_pair} @ ${limit_price:.2f}")
            exchange_order = await exchange.place_limit_order(
                bot.trading_pair, side, amount_base, limit_price
            )

        if not exchange_order:
            # Retrieve the specific rejection reason the exchange set before
            # returning None; fall back to a generic message if unavailable.
            rejection = getattr(exchange, "last_order_error", None) or "exchange rejected it"
            reason_text = f"failed to place {signal.action} order: {rejection}"
            logger.error(f"Bot {bot.id}: {reason_text}")
            diagnostics_store.record_execution(
                bot.id, signal.action, success=False,
                reason=reason_text,
            )
            await self._record_trade_outcome(
                bot,
                f"place_order_failed:{signal.action}",
                reason_text,
            )
            return None

        # Map order type
        order_type_map = {
            ("buy", "market"): OrderType.MARKET_BUY,
            ("sell", "market"): OrderType.MARKET_SELL,
            ("buy", "limit"): OrderType.LIMIT_BUY,
            ("sell", "limit"): OrderType.LIMIT_SELL,
        }
        order_type = order_type_map.get((signal.action, signal.order_type), OrderType.MARKET_BUY)

        # Create order record WITH EXECUTION COST MODELING
        order = Order(
            bot_id=bot.id,
            exchange_order_id=exchange_order.id,
            order_type=order_type,
            trading_pair=bot.trading_pair,
            amount=exchange_order.amount,
            price=exchange_order.price,
            limit_price=signal.limit_price,  # Store limit price if provided
            fees=exchange_order.fee,
            # Map exchange order status to our OrderStatus
            # Treat "partial" fills as FILLED (with amount reflecting actual fill)
            status=OrderStatus.FILLED if exchange_order.status in ["closed", "partial"] else OrderStatus.PENDING,
            strategy_used=bot.strategy,
            is_simulated=bot.is_dry_run,
            reason=signal.reason,  # NEW: Track trade reason
            # NEW: Attach modeled execution costs
            modeled_exchange_fee=cost_estimate.exchange_fee,
            modeled_spread_cost=cost_estimate.spread_cost,
            modeled_slippage_cost=cost_estimate.slippage_cost,
            modeled_total_cost=cost_estimate.total_cost,
        )

        if order.status == OrderStatus.FILLED:
            order.filled_at = self.clock.now()

        # add-strategy-decision-framework Phase 0.6: persist this cycle's
        # decision explanation onto the order (closes the Pillar 8
        # persistence gap - purely additive, never affects the trade itself).
        order.decision_explanation, order.edge_management_category = (
            self._decision_explanation_for_order(bot.id)
        )

        session.add(order)
        await session.flush()  # Get order.id for trade recording

        # === ACCOUNTING-GRADE LEDGER INTEGRATION ===
        # Record trade, tax lots, invariants, wallet and position for FILLED
        # orders. PENDING orders (e.g. a resting limit order, or a market order
        # the exchange has not yet confirmed) are finalized later by
        # _resolve_pending_orders once the exchange confirms the fill.
        if order.status == OrderStatus.FILLED:
            finalized = await self._finalize_filled_order(
                session, bot, order, exchange_order, cost_estimate, signal.action
            )
            if not finalized:
                # Invariant validation failed; the transaction is already rolled
                # back. The order is left unresolved for later reconciliation.
                diagnostics_store.record_execution(
                    bot.id, signal.action, success=False,
                    reason="accounting invariant failed while finalizing the fill",
                )
                await self._record_trade_outcome(
                    bot, "finalize_failed",
                    "accounting invariant failed while finalizing the fill",
                )
                return None

        # Commit all changes (order, trade, ledger entries, tax lots, gains)
        await session.commit()

        # Observe-only: a successful execution (main market/limit path).
        diagnostics_store.record_execution(bot.id, signal.action, success=True)

        # Execution succeeded - reset the repeated-rejection breaker.
        await self._record_trade_outcome(bot, None)

        # Export to CSV (async, best-effort - failures don't block trading)
        try:
            csv_exporter = CSVExportService(session)
            # L3: absolute, CWD-independent path under the canonical logs dir.
            log_path = self._trades_csv_path(bot)
            await csv_exporter.export_trades_csv(bot.id, log_path, bot.is_dry_run)
        except Exception as e:
            logger.warning(f"Bot {bot.id}: Failed to export trades CSV: {e}")

        logger.info(
            f"Bot {bot.id}: Executed {signal.action} order - "
            f"{exchange_order.amount:.6f} @ ${exchange_order.price:.2f} "
            f"(costs: ${cost_estimate.total_cost:.4f})"
        )

        # Log trade to per-bot file
        if bot.id in self._bot_loggers:
            self._bot_loggers[bot.id].log_trade(TradeLogEntry(
                timestamp=self.clock.now(),
                bot_id=bot.id,
                bot_name=bot.name,
                order_id=order.id,
                order_type=order_type.value,
                trading_pair=bot.trading_pair,
                amount=exchange_order.amount,
                price=exchange_order.price,
                fees=exchange_order.fee,
                status=order.status.value,
                strategy=bot.strategy,
                running_balance=order.running_balance_after,
                is_simulated=bot.is_dry_run,
            ))

        return order

    async def _finalize_filled_order(
        self,
        session: AsyncSession,
        bot: Bot,
        order: Order,
        exchange_order,
        cost_estimate,
        action: str,
    ) -> bool:
        """Record the accounting consequences of a FILLED order.

        Single source of truth for a fill: creates the Trade + ledger entries,
        processes tax lots, validates invariants, updates the wallet and the
        position cache. Used by both _execute_trade and the order-recovery paths
        (_resolve_pending_orders, _reconcile_orders_with_exchange) so a fill is
        recorded identically however it is discovered.

        Does NOT commit. Returns True on success; on accounting-invariant failure
        it rolls back, marks the order FAILED, and returns False.
        """
        # Parse trading pair to get base and quote assets
        base_asset, quote_asset = bot.trading_pair.split('/')

        # Determine owner_id (TODO: Get from Bot model when owner_id field exists)
        owner_id = str(bot.id)  # FIXME: Use bot.owner_id when available

        # Record trade execution (creates Trade record + ledger entries)
        trade_recorder = TradeRecorderService(session)
        # For partial fills, use the actual filled amount, not the requested amount
        filled_amount = getattr(exchange_order, 'filled', exchange_order.amount) or exchange_order.amount
        # Prefer the exchange-reported notional cost; fall back to filled * price
        reported_cost = getattr(exchange_order, 'cost', None)
        executed_cost = (
            reported_cost
            if isinstance(reported_cost, (int, float)) and reported_cost > 0
            else filled_amount * exchange_order.price
        )
        # Record the fee in the currency the exchange actually charged it
        # (only trust a real string; fall back to the quote asset)
        reported_fee_currency = getattr(exchange_order, 'fee_currency', None)
        fee_asset = (
            reported_fee_currency
            if isinstance(reported_fee_currency, str) and reported_fee_currency
            else quote_asset
        )
        trade = await trade_recorder.record_trade(
            order_id=order.id,
            owner_id=owner_id,
            bot_id=bot.id,
            exchange=bot.exchange if hasattr(bot, 'exchange') else 'simulated',
            trading_pair=bot.trading_pair,
            side=TradeSide.BUY if action == "buy" else TradeSide.SELL,
            base_asset=base_asset,
            quote_asset=quote_asset,
            base_amount=filled_amount,  # Use actual filled amount
            quote_amount=executed_cost,
            price=exchange_order.price,
            fee_amount=exchange_order.fee,
            fee_asset=fee_asset,
            modeled_cost=cost_estimate.total_cost,
            exchange_trade_id=exchange_order.id,
            executed_at=self.clock.now(),
            strategy_used=bot.strategy,
        )

        # Process tax lots (FIFO cost basis tracking)
        tax_engine = FIFOTaxEngine(session)
        if action == "buy":
            # BUY creates a new tax lot
            await tax_engine.process_buy(trade=trade, bot_id=bot.id)
            logger.info(
                f"Bot {bot.id}: Created tax lot for {trade.base_amount:.8f} {base_asset} "
                f"@ ${trade.get_cost_basis_per_unit():.2f}/unit"
            )
        else:
            # SELL consumes tax lots in FIFO order and records realized gains.
            # CR-3: the FIFO realized P&L is authoritative; it (not a separate
            # average-cost calc) is what updates bot.total_pnl/current_balance,
            # so the operational balance stays consistent with the ledger.
            realized_pnl = 0.0
            realized_gains = await tax_engine.process_sell(trade=trade, bot_id=bot.id)
            if realized_gains:
                # Handle different return formats for compatibility with tests:
                # - List of RealizedGain objects (production)
                # - List of floats (some tests)
                # - Tuple (realized_gain_float, consumed_lots) (legacy test format)
                if isinstance(realized_gains, tuple):
                    # Legacy format: (realized_gain, consumed_lots)
                    total_gain = realized_gains[0] if realized_gains else 0.0
                    lot_count = len(realized_gains[1]) if len(realized_gains) > 1 else 0
                else:
                    # List format: support both objects with .gain_loss and plain floats
                    total_gain = sum(
                        g.gain_loss if hasattr(g, "gain_loss") else g
                        for g in realized_gains
                    )
                    lot_count = len(realized_gains)

                realized_pnl = total_gain
                logger.info(
                    f"Bot {bot.id}: Realized gain/loss ${total_gain:+.2f} "
                    f"from {lot_count} tax lot(s)"
                )

        # === ACCOUNTING INVARIANT VALIDATION ===
        # CRITICAL: Validate all accounting invariants before updating cached state
        # If validation fails, roll back and signal failure to the caller.
        invariant_validator = LedgerInvariantService(session)
        try:
            await invariant_validator.validate_trade(trade.id)
        except Exception as e:
            logger.critical(
                f"Bot {bot.id}: ACCOUNTING VALIDATION FAILED for trade {trade.id}. "
                f"Rolling back transaction. Error: {e}"
            )
            # Rollback the transaction to prevent corrupt state
            await session.rollback()
            # Mark order as failed
            order.status = OrderStatus.FAILED
            return False

        # Update wallet (LEGACY - kept for backward compatibility)
        # NOTE: Ledger entries are already created by trade_recorder
        wallet = VirtualWalletService(session)
        total_cost = exchange_order.fee + cost_estimate.total_cost

        if action == "buy":
            await wallet.record_trade_result(bot.id, -total_cost, 0)
        else:
            await wallet.record_trade_result(bot.id, -total_cost, 0)

        # Update running balance
        result = await session.execute(select(Bot).where(Bot.id == bot.id))
        updated_bot = result.scalar_one_or_none()
        if updated_bot:
            order.running_balance_after = updated_bot.current_balance

        # Update/create position (derived state cache)
        # Use the actual filled amount so partial fills cannot
        # desynchronize positions from the ledger
        if action == "buy":
            owning_strategy = self._resolve_owning_strategy(bot, order.reason)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session,
                owning_strategy=owning_strategy,
                entry_reason=order.reason,
                entry_strategy_state=self._snapshot_entry_strategy_state(owning_strategy, bot.id),
            )
        else:
            await self._close_or_reduce_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session, wallet, realized_pnl=realized_pnl
            )

        return True

    # === ORDER RECOVERY / RECONCILIATION (C2, H2) ===
    # Guarantee that every order and position can be recovered after a failure
    # by treating the exchange as the source of truth.

    @staticmethod
    def _action_for_order_type(order_type: OrderType) -> str:
        """Map an order type to a buy/sell action."""
        return "buy" if order_type in (OrderType.MARKET_BUY, OrderType.LIMIT_BUY) else "sell"

    def _cost_estimate_for(self, bot: Bot, action: str, notional: float, price: float):
        """Build a cost estimate for a recovered fill (mirrors _execute_trade,
        including its market_spread_pct/slippage_pct - see
        add-trading-safety-boundaries)."""
        _fee_raw = getattr(bot, 'exchange_fee', 0.1)
        _fee_pct = float(_fee_raw) if isinstance(_fee_raw, (int, float)) else 0.1
        _spread_raw = getattr(bot, 'market_spread_pct', 0.0)
        _spread_pct = float(_spread_raw) if isinstance(_spread_raw, (int, float)) else 0.0
        _slippage_raw = getattr(bot, 'slippage_pct', 0.0)
        _slippage_pct = float(_slippage_raw) if isinstance(_slippage_raw, (int, float)) else 0.0
        cost_model = get_cost_model(
            exchange_fee_pct=_fee_pct,
            market_spread_pct=_spread_pct,
            slippage_pct=_slippage_pct,
        )
        return cost_model.estimate_cost(side=action, notional_usd=notional, price=price)

    async def _resolve_pending_orders(
        self,
        bot_id: int,
        exchange: ExchangeService,
        session: AsyncSession,
    ) -> int:
        """Resolve locally-PENDING orders against the exchange (H2).

        For each PENDING order, ask the exchange for its true state:
        - filled/closed -> finalize (record trade, tax lots, position)
        - canceled/expired/rejected -> mark CANCELLED
        - still open -> leave untouched
        Returns the number of orders whose state changed.
        """
        result = await session.execute(
            select(Order).where(
                Order.bot_id == bot_id,
                Order.status == OrderStatus.PENDING,
            )
        )
        pending = result.scalars().all()
        resolved = 0

        for order in pending:
            if not order.exchange_order_id:
                continue
            ex_order = await exchange.get_order(order.exchange_order_id, order.trading_pair)
            if ex_order is None:
                continue

            status = (getattr(ex_order, "status", "") or "").lower()
            filled = getattr(ex_order, "filled", 0) or 0

            if status in ("closed", "filled", "partial") and (filled or ex_order.amount):
                bot = (
                    await session.execute(select(Bot).where(Bot.id == bot_id))
                ).scalar_one_or_none()
                if bot is None:
                    continue
                action = self._action_for_order_type(order.order_type)
                # Sync the local order to the exchange's reported fill.
                order.status = OrderStatus.FILLED
                order.filled_at = self.clock.now()
                order.amount = filled or ex_order.amount or order.amount
                order.price = ex_order.price or order.price
                order.fees = ex_order.fee or order.fees
                notional = (getattr(ex_order, "cost", None) or order.amount * order.price)
                cost_estimate = self._cost_estimate_for(bot, action, notional, order.price)
                ok = await self._finalize_filled_order(
                    session, bot, order, ex_order, cost_estimate, action
                )
                if ok:
                    await session.commit()
                    resolved += 1
                    logger.info(
                        f"Bot {bot_id}: resolved pending order {order.exchange_order_id} -> FILLED"
                    )
            elif status in ("canceled", "cancelled", "expired", "rejected"):
                order.status = OrderStatus.CANCELLED
                await session.commit()
                resolved += 1
                logger.info(
                    f"Bot {bot_id}: pending order {order.exchange_order_id} -> {status.upper()}"
                )
            # else: still open on the exchange; leave it for a later cycle.

        return resolved

    async def _reconcile_orders_with_exchange(
        self,
        bot: Bot,
        exchange: ExchangeService,
        session: AsyncSession,
    ) -> int:
        """Import exchange fills that have no local order (C2 recovery).

        Covers the irreducible window where a fill succeeded on the exchange but
        the local write was lost (e.g. a crash between fill and commit). The
        exchange is the source of truth: any recent filled order whose id is not
        recorded locally is imported and finalized. Returns the import count.
        """
        if not hasattr(exchange, "get_recent_orders"):
            return 0

        recent = await exchange.get_recent_orders(bot.trading_pair, limit=50)
        if not recent:
            return 0

        known_result = await session.execute(
            select(Order.exchange_order_id).where(Order.bot_id == bot.id)
        )
        known_ids = {row[0] for row in known_result.all() if row[0]}

        imported = 0
        for ex_order in recent:
            status = (getattr(ex_order, "status", "") or "").lower()
            filled = getattr(ex_order, "filled", 0) or 0
            if status not in ("closed", "filled", "partial"):
                continue
            if not (filled or ex_order.amount):
                continue
            if ex_order.id in known_ids:
                continue

            action = "buy" if getattr(ex_order, "side", "") == "buy" else "sell"
            is_limit = getattr(ex_order, "type", "market") == "limit"
            order_type = {
                ("buy", False): OrderType.MARKET_BUY,
                ("sell", False): OrderType.MARKET_SELL,
                ("buy", True): OrderType.LIMIT_BUY,
                ("sell", True): OrderType.LIMIT_SELL,
            }[(action, is_limit)]

            order = Order(
                bot_id=bot.id,
                exchange_order_id=ex_order.id,
                order_type=order_type,
                trading_pair=bot.trading_pair,
                amount=filled or ex_order.amount,
                price=ex_order.price,
                fees=ex_order.fee or 0.0,
                status=OrderStatus.FILLED,
                strategy_used=bot.strategy,
                is_simulated=bot.is_dry_run,
                reason="recovered: imported from exchange",
                filled_at=self.clock.now(),
            )
            session.add(order)
            await session.flush()

            notional = (getattr(ex_order, "cost", None) or order.amount * order.price)
            cost_estimate = self._cost_estimate_for(bot, action, notional, order.price)
            ok = await self._finalize_filled_order(
                session, bot, order, ex_order, cost_estimate, action
            )
            if ok:
                await session.commit()
                known_ids.add(ex_order.id)
                imported += 1
                logger.warning(
                    f"Bot {bot.id}: imported untracked exchange fill "
                    f"{ex_order.id} ({action} {order.amount}) during recovery"
                )

        return imported

    async def _recover_bot_orders(
        self,
        bot: Bot,
        exchange: ExchangeService,
        session: AsyncSession,
    ) -> int:
        """Full order recovery for a bot: resolve pending + import missing fills.

        Run on startup/resume so no order or position is left desynchronized
        from the exchange after a failure.
        """
        recovered = 0
        try:
            recovered += await self._resolve_pending_orders(bot.id, exchange, session)
            recovered += await self._reconcile_orders_with_exchange(bot, exchange, session)
        except Exception as e:
            logger.error(f"Bot {bot.id}: order recovery failed: {e}")
        if recovered:
            logger.info(f"Bot {bot.id}: order recovery reconciled {recovered} order(s)")
        return recovered

    # === OPERATIONAL ALERTING & LIFECYCLE (M1, M2, L4) ===

    async def _emit_alert(
        self,
        session: AsyncSession,
        bot_id: Optional[int],
        alert_type: str,
        message: str,
        *,
        email_subject: Optional[str] = None,
    ) -> None:
        """Persist an operational Alert and best-effort send an email.

        Never raises: alerting must not take down the trading loop. An email is
        attempted only when a subject is given and email delivery is enabled.
        """
        email_sent = False
        try:
            if email_subject and email_service.is_enabled():
                email_sent = bool(
                    email_service.send_email(email_subject, f"<p>{message}</p>", message)
                )
        except Exception as e:
            logger.error(f"Failed to send alert email for bot {bot_id}: {e}")
        try:
            session.add(Alert(
                bot_id=bot_id,
                alert_type=alert_type,
                message=message,
                email_sent=email_sent,
            ))
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to persist alert for bot {bot_id}: {e}")

    async def _pause_bot_for_failures(
        self, bot_id: int, failures: int, last_error: str
    ) -> None:
        """Circuit breaker (M1): pause a persistently failing bot and alert."""
        reason = (
            f"Paused by failure circuit breaker after {failures} consecutive "
            f"errors. Last error: {last_error}"
        )
        logger.critical(f"Bot {bot_id}: {reason}")
        diagnostics_store.record_pause(bot_id, reason)
        decision_status_store.update(bot_id, DecisionState.RISK_LIMIT, reason=reason)
        try:
            async with async_session_maker() as session:
                result = await session.execute(select(Bot).where(Bot.id == bot_id))
                bot = result.scalar_one_or_none()
                if bot:
                    bot.status = BotStatus.PAUSED
                    bot.paused_at = self.clock.now()
                    bot.updated_at = self.clock.now()
                    await session.commit()
                    await self._emit_alert(
                        session, bot_id, "failure_circuit_breaker", reason,
                        email_subject=f"TradingBot: bot {bot_id} paused after repeated failures",
                    )
        except Exception as e:
            logger.error(f"Bot {bot_id}: failed to pause via circuit breaker: {e}")
        finally:
            self._stop_flags[bot_id] = True

    # ------------------------------------------------------------------
    # Recovery mode helpers
    # ------------------------------------------------------------------

    async def _enter_recovery_mode(
        self, bot, bot_id: int, reason: str, session
    ) -> None:
        """Transition a bot into RECOVERY_MODE after consecutive losses.

        Stores recovery state in both the in-memory dict (fast path) and
        bot.strategy_state (survives a restart).  Does NOT stop the loop.
        """
        recovery_state = {
            "active": True,
            "entered_at": self.clock.now().isoformat(),
            "trigger_reason": reason,
            "paper_position": None,
            "paper_trades": [],
            "consecutive_paper_wins": 0,
        }
        self._recovery_states[bot_id] = recovery_state

        ss = dict(bot.strategy_state or {})
        ss["recovery_mode"] = recovery_state
        bot.strategy_state = ss
        bot.status = BotStatus.RECOVERY_MODE
        await session.commit()

        logger.warning(
            f"Bot {bot_id}: Entered RECOVERY_MODE — {reason}. "
            "All order signals will be paper-traded until recovery criteria are met."
        )
        diagnostics_store.record_recovery_entered(bot_id, reason)
        decision_status_store.update(
            bot_id, DecisionState.RECOVERY_MODE_PAPER_TRADING,
            reason=reason, symbol=bot.trading_pair,
        )
        email_service.send_bot_paused_alert(
            bot_id=bot_id,
            bot_name=bot.name,
            reason=f"[RECOVERY MODE] {reason} — paper trading until recovery criteria met",
            pnl=bot.total_pnl,
            trading_pair=bot.trading_pair,
        )

    async def _exit_recovery_mode(
        self, bot, bot_id: int, reason: str, session
    ) -> None:
        """Transition a bot out of RECOVERY_MODE and resume real trading."""
        self._recovery_states.pop(bot_id, None)

        ss = dict(bot.strategy_state or {})
        ss.pop("recovery_mode", None)
        bot.strategy_state = ss
        bot.status = BotStatus.RUNNING
        await session.commit()

        logger.info(
            f"Bot {bot_id}: Exited RECOVERY_MODE — {reason}. Resuming real trading."
        )
        diagnostics_store.record_recovery_exited(bot_id, reason)
        decision_status_store.update(
            bot_id, DecisionState.EVALUATING,
            reason=f"Resumed real trading after recovery: {reason}",
            symbol=bot.trading_pair,
        )

    def _check_recovery_exit(self, recovery_state: dict) -> tuple:
        """Evaluate recovery exit criteria.  Returns (should_exit, reason_str)."""
        from .risk_management import RiskManagementService
        return RiskManagementService.check_recovery_exit(
            paper_trades=recovery_state.get("paper_trades", []),
            consecutive_paper_wins=recovery_state.get("consecutive_paper_wins", 0),
        )

    async def _process_paper_trade(
        self, bot, bot_id: int, signal, current_price: float, session
    ) -> None:
        """Simulate a trade signal as a paper (virtual) trade during RECOVERY_MODE.

        BUY → open paper position (records entry price + amount).
        SELL → close paper position, compute P&L, evaluate recovery exit criteria.
        """
        recovery = self._recovery_states.get(bot_id)
        if not recovery:
            logger.warning(f"Bot {bot_id}: _process_paper_trade called but no recovery state")
            return

        _fee_raw_pt = getattr(bot, 'exchange_fee', 0.1)
        taker_fee_rate = (
            float(_fee_raw_pt) if isinstance(_fee_raw_pt, (int, float)) else 0.1
        ) / 100.0

        if signal.action == "buy":
            if recovery.get("paper_position") is None:
                amount_usd = signal.amount or (bot.current_balance * 0.10)
                recovery["paper_position"] = {
                    "entry_price": current_price,
                    "amount_usd": amount_usd,
                    "trading_pair": bot.trading_pair,
                    "entered_at": self.clock.now().isoformat(),
                }
                logger.info(
                    f"Bot {bot_id}: [PAPER] BUY @ {current_price:.4f} "
                    f"(notional ${amount_usd:.2f})"
                )
            # Already in a paper position — ignore duplicate buy signals.

        elif signal.action == "sell":
            pp = recovery.get("paper_position")
            if pp is None:
                return  # No open paper position to close.

            entry_price = pp["entry_price"]
            amount_usd = pp["amount_usd"]
            gross_return_pct = (current_price / entry_price - 1.0)
            fees = amount_usd * taker_fee_rate * 2  # entry + exit
            gain_loss_usd = amount_usd * gross_return_pct - fees
            win = gain_loss_usd > 0

            trade_record = {
                "gain_loss_usd": gain_loss_usd,
                "win": win,
                "entry_price": entry_price,
                "exit_price": current_price,
                "timestamp": self.clock.now().isoformat(),
            }
            recovery["paper_trades"].append(trade_record)

            if win:
                recovery["consecutive_paper_wins"] = recovery.get("consecutive_paper_wins", 0) + 1
            else:
                recovery["consecutive_paper_wins"] = 0

            recovery["paper_position"] = None

            logger.info(
                f"Bot {bot_id}: [PAPER] SELL @ {current_price:.4f} "
                f"(P&L ${gain_loss_usd:+.2f}, win={win}, "
                f"consecutive_wins={recovery['consecutive_paper_wins']})"
            )
            diagnostics_store.record_paper_trade(
                bot_id,
                gain_loss_usd=gain_loss_usd,
                win=win,
                entry_price=entry_price,
                exit_price=current_price,
            )
            decision_status_store.update(
                bot_id,
                DecisionState.RECOVERY_MODE_PAPER_TRADING,
                reason=(
                    f"Paper trade {'WIN' if win else 'LOSS'} ${gain_loss_usd:+.2f} — "
                    f"{recovery['consecutive_paper_wins']} consecutive win(s)"
                ),
                symbol=bot.trading_pair,
            )

            # Persist updated state before potentially exiting recovery.
            ss = dict(bot.strategy_state or {})
            ss["recovery_mode"] = recovery
            bot.strategy_state = ss
            await session.commit()

            # Evaluate exit criteria.
            should_exit, exit_reason = self._check_recovery_exit(recovery)
            if should_exit:
                await self._exit_recovery_mode(bot, bot_id, exit_reason, session)
                return

            # Warn if stuck > 7 days.
            try:
                entered_at = datetime.fromisoformat(recovery["entered_at"])
                days_stuck = (self.clock.now() - entered_at).days
                if days_stuck >= 7:
                    logger.warning(
                        f"Bot {bot_id}: RECOVERY_MODE for {days_stuck} days — "
                        f"still monitoring. Criteria: {exit_reason}"
                    )
            except (KeyError, ValueError):
                pass
            return

        # Persist state for buy path too.
        ss = dict(bot.strategy_state or {})
        ss["recovery_mode"] = recovery
        bot.strategy_state = ss
        await session.commit()

    async def _record_trade_outcome(
        self, bot, reason_key: Optional[str], reason_text: str = ""
    ) -> None:
        """Track consecutive identical trade rejections; break a stuck loop.

        Every executable trade - from both the strategy-signal path and the
        stop-loss path - funnels through ``_execute_trade``. A trade that is
        rejected or fails for the SAME reason on every tick (a sub-minimum order,
        an un-settleable exit, a portfolio cap, ...) would otherwise retry forever
        without an exception ever being raised, so the loop-level failure breaker
        never trips. This counts identical consecutive rejections and, on the
        ``MAX_CONSECUTIVE_REJECTIONS`` threshold, pauses the bot with the reason
        surfaced in Decision Status.

        ``reason_key`` is ``None`` on a successful execution, which resets the
        counter (a single transient rejection cannot accumulate to a pause).
        """
        tracker = self._state_store("_exec_rejections")
        if reason_key is None:
            tracker.pop(bot.id, None)
            return
        entry = tracker.get(bot.id)
        if entry and entry.get("key") == reason_key:
            entry["count"] += 1
        else:
            entry = {"key": reason_key, "count": 1}
            tracker[bot.id] = entry
        if entry["count"] >= MAX_CONSECUTIVE_REJECTIONS:
            tracker.pop(bot.id, None)
            await self._pause_bot_for_repeated_rejection(
                bot.id, entry["count"], reason_text or reason_key
            )

    async def _pause_bot_for_repeated_rejection(
        self, bot_id: int, count: int, reason_text: str
    ) -> None:
        """Pause a bot stuck rejecting the same trade every tick, and alert."""
        reason = (
            f"Paused after {count} consecutive identical trade rejections: "
            f"{reason_text}. Resolve the cause, then resume."
        )
        logger.critical(f"Bot {bot_id}: {reason}")
        diagnostics_store.record_pause(bot_id, reason)
        decision_status_store.update(bot_id, DecisionState.PAUSED, reason=reason)
        try:
            async with async_session_maker() as session:
                result = await session.execute(select(Bot).where(Bot.id == bot_id))
                bot = result.scalar_one_or_none()
                if bot:
                    bot.status = BotStatus.PAUSED
                    bot.paused_at = self.clock.now()
                    bot.updated_at = self.clock.now()
                    await session.commit()
                    await self._emit_alert(
                        session, bot_id, "repeated_rejection_breaker", reason,
                        email_subject=(
                            f"TradingBot: bot {bot_id} paused after repeated "
                            "trade rejections"
                        ),
                    )
        except Exception as e:
            logger.error(
                f"Bot {bot_id}: failed to pause via rejection breaker: {e}"
            )
        finally:
            self._stop_flags[bot_id] = True

    def _cleanup_bot_state(self, bot_id: int) -> None:
        """Drop a bot's in-memory state to prevent unbounded growth (L4)."""
        state_dicts = (
            "_price_histories", "_trend_states",
            "_grid_states", "_mean_reversion_states", "_volatility_breakout_states",
            "_twap_states", "_vwap_states", "_auto_states", "_last_pending_resolve",
            "_bot_loggers", "_exec_rejections", "_explanations",
        )
        for attr in state_dicts:
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(bot_id, None)
        # Drop the transient decision status too (bot stopped/deleted).
        decision_status_store.clear(bot_id)

    def cleanup_bot_state(self, bot_id: int) -> None:
        """Public hook to release a bot's in-memory state (e.g. on delete)."""
        self._cleanup_bot_state(bot_id)

    @staticmethod
    def _trades_csv_path(bot) -> Path:
        """Absolute path of a bot's trades CSV, independent of the process CWD (L3).

        Anchored to the canonical per-bot log directory so logs are never written
        to a stray relative location when the server is started elsewhere.
        """
        suffix = "simulated" if bot.is_dry_run else "live"
        return ensure_bot_log_directory(bot.id) / f"trades_{suffix}.csv"

    # === EXECUTION LAYER METHODS ===
    # These methods implement HOW to execute trades, not WHAT/WHY to trade.
    # Strategies decide alpha, execution layer implements mechanics.

    async def _execute_twap(
        self,
        bot: Bot,
        exchange: ExchangeService,
        signal: TradeSignal,
        current_price: float,
        session: AsyncSession,
    ) -> Optional[Order]:
        """Execute trade using Time-Weighted Average Price (TWAP).

        TWAP splits a large order into equal-sized slices distributed evenly over time.
        This is a pure execution algorithm - it does NOT make trading decisions.

        Execution parameters (from signal.execution_params):
            duration_minutes: Total execution period (default: 60)
            slice_count: Number of order slices (default: 10)
            slice_interval_seconds: Seconds between slices (calculated from duration/count)

        State tracking:
            Maintains per-bot TWAP execution state for multi-slice orders

        Args:
            bot: The bot model
            exchange: Exchange service
            signal: Trade signal (action, amount)
            current_price: Current market price
            session: Database session

        Returns:
            Order if slice executed, None if waiting or complete
        """
        # Extract execution parameters with defaults
        params = signal.execution_params or {}
        duration_minutes = params.get("duration_minutes", 60)
        slice_count = params.get("slice_count", 10)
        total_amount = signal.amount

        # Get or initialize TWAP state
        twap_state = self._get_execution_state(bot.id, "twap")

        # Initialize TWAP execution if starting new order
        if "start_time" not in twap_state or twap_state.get("completed", False):
            twap_state.clear()
            twap_state.update({
                "start_time": self.clock.now(),
                "slices_executed": 0,
                "total_executed_usd": 0.0,
                "target_amount_usd": total_amount,
                "action": signal.action,
                "slice_count": slice_count,
                "duration_minutes": duration_minutes,
                "prices": [],
                "completed": False,
            })
            self._save_execution_state(bot.id, "twap", twap_state)

            logger.info(
                f"Bot {bot.id}: TWAP execution started - "
                f"${total_amount:.2f} {signal.action} over {duration_minutes} min in {slice_count} slices "
                f"(≈${total_amount/slice_count:.2f}/slice every {duration_minutes*60/slice_count:.0f}s)"
            )

        slices_executed = twap_state["slices_executed"]
        total_executed = twap_state["total_executed_usd"]
        target_amount = twap_state["target_amount_usd"]

        # Check if TWAP is complete
        if slices_executed >= slice_count:
            avg_price = sum(twap_state["prices"]) / len(twap_state["prices"]) if twap_state["prices"] else current_price
            twap_state["completed"] = True
            self._save_execution_state(bot.id, "twap", twap_state)

            logger.info(
                f"Bot {bot.id}: TWAP execution complete - "
                f"{slices_executed}/{slice_count} slices executed, "
                f"Total: ${total_executed:.2f}, Avg price: ${avg_price:.2f}"
            )
            return None  # No more orders to place

        # Calculate slice interval
        slice_interval_seconds = (duration_minutes * 60) / slice_count
        start_time = twap_state["start_time"]
        elapsed_seconds = (self.clock.now() - start_time).total_seconds()

        # Check if enough time has passed for next slice
        expected_time_for_next_slice = slices_executed * slice_interval_seconds
        if elapsed_seconds < expected_time_for_next_slice:
            wait_seconds = expected_time_for_next_slice - elapsed_seconds
            logger.debug(
                f"Bot {bot.id}: TWAP waiting - "
                f"Slice {slices_executed + 1}/{slice_count} in {wait_seconds:.0f}s"
            )
            return None  # Not time for next slice yet

        # Check if bot is still active (defensive)
        if bot.id in self._stop_flags and self._stop_flags[bot.id]:
            logger.warning(
                f"Bot {bot.id}: TWAP execution interrupted - bot stopped "
                f"({slices_executed}/{slice_count} slices executed)"
            )
            twap_state["completed"] = True
            self._save_execution_state(bot.id, "twap", twap_state)
            return None

        # Calculate slice amount (equal distribution with remainder handling)
        remaining_slices = slice_count - slices_executed
        remaining_amount = target_amount - total_executed
        slice_amount = remaining_amount / remaining_slices

        # Validate sufficient balance for buys
        if signal.action == "buy" and slice_amount > bot.current_balance:
            slice_amount = bot.current_balance
            if slice_amount < 1.0:  # Min $1 slice
                logger.warning(
                    f"Bot {bot.id}: TWAP execution stopped - insufficient balance "
                    f"({slices_executed}/{slice_count} slices executed, ${total_executed:.2f}/${target_amount:.2f})"
                )
                twap_state["completed"] = True
                self._save_execution_state(bot.id, "twap", twap_state)
                return None

        # Execute slice using market order
        amount_base = slice_amount / current_price
        side = OrderSide.BUY if signal.action == "buy" else OrderSide.SELL

        logger.info(
            f"Bot {bot.id}: TWAP executing slice {slices_executed + 1}/{slice_count} - "
            f"${slice_amount:.2f} {signal.action} @ ${current_price:.2f}"
        )

        exchange_order = await exchange.place_market_order(
            bot.trading_pair, side, amount_base, reference_price=current_price
        )

        if not exchange_order:
            logger.error(f"Bot {bot.id}: TWAP slice {slices_executed + 1} failed to execute")
            return None

        # Update TWAP state
        twap_state["slices_executed"] = slices_executed + 1
        twap_state["total_executed_usd"] = total_executed + slice_amount
        twap_state["prices"].append(current_price)
        self._save_execution_state(bot.id, "twap", twap_state)

        # Create order record (same as standard execution)
        order_type_map = {
            "buy": OrderType.MARKET_BUY,
            "sell": OrderType.MARKET_SELL,
        }
        order_type = order_type_map.get(signal.action, OrderType.MARKET_BUY)

        order = Order(
            bot_id=bot.id,
            exchange_order_id=exchange_order.id,
            order_type=order_type,
            trading_pair=bot.trading_pair,
            amount=exchange_order.amount,
            price=exchange_order.price,
            fees=exchange_order.fee,
            status=OrderStatus.FILLED if exchange_order.status == "closed" else OrderStatus.PENDING,
            strategy_used=f"{bot.strategy} (TWAP {slices_executed + 1}/{slice_count})",
            is_simulated=bot.is_dry_run,
        )

        if order.status == OrderStatus.FILLED:
            order.filled_at = self.clock.now()

        # add-strategy-decision-framework Phase 0.6: see the standard
        # execution path's identical comment above.
        order.decision_explanation, order.edge_management_category = (
            self._decision_explanation_for_order(bot.id)
        )

        session.add(order)

        # Update wallet and positions (same as standard execution)
        # Use the actual filled amount so partial fills cannot desync positions
        filled_amount = getattr(exchange_order, 'filled', exchange_order.amount) or exchange_order.amount
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            owning_strategy = self._resolve_owning_strategy(bot, signal.reason)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session,
                owning_strategy=owning_strategy,
                entry_reason=signal.reason,
                entry_strategy_state=self._snapshot_entry_strategy_state(owning_strategy, bot.id),
            )
        else:
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._close_or_reduce_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session, wallet
            )

        # Update running balance
        result = await session.execute(select(Bot).where(Bot.id == bot.id))
        updated_bot = result.scalar_one_or_none()
        if updated_bot:
            order.running_balance_after = updated_bot.current_balance

        await session.commit()

        logger.info(
            f"Bot {bot.id}: TWAP slice {slices_executed + 1}/{slice_count} executed - "
            f"{exchange_order.amount:.6f} @ ${exchange_order.price:.2f}, "
            f"Progress: ${twap_state['total_executed_usd']:.2f}/${target_amount:.2f}"
        )

        return order

    async def _execute_vwap(
        self,
        bot: Bot,
        exchange: ExchangeService,
        signal: TradeSignal,
        current_price: float,
        session: AsyncSession,
    ) -> Optional[Order]:
        """Execute trade using Volume-Weighted Average Price (VWAP) benchmarking.

        VWAP execution is used for BENCHMARKING, not decision-making.
        This compares achieved execution price vs VWAP to measure execution quality.

        IMPORTANT: VWAP does NOT decide whether to trade (that's the strategy's job).
        It only affects HOW we execute a trade that was already decided.

        In this implementation:
        - Falls back to market execution (no volume data available)
        - Logs VWAP benchmark for comparison
        - Future: Could implement participation rate limiting based on volume

        Execution parameters (from signal.execution_params):
            lookback_minutes: Period for VWAP calculation (default: 30)
            max_participation_rate: Max % of volume per interval (future use)

        Args:
            bot: The bot model
            exchange: Exchange service
            signal: Trade signal (action, amount)
            current_price: Current market price
            session: Database session

        Returns:
            Order if executed, None otherwise
        """
        params = signal.execution_params or {}
        lookback_minutes = params.get("lookback_minutes", 30)

        # Get VWAP state for benchmarking
        vwap_state = self._get_execution_state(bot.id, "vwap")

        # Initialize if needed
        if "price_volume_data" not in vwap_state:
            vwap_state["price_volume_data"] = []

        # Simulate volume data (in production, fetch from exchange)
        # This is a placeholder - real implementation would use exchange.fetch_ohlcv()
        simulated_volume = 1000.0  # Placeholder volume

        vwap_state["price_volume_data"].append({
            "timestamp": self.clock.now(),
            "price": current_price,
            "volume": simulated_volume,
        })

        # Keep only recent data
        cutoff = self.clock.now() - timedelta(minutes=lookback_minutes)
        vwap_state["price_volume_data"] = [
            pv for pv in vwap_state["price_volume_data"]
            if pv["timestamp"] > cutoff
        ]

        self._save_execution_state(bot.id, "vwap", vwap_state)

        # Calculate VWAP benchmark
        vwap_benchmark = None
        if len(vwap_state["price_volume_data"]) >= 5:
            total_pv = sum(pv["price"] * pv["volume"] for pv in vwap_state["price_volume_data"])
            total_volume = sum(pv["volume"] for pv in vwap_state["price_volume_data"])
            if total_volume > 0:
                vwap_benchmark = total_pv / total_volume

        # Execute using market order (fallback when no volume data)
        benchmark_str = f"${vwap_benchmark:.2f}" if vwap_benchmark is not None else "N/A"
        logger.info(
            f"Bot {bot.id}: VWAP execution - "
            f"Using market order (no real volume data available). "
            f"Benchmark VWAP: {benchmark_str}"
        )

        # Delegate to standard market execution
        amount_base = signal.amount / current_price
        side = OrderSide.BUY if signal.action == "buy" else OrderSide.SELL

        exchange_order = await exchange.place_market_order(
            bot.trading_pair, side, amount_base, reference_price=current_price
        )

        if not exchange_order:
            logger.error(f"Bot {bot.id}: VWAP execution failed")
            return None

        # Log execution quality vs VWAP benchmark
        if vwap_benchmark:
            deviation_pct = ((exchange_order.price - vwap_benchmark) / vwap_benchmark) * 100
            quality = "better" if (signal.action == "buy" and exchange_order.price < vwap_benchmark) or \
                                  (signal.action == "sell" and exchange_order.price > vwap_benchmark) else "worse"

            logger.info(
                f"Bot {bot.id}: VWAP execution quality - "
                f"Achieved: ${exchange_order.price:.2f}, Benchmark: ${vwap_benchmark:.2f}, "
                f"Deviation: {deviation_pct:+.2f}% ({quality} than VWAP)"
            )

        # Create order record (same as standard execution)
        order_type_map = {
            "buy": OrderType.MARKET_BUY,
            "sell": OrderType.MARKET_SELL,
        }
        order_type = order_type_map.get(signal.action, OrderType.MARKET_BUY)

        order = Order(
            bot_id=bot.id,
            exchange_order_id=exchange_order.id,
            order_type=order_type,
            trading_pair=bot.trading_pair,
            amount=exchange_order.amount,
            price=exchange_order.price,
            fees=exchange_order.fee,
            status=OrderStatus.FILLED if exchange_order.status == "closed" else OrderStatus.PENDING,
            strategy_used=f"{bot.strategy} (VWAP)",
            is_simulated=bot.is_dry_run,
        )

        if order.status == OrderStatus.FILLED:
            order.filled_at = self.clock.now()

        # add-strategy-decision-framework Phase 0.6: see the standard
        # execution path's identical comment above.
        order.decision_explanation, order.edge_management_category = (
            self._decision_explanation_for_order(bot.id)
        )

        session.add(order)

        # Update wallet and positions (same as standard execution)
        # Use the actual filled amount so partial fills cannot desync positions
        filled_amount = getattr(exchange_order, 'filled', exchange_order.amount) or exchange_order.amount
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            owning_strategy = self._resolve_owning_strategy(bot, signal.reason)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session,
                owning_strategy=owning_strategy,
                entry_reason=signal.reason,
                entry_strategy_state=self._snapshot_entry_strategy_state(owning_strategy, bot.id),
            )
        else:
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._close_or_reduce_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session, wallet
            )

        # Update running balance
        result = await session.execute(select(Bot).where(Bot.id == bot.id))
        updated_bot = result.scalar_one_or_none()
        if updated_bot:
            order.running_balance_after = updated_bot.current_balance

        await session.commit()

        return order

    def _get_execution_state(self, bot_id: int, execution_type: str) -> dict:
        """Get execution state for a bot and execution type.

        Args:
            bot_id: Bot ID
            execution_type: "twap" or "vwap"

        Returns:
            State dictionary (mutable)
        """
        if not hasattr(self, "_execution_states"):
            self._execution_states = {}
        if bot_id not in self._execution_states:
            self._execution_states[bot_id] = {}
        if execution_type not in self._execution_states[bot_id]:
            self._execution_states[bot_id][execution_type] = {}
        return self._execution_states[bot_id][execution_type]

    def _save_execution_state(self, bot_id: int, execution_type: str, state: dict) -> None:
        """Save execution state for a bot and execution type.

        Args:
            bot_id: Bot ID
            execution_type: "twap" or "vwap"
            state: State dictionary to save
        """
        if not hasattr(self, "_execution_states"):
            self._execution_states = {}
        if bot_id not in self._execution_states:
            self._execution_states[bot_id] = {}
        self._execution_states[bot_id][execution_type] = state

    # Per-bot in-memory state attribute a strategy uses to remember its own
    # exit-relevant fields (locked entry ATR, trailing-stop anchor, ...).
    # Used only to opportunistically snapshot that state onto the Position at
    # entry time (entry_strategy_state) for forensic/debugging purposes - the
    # engine still reads the live dict during actual exit decisions. Strategies
    # without per-bot exit state (e.g. dca_accumulator, which has no stop or
    # target) simply have no entry here.
    _STRATEGY_STATE_ATTR = {
        "trend_following": "_trend_states",
        "mean_reversion": "_mean_reversion_states",
        "volatility_breakout": "_volatility_breakout_states",
        "adaptive_grid": "_grid_states",
        "dip_recovery": "_dip_recovery_states",
    }

    def _resolve_owning_strategy(self, bot: Bot, reason: Optional[str]) -> str:
        """Resolve the strategy that actually made this trade decision.

        For fixed-strategy bots this is just ``bot.strategy``. For an
        auto_mode bot, ``bot.strategy`` reads "auto_mode" for every sub-
        strategy it ever dispatches to, so the real sub-strategy name is
        recovered from the "[Auto:strategy_name|regime]" prefix _strategy_auto
        stamps onto the signal's reason (mirrors the parsing already used by
        _update_strategy_performance_metrics for historical attribution -
        this makes the same fact available at write time instead of only
        recoverable later from text).
        """
        if reason and reason.startswith("[Auto:"):
            try:
                return reason.split("|")[0].replace("[Auto:", "").strip()
            except Exception:
                pass
        return bot.strategy

    def _snapshot_entry_strategy_state(self, owning_strategy: str, bot_id: int) -> Optional[dict]:
        """Best-effort copy of the owning strategy's own per-bot state at
        entry time, for Position.entry_strategy_state. Returns None if the
        strategy keeps no such state (e.g. dca_accumulator)."""
        attr = self._STRATEGY_STATE_ATTR.get(owning_strategy)
        if not attr:
            return None
        store = getattr(self, attr, None)
        if not isinstance(store, dict):
            return None
        state = store.get(bot_id)
        return dict(state) if isinstance(state, dict) else None

    async def _open_or_add_position(
        self,
        bot_id: int,
        trading_pair: str,
        amount: float,
        price: float,
        session: AsyncSession,
        owning_strategy: Optional[str] = None,
        entry_reason: Optional[str] = None,
        entry_strategy_state: Optional[dict] = None,
    ) -> None:
        """Open or add to a position."""
        result = await session.execute(
            select(Position).where(
                Position.bot_id == bot_id,
                Position.trading_pair == trading_pair,
            )
        )
        position = result.scalar_one_or_none()

        if position:
            # Average into existing position. Ownership is NOT overwritten -
            # while a position is open, Auto Mode only ever dispatches to its
            # owning strategy (see _strategy_auto), so any add-to-position buy
            # can only come from the same strategy that already owns it; the
            # original entry_reason/entry_strategy_state stay authoritative.
            total_value = position.amount * position.entry_price + amount * price
            total_amount = position.amount + amount
            position.entry_price = total_value / total_amount
            position.amount = total_amount
            position.current_price = price
            position.unrealized_pnl = position.calculate_unrealized_pnl()
        else:
            # Create new position
            position = Position(
                bot_id=bot_id,
                trading_pair=trading_pair,
                side=PositionSide.LONG,
                entry_price=price,
                current_price=price,
                amount=amount,
                unrealized_pnl=0,
                owning_strategy=owning_strategy,
                entry_reason=entry_reason,
                entry_strategy_state=entry_strategy_state,
            )
            session.add(position)

    async def _close_or_reduce_position(
        self,
        bot_id: int,
        trading_pair: str,
        amount: float,
        price: float,
        session: AsyncSession,
        wallet: VirtualWalletService,
        realized_pnl: Optional[float] = None,
    ) -> None:
        """Close or reduce a position and realize P&L.

        CR-3: when ``realized_pnl`` is provided (the FIFO realized gain from the
        ledger), it is the authoritative P&L recorded to the wallet. The
        average-cost fallback is kept only for callers that don't supply it.
        """
        result = await session.execute(
            select(Position).where(
                Position.bot_id == bot_id,
                Position.trading_pair == trading_pair,
            )
        )
        position = result.scalar_one_or_none()

        if not position:
            return

        # Get bot info for logging
        bot_result = await session.execute(select(Bot).where(Bot.id == bot_id))
        bot = bot_result.scalar_one_or_none()

        sell_amount = min(amount, position.amount)
        # Prefer the FIFO realized P&L (ledger truth); fall back to average-cost
        # only when no FIFO value was supplied.
        if realized_pnl is not None:
            pnl = realized_pnl
        else:
            pnl = (price - position.entry_price) * sell_amount

        # Record P&L
        await wallet.record_trade_result(bot_id, pnl, 0)

        # Log fiscal entry for tax purposes
        if bot and bot_id in self._bot_loggers:
            # Extract token from trading pair (e.g., "BTC/USDT" -> "BTC")
            token = trading_pair.split('/')[0] if '/' in trading_pair else trading_pair

            # Calculate holding period
            holding_days = None
            if position.created_at:
                holding_days = (self.clock.now() - position.created_at).days

            proceeds = sell_amount * price
            cost_basis = sell_amount * position.entry_price

            self._bot_loggers[bot_id].log_fiscal_entry(FiscalLogEntry(
                date=self.clock.now(),
                trading_pair=trading_pair,
                token=token,
                buy_date=position.created_at,
                buy_price=position.entry_price,
                sale_price=price,
                amount=sell_amount,
                proceeds=proceeds,
                cost_basis=cost_basis,
                gain_loss=pnl,
                holding_period_days=holding_days,
                is_simulated=bot.is_dry_run,
            ))

        # Update position
        position.amount -= sell_amount
        if position.amount <= 0.000001:  # Close position
            await session.delete(position)
        else:
            position.current_price = price
            position.unrealized_pnl = position.calculate_unrealized_pnl()

    async def _check_positions_stop_loss(
        self,
        bot_id: int,
        exchange: ExchangeService,
        risk_mgr: RiskManagementService,
        session: AsyncSession,
    ) -> None:
        """Check all positions for stop loss."""
        positions = await self._get_bot_positions(bot_id, session)

        for pos in positions:
            ticker = await exchange.get_ticker(pos.trading_pair)
            if not ticker:
                continue

            risk = await risk_mgr.check_stop_loss(
                bot_id,
                pos.entry_price,
                ticker.last,
                pos.amount,
                pos.side == PositionSide.LONG,
            )

            if risk.should_close:
                logger.warning(f"Bot {bot_id}: Stop loss triggered - {risk.reason}")
                # Close position
                result = await session.execute(select(Bot).where(Bot.id == bot_id))
                bot = result.scalar_one_or_none()
                if bot:
                    signal = TradeSignal(
                        action="sell",
                        amount=pos.amount * ticker.last,
                        order_type="market",
                        reason=risk.reason,
                    )
                    await self._execute_trade(bot, exchange, signal, ticker.last, session)

    async def _get_bot_positions(
        self,
        bot_id: int,
        session: AsyncSession,
    ) -> list:
        """Get all positions for a bot.

        In RECOVERY_MODE, if the paper-trade state holds an open position, a
        lightweight virtual Position substitute is prepended so strategies see a
        held position and produce sensible exit signals (otherwise they would
        issue an infinite stream of BUY signals against an empty position list).
        """
        result = await session.execute(
            select(Position).where(Position.bot_id == bot_id)
        )
        real_positions = list(result.scalars().all())

        recovery = self._recovery_states.get(bot_id)
        if recovery and recovery.get("active") and recovery.get("paper_position"):
            pp = recovery["paper_position"]
            # Retrieve the bot's trading pair from the first real position if
            # available, otherwise fall back to recovery state or skip.
            trading_pair = pp.get("trading_pair", "")
            if not trading_pair and real_positions:
                trading_pair = real_positions[0].trading_pair

            if trading_pair:
                virt = _VirtualPosition(
                    bot_id=bot_id,
                    trading_pair=trading_pair,
                    entry_price=pp["entry_price"],
                    amount_usd=pp["amount_usd"],
                )
                return [virt] + real_positions

        return real_positions

    async def _get_last_order(
        self,
        bot_id: int,
        session: AsyncSession,
    ) -> Optional[Order]:
        """Get the most recent order for a bot."""
        result = await session.execute(
            select(Order)
            .where(Order.bot_id == bot_id)
            .order_by(Order.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _get_order_count(
        self,
        bot_id: int,
        session: AsyncSession,
    ) -> int:
        """Get total order count for a bot."""
        from sqlalchemy import func
        result = await session.execute(
            select(func.count(Order.id))
            .where(Order.bot_id == bot_id)
        )
        return result.scalar() or 0

    async def _cancel_pending_orders(
        self,
        bot_id: int,
        session: AsyncSession,
    ) -> int:
        """Cancel all pending orders for a bot."""
        result = await session.execute(
            select(Order).where(
                Order.bot_id == bot_id,
                Order.status == OrderStatus.PENDING,
            )
        )
        pending_orders = result.scalars().all()

        exchange = self._exchange_services.get(bot_id)
        cancelled = 0

        for order in pending_orders:
            if exchange and order.exchange_order_id:
                await exchange.cancel_order(order.exchange_order_id, order.trading_pair)
            order.status = OrderStatus.CANCELLED
            cancelled += 1

        await session.commit()
        return cancelled

    async def _take_pnl_snapshot(
        self,
        bot_id: int,
        session: AsyncSession,
    ) -> None:
        """Take a P&L snapshot for the bot."""
        result = await session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return

        # Only take snapshot every 5 minutes
        last_snapshot = await session.execute(
            select(PnLSnapshot)
            .where(PnLSnapshot.bot_id == bot_id)
            .order_by(PnLSnapshot.snapshot_at.desc())
            .limit(1)
        )
        last = last_snapshot.scalar_one_or_none()

        if last:
            time_since = self.clock.now() - last.snapshot_at
            if time_since.total_seconds() < 300:  # 5 minutes
                return

        snapshot = PnLSnapshot(
            bot_id=bot_id,
            total_pnl=bot.total_pnl,
        )
        session.add(snapshot)
        await session.commit()

    async def _get_next_strategy(self, current_strategy: str) -> str:
        """Get next strategy for rotation.

        Args:
            current_strategy: Current strategy name

        Returns:
            Next strategy name
        """
        # H-3: rotate ONLY among valid alpha strategies. Rotating into an
        # execution algorithm (twap/vwap) would leave the bot with no executor
        # and silently stop it from trading.
        strategies = list(_ALPHA_STRATEGIES)

        try:
            idx = strategies.index(current_strategy)
            return strategies[(idx + 1) % len(strategies)]
        except ValueError:
            return strategies[0]

    async def resume_bots_on_startup(self) -> int:
        """Resume all bots that were active when the server stopped.

        This restarts the execution loop for every bot whose persisted status is
        an ACTIVE one — RUNNING *and* RECOVERY_MODE. RECOVERY_MODE is a live,
        paper-trading state (the bot must keep evaluating the market every tick),
        NOT an idle one: if it were excluded here, a recovery bot would have no
        loop task after a restart and would freeze forever — 0 evaluations, 0
        paper trades — which is exactly the defect this guards against. The
        recovery state itself is restored from ``bot.strategy_state`` on the
        loop's first iteration (see ``_run_bot_loop``).

        Returns:
            Number of bots resumed
        """
        resumed = 0

        # Collect the bot ids in a short-lived session, then resume each bot in
        # ITS OWN session (M-3): a failure or mid-loop commit for one bot must
        # not poison the session used for the others.
        async with async_session_maker() as session:
            result = await session.execute(
                select(Bot.id).where(
                    Bot.status.in_([BotStatus.RUNNING, BotStatus.RECOVERY_MODE])
                )
            )
            bot_ids = [row[0] for row in result.all()]

        if not bot_ids:
            logger.info("No bots to resume on startup")
            return 0

        logger.info(f"Found {len(bot_ids)} bot(s) to resume")

        for bot_id in bot_ids:
            try:
                async with async_session_maker() as session:
                    bot = (
                        await session.execute(select(Bot).where(Bot.id == bot_id))
                    ).scalar_one_or_none()
                    if not bot:
                        continue

                    # Restore strategy state from database if available
                    await self._restore_strategy_state(bot)

                    # Create exchange service
                    if bot.is_dry_run:
                        exchange = self._make_simulated_exchange(bot.budget)
                        # CR-2: restore persisted simulated balances so a dry run
                        # resumes from where it left off instead of resetting to
                        # the initial budget (which would desync DB vs simulator).
                        # Also reconciles base holdings up to the open positions so
                        # a resumed exit can always settle.
                        await self._seed_sim_exchange(bot, exchange, session)
                    else:
                        exchange = ExchangeService()
                        if not exchange.has_credentials():
                            logger.error(
                                f"Bot {bot.id}: cannot resume live bot without "
                                "exchange credentials; marking STOPPED"
                            )
                            bot.status = BotStatus.STOPPED
                            await session.commit()
                            continue

                    if not await exchange.connect():
                        await exchange.disconnect()
                        logger.error(
                            f"Bot {bot.id}: exchange connection failed on resume; "
                            "marking STOPPED"
                        )
                        bot.status = BotStatus.STOPPED
                        await session.commit()
                        continue

                    self._exchange_services[bot.id] = exchange

                    # C2/H2: before resuming trading, reconcile orders against the
                    # exchange - resolve any orders left pending and import fills
                    # that never reached the database (e.g. a crash between fill
                    # and commit). This guarantees positions/orders are recovered
                    # before the strategy acts on possibly-stale local state.
                    await self._recover_bot_orders(bot, exchange, session)

                    # Initialize per-bot file logger
                    ensure_bot_log_directory(bot.id)
                    self._bot_loggers[bot.id] = BotLoggingService(
                        bot.id, bot.name, bot.is_dry_run
                    )
                    self._bot_loggers[bot.id].log_activity(
                        f"Bot resumed after server restart"
                    )

                    decision_status_store.update(
                        bot.id, DecisionState.EVALUATING,
                        reason="Bot resumed after restart", symbol=bot.trading_pair,
                    )

                    # Start bot task
                    self._stop_flags[bot.id] = False
                    task = asyncio.create_task(self._run_bot_loop(bot.id))
                    self._running_bots[bot.id] = task

                    resumed += 1
                    logger.info(f"Resumed bot {bot.id} ({bot.name})")

            except Exception as e:
                logger.error(f"Failed to resume bot {bot_id}: {e}")

        logger.info(f"Resumed {resumed} bot(s) on startup")
        return resumed

    async def _restore_strategy_state(self, bot) -> None:
        """Restore a bot's strategy runtime state on resume.

        Prefers the dedicated Bot.strategy_state column (current format). Falls
        back to legacy state previously embedded in strategy_params so bots saved
        by older builds still resume correctly.

        Args:
            bot: The Bot model being resumed.
        """
        strategy_state = getattr(bot, "strategy_state", None)
        if strategy_state:
            self._restore_bot_state(bot.id, strategy_state)
            return

        # Backward compatibility: older builds stored state inside strategy_params.
        self._restore_legacy_state(bot.id, bot.strategy_params)

    def _state_store(self, attr: str) -> dict:
        """Return (creating if needed) a per-bot state dict by attribute name."""
        store = getattr(self, attr, None)
        if store is None:
            store = {}
            setattr(self, attr, store)
        return store

    def _collect_bot_state(self, bot_id: int) -> dict:
        """Collect all persistable in-memory state for a bot into one dict.

        Returns JSON-safe data (datetimes tagged) ready to store in
        Bot.strategy_state.
        """
        state: Dict[str, Any] = {}
        for attr in _PERSISTED_STATE_ATTRS:
            store = getattr(self, attr, None)
            if isinstance(store, dict) and bot_id in store:
                state[attr] = store[bot_id]

        histories = getattr(self, "_price_histories", None)
        if isinstance(histories, dict) and bot_id in histories:
            state["_price_histories"] = histories[bot_id][-_PERSISTED_PRICE_HISTORY_LEN:]

        # CR-2: persist dry-run simulator balances so they survive a restart.
        exchange = self._exchange_services.get(bot_id)
        if exchange is not None and hasattr(exchange, "export_state"):
            state["_sim_state"] = exchange.export_state()

        # Recovery state lives in its own in-memory dict (self._recovery_states),
        # not in _PERSISTED_STATE_ATTRS, because it is written eagerly by
        # _enter_recovery_mode/_process_paper_trade on every transition rather
        # than collected here. It must still ride along in this snapshot: this
        # dict wholesale-replaces Bot.strategy_state (see _save_bot_state), so
        # omitting it would erase recovery_mode on the very next checkpoint.
        recovery = self._recovery_states.get(bot_id)
        if recovery is not None:
            state["recovery_mode"] = recovery

        return _to_jsonable(state)

    def _restore_bot_state(self, bot_id: int, strategy_state: dict) -> None:
        """Restore a bot's state from a saved strategy_state dict."""
        data = _from_jsonable(strategy_state)
        if not isinstance(data, dict):
            return

        for attr in _PERSISTED_STATE_ATTRS:
            if attr in data:
                self._state_store(attr)[bot_id] = data[attr]

        if "_price_histories" in data:
            self._state_store("_price_histories")[bot_id] = data["_price_histories"]

        logger.debug(f"Bot {bot_id}: Restored strategy state ({list(data.keys())})")

    def _restore_legacy_state(self, bot_id: int, strategy_params: dict) -> None:
        """Restore state from the legacy strategy_params embedding (pre-3)."""
        if not strategy_params:
            return

        for legacy_key, attr in _LEGACY_STATE_KEYS.items():
            if legacy_key in strategy_params:
                self._state_store(attr)[bot_id] = _from_jsonable(strategy_params[legacy_key])
                logger.debug(f"Bot {bot_id}: Restored legacy {legacy_key}")

        if "_price_history" in strategy_params:
            self._state_store("_price_histories")[bot_id] = strategy_params["_price_history"]
            logger.debug(f"Bot {bot_id}: Restored legacy price history")

    async def graceful_shutdown(self) -> int:
        """Perform graceful shutdown - save state and stop all bots.

        This stops all bot execution loops but keeps their status as RUNNING
        in the database so they can be resumed on next startup.

        Returns:
            Number of bots shut down
        """
        logger.info("Starting graceful shutdown...")
        shutdown_count = 0

        # Get list of running bots
        bot_ids = list(self._running_bots.keys())

        if not bot_ids:
            logger.info("No running bots to shut down")
            return 0

        logger.info(f"Shutting down {len(bot_ids)} bot(s)")

        # Save state for each bot before stopping
        async with async_session_maker() as session:
            for bot_id in bot_ids:
                try:
                    # Save strategy state to database
                    await self._save_bot_state(bot_id, session)
                    shutdown_count += 1
                except Exception as e:
                    logger.error(f"Failed to save state for bot {bot_id}: {e}")

            await session.commit()

        # Stop all bot loops (but don't change status)
        for bot_id in bot_ids:
            self._stop_flags[bot_id] = True

        # Wait for all tasks to complete
        tasks = list(self._running_bots.values())
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for bot tasks, cancelling...")
                for task in tasks:
                    task.cancel()

        # Disconnect exchange services
        for bot_id, exchange in list(self._exchange_services.items()):
            try:
                await exchange.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect exchange for bot {bot_id}: {e}")

        # Clear internal state
        self._running_bots.clear()
        self._exchange_services.clear()
        self._stop_flags.clear()

        logger.info(f"Graceful shutdown complete. Saved state for {shutdown_count} bot(s)")
        return shutdown_count

    async def _save_bot_state(self, bot_id: int, session: AsyncSession) -> None:
        """Save bot strategy state to database.

        Args:
            bot_id: Bot ID
            session: Database session
        """
        result = await session.execute(select(Bot).where(Bot.id == bot_id))
        bot = result.scalar_one_or_none()

        if not bot:
            return

        # Persist ALL strategy runtime state (trailing stops, locked entry ATR,
        # cooldowns, price history) into the dedicated strategy_state column.
        new_state = self._collect_bot_state(bot_id)

        # _collect_bot_state already carries recovery_mode when this process has
        # a live in-memory copy (self._recovery_states). In the narrow window
        # right after a restart — bot resumed into RECOVERY_MODE but the loop's
        # first-tick restore (see _run_bot_loop) hasn't repopulated
        # self._recovery_states yet — fall back to whatever is already
        # persisted so this checkpoint cannot erase it.
        existing = bot.strategy_state or {}
        if "recovery_mode" not in new_state and "recovery_mode" in existing:
            new_state["recovery_mode"] = existing["recovery_mode"]

        bot.strategy_state = new_state

        # M5: strategy_params is user config only. Strip any runtime state that
        # older builds may have embedded there so it cannot pollute config or
        # fail parameter validation on a later edit.
        if bot.strategy_params:
            cleaned = {k: v for k, v in bot.strategy_params.items() if not k.startswith("_")}
            if cleaned != bot.strategy_params:
                bot.strategy_params = cleaned

        bot.updated_at = self.clock.now()

        logger.debug(f"Saved state for bot {bot_id}")


# Global trading engine instance
trading_engine = TradingEngine()
