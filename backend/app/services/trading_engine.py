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
from .funding_diagnostic import compute_funding_stats
from .portfolio_risk import PortfolioRiskService
from .strategy_capacity import StrategyCapacityService
from .ledger_writer import LedgerWriterService
from .accounting import TradeRecorderService, FIFOTaxEngine, CSVExportService
from .ledger_invariants import LedgerInvariantService, ValidationError
from .decision_status import decision_status_store, DecisionState
from .diagnostics import (
    diagnostics_store,
    BLOCK_RISK_MANAGER,
    BLOCK_MIN_ORDER_SIZE,
    BLOCK_INSUFFICIENT_BALANCE,
    BLOCK_POSITION_LIMITS,
    BLOCK_OTHER,
    DATA_UNAVAILABLE,
)

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


# Repeated-rejection circuit breaker. When the SAME executable trade is rejected
# or fails for the SAME reason this many consecutive times, the bot is paused and
# the reason surfaced in Decision Status, instead of retrying the doomed action
# every tick forever (e.g. a sub-minimum order, or an un-settleable exit). This
# is independent of the loop-level failure breaker (which counts raised
# exceptions); a rejection is a clean None return, not an exception.
MAX_CONSECUTIVE_REJECTIONS = 5


# === STRATEGY STATE PERSISTENCE (C3/H1, M5) ===
# Per-bot, in-memory strategy state attributes that must survive a restart so
# resumed bots keep their risk state (trailing stops, locked entry ATR,
# cooldowns) and price history. Stored in Bot.strategy_state (a dedicated JSON
# column), NEVER in strategy_params (which is user config). Transient caches
# (e.g. _funding_cache) are intentionally excluded - they are safe to rebuild.
_PERSISTED_STATE_ATTRS = (
    "_grid_states",
    "_mean_reversion_states",
    "_trend_states",
    "_funding_states",
    "_volatility_breakout_states",
    "_twap_states",
    "_vwap_states",
    "_auto_states",
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
    "funding_carry",
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


def validate_funding_carry_params(params: dict) -> list:
    """Validate funding_carry strategy parameters.

    Returns a list of human-readable error strings (empty when valid). Kept as a
    pure function so it can be reused by configuration checks and tests without
    constructing an engine. Missing keys are valid (defaults apply).

    Args:
        params: The bot's strategy_params dict.
    """
    errors = []
    min_funding = params.get("min_funding_rate")
    max_funding = params.get("max_funding_rate")
    lookback = params.get("funding_lookback_periods")
    max_alloc = params.get("max_allocation_percent")
    cooldown = params.get("cooldown_seconds")
    refresh = params.get("funding_refresh_seconds")
    allowed = params.get("allowed_regimes")

    if (
        isinstance(min_funding, (int, float))
        and isinstance(max_funding, (int, float))
        and min_funding > max_funding
    ):
        errors.append(
            f"min_funding_rate ({min_funding}) must not exceed "
            f"max_funding_rate ({max_funding})"
        )
    if lookback is not None and (not isinstance(lookback, int) or lookback < 1):
        errors.append("funding_lookback_periods must be an integer >= 1")
    if max_alloc is not None and (
        not isinstance(max_alloc, (int, float)) or not 0 < max_alloc <= 100
    ):
        errors.append("max_allocation_percent must be in (0, 100]")
    if cooldown is not None and (not isinstance(cooldown, (int, float)) or cooldown < 0):
        errors.append("cooldown_seconds must be >= 0")
    if refresh is not None and (not isinstance(refresh, (int, float)) or refresh <= 0):
        errors.append("funding_refresh_seconds must be > 0")
    if allowed is not None and (
        not isinstance(allowed, list) or not all(isinstance(r, str) for r in allowed)
    ):
        errors.append("allowed_regimes must be a list of regime strings")

    return errors


class TradingEngine:
    """Engine for executing trading bots."""

    def __init__(self):
        """Initialize trading engine."""
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
            bot.started_at = datetime.utcnow()
            bot.paused_at = None
            bot.updated_at = datetime.utcnow()
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
                bot.paused_at = datetime.utcnow()
                bot.updated_at = datetime.utcnow()
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
            bot.updated_at = datetime.utcnow()
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

                    if not bot or bot.status != BotStatus.RUNNING:
                        logger.info(f"Bot {bot_id}: No longer running, stopping loop")
                        break

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
                        bot.paused_at = datetime.utcnow()
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
                        proceed = True
                        if signal.action == "buy":
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
                    now = datetime.utcnow()
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

        return await executor(bot, current_price, params, session)

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
            "funding_carry": self._strategy_funding_carry,
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

        Infinite accumulation strategy: Buys at regular clock-driven intervals
        regardless of price. Continues indefinitely until balance is exhausted
        or bot is manually stopped.

        IMPORTANT: This strategy is INFINITE by design. It will keep buying until:
        - Balance falls below minimum order size
        - Bot is manually stopped by operator
        This is intentional, not a bug.

        DCA NEVER SELLS. Exits are handled by other strategies or manual intervention.

        Regime-aware by default: Can pause during unfavorable market conditions
        (e.g., strong downtrends) to protect capital.

        WARNING: Do NOT use this strategy inside auto_mode. DCA is clock-driven
        and conflicts with regime-based strategy rotation.

        Parameters:
            interval_minutes: Time between buys (default: 60)
            amount_percent: Percent of budget per buy (default: 10)
            amount_usd: Fixed USD amount per buy (overrides amount_percent if set)
            immediate_first_buy: Execute first buy immediately (default: True)
            regime_filter_enabled: Enable regime filter to pause during bad conditions (default: True)
            allowed_regimes: List of allowed trend states (default: ["trend_up", "trend_flat"])
        """
        # Defensive check: Block usage inside auto_mode
        if bot.strategy == "auto_mode":
            logger.warning(
                f"Bot {bot.id}: DCA strategy invoked inside auto_mode. "
                f"This is NOT recommended - DCA conflicts with regime-based rotation."
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason="DCA: Not intended for use inside auto_mode"
            )

        interval_minutes = params.get("interval_minutes", 60)
        amount_percent = params.get("amount_percent", 10) / 100
        amount_usd = params.get("amount_usd")  # Fixed amount in USD
        immediate_first_buy = params.get("immediate_first_buy", True)
        regime_filter_enabled = params.get("regime_filter_enabled", True)
        allowed_regimes = params.get("allowed_regimes", ["trend_up", "trend_flat"])

        # === REGIME FILTER (pause during unfavorable conditions) ===
        # Protects capital by not buying during strong downtrends or adverse regimes
        if regime_filter_enabled:
            # Get price history for regime detection
            price_history = self._get_price_history(bot.id)

            # Add current price to history for regime detection
            price_history_with_current = price_history + [{
                "timestamp": datetime.utcnow().isoformat(),
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
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"DCA: Paused (regime={trend_regime_name}, waiting for {allowed_regimes})"
                )

        # Get order history for this bot
        last_order = await self._get_last_order(bot.id, session)
        order_count = await self._get_order_count(bot.id, session)

        # === TIME-BASED INTERVAL LOGIC (hardened) ===
        # Ensures clock-stable behavior: one buy max per interval, no catch-up
        interval_seconds = interval_minutes * 60
        now = datetime.utcnow()

        if last_order:
            # Defensive: If last order timestamp is in the future, treat as no last order
            # This handles clock skew or bad data gracefully
            if last_order.created_at > now:
                logger.warning(
                    f"Bot {bot.id}: DCA last order timestamp is in future "
                    f"({last_order.created_at} > {now}). Treating as no previous order."
                )
                last_order = None

        if last_order:
            time_since_last = now - last_order.created_at
            seconds_since_last = time_since_last.total_seconds()

            if seconds_since_last < interval_seconds:
                remaining_seconds = interval_seconds - seconds_since_last
                next_buy_time = last_order.created_at + timedelta(seconds=interval_seconds)
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"DCA: Next buy in {remaining_seconds/60:.1f} min (at {next_buy_time.strftime('%H:%M:%S')})"
                )
        elif not immediate_first_buy:
            # No orders yet but not immediate - check time since bot started
            # Defensive: If bot.started_at is None, allow immediate buy (treat as edge case)
            if bot.started_at:
                time_since_start = now - bot.started_at
                if time_since_start.total_seconds() < interval_seconds:
                    remaining_seconds = interval_seconds - time_since_start.total_seconds()
                    first_buy_time = bot.started_at + timedelta(seconds=interval_seconds)
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason=f"DCA: First buy in {remaining_seconds/60:.1f} min (at {first_buy_time.strftime('%H:%M:%S')})"
                    )
            else:
                # Edge case: bot.started_at is missing - allow immediate buy
                logger.info(
                    f"Bot {bot.id}: DCA bot.started_at is None. "
                    f"Allowing immediate first buy (edge case handling)."
                )

        # === AMOUNT CALCULATION ===
        # Cap is _BUY_BALANCE_FRACTION of balance (not the full balance) so the
        # simulated exchange fee (cost * 0.1 %) + bid/ask spread cannot push the
        # total deduction over the available funds.
        max_buy = bot.current_balance * _BUY_BALANCE_FRACTION
        if amount_usd and amount_usd > 0:
            # Use fixed USD amount
            buy_amount = min(amount_usd, max_buy)
        else:
            # Use percentage of current balance
            buy_amount = bot.current_balance * amount_percent

        # === MINIMUM ORDER FLOOR (executable, shared $10 minimum) ===
        # A DCA buy must clear the same MIN_ORDER_USD the execution layer enforces
        # (_execute_trade STEP 6). Emitting a sub-minimum buy here would be
        # rejected downstream every tick without ever recording an order, so the
        # interval gate never advances and the same "$9.95 REJECTED" repeats
        # forever. Floor up to the minimum when affordable; otherwise HOLD - the
        # infinite accumulation has reached its natural end (budget exhausted).
        if buy_amount < MIN_ORDER_USD:
            if bot.current_balance >= MIN_ORDER_USD:
                buy_amount = MIN_ORDER_USD
            else:
                logger.info(
                    f"Bot {bot.id}: DCA infinite accumulation complete - "
                    f"Balance ${bot.current_balance:.2f} < ${MIN_ORDER_USD:.0f} minimum order"
                )
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=(
                        f"DCA: balance ${bot.current_balance:.2f} below "
                        f"${MIN_ORDER_USD:.0f} minimum order (accumulation complete)"
                    ),
                )

        # Defensive: Cap at fee-adjusted balance ceiling so cost + fee cannot
        # exceed available funds regardless of the amount_percent path above.
        if buy_amount > max_buy:
            buy_amount = max_buy

        # === EXECUTE BUY (infinite accumulation continues) ===
        logger.info(
            f"Bot {bot.id}: DCA buy #{order_count + 1} - "
            f"${buy_amount:.2f} ({"fixed" if amount_usd else f"{amount_percent*100:.1f}%"}) "
            f"at ${current_price:.2f} | "
            f"Balance: ${bot.current_balance:.2f} → ${bot.current_balance - buy_amount:.2f}"
        )

        return TradeSignal(
            action="buy",
            amount=buy_amount,
            order_type="market",
            reason=f"DCA buy #{order_count + 1}: ${buy_amount:.2f} @ ${current_price:.2f}",
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

        DESIGN PHILOSOPHY:
        - Grid is a MANUFACTURING PROCESS: converts cash to crypto at favorable prices
        - Long-biased for crypto: more buy levels below, fewer sell levels above
        - Capital-bounded: hard kill switches prevent runaway losses
        - Regime-aware: pauses in trends/high volatility, operates in flat/normal markets
        - Depth-aware sizing: larger orders at deeper discounts (convex payoff)
        - Bar-based: one order max per bar (no cascading, no tick noise)
        - Fast exits: admits failure quickly via kill switches

        RISK CONTROLS:
        1. Max drawdown kill switch (% of initial capital)
        2. ATR distance kill switch (price escapes grid range)
        3. Regime gating (only operates in trend_flat + volatility_normal)
        4. Re-centering logic when price escapes range

        CRITICAL: All logic operates on AGGREGATED PSEUDO-BARS, not tick data.
        Bar interval defines time granularity (default: 60 seconds per bar).

        Parameters:
            bar_interval_seconds: Seconds per bar for aggregation (default: 60)
            grid_count: Total grid levels (default: 10, long-biased split: 7 buy / 3 sell)
            grid_spacing_percent: Spacing between adjacent levels % (default: 1.0)
            range_percent: Total grid range % from center (default: 10)
            base_order_size_percent: Base order size % of budget (default: 5)
            depth_multiplier: Multiplier for deeper levels (default: 1.5, convex sizing)
            max_drawdown_percent: Max drawdown % before kill (default: 15)
            kill_atr_multiplier: ATR distance for kill switch (default: 3.0)
            atr_period: ATR period for kill switch (default: 14 bars)
            regime_filter_enabled: Enable regime gating (default: True)
            allowed_regimes: Allowed regimes (default: ["trend_flat", "volatility_normal"])
            cooldown_after_kill_hours: Hours to wait after kill switch (default: 2)
        """
        # === PARAMETER EXTRACTION ===
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        grid_count = params.get("grid_count", 10)
        grid_spacing_pct = params.get("grid_spacing_percent", 0.3) / 100
        range_pct = params.get("range_percent", 10) / 100
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
                "last_recenter_time": datetime.utcnow(),
                "total_trades": 0,
                # TELEMETRY METRICS (for dashboards and auto-mode learning)
                "lifetime_return_pct": 0.0,  # Total return since inception (%)
                "lifetime_max_drawdown_pct": 0.0,  # Worst drawdown ever experienced (%)
                # COOLDOWN TRACKING (prevents immediate re-entry after kill)
                "last_kill_switch_time": None,  # Timestamp of last kill switch activation
                "kill_switch_count": 0,  # Number of times kill switch has fired
                # ATR LOCKING (refreshed on recenter)
                "atr_at_recenter": None,  # ATR value captured at last recenter
            })
            logger.info(
                f"Bot {bot.id}: Adaptive Grid initialized - "
                f"Center: ${current_price:.2f}, Capital: ${bot.budget:.2f}, "
                f"Long-biased: {buy_levels} buy / {sell_levels} sell levels"
            )

        # === BAR AGGREGATION (60-second bars) ===
        now = datetime.utcnow()
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
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: Cooldown after kill ({remaining_minutes:.0f}min remaining)"
                )
            else:
                # Cooldown expired, clear the timestamp
                logger.info(
                    f"Bot {bot.id}: Grid cooldown expired - "
                    f"Resuming normal operation after {cooldown_after_kill_hours}h pause"
                )
                state["last_kill_switch_time"] = None

        # === REGIME GATING (operate only in flat/normal markets) ===
        if regime_filter_enabled:
            # Build price history from bars for regime detection
            price_history_from_bars = [{"price": b["close"], "timestamp": b["start_ts"]} for b in completed_bars]
            price_history_with_current = price_history_from_bars + [{"price": current_price, "timestamp": now}]

            current_regime = self._detect_market_regime(price_history_with_current, None)
            trend_state = current_regime.get("trend_state", "flat")
            volatility_state = current_regime.get("volatility_state", "medium")

            # Map to user-friendly names
            regime_names = [f"trend_{trend_state}", f"volatility_{volatility_state}"]

            # Check if ANY required regime is met (OR logic across trend/volatility)
            regime_allowed = any(r in allowed_regimes for r in regime_names)

            if not regime_allowed:
                logger.info(
                    f"Bot {bot.id}: Grid PAUSED (regime unsuitable) - "
                    f"Current: {regime_names}, Allowed: {allowed_regimes}. "
                    f"Grid operates in flat/normal markets only (range-bound conditions)."
                )
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: Paused (regime={regime_names}, need flat/normal markets)"
                )

        # === CALCULATE ATR (for kill switch) ===
        atr = None
        if len(completed_bars) >= atr_period:
            atr = self.calculate_atr_proxy(completed_bars, atr_period)

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

            logger.warning(
                f"Bot {bot.id}: Grid KILL SWITCH #{state['kill_switch_count']} ACTIVATED (drawdown) - "
                f"Drawdown: {drawdown*100:.1f}% > Max: {max_drawdown_pct*100:.1f}%. "
                f"Grid admits failure. Liquidating all virtual positions. "
                f"Cooldown: {cooldown_after_kill_hours}h"
            )
            # Reset grid (liquidate virtual positions, restart)
            state["virtual_cash"] = current_portfolio_value
            state["virtual_crypto"] = 0.0
            state["center_price"] = bar_close_price
            state["grid_levels"] = {}
            state["peak_portfolio_value"] = current_portfolio_value
            state["last_recenter_time"] = now
            self._save_grid_state(bot.id, state)

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Grid: Kill switch (drawdown {drawdown*100:.1f}%), {cooldown_after_kill_hours}h cooldown"
            )

        # === KILL SWITCH 2: ATR DISTANCE (range escape with re-centering) ===
        center_price = state["center_price"]
        if atr is not None:
            atr_distance = abs(bar_close_price - center_price)
            kill_distance = atr * kill_atr_mult

            if atr_distance > kill_distance:
                # Activate cooldown and track kill switch event
                state["last_kill_switch_time"] = now
                state["kill_switch_count"] = state.get("kill_switch_count", 0) + 1

                # Lock ATR at recenter (ensures fresh baseline for new grid)
                state["atr_at_recenter"] = atr

                logger.warning(
                    f"Bot {bot.id}: Grid KILL SWITCH #{state['kill_switch_count']} ACTIVATED (range escape) - "
                    f"Price ${bar_close_price:.2f} is {atr_distance:.2f} from center ${center_price:.2f} "
                    f"(> {kill_atr_mult}x ATR = {kill_distance:.2f}). Re-centering grid. "
                    f"ATR locked at {atr:.2f}. Cooldown: {cooldown_after_kill_hours}h"
                )
                # Re-center grid
                state["center_price"] = bar_close_price
                state["grid_levels"] = {}
                state["last_recenter_time"] = now
                self._save_grid_state(bot.id, state)

                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: Re-centered at ${bar_close_price:.2f} (range escape), {cooldown_after_kill_hours}h cooldown"
                )

        # === ONE ORDER PER BAR CHECK ===
        last_order_bar = state.get("last_order_bar")
        if last_order_bar is not None and last_order_bar == completed_bars[-1]["start_ts"]:
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Grid: Already traded this bar (one order per bar limit)"
            )

        # === CALCULATE GRID LEVELS (if not set or needs refresh) ===
        if not state["grid_levels"]:
            grid_levels = {}

            # Create buy levels (below center, long-biased)
            for i in range(1, buy_levels + 1):
                level_price = center_price * (1 - grid_spacing_pct * i)
                grid_levels[-i] = {
                    "price": level_price,
                    "side": "buy",
                    "filled": False,
                    "depth": i,  # Track depth for sizing
                }

            # Create sell levels (above center)
            for i in range(1, sell_levels + 1):
                level_price = center_price * (1 + grid_spacing_pct * i)
                grid_levels[i] = {
                    "price": level_price,
                    "side": "sell",
                    "filled": False,
                    "depth": i,
                }

            state["grid_levels"] = grid_levels
            logger.info(
                f"Bot {bot.id}: Grid levels created - "
                f"{buy_levels} buy levels below ${center_price:.2f}, "
                f"{sell_levels} sell levels above"
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
            logger.info(
                f"Bot {bot.id}: Grid no-trade diag - close=${bar_close_price:.2f} "
                f"center=${center_price:.2f} nearest_buy=${nearest_buy or 0:.2f} "
                f"(price {buy_gap} vs nearest buy) virtual_cash=${state['virtual_cash']:.2f} "
                f"virtual_crypto={state['virtual_crypto']:.6f}"
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Grid: No levels triggered this bar"
            )

        # === EXECUTE ORDER AT NEAREST LEVEL ===
        level_data = grid_levels[nearest_level]
        depth = level_data["depth"]

        # Depth-aware sizing: larger orders at deeper discounts (convex payoff)
        # Example: depth=1 → 1.0x, depth=2 → 1.5x, depth=3 → 2.25x, etc.
        size_multiplier = depth_multiplier ** (depth - 1)
        order_size_usd = min(
            initial_capital * base_order_size_pct * size_multiplier,
            bot.current_balance * _BUY_BALANCE_FRACTION,  # never exceed real funds
        )

        if level_data["side"] == "buy":
            # === BUY ORDER (accumulate crypto) ===
            # Check virtual wallet has cash
            if state["virtual_cash"] < order_size_usd:
                logger.info(
                    f"Bot {bot.id}: Grid BUY skipped at level {nearest_level} - "
                    f"Insufficient virtual cash: ${state['virtual_cash']:.2f} < ${order_size_usd:.2f}"
                )
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: Insufficient virtual cash for buy at level {nearest_level}"
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
                f"(size_mult={size_multiplier:.2f}x), Virtual: ${state['virtual_cash']:.2f} cash + "
                f"{state['virtual_crypto']:.4f} crypto | "
                f"Lifetime: {state['lifetime_return_pct']:+.2f}% return, "
                f"{state['lifetime_max_drawdown_pct']:.2f}% max DD"
            )

            return TradeSignal(
                action="buy",
                amount=order_size_usd,
                order_type="market",
                reason=f"Grid: Buy at level {nearest_level} (${bar_close_price:.2f}, depth={depth})",
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
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: Insufficient virtual crypto for sell at level {nearest_level}"
                )

            # Execute virtual sell
            state["virtual_crypto"] -= crypto_to_sell
            state["virtual_cash"] += order_size_usd
            state["grid_levels"][nearest_level]["filled"] = True
            state["last_order_bar"] = completed_bars[-1]["start_ts"]
            state["total_trades"] += 1

            self._save_grid_state(bot.id, state)

            logger.info(
                f"Bot {bot.id}: Grid SELL at level {nearest_level} (depth={depth}) - "
                f"Price: ${bar_close_price:.2f}, Amount: ${order_size_usd:.2f} "
                f"(size_mult={size_multiplier:.2f}x), Virtual: ${state['virtual_cash']:.2f} cash + "
                f"{state['virtual_crypto']:.4f} crypto | "
                f"Lifetime: {state['lifetime_return_pct']:+.2f}% return, "
                f"{state['lifetime_max_drawdown_pct']:.2f}% max DD"
            )

            return TradeSignal(
                action="sell",
                amount=order_size_usd,
                order_type="market",
                reason=f"Grid: Sell at level {nearest_level} (${bar_close_price:.2f}, depth={depth})",
            )

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

    async def _strategy_mean_reversion(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Mean Reversion strategy - Institutional Grade.

        ANTI-TREND, BOUNDED-RISK, REGIME-AWARE strategy designed for range-bound markets.
        Takes quick profits on mean reversion, exits aggressively when wrong.

        NOT designed to hold through trends. Regime gating force-exits on trend flips.

        Uses bar-aggregated Bollinger Bands with hard stop (locked entry_atr) and
        time stop (max_hold_bars) to bound downside risk.

        CRITICAL: All logic operates on AGGREGATED PSEUDO-BARS, not tick data.
        Bar interval defines time granularity (default: 60 seconds per bar).

        Regime-aware: Only allowed in trend_flat and volatility_high regimes.
        Force-exits immediately if regime flips to trend_up or trend_down.

        This is a DEFENSIVE STATISTICAL STRATEGY, not a conviction trade.

        Parameters:
            bar_interval_seconds: Time per bar for aggregation (default: 60)
            bollinger_period: Bollinger Band period in bars (default: 20)
            bollinger_std: Standard deviation multiplier (default: 2.0)
            atr_period: ATR period for hard stop (default: 14)
            atr_stop_multiplier: ATR stop multiplier (default: 2.0)
            max_hold_bars: Maximum bars to hold position (time stop, default: 10)
            order_size_percent: Percent of balance per order (default: 20)
            exit_at_mean: Exit at mean vs upper band (default: True)
            regime_filter_enabled: Enable regime gating (default: True)
            cooldown_seconds: Seconds between trades (default: 300)
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
            "hard_stop": None,  # ATR-based hard stop
            "bars_since_entry": 0,  # Time stop counter
            "last_exit_time": None,  # For cooldown
        })

        now = datetime.utcnow()

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

        # === REGIME GATING (MANDATORY FOR MEAN REVERSION) ===
        # Mean reversion only allowed in: trend_flat, volatility_high
        # Force-exit immediately if regime flips to trend_up or trend_down
        allowed_regimes = ["trend_flat", "volatility_high"]

        regime_allows_entry = True
        force_exit_regime = False
        regime_name = "regime_filter_disabled"

        if regime_filter_enabled:
            # Detect regime from THIS strategy's own completed bar closes. The
            # shared tick price-history buffer (_get_price_history) is only
            # populated by trend_following/funding_carry, so a standalone
            # mean-reversion bot fed it an empty series -> _detect_market_regime
            # returned the neutral 'flat/medium' default forever. That silently
            # disabled BOTH the regime entry gate AND the trend force-exit
            # (mean reversion would happily enter and hold through a downtrend).
            # By here we already have >= period bars (checked above), so the
            # detector has enough data.
            bar_closes = [b["close"] for b in state["bars"]] + [current_price]
            current_regime = self._detect_market_regime(bar_closes, None)
            trend_state = current_regime.get("trend_state", "flat")
            volatility_state = current_regime.get("volatility_state", "medium")

            # Map to regime names
            trend_regime = f"trend_{trend_state}"
            volatility_regime = f"volatility_{volatility_state}" if volatility_state == "high" else None

            # Check if allowed
            regime_allows_entry = (trend_regime in allowed_regimes or
                                   (volatility_regime and volatility_regime in allowed_regimes))

            # Force exit if trending (mean reversion not suitable)
            force_exit_regime = trend_state in ["up", "down"]
            regime_name = f"{trend_regime}, vol={volatility_state}"

            logger.info(
                f"Bot {bot.id}: Mean Reversion regime - {regime_name}, "
                f"Entry allowed: {regime_allows_entry}, Force exit: {force_exit_regime}"
            )

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

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
                    datetime.utcnow() - state["last_exit_time"]).total_seconds()))
            logger.info(
                f"Bot {bot.id}: MR no-trade diag - close=${last_bar_close:.2f} "
                f"lower_band=${lower_band:.2f} gap_to_entry={entry_gap_pct:+.3f}% "
                f"regime={regime_name} entry_allowed={regime_allows_entry} "
                f"cooldown={cd_remaining}s has_position={has_position} bars={len(state['bars'])}"
            )

        # === POSITION EXIT LOGIC (BOUNDED RISK) ===
        # Exits: Mean reached | Hard stop | Time stop | Regime flip
        if has_position:
            for pos in positions:
                # Increment bars_since_entry only when bar completes
                if bar_completed and state["entry_price"] is not None:
                    state["bars_since_entry"] += 1

                # CRITICAL: Use LOCKED entry_atr (risk never expands)
                entry_atr_locked = state.get("entry_atr", atr)  # Fallback for legacy positions
                hard_stop = state.get("hard_stop", None)

                # === EXIT CONDITION 1: Regime Flip (FORCE EXIT) ===
                # If market starts trending, mean reversion is wrong - exit immediately
                if force_exit_regime:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Mean Reversion EXIT (regime flip) - "
                        f"Regime: {regime_name}, Mean reversion not suitable for trends"
                    )

                    # Clear state
                    state["entry_price"] = None
                    state["entry_atr"] = None
                    state["hard_stop"] = None
                    state["bars_since_entry"] = 0
                    state["last_exit_time"] = datetime.utcnow()
                    self._mean_reversion_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Mean Reversion: Force exit (regime={regime_name})",
                    )

                # === EXIT CONDITION 2: Mean Reached (TARGET) ===
                # Determine exit level
                if exit_at_mean:
                    exit_level = sma
                    exit_label = "mean"
                else:
                    exit_level = upper_band
                    exit_label = "upper band"

                if last_bar_close >= exit_level:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Mean Reversion EXIT (target reached) - "
                        f"Bar close ${last_bar_close:.2f} >= {exit_label} ${exit_level:.2f}"
                    )

                    # Clear state
                    state["entry_price"] = None
                    state["entry_atr"] = None
                    state["hard_stop"] = None
                    state["bars_since_entry"] = 0
                    state["last_exit_time"] = datetime.utcnow()
                    self._mean_reversion_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Mean Reversion: {exit_label} reached (${exit_level:.2f})",
                    )

                # === EXIT CONDITION 3: Hard Stop (ATR-BASED) ===
                # Stop is locked at entry, never widens
                if hard_stop is not None and current_price <= hard_stop:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Mean Reversion EXIT (hard stop) - "
                        f"Price ${current_price:.2f} <= Stop ${hard_stop:.2f}, "
                        f"Entry ATR (locked): ${entry_atr_locked:.4f}"
                    )

                    # Clear state
                    state["entry_price"] = None
                    state["entry_atr"] = None
                    state["hard_stop"] = None
                    state["bars_since_entry"] = 0
                    state["last_exit_time"] = datetime.utcnow()
                    self._mean_reversion_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Mean Reversion: Hard stop (${hard_stop:.2f})",
                    )

                # === EXIT CONDITION 4: Time Stop (MAX HOLD BARS) ===
                # If mean not reached within N bars, exit anyway (bounded holding)
                if state["bars_since_entry"] >= max_hold_bars:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Mean Reversion EXIT (time stop) - "
                        f"Held {state['bars_since_entry']} bars >= max {max_hold_bars}"
                    )

                    # Clear state
                    state["entry_price"] = None
                    state["entry_atr"] = None
                    state["hard_stop"] = None
                    state["bars_since_entry"] = 0
                    state["last_exit_time"] = datetime.utcnow()
                    self._mean_reversion_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Mean Reversion: Time stop ({state['bars_since_entry']} bars)",
                    )

            # Update state and hold
            self._mean_reversion_states[bot.id] = state

            # A conditional inside the f-string format spec ({x:.2f if ...}) is a
            # ValueError ("Invalid format specifier"); format the optional stop
            # outside the f-string so a held position never raises here.
            stop_str = f"${hard_stop:.2f}" if hard_stop is not None else "N/A"
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Holding, target ${exit_level:.2f}, stop {stop_str}, bars {state['bars_since_entry']}/{max_hold_bars}"
            )

        # === ENTRY LOGIC (BAR-BASED) ===
        # Entry: bar close <= lower BB, regime allowed, cooldown elapsed

        # Regime gate: Block entries if wrong market conditions
        if not regime_allows_entry:
            self._mean_reversion_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Waiting for suitable regime (current: {regime_name})"
            )

        # Cooldown check
        if state["last_exit_time"] is not None:
            time_since_exit = (datetime.utcnow() - state["last_exit_time"]).total_seconds()
            if time_since_exit < cooldown_seconds:
                remaining = int(cooldown_seconds - time_since_exit)
                self._mean_reversion_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Mean Reversion: Cooldown ({remaining}s remaining)"
                )

        # Entry condition: bar close <= lower Bollinger Band
        if last_bar_close <= lower_band:
            # Fixed percentage position sizing, capped at _BUY_BALANCE_FRACTION
            # of balance so the simulated exchange fee cannot push cost + fee
            # over available funds, then floored to the executable minimum.
            buy_amount = min(
                bot.current_balance * order_size_percent,
                bot.current_balance * _BUY_BALANCE_FRACTION,
            )

            if buy_amount < MIN_ORDER_USD:
                if bot.current_balance >= MIN_ORDER_USD:
                    buy_amount = MIN_ORDER_USD
                else:
                    self._mean_reversion_states[bot.id] = state
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason=(
                            f"Mean Reversion: balance ${bot.current_balance:.2f} "
                            f"below ${MIN_ORDER_USD:.0f} minimum order"
                        ),
                    )

            logger.info(
                f"Bot {bot.id}: Mean Reversion ENTRY - "
                f"Bar close ${last_bar_close:.2f} <= lower BB ${lower_band:.2f}, "
                f"Entry ATR locked at ${atr:.4f}, Position: ${buy_amount:.2f}"
            )

            # Initialize state with LOCKED entry_atr and hard stop
            state["entry_price"] = current_price
            state["entry_atr"] = atr  # LOCKED - stop distance will always use this
            state["hard_stop"] = current_price - (atr * atr_stop_mult)
            state["bars_since_entry"] = 0
            self._mean_reversion_states[bot.id] = state

            return TradeSignal(
                action="buy",
                amount=buy_amount,
                order_type="market",
                reason=f"Mean Reversion: Entry at lower band (${lower_band:.2f})",
            )

        # Update state and hold
        self._mean_reversion_states[bot.id] = state

        return TradeSignal(
            action="hold",
            amount=0,
            reason=f"Mean Reversion: Waiting for lower band (current: ${last_bar_close:.2f}, target: ${lower_band:.2f})"
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
        now = datetime.utcnow()
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

    # Note: _strategy_momentum (breakdown_momentum) was removed (stub implementation, overlaps with other strategies)

    async def _strategy_trend_following(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Trend Following (time-series momentum) strategy - Institutional Grade.

        Conservative long-only momentum strategy using EMA crossover and ATR-based stops.
        Hardened with noise-resistant entry/exit confirmation, locked entry ATR (prevents
        risk expansion), and re-entry cooldown (prevents churn).

        Entry: price > EMA(long) AND EMA(short) > EMA(long), confirmed over N loops
        Exit: price < EMA(long) (confirmed) OR trailing stop hit (immediate)
        Trailing stop distance is LOCKED at entry ATR to ensure risk never increases.

        Parameters:
            short_period: EMA short period (default: 50)
            long_period: EMA long period (default: 200)
            atr_period: ATR period (default: 14)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
            risk_percent: Percent of capital to risk per trade (default: 1.0)
            entry_confirmation_loops: Consecutive loops required for entry (default: 3)
            exit_confirmation_loops: Consecutive loops required for exit (default: 2)
            cooldown_seconds: Seconds to wait after exit before re-entry (default: 300)
        """
        short_period = params.get("short_period", 50)
        long_period = params.get("long_period", 100)
        atr_period = params.get("atr_period", 14)
        atr_multiplier = params.get("atr_multiplier", 2.0)
        risk_percent = params.get("risk_percent", 1.0) / 100
        entry_confirmation_loops = params.get("entry_confirmation_loops", 3)
        exit_confirmation_loops = params.get("exit_confirmation_loops", 2)
        cooldown_seconds = params.get("cooldown_seconds", 300)  # 5 minutes default

        # Get price history
        price_history = self._get_price_history(bot.id)

        # Add current price to history
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history, max_len=max(long_period + 50, 250))

        # Need enough data for EMA calculation
        if len(price_history) < long_period:
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

        # Calculate ATR proxy (Average True Range approximation)
        def calculate_atr_proxy(prices: list, period: int) -> float:
            """Calculate ATR proxy from price-only data.

            Since only prices are available (no OHLC), we approximate True Range
            as the absolute difference between consecutive prices.
            This is NOT a true ATR but serves as a reasonable volatility proxy
            for tick-based data.

            Returns 0 if insufficient data.
            """
            if len(prices) < period + 1:
                return 0.0

            # Calculate true range approximations: |current_price - previous_price|
            true_ranges = []
            for i in range(1, len(prices)):
                tr = abs(prices[i] - prices[i-1])
                true_ranges.append(tr)

            # Return rolling average over last 'period' true ranges
            if len(true_ranges) < period:
                return 0.0

            recent_trs = true_ranges[-period:]
            return sum(recent_trs) / len(recent_trs)

        # Calculate indicators
        ema_short = calculate_ema(price_history, short_period)
        ema_long = calculate_ema(price_history, long_period)
        atr = calculate_atr_proxy(price_history, atr_period)

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # Get or initialize trend-following state
        # State tracks: trailing stop, entry ATR (locked at entry to prevent risk expansion),
        # entry/exit confirmation counters (noise defense), and cooldown timer (anti-churn)
        if not hasattr(self, "_trend_states"):
            self._trend_states = {}

        state = self._trend_states.get(bot.id, {
            "trailing_stop": None,
            "highest_price": None,
            "entry_atr": None,  # Locked at entry - risk must never increase
            "entry_time": None,
            "last_exit_time": None,
            "entry_confirmation_count": 0,  # Consecutive loops with entry conditions met
            "exit_confirmation_count": 0,   # Consecutive loops with exit conditions met
        })

        logger.debug(
            f"Bot {bot.id}: Trend Following - Price: ${current_price:.2f}, "
            f"EMA({short_period}): ${ema_short:.2f}, EMA({long_period}): ${ema_long:.2f}, "
            f"ATR: ${atr:.2f}"
        )

        # Trading logic
        if not has_position:
            # No position - look for entry signal

            # Check re-entry cooldown (anti-churn protection)
            if state["last_exit_time"] is not None:
                time_since_exit = (datetime.utcnow() - state["last_exit_time"]).total_seconds()
                if time_since_exit < cooldown_seconds:
                    remaining = int(cooldown_seconds - time_since_exit)
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason=f"Trend Following: Re-entry cooldown active ({remaining}s remaining)"
                    )

            # Entry conditions: price > EMA(long) AND EMA(short) > EMA(long)
            entry_conditions_met = current_price > ema_long and ema_short > ema_long

            if entry_conditions_met:
                # Entry hysteresis: require consecutive confirmations (noise defense)
                state["entry_confirmation_count"] = state.get("entry_confirmation_count", 0) + 1
                self._trend_states[bot.id] = state

                if state["entry_confirmation_count"] < entry_confirmation_loops:
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason=f"Trend Following: Entry confirmation {state['entry_confirmation_count']}/{entry_confirmation_loops}"
                    )

                # Confirmed entry - proceed.
                # Volatility-adjusted position sizing: risk a fixed % of capital;
                # the ATR-based stop distance determines how many COINS that risk
                # buys, which we convert to a quote-notional order.
                #
                # BUGFIX: the conversion to notional (* current_price) was
                # missing, so position_size was a coin count used as if it were
                # USD. On a high-priced asset that produced sub-$1 "orders"
                # (rejected as < $10 minimum), and when ATR collapsed toward 0 it
                # blew up and was silently capped at the whole balance.
                risk_amount = bot.current_balance * risk_percent

                if atr > 0:
                    stop_distance = atr * atr_multiplier           # USD per coin
                    position_coins = risk_amount / stop_distance   # base coins
                    position_size = position_coins * current_price  # quote USD notional
                else:
                    # ATR unavailable (flat/illiquid): risk the nominal amount
                    # rather than dividing by ~0.
                    position_size = risk_amount

                # Never deploy more than the available balance, less the
                # execution cost buffer (fee + spread) so the simulated exchange
                # cannot reject the order for insufficient funds.
                buy_amount = min(position_size, bot.current_balance * _BUY_BALANCE_FRACTION)

                # Minimum-order floor: a risk-based size below the exchange
                # minimum cannot execute. Floor to the minimum when the balance
                # can afford it; otherwise HOLD with a clear reason instead of
                # emitting a doomed sub-minimum order that the engine rejects
                # every loop (which presents as a stuck bot).
                if buy_amount < MIN_ORDER_USD:
                    if bot.current_balance >= MIN_ORDER_USD:
                        buy_amount = MIN_ORDER_USD
                    else:
                        return TradeSignal(
                            action="hold",
                            amount=0,
                            reason=(
                                f"Trend Following: balance ${bot.current_balance:.2f} "
                                f"below ${MIN_ORDER_USD:.0f} minimum order"
                            ),
                        )

                logger.info(
                    f"Bot {bot.id}: Trend Following ENTRY - "
                    f"Price ${current_price:.2f} > EMA({long_period}) ${ema_long:.2f}, "
                    f"EMA({short_period}) ${ema_short:.2f} > EMA({long_period}), "
                    f"Entry ATR: ${atr:.4f}, Position: ${buy_amount:.2f}"
                )

                # Initialize state with LOCKED entry_atr (risk must never increase)
                trailing_stop_price = current_price - (atr * atr_multiplier)
                self._trend_states[bot.id] = {
                    "trailing_stop": trailing_stop_price,
                    "highest_price": current_price,
                    "entry_atr": atr,  # LOCKED - trailing stop distance will always use this
                    "entry_time": datetime.utcnow(),
                    "last_exit_time": None,
                    "entry_confirmation_count": 0,
                    "exit_confirmation_count": 0,
                }

                return TradeSignal(
                    action="buy",
                    amount=buy_amount,
                    order_type="market",
                    reason=f"Trend Following: Bullish trend confirmed ({entry_confirmation_loops} loops)",
                )
            else:
                # Entry conditions not met - reset confirmation counter
                state["entry_confirmation_count"] = 0
                self._trend_states[bot.id] = state

            # Check entry conditions and provide feedback
            if current_price <= ema_long:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Trend Following: Price ${current_price:.2f} below EMA({long_period}) ${ema_long:.2f}"
                )
            elif ema_short <= ema_long:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Trend Following: Waiting for EMA crossover (short ${ema_short:.2f} <= long ${ema_long:.2f})"
                )
            else:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="Trend Following: Waiting for entry conditions"
                )

        else:
            # Have position - manage exit
            for pos in positions:
                # CRITICAL: Use LOCKED entry_atr for trailing stop distance (risk must never increase)
                entry_atr_locked = state.get("entry_atr", atr)  # Fallback to current ATR for legacy positions

                # Update trailing stop if price made new high
                # Trailing stop distance is ALWAYS based on entry_atr, not current ATR
                if state["highest_price"] is None or current_price > state["highest_price"]:
                    state["highest_price"] = current_price
                    state["trailing_stop"] = current_price - (entry_atr_locked * atr_multiplier)
                    state["exit_confirmation_count"] = 0  # Reset exit confirmation on new high
                    self._trend_states[bot.id] = state

                # Exit condition 1: Trailing stop hit (hard stop - no confirmation needed)
                if state["trailing_stop"] is not None and current_price <= state["trailing_stop"]:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Trend Following EXIT (trailing stop) - "
                        f"Price ${current_price:.2f} <= Stop ${state['trailing_stop']:.2f}, "
                        f"Entry ATR: ${entry_atr_locked:.4f}"
                    )

                    # Set last_exit_time for cooldown, reset state
                    self._trend_states[bot.id] = {
                        "trailing_stop": None,
                        "highest_price": None,
                        "entry_atr": None,
                        "entry_time": None,
                        "last_exit_time": datetime.utcnow(),
                        "entry_confirmation_count": 0,
                        "exit_confirmation_count": 0,
                    }

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Trend Following: Exit on trailing stop (${state['trailing_stop']:.2f})",
                    )

                # Exit condition 2: Price below EMA(long) - trend break (requires confirmation)
                if current_price < ema_long:
                    # Exit confirmation: require consecutive loops (anti-whipsaw)
                    state["exit_confirmation_count"] = state.get("exit_confirmation_count", 0) + 1
                    self._trend_states[bot.id] = state

                    if state["exit_confirmation_count"] < exit_confirmation_loops:
                        return TradeSignal(
                            action="hold",
                            amount=0,
                            reason=f"Trend Following: Exit confirmation {state['exit_confirmation_count']}/{exit_confirmation_loops} (price < EMA)"
                        )

                    # Confirmed exit
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Trend Following EXIT (trend break confirmed) - "
                        f"Price ${current_price:.2f} < EMA({long_period}) ${ema_long:.2f}, "
                        f"Confirmed over {exit_confirmation_loops} loops"
                    )

                    # Set last_exit_time for cooldown, reset state
                    self._trend_states[bot.id] = {
                        "trailing_stop": None,
                        "highest_price": None,
                        "entry_atr": None,
                        "entry_time": None,
                        "last_exit_time": datetime.utcnow(),
                        "entry_confirmation_count": 0,
                        "exit_confirmation_count": 0,
                    }

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Trend Following: Exit on confirmed trend break (price < EMA({long_period}))",
                    )
                else:
                    # Price still above EMA(long) - reset exit confirmation
                    state["exit_confirmation_count"] = 0
                    self._trend_states[bot.id] = state

            # Hold position - trend still valid.
            # A conditional inside an f-string format spec ({x:.2f if ...}) is a
            # ValueError ("Invalid format specifier"), raised on EVERY hold tick
            # once a position is open -> the failure breaker pauses the bot.
            # Format the optional stop outside the f-string.
            stop_str = (
                f"${state['trailing_stop']:.2f}"
                if state["trailing_stop"] is not None else "N/A"
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Trend Following: Holding position, stop at {stop_str}"
            )

    async def _get_funding_signal(
        self,
        bot: Bot,
        lookback_periods: int,
        refresh_seconds: float,
    ) -> Optional[float]:
        """Return the mean funding rate over the lookback, cached per bot.

        Fetches perpetual funding-rate history via the bot's exchange service
        (reusing ExchangeService.get_funding_rate_history) and averages the most
        recent `lookback_periods` windows using the shared compute_funding_stats
        helper. Cached for `refresh_seconds` to avoid hitting the API every loop.

        Returns None when no exchange service or funding data is available, so
        the strategy never trades on missing data.
        """
        if not hasattr(self, "_funding_cache"):
            self._funding_cache = {}

        now = datetime.utcnow()
        cached = self._funding_cache.get(bot.id)
        if cached and (now - cached["time"]).total_seconds() < refresh_seconds:
            return cached["mean_rate"]

        exchange = self._exchange_services.get(bot.id)
        if exchange is None:
            return None

        swap_symbol = exchange.to_swap_symbol(bot.trading_pair)
        history = await exchange.get_funding_rate_history(
            swap_symbol, limit=max(lookback_periods, 1)
        )
        if not history:
            return None

        recent = history[-lookback_periods:]
        rates = [h.funding_rate for h in recent]
        interval_hours = recent[-1].interval_hours
        mean_rate = compute_funding_stats(rates, interval_hours).mean_rate

        self._funding_cache[bot.id] = {"time": now, "mean_rate": mean_rate}
        return mean_rate

    async def _strategy_funding_carry(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Funding Carry (funding-aware trend) strategy.

        Long-only SPOT strategy that uses perpetual funding rates as a
        positioning/crowdedness signal, gated by a directional trend filter.

        IMPORTANT: This does NOT short, hedge, or directly harvest funding (a
        market-neutral basis trade needs a perpetual leg, intentionally out of
        scope). Funding is used purely as a FILTER on spot entries. True funding
        harvesting would require the perp leg; this strategy instead aims to
        improve risk-adjusted entries, not trade frequency.

        Entry (ALL required):
            - mean funding over `funding_lookback_periods` lies within the
              favourable band [min_funding_rate, max_funding_rate] (avoids
              both falling-knife and over-crowded/euphoric regimes)
            - market regime trend_state is in `allowed_regimes`
            - no open position, not in cooldown, sufficient balance
        Exit (ANY):
            - funding leaves the favourable band
            - market regime no longer favourable

        Parameters:
            min_funding_rate: Lower bound of favourable funding band (default -0.0005)
            max_funding_rate: Upper bound of favourable funding band (default 0.0005)
            funding_lookback_periods: Funding windows to average (default 3)
            allowed_regimes: Favourable trend regimes (default ["trend_up"])
            max_allocation_percent: Max % of balance per position (default 20)
            cooldown_seconds: Seconds to wait after exit before re-entry (default 300)
            funding_refresh_seconds: Funding-rate cache TTL in seconds (default 300)
        """
        min_funding = params.get("min_funding_rate", -0.0005)
        max_funding = params.get("max_funding_rate", 0.0005)
        lookback = int(params.get("funding_lookback_periods", 3))
        allowed_regimes = params.get("allowed_regimes", ["trend_up", "trend_flat"])
        max_alloc = params.get("max_allocation_percent", 20.0) / 100.0
        cooldown_seconds = params.get("cooldown_seconds", 300)
        refresh_seconds = params.get("funding_refresh_seconds", 300)

        # Defensive: a misconfigured band would silently block all trades.
        if min_funding > max_funding:
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Funding Carry: invalid config (min_funding_rate > max_funding_rate)",
            )

        # === DIRECTIONAL FILTER (reuses tick-based regime detection) ===
        price_history = self._get_price_history(bot.id)
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history, max_len=250)

        regime = self._detect_market_regime(price_history, None)
        trend_regime_name = f"trend_{regime.get('trend_state', 'flat')}"
        market_favorable = trend_regime_name in allowed_regimes

        # === FUNDING FILTER ===
        mean_funding = await self._get_funding_signal(bot, lookback, refresh_seconds)
        if not hasattr(self, "_funding_unavailable_warned"):
            self._funding_unavailable_warned = set()
        if mean_funding is None:
            # L1: a pair with no perpetual market yields a permanent silent hold.
            # Warn once per outage so the operator can see why nothing trades.
            if bot.id not in self._funding_unavailable_warned:
                self._funding_unavailable_warned.add(bot.id)
                msg = (
                    f"Funding Carry: no funding-rate data for {bot.trading_pair} "
                    "(no perpetual market or unsupported); holding until data is available."
                )
                logger.warning(f"Bot {bot.id}: {msg}")
                if bot.id in self._bot_loggers:
                    self._bot_loggers[bot.id].log_activity(msg)
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Funding Carry: funding-rate data unavailable (holding)",
            )
        # Data available again: clear the one-time warning latch.
        self._funding_unavailable_warned.discard(bot.id)
        funding_favorable = min_funding <= mean_funding <= max_funding

        # Per-bot state (cooldown tracking), mirroring other strategies.
        if not hasattr(self, "_funding_states"):
            self._funding_states = {}
        state = self._funding_states.get(bot.id, {"last_exit_time": None})

        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        funding_pct = mean_funding * 100.0

        if not has_position:
            # --- Entry path ---
            if state.get("last_exit_time") is not None:
                elapsed = (datetime.utcnow() - state["last_exit_time"]).total_seconds()
                if elapsed < cooldown_seconds:
                    remaining = int(cooldown_seconds - elapsed)
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason=f"Funding Carry: re-entry cooldown ({remaining}s remaining)",
                    )

            if not funding_favorable:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=(
                        f"Funding Carry: funding {funding_pct:.5f}% outside favourable "
                        f"band [{min_funding * 100:.5f}%, {max_funding * 100:.5f}%]"
                    ),
                )

            if not market_favorable:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=(
                        f"Funding Carry: regime {trend_regime_name} not in {allowed_regimes}"
                    ),
                )

            buy_amount = min(bot.current_balance * max_alloc, bot.current_balance * _BUY_BALANCE_FRACTION)
            if buy_amount < 1:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="Funding Carry: insufficient balance for entry",
                )

            logger.info(
                f"Bot {bot.id}: Funding Carry ENTRY - funding {funding_pct:.5f}% in band, "
                f"regime {trend_regime_name}, position ${buy_amount:.2f}"
            )
            self._funding_states[bot.id] = {"last_exit_time": None}
            return TradeSignal(
                action="buy",
                amount=buy_amount,
                order_type="market",
                reason=(
                    f"Funding Carry: favourable funding ({funding_pct:.5f}%) "
                    f"and regime ({trend_regime_name})"
                ),
            )

        # --- Exit path: leave when either condition stops being favourable ---
        if funding_favorable and market_favorable:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=(
                    f"Funding Carry: holding (funding {funding_pct:.5f}%, {trend_regime_name})"
                ),
            )

        pos = positions[0]
        sell_amount = pos.amount * current_price
        exit_reason = (
            "funding left favourable band"
            if not funding_favorable
            else f"regime {trend_regime_name} unfavourable"
        )
        logger.info(
            f"Bot {bot.id}: Funding Carry EXIT - {exit_reason} "
            f"(funding {funding_pct:.5f}%)"
        )
        self._funding_states[bot.id] = {"last_exit_time": datetime.utcnow()}
        return TradeSignal(
            action="sell",
            amount=sell_amount,
            order_type="market",
            reason=f"Funding Carry: exit ({exit_reason})",
        )

    async def _strategy_volatility_breakout(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Volatility Breakout (volatility expansion) strategy - Institutional Grade.

        RARE, CONVEX, REGIME-AWARE strategy that:
        - Trades volatility expansion after compression
        - Enters rarely (default: few trades per month)
        - Complements trend_following (often enters earlier)
        - Long-only, upper-band breakouts only

        Uses bar-aggregated Bollinger Band compression + breakout detection with
        ATR-based trailing stops that can ONLY TIGHTEN (risk never expands).

        CRITICAL: All logic operates on AGGREGATED PSEUDO-BARS, not tick data.
        Bar interval defines time granularity (default: 60 seconds per bar).

        Regime-aware by default: Pauses during unfavorable regimes (e.g. volatility
        contracting, strong downtrends) to prevent entries in wrong conditions.

        Parameters:
            bar_interval_seconds: Time per bar for aggregation (default: 60)
            bb_period: Bollinger Band period in bars (default: 20)
            bb_std: Bollinger Band standard deviation (default: 2.0)
            atr_period: ATR period in bars (default: 14)
            compression_method: "bb_width" or "atr_average" (default: "bb_width")
            compression_percentile: BB width percentile threshold % (default: 20)
            atr_threshold_multiplier: ATR threshold vs average (default: 0.8)
            min_compression_bars: Minimum bars of compression (default: 20, SPARSE)
            atr_stop_multiplier: ATR stop loss multiplier (default: 2.0)
            risk_percent: Percent of capital to risk (default: 1.0)
            cooldown_hours: Hours between breakout attempts (default: 72, SPARSE)
            failed_breakout_bars: Bars to detect failed breakout (default: 3)
            regime_filter_enabled: Enable regime gating (default: True)
            allowed_regimes: Allowed volatility regimes (default: ["volatility_expanding"])
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
        atr_stop_mult = params.get("atr_stop_multiplier", 2.0)
        risk_percent = params.get("risk_percent", 1.0) / 100
        cooldown_hours = params.get("cooldown_hours", 24)
        failed_breakout_bars = params.get("failed_breakout_bars", 3)
        regime_filter_enabled = params.get("regime_filter_enabled", True)
        allowed_regimes = params.get("allowed_regimes", ["volatility_expanding"])

        # === BAR AGGREGATION SYSTEM ===
        # Aggregate tick prices into fixed-time bars (OHLC)
        # This converts 1-second ticks into meaningful time-based bars

        # Initialize state tracking (INSTITUTIONAL STRUCTURE)
        if not hasattr(self, "_volatility_breakout_states"):
            self._volatility_breakout_states = {}

        state = self._volatility_breakout_states.get(bot.id, {
            "bars": [],  # List of {"open", "high", "low", "close", "start_ts"}
            "current_bar": None,  # Bar being built
            "bb_width_history": [],
            "atr_history": [],
            "compression_active": False,
            "compression_bars": 0,
            "compression_start": None,
            "breakout_armed": False,  # latched after compression, survives breakout bar
            "entry_price": None,
            "entry_atr": None,  # LOCKED at entry - risk never expands
            "highest_price": None,  # For monotonic trailing stop
            "trailing_stop": None,
            "bars_since_entry": 0,
            "last_breakout_attempt": None,
        })

        now = datetime.utcnow()

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

        # === REGIME GATING (pauses entries during wrong conditions) ===
        # Uses system-wide regime detection to avoid entries in adverse markets
        if regime_filter_enabled:
            # Detect regime from THIS strategy's own bar closes (same fix as
            # mean_reversion): the shared tick buffer is empty for a standalone
            # breakout bot, which pinned volatility to 'medium' -> 'volatility_
            # normal' and, with the default allow-list of ['volatility_expanding']
            # only, PERMANENTLY blocked every entry. >= bb_period bars exist here.
            bar_closes = [b["close"] for b in state["bars"]] + [current_price]
            current_regime = self._detect_market_regime(bar_closes, None)
            volatility_state = current_regime.get("volatility_state", "medium")

            # Map volatility_state to regime names
            # volatility_state values: "low", "medium", "high"
            # user config values: "volatility_contracting", "volatility_normal", "volatility_expanding"
            volatility_regime_map = {
                "low": "volatility_contracting",
                "medium": "volatility_normal",
                "high": "volatility_expanding",
            }
            volatility_regime_name = volatility_regime_map.get(volatility_state, "volatility_normal")

            # Block entries (not exits) if regime is wrong
            # This is a PAUSE, not an exit trigger (hold existing positions)
            regime_allows_entry = volatility_regime_name in allowed_regimes

            if not regime_allows_entry:
                logger.debug(
                    f"Bot {bot.id}: Volatility Breakout regime gate ACTIVE - "
                    f"Current: {volatility_regime_name}, Allowed: {allowed_regimes}"
                )

        else:
            regime_allows_entry = True
            volatility_regime_name = "regime_filter_disabled"

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # Get last completed bar close for logic (current bar is incomplete)
        last_bar_close = state["bars"][-1]["close"] if state["bars"] else current_price

        logger.debug(
            f"Bot {bot.id}: Volatility Breakout - Bar close: ${last_bar_close:.2f}, "
            f"BB: [${lower_band:.2f}, ${sma:.2f}, ${upper_band:.2f}], "
            f"Width: {bb_width:.4f}, ATR: ${atr:.2f}, Regime: {volatility_regime_name}"
        )

        # === POSITION EXIT LOGIC (INSTITUTIONAL GRADE) ===
        if has_position:
            for pos in positions:
                # Increment bars_since_entry only when bar completes (not on every tick)
                if bar_completed and state["entry_price"] is not None:
                    state["bars_since_entry"] += 1

                # CRITICAL: Use LOCKED entry_atr (risk never expands)
                entry_atr_locked = state.get("entry_atr", atr)  # Fallback for legacy positions

                # === EXIT CONDITION 1: Failed Breakout (BAR-BASED) ===
                # Within N bars after entry, if bar CLOSES back inside BB, exit immediately
                # Uses bar close, not tick noise
                if state["bars_since_entry"] <= failed_breakout_bars:
                    if last_bar_close < upper_band:
                        sell_amount = pos.amount * current_price

                        logger.info(
                            f"Bot {bot.id}: Volatility Breakout EXIT (failed breakout) - "
                            f"Bar close ${last_bar_close:.2f} < BB upper ${upper_band:.2f} "
                            f"after {state['bars_since_entry']} bars (threshold: {failed_breakout_bars})"
                        )

                        # Clear state
                        state["trailing_stop"] = None
                        state["highest_price"] = None
                        state["entry_price"] = None
                        state["entry_atr"] = None
                        state["bars_since_entry"] = 0
                        state["compression_active"] = False
                        state["compression_bars"] = 0
                        self._volatility_breakout_states[bot.id] = state

                        return TradeSignal(
                            action="sell",
                            amount=sell_amount,
                            order_type="market",
                            reason=f"Volatility Breakout: Failed breakout (bar {state['bars_since_entry']}/{failed_breakout_bars})",
                        )

                # === TRAILING STOP: MONOTONIC TIGHTENING (FIXED BUG) ===
                # Track highest_price explicitly
                # Trailing stop can ONLY move upward (risk never expands)

                # Initialize highest_price if needed
                if state["highest_price"] is None:
                    state["highest_price"] = current_price

                # Update highest_price if new high
                if current_price > state["highest_price"]:
                    state["highest_price"] = current_price
                    # Recompute trailing stop using LOCKED entry_atr
                    state["trailing_stop"] = state["highest_price"] - (entry_atr_locked * atr_stop_mult)
                    logger.debug(
                        f"Bot {bot.id}: Volatility Breakout trailing stop updated - "
                        f"New high ${state['highest_price']:.2f}, Stop ${state['trailing_stop']:.2f}"
                    )

                # Initialize trailing stop if not set (legacy positions)
                if state["trailing_stop"] is None:
                    state["trailing_stop"] = current_price - (entry_atr_locked * atr_stop_mult)

                # === EXIT CONDITION 2: Trailing Stop Hit ===
                if current_price <= state["trailing_stop"]:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Volatility Breakout EXIT (trailing stop) - "
                        f"Price ${current_price:.2f} <= Stop ${state['trailing_stop']:.2f}, "
                        f"Entry ATR (locked): ${entry_atr_locked:.4f}"
                    )

                    # Clear state
                    state["trailing_stop"] = None
                    state["highest_price"] = None
                    state["entry_price"] = None
                    state["entry_atr"] = None
                    state["bars_since_entry"] = 0
                    state["compression_active"] = False
                    state["compression_bars"] = 0
                    self._volatility_breakout_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Volatility Breakout: Trailing stop (${state['trailing_stop']:.2f})",
                    )

            # Update state and hold.
            # Same f-string format-spec crash class as trend_following/mean
            # reversion: a conditional in the spec raises on every hold tick.
            self._volatility_breakout_states[bot.id] = state

            stop_str = (
                f"${state['trailing_stop']:.2f}"
                if state["trailing_stop"] is not None else "N/A"
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Holding position, stop at {stop_str}"
            )

        # === ENTRY LOGIC (RARE) ===
        # The volatility "regime" veto was REMOVED as a hard entry gate here.
        # Two bugs made it pathological: (1) it ran BEFORE compression tracking
        # and early-returned, so a regime-blocked bot never accumulated a single
        # compression bar; (2) it demanded 'volatility_expanding' at the very
        # instant the compression check demands LOW volatility - a contradiction
        # that meant the strategy could never enter. The compression-then-
        # breakout sequence already encodes the volatility thesis (compression =
        # low vol, a close above the upper band = the expansion), so the separate
        # volatility veto was redundant. volatility_regime_name is kept for logs.

        # === COMPRESSION DETECTION (BAR-BASED) - ALWAYS TRACKED ===
        # Must run regardless of regime/cooldown, or compression_bars can never
        # reach min_compression_bars.
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
                avg_atr = sum(state["atr_history"][-20:]) / 20
                is_compressed = atr <= (avg_atr * atr_threshold_mult)

        # Track compression duration (BAR-BASED)
        if bar_completed:
            if is_compressed:
                if not state["compression_active"]:
                    state["compression_active"] = True
                    state["compression_start"] = datetime.utcnow().isoformat()
                    state["compression_bars"] = 1
                    logger.info(
                        f"Bot {bot.id}: Volatility Breakout compression STARTED - "
                        f"Method: {compression_method}, BB width: {bb_width:.4f}"
                    )
                else:
                    state["compression_bars"] += 1
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

        # Latch a "breakout armed" flag once compression has persisted long
        # enough. It MUST survive the expansion that follows: a genuine breakout
        # bar has wide close-dispersion (high BB width) and therefore reads as
        # NOT compressed, which would otherwise reset compression on the very bar
        # we want to act on (so the strategy could never fire). Arming decouples
        # "we were compressed" from "now we break out". It disarms on entry or if
        # price falls back to the mean (the setup has gone stale).
        if state["compression_active"] and state["compression_bars"] >= min_compression_bars:
            state["breakout_armed"] = True
        if state.get("breakout_armed") and last_bar_close < sma:
            state["breakout_armed"] = False
        compression_satisfied = bool(state.get("breakout_armed"))

        # A confirmed breakout = armed by prior compression AND a bar close above
        # the upper band. This is the only behavioural entry gate now (+ cooldown).
        is_breakout = compression_satisfied and last_bar_close > upper_band

        # PRODUCTION DIAGNOSIS (once per completed bar): why this (rare) strategy
        # is/ isn't entering - compression progress, arming, and the breakout gap.
        if bar_completed:
            upper_gap_pct = (
                (last_bar_close - upper_band) / upper_band * 100 if upper_band > 0 else 0.0
            )
            logger.info(
                f"Bot {bot.id}: VB no-trade diag - close=${last_bar_close:.2f} "
                f"upper=${upper_band:.2f} gap_to_breakout={upper_gap_pct:+.3f}% "
                f"width={bb_width:.5f} compressed={is_compressed} "
                f"comp_bars={state['compression_bars']}/{min_compression_bars} "
                f"armed={bool(state.get('breakout_armed'))} regime={volatility_regime_name} "
                f"bars={len(state['bars'])}"
            )

        # Cooldown only gates an ACTUAL breakout entry (sparse trading) - it no
        # longer blocks compression tracking.
        if is_breakout and state["last_breakout_attempt"] is not None:
            last_attempt = datetime.fromisoformat(state["last_breakout_attempt"])
            hours_since = (datetime.utcnow() - last_attempt).total_seconds() / 3600
            if hours_since < cooldown_hours:
                self._volatility_breakout_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Volatility Breakout: Cooldown ({cooldown_hours - hours_since:.1f}h remaining)"
                )

        # === BREAKOUT ENTRY CONDITION (LONG-ONLY, UPPER BAND) ===
        if is_breakout:
            # Volatility-adjusted position sizing: risk a fixed % of capital; the
            # ATR-based stop distance gives the COIN count, converted to a quote
            # notional. BUGFIX: the * current_price conversion was missing (same
            # unit bug as trend_following), producing sub-$1 orders or, on ATR
            # collapse, a blow-up silently capped at the balance.
            risk_amount = bot.current_balance * risk_percent

            if atr > 0:
                stop_distance = atr * atr_stop_mult            # USD per coin
                position_coins = risk_amount / stop_distance   # base coins
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
                        action="hold",
                        amount=0,
                        reason=(
                            f"Volatility Breakout: balance ${bot.current_balance:.2f} "
                            f"below ${MIN_ORDER_USD:.0f} minimum order"
                        ),
                    )

            # Calculate BB width percentile for logging
            percentile_rank = "N/A"
            if len(state["bb_width_history"]) >= 20:
                sorted_widths = sorted(state["bb_width_history"])
                rank = sorted_widths.index(min(sorted_widths, key=lambda x: abs(x - bb_width)))
                percentile_rank = f"{int((rank / len(sorted_widths)) * 100)}th"

            logger.info(
                f"Bot {bot.id}: Volatility Breakout ENTRY - "
                f"{state['compression_bars']} bars compression (BB width {percentile_rank} percentile), "
                f"Bar close ${last_bar_close:.2f} > BB upper ${upper_band:.2f}, "
                f"Entry ATR locked at ${atr:.4f}, Position: ${buy_amount:.2f}"
            )

            # Initialize state with LOCKED entry_atr (risk never expands)
            state["trailing_stop"] = current_price - (atr * atr_stop_mult)
            state["highest_price"] = current_price
            state["entry_price"] = current_price
            state["entry_atr"] = atr  # LOCKED - trailing stop distance will always use this
            state["bars_since_entry"] = 0
            state["breakout_armed"] = False  # consumed this setup
            state["last_breakout_attempt"] = datetime.utcnow().isoformat()

            self._volatility_breakout_states[bot.id] = state

            return TradeSignal(
                action="buy",
                amount=buy_amount,
                order_type="market",
                reason=f"Volatility Breakout: {state['compression_bars']} bars compression, breakout confirmed",
            )

        # Update state and hold (EXPLAINABLE REASONS)
        self._volatility_breakout_states[bot.id] = state

        # Provide clear feedback on why not entering
        if compression_satisfied:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Compression satisfied ({state['compression_bars']} bars), waiting for breakout > ${upper_band:.2f}"
            )
        elif state["compression_active"]:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Compression building ({state['compression_bars']}/{min_compression_bars} bars)"
            )
        else:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Watching for compression (BB width: {bb_width:.4f})"
            )

    # Note: TWAP and VWAP strategy methods removed.
    # TWAP/VWAP are execution algorithms, not alpha strategies.
    # They now exist in the execution layer as _execute_twap() and _execute_vwap().


    # Note: _strategy_arbitrage and _strategy_event were removed (placeholders without implementation)

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
        now = datetime.utcnow()

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
        scored_strategies = []
        for strategy_name in eligible_strategies:
            caps = capabilities[strategy_name]
            metrics = strategy_metrics.get(strategy_name, {})
            effective_priority = self._score_strategy(strategy_name, caps, metrics)
            scored_strategies.append((strategy_name, effective_priority, caps, metrics))

        # Sort by effective priority (descending)
        scored_strategies.sort(key=lambda x: x[1], reverse=True)

        # === FORCE-EXIT CHECK ===
        current_strategy = auto_state["current_strategy"]
        force_exit_signal = None

        # Check if current strategy is ineligible
        if current_strategy not in eligible_strategies:
            force_exit_reason = ineligible_reasons.get(current_strategy, "unknown")
            logger.warning(
                f"Bot {bot.id}: Auto Mode FORCE EXIT - {current_strategy} became ineligible: {force_exit_reason}"
            )
            force_exit_signal = TradeSignal(
                action="sell",
                amount=0,  # Sell all
                reason=f"Auto Mode FORCE EXIT: {current_strategy} ineligible ({force_exit_reason})"
            )

            # Record failure
            await self._record_strategy_failure(
                bot_id=bot.id,
                auto_state=auto_state,
                strategy_name=current_strategy,
                reason=force_exit_reason,
                now=now,
                cooldown_hours=cooldown_hours_default,
                session=session
            )

        # === STRATEGY SELECTION WITH INERTIA ===
        best_strategy, best_priority, best_caps, best_metrics = scored_strategies[0]
        selected_strategy = current_strategy
        should_switch = False
        switch_reason = ""

        if current_strategy not in eligible_strategies:
            # Must switch, current is ineligible
            selected_strategy = best_strategy
            should_switch = True
            switch_reason = f"current strategy ineligible, switching to {best_strategy}"
        elif best_strategy != current_strategy:
            # Check if better strategy available
            current_priority = next((s[1] for s in scored_strategies if s[0] == current_strategy), 0)

            if best_priority > current_priority:
                # Check min switch interval
                last_switch = auto_state.get("last_switch_time")
                if last_switch:
                    time_since_switch = (now - datetime.fromisoformat(last_switch)).total_seconds() / 60
                    if time_since_switch < min_switch_interval:
                        switch_reason = f"better strategy {best_strategy} available but too soon (last switch {time_since_switch:.1f}m ago)"
                    else:
                        selected_strategy = best_strategy
                        should_switch = True
                        switch_reason = f"higher priority strategy {best_strategy} (priority {best_priority:.2f} > {current_priority:.2f})"
                else:
                    selected_strategy = best_strategy
                    should_switch = True
                    switch_reason = f"higher priority strategy {best_strategy} (priority {best_priority:.2f})"
            else:
                switch_reason = f"current strategy {current_strategy} still optimal (priority {current_priority:.2f})"
        else:
            switch_reason = f"current strategy {current_strategy} is best (priority {best_priority:.2f})"

        # === STRATEGY SWITCHING ===
        if should_switch:
            logger.info(
                f"Bot {bot.id}: Auto Mode SWITCHING - {current_strategy} → {selected_strategy} ({switch_reason})"
            )
            auto_state["current_strategy"] = selected_strategy
            auto_state["last_switch_time"] = now.isoformat()
            current_strategy = selected_strategy
            # Record activation time so inactivity penalty has a baseline
            if current_strategy not in strategy_metrics:
                strategy_metrics[current_strategy] = {}
            strategy_metrics[current_strategy]["activated_at"] = now.isoformat()
            auto_state["strategy_metrics"] = strategy_metrics
        else:
            logger.debug(f"Bot {bot.id}: Auto Mode HOLDING {current_strategy} ({switch_reason})")

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

        # === FORCE EXIT EXECUTION ===
        if force_exit_signal:
            return force_exit_signal

        # === STRATEGY EXECUTION ===
        strategy_executor = self._get_strategy_executor(current_strategy)

        if not strategy_executor:
            logger.warning(f"Bot {bot.id}: Auto Mode - unknown strategy {current_strategy}")
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Auto Mode: Invalid strategy {current_strategy}"
            )

        # Execute the selected strategy
        signal = await strategy_executor(bot, current_price, params, session)

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

        # === TREND STATE (EMA slope, same thresholds as bar-based variant) ===
        ema_20 = self._calculate_ema(prices, 20)
        ema_50 = self._calculate_ema(prices, 50) if n >= 50 else ema_20

        ema_20_current = ema_20[-1]
        ema_20_prev = ema_20[-5] if len(ema_20) >= 5 else ema_20[0]
        ema_slope_pct = ((ema_20_current - ema_20_prev) / ema_20_prev) * 100 if ema_20_prev > 0 else 0

        if ema_slope_pct > 0.5 and ema_20_current > ema_50[-1]:
            trend_state = "up"
        elif ema_slope_pct < -0.5 and ema_20_current < ema_50[-1]:
            trend_state = "down"
        else:
            trend_state = "flat"

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

        # === TREND STATE (using EMA slope on bar closes) ===
        ema_20 = self._calculate_ema(closes, 20)
        ema_50 = self._calculate_ema(closes, 50) if n >= 50 else ema_20

        ema_20_current = ema_20[-1]
        ema_20_prev = ema_20[-5] if len(ema_20) >= 5 else ema_20[0]
        ema_slope_pct = ((ema_20_current - ema_20_prev) / ema_20_prev) * 100 if ema_20_prev > 0 else 0

        if ema_slope_pct > 0.5 and ema_20_current > ema_50[-1]:
            new_trend = "up"
        elif ema_slope_pct < -0.5 and ema_20_current < ema_50[-1]:
            new_trend = "down"
        else:
            new_trend = "flat"

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
        else:
            new_volatility = "medium"

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
                    "liquidity_state": new_liquidity,
                    "persistence_bars": 0
                }
            else:
                return {
                    "trend_state": current_regime.get("trend_state"),
                    "volatility_state": current_regime.get("volatility_state"),
                    "liquidity_state": current_regime.get("liquidity_state"),
                    "persistence_bars": persistence_bars,
                }
        else:
            return {
                "trend_state": new_trend,
                "volatility_state": new_volatility,
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
        liquidity = current_regime.get("liquidity_state", "normal")

        regime_tags = [
            f"trend_{trend}",
            f"volatility_{volatility}",
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

    def _score_strategy(
        self,
        strategy_name: str,
        capabilities: dict,
        strategy_metrics: dict
    ) -> float:
        """Calculate risk-adjusted effective priority for a strategy.

        Formula:
            effective_priority = base_priority + performance_bonus - risk_penalty

        Where:
            - performance_bonus is proportional to recent_pnl_pct (capped)
            - risk_penalty grows aggressively with drawdown and failures
            - Risk penalty MUST dominate performance bonus

        Args:
            strategy_name: Name of strategy
            capabilities: Strategy capability dict
            strategy_metrics: Performance metrics

        Returns:
            Effective priority score (float)
        """
        base_priority = capabilities.get("priority", 0)

        # Performance bonus (capped at +2.0)
        recent_pnl_pct = strategy_metrics.get("recent_pnl_pct", 0.0)
        performance_bonus = min(2.0, max(-2.0, recent_pnl_pct / 5.0))  # +/-20% PnL = +/-4 bonus, capped at +/-2

        # Risk penalty (aggressive)
        risk_penalty = 0.0

        # Drawdown penalty (exponential)
        max_drawdown_pct = strategy_metrics.get("max_drawdown_pct", 0.0)
        if max_drawdown_pct > 0:
            # Penalty grows exponentially: 5% = -1, 10% = -4, 15% = -9, 20% = -16
            risk_penalty += (max_drawdown_pct / 5.0) ** 2

        # Failure penalty (linear and heavy)
        failure_count = strategy_metrics.get("failure_count", 0)
        risk_penalty += failure_count * 5.0  # Each failure = -5 priority

        # Recent exit penalty (discourage recently exited strategies)
        last_exit_time = strategy_metrics.get("last_exit_time")
        if last_exit_time:
            try:
                exit_time = datetime.fromisoformat(last_exit_time)
                hours_since_exit = (datetime.utcnow() - exit_time).total_seconds() / 3600
                if hours_since_exit < 1:
                    risk_penalty += 3.0  # Heavy penalty if exited < 1h ago
                elif hours_since_exit < 6:
                    risk_penalty += 1.0  # Moderate penalty if exited < 6h ago
            except (ValueError, TypeError):
                pass

        # Inactivity penalty: a strategy that produces no trades while selected
        # loses score at 0.15 priority/hour after a 2-hour grace period (capped
        # at -4.0). This prevents auto mode from parking indefinitely on an
        # eligible but non-participating strategy. last_trade_time is set each
        # time the sub-strategy returns a buy/sell; activated_at is set on
        # strategy switch. The baseline is the more recent of the two.
        last_trade_str = strategy_metrics.get("last_trade_time")
        activated_str = strategy_metrics.get("activated_at")
        baseline_str = last_trade_str or activated_str
        if baseline_str:
            try:
                baseline_dt = datetime.fromisoformat(baseline_str)
                hours_inactive = (datetime.utcnow() - baseline_dt).total_seconds() / 3600
                if hours_inactive > 2.0:
                    risk_penalty += min((hours_inactive - 2.0) * 0.15, 4.0)
            except (ValueError, TypeError):
                pass

        effective_priority = base_priority + performance_bonus - risk_penalty

        return effective_priority

    async def _update_strategy_performance_metrics(
        self,
        bot_id: int,
        auto_state: dict,
        session: AsyncSession,
        performance_window: int
    ) -> None:
        """Update per-strategy performance metrics on bar close.

        Tracks:
        - recent_pnl_pct: Rolling window PnL percentage
        - max_drawdown_pct: Maximum drawdown observed
        - failure_count: Number of failures
        - last_exit_time: Timestamp of last exit

        Args:
            bot_id: Bot ID
            auto_state: Auto mode state
            session: Database session
            performance_window: Number of bars for rolling window
        """
        # Get recent orders to calculate PnL
        query = select(Order).where(
            Order.bot_id == bot_id,
            Order.status == OrderStatus.FILLED
        ).order_by(Order.filled_at.desc()).limit(50)

        result = await session.execute(query)
        orders = result.scalars().all()

        if not orders:
            return

        # Group orders by strategy (extracted from reason field)
        strategy_pnl = {}
        strategy_orders = {}

        for order in orders:
            # Extract strategy from reason (format: "[Auto:strategy_name|regime] reason")
            reason = order.reason or ""
            if reason.startswith("[Auto:"):
                strategy_part = reason.split("|")[0].replace("[Auto:", "")
                strategy_name = strategy_part.strip()

                if strategy_name not in strategy_pnl:
                    strategy_pnl[strategy_name] = []
                    strategy_orders[strategy_name] = []

                # Calculate order PnL (simplified)
                if order.running_balance_after is not None:
                    strategy_orders[strategy_name].append(order)

        # Update metrics for each strategy
        strategy_metrics = auto_state.get("strategy_metrics", {})

        for strategy_name, orders_list in strategy_orders.items():
            if strategy_name not in strategy_metrics:
                strategy_metrics[strategy_name] = {
                    "recent_pnl_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "failure_count": 0,
                    "last_exit_time": None,
                    "cooldown_until": None
                }

            metrics = strategy_metrics[strategy_name]

            # Calculate recent PnL (simplified - use running balance changes)
            if len(orders_list) >= 2:
                recent_orders = orders_list[:min(performance_window, len(orders_list))]
                start_balance = recent_orders[-1].running_balance_after
                end_balance = recent_orders[0].running_balance_after

                if start_balance and start_balance > 0:
                    pnl_pct = ((end_balance - start_balance) / start_balance) * 100
                    metrics["recent_pnl_pct"] = pnl_pct

                    # Update max drawdown
                    if pnl_pct < 0:
                        metrics["max_drawdown_pct"] = max(metrics["max_drawdown_pct"], abs(pnl_pct))

        auto_state["strategy_metrics"] = strategy_metrics

        # Persist all updated metrics to database
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

        if existing:
            # Update existing record
            existing.recent_pnl_pct = metrics.get("recent_pnl_pct", 0.0)
            existing.max_drawdown_pct = metrics.get("max_drawdown_pct", 0.0)
            existing.failure_count = metrics.get("failure_count", 0)

            # Handle datetime conversion
            last_exit = metrics.get("last_exit_time")
            if last_exit:
                if isinstance(last_exit, str):
                    existing.last_exit_time = datetime.fromisoformat(last_exit)
                else:
                    existing.last_exit_time = last_exit

            cooldown = metrics.get("cooldown_until")
            if cooldown:
                if isinstance(cooldown, str):
                    existing.cooldown_until = datetime.fromisoformat(cooldown)
                else:
                    existing.cooldown_until = cooldown

            existing.last_updated = datetime.utcnow()
        else:
            # Create new record
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
        log_msg += f"Reason: {switch_reason}\n"
        log_msg += f"\n"

        log_msg += f"Eligible Strategies ({len(eligible_strategies)}):\n"
        for name, score, caps, metrics in scored_strategies:
            pnl = metrics.get("recent_pnl_pct", 0.0)
            dd = metrics.get("max_drawdown_pct", 0.0)
            failures = metrics.get("failure_count", 0)
            log_msg += f"  - {name:30s} score={score:6.2f} pnl={pnl:+6.2f}% dd={dd:5.2f}% failures={failures}\n"

        if ineligible_reasons:
            log_msg += f"\nIneligible Strategies ({len(ineligible_reasons)}):\n"
            for name, reason in ineligible_reasons.items():
                log_msg += f"  - {name:30s} {reason}\n"

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
                "allowed_regimes": ["trend_up"],
                "priority": 4,
                "typical_holding_time": "long",
                "description": "Best for sustained uptrends with clear momentum"
            },
            "volatility_breakout": {
                "allowed_regimes": ["volatility_high", "volatility_expanding"],
                "priority": 3,
                "typical_holding_time": "medium",
                "description": "Captures breakouts after volatility compression"
            },
            "mean_reversion": {
                "allowed_regimes": ["trend_flat", "volatility_high"],
                "priority": 2,
                "typical_holding_time": "short",
                "description": "Profits from price mean reversion in choppy markets"
            },
            "adaptive_grid": {
                "allowed_regimes": ["trend_flat", "volatility_medium"],
                "priority": 2,
                "typical_holding_time": "medium",
                "description": "Range-bound grid trading for sideways markets"
            },
            # Note: VWAP removed - it is an execution algorithm, not an alpha strategy
            "dca_accumulator": {
                "allowed_regimes": ["all"],
                "priority": 0,
                "typical_holding_time": "long",
                "description": "Safe default accumulator for all market conditions"
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
                        datetime.utcnow() - datetime.fromisoformat(last_switch)
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
        # Get cost model (defaults to 0 cost, preserving current behavior)
        cost_model = get_cost_model(
            exchange_fee_pct=getattr(bot, 'exchange_fee', 0.0) or 0.0,
            market_spread_pct=0.0,  # TODO: Make configurable per bot
            slippage_pct=0.0,       # TODO: Make configurable per bot
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
                bot.trading_pair, side, amount_base
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
            order.filled_at = datetime.utcnow()

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
                timestamp=datetime.utcnow(),
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
            executed_at=datetime.utcnow(),
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
            await self._open_or_add_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session
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
        """Build a cost estimate for a recovered fill (mirrors _execute_trade)."""
        cost_model = get_cost_model(
            exchange_fee_pct=getattr(bot, 'exchange_fee', 0.0) or 0.0,
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
                order.filled_at = datetime.utcnow()
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
                filled_at=datetime.utcnow(),
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
                    bot.paused_at = datetime.utcnow()
                    bot.updated_at = datetime.utcnow()
                    await session.commit()
                    await self._emit_alert(
                        session, bot_id, "failure_circuit_breaker", reason,
                        email_subject=f"TradingBot: bot {bot_id} paused after repeated failures",
                    )
        except Exception as e:
            logger.error(f"Bot {bot_id}: failed to pause via circuit breaker: {e}")
        finally:
            self._stop_flags[bot_id] = True

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
                    bot.paused_at = datetime.utcnow()
                    bot.updated_at = datetime.utcnow()
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
            "_price_histories", "_funding_cache", "_funding_states", "_trend_states",
            "_grid_states", "_mean_reversion_states", "_volatility_breakout_states",
            "_twap_states", "_vwap_states", "_auto_states", "_last_pending_resolve",
            "_bot_loggers", "_exec_rejections",
        )
        for attr in state_dicts:
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(bot_id, None)
        warned = getattr(self, "_funding_unavailable_warned", None)
        if isinstance(warned, set):
            warned.discard(bot_id)
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
                "start_time": datetime.utcnow(),
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
        elapsed_seconds = (datetime.utcnow() - start_time).total_seconds()

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

        exchange_order = await exchange.place_market_order(bot.trading_pair, side, amount_base)

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
            order.filled_at = datetime.utcnow()

        session.add(order)

        # Update wallet and positions (same as standard execution)
        # Use the actual filled amount so partial fills cannot desync positions
        filled_amount = getattr(exchange_order, 'filled', exchange_order.amount) or exchange_order.amount
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session
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
            "timestamp": datetime.utcnow(),
            "price": current_price,
            "volume": simulated_volume,
        })

        # Keep only recent data
        cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)
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

        exchange_order = await exchange.place_market_order(bot.trading_pair, side, amount_base)

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
            order.filled_at = datetime.utcnow()

        session.add(order)

        # Update wallet and positions (same as standard execution)
        # Use the actual filled amount so partial fills cannot desync positions
        filled_amount = getattr(exchange_order, 'filled', exchange_order.amount) or exchange_order.amount
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, filled_amount,
                exchange_order.price, session
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

    async def _open_or_add_position(
        self,
        bot_id: int,
        trading_pair: str,
        amount: float,
        price: float,
        session: AsyncSession,
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
            # Average into existing position
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
                holding_days = (datetime.utcnow() - position.created_at).days

            proceeds = sell_amount * price
            cost_basis = sell_amount * position.entry_price

            self._bot_loggers[bot_id].log_fiscal_entry(FiscalLogEntry(
                date=datetime.utcnow(),
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
        """Get all positions for a bot."""
        result = await session.execute(
            select(Position).where(Position.bot_id == bot_id)
        )
        return result.scalars().all()

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
            time_since = datetime.utcnow() - last.snapshot_at
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
        """Resume all bots that were running when server stopped.

        This queries for bots with RUNNING status and restarts their execution loops.

        Returns:
            Number of bots resumed
        """
        resumed = 0

        # Collect the bot ids in a short-lived session, then resume each bot in
        # ITS OWN session (M-3): a failure or mid-loop commit for one bot must
        # not poison the session used for the others.
        async with async_session_maker() as session:
            result = await session.execute(
                select(Bot.id).where(Bot.status == BotStatus.RUNNING)
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
        bot.strategy_state = self._collect_bot_state(bot_id)

        # M5: strategy_params is user config only. Strip any runtime state that
        # older builds may have embedded there so it cannot pollute config or
        # fail parameter validation on a later edit.
        if bot.strategy_params:
            cleaned = {k: v for k, v in bot.strategy_params.items() if not k.startswith("_")}
            if cleaned != bot.strategy_params:
                bot.strategy_params = cleaned

        bot.updated_at = datetime.utcnow()

        logger.debug(f"Saved state for bot {bot_id}")


# Global trading engine instance
trading_engine = TradingEngine()
