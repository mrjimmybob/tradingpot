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
    Trade, TradeSide,
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
from .portfolio_risk import PortfolioRiskService
from .strategy_capacity import StrategyCapacityService
from .ledger_writer import LedgerWriterService
from .accounting import TradeRecorderService, FIFOTaxEngine, CSVExportService
from .ledger_invariants import LedgerInvariantService, ValidationError

logger = logging.getLogger(__name__)


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
    price: Optional[float] = None  # For limit orders
    order_type: str = "market"  # "market" or "limit" (legacy)
    reason: str = ""

    # Execution layer fields (new)
    execution: Optional[str] = None  # "market", "twap", "vwap" (None defaults to "market")
    execution_params: Optional[dict] = None  # Execution-specific parameters


class TradingEngine:
    """Engine for executing trading bots."""

    def __init__(self):
        """Initialize trading engine."""
        self._running_bots: Dict[int, asyncio.Task] = {}
        self._exchange_services: Dict[int, ExchangeService] = {}
        self._stop_flags: Dict[int, bool] = {}
        self._bot_loggers: Dict[int, BotLoggingService] = {}

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

            # Update bot status
            bot.status = BotStatus.RUNNING
            bot.started_at = datetime.utcnow()
            bot.paused_at = None
            bot.updated_at = datetime.utcnow()
            await session.commit()

            # Create exchange service
            if bot.is_dry_run:
                exchange = SimulatedExchangeService(initial_balance=bot.budget)
            else:
                exchange = ExchangeService()

            await exchange.connect()
            self._exchange_services[bot_id] = exchange

            # Initialize per-bot file logger
            ensure_bot_log_directory(bot_id)
            self._bot_loggers[bot_id] = BotLoggingService(
                bot_id, bot.name, bot.is_dry_run
            )
            self._bot_loggers[bot_id].log_activity(
                f"Bot started with strategy '{bot.strategy}' on {bot.trading_pair}"
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
                        logger.warning(f"Bot {bot_id}: Could not get ticker for {bot.trading_pair}")
                        await asyncio.sleep(5)
                        continue

                    # Generate trading signal from strategy
                    signal = await self._execute_strategy(bot, ticker.last, session)

                    if signal and signal.action != "hold":
                        # Validate trade with virtual wallet
                        validation = await wallet.validate_trade(bot_id, signal.amount)

                        if validation.is_valid:
                            # Execute trade
                            order = await self._execute_trade(
                                bot, exchange, signal, ticker.last, session
                            )

                            if order:
                                # Check stop loss for any open positions
                                await self._check_positions_stop_loss(
                                    bot_id, exchange, risk_mgr, session
                                )
                        else:
                            logger.warning(f"Bot {bot_id}: Trade rejected - {validation.reason}")

                    # Take P&L snapshot periodically
                    await self._take_pnl_snapshot(bot_id, session)

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
                self._stop_flags[bot_id] = True
                # TODO: Send alert to admins via email/alert system
                break  # Exit loop immediately

            except Exception as e:
                logger.error(f"Bot {bot_id}: Error in execution loop: {e}")

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
            "cross_sectional_momentum": self._strategy_cross_sectional_momentum,
            "volatility_breakout": self._strategy_volatility_breakout,
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
        if amount_usd and amount_usd > 0:
            # Use fixed USD amount
            buy_amount = min(amount_usd, bot.current_balance)
        else:
            # Use percentage of current balance
            buy_amount = bot.current_balance * amount_percent

        # === MINIMUM ORDER CHECK (safety floor, not exchange-accurate) ===
        # This is a placeholder minimum to prevent dust orders.
        # Exchange-level minimum notional validation happens downstream.
        # Do NOT rely on this for exchange-specific min-notional requirements.
        min_notional_usd_placeholder = 1.0  # Safety floor: $1 minimum

        if buy_amount < min_notional_usd_placeholder:
            if bot.current_balance < min_notional_usd_placeholder:
                # Infinite DCA has reached its natural end: balance exhausted
                logger.info(
                    f"Bot {bot.id}: DCA infinite accumulation complete - "
                    f"Balance ${bot.current_balance:.2f} < minimum ${min_notional_usd_placeholder}"
                )
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="DCA: Budget exhausted (infinite accumulation complete)"
                )
            # Calculated amount is too small (likely due to low amount_percent)
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"DCA: Calculated amount ${buy_amount:.2f} below minimum ${min_notional_usd_placeholder}"
            )

        # Defensive: Cap at available balance (should already be handled above, but extra safety)
        if buy_amount > bot.current_balance:
            buy_amount = bot.current_balance

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
        grid_spacing_pct = params.get("grid_spacing_percent", 1.0) / 100
        range_pct = params.get("range_percent", 10) / 100
        base_order_size_pct = params.get("base_order_size_percent", 5) / 100
        depth_multiplier = params.get("depth_multiplier", 1.5)
        max_drawdown_pct = params.get("max_drawdown_percent", 15) / 100
        kill_atr_mult = params.get("kill_atr_multiplier", 3.0)
        atr_period = params.get("atr_period", 14)
        regime_filter_enabled = params.get("regime_filter_enabled", True)
        allowed_regimes = params.get("allowed_regimes", ["trend_flat", "volatility_normal"])
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
            return TradeSignal(action="hold", reason="Grid: Starting new bar")

        # Update current bar
        current_bar["high"] = max(current_bar["high"], current_price)
        current_bar["low"] = min(current_bar["low"], current_price)
        current_bar["close"] = current_price

        # Check if bar is complete
        bar_duration = (now - current_bar["start_ts"]).total_seconds()
        if bar_duration < bar_interval_seconds:
            return TradeSignal(
                action="hold",
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
                    reason=f"Grid: Re-centered at ${bar_close_price:.2f} (range escape), {cooldown_after_kill_hours}h cooldown"
                )

        # === ONE ORDER PER BAR CHECK ===
        last_order_bar = state.get("last_order_bar")
        if last_order_bar is not None and last_order_bar == completed_bars[-1]["start_ts"]:
            return TradeSignal(
                action="hold",
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
            return TradeSignal(
                action="hold",
                reason="Grid: No levels triggered this bar"
            )

        # === EXECUTE ORDER AT NEAREST LEVEL ===
        level_data = grid_levels[nearest_level]
        depth = level_data["depth"]

        # Depth-aware sizing: larger orders at deeper discounts (convex payoff)
        # Example: depth=1 → 1.0x, depth=2 → 1.5x, depth=3 → 2.25x, etc.
        size_multiplier = depth_multiplier ** (depth - 1)
        order_size_usd = initial_capital * base_order_size_pct * size_multiplier

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
        std_mult = params.get("bollinger_std", 2.0)
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
            # Get price history for regime detection
            price_history_for_regime = self._get_price_history(bot.id)
            price_history_for_regime.append(current_price)

            # Detect current market regime
            current_regime = self._detect_market_regime(price_history_for_regime, None)
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

            logger.debug(
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

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Holding, target ${exit_level:.2f}, stop ${hard_stop:.2f if hard_stop else 'N/A'}, bars {state['bars_since_entry']}/{max_hold_bars}"
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
            # Fixed percentage position sizing
            buy_amount = bot.current_balance * order_size_percent

            if buy_amount < 1:
                self._mean_reversion_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="Mean Reversion: Insufficient balance for entry"
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
        long_period = params.get("long_period", 200)
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

                # Confirmed entry - proceed
                # Volatility-adjusted position sizing
                # Risk fixed percentage of capital, position size based on ATR
                risk_amount = bot.current_balance * risk_percent

                if atr > 0:
                    # Position size = risk_amount / (ATR * atr_multiplier)
                    # This gives us the position size in quote currency (USDT)
                    position_size = risk_amount / (atr * atr_multiplier)
                else:
                    # Fallback: use risk_amount directly if ATR is 0
                    position_size = risk_amount

                # Cap position size at available balance
                buy_amount = min(position_size, bot.current_balance)

                if buy_amount < 1:
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason="Trend Following: Insufficient balance for entry"
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

            # Hold position - trend still valid
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Trend Following: Holding position, stop at ${state['trailing_stop']:.2f if state['trailing_stop'] else 'N/A'}"
            )

    async def _strategy_cross_sectional_momentum(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Cross-Sectional Momentum (relative strength) strategy - Institutional Grade.

        Ranks assets by relative performance and only holds positions in top performers.
        Each bot tracks its own trading_pair and enters/exits based on whether
        that pair is in the top N ranked assets.

        IMPORTANT: This strategy operates on SAMPLE-based data (1 sample ≈ 1 loop ≈ 1 second).
        Momentum is calculated as simple return over the last N samples, NOT calendar days.

        Rebalance behavior:
        - New bots: May enter immediately if ranked in top_n
        - Existing bots: Exit only at rebalance time
        - Rank hysteresis: Enter at top_n, exit at top_n + rank_buffer (prevents churn)

        Parameters:
            universe: List of symbols to compare (default: common pairs)
            lookback_samples: Samples for momentum calculation (default: 3600 ≈ 1 hour @ 1s polling)
            lookback_days: [DEPRECATED] Use lookback_samples instead
            top_n: Number of top assets to hold (default: 3)
            rank_buffer: Hysteresis buffer for exits (default: 1, exit at top_n + buffer)
            rebalance_hours: Hours between rebalances (default: 168 = weekly)
            allocation_percent: Percent of capital to allocate (default: 100)
            trend_filter_enabled: Enable global trend filter (default: False)
            trend_filter_symbol: Symbol for trend filter (default: BTC/USDT)
            trend_filter_ema: EMA period for trend filter (default: 200)
        """
        # Get parameters with backward compatibility
        universe = params.get("universe", [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
            "DOGE/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT"
        ])

        # Handle deprecated lookback_days parameter
        if "lookback_days" in params and "lookback_samples" not in params:
            logger.warning(
                f"Bot {bot.id}: Cross-Sectional parameter 'lookback_days' is DEPRECATED. "
                f"Use 'lookback_samples' instead. Note: 1 sample ≈ 1 second, not 1 day!"
            )
            lookback_samples = params.get("lookback_days", 3600)
        else:
            lookback_samples = params.get("lookback_samples", 3600)  # 1 hour @ 1s polling

        top_n = params.get("top_n", 3)
        rank_buffer = params.get("rank_buffer", 1)  # Hysteresis: exit at top_n + buffer
        rebalance_hours = params.get("rebalance_hours", 168)  # Weekly
        allocation_percent = params.get("allocation_percent", 100) / 100
        trend_filter_enabled = params.get("trend_filter_enabled", False)
        trend_filter_symbol = params.get("trend_filter_symbol", "BTC/USDT")
        trend_filter_ema = params.get("trend_filter_ema", 200)

        # Universe consistency guard: warn if bot's trading_pair was auto-added
        original_universe_size = len(universe)
        if bot.trading_pair not in universe:
            logger.warning(
                f"Bot {bot.id}: Cross-Sectional auto-added {bot.trading_pair} to universe "
                f"(not in configured universe). This may indicate misconfiguration."
            )
            universe.append(bot.trading_pair)

        # Initialize state tracking for cross-sectional data
        if not hasattr(self, "_cross_sectional_states"):
            self._cross_sectional_states = {}

        state = self._cross_sectional_states.get(bot.id, {
            "price_data": {},  # {symbol: [{"price": float, "timestamp": str}]}
            "last_rebalance": None,
            "current_rank": None,
            "last_top_n": [],
            "first_rebalance_logged": False,  # Log universe composition once
        })

        # Get exchange service for fetching multiple ticker prices
        exchange = self._exchange_services.get(bot.id)
        if not exchange:
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Cross-Sectional: Exchange not available"
            )

        # Fetch current prices for all symbols in universe
        try:
            current_prices = {}
            for symbol in universe:
                ticker = await exchange.get_ticker(symbol)
                if ticker:
                    current_prices[symbol] = ticker.last

                    # Store price in history
                    if symbol not in state["price_data"]:
                        state["price_data"][symbol] = []

                    state["price_data"][symbol].append({
                        "price": ticker.last,
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    # Data retention: Keep 2× lookback_samples as buffer (minimum 100 samples)
                    # This ensures sufficient history while preventing unbounded growth
                    max_samples = max(lookback_samples * 2, 100)
                    state["price_data"][symbol] = state["price_data"][symbol][-max_samples:]

        except Exception as e:
            logger.error(f"Bot {bot.id}: Cross-Sectional failed to fetch prices: {e}")
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Cross-Sectional: Error fetching prices"
            )

        # Check if we have enough historical data for momentum calculation
        # Require at least lookback_samples data points (minimum 30 for stability)
        min_data_points = max(min(lookback_samples, 30), 2)
        symbols_with_data = [
            sym for sym, prices in state["price_data"].items()
            if len(prices) >= min_data_points
        ]

        if len(symbols_with_data) < 2:
            # Not enough data yet - need at least 2 symbols for ranking
            self._cross_sectional_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Cross-Sectional: Collecting data ({len(symbols_with_data)}/{len(universe)} symbols ready)"
            )

        # Check if it's time to rebalance
        now = datetime.utcnow()
        should_rebalance = False

        if state["last_rebalance"] is None:
            should_rebalance = True
        else:
            last_rebalance_time = datetime.fromisoformat(state["last_rebalance"])
            hours_since = (now - last_rebalance_time).total_seconds() / 3600
            if hours_since >= rebalance_hours:
                should_rebalance = True

        # Calculate momentum for all symbols with sufficient data
        def calculate_momentum(prices: list, lookback: int) -> float:
            """Calculate simple total return over lookback samples.

            Momentum = (current_price - start_price) / start_price
            where start_price is from N samples ago (1 sample ≈ 1 second).

            Returns 0.0 for invalid/insufficient data.
            """
            if len(prices) < 2:
                return 0.0

            # Use min of lookback and available data
            n = min(lookback, len(prices))
            if n < 2:
                return 0.0

            # Simple return: (current - start) / start
            start_price = prices[-n]["price"]
            end_price = prices[-1]["price"]

            if start_price <= 0:
                return 0.0

            momentum = (end_price - start_price) / start_price

            # Defensive: clamp NaN/inf results to 0
            if not isinstance(momentum, (int, float)) or momentum != momentum:  # NaN check
                return 0.0
            if momentum == float('inf') or momentum == float('-inf'):
                return 0.0

            return momentum

        # Rank all symbols by momentum
        momentum_scores = {}
        for symbol in symbols_with_data:
            prices = state["price_data"][symbol]
            momentum = calculate_momentum(prices, lookback_samples)
            momentum_scores[symbol] = momentum

        # Sort by momentum (descending)
        ranked_symbols = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get top N symbols
        top_n_symbols = [sym for sym, score in ranked_symbols[:top_n]]

        # Find rank of bot's trading pair
        bot_rank = None
        bot_momentum = momentum_scores.get(bot.trading_pair, 0.0)
        for idx, (sym, score) in enumerate(ranked_symbols):
            if sym == bot.trading_pair:
                bot_rank = idx + 1
                break

        # Log universe composition at first rebalance (observability)
        if not state.get("first_rebalance_logged", False):
            logger.info(
                f"Bot {bot.id}: Cross-Sectional universe initialized - "
                f"{len(universe)} symbols: {', '.join(universe[:5])}{'...' if len(universe) > 5 else ''}"
            )
            state["first_rebalance_logged"] = True

        # Check trend filter if enabled
        trend_filter_passed = True
        if trend_filter_enabled:
            filter_symbol_data = state["price_data"].get(trend_filter_symbol, [])
            if len(filter_symbol_data) >= trend_filter_ema:
                # Calculate EMA for trend filter
                prices = [p["price"] for p in filter_symbol_data]

                def calculate_ema(prices: list, period: int) -> float:
                    """Calculate EMA."""
                    if len(prices) < period:
                        return sum(prices) / len(prices)

                    k = 2 / (period + 1)
                    ema = sum(prices[:period]) / period
                    for price in prices[period:]:
                        ema = (price * k) + (ema * (1 - k))
                    return ema

                ema = calculate_ema(prices, trend_filter_ema)
                current_filter_price = filter_symbol_data[-1]["price"]
                trend_filter_passed = current_filter_price > ema

                if not trend_filter_passed:
                    logger.info(
                        f"Bot {bot.id}: Cross-Sectional trend filter FAILED - "
                        f"{trend_filter_symbol} ${current_filter_price:.2f} < EMA ${ema:.2f}"
                    )

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # Rank-based decision logic with hysteresis (stability)
        # Entry rule: rank ≤ top_n
        # Exit rule: rank > (top_n + rank_buffer)
        # This prevents churn when a symbol oscillates around the top_n threshold
        is_in_top_n = bot.trading_pair in top_n_symbols
        is_outside_buffer = bot_rank is not None and bot_rank > (top_n + rank_buffer)

        # Update state
        state["current_rank"] = bot_rank
        state["last_top_n"] = top_n_symbols

        if should_rebalance:
            state["last_rebalance"] = now.isoformat()
            # Log rebalance with ranked list for explainability
            ranked_str = ", ".join([f"{sym}(#{i+1}:{score:.2%})" for i, (sym, score) in enumerate(ranked_symbols[:min(top_n + 2, len(ranked_symbols))])])
            logger.info(
                f"Bot {bot.id}: Cross-Sectional REBALANCE - "
                f"Top {top_n}: {', '.join(top_n_symbols)} | "
                f"Rankings: {ranked_str}"
            )

        self._cross_sectional_states[bot.id] = state

        # Exit if trend filter fails (regardless of rank)
        if trend_filter_enabled and not trend_filter_passed:
            if has_position:
                for pos in positions:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Cross-Sectional EXIT (trend filter) - "
                        f"Selling {bot.trading_pair}"
                    )

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason="Cross-Sectional: Exit due to trend filter failure",
                    )

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Cross-Sectional: Trend filter failed, staying out"
            )

        # REBALANCE LOGIC (institutional-grade with hysteresis)
        #
        # Entry behavior:
        #   - New bots (no position): May enter IMMEDIATELY if rank ≤ top_n
        #   - Existing bots: Only evaluated at rebalance time
        #
        # Exit behavior (with rank hysteresis for stability):
        #   - Exit ONLY at rebalance time
        #   - Exit when rank > (top_n + rank_buffer)
        #   - Hold when rank ∈ [top_n + 1, top_n + rank_buffer] (buffer zone)
        #
        # This prevents churn from minor rank fluctuations near the threshold.

        if should_rebalance or not has_position:
            # ENTRY: pair is in top_n AND no position
            if is_in_top_n and not has_position:
                # Calculate position size
                buy_amount = bot.current_balance * allocation_percent

                if buy_amount < 1:
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason="Cross-Sectional: Insufficient balance for entry"
                    )

                logger.info(
                    f"Bot {bot.id}: Cross-Sectional ENTRY - "
                    f"{bot.trading_pair} ranked #{bot_rank}/{len(ranked_symbols)} (top {top_n}), "
                    f"momentum: {bot_momentum:.2%} over {lookback_samples} samples"
                )

                return TradeSignal(
                    action="buy",
                    amount=buy_amount,
                    order_type="market",
                    reason=f"Cross-Sectional: Entry at rank #{bot_rank} (momentum: {bot_momentum:.2%})",
                )

            # EXIT: pair fell OUTSIDE buffer zone (rank > top_n + rank_buffer) AND has position
            elif is_outside_buffer and has_position:
                for pos in positions:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Cross-Sectional EXIT - "
                        f"{bot.trading_pair} rank #{bot_rank} > threshold (top_n={top_n} + buffer={rank_buffer}={top_n + rank_buffer}), "
                        f"momentum: {bot_momentum:.2%}"
                    )

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Cross-Sectional: Exit, rank #{bot_rank} outside buffer (threshold: {top_n + rank_buffer})",
                    )

        # HOLD LOGIC (policy-based reasons)
        if has_position:
            # Holding position - explain why not exiting
            if is_in_top_n:
                # Still in top_n - no reason to exit
                reason = f"Cross-Sectional: Holding rank #{bot_rank} (in top {top_n})"
            elif not is_outside_buffer:
                # In buffer zone [top_n+1, top_n+rank_buffer] - hysteresis prevents exit
                reason = f"Cross-Sectional: Holding rank #{bot_rank} (in buffer, threshold={top_n + rank_buffer})"
            else:
                # Outside buffer but not rebalance time - wait for rebalance
                hours_remaining = rebalance_hours - ((now - datetime.fromisoformat(state['last_rebalance'])).total_seconds() / 3600)
                reason = f"Cross-Sectional: Holding rank #{bot_rank}, rebalance in {hours_remaining:.1f}h"

            return TradeSignal(
                action="hold",
                amount=0,
                reason=reason
            )
        else:
            # No position - explain why not entering
            if is_in_top_n:
                # In top_n but not rebalance time - wait
                hours_remaining = rebalance_hours - ((now - datetime.fromisoformat(state['last_rebalance'])).total_seconds() / 3600)
                reason = f"Cross-Sectional: Rank #{bot_rank} (in top {top_n}), waiting for rebalance in {hours_remaining:.1f}h"
            else:
                # Not in top_n - no entry
                reason = f"Cross-Sectional: Rank #{bot_rank}/{len(ranked_symbols)}, not in top {top_n}"

            return TradeSignal(
                action="hold",
                amount=0,
                reason=reason
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
        min_compression_bars = params.get("min_compression_bars", 20)  # SPARSE: was 5
        atr_stop_mult = params.get("atr_stop_multiplier", 2.0)
        risk_percent = params.get("risk_percent", 1.0) / 100
        cooldown_hours = params.get("cooldown_hours", 72)  # SPARSE: was 24
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
            # Get price history for regime detection (uses existing tick data)
            price_history_for_regime = self._get_price_history(bot.id)
            price_history_for_regime.append(current_price)

            # Detect current market regime
            current_regime = self._detect_market_regime(price_history_for_regime, None)
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

            # Update state and hold
            self._volatility_breakout_states[bot.id] = state

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Holding position, stop at ${state['trailing_stop']:.2f if state['trailing_stop'] else 'N/A'}"
            )

        # === ENTRY LOGIC (RARE, REGIME-AWARE) ===

        # Regime gate: Block entries if wrong market conditions
        if not regime_allows_entry:
            self._volatility_breakout_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: PAUSED (regime={volatility_regime_name}, waiting for {allowed_regimes})"
            )

        # Cooldown period (sparse trading)
        if state["last_breakout_attempt"] is not None:
            last_attempt = datetime.fromisoformat(state["last_breakout_attempt"])
            hours_since = (datetime.utcnow() - last_attempt).total_seconds() / 3600

            if hours_since < cooldown_hours:
                self._volatility_breakout_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Volatility Breakout: Cooldown ({cooldown_hours - hours_since:.1f}h remaining)"
                )

        # === COMPRESSION DETECTION (BAR-BASED) ===
        # Only update compression state when bar completes (not on every tick)
        is_compressed = False

        if compression_method == "bb_width":
            # Use Bollinger Band width percentile
            if len(state["bb_width_history"]) >= 20:
                # Calculate percentile threshold
                sorted_widths = sorted(state["bb_width_history"])
                percentile_index = int(len(sorted_widths) * (compression_percentile / 100))
                percentile_value = sorted_widths[percentile_index]

                is_compressed = bb_width <= percentile_value

                logger.debug(
                    f"Bot {bot.id}: Volatility Breakout compression check - "
                    f"BB width: {bb_width:.4f}, {compression_percentile}th percentile: {percentile_value:.4f}, "
                    f"Compressed: {is_compressed}"
                )

        elif compression_method == "atr_average":
            # Use ATR below its rolling average
            if len(state["atr_history"]) >= 20:
                avg_atr = sum(state["atr_history"][-20:]) / 20
                is_compressed = atr <= (avg_atr * atr_threshold_mult)

                logger.debug(
                    f"Bot {bot.id}: Volatility Breakout compression check - "
                    f"ATR: {atr:.4f}, Avg ATR: {avg_atr:.4f}, "
                    f"Threshold: {avg_atr * atr_threshold_mult:.4f}, Compressed: {is_compressed}"
                )

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

        # Check if compression has persisted long enough
        compression_satisfied = (
            state["compression_active"] and
            state["compression_bars"] >= min_compression_bars
        )

        # === BREAKOUT ENTRY CONDITION (LONG-ONLY, UPPER BAND) ===
        # Requires: sufficient compression + breakout above upper BB
        if compression_satisfied and last_bar_close > upper_band:
            # Volatility-adjusted position sizing
            risk_amount = bot.current_balance * risk_percent

            if atr > 0:
                # Position size = risk_amount / (ATR * stop_multiplier)
                position_size = risk_amount / (atr * atr_stop_mult)
            else:
                position_size = risk_amount

            # Cap at available balance
            buy_amount = min(position_size, bot.current_balance)

            if buy_amount < 1:
                self._volatility_breakout_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="Volatility Breakout: Insufficient balance for entry"
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
            self._record_strategy_failure(auto_state, current_strategy, force_exit_reason, now, cooldown_hours_default)

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

        return signal

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

    def _record_strategy_failure(
        self,
        auto_state: dict,
        strategy_name: str,
        reason: str,
        now: datetime,
        cooldown_hours: float
    ) -> None:
        """Record strategy failure and apply cooldown.

        Args:
            auto_state: Auto mode state
            strategy_name: Name of failed strategy
            reason: Failure reason
            now: Current datetime
            cooldown_hours: Hours to apply cooldown
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
            "cross_sectional_momentum": {
                "allowed_regimes": ["trend_up"],
                "priority": 5,
                "typical_holding_time": "long",
                "description": "Top performer selector in bull markets"
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
        # === STEP 3: PORTFOLIO RISK CAPS CHECK ===
        portfolio_risk = PortfolioRiskService(session)
        portfolio_check = await portfolio_risk.check_portfolio_risk(
            bot.id,
            signal.amount,
            signal.action,
        )

        if not portfolio_check.ok:
            logger.warning(
                f"Bot {bot.id}: Trade REJECTED by portfolio risk caps - "
                f"{portfolio_check.violated_cap}: {portfolio_check.details}"
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
                logger.warning(
                    f"Bot {bot.id}: Trade REJECTED by strategy capacity limits - "
                    f"{capacity_check.reason}"
                )
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
            exchange_fee_pct=bot.exchange_fee or 0.0,
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
        # Ensure order size is still meaningful after adjustments
        min_order_size = 10.0  # $10 minimum
        if signal.amount < min_order_size:
            logger.warning(
                f"Bot {bot.id}: Trade REJECTED - order size ${signal.amount:.2f} < ${min_order_size:.2f} minimum"
            )
            return None

        # === STEP 7: EXECUTION LAYER ROUTING ===
        execution_mode = signal.execution or "market"

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
        # Calculate amount in base currency
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
            logger.debug(f"Bot {bot.id}: Placing limit order: {side} {amount_base:.6f} {bot.trading_pair} @ ${signal.price or current_price:.2f}")
            exchange_order = await exchange.place_limit_order(
                bot.trading_pair, side, amount_base, signal.price or current_price
            )

        if not exchange_order:
            logger.error(f"Bot {bot.id}: Failed to place order")
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
            fees=exchange_order.fee,
            status=OrderStatus.FILLED if exchange_order.status == "closed" else OrderStatus.PENDING,
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
        # CRITICAL: This is the single source of truth for all financial transactions
        # All balance changes, tax lots, and realized gains are recorded here

        # Parse trading pair to get base and quote assets
        base_asset, quote_asset = bot.trading_pair.split('/')

        # Determine owner_id (TODO: Get from Bot model when owner_id field exists)
        owner_id = str(bot.id)  # FIXME: Use bot.owner_id when available

        # Record trade execution (creates Trade record + ledger entries)
        trade_recorder = TradeRecorderService(session)
        trade = await trade_recorder.record_trade(
            order_id=order.id,
            owner_id=owner_id,
            bot_id=bot.id,
            exchange=bot.exchange if hasattr(bot, 'exchange') else 'simulated',
            trading_pair=bot.trading_pair,
            side=TradeSide.BUY if signal.action == "buy" else TradeSide.SELL,
            base_asset=base_asset,
            quote_asset=quote_asset,
            base_amount=exchange_order.amount,
            quote_amount=signal.amount,
            price=exchange_order.price,
            fee_amount=exchange_order.fee,
            fee_asset=quote_asset,
            modeled_cost=cost_estimate.total_cost,
            exchange_trade_id=exchange_order.id,
            executed_at=datetime.utcnow(),
            strategy_used=bot.strategy,
        )

        # Process tax lots (FIFO cost basis tracking)
        tax_engine = FIFOTaxEngine(session)
        if signal.action == "buy":
            # BUY creates a new tax lot
            await tax_engine.process_buy(trade)
            logger.info(
                f"Bot {bot.id}: Created tax lot for {trade.base_amount:.8f} {base_asset} "
                f"@ ${trade.get_cost_basis_per_unit():.2f}/unit"
            )
        else:
            # SELL consumes tax lots in FIFO order and records realized gains
            realized_gains = await tax_engine.process_sell(trade)
            if realized_gains:
                total_gain = sum(g.gain_loss for g in realized_gains)
                logger.info(
                    f"Bot {bot.id}: Realized gain/loss ${total_gain:+.2f} "
                    f"from {len(realized_gains)} tax lot(s)"
                )

        # === ACCOUNTING INVARIANT VALIDATION ===
        # CRITICAL: Validate all accounting invariants before updating cached state
        # If validation fails, exception is raised and trading HALTS
        invariant_validator = LedgerInvariantService(session)
        try:
            await invariant_validator.validate_trade(trade.id)
        except ValidationError as e:
            logger.critical(
                f"Bot {bot.id}: ACCOUNTING VALIDATION FAILED for trade {trade.id}. "
                f"Trading HALTED. Error: {e}"
            )
            # Rollback the transaction to prevent corrupt state
            await session.rollback()
            raise  # Re-raise to halt trading loop

        # Update wallet (LEGACY - kept for backward compatibility)
        # NOTE: Ledger entries are already created by trade_recorder
        wallet = VirtualWalletService(session)
        total_cost = exchange_order.fee + cost_estimate.total_cost

        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -total_cost, 0)
        else:
            await wallet.record_trade_result(bot.id, -total_cost, 0)

        # Update running balance
        result = await session.execute(select(Bot).where(Bot.id == bot.id))
        updated_bot = result.scalar_one_or_none()
        if updated_bot:
            order.running_balance_after = updated_bot.current_balance

        # Update/create position (derived state cache)
        if signal.action == "buy":
            await self._open_or_add_position(
                bot.id, bot.trading_pair, exchange_order.amount,
                exchange_order.price, session
            )
        else:
            await self._close_or_reduce_position(
                bot.id, bot.trading_pair, exchange_order.amount,
                exchange_order.price, session, wallet
            )

        # Commit all changes (order, trade, ledger entries, tax lots, gains)
        await session.commit()

        # Export to CSV (async, best-effort - failures don't block trading)
        try:
            csv_exporter = CSVExportService(session)
            # Include is_simulated in filename to prevent mixing live/simulated data
            suffix = "simulated" if bot.is_dry_run else "live"
            log_path = Path(f"backend/logs/{bot.id}/trades_{suffix}.csv")
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
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, exchange_order.amount,
                exchange_order.price, session
            )
        else:
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._close_or_reduce_position(
                bot.id, bot.trading_pair, exchange_order.amount,
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
        logger.info(
            f"Bot {bot.id}: VWAP execution - "
            f"Using market order (no real volume data available). "
            f"Benchmark VWAP: ${vwap_benchmark:.2f if vwap_benchmark else 'N/A'}"
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
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._open_or_add_position(
                bot.id, bot.trading_pair, exchange_order.amount,
                exchange_order.price, session
            )
        else:
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
            await self._close_or_reduce_position(
                bot.id, bot.trading_pair, exchange_order.amount,
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
    ) -> None:
        """Close or reduce a position and realize P&L."""
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

        # Calculate realized P&L
        sell_amount = min(amount, position.amount)
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
        strategies = [
            "dca_accumulator",
            "adaptive_grid",
            "mean_reversion",
            "twap",
        ]

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

        async with async_session_maker() as session:
            # Find all bots that were running
            result = await session.execute(
                select(Bot).where(Bot.status == BotStatus.RUNNING)
            )
            running_bots = result.scalars().all()

            if not running_bots:
                logger.info("No bots to resume on startup")
                return 0

            logger.info(f"Found {len(running_bots)} bot(s) to resume")

            for bot in running_bots:
                try:
                    # Restore strategy state from database if available
                    await self._restore_strategy_state(bot.id, bot.strategy_params)

                    # Create exchange service
                    if bot.is_dry_run:
                        exchange = SimulatedExchangeService(initial_balance=bot.budget)
                    else:
                        exchange = ExchangeService()

                    await exchange.connect()
                    self._exchange_services[bot.id] = exchange

                    # Initialize per-bot file logger
                    ensure_bot_log_directory(bot.id)
                    self._bot_loggers[bot.id] = BotLoggingService(
                        bot.id, bot.name, bot.is_dry_run
                    )
                    self._bot_loggers[bot.id].log_activity(
                        f"Bot resumed after server restart"
                    )

                    # Start bot task
                    self._stop_flags[bot.id] = False
                    task = asyncio.create_task(self._run_bot_loop(bot.id))
                    self._running_bots[bot.id] = task

                    resumed += 1
                    logger.info(f"Resumed bot {bot.id} ({bot.name})")

                except Exception as e:
                    logger.error(f"Failed to resume bot {bot.id}: {e}")

        logger.info(f"Resumed {resumed} bot(s) on startup")
        return resumed

    async def _restore_strategy_state(self, bot_id: int, strategy_params: dict) -> None:
        """Restore strategy state from saved data.

        Args:
            bot_id: Bot ID
            strategy_params: Strategy params which may contain saved state
        """
        if not strategy_params:
            return

        # Restore grid state
        if "_grid_state" in strategy_params:
            if not hasattr(self, "_grid_states"):
                self._grid_states = {}
            self._grid_states[bot_id] = strategy_params["_grid_state"]
            logger.debug(f"Bot {bot_id}: Restored grid state")

        # Restore TWAP state
        if "_twap_state" in strategy_params:
            if not hasattr(self, "_twap_states"):
                self._twap_states = {}
            self._twap_states[bot_id] = strategy_params["_twap_state"]
            logger.debug(f"Bot {bot_id}: Restored TWAP state")

        # Restore VWAP state
        if "_vwap_state" in strategy_params:
            if not hasattr(self, "_vwap_states"):
                self._vwap_states = {}
            self._vwap_states[bot_id] = strategy_params["_vwap_state"]
            logger.debug(f"Bot {bot_id}: Restored VWAP state")

        # Restore Auto mode state
        if "_auto_state" in strategy_params:
            if not hasattr(self, "_auto_states"):
                self._auto_states = {}
            self._auto_states[bot_id] = strategy_params["_auto_state"]
            logger.debug(f"Bot {bot_id}: Restored Auto mode state")

        # Restore price history
        if "_price_history" in strategy_params:
            if not hasattr(self, "_price_histories"):
                self._price_histories = {}
            self._price_histories[bot_id] = strategy_params["_price_history"]
            logger.debug(f"Bot {bot_id}: Restored price history")

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

        # Collect strategy state
        strategy_params = dict(bot.strategy_params) if bot.strategy_params else {}

        # Save grid state
        if hasattr(self, "_grid_states") and bot_id in self._grid_states:
            strategy_params["_grid_state"] = self._grid_states[bot_id]

        # Save TWAP state
        if hasattr(self, "_twap_states") and bot_id in self._twap_states:
            strategy_params["_twap_state"] = self._twap_states[bot_id]

        # Save VWAP state
        if hasattr(self, "_vwap_states") and bot_id in self._vwap_states:
            strategy_params["_vwap_state"] = self._vwap_states[bot_id]

        # Save Auto mode state
        if hasattr(self, "_auto_states") and bot_id in self._auto_states:
            strategy_params["_auto_state"] = self._auto_states[bot_id]

        # Save price history (limited to last 50 to avoid bloat)
        if hasattr(self, "_price_histories") and bot_id in self._price_histories:
            strategy_params["_price_history"] = self._price_histories[bot_id][-50:]

        # Update database
        bot.strategy_params = strategy_params
        bot.updated_at = datetime.utcnow()

        logger.debug(f"Saved state for bot {bot_id}")


# Global trading engine instance
trading_engine = TradingEngine()
