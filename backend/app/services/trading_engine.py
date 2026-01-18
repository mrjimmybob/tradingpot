"""Trading engine for bot execution and management."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models import (
    Bot, BotStatus, Order, OrderType, OrderStatus,
    Position, PositionSide, PnLSnapshot,
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

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trading signal from strategy."""
    action: str  # "buy", "sell", "hold"
    amount: float  # Amount in quote currency (e.g., USDT)
    price: Optional[float] = None  # For limit orders
    order_type: str = "market"  # "market" or "limit"
    reason: str = ""


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
        """
        strategies = {
            "dca_accumulator": self._strategy_dca,
            "adaptive_grid": self._strategy_grid,
            "mean_reversion": self._strategy_mean_reversion,
            "trend_following": self._strategy_trend_following,
            "cross_sectional_momentum": self._strategy_cross_sectional_momentum,
            "volatility_breakout": self._strategy_volatility_breakout,
            "twap": self._strategy_twap,
            "vwap": self._strategy_vwap,
            "scalping": self._strategy_scalping,
            "auto_mode": self._strategy_auto,
        }
        return strategies.get(strategy_name)

    async def _strategy_dca(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """DCA (Dollar Cost Averaging) strategy.

        Buys at regular intervals regardless of price.

        Parameters:
            interval_minutes: Time between buys (default: 60)
            amount_percent: Percent of budget per buy (default: 10)
            amount_usd: Fixed USD amount per buy (overrides amount_percent if set)
            immediate_first_buy: Execute first buy immediately (default: True)
        """
        interval_minutes = params.get("interval_minutes", 60)
        amount_percent = params.get("amount_percent", 10) / 100
        amount_usd = params.get("amount_usd")  # Fixed amount in USD
        immediate_first_buy = params.get("immediate_first_buy", True)

        # Get order history for this bot
        last_order = await self._get_last_order(bot.id, session)
        order_count = await self._get_order_count(bot.id, session)

        # Check if it's time to buy
        if last_order:
            time_since_last = datetime.utcnow() - last_order.created_at
            seconds_since_last = time_since_last.total_seconds()
            interval_seconds = interval_minutes * 60

            if seconds_since_last < interval_seconds:
                remaining = interval_seconds - seconds_since_last
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"DCA: Next buy in {remaining/60:.1f} minutes"
                )
        elif not immediate_first_buy:
            # No orders yet but not immediate - check time since bot started
            if bot.started_at:
                time_since_start = datetime.utcnow() - bot.started_at
                if time_since_start.total_seconds() < interval_minutes * 60:
                    remaining = (interval_minutes * 60) - time_since_start.total_seconds()
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason=f"DCA: First buy in {remaining/60:.1f} minutes"
                    )

        # Calculate buy amount
        if amount_usd and amount_usd > 0:
            # Use fixed USD amount
            buy_amount = min(amount_usd, bot.current_balance)
        else:
            # Use percentage of current balance
            buy_amount = bot.current_balance * amount_percent

        # Minimum order check
        min_order = 1.0  # Minimum $1 order
        if buy_amount < min_order:
            if bot.current_balance < min_order:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="DCA: Budget exhausted"
                )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"DCA: Amount ${buy_amount:.2f} below minimum ${min_order}"
            )

        # Check we don't exceed available balance
        if buy_amount > bot.current_balance:
            buy_amount = bot.current_balance

        logger.info(
            f"Bot {bot.id}: DCA buy #{order_count + 1} - "
            f"${buy_amount:.2f} at ${current_price:.2f}"
        )

        return TradeSignal(
            action="buy",
            amount=buy_amount,
            order_type="market",
            reason=f"DCA buy #{order_count + 1} at ${current_price:.2f}",
        )

    async def _strategy_grid(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Adaptive Grid trading strategy.

        Creates a grid of buy/sell levels around the current price.
        Buys when price drops to lower grid levels, sells when price rises to upper levels.

        Parameters:
            grid_count: Number of grid levels (default: 10)
            grid_spacing_percent: Spacing between levels as % (default: 1.0)
            range_percent: Total grid range as % (default: 10)
            order_size_percent: Percent of budget per grid order (default: 10)
        """
        grid_count = params.get("grid_count", 10)
        grid_spacing = params.get("grid_spacing_percent", 1.0) / 100
        range_percent = params.get("range_percent", 10) / 100
        order_size_percent = params.get("order_size_percent", 10) / 100

        # Get grid state from bot's strategy params or initialize
        grid_state = self._get_grid_state(bot.id, params)

        # Calculate grid levels if not set
        if "center_price" not in grid_state or grid_state.get("needs_recenter", False):
            grid_state["center_price"] = current_price
            grid_state["needs_recenter"] = False
            grid_state["last_trade_level"] = 0
            logger.info(f"Bot {bot.id}: Grid centered at ${current_price:.2f}")

        center_price = grid_state["center_price"]
        last_trade_level = grid_state.get("last_trade_level", 0)

        # Calculate which grid level the current price is at
        # Positive = above center, Negative = below center
        if current_price > center_price:
            price_diff_pct = (current_price - center_price) / center_price
            current_level = int(price_diff_pct / grid_spacing)
        else:
            price_diff_pct = (center_price - current_price) / center_price
            current_level = -int(price_diff_pct / grid_spacing)

        # Check if price crossed a new grid level
        level_change = current_level - last_trade_level

        if level_change == 0:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Grid: At level {current_level}, waiting for move"
            )

        # Calculate order amount
        order_amount = bot.budget * order_size_percent

        # Ensure we have enough balance
        positions = await self._get_bot_positions(bot.id, session)
        total_position_value = sum(p.amount * p.current_price for p in positions)

        if level_change > 0:
            # Price went UP - SELL at upper grid level
            if total_position_value < order_amount * 0.5:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: No position to sell at level {current_level}"
                )

            sell_amount = min(order_amount, total_position_value * 0.5)

            # Update grid state
            grid_state["last_trade_level"] = current_level
            self._save_grid_state(bot.id, grid_state)

            logger.info(
                f"Bot {bot.id}: Grid SELL at level {current_level} "
                f"(${current_price:.2f}), amount ${sell_amount:.2f}"
            )

            return TradeSignal(
                action="sell",
                amount=sell_amount,
                order_type="market",
                reason=f"Grid: Sell at level {current_level} (${current_price:.2f})",
            )

        else:
            # Price went DOWN - BUY at lower grid level
            if bot.current_balance < order_amount * 0.5:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Grid: Insufficient balance for buy at level {current_level}"
                )

            buy_amount = min(order_amount, bot.current_balance * 0.5)

            # Update grid state
            grid_state["last_trade_level"] = current_level
            self._save_grid_state(bot.id, grid_state)

            logger.info(
                f"Bot {bot.id}: Grid BUY at level {current_level} "
                f"(${current_price:.2f}), amount ${buy_amount:.2f}"
            )

            return TradeSignal(
                action="buy",
                amount=buy_amount,
                order_type="market",
                reason=f"Grid: Buy at level {current_level} (${current_price:.2f})",
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
        """Mean Reversion strategy with Bollinger bands.

        Buys when price touches lower band, sells when price returns to mean or upper band.

        Parameters:
            bollinger_period: Number of periods for SMA (default: 20)
            bollinger_std: Standard deviation multiplier (default: 2.0)
            order_size_percent: Percent of budget per order (default: 20)
            exit_at_mean: Exit at mean instead of upper band (default: True)
        """
        period = params.get("bollinger_period", 20)
        std_mult = params.get("bollinger_std", 2.0)
        order_size_percent = params.get("order_size_percent", 20) / 100
        exit_at_mean = params.get("exit_at_mean", True)

        # Get price history
        price_history = self._get_price_history(bot.id)

        # Add current price to history
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history)

        # Need enough data for Bollinger calculation
        if len(price_history) < period:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Collecting data ({len(price_history)}/{period})"
            )

        # Calculate Bollinger bands
        recent_prices = price_history[-period:]
        sma = sum(recent_prices) / len(recent_prices)

        # Calculate standard deviation
        variance = sum((p - sma) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = variance ** 0.5

        upper_band = sma + (std_mult * std_dev)
        lower_band = sma - (std_mult * std_dev)

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        logger.debug(
            f"Bot {bot.id}: Mean Reversion - Price: ${current_price:.2f}, "
            f"SMA: ${sma:.2f}, Upper: ${upper_band:.2f}, Lower: ${lower_band:.2f}"
        )

        # Trading logic
        if not has_position:
            # No position - look for buy opportunity
            if current_price <= lower_band:
                # Price at or below lower band - BUY
                buy_amount = bot.current_balance * order_size_percent

                if buy_amount < 1:
                    return TradeSignal(
                        action="hold",
                        amount=0,
                        reason="Mean Reversion: Insufficient balance"
                    )

                logger.info(
                    f"Bot {bot.id}: Mean Reversion BUY - "
                    f"Price ${current_price:.2f} <= Lower band ${lower_band:.2f}"
                )

                return TradeSignal(
                    action="buy",
                    amount=buy_amount,
                    order_type="market",
                    reason=f"Mean Reversion: Buy at lower band (${lower_band:.2f})",
                )

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Waiting for lower band (${lower_band:.2f})"
            )

        else:
            # Have position - look for exit
            for pos in positions:
                # Determine exit level
                if exit_at_mean:
                    exit_level = sma
                    exit_label = "mean"
                else:
                    exit_level = upper_band
                    exit_label = "upper band"

                if current_price >= exit_level:
                    # Price at or above exit level - SELL
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Mean Reversion SELL - "
                        f"Price ${current_price:.2f} >= {exit_label} ${exit_level:.2f}"
                    )

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Mean Reversion: Sell at {exit_label} (${exit_level:.2f})",
                    )

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Mean Reversion: Holding, target ${sma:.2f if exit_at_mean else upper_band:.2f}"
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
        """Trend Following (time-series momentum) strategy.

        Conservative long-only momentum strategy using EMA crossover and ATR-based stops.
        Enters when price > EMA(long) and EMA(short) > EMA(long).
        Exits when price closes below EMA(long) or trailing stop hit.

        Parameters:
            short_period: EMA short period (default: 50)
            long_period: EMA long period (default: 200)
            atr_period: ATR period (default: 14)
            atr_multiplier: ATR multiplier for stop loss (default: 2.0)
            risk_percent: Percent of capital to risk per trade (default: 1.0)
        """
        short_period = params.get("short_period", 50)
        long_period = params.get("long_period", 200)
        atr_period = params.get("atr_period", 14)
        atr_multiplier = params.get("atr_multiplier", 2.0)
        risk_percent = params.get("risk_percent", 1.0) / 100

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

        # Calculate ATR (Average True Range) for volatility
        def calculate_atr(prices: list, period: int) -> float:
            """Calculate ATR from price history."""
            if len(prices) < period + 1:
                # Not enough data, use simple range
                return max(prices[-period:]) - min(prices[-period:]) if len(prices) >= period else 0

            true_ranges = []
            for i in range(1, len(prices)):
                high_low = abs(prices[i] - prices[i-1])
                true_ranges.append(high_low)

            # Use last 'period' true ranges
            recent_trs = true_ranges[-period:]
            return sum(recent_trs) / len(recent_trs)

        # Calculate indicators
        ema_short = calculate_ema(price_history, short_period)
        ema_long = calculate_ema(price_history, long_period)
        atr = calculate_atr(price_history, atr_period)

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        # Get or initialize trailing stop state
        if not hasattr(self, "_trend_states"):
            self._trend_states = {}

        state = self._trend_states.get(bot.id, {"trailing_stop": None, "highest_price": None})

        logger.debug(
            f"Bot {bot.id}: Trend Following - Price: ${current_price:.2f}, "
            f"EMA({short_period}): ${ema_short:.2f}, EMA({long_period}): ${ema_long:.2f}, "
            f"ATR: ${atr:.2f}"
        )

        # Trading logic
        if not has_position:
            # No position - look for entry signal
            # Entry: price > EMA(long) AND EMA(short) > EMA(long)
            if current_price > ema_long and ema_short > ema_long:
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
                    f"EMA({short_period}) ${ema_short:.2f} > EMA({long_period})"
                )

                # Initialize trailing stop
                trailing_stop_price = current_price - (atr * atr_multiplier)
                self._trend_states[bot.id] = {
                    "trailing_stop": trailing_stop_price,
                    "highest_price": current_price,
                }

                return TradeSignal(
                    action="buy",
                    amount=buy_amount,
                    order_type="market",
                    reason=f"Trend Following: Bullish trend detected (EMA cross confirmed)",
                )

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
                # Update trailing stop if price made new high
                if state["highest_price"] is None or current_price > state["highest_price"]:
                    state["highest_price"] = current_price
                    state["trailing_stop"] = current_price - (atr * atr_multiplier)
                    self._trend_states[bot.id] = state

                # Exit condition 1: Price closes below EMA(long)
                if current_price < ema_long:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Trend Following EXIT (trend break) - "
                        f"Price ${current_price:.2f} < EMA({long_period}) ${ema_long:.2f}"
                    )

                    # Clear state
                    self._trend_states[bot.id] = {"trailing_stop": None, "highest_price": None}

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Trend Following: Exit on trend break (price < EMA({long_period}))",
                    )

                # Exit condition 2: Trailing stop hit
                if state["trailing_stop"] is not None and current_price <= state["trailing_stop"]:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Trend Following EXIT (trailing stop) - "
                        f"Price ${current_price:.2f} <= Stop ${state['trailing_stop']:.2f}"
                    )

                    # Clear state
                    self._trend_states[bot.id] = {"trailing_stop": None, "highest_price": None}

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Trend Following: Exit on trailing stop (${state['trailing_stop']:.2f})",
                    )

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
        """Cross-Sectional Momentum (relative strength) strategy.

        Ranks assets by relative performance and only holds positions in top performers.
        Each bot tracks its own trading_pair and enters/exits based on whether
        that pair is in the top N ranked assets.

        Parameters:
            universe: List of symbols to compare (default: common pairs)
            lookback_days: Days to calculate momentum (default: 60)
            top_n: Number of top assets to hold (default: 3)
            rebalance_hours: Hours between rebalances (default: 168 = weekly)
            allocation_percent: Percent of capital to allocate (default: 100)
            trend_filter_enabled: Enable global trend filter (default: False)
            trend_filter_symbol: Symbol for trend filter (default: BTC/USDT)
            trend_filter_ema: EMA period for trend filter (default: 200)
        """
        # Get parameters
        universe = params.get("universe", [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
            "DOGE/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT", "MATIC/USDT"
        ])
        lookback_days = params.get("lookback_days", 60)
        top_n = params.get("top_n", 3)
        rebalance_hours = params.get("rebalance_hours", 168)  # Weekly
        allocation_percent = params.get("allocation_percent", 100) / 100
        trend_filter_enabled = params.get("trend_filter_enabled", False)
        trend_filter_symbol = params.get("trend_filter_symbol", "BTC/USDT")
        trend_filter_ema = params.get("trend_filter_ema", 200)

        # Ensure bot's trading pair is in the universe
        if bot.trading_pair not in universe:
            universe.append(bot.trading_pair)

        # Initialize state tracking for cross-sectional data
        if not hasattr(self, "_cross_sectional_states"):
            self._cross_sectional_states = {}

        state = self._cross_sectional_states.get(bot.id, {
            "price_data": {},  # {symbol: [prices]}
            "last_rebalance": None,
            "current_rank": None,
            "last_top_n": [],
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

                    # Keep only last 90 days of data (assuming ~daily samples)
                    state["price_data"][symbol] = state["price_data"][symbol][-90:]

        except Exception as e:
            logger.error(f"Bot {bot.id}: Cross-Sectional failed to fetch prices: {e}")
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Cross-Sectional: Error fetching prices"
            )

        # Check if we have enough historical data
        min_data_points = min(lookback_days, 30)  # Need at least 30 data points
        symbols_with_data = [
            sym for sym, prices in state["price_data"].items()
            if len(prices) >= min_data_points
        ]

        if len(symbols_with_data) < 2:
            # Not enough data yet
            self._cross_sectional_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Cross-Sectional: Collecting data ({len(symbols_with_data)} symbols ready)"
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
            """Calculate simple total return over lookback period."""
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

            return (end_price - start_price) / start_price

        # Rank all symbols by momentum
        momentum_scores = {}
        for symbol in symbols_with_data:
            prices = state["price_data"][symbol]
            momentum = calculate_momentum(prices, lookback_days)
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
        for idx, (sym, score) in enumerate(ranked_symbols):
            if sym == bot.trading_pair:
                bot_rank = idx + 1
                break

        logger.debug(
            f"Bot {bot.id}: Cross-Sectional - {bot.trading_pair} ranked #{bot_rank} "
            f"with {momentum_scores.get(bot.trading_pair, 0):.2%} momentum"
        )

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

        # Decision logic
        is_in_top_n = bot.trading_pair in top_n_symbols

        # Update state
        state["current_rank"] = bot_rank
        state["last_top_n"] = top_n_symbols

        if should_rebalance:
            state["last_rebalance"] = now.isoformat()
            logger.info(
                f"Bot {bot.id}: Cross-Sectional REBALANCE - Top {top_n}: {', '.join(top_n_symbols)}"
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

        # Trading logic based on rebalancing schedule
        if should_rebalance or not has_position:
            # Entry: pair is in top N and no position
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
                    f"{bot.trading_pair} ranked #{bot_rank} (top {top_n}), "
                    f"momentum: {momentum_scores.get(bot.trading_pair, 0):.2%}"
                )

                return TradeSignal(
                    action="buy",
                    amount=buy_amount,
                    order_type="market",
                    reason=f"Cross-Sectional: Entry at rank #{bot_rank} (momentum: {momentum_scores.get(bot.trading_pair, 0):.2%})",
                )

            # Exit: pair fell out of top N and has position
            elif not is_in_top_n and has_position:
                for pos in positions:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Cross-Sectional EXIT - "
                        f"{bot.trading_pair} fell to rank #{bot_rank} (out of top {top_n})"
                    )

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Cross-Sectional: Exit, fell to rank #{bot_rank}",
                    )

        # Hold current state
        if has_position:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Cross-Sectional: Holding rank #{bot_rank}, next rebalance in {rebalance_hours - ((now - datetime.fromisoformat(state['last_rebalance'])).total_seconds() / 3600):.1f}h"
            )
        else:
            if is_in_top_n:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Cross-Sectional: In top {top_n} (#{bot_rank}), waiting for rebalance"
                )
            else:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Cross-Sectional: Rank #{bot_rank}, not in top {top_n}"
                )

    async def _strategy_volatility_breakout(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Volatility Breakout (volatility expansion) strategy.

        Enters on price breakouts following low-volatility compression regimes.
        Uses Bollinger Bands to detect compression and breakouts, ATR for stops.

        Parameters:
            bb_period: Bollinger Band period (default: 20)
            bb_std: Bollinger Band standard deviation (default: 2.0)
            atr_period: ATR period (default: 14)
            compression_method: "bb_width" or "atr_average" (default: "bb_width")
            compression_percentile: BB width percentile threshold (default: 20)
            atr_threshold_multiplier: ATR threshold vs average (default: 0.8)
            min_compression_bars: Minimum bars of compression (default: 5)
            atr_stop_multiplier: ATR stop loss multiplier (default: 2.0)
            risk_percent: Percent of capital to risk (default: 1.0)
            cooldown_hours: Hours to wait between breakout attempts (default: 24)
            failed_breakout_bars: Bars to check for failed breakout (default: 3)
        """
        # Get parameters
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

        # Get price history
        price_history = self._get_price_history(bot.id)

        # Add current price to history
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history, max_len=max(bb_period + 100, 150))

        # Need enough data for calculations
        if len(price_history) < bb_period:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Collecting data ({len(price_history)}/{bb_period})"
            )

        # Calculate Bollinger Bands
        def calculate_bollinger_bands(prices: list, period: int, std_mult: float):
            """Calculate Bollinger Bands."""
            recent_prices = prices[-period:]
            sma = sum(recent_prices) / len(recent_prices)

            # Calculate standard deviation
            variance = sum((p - sma) ** 2 for p in recent_prices) / len(recent_prices)
            std_dev = variance ** 0.5

            upper_band = sma + (std_mult * std_dev)
            lower_band = sma - (std_mult * std_dev)
            bandwidth = (upper_band - lower_band) / sma if sma > 0 else 0

            return sma, upper_band, lower_band, bandwidth

        # Calculate ATR
        def calculate_atr(prices: list, period: int) -> float:
            """Calculate ATR from price history."""
            if len(prices) < period + 1:
                return max(prices[-period:]) - min(prices[-period:]) if len(prices) >= period else 0

            true_ranges = []
            for i in range(1, len(prices)):
                high_low = abs(prices[i] - prices[i-1])
                true_ranges.append(high_low)

            recent_trs = true_ranges[-period:]
            return sum(recent_trs) / len(recent_trs)

        # Current indicators
        sma, upper_band, lower_band, bb_width = calculate_bollinger_bands(
            price_history, bb_period, bb_std
        )
        atr = calculate_atr(price_history, atr_period)

        # Initialize state tracking
        if not hasattr(self, "_volatility_breakout_states"):
            self._volatility_breakout_states = {}

        state = self._volatility_breakout_states.get(bot.id, {
            "bb_width_history": [],
            "atr_history": [],
            "compression_detected": False,
            "compression_start": None,
            "compression_bars": 0,
            "last_breakout_attempt": None,
            "trailing_stop": None,
            "entry_price": None,
            "bars_since_entry": 0,
        })

        # Track historical Bollinger width and ATR for compression detection
        state["bb_width_history"].append(bb_width)
        state["bb_width_history"] = state["bb_width_history"][-100:]  # Keep last 100

        state["atr_history"].append(atr)
        state["atr_history"] = state["atr_history"][-100:]  # Keep last 100

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        logger.debug(
            f"Bot {bot.id}: Volatility Breakout - Price: ${current_price:.2f}, "
            f"BB: [${lower_band:.2f}, ${sma:.2f}, ${upper_band:.2f}], "
            f"Width: {bb_width:.4f}, ATR: ${atr:.2f}"
        )

        # === POSITION EXIT LOGIC ===
        if has_position:
            for pos in positions:
                # Update trailing stop if price made new high
                if state["entry_price"] is not None:
                    state["bars_since_entry"] += 1

                    # Check for failed breakout (price closes back inside BB shortly after entry)
                    if state["bars_since_entry"] <= failed_breakout_bars:
                        if current_price < upper_band:
                            sell_amount = pos.amount * current_price

                            logger.info(
                                f"Bot {bot.id}: Volatility Breakout EXIT (failed breakout) - "
                                f"Price ${current_price:.2f} fell back inside BB after {state['bars_since_entry']} bars"
                            )

                            # Clear state
                            state["trailing_stop"] = None
                            state["entry_price"] = None
                            state["bars_since_entry"] = 0
                            state["compression_detected"] = False
                            state["compression_bars"] = 0
                            self._volatility_breakout_states[bot.id] = state

                            return TradeSignal(
                                action="sell",
                                amount=sell_amount,
                                order_type="market",
                                reason="Volatility Breakout: Failed breakout, price back inside BB",
                            )

                # Update trailing stop
                if state["trailing_stop"] is None or current_price > pos.entry_price:
                    state["trailing_stop"] = current_price - (atr * atr_stop_mult)
                    if current_price > pos.entry_price:
                        # Only update trailing stop higher if profitable
                        state["trailing_stop"] = max(
                            state["trailing_stop"],
                            state.get("trailing_stop", 0)
                        )

                # Check trailing stop
                if state["trailing_stop"] is not None and current_price <= state["trailing_stop"]:
                    sell_amount = pos.amount * current_price

                    logger.info(
                        f"Bot {bot.id}: Volatility Breakout EXIT (trailing stop) - "
                        f"Price ${current_price:.2f} <= Stop ${state['trailing_stop']:.2f}"
                    )

                    # Clear state
                    state["trailing_stop"] = None
                    state["entry_price"] = None
                    state["bars_since_entry"] = 0
                    state["compression_detected"] = False
                    state["compression_bars"] = 0
                    self._volatility_breakout_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Volatility Breakout: Trailing stop hit (${state['trailing_stop']:.2f})",
                    )

            # Update state and hold
            self._volatility_breakout_states[bot.id] = state

            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Holding position, stop at ${state['trailing_stop']:.2f if state['trailing_stop'] else 'N/A'}"
            )

        # === ENTRY LOGIC (no position) ===

        # Check cooldown period
        if state["last_breakout_attempt"] is not None:
            last_attempt = datetime.fromisoformat(state["last_breakout_attempt"])
            hours_since = (datetime.utcnow() - last_attempt).total_seconds() / 3600

            if hours_since < cooldown_hours:
                self._volatility_breakout_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Volatility Breakout: Cooldown active ({cooldown_hours - hours_since:.1f}h remaining)"
                )

        # Detect volatility compression
        is_compressed = False

        if compression_method == "bb_width":
            # Use Bollinger Band width percentile
            if len(state["bb_width_history"]) >= 20:
                # Calculate percentile
                sorted_widths = sorted(state["bb_width_history"])
                percentile_index = int(len(sorted_widths) * (compression_percentile / 100))
                percentile_value = sorted_widths[percentile_index]

                is_compressed = bb_width <= percentile_value

        elif compression_method == "atr_average":
            # Use ATR below its rolling average
            if len(state["atr_history"]) >= 20:
                avg_atr = sum(state["atr_history"][-20:]) / 20
                is_compressed = atr <= (avg_atr * atr_threshold_mult)

        # Track compression duration
        if is_compressed:
            if not state["compression_detected"]:
                state["compression_detected"] = True
                state["compression_start"] = datetime.utcnow().isoformat()
                state["compression_bars"] = 1
            else:
                state["compression_bars"] += 1
        else:
            # Compression ended
            if state["compression_detected"]:
                state["compression_detected"] = False
                state["compression_start"] = None
                state["compression_bars"] = 0

        # Check if compression has persisted long enough
        compression_satisfied = (
            state["compression_detected"] and
            state["compression_bars"] >= min_compression_bars
        )

        # Breakout entry condition
        if compression_satisfied and current_price > upper_band:
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

            logger.info(
                f"Bot {bot.id}: Volatility Breakout ENTRY - "
                f"Breakout after {state['compression_bars']} bars compression, "
                f"Price ${current_price:.2f} > Upper BB ${upper_band:.2f}"
            )

            # Initialize trailing stop
            state["trailing_stop"] = current_price - (atr * atr_stop_mult)
            state["entry_price"] = current_price
            state["bars_since_entry"] = 0
            state["last_breakout_attempt"] = datetime.utcnow().isoformat()

            self._volatility_breakout_states[bot.id] = state

            return TradeSignal(
                action="buy",
                amount=buy_amount,
                order_type="market",
                reason=f"Volatility Breakout: Breakout after {state['compression_bars']} bars compression",
            )

        # Update state and hold
        self._volatility_breakout_states[bot.id] = state

        # Provide feedback on current state
        if compression_satisfied:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Volatility Breakout: Compression active ({state['compression_bars']} bars), waiting for breakout above ${upper_band:.2f}"
            )
        elif state["compression_detected"]:
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

    async def _strategy_twap(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Time-Weighted Average Price (TWAP) execution strategy.

        Executes a large order by splitting it into equal-sized slices
        distributed evenly over a time period.

        Parameters:
            execution_period_minutes: Total execution period (default: 60)
            slice_count: Number of order slices (default: 10)
            total_amount_usd: Total amount to buy in USD (default: uses full budget)
            side: Buy or sell (default: "buy")
        """
        execution_period = params.get("execution_period_minutes", 60)
        slice_count = params.get("slice_count", 10)
        total_amount = params.get("total_amount_usd", bot.budget)
        side = params.get("side", "buy")

        # Get TWAP execution state
        twap_state = self._get_twap_state(bot.id)

        # Initialize execution if needed
        if "start_time" not in twap_state:
            twap_state["start_time"] = datetime.utcnow().isoformat()
            twap_state["slices_executed"] = 0
            twap_state["total_executed"] = 0.0
            twap_state["target_amount"] = total_amount
            twap_state["prices"] = []
            self._save_twap_state(bot.id, twap_state)
            logger.info(
                f"Bot {bot.id}: TWAP started - "
                f"${total_amount:.2f} over {execution_period} minutes in {slice_count} slices"
            )

        slices_executed = twap_state.get("slices_executed", 0)
        total_executed = twap_state.get("total_executed", 0.0)
        target_amount = twap_state.get("target_amount", total_amount)

        # Check if TWAP is complete
        if slices_executed >= slice_count:
            avg_price = (
                sum(twap_state["prices"]) / len(twap_state["prices"])
                if twap_state["prices"] else current_price
            )
            logger.info(
                f"Bot {bot.id}: TWAP complete - "
                f"Executed ${total_executed:.2f}, Avg price ${avg_price:.2f}"
            )
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"TWAP: Complete ({slices_executed}/{slice_count} slices, avg ${avg_price:.2f})"
            )

        # Calculate slice timing
        slice_interval_minutes = execution_period / slice_count

        # Check timing for next slice
        last_order = await self._get_last_order(bot.id, session)

        if last_order and slices_executed > 0:
            time_since_last = (datetime.utcnow() - last_order.created_at).total_seconds() / 60
            if time_since_last < slice_interval_minutes:
                remaining = slice_interval_minutes - time_since_last
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"TWAP: Slice {slices_executed + 1}/{slice_count} in {remaining:.1f} min"
                )

        # Calculate slice amount
        remaining_slices = slice_count - slices_executed
        remaining_amount = target_amount - total_executed
        slice_amount = remaining_amount / remaining_slices

        # Validate we have enough balance for buys
        if side == "buy" and slice_amount > bot.current_balance:
            slice_amount = bot.current_balance
            if slice_amount < 1:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="TWAP: Insufficient balance for next slice"
                )

        # Execute slice
        twap_state["slices_executed"] = slices_executed + 1
        twap_state["total_executed"] = total_executed + slice_amount
        twap_state["prices"].append(current_price)
        self._save_twap_state(bot.id, twap_state)

        logger.info(
            f"Bot {bot.id}: TWAP slice {slices_executed + 1}/{slice_count} - "
            f"${slice_amount:.2f} at ${current_price:.2f}"
        )

        return TradeSignal(
            action=side,
            amount=slice_amount,
            order_type="market",
            reason=f"TWAP: Slice {slices_executed + 1}/{slice_count} at ${current_price:.2f}",
        )

    def _get_twap_state(self, bot_id: int) -> dict:
        """Get TWAP execution state for a bot."""
        if not hasattr(self, "_twap_states"):
            self._twap_states = {}
        return self._twap_states.get(bot_id, {})

    def _save_twap_state(self, bot_id: int, state: dict) -> None:
        """Save TWAP execution state for a bot."""
        if not hasattr(self, "_twap_states"):
            self._twap_states = {}
        self._twap_states[bot_id] = state

    async def _strategy_vwap(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Volume-Weighted Average Price (VWAP) strategy.

        Tracks VWAP and executes orders based on price deviation from VWAP.
        - Buys when price is significantly below VWAP (undervalued)
        - Sells when price is significantly above VWAP (overvalued)

        Parameters:
            lookback_period_minutes: Period for VWAP calculation (default: 30)
            deviation_threshold_percent: Min deviation to trigger trade (default: 0.5)
            order_size_percent: Order size as percent of budget (default: 20)
        """
        lookback_period = params.get("lookback_period_minutes", 30)
        deviation_threshold = params.get("deviation_threshold_percent", 0.5) / 100
        order_size_percent = params.get("order_size_percent", 20) / 100

        # Get VWAP state
        vwap_state = self._get_vwap_state(bot.id)

        # Initialize if needed
        if "price_volume_data" not in vwap_state:
            vwap_state["price_volume_data"] = []

        # Add current data point (simulate volume based on price volatility)
        # In production, this would use real volume data from the exchange
        simulated_volume = self._estimate_volume(vwap_state, current_price)

        vwap_state["price_volume_data"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "price": current_price,
            "volume": simulated_volume,
        })

        # Keep only data within lookback period
        cutoff = datetime.utcnow() - timedelta(minutes=lookback_period)
        vwap_state["price_volume_data"] = [
            pv for pv in vwap_state["price_volume_data"]
            if datetime.fromisoformat(pv["timestamp"]) > cutoff
        ]

        self._save_vwap_state(bot.id, vwap_state)

        # Need enough data
        if len(vwap_state["price_volume_data"]) < 5:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"VWAP: Collecting data ({len(vwap_state['price_volume_data'])}/5)"
            )

        # Calculate VWAP
        total_pv = sum(pv["price"] * pv["volume"] for pv in vwap_state["price_volume_data"])
        total_volume = sum(pv["volume"] for pv in vwap_state["price_volume_data"])

        if total_volume == 0:
            return TradeSignal(
                action="hold",
                amount=0,
                reason="VWAP: No volume data"
            )

        vwap = total_pv / total_volume

        # Calculate deviation from VWAP
        deviation = (current_price - vwap) / vwap

        logger.debug(
            f"Bot {bot.id}: VWAP - Current: ${current_price:.2f}, "
            f"VWAP: ${vwap:.2f}, Deviation: {deviation*100:.2f}%"
        )

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0
        total_position_value = sum(p.amount * p.current_price for p in positions)

        # Trading logic based on VWAP deviation
        if deviation <= -deviation_threshold:
            # Price significantly BELOW VWAP - BUY opportunity
            buy_amount = bot.current_balance * order_size_percent

            if buy_amount < 1:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="VWAP: Insufficient balance for buy"
                )

            logger.info(
                f"Bot {bot.id}: VWAP BUY - "
                f"Price ${current_price:.2f} is {abs(deviation)*100:.2f}% below VWAP ${vwap:.2f}"
            )

            return TradeSignal(
                action="buy",
                amount=buy_amount,
                order_type="market",
                reason=f"VWAP: Buy {abs(deviation)*100:.1f}% below VWAP (${vwap:.2f})",
            )

        elif deviation >= deviation_threshold and has_position:
            # Price significantly ABOVE VWAP - SELL opportunity
            sell_amount = min(total_position_value * 0.5, total_position_value)

            if sell_amount < 1:
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="VWAP: Position too small to sell"
                )

            logger.info(
                f"Bot {bot.id}: VWAP SELL - "
                f"Price ${current_price:.2f} is {deviation*100:.2f}% above VWAP ${vwap:.2f}"
            )

            return TradeSignal(
                action="sell",
                amount=sell_amount,
                order_type="market",
                reason=f"VWAP: Sell {deviation*100:.1f}% above VWAP (${vwap:.2f})",
            )

        return TradeSignal(
            action="hold",
            amount=0,
            reason=f"VWAP: ${vwap:.2f}, deviation {deviation*100:.2f}% (threshold: {deviation_threshold*100:.1f}%)"
        )

    def _get_vwap_state(self, bot_id: int) -> dict:
        """Get VWAP state for a bot."""
        if not hasattr(self, "_vwap_states"):
            self._vwap_states = {}
        return self._vwap_states.get(bot_id, {})

    def _save_vwap_state(self, bot_id: int, state: dict) -> None:
        """Save VWAP state for a bot."""
        if not hasattr(self, "_vwap_states"):
            self._vwap_states = {}
        self._vwap_states[bot_id] = state

    def _estimate_volume(self, state: dict, current_price: float) -> float:
        """Estimate volume based on price movement (simulation)."""
        # In production, use real volume data from exchange
        # Here we simulate volume based on price volatility
        data = state.get("price_volume_data", [])
        if len(data) < 2:
            return 1000.0  # Base volume

        last_price = data[-1]["price"]
        price_change = abs(current_price - last_price) / last_price if last_price > 0 else 0

        # Higher volume on larger price movements
        base_volume = 1000.0
        volatility_multiplier = 1 + (price_change * 50)  # Up to 2x on 2% move

        return base_volume * volatility_multiplier

    async def _strategy_scalping(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Scalping strategy - Conservative, tightly constrained tactical strategy.

        Designed for spot-only, long-only trading with strict risk controls.
        Captures small profits from micro-breakouts and short-term momentum.

        Behavioral constraints (mandatory):
        - No averaging down
        - No pyramiding (one position at a time)
        - Hard cooldown between trades
        - Maximum trades per hour/day limits
        - Very small position sizing

        Parameters:
            short_ema: Short EMA period for momentum (default: 5)
            long_ema: Long EMA period for momentum (default: 15)
            take_profit_percent: Profit target (default: 0.5%)
            stop_loss_percent: Stop loss (default: 0.5%, risk-reward 1:1)
            max_position_time_seconds: Time-based exit (default: 300 = 5 minutes)
            position_size_percent: Position size (default: 5% of balance, very small)
            cooldown_minutes: Cooldown between trades (default: 10 minutes)
            max_trades_per_hour: Maximum trades per hour (default: 3)
            max_trades_per_day: Maximum trades per day (default: 20)
            trend_filter_ema: Global trend filter EMA (default: 50, 0 = disabled)
        """
        # Get parameters
        short_ema = params.get("short_ema", 5)
        long_ema = params.get("long_ema", 15)
        take_profit = params.get("take_profit_percent", 0.5) / 100
        stop_loss = params.get("stop_loss_percent", 0.5) / 100
        max_position_time = params.get("max_position_time_seconds", 300)
        position_size_pct = params.get("position_size_percent", 5) / 100
        cooldown_minutes = params.get("cooldown_minutes", 10)
        max_trades_hour = params.get("max_trades_per_hour", 3)
        max_trades_day = params.get("max_trades_per_day", 20)
        trend_filter_ema = params.get("trend_filter_ema", 50)

        # Get price history
        price_history = self._get_price_history(bot.id)
        price_history.append(current_price)
        self._save_price_history(bot.id, price_history, max_len=max(trend_filter_ema + 20, 100))

        # Need minimum data for EMA calculation
        min_data = max(long_ema, trend_filter_ema if trend_filter_ema > 0 else 0)
        if len(price_history) < min_data:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Scalping: Collecting data ({len(price_history)}/{min_data})"
            )

        # Calculate EMAs
        def calculate_ema(prices: list, period: int) -> float:
            """Calculate EMA."""
            if len(prices) < period:
                return sum(prices) / len(prices)
            k = 2 / (period + 1)
            ema = sum(prices[:period]) / period
            for price in prices[period:]:
                ema = (price * k) + (ema * (1 - k))
            return ema

        ema_short = calculate_ema(price_history, short_ema)
        ema_long = calculate_ema(price_history, long_ema)

        # Global trend filter (optional)
        trend_filter_passed = True
        if trend_filter_ema > 0:
            ema_trend = calculate_ema(price_history, trend_filter_ema)
            trend_filter_passed = current_price > ema_trend

        # Initialize scalping state
        if not hasattr(self, "_scalping_states"):
            self._scalping_states = {}

        state = self._scalping_states.get(bot.id, {
            "entry_time": None,
            "entry_price": None,
            "last_trade_time": None,
            "trades_today": [],
            "trades_this_hour": [],
        })

        # Get current positions
        positions = await self._get_bot_positions(bot.id, session)
        has_position = len(positions) > 0

        now = datetime.utcnow()

        # Clean up old trade tracking (remove trades older than 1 day)
        cutoff_day = (now - timedelta(days=1)).isoformat()
        state["trades_today"] = [t for t in state["trades_today"] if t > cutoff_day]

        # Clean up trades older than 1 hour
        cutoff_hour = (now - timedelta(hours=1)).isoformat()
        state["trades_this_hour"] = [t for t in state["trades_this_hour"] if t > cutoff_hour]

        logger.debug(
            f"Bot {bot.id}: Scalping - Price: ${current_price:.2f}, "
            f"EMA({short_ema}): ${ema_short:.2f}, EMA({long_ema}): ${ema_long:.2f}, "
            f"Trades today: {len(state['trades_today'])}, this hour: {len(state['trades_this_hour'])}"
        )

        # === EXIT LOGIC (position held) ===
        if has_position:
            for pos in positions:
                # Update entry tracking if not set
                if state["entry_time"] is None:
                    state["entry_time"] = pos.created_at.isoformat() if hasattr(pos, 'created_at') else now.isoformat()
                    state["entry_price"] = pos.entry_price

                entry_time = datetime.fromisoformat(state["entry_time"])
                time_in_position = (now - entry_time).total_seconds()

                gain = (current_price - pos.entry_price) / pos.entry_price
                sell_amount = pos.amount * current_price

                # Exit 1: Take profit hit
                if gain >= take_profit:
                    logger.info(
                        f"Bot {bot.id}: Scalping EXIT (take profit) - "
                        f"Gain: {gain*100:.2f}% at ${current_price:.2f}"
                    )

                    # Track trade and reset state
                    state["trades_today"].append(now.isoformat())
                    state["trades_this_hour"].append(now.isoformat())
                    state["last_trade_time"] = now.isoformat()
                    state["entry_time"] = None
                    state["entry_price"] = None
                    self._scalping_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Scalping: Take profit {gain*100:.2f}%",
                    )

                # Exit 2: Stop loss hit
                if gain <= -stop_loss:
                    logger.info(
                        f"Bot {bot.id}: Scalping EXIT (stop loss) - "
                        f"Loss: {gain*100:.2f}% at ${current_price:.2f}"
                    )

                    # Track trade and reset state
                    state["trades_today"].append(now.isoformat())
                    state["trades_this_hour"].append(now.isoformat())
                    state["last_trade_time"] = now.isoformat()
                    state["entry_time"] = None
                    state["entry_price"] = None
                    self._scalping_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Scalping: Stop loss {gain*100:.2f}%",
                    )

                # Exit 3: Time-based exit
                if time_in_position >= max_position_time:
                    logger.info(
                        f"Bot {bot.id}: Scalping EXIT (time limit) - "
                        f"Held for {time_in_position:.0f}s, P&L: {gain*100:.2f}%"
                    )

                    # Track trade and reset state
                    state["trades_today"].append(now.isoformat())
                    state["trades_this_hour"].append(now.isoformat())
                    state["last_trade_time"] = now.isoformat()
                    state["entry_time"] = None
                    state["entry_price"] = None
                    self._scalping_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason=f"Scalping: Time exit at {time_in_position:.0f}s (P&L: {gain*100:.2f}%)",
                    )

                # Exit 4: Trend filter failure (macro risk)
                if not trend_filter_passed:
                    logger.info(
                        f"Bot {bot.id}: Scalping EXIT (trend filter) - "
                        f"Price fell below global trend"
                    )

                    state["entry_time"] = None
                    state["entry_price"] = None
                    self._scalping_states[bot.id] = state

                    return TradeSignal(
                        action="sell",
                        amount=sell_amount,
                        order_type="market",
                        reason="Scalping: Exit on trend filter failure",
                    )

            # Hold position, waiting for exit condition
            self._scalping_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Scalping: Holding, target +{take_profit*100:.2f}%, stop -{stop_loss*100:.2f}%"
            )

        # === ENTRY LOGIC (no position) ===

        # Check trade limits (hard constraints to prevent overtrading)
        if len(state["trades_today"]) >= max_trades_day:
            self._scalping_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Scalping: Daily trade limit reached ({max_trades_day})"
            )

        if len(state["trades_this_hour"]) >= max_trades_hour:
            self._scalping_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Scalping: Hourly trade limit reached ({max_trades_hour})"
            )

        # Check cooldown period
        if state["last_trade_time"] is not None:
            last_trade = datetime.fromisoformat(state["last_trade_time"])
            minutes_since_trade = (now - last_trade).total_seconds() / 60

            if minutes_since_trade < cooldown_minutes:
                self._scalping_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason=f"Scalping: Cooldown active ({cooldown_minutes - minutes_since_trade:.1f}m remaining)"
                )

        # Check trend filter
        if not trend_filter_passed:
            self._scalping_states[bot.id] = state
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Scalping: Trend filter failed, staying out"
            )

        # Entry signal: Short EMA crosses above Long EMA (micro momentum)
        # Also check that price is above short EMA (confirmation)
        if ema_short > ema_long and current_price > ema_short:
            # Calculate position size (very small, conservative)
            position_amount = bot.current_balance * position_size_pct

            if position_amount < 1:
                self._scalping_states[bot.id] = state
                return TradeSignal(
                    action="hold",
                    amount=0,
                    reason="Scalping: Insufficient balance for entry"
                )

            logger.info(
                f"Bot {bot.id}: Scalping ENTRY - "
                f"EMA({short_ema}) ${ema_short:.2f} > EMA({long_ema}) ${ema_long:.2f}, "
                f"Price: ${current_price:.2f}"
            )

            # Set entry tracking
            state["entry_time"] = now.isoformat()
            state["entry_price"] = current_price
            self._scalping_states[bot.id] = state

            return TradeSignal(
                action="buy",
                amount=position_amount,
                order_type="market",
                reason=f"Scalping: EMA cross entry (target +{take_profit*100:.2f}%)",
            )

        # No entry signal
        self._scalping_states[bot.id] = state

        if ema_short <= ema_long:
            return TradeSignal(
                action="hold",
                amount=0,
                reason=f"Scalping: Waiting for momentum (EMA({short_ema}) <= EMA({long_ema}))"
            )
        else:
            return TradeSignal(
                action="hold",
                amount=0,
                reason="Scalping: Waiting for price confirmation above EMA"
            )

    # Note: _strategy_arbitrage and _strategy_event were removed (placeholders without implementation)

    async def _strategy_auto(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Auto Mode - Factor-based strategy selection.

        Analyzes market conditions and selects the most appropriate strategy:
        - High volatility -> Mean Reversion (capture swings)
        - Strong trend -> Grid (ride the trend)
        - Low volatility -> DCA (accumulate steadily)
        - High volume -> VWAP (follow smart money)

        Parameters:
            factor_precedence: Order of factor importance (default: ["trend", "volatility", "volume"])
            disabled_factors: Factors to ignore (default: [])
            switch_threshold: Confidence threshold to switch strategy (default: 0.7)
            min_switch_interval_minutes: Minimum time between strategy switches (default: 15)
        """
        factor_precedence = params.get("factor_precedence", ["trend", "volatility", "volume"])
        disabled_factors = params.get("disabled_factors", [])
        switch_threshold = params.get("switch_threshold", 0.7)
        min_switch_interval = params.get("min_switch_interval_minutes", 15)

        # Get auto mode state
        auto_state = self._get_auto_state(bot.id)

        # Initialize state if needed
        if "current_strategy" not in auto_state:
            auto_state["current_strategy"] = "dca_accumulator"
            auto_state["last_switch_time"] = None
            auto_state["price_history"] = []
            self._save_auto_state(bot.id, auto_state)

        # Update price history
        auto_state["price_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "price": current_price,
        })

        # Keep last 100 price points
        auto_state["price_history"] = auto_state["price_history"][-100:]

        # Analyze market factors
        factors = self._analyze_market_factors(auto_state["price_history"])

        # Determine best strategy based on factors
        best_strategy, confidence = self._select_strategy_by_factors(
            factors, factor_precedence, disabled_factors
        )

        # Check if we should switch strategies
        current_strategy = auto_state["current_strategy"]
        should_switch = False

        if best_strategy != current_strategy and confidence >= switch_threshold:
            # Check minimum switch interval
            last_switch = auto_state.get("last_switch_time")
            if last_switch:
                time_since_switch = (
                    datetime.utcnow() - datetime.fromisoformat(last_switch)
                ).total_seconds() / 60
                should_switch = time_since_switch >= min_switch_interval
            else:
                should_switch = True

        if should_switch:
            logger.info(
                f"Bot {bot.id}: Auto Mode switching from {current_strategy} "
                f"to {best_strategy} (confidence: {confidence:.0%})"
            )
            auto_state["current_strategy"] = best_strategy
            auto_state["last_switch_time"] = datetime.utcnow().isoformat()
            current_strategy = best_strategy

        self._save_auto_state(bot.id, auto_state)

        # Get the strategy executor and run it
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
            signal.reason = f"[Auto:{current_strategy}] {signal.reason}"

        return signal

    def _analyze_market_factors(self, price_history: list) -> dict:
        """Analyze market factors from price history.

        Returns dict with factor scores (-1 to +1):
        - trend: positive = uptrend, negative = downtrend
        - volatility: 0 = low, 1 = high
        - volume_trend: positive = increasing, negative = decreasing
        """
        if len(price_history) < 10:
            return {"trend": 0, "volatility": 0.5, "volume_trend": 0}

        prices = [p["price"] for p in price_history]

        # Calculate trend (linear regression slope normalized)
        n = len(prices)
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n

        numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            trend = 0
        else:
            slope = numerator / denominator
            # Normalize: divide by mean price and scale
            trend = (slope / y_mean) * 100  # Percent per data point
            trend = max(-1, min(1, trend * 10))  # Scale to -1 to +1

        # Calculate volatility (normalized standard deviation)
        mean_price = sum(prices) / n
        variance = sum((p - mean_price) ** 2 for p in prices) / n
        std_dev = variance ** 0.5
        volatility = std_dev / mean_price if mean_price > 0 else 0

        # Normalize volatility to 0-1 (assuming 0-5% std dev range)
        volatility_normalized = min(1, volatility * 20)

        # Volume trend (simulated based on price changes)
        if len(prices) >= 20:
            recent_range = max(prices[-10:]) - min(prices[-10:])
            older_range = max(prices[-20:-10]) - min(prices[-20:-10])
            volume_trend = (recent_range - older_range) / older_range if older_range > 0 else 0
            volume_trend = max(-1, min(1, volume_trend * 5))
        else:
            volume_trend = 0

        return {
            "trend": trend,
            "volatility": volatility_normalized,
            "volume_trend": volume_trend,
        }

    def _select_strategy_by_factors(
        self,
        factors: dict,
        precedence: list,
        disabled: list
    ) -> tuple:
        """Select best strategy based on market factors.

        Returns (strategy_name, confidence_score)
        """
        trend = factors.get("trend", 0)
        volatility = factors.get("volatility", 0.5)
        volume_trend = factors.get("volume_trend", 0)

        # Strategy scores based on factors
        strategy_scores = {
            "dca_accumulator": 0.5,  # Base score - always reasonable
            "adaptive_grid": 0.3,
            "mean_reversion": 0.3,
            "vwap": 0.3,
        }

        # Adjust scores based on factors
        if "trend" not in disabled:
            if abs(trend) > 0.5:
                # Strong trend - favor grid trading
                strategy_scores["adaptive_grid"] += 0.4
            else:
                # No clear trend - favor DCA
                strategy_scores["dca_accumulator"] += 0.2

        if "volatility" not in disabled:
            if volatility > 0.7:
                # High volatility - favor mean reversion
                strategy_scores["mean_reversion"] += 0.5
            elif volatility < 0.3:
                # Low volatility - favor DCA
                strategy_scores["dca_accumulator"] += 0.3
            else:
                # Medium volatility - favor grid
                strategy_scores["adaptive_grid"] += 0.2

        if "volume" not in disabled:
            if abs(volume_trend) > 0.5:
                # Significant volume change - favor VWAP
                strategy_scores["vwap"] += 0.4

        # Apply precedence weighting
        for i, factor in enumerate(precedence):
            weight = 1.0 - (i * 0.2)  # 1.0, 0.8, 0.6 for top 3
            if factor == "trend" and "trend" not in disabled:
                if abs(trend) > 0.5:
                    strategy_scores["adaptive_grid"] *= (1 + weight * 0.2)
            elif factor == "volatility" and "volatility" not in disabled:
                if volatility > 0.6:
                    strategy_scores["mean_reversion"] *= (1 + weight * 0.2)
            elif factor == "volume" and "volume" not in disabled:
                if abs(volume_trend) > 0.4:
                    strategy_scores["vwap"] *= (1 + weight * 0.2)

        # Find best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        confidence = strategy_scores[best_strategy]

        # Normalize confidence to 0-1
        confidence = min(1.0, confidence / 1.5)

        return best_strategy, confidence

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
        """Execute a trade based on signal.

        Args:
            bot: The bot model
            exchange: Exchange service
            signal: Trade signal
            current_price: Current market price
            session: Database session

        Returns:
            Order if executed, None otherwise
        """
        # Calculate amount in base currency
        amount_base = signal.amount / current_price

        # Determine order side
        side = OrderSide.BUY if signal.action == "buy" else OrderSide.SELL

        # Place order
        if signal.order_type == "market":
            exchange_order = await exchange.place_market_order(
                bot.trading_pair, side, amount_base
            )
        else:
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

        # Create order record
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
        )

        if order.status == OrderStatus.FILLED:
            order.filled_at = datetime.utcnow()

        session.add(order)

        # Update wallet
        wallet = VirtualWalletService(session)
        if signal.action == "buy":
            # Buying uses quote currency
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)
        else:
            # Selling - calculate P&L
            # For simplicity, just record fees for now
            await wallet.record_trade_result(bot.id, -exchange_order.fee, 0)

        # Update running balance
        result = await session.execute(select(Bot).where(Bot.id == bot.id))
        updated_bot = result.scalar_one_or_none()
        if updated_bot:
            order.running_balance_after = updated_bot.current_balance

        # Update/create position
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

        await session.commit()

        logger.info(
            f"Bot {bot.id}: Executed {signal.action} order - "
            f"{exchange_order.amount:.6f} @ ${exchange_order.price:.2f}"
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
            "scalping",
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
