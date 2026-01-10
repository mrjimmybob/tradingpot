"""Trading engine for bot execution and management."""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import (
    Bot, BotStatus, Order, OrderType, OrderStatus,
    Position, PositionSide, PnLSnapshot,
    async_session_maker,
)
from .exchange import ExchangeService, SimulatedExchangeService, OrderSide
from .virtual_wallet import VirtualWalletService
from .risk_management import RiskManagementService, RiskAction

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
            "breakdown_momentum": self._strategy_momentum,
            "twap": self._strategy_twap,
            "vwap": self._strategy_vwap,
            "scalping": self._strategy_scalping,
            "arbitrage": self._strategy_arbitrage,
            "event_filler": self._strategy_event,
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
        """
        interval_minutes = params.get("interval_minutes", 60)
        amount_percent = params.get("amount_percent", 10) / 100

        # Check if it's time to buy
        last_order = await self._get_last_order(bot.id, session)

        if last_order:
            time_since_last = datetime.utcnow() - last_order.created_at
            if time_since_last.total_seconds() < interval_minutes * 60:
                return TradeSignal(action="hold", amount=0, reason="Waiting for interval")

        # Calculate buy amount
        buy_amount = bot.current_balance * amount_percent

        if buy_amount < 1:  # Minimum $1
            return TradeSignal(action="hold", amount=0, reason="Insufficient balance")

        return TradeSignal(
            action="buy",
            amount=buy_amount,
            order_type="market",
            reason=f"DCA buy at ${current_price:.2f}",
        )

    async def _strategy_grid(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Adaptive grid trading strategy."""
        grid_count = params.get("grid_count", 10)
        grid_spacing = params.get("grid_spacing_percent", 1.0) / 100
        range_percent = params.get("range_percent", 10) / 100

        # Calculate grid levels around current price
        upper = current_price * (1 + range_percent / 2)
        lower = current_price * (1 - range_percent / 2)

        # Simplified: alternate between buy and sell based on position
        positions = await self._get_bot_positions(bot.id, session)

        if not positions:
            # No position - buy
            amount = bot.current_balance * 0.1
            return TradeSignal(
                action="buy",
                amount=amount,
                order_type="market",
                reason="Grid: Initial buy",
            )

        # Check if price moved enough for a trade
        for pos in positions:
            price_change = (current_price - pos.entry_price) / pos.entry_price
            if price_change >= grid_spacing:
                # Sell portion
                sell_amount = pos.amount * pos.entry_price * 0.5
                return TradeSignal(
                    action="sell",
                    amount=sell_amount,
                    order_type="market",
                    reason=f"Grid: Sell on {price_change*100:.2f}% gain",
                )
            elif price_change <= -grid_spacing:
                # Buy more
                buy_amount = bot.current_balance * 0.1
                return TradeSignal(
                    action="buy",
                    amount=buy_amount,
                    order_type="market",
                    reason=f"Grid: Buy on {price_change*100:.2f}% dip",
                )

        return TradeSignal(action="hold", amount=0, reason="Grid: Waiting for price movement")

    async def _strategy_mean_reversion(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Mean reversion strategy with Bollinger bands."""
        # Simplified implementation
        period = params.get("bollinger_period", 20)
        std_mult = params.get("bollinger_std", 2.0)

        # For now, use a simple approach based on price history
        # In production, would calculate actual Bollinger bands
        return TradeSignal(action="hold", amount=0, reason="Mean reversion: Calculating bands")

    async def _strategy_momentum(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Breakdown momentum strategy."""
        breakout_threshold = params.get("breakout_threshold_percent", 2.0) / 100

        # Get recent price history to detect breakout
        # Simplified implementation
        return TradeSignal(action="hold", amount=0, reason="Momentum: Watching for breakout")

    async def _strategy_twap(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Time-weighted average price execution."""
        execution_period = params.get("execution_period_minutes", 60)
        slice_count = params.get("slice_count", 10)

        # Calculate slice amount
        slice_amount = bot.current_balance / slice_count

        # Check timing
        last_order = await self._get_last_order(bot.id, session)
        slice_interval = execution_period / slice_count

        if last_order:
            time_since_last = (datetime.utcnow() - last_order.created_at).total_seconds() / 60
            if time_since_last < slice_interval:
                return TradeSignal(action="hold", amount=0, reason="TWAP: Waiting for slice interval")

        return TradeSignal(
            action="buy",
            amount=slice_amount,
            order_type="market",
            reason=f"TWAP: Slice buy at ${current_price:.2f}",
        )

    async def _strategy_vwap(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Volume-weighted average price strategy."""
        return TradeSignal(action="hold", amount=0, reason="VWAP: Calculating target")

    async def _strategy_scalping(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Scalping strategy for quick profits."""
        take_profit = params.get("take_profit_percent", 0.5) / 100

        positions = await self._get_bot_positions(bot.id, session)

        if not positions:
            # Enter position
            amount = bot.current_balance * 0.2
            return TradeSignal(
                action="buy",
                amount=amount,
                order_type="market",
                reason="Scalp: Entry",
            )

        # Check for take profit
        for pos in positions:
            gain = (current_price - pos.entry_price) / pos.entry_price
            if gain >= take_profit:
                return TradeSignal(
                    action="sell",
                    amount=pos.amount * pos.entry_price,
                    order_type="market",
                    reason=f"Scalp: Take profit at {gain*100:.2f}%",
                )

        return TradeSignal(action="hold", amount=0, reason="Scalp: Waiting for target")

    async def _strategy_arbitrage(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Arbitrage strategy."""
        return TradeSignal(action="hold", amount=0, reason="Arbitrage: Scanning for opportunities")

    async def _strategy_event(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Event-driven strategy."""
        return TradeSignal(action="hold", amount=0, reason="Event: Monitoring")

    async def _strategy_auto(
        self,
        bot: Bot,
        current_price: float,
        params: dict,
        session: AsyncSession,
    ) -> Optional[TradeSignal]:
        """Auto mode - selects strategy based on market conditions."""
        # Default to DCA for simplicity
        return await self._strategy_dca(bot, current_price, params, session)

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

        # Calculate realized P&L
        sell_amount = min(amount, position.amount)
        pnl = (price - position.entry_price) * sell_amount

        # Record P&L
        await wallet.record_trade_result(bot_id, pnl, 0)

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
            "breakdown_momentum",
            "twap",
            "scalping",
        ]

        try:
            idx = strategies.index(current_strategy)
            return strategies[(idx + 1) % len(strategies)]
        except ValueError:
            return strategies[0]


# Global trading engine instance
trading_engine = TradingEngine()
