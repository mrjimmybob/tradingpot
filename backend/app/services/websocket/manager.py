"""
WebSocket manager for coordinating exchange connections and frontend broadcasts.

Provides:
- Exchange WebSocket connection management
- Market data aggregation from multiple exchanges
- Frontend WebSocket server for real-time UI updates
- Bot status and P&L broadcasting
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Set, List, Any
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .base import (
    BaseWebSocketConnector,
    DepthUpdate,
    TradeUpdate,
    TickerUpdate,
    KlineUpdate,
)
from .mexc import MEXCWebSocketConnector
from .market_data import MarketDataService, MarketIndicators

logger = logging.getLogger(__name__)


@dataclass
class BotUpdate:
    """Bot status update for frontend."""
    type: str = "bot_update"
    bot_id: int = 0
    status: str = ""
    pnl: float = 0.0
    current_balance: float = 0.0
    positions: List[dict] = None
    timestamp: str = ""

    def __post_init__(self):
        if self.positions is None:
            self.positions = []
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class PriceUpdate:
    """Price update for frontend."""
    type: str = "price_update"
    symbol: str = ""
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    change_24h: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class IndicatorUpdate:
    """Market indicator update for frontend."""
    type: str = "indicator_update"
    symbol: str = ""
    sentiment: float = 0.0
    risk: float = 0.0
    signal: str = "neutral"
    orderbook_imbalance: float = 0.0
    volume_delta: float = 0.0
    spread_percent: float = 0.0
    volatility_regime: str = "normal"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class StatsUpdate:
    """Global stats update for frontend."""
    type: str = "stats_update"
    total_bots: int = 0
    running_bots: int = 0
    total_pnl: float = 0.0
    active_trades: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class WebSocketManager:
    """
    Central manager for all WebSocket operations.

    Handles:
    - Exchange WebSocket connections (MEXC, etc.)
    - Market data processing and indicator computation
    - Frontend client connections
    - Real-time broadcasting to UI
    """

    def __init__(self):
        # Exchange connectors
        self._connectors: Dict[str, BaseWebSocketConnector] = {}

        # Market data service
        self._market_data = MarketDataService()

        # Frontend WebSocket clients
        self._clients: Set[WebSocket] = set()
        self._client_subscriptions: Dict[WebSocket, Set[str]] = {}

        # Symbol tracking
        self._tracked_symbols: Set[str] = set()

        # Latest data cache for new clients
        self._price_cache: Dict[str, PriceUpdate] = {}
        self._indicator_cache: Dict[str, IndicatorUpdate] = {}

        # Background tasks
        self._broadcast_task: Optional[asyncio.Task] = None
        self._bot_update_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start WebSocket manager."""
        if self._running:
            return

        self._running = True

        # Initialize MEXC connector
        mexc = MEXCWebSocketConnector()
        self._connectors["MEXC"] = mexc

        # Connect market data handlers
        mexc.on_depth(self._market_data.handle_depth)
        mexc.on_trade(self._market_data.handle_trade)
        mexc.on_kline(self._market_data.handle_kline)
        mexc.on_ticker(self._market_data.handle_ticker)

        # Connect price update handler for frontend
        mexc.on_trade(self._handle_trade_for_frontend)
        mexc.on_ticker(self._handle_ticker_for_frontend)

        # Connect indicator handler for frontend
        self._market_data.on_indicators(self._handle_indicators)

        # Start market data service
        await self._market_data.start()

        # Connect to exchange
        await mexc.connect()

        # Start background tasks
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        self._bot_update_task = asyncio.create_task(self._bot_status_loop())

        logger.info("WebSocketManager started")

    async def stop(self) -> None:
        """Stop WebSocket manager."""
        self._running = False

        # Cancel background tasks
        if self._broadcast_task:
            self._broadcast_task.cancel()
            self._broadcast_task = None

        if self._bot_update_task:
            self._bot_update_task.cancel()
            self._bot_update_task = None

        # Stop market data service
        await self._market_data.stop()

        # Disconnect all exchange connectors
        for connector in self._connectors.values():
            await connector.disconnect()
        self._connectors.clear()

        # Close all client connections
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

        logger.info("WebSocketManager stopped")

    async def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to market data for a symbol."""
        if symbol in self._tracked_symbols:
            return True

        # Subscribe on all connectors
        success = False
        for connector in self._connectors.values():
            if await connector.subscribe_all(symbol):
                success = True

        if success:
            self._tracked_symbols.add(symbol)
            logger.info(f"Subscribed to market data for {symbol}")

        return success

    async def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from market data for a symbol."""
        if symbol not in self._tracked_symbols:
            return True

        for connector in self._connectors.values():
            await connector.unsubscribe_depth(symbol)

        self._tracked_symbols.discard(symbol)
        return True

    # Frontend WebSocket handling
    async def connect_client(self, websocket: WebSocket) -> None:
        """Handle new frontend WebSocket connection."""
        await websocket.accept()
        self._clients.add(websocket)
        self._client_subscriptions[websocket] = set()

        # Send cached data to new client
        await self._send_cached_data(websocket)

        logger.info(f"Frontend client connected. Total clients: {len(self._clients)}")

    async def disconnect_client(self, websocket: WebSocket) -> None:
        """Handle frontend WebSocket disconnection."""
        self._clients.discard(websocket)
        self._client_subscriptions.pop(websocket, None)
        logger.info(f"Frontend client disconnected. Total clients: {len(self._clients)}")

    async def handle_client_message(
        self, websocket: WebSocket, message: str
    ) -> None:
        """Handle message from frontend client."""
        try:
            data = json.loads(message)
            action = data.get("action")

            if action == "subscribe":
                symbols = data.get("symbols", [])
                for symbol in symbols:
                    await self.subscribe_symbol(symbol)
                    self._client_subscriptions[websocket].add(symbol)

            elif action == "unsubscribe":
                symbols = data.get("symbols", [])
                for symbol in symbols:
                    self._client_subscriptions[websocket].discard(symbol)

            elif action == "ping":
                await websocket.send_json({"type": "pong"})

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def _send_cached_data(self, websocket: WebSocket) -> None:
        """Send cached price/indicator data to new client."""
        try:
            # Send all cached prices
            for price_update in self._price_cache.values():
                await websocket.send_json(asdict(price_update))

            # Send all cached indicators
            for indicator_update in self._indicator_cache.values():
                await websocket.send_json(asdict(indicator_update))

        except Exception as e:
            logger.error(f"Error sending cached data: {e}")

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self._clients:
            return

        # Get relevant symbol if present
        symbol = message.get("symbol")

        disconnected = set()
        for client in self._clients:
            try:
                # If message has a symbol, only send to subscribed clients
                if symbol:
                    subs = self._client_subscriptions.get(client, set())
                    if symbol not in subs and "*" not in subs:
                        continue

                await client.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.add(client)

        # Clean up disconnected clients
        for client in disconnected:
            await self.disconnect_client(client)

    async def broadcast_bot_update(self, update: BotUpdate) -> None:
        """Broadcast bot status update."""
        await self.broadcast(asdict(update))

    async def broadcast_stats_update(self, update: StatsUpdate) -> None:
        """Broadcast global stats update."""
        await self.broadcast(asdict(update))

    # Exchange data handlers
    async def _handle_trade_for_frontend(self, trade: TradeUpdate) -> None:
        """Handle trade update and broadcast price to frontend."""
        price_update = PriceUpdate(
            symbol=trade.symbol,
            price=trade.price,
            timestamp=trade.timestamp.isoformat(),
        )

        self._price_cache[trade.symbol] = price_update
        await self.broadcast(asdict(price_update))

    async def _handle_ticker_for_frontend(self, ticker: TickerUpdate) -> None:
        """Handle ticker update and broadcast to frontend."""
        price_update = PriceUpdate(
            symbol=ticker.symbol,
            price=ticker.last_price,
            bid=ticker.bid_price,
            ask=ticker.ask_price,
            change_24h=ticker.price_change_percent_24h,
            timestamp=ticker.timestamp.isoformat(),
        )

        self._price_cache[ticker.symbol] = price_update
        await self.broadcast(asdict(price_update))

    async def _handle_indicators(self, indicators: MarketIndicators) -> None:
        """Handle indicator update and broadcast to frontend."""
        update = IndicatorUpdate(
            symbol=indicators.symbol,
            sentiment=indicators.sentiment_score,
            risk=indicators.risk_score,
            signal=indicators.signal,
            orderbook_imbalance=(
                indicators.orderbook.imbalance_ratio
                if indicators.orderbook else 0
            ),
            volume_delta=(
                indicators.volume_delta.normalized_delta
                if indicators.volume_delta else 0
            ),
            spread_percent=(
                indicators.spread.spread_percent
                if indicators.spread else 0
            ),
            volatility_regime=(
                indicators.volatility.vol_regime
                if indicators.volatility else "normal"
            ),
            timestamp=indicators.timestamp.isoformat(),
        )

        self._indicator_cache[indicators.symbol] = update
        await self.broadcast(asdict(update))

    async def _broadcast_loop(self) -> None:
        """Background loop for periodic broadcasts."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Broadcast interval

                # Re-broadcast cached data periodically for clients that might have missed updates
                # (This is a safety net, not the primary delivery mechanism)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")

    async def _bot_status_loop(self) -> None:
        """Background loop for bot status updates."""
        while self._running:
            try:
                await asyncio.sleep(2)  # Update every 2 seconds

                # Import here to avoid circular imports
                from ...models import async_session_maker, Bot, BotStatus, Position

                async with async_session_maker() as session:
                    # Get all running bots
                    result = await session.execute(
                        select(Bot).where(Bot.status == BotStatus.RUNNING)
                    )
                    running_bots = result.scalars().all()

                    # Broadcast updates for each running bot
                    for bot in running_bots:
                        # Get positions
                        pos_result = await session.execute(
                            select(Position).where(Position.bot_id == bot.id)
                        )
                        positions = pos_result.scalars().all()

                        update = BotUpdate(
                            bot_id=bot.id,
                            status=bot.status.value,
                            pnl=bot.total_pnl,
                            current_balance=bot.current_balance,
                            positions=[
                                {
                                    "trading_pair": p.trading_pair,
                                    "side": p.side.value,
                                    "entry_price": p.entry_price,
                                    "current_price": p.current_price,
                                    "amount": p.amount,
                                    "unrealized_pnl": p.unrealized_pnl,
                                }
                                for p in positions
                            ],
                        )

                        await self.broadcast_bot_update(update)

                    # Broadcast global stats
                    total_bots = len(running_bots)
                    total_pnl = sum(b.total_pnl for b in running_bots)

                    stats = StatsUpdate(
                        total_bots=total_bots,
                        running_bots=total_bots,
                        total_pnl=total_pnl,
                        active_trades=sum(
                            len([p for p in await session.execute(
                                select(Position).where(Position.bot_id == b.id)
                            ).scalars().all() for p in [p]])
                            for b in running_bots
                        ) if running_bots else 0,
                    )

                    await self.broadcast_stats_update(stats)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bot status loop error: {e}")
                await asyncio.sleep(5)

    def get_market_indicators(self, symbol: str) -> Optional[MarketIndicators]:
        """Get current market indicators for a symbol."""
        return self._market_data.get_indicators(symbol)

    def get_all_market_indicators(self) -> Dict[str, MarketIndicators]:
        """Get market indicators for all tracked symbols."""
        return self._market_data.get_all_indicators()


# Global WebSocket manager instance
ws_manager = WebSocketManager()
