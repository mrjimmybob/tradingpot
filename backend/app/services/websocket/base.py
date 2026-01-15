"""
Base WebSocket connector for exchange-agnostic market data ingestion.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Callable, Awaitable, Any

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    DEPTH = "depth"
    TRADE = "trade"
    TICKER = "ticker"
    KLINE = "kline"
    BOOK_TICKER = "book_ticker"
    STATUS = "status"


@dataclass
class WebSocketMessage:
    """Base WebSocket message."""
    type: MessageType
    symbol: str
    timestamp: datetime
    exchange: str
    data: Dict[str, Any]


@dataclass
class DepthLevel:
    """Single orderbook level."""
    price: float
    quantity: float


@dataclass
class DepthUpdate:
    """Orderbook depth update."""
    symbol: str
    timestamp: datetime
    exchange: str
    bids: List[DepthLevel]  # Buy orders (highest first)
    asks: List[DepthLevel]  # Sell orders (lowest first)
    is_snapshot: bool = False
    version: Optional[int] = None

    @property
    def best_bid(self) -> Optional[DepthLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[DepthLevel]:
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def spread_percent(self) -> Optional[float]:
        if self.best_bid and self.best_ask and self.best_bid.price > 0:
            return (self.spread / self.best_bid.price) * 100
        return None


@dataclass
class TradeUpdate:
    """Trade execution update."""
    symbol: str
    timestamp: datetime
    exchange: str
    price: float
    quantity: float
    is_buyer_maker: bool  # True = sell aggressor, False = buy aggressor
    trade_id: Optional[str] = None

    @property
    def side(self) -> str:
        """Return 'buy' or 'sell' based on aggressor."""
        return "sell" if self.is_buyer_maker else "buy"

    @property
    def value(self) -> float:
        """Trade value in quote currency."""
        return self.price * self.quantity


@dataclass
class TickerUpdate:
    """24h ticker update."""
    symbol: str
    timestamp: datetime
    exchange: str
    last_price: float
    bid_price: float
    ask_price: float
    high_24h: float
    low_24h: float
    volume_24h: float
    price_change_24h: float
    price_change_percent_24h: float


@dataclass
class KlineUpdate:
    """Candlestick/Kline update."""
    symbol: str
    timestamp: datetime
    exchange: str
    interval: str  # "1m", "5m", "15m", "1h", etc.
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    is_closed: bool = False

    @property
    def range(self) -> float:
        """Price range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Candle body (close - open)."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open


# Type alias for message handlers
MessageHandler = Callable[[WebSocketMessage], Awaitable[None]]
DepthHandler = Callable[[DepthUpdate], Awaitable[None]]
TradeHandler = Callable[[TradeUpdate], Awaitable[None]]
TickerHandler = Callable[[TickerUpdate], Awaitable[None]]
KlineHandler = Callable[[KlineUpdate], Awaitable[None]]


@dataclass
class ConnectionState:
    """WebSocket connection state."""
    is_connected: bool = False
    is_connecting: bool = False
    reconnect_attempts: int = 0
    last_connected: Optional[datetime] = None
    last_message: Optional[datetime] = None
    subscriptions: set = field(default_factory=set)


class BaseWebSocketConnector(ABC):
    """
    Abstract base class for exchange WebSocket connectors.

    Provides exchange-agnostic interface for:
    - Connection management with auto-reconnect
    - Subscription management
    - Message parsing and routing
    - Keep-alive handling
    """

    def __init__(
        self,
        exchange_name: str,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        ping_interval: float = 30.0,
    ):
        self.exchange_name = exchange_name
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ping_interval = ping_interval

        self.state = ConnectionState()
        self._ws = None
        self._ping_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Message handlers
        self._depth_handlers: List[DepthHandler] = []
        self._trade_handlers: List[TradeHandler] = []
        self._ticker_handlers: List[TickerHandler] = []
        self._kline_handlers: List[KlineHandler] = []
        self._raw_handlers: List[MessageHandler] = []

        # Symbol subscriptions
        self._depth_subscriptions: set = set()
        self._trade_subscriptions: set = set()
        self._ticker_subscriptions: set = set()
        self._kline_subscriptions: Dict[str, set] = {}  # symbol -> intervals

    @property
    @abstractmethod
    def ws_url(self) -> str:
        """WebSocket endpoint URL."""
        pass

    @abstractmethod
    async def _send_subscribe(self, channels: List[str]) -> None:
        """Send subscription message to exchange."""
        pass

    @abstractmethod
    async def _send_unsubscribe(self, channels: List[str]) -> None:
        """Send unsubscription message to exchange."""
        pass

    @abstractmethod
    async def _send_ping(self) -> None:
        """Send keep-alive ping to exchange."""
        pass

    @abstractmethod
    def _parse_message(self, raw_message: str) -> Optional[WebSocketMessage]:
        """Parse raw message into WebSocketMessage."""
        pass

    @abstractmethod
    def _get_depth_channel(self, symbol: str) -> str:
        """Get depth/orderbook channel name for symbol."""
        pass

    @abstractmethod
    def _get_trade_channel(self, symbol: str) -> str:
        """Get trade channel name for symbol."""
        pass

    @abstractmethod
    def _get_ticker_channel(self, symbol: str) -> str:
        """Get ticker channel name for symbol."""
        pass

    @abstractmethod
    def _get_kline_channel(self, symbol: str, interval: str) -> str:
        """Get kline channel name for symbol and interval."""
        pass

    # Handler registration
    def on_depth(self, handler: DepthHandler) -> None:
        """Register depth update handler."""
        self._depth_handlers.append(handler)

    def on_trade(self, handler: TradeHandler) -> None:
        """Register trade update handler."""
        self._trade_handlers.append(handler)

    def on_ticker(self, handler: TickerHandler) -> None:
        """Register ticker update handler."""
        self._ticker_handlers.append(handler)

    def on_kline(self, handler: KlineHandler) -> None:
        """Register kline update handler."""
        self._kline_handlers.append(handler)

    def on_message(self, handler: MessageHandler) -> None:
        """Register raw message handler."""
        self._raw_handlers.append(handler)

    # Subscription management
    async def subscribe_depth(self, symbol: str) -> bool:
        """Subscribe to orderbook depth updates."""
        if symbol in self._depth_subscriptions:
            return True

        channel = self._get_depth_channel(symbol)
        try:
            await self._send_subscribe([channel])
            self._depth_subscriptions.add(symbol)
            self.state.subscriptions.add(channel)
            logger.info(f"{self.exchange_name}: Subscribed to depth for {symbol}")
            return True
        except Exception as e:
            logger.error(f"{self.exchange_name}: Failed to subscribe depth {symbol}: {e}")
            return False

    async def subscribe_trades(self, symbol: str) -> bool:
        """Subscribe to trade updates."""
        if symbol in self._trade_subscriptions:
            return True

        channel = self._get_trade_channel(symbol)
        try:
            await self._send_subscribe([channel])
            self._trade_subscriptions.add(symbol)
            self.state.subscriptions.add(channel)
            logger.info(f"{self.exchange_name}: Subscribed to trades for {symbol}")
            return True
        except Exception as e:
            logger.error(f"{self.exchange_name}: Failed to subscribe trades {symbol}: {e}")
            return False

    async def subscribe_ticker(self, symbol: str) -> bool:
        """Subscribe to ticker updates."""
        if symbol in self._ticker_subscriptions:
            return True

        channel = self._get_ticker_channel(symbol)
        try:
            await self._send_subscribe([channel])
            self._ticker_subscriptions.add(symbol)
            self.state.subscriptions.add(channel)
            logger.info(f"{self.exchange_name}: Subscribed to ticker for {symbol}")
            return True
        except Exception as e:
            logger.error(f"{self.exchange_name}: Failed to subscribe ticker {symbol}: {e}")
            return False

    async def subscribe_kline(self, symbol: str, interval: str = "1m") -> bool:
        """Subscribe to kline/candlestick updates."""
        if symbol not in self._kline_subscriptions:
            self._kline_subscriptions[symbol] = set()

        if interval in self._kline_subscriptions[symbol]:
            return True

        channel = self._get_kline_channel(symbol, interval)
        try:
            await self._send_subscribe([channel])
            self._kline_subscriptions[symbol].add(interval)
            self.state.subscriptions.add(channel)
            logger.info(f"{self.exchange_name}: Subscribed to kline {interval} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"{self.exchange_name}: Failed to subscribe kline {symbol}: {e}")
            return False

    async def subscribe_all(self, symbol: str, kline_intervals: List[str] = None) -> bool:
        """Subscribe to all market data for a symbol."""
        kline_intervals = kline_intervals or ["1m", "5m", "15m"]

        results = await asyncio.gather(
            self.subscribe_depth(symbol),
            self.subscribe_trades(symbol),
            self.subscribe_ticker(symbol),
            *[self.subscribe_kline(symbol, interval) for interval in kline_intervals],
            return_exceptions=True,
        )

        return all(r is True for r in results)

    async def unsubscribe_depth(self, symbol: str) -> bool:
        """Unsubscribe from orderbook depth updates."""
        if symbol not in self._depth_subscriptions:
            return True

        channel = self._get_depth_channel(symbol)
        try:
            await self._send_unsubscribe([channel])
            self._depth_subscriptions.discard(symbol)
            self.state.subscriptions.discard(channel)
            return True
        except Exception as e:
            logger.error(f"{self.exchange_name}: Failed to unsubscribe depth {symbol}: {e}")
            return False

    # Connection management
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        if self.state.is_connected:
            return True

        if self.state.is_connecting:
            # Wait for existing connection attempt
            for _ in range(50):  # 5 seconds max
                await asyncio.sleep(0.1)
                if self.state.is_connected:
                    return True
            return False

        self.state.is_connecting = True

        try:
            import websockets
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=None,  # We handle ping ourselves
                ping_timeout=None,
                close_timeout=10,
            )

            self.state.is_connected = True
            self.state.is_connecting = False
            self.state.last_connected = datetime.utcnow()
            self.state.reconnect_attempts = 0

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())

            logger.info(f"{self.exchange_name}: WebSocket connected to {self.ws_url}")
            return True

        except Exception as e:
            self.state.is_connecting = False
            logger.error(f"{self.exchange_name}: Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self.state.is_connected = False

        # Cancel background tasks
        if self._ping_task:
            self._ping_task.cancel()
            self._ping_task = None

        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info(f"{self.exchange_name}: WebSocket disconnected")

    async def _receive_loop(self) -> None:
        """Main message receive loop."""
        while self.state.is_connected and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self.ping_interval * 2,
                )
                self.state.last_message = datetime.utcnow()

                # Parse and route message
                await self._handle_message(message)

            except asyncio.TimeoutError:
                logger.warning(f"{self.exchange_name}: No message received, checking connection")
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.exchange_name}: Receive error: {e}")
                self.state.is_connected = False
                break

        # Trigger reconnect if disconnected unexpectedly
        if not self.state.is_connected and self.state.reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _ping_loop(self) -> None:
        """Keep-alive ping loop."""
        while self.state.is_connected:
            try:
                await asyncio.sleep(self.ping_interval)
                if self.state.is_connected:
                    await self._send_ping()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"{self.exchange_name}: Ping error: {e}")

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        while self.state.reconnect_attempts < self.max_reconnect_attempts:
            self.state.reconnect_attempts += 1
            delay = min(
                self.reconnect_delay * (2 ** (self.state.reconnect_attempts - 1)),
                self.max_reconnect_delay,
            )

            logger.info(
                f"{self.exchange_name}: Reconnect attempt {self.state.reconnect_attempts}"
                f"/{self.max_reconnect_attempts} in {delay:.1f}s"
            )

            await asyncio.sleep(delay)

            if await self.connect():
                # Resubscribe to all channels
                await self._resubscribe()
                return

        logger.error(f"{self.exchange_name}: Max reconnect attempts reached")

    async def _resubscribe(self) -> None:
        """Resubscribe to all previous subscriptions after reconnect."""
        channels = list(self.state.subscriptions)
        if channels:
            try:
                await self._send_subscribe(channels)
                logger.info(f"{self.exchange_name}: Resubscribed to {len(channels)} channels")
            except Exception as e:
                logger.error(f"{self.exchange_name}: Resubscribe failed: {e}")

    async def _handle_message(self, raw_message: str) -> None:
        """Parse and route incoming message to appropriate handlers."""
        try:
            parsed = self._parse_message(raw_message)
            if not parsed:
                return

            # Route to type-specific handlers
            if parsed.type == MessageType.DEPTH:
                depth = self._parse_depth_update(parsed)
                if depth:
                    for handler in self._depth_handlers:
                        await handler(depth)

            elif parsed.type == MessageType.TRADE:
                trade = self._parse_trade_update(parsed)
                if trade:
                    for handler in self._trade_handlers:
                        await handler(trade)

            elif parsed.type == MessageType.TICKER:
                ticker = self._parse_ticker_update(parsed)
                if ticker:
                    for handler in self._ticker_handlers:
                        await handler(ticker)

            elif parsed.type == MessageType.KLINE:
                kline = self._parse_kline_update(parsed)
                if kline:
                    for handler in self._kline_handlers:
                        await handler(kline)

            # Also send to raw handlers
            for handler in self._raw_handlers:
                await handler(parsed)

        except Exception as e:
            logger.error(f"{self.exchange_name}: Error handling message: {e}")

    def _parse_depth_update(self, msg: WebSocketMessage) -> Optional[DepthUpdate]:
        """Parse WebSocketMessage into DepthUpdate. Override if needed."""
        data = msg.data
        return DepthUpdate(
            symbol=msg.symbol,
            timestamp=msg.timestamp,
            exchange=msg.exchange,
            bids=[DepthLevel(b["price"], b["quantity"]) for b in data.get("bids", [])],
            asks=[DepthLevel(a["price"], a["quantity"]) for a in data.get("asks", [])],
            is_snapshot=data.get("is_snapshot", False),
            version=data.get("version"),
        )

    def _parse_trade_update(self, msg: WebSocketMessage) -> Optional[TradeUpdate]:
        """Parse WebSocketMessage into TradeUpdate. Override if needed."""
        data = msg.data
        return TradeUpdate(
            symbol=msg.symbol,
            timestamp=msg.timestamp,
            exchange=msg.exchange,
            price=data["price"],
            quantity=data["quantity"],
            is_buyer_maker=data.get("is_buyer_maker", False),
            trade_id=data.get("trade_id"),
        )

    def _parse_ticker_update(self, msg: WebSocketMessage) -> Optional[TickerUpdate]:
        """Parse WebSocketMessage into TickerUpdate. Override if needed."""
        data = msg.data
        return TickerUpdate(
            symbol=msg.symbol,
            timestamp=msg.timestamp,
            exchange=msg.exchange,
            last_price=data["last_price"],
            bid_price=data.get("bid_price", 0),
            ask_price=data.get("ask_price", 0),
            high_24h=data.get("high_24h", 0),
            low_24h=data.get("low_24h", 0),
            volume_24h=data.get("volume_24h", 0),
            price_change_24h=data.get("price_change_24h", 0),
            price_change_percent_24h=data.get("price_change_percent_24h", 0),
        )

    def _parse_kline_update(self, msg: WebSocketMessage) -> Optional[KlineUpdate]:
        """Parse WebSocketMessage into KlineUpdate. Override if needed."""
        data = msg.data
        return KlineUpdate(
            symbol=msg.symbol,
            timestamp=msg.timestamp,
            exchange=msg.exchange,
            interval=data["interval"],
            open_time=data["open_time"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            close_time=data["close_time"],
            is_closed=data.get("is_closed", False),
        )
