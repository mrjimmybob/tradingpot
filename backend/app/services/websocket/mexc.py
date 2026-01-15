"""
MEXC WebSocket connector for real-time market data.

MEXC WebSocket API Documentation:
- Endpoint: wss://wbs.mexc.com/ws
- Max subscriptions per connection: 30
- Ping interval: 30 seconds
- Auto-disconnect on idle: 30 seconds (no sub) / 60 seconds (no data)
"""
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from .base import (
    BaseWebSocketConnector,
    WebSocketMessage,
    MessageType,
    DepthUpdate,
    TradeUpdate,
    DepthLevel,
)

logger = logging.getLogger(__name__)


class MEXCWebSocketConnector(BaseWebSocketConnector):
    """
    MEXC exchange WebSocket connector.

    Supports:
    - Aggregated depth (orderbook)
    - Trade stream (aggre.deals)
    - Book ticker (best bid/ask)
    - Kline/candlestick streams
    """

    WS_URL = "wss://wbs.mexc.com/ws"
    MAX_SUBSCRIPTIONS = 30

    # Channel format templates
    DEPTH_CHANNEL = "spot@public.limit.depth.v3.api@{symbol}@20"  # 20 levels
    DEPTH_INCREMENTAL_CHANNEL = "spot@public.increase.depth.v3.api@{symbol}"
    TRADE_CHANNEL = "spot@public.deals.v3.api@{symbol}"
    BOOK_TICKER_CHANNEL = "spot@public.bookTicker.v3.api@{symbol}"
    KLINE_CHANNEL = "spot@public.kline.v3.api@{symbol}@{interval}"

    # Interval mapping (MEXC format)
    INTERVAL_MAP = {
        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "30m": "Min30",
        "1h": "Min60",
        "4h": "Hour4",
        "1d": "Day1",
        "1w": "Week1",
        "1M": "Month1",
    }

    def __init__(self):
        super().__init__(
            exchange_name="MEXC",
            max_reconnect_attempts=10,
            reconnect_delay=1.0,
            max_reconnect_delay=60.0,
            ping_interval=25.0,  # MEXC disconnects after 30s without ping
        )
        self._subscription_count = 0

    @property
    def ws_url(self) -> str:
        return self.WS_URL

    async def _send_subscribe(self, channels: List[str]) -> None:
        """Send subscription to MEXC."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        # Check subscription limit
        new_count = self._subscription_count + len(channels)
        if new_count > self.MAX_SUBSCRIPTIONS:
            raise ValueError(
                f"Subscription limit exceeded: {new_count} > {self.MAX_SUBSCRIPTIONS}"
            )

        message = {
            "method": "SUBSCRIPTION",
            "params": channels,
        }

        await self._ws.send(json.dumps(message))
        self._subscription_count = new_count
        logger.debug(f"MEXC: Sent subscription for {len(channels)} channels")

    async def _send_unsubscribe(self, channels: List[str]) -> None:
        """Send unsubscription to MEXC."""
        if not self._ws:
            return

        message = {
            "method": "UNSUBSCRIPTION",
            "params": channels,
        }

        await self._ws.send(json.dumps(message))
        self._subscription_count = max(0, self._subscription_count - len(channels))

    async def _send_ping(self) -> None:
        """Send keep-alive ping to MEXC."""
        if not self._ws:
            return

        await self._ws.send(json.dumps({"method": "PING"}))
        logger.debug("MEXC: Sent PING")

    def _get_depth_channel(self, symbol: str) -> str:
        """Get depth channel for symbol."""
        return self.DEPTH_CHANNEL.format(symbol=symbol.upper())

    def _get_trade_channel(self, symbol: str) -> str:
        """Get trade channel for symbol."""
        return self.TRADE_CHANNEL.format(symbol=symbol.upper())

    def _get_ticker_channel(self, symbol: str) -> str:
        """Get book ticker channel for symbol."""
        return self.BOOK_TICKER_CHANNEL.format(symbol=symbol.upper())

    def _get_kline_channel(self, symbol: str, interval: str) -> str:
        """Get kline channel for symbol and interval."""
        mexc_interval = self.INTERVAL_MAP.get(interval, "Min1")
        return self.KLINE_CHANNEL.format(
            symbol=symbol.upper(),
            interval=mexc_interval,
        )

    def _parse_message(self, raw_message: str) -> Optional[WebSocketMessage]:
        """Parse MEXC WebSocket message."""
        try:
            data = json.loads(raw_message)

            # Handle ping/pong responses
            if "msg" in data and data.get("msg") == "PONG":
                logger.debug("MEXC: Received PONG")
                return None

            # Handle subscription responses
            if "code" in data and "msg" in data:
                if data["code"] == 0:
                    logger.debug(f"MEXC: Subscription confirmed: {data['msg']}")
                else:
                    logger.warning(f"MEXC: Subscription error: {data}")
                return None

            # Handle data messages
            channel = data.get("c") or data.get("channel", "")
            symbol = data.get("s") or data.get("symbol", "")

            # Determine message type from channel
            if "depth" in channel.lower():
                return self._parse_depth_message(data, symbol)
            elif "deals" in channel.lower():
                return self._parse_trade_message(data, symbol)
            elif "bookticker" in channel.lower():
                return self._parse_book_ticker_message(data, symbol)
            elif "kline" in channel.lower():
                return self._parse_kline_message(data, symbol)

            return None

        except json.JSONDecodeError:
            logger.warning(f"MEXC: Invalid JSON: {raw_message[:100]}")
            return None
        except Exception as e:
            logger.error(f"MEXC: Parse error: {e}")
            return None

    def _parse_depth_message(
        self, data: Dict[str, Any], symbol: str
    ) -> Optional[WebSocketMessage]:
        """Parse depth/orderbook message from MEXC."""
        try:
            # MEXC depth format
            depth_data = data.get("d", {})
            bids_raw = depth_data.get("bids", [])
            asks_raw = depth_data.get("asks", [])

            # Parse price levels [price, quantity]
            if not bids_raw:
                bids = []
            elif isinstance(bids_raw[0], dict):
                bids = [
                    {"price": float(b["p"]), "quantity": float(b["v"])}
                    for b in bids_raw
                ]
            else:
                bids = [
                    {"price": float(b[0]), "quantity": float(b[1])}
                    for b in bids_raw
                ]

            if not asks_raw:
                asks = []
            elif isinstance(asks_raw[0], dict):
                asks = [
                    {"price": float(a["p"]), "quantity": float(a["v"])}
                    for a in asks_raw
                ]
            else:
                asks = [
                    {"price": float(a[0]), "quantity": float(a[1])}
                    for a in asks_raw
                ]

            timestamp = datetime.fromtimestamp(
                data.get("t", 0) / 1000
            ) if data.get("t") else datetime.utcnow()

            return WebSocketMessage(
                type=MessageType.DEPTH,
                symbol=symbol,
                timestamp=timestamp,
                exchange="MEXC",
                data={
                    "bids": bids,
                    "asks": asks,
                    "is_snapshot": True,
                    "version": depth_data.get("r"),
                },
            )
        except Exception as e:
            logger.error(f"MEXC: Depth parse error: {e}")
            return None

    def _parse_trade_message(
        self, data: Dict[str, Any], symbol: str
    ) -> Optional[WebSocketMessage]:
        """Parse trade message from MEXC."""
        try:
            deals = data.get("d", {}).get("deals", [])
            if not deals:
                return None

            # Take the most recent trade
            trade = deals[-1]

            timestamp = datetime.fromtimestamp(
                trade.get("t", 0) / 1000
            ) if trade.get("t") else datetime.utcnow()

            # MEXC trade type: 1 = buy, 2 = sell (maker side)
            trade_type = trade.get("S", 1)
            is_buyer_maker = trade_type == 2  # Sell aggressor = buyer is maker

            return WebSocketMessage(
                type=MessageType.TRADE,
                symbol=symbol,
                timestamp=timestamp,
                exchange="MEXC",
                data={
                    "price": float(trade.get("p", 0)),
                    "quantity": float(trade.get("v", 0)),
                    "is_buyer_maker": is_buyer_maker,
                    "trade_id": str(trade.get("t", "")),
                },
            )
        except Exception as e:
            logger.error(f"MEXC: Trade parse error: {e}")
            return None

    def _parse_book_ticker_message(
        self, data: Dict[str, Any], symbol: str
    ) -> Optional[WebSocketMessage]:
        """Parse book ticker (best bid/ask) message from MEXC."""
        try:
            ticker_data = data.get("d", {})

            timestamp = datetime.fromtimestamp(
                data.get("t", 0) / 1000
            ) if data.get("t") else datetime.utcnow()

            return WebSocketMessage(
                type=MessageType.TICKER,
                symbol=symbol,
                timestamp=timestamp,
                exchange="MEXC",
                data={
                    "last_price": float(ticker_data.get("c", 0)),  # Last price if available
                    "bid_price": float(ticker_data.get("b", 0)),  # Best bid
                    "ask_price": float(ticker_data.get("a", 0)),  # Best ask
                    "bid_qty": float(ticker_data.get("B", 0)),  # Best bid qty
                    "ask_qty": float(ticker_data.get("A", 0)),  # Best ask qty
                    "high_24h": 0,
                    "low_24h": 0,
                    "volume_24h": 0,
                    "price_change_24h": 0,
                    "price_change_percent_24h": 0,
                },
            )
        except Exception as e:
            logger.error(f"MEXC: Book ticker parse error: {e}")
            return None

    def _parse_kline_message(
        self, data: Dict[str, Any], symbol: str
    ) -> Optional[WebSocketMessage]:
        """Parse kline/candlestick message from MEXC."""
        try:
            kline_data = data.get("d", {}).get("k", {})
            if not kline_data:
                return None

            # Get interval from channel
            channel = data.get("c", "")
            interval = "1m"  # Default
            for k, v in self.INTERVAL_MAP.items():
                if v in channel:
                    interval = k
                    break

            timestamp = datetime.fromtimestamp(
                kline_data.get("t", 0) / 1000
            ) if kline_data.get("t") else datetime.utcnow()

            close_time = datetime.fromtimestamp(
                kline_data.get("T", 0) / 1000
            ) if kline_data.get("T") else timestamp

            return WebSocketMessage(
                type=MessageType.KLINE,
                symbol=symbol,
                timestamp=timestamp,
                exchange="MEXC",
                data={
                    "interval": interval,
                    "open_time": timestamp,
                    "open": float(kline_data.get("o", 0)),
                    "high": float(kline_data.get("h", 0)),
                    "low": float(kline_data.get("l", 0)),
                    "close": float(kline_data.get("c", 0)),
                    "volume": float(kline_data.get("v", 0)),
                    "close_time": close_time,
                    "is_closed": kline_data.get("x", False),
                },
            )
        except Exception as e:
            logger.error(f"MEXC: Kline parse error: {e}")
            return None

    def _parse_depth_update(self, msg: WebSocketMessage) -> Optional[DepthUpdate]:
        """Parse WebSocketMessage into DepthUpdate."""
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
