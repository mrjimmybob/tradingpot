"""Exchange service for interacting with crypto exchanges via ccxt."""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import ccxt.async_support as ccxt
import yaml

logger = logging.getLogger(__name__)


def _clean_credential(value: Optional[str]) -> str:
    """Normalize a credential value, treating placeholders as unset.

    Placeholder values like "YOUR_MEXC_API_KEY" must never be sent to the
    exchange as if they were real credentials.
    """
    value = (value or "").strip()
    if value.upper().startswith("YOUR_"):
        return ""
    return value


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class ExchangeOrder:
    """Exchange order result."""
    id: str
    symbol: str
    side: str
    type: str
    amount: float
    price: float
    cost: float
    fee: float
    fee_currency: str
    status: str
    timestamp: datetime
    filled: float
    remaining: float


@dataclass
class Balance:
    """Account balance for a currency."""
    currency: str
    free: float
    used: float
    total: float


@dataclass
class Ticker:
    """Market ticker data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime


class ExchangeService:
    """Service for interacting with crypto exchanges."""

    def __init__(self, exchange_id: str = "mexc", config_path: str = "config/exchanges.yaml"):
        """Initialize the exchange service.

        Args:
            exchange_id: The exchange identifier (e.g., 'mexc', 'binance')
            config_path: Path to the exchanges configuration file
        """
        self.exchange_id = exchange_id
        self.config_path = config_path
        self.exchange: Optional[ccxt.Exchange] = None
        self._config: Dict[str, Any] = {}
        self._connected = False
        self._retry_count = 3
        self._retry_delay = 1.0
        self._rate_limit_remaining = 1200
        self._last_request_time = 0

    async def connect(self) -> bool:
        """Connect to the exchange.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            # Load configuration
            self._config = self._load_config()

            if not self._config:
                logger.warning(f"No configuration found for {self.exchange_id}, using defaults")
                self._config = {
                    "api_key": "",
                    "api_secret": "",
                    "sandbox": False,
                    "retry_count": 3,
                    "retry_delay": 1.0,
                }

            self._retry_count = self._config.get("retry_count", 3)
            self._retry_delay = self._config.get("retry_delay", 1.0)

            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id, None)
            if not exchange_class:
                logger.error(f"Exchange {self.exchange_id} not supported by ccxt")
                return False

            self.exchange = exchange_class({
                "apiKey": self._config.get("api_key", ""),
                "secret": self._config.get("api_secret", ""),
                "sandbox": self._config.get("sandbox", False),
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                }
            })

            # Test connection by loading markets
            await self._execute_with_retry(self.exchange.load_markets)
            self._connected = True
            logger.info(f"Connected to {self.exchange_id} exchange")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
            self._connected = False
            logger.info(f"Disconnected from {self.exchange_id}")

    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected and self.exchange is not None

    async def reconnect(self) -> bool:
        """Reconnect to the exchange."""
        await self.disconnect()
        return await self.connect()

    def _load_config(self) -> Dict[str, Any]:
        """Load exchange configuration from YAML file.

        Credentials are resolved as: environment variable
        (<EXCHANGE_ID>_API_KEY / <EXCHANGE_ID>_API_SECRET) -> YAML value -> unset.
        Placeholder values (starting with "YOUR_") are treated as unset.
        """
        config: Dict[str, Any] = {}
        try:
            with open(self.config_path, "r") as f:
                loaded = yaml.safe_load(f) or {}
                config = loaded.get(self.exchange_id, {}) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

        env_prefix = self.exchange_id.upper()
        config["api_key"] = _clean_credential(
            os.environ.get(f"{env_prefix}_API_KEY") or config.get("api_key")
        )
        config["api_secret"] = _clean_credential(
            os.environ.get(f"{env_prefix}_API_SECRET") or config.get("api_secret")
        )
        return config

    def has_credentials(self) -> bool:
        """Whether usable (non-placeholder) API credentials are configured."""
        config = self._config or self._load_config()
        return bool(config.get("api_key")) and bool(config.get("api_secret"))

    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function

        Raises:
            The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self._retry_count):
            try:
                return await func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit exceeded, waiting... (attempt {attempt + 1})")
                await asyncio.sleep(self._retry_delay * (attempt + 1) * 2)
                last_exception = e
            except ccxt.NetworkError as e:
                logger.warning(f"Network error, retrying... (attempt {attempt + 1}): {e}")
                await asyncio.sleep(self._retry_delay * (attempt + 1))
                last_exception = e
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        raise last_exception

    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get current ticker for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')

        Returns:
            Ticker data or None if failed
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return None

        try:
            ticker = await self._execute_with_retry(
                self.exchange.fetch_ticker, symbol
            )
            return Ticker(
                symbol=ticker["symbol"],
                bid=ticker.get("bid", 0) or 0,
                ask=ticker.get("ask", 0) or 0,
                last=ticker.get("last", 0) or 0,
                volume=ticker.get("baseVolume", 0) or 0,
                timestamp=datetime.fromtimestamp(ticker["timestamp"] / 1000) if ticker.get("timestamp") else datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    async def get_balance(self, currency: str = "USDT") -> Optional[Balance]:
        """Get account balance for a currency.

        Args:
            currency: The currency to check balance for

        Returns:
            Balance data or None if failed
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return None

        try:
            balance = await self._execute_with_retry(self.exchange.fetch_balance)
            currency_balance = balance.get(currency, {})
            return Balance(
                currency=currency,
                free=currency_balance.get("free", 0) or 0,
                used=currency_balance.get("used", 0) or 0,
                total=currency_balance.get("total", 0) or 0,
            )
        except Exception as e:
            logger.error(f"Failed to get balance for {currency}: {e}")
            return None

    async def get_all_balances(self) -> Dict[str, Balance]:
        """Get all non-zero balances.

        Returns:
            Dictionary of currency to Balance
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return {}

        try:
            balance = await self._execute_with_retry(self.exchange.fetch_balance)
            result = {}
            for currency, data in balance.get("total", {}).items():
                if data and data > 0:
                    result[currency] = Balance(
                        currency=currency,
                        free=balance.get(currency, {}).get("free", 0) or 0,
                        used=balance.get(currency, {}).get("used", 0) or 0,
                        total=data,
                    )
            return result
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return {}

    def _preflight_order(
        self,
        symbol: str,
        amount: float,
        price: float,
    ) -> Optional[float]:
        """Validate and normalize a live order against exchange rules.

        Rounds the amount to the market's precision and checks the market's
        minimum amount and minimum cost limits, so invalid orders are
        rejected locally instead of bouncing off the exchange.

        Args:
            symbol: Trading pair symbol
            amount: Requested amount in base currency
            price: Price used to estimate notional cost (limit or last)

        Returns:
            The precision-adjusted amount, or None if the order violates
            exchange limits.
        """
        # Best-effort local validation: only enforce limits the exchange
        # metadata actually provides (the exchange remains the final
        # validator). Malformed/missing metadata never blocks an order.
        markets = self.exchange.markets if isinstance(self.exchange.markets, dict) else {}
        market = markets.get(symbol)

        # Round to exchange precision (ccxt returns a string)
        try:
            amount = float(self.exchange.amount_to_precision(symbol, amount))
        except Exception as e:
            logger.warning(f"Could not apply amount precision for {symbol}: {e}")

        if amount <= 0:
            logger.error(f"Order rejected: amount rounds to zero for {symbol}")
            return None

        if isinstance(market, dict):
            limits = market.get("limits") or {}
            min_amount = (limits.get("amount") or {}).get("min")
            min_cost = (limits.get("cost") or {}).get("min")

            if isinstance(min_amount, (int, float)) and amount < min_amount:
                logger.error(
                    f"Order rejected: amount {amount} below exchange minimum "
                    f"{min_amount} for {symbol}"
                )
                return None

            if (
                isinstance(min_cost, (int, float))
                and isinstance(price, (int, float))
                and price > 0
                and amount * price < min_cost
            ):
                logger.error(
                    f"Order rejected: notional {amount * price:.8f} below exchange "
                    f"minimum cost {min_cost} for {symbol}"
                )
                return None

        return amount

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
    ) -> Optional[ExchangeOrder]:
        """Place a market order.

        Args:
            symbol: Trading pair symbol
            side: Buy or sell
            amount: Amount to trade (in base currency)

        Returns:
            Order result or None if failed
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return None

        try:
            # Pre-flight: precision + min limits (use last price for notional)
            ticker = await self.get_ticker(symbol)
            estimate_price = ticker.last if ticker else 0
            amount = self._preflight_order(symbol, amount, estimate_price)
            if amount is None:
                return None

            order = await self._execute_with_retry(
                self.exchange.create_order,
                symbol,
                "market",
                side.value,
                amount,
            )
            # Audit trail for live execution: keep the full raw response
            logger.info(f"Raw exchange response for market {side.value} {symbol}: {order}")
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"Failed to place market {side.value} order for {symbol}: {e}")
            return None

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
    ) -> Optional[ExchangeOrder]:
        """Place a limit order.

        Args:
            symbol: Trading pair symbol
            side: Buy or sell
            amount: Amount to trade (in base currency)
            price: Limit price

        Returns:
            Order result or None if failed
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return None

        try:
            # Pre-flight: precision + min limits at the limit price
            try:
                price = float(self.exchange.price_to_precision(symbol, price))
            except Exception as e:
                logger.warning(f"Could not apply price precision for {symbol}: {e}")
            amount = self._preflight_order(symbol, amount, price)
            if amount is None:
                return None

            order = await self._execute_with_retry(
                self.exchange.create_order,
                symbol,
                "limit",
                side.value,
                amount,
                price,
            )
            # Audit trail for live execution: keep the full raw response
            logger.info(f"Raw exchange response for limit {side.value} {symbol}: {order}")
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"Failed to place limit {side.value} order for {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: The exchange order ID
            symbol: Trading pair symbol

        Returns:
            True if cancelled successfully
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return False

        try:
            await self._execute_with_retry(
                self.exchange.cancel_order, order_id, symbol
            )
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[ExchangeOrder]:
        """Get order status.

        Args:
            order_id: The exchange order ID
            symbol: Trading pair symbol

        Returns:
            Order data or None if failed
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return None

        try:
            order = await self._execute_with_retry(
                self.exchange.fetch_order, order_id, symbol
            )
            return self._parse_order(order)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        if not self.is_connected():
            logger.error("Not connected to exchange")
            return []

        try:
            orders = await self._execute_with_retry(
                self.exchange.fetch_open_orders, symbol
            )
            return [self._parse_order(o) for o in orders]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def _parse_order(self, order: Dict[str, Any]) -> ExchangeOrder:
        """Parse ccxt order response to ExchangeOrder."""
        fee = order.get("fee", {}) or {}
        return ExchangeOrder(
            id=str(order.get("id", "")),
            symbol=order.get("symbol", ""),
            side=order.get("side", ""),
            type=order.get("type", ""),
            amount=order.get("amount", 0) or 0,
            price=order.get("price", 0) or order.get("average", 0) or 0,
            cost=order.get("cost", 0) or 0,
            fee=fee.get("cost", 0) or 0,
            fee_currency=fee.get("currency", "USDT"),
            status=order.get("status", "unknown"),
            timestamp=datetime.fromtimestamp(order["timestamp"] / 1000) if order.get("timestamp") else datetime.utcnow(),
            filled=order.get("filled", 0) or 0,
            remaining=order.get("remaining", 0) or 0,
        )

    async def get_trading_pairs(self) -> List[str]:
        """Get all available trading pairs.

        Returns:
            List of trading pair symbols
        """
        if not self.is_connected():
            return []

        try:
            if not self.exchange.markets:
                await self._execute_with_retry(self.exchange.load_markets)
            return list(self.exchange.markets.keys())
        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            return []


class SimulatedExchangeService(ExchangeService):
    """Simulated exchange service for dry run mode.

    Market data is REAL (fetched from the exchange's public API, which
    requires no credentials); balances, order fills, and order history are
    simulated. Never fabricates prices: if real market data is unavailable,
    get_ticker returns None and the caller must skip trading.
    """

    def __init__(self, initial_balance: float = 10000.0, ticker_cache_ttl: float = 2.0):
        """Initialize simulated exchange.

        Args:
            initial_balance: Initial USDT balance for simulation
            ticker_cache_ttl: Seconds to cache tickers, limiting public API
                load from per-second bot loops
        """
        super().__init__()
        self._simulated_balance = {"USDT": initial_balance}
        self._simulated_orders: Dict[str, ExchangeOrder] = {}
        self._order_counter = 0
        self._ticker_cache_ttl = ticker_cache_ttl
        self._ticker_cache: Dict[str, tuple] = {}

    async def connect(self) -> bool:
        """Connect to the exchange's public API for real market data."""
        connected = await super().connect()
        if connected:
            logger.info("Simulated exchange connected (real market data, simulated fills)")
        else:
            logger.error("Simulated exchange: failed to connect to public market data API")
        return connected

    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get a real ticker from the public API, cached for a short TTL."""
        cached = self._ticker_cache.get(symbol)
        if cached and (time.monotonic() - cached[0]) < self._ticker_cache_ttl:
            return cached[1]

        ticker = await super().get_ticker(symbol)
        if ticker:
            self._ticker_cache[symbol] = (time.monotonic(), ticker)
        return ticker

    async def get_balance(self, currency: str = "USDT") -> Optional[Balance]:
        """Get simulated balance."""
        balance = self._simulated_balance.get(currency, 0)
        return Balance(
            currency=currency,
            free=balance,
            used=0,
            total=balance,
        )

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
    ) -> Optional[ExchangeOrder]:
        """Place simulated market order."""
        ticker = await self.get_ticker(symbol)
        if not ticker:
            return None

        price = ticker.ask if side == OrderSide.BUY else ticker.bid
        cost = amount * price
        fee = cost * 0.001  # 0.1% fee

        # Update simulated balance
        base, quote = symbol.split("/")
        if side == OrderSide.BUY:
            if self._simulated_balance.get(quote, 0) < cost + fee:
                logger.error("Insufficient simulated balance")
                return None
            self._simulated_balance[quote] = self._simulated_balance.get(quote, 0) - cost - fee
            self._simulated_balance[base] = self._simulated_balance.get(base, 0) + amount
        else:
            if self._simulated_balance.get(base, 0) < amount:
                logger.error("Insufficient simulated balance")
                return None
            self._simulated_balance[base] = self._simulated_balance.get(base, 0) - amount
            self._simulated_balance[quote] = self._simulated_balance.get(quote, 0) + cost - fee

        self._order_counter += 1
        order = ExchangeOrder(
            id=f"sim_{self._order_counter}",
            symbol=symbol,
            side=side.value,
            type="market",
            amount=amount,
            price=price,
            cost=cost,
            fee=fee,
            fee_currency=quote,
            status="closed",
            timestamp=datetime.utcnow(),
            filled=amount,
            remaining=0,
        )
        self._simulated_orders[order.id] = order
        return order

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        price: float,
    ) -> Optional[ExchangeOrder]:
        """Place simulated limit order (immediately filled for simplicity)."""
        # For simulation, treat limit orders as immediately filled
        return await self.place_market_order(symbol, side, amount)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel simulated order."""
        if order_id in self._simulated_orders:
            self._simulated_orders[order_id].status = "canceled"
            return True
        return False

    async def get_order(self, order_id: str, symbol: str) -> Optional[ExchangeOrder]:
        """Get simulated order."""
        return self._simulated_orders.get(order_id)

    def set_balance(self, currency: str, amount: float) -> None:
        """Set simulated balance for testing."""
        self._simulated_balance[currency] = amount
