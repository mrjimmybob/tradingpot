"""Exchange service for interacting with crypto exchanges via ccxt."""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import ccxt.async_support as ccxt
import yaml

logger = logging.getLogger(__name__)


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
        """Load exchange configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                return config.get(self.exchange_id, {})
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

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
            order = await self._execute_with_retry(
                self.exchange.create_order,
                symbol,
                "market",
                side.value,
                amount,
            )
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
            order = await self._execute_with_retry(
                self.exchange.create_order,
                symbol,
                "limit",
                side.value,
                amount,
                price,
            )
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
    """Simulated exchange service for dry run mode."""

    def __init__(self, initial_balance: float = 10000.0):
        """Initialize simulated exchange.

        Args:
            initial_balance: Initial USDT balance for simulation
        """
        super().__init__()
        self._simulated_balance = {"USDT": initial_balance}
        self._simulated_orders: Dict[str, ExchangeOrder] = {}
        self._order_counter = 0
        self._connected = True

    async def connect(self) -> bool:
        """Simulated connection always succeeds."""
        self._connected = True
        logger.info("Connected to simulated exchange")
        return True

    async def disconnect(self) -> None:
        """Simulated disconnect."""
        self._connected = False
        logger.info("Disconnected from simulated exchange")

    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get simulated ticker (returns realistic mock data)."""
        # Return mock ticker data based on symbol
        mock_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2500.0,
            "SOL/USDT": 100.0,
            "XRP/USDT": 0.55,
            "ADA/USDT": 0.45,
            "DOGE/USDT": 0.08,
        }
        price = mock_prices.get(symbol, 100.0)
        spread = price * 0.001  # 0.1% spread

        return Ticker(
            symbol=symbol,
            bid=price - spread,
            ask=price + spread,
            last=price,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

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
