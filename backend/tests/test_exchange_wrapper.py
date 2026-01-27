"""Unit tests for exchange wrapper service.

Tests the boundary between bot logic and CCXT library.
All CCXT interactions are fully mocked - no real exchange calls.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import ccxt.async_support as ccxt

from app.services.exchange import ExchangeService, OrderSide


def create_mock_ccxt_order(
    order_id: str = "12345",
    symbol: str = "BTC/USDT",
    side: str = "buy",
    order_type: str = "market",
    amount: float = 0.1,
    price: float = 50000.0,
    cost: float = 5000.0,
    status: str = "closed",
    filled: float = None,
    remaining: float = None,
) -> dict:
    """Create a mock CCXT order response."""
    if filled is None:
        filled = amount if status == "closed" else 0.0
    if remaining is None:
        remaining = 0.0 if status == "closed" else amount

    return {
        "id": order_id,
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "amount": amount,
        "price": price,
        "average": price,
        "cost": cost,
        "filled": filled,
        "remaining": remaining,
        "status": status,
        "timestamp": int(datetime(2025, 6, 1, 12, 0, 0).timestamp() * 1000),
        "fee": {"cost": 5.0, "currency": "USDT"},
    }


@pytest.fixture
def mock_exchange_connected():
    """Create a connected mock exchange service."""
    with patch("app.services.exchange.ccxt") as mock_ccxt_module:
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange_class = Mock(return_value=mock_exchange)
        mock_ccxt_module.mexc = mock_exchange_class
        
        with patch.object(ExchangeService, "_load_config", return_value={}):
            service = ExchangeService(exchange_id="mexc")
            yield service, mock_exchange


class TestOrderPlacementSuccess:
    """Test successful order placement scenarios."""
    
    @pytest.mark.asyncio
    async def test_market_buy_order_success(self):
        """Market buy order success."""
        mock_order = create_mock_ccxt_order(
            order_id="buy123",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            amount=0.1,
            price=50000.0,
            status="closed",
        )
        
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(return_value=mock_order)
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is not None
        assert result.id == "buy123"
        assert result.symbol == "BTC/USDT"
        assert result.side == "buy"
        assert result.type == "market"
        assert result.amount == 0.1
        assert result.price == 50000.0
        assert result.status == "closed"
        mock_exchange.create_order.assert_called_once_with(
            "BTC/USDT", "market", "buy", 0.1
        )
    
    @pytest.mark.asyncio
    async def test_market_sell_order_success(self):
        """Market sell order success."""
        mock_order = create_mock_ccxt_order(
            order_id="sell456",
            symbol="BTC/USDT",
            side="sell",
            order_type="market",
            amount=0.05,
            price=51000.0,
            status="closed",
        )
        
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(return_value=mock_order)
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.SELL, amount=0.05
                )
        
        assert result is not None
        assert result.side == "sell"
        assert result.amount == 0.05
        mock_exchange.create_order.assert_called_once_with(
            "BTC/USDT", "market", "sell", 0.05
        )
    
    @pytest.mark.asyncio
    async def test_limit_order_success(self):
        """Limit order success."""
        mock_order = create_mock_ccxt_order(
            order_id="limit789",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            amount=0.1,
            price=48000.0,
            status="open",
            filled=0.0,
            remaining=0.1,
        )
        
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(return_value=mock_order)
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_limit_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1, price=48000.0
                )
        
        assert result is not None
        assert result.id == "limit789"
        assert result.type == "limit"
        assert result.price == 48000.0
        assert result.status == "open"
        mock_exchange.create_order.assert_called_once_with(
            "BTC/USDT", "limit", "buy", 0.1, 48000.0
        )
    
    @pytest.mark.asyncio
    async def test_returned_order_contains_expected_fields(self):
        """Returned order object contains expected fields."""
        mock_order = create_mock_ccxt_order(
            order_id="full123",
            symbol="ETH/USDT",
            side="buy",
            order_type="market",
            amount=1.5,
            price=2500.0,
            cost=3750.0,
            status="closed",
        )
        
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(return_value=mock_order)
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="ETH/USDT", side=OrderSide.BUY, amount=1.5
                )
        
        assert hasattr(result, "id")
        assert hasattr(result, "price")
        assert hasattr(result, "amount")
        assert hasattr(result, "status")
        assert result.id == "full123"
        assert result.price == 2500.0
        assert result.amount == 1.5
        assert result.status == "closed"


class TestOrderPlacementFailure:
    """Test order placement failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_exchange_rejects_order_throws_exception(self):
        """Exchange rejects order (throws exception)."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.ExchangeError("Order rejected by exchange")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_invalid_symbol(self):
        """Invalid symbol (e.g. "FAKE/USDT")."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.BadSymbol("Market symbol FAKE/USDT is invalid")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="FAKE/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_error_from_exchange(self):
        """Insufficient balance error from exchange."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.InsufficientFunds("Insufficient balance")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=100.0
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_order_placement_returns_malformed_response(self):
        """Order placement returns malformed response."""
        malformed_order = {
            "id": "malformed123",
        }
        
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(return_value=malformed_order)
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is not None
        assert result.id == "malformed123"
        assert result.price == 0.0
        assert result.amount == 0.0


class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.mark.asyncio
    async def test_network_timeout(self):
        """Network timeout."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.NetworkError("Connection timeout")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService, "_load_config", return_value={"retry_count": 3}
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
        assert mock_exchange.create_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Authentication error."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.AuthenticationError("Invalid API key")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_generic_ccxt_exception(self):
        """Generic CCXT exception."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.ExchangeError("Unknown exchange error")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
        assert mock_exchange.create_order.call_count == 1
    
    @pytest.mark.asyncio
    async def test_errors_are_caught(self):
        """Ensure errors are caught."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_errors_wrapped_or_reraised_correctly(self):
        """Ensure errors are wrapped or re-raised correctly."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.ExchangeError("Exchange error")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None


class TestRateLimitHandling:
    """Test rate limit handling."""
    
    @pytest.mark.asyncio
    async def test_simulate_rate_limit_exception(self):
        """Simulate rate-limit exception."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_order = create_mock_ccxt_order()
            mock_exchange.create_order = AsyncMock(
                side_effect=[
                    ccxt.RateLimitExceeded("Rate limit exceeded"),
                    ccxt.RateLimitExceeded("Rate limit exceeded"),
                    mock_order,
                ]
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService,
                "_load_config",
                return_value={"retry_count": 3, "retry_delay": 0.01},
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    result = await service.place_market_order(
                        symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                    )
        
        assert result is not None
        assert mock_exchange.create_order.call_count == 3
        assert mock_sleep.call_count == 2
    
    @pytest.mark.asyncio
    async def test_verify_retry_logic(self):
        """Verify retry logic."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_order = create_mock_ccxt_order()
            mock_exchange.create_order = AsyncMock(
                side_effect=[
                    ccxt.RateLimitExceeded("Rate limit"),
                    mock_order,
                ]
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService,
                "_load_config",
                return_value={"retry_count": 3, "retry_delay": 0.01},
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await service.place_market_order(
                        symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                    )
        
        assert result is not None
        assert mock_exchange.create_order.call_count == 2
    
    @pytest.mark.asyncio
    async def test_graceful_failure_with_clear_error_result(self):
        """Graceful failure with clear error result."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.RateLimitExceeded("Rate limit exceeded")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService,
                "_load_config",
                return_value={"retry_count": 3, "retry_delay": 0.01},
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await service.place_market_order(
                        symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                    )
        
        assert result is None
        assert mock_exchange.create_order.call_count == 3


class TestInputValidation:
    """Test input validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol format."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.BadSymbol("Invalid symbol format")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="INVALID_FORMAT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_zero_order_amount(self):
        """Zero order amount."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.InvalidOrder("Amount must be positive")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.0
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_negative_order_amount(self):
        """Negative order amount."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.InvalidOrder("Amount cannot be negative")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=-0.1
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_zero_price_for_limit_order(self):
        """Zero price for limit order."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.InvalidOrder("Price must be positive")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_limit_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1, price=0.0
                )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_negative_price_for_limit_order(self):
        """Negative price for limit order."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.InvalidOrder("Price cannot be negative")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_limit_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1, price=-50000.0
                )
        
        assert result is None


class TestNotConnectedBehavior:
    """Test behavior when not connected."""
    
    @pytest.mark.asyncio
    async def test_place_order_not_connected(self):
        """Place order not connected."""
        service = ExchangeService(exchange_id="mexc")
        
        result = await service.place_market_order(
            symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_ticker_not_connected(self):
        """Get ticker not connected."""
        service = ExchangeService(exchange_id="mexc")
        
        result = await service.get_ticker("BTC/USDT")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self):
        """Get balance not connected."""
        service = ExchangeService(exchange_id="mexc")
        
        result = await service.get_balance("USDT")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cancel_order_not_connected(self):
        """Cancel order not connected."""
        service = ExchangeService(exchange_id="mexc")
        
        result = await service.cancel_order("12345", "BTC/USDT")
        
        assert result is False


class TestRetryLogic:
    """Test retry logic for transient errors."""
    
    @pytest.mark.asyncio
    async def test_network_error_triggers_retry(self):
        """Network error triggers retry."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_order = create_mock_ccxt_order()
            mock_exchange.create_order = AsyncMock(
                side_effect=[
                    ccxt.NetworkError("Timeout"),
                    mock_order,
                ]
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService, "_load_config", return_value={"retry_count": 3}
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is not None
        assert mock_exchange.create_order.call_count == 2
    
    @pytest.mark.asyncio
    async def test_exchange_error_does_not_trigger_retry(self):
        """Exchange error does not trigger retry."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.ExchangeError("Invalid order")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
        assert mock_exchange.create_order.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retries_eventually_succeed(self):
        """Retries eventually succeed."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_order = create_mock_ccxt_order()
            mock_exchange.create_order = AsyncMock(
                side_effect=[
                    ccxt.NetworkError("Error 1"),
                    ccxt.NetworkError("Error 2"),
                    mock_order,
                ]
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService, "_load_config", return_value={"retry_count": 3}
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is not None
        assert mock_exchange.create_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_all_retries_fail_returns_none(self):
        """All retries fail returns None."""
        with patch("app.services.exchange.ccxt") as mock_ccxt_module:
            mock_ccxt_module.NetworkError = ccxt.NetworkError
            mock_ccxt_module.ExchangeError = ccxt.ExchangeError
            mock_ccxt_module.RateLimitExceeded = ccxt.RateLimitExceeded
            
            mock_exchange = AsyncMock()
            mock_exchange.load_markets = AsyncMock()
            mock_exchange.create_order = AsyncMock(
                side_effect=ccxt.NetworkError("Connection error")
            )
            
            mock_exchange_class = Mock(return_value=mock_exchange)
            mock_ccxt_module.mexc = mock_exchange_class
            
            with patch.object(
                ExchangeService, "_load_config", return_value={"retry_count": 3}
            ):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                
                result = await service.place_market_order(
                    symbol="BTC/USDT", side=OrderSide.BUY, amount=0.1
                )
        
        assert result is None
        assert mock_exchange.create_order.call_count == 3
