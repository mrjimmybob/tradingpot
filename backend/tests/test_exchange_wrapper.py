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


class TestCredentialResolution:
    """Credentials: env vars override YAML, placeholders treated as unset."""

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        """Environment variables win over YAML values."""
        config_file = tmp_path / "exchanges.yaml"
        config_file.write_text(
            'mexc:\n  api_key: "yaml_key"\n  api_secret: "yaml_secret"\n'
        )
        monkeypatch.setenv("MEXC_API_KEY", "env_key")
        monkeypatch.setenv("MEXC_API_SECRET", "env_secret")

        service = ExchangeService(exchange_id="mexc", config_path=str(config_file))
        config = service._load_config()

        assert config["api_key"] == "env_key"
        assert config["api_secret"] == "env_secret"
        assert service.has_credentials()

    def test_yaml_used_when_no_env(self, tmp_path, monkeypatch):
        """YAML values are used when no env vars are set."""
        monkeypatch.delenv("MEXC_API_KEY", raising=False)
        monkeypatch.delenv("MEXC_API_SECRET", raising=False)
        config_file = tmp_path / "exchanges.yaml"
        config_file.write_text(
            'mexc:\n  api_key: "real_key"\n  api_secret: "real_secret"\n'
        )

        service = ExchangeService(exchange_id="mexc", config_path=str(config_file))
        config = service._load_config()

        assert config["api_key"] == "real_key"
        assert service.has_credentials()

    def test_placeholder_treated_as_unset(self, tmp_path, monkeypatch):
        """Placeholder credentials (YOUR_*) must never be used."""
        monkeypatch.delenv("MEXC_API_KEY", raising=False)
        monkeypatch.delenv("MEXC_API_SECRET", raising=False)
        config_file = tmp_path / "exchanges.yaml"
        config_file.write_text(
            'mexc:\n  api_key: "YOUR_MEXC_API_KEY"\n  api_secret: "YOUR_MEXC_API_SECRET"\n'
        )

        service = ExchangeService(exchange_id="mexc", config_path=str(config_file))
        config = service._load_config()

        assert config["api_key"] == ""
        assert config["api_secret"] == ""
        assert not service.has_credentials()

    def test_missing_config_file_means_no_credentials(self, tmp_path, monkeypatch):
        """No config file and no env vars -> no credentials."""
        monkeypatch.delenv("MEXC_API_KEY", raising=False)
        monkeypatch.delenv("MEXC_API_SECRET", raising=False)

        service = ExchangeService(
            exchange_id="mexc", config_path=str(tmp_path / "missing.yaml")
        )

        assert not service.has_credentials()


class TestSimulatedExchangeRealMarketData:
    """Dry-run market data is real (public API); fills stay simulated."""

    def _make_service_with_mock_client(self, ttl: float = 2.0):
        """SimulatedExchangeService wired to a mocked ccxt client."""
        from app.services.exchange import SimulatedExchangeService

        service = SimulatedExchangeService(
            initial_balance=10000.0, ticker_cache_ttl=ttl
        )
        service._connected = True
        mock_client = AsyncMock()
        mock_client.fetch_ticker = AsyncMock(
            return_value={
                "symbol": "BTC/USDT",
                "bid": 99990.0,
                "ask": 100010.0,
                "last": 100000.0,
                "baseVolume": 1234.0,
                "timestamp": int(datetime(2026, 6, 1).timestamp() * 1000),
            }
        )
        service.exchange = mock_client
        return service, mock_client

    @pytest.mark.asyncio
    async def test_ticker_comes_from_public_api(self):
        """Ticker is fetched from the exchange, not hardcoded."""
        service, mock_client = self._make_service_with_mock_client()

        ticker = await service.get_ticker("BTC/USDT")

        mock_client.fetch_ticker.assert_awaited_once_with("BTC/USDT")
        assert ticker.last == 100000.0
        assert ticker.bid == 99990.0
        assert ticker.ask == 100010.0

    @pytest.mark.asyncio
    async def test_ticker_cached_within_ttl(self):
        """Second request within TTL does not hit the API again."""
        service, mock_client = self._make_service_with_mock_client(ttl=60.0)

        first = await service.get_ticker("BTC/USDT")
        second = await service.get_ticker("BTC/USDT")

        assert mock_client.fetch_ticker.await_count == 1
        assert first.last == second.last

    @pytest.mark.asyncio
    async def test_no_fabricated_price_when_disconnected(self):
        """No fallback price: unavailable market data returns None."""
        from app.services.exchange import SimulatedExchangeService

        service = SimulatedExchangeService(initial_balance=10000.0)
        # Not connected, no client: must return None, never a made-up price
        result = await service.get_ticker("BTC/USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_market_order_fills_at_real_price(self):
        """Simulated fills use the real ticker price and mutate only simulated balances."""
        service, _ = self._make_service_with_mock_client()

        order = await service.place_market_order(
            symbol="BTC/USDT", side=OrderSide.BUY, amount=0.01
        )

        assert order is not None
        assert order.price == 100010.0  # real ask
        balance = await service.get_balance("USDT")
        assert balance.total < 10000.0
        btc = await service.get_balance("BTC")
        assert btc.total == 0.01


class TestOrderPreflight:
    """Live orders are validated against exchange precision and limits
    before submission; invalid orders never reach the exchange."""

    def _connected_service(self, markets: dict, ticker_price: float = 50000.0):
        service = ExchangeService(exchange_id="mexc")
        service._connected = True
        client = AsyncMock()
        client.fetch_ticker = AsyncMock(
            return_value={
                "symbol": "BTC/USDT",
                "bid": ticker_price - 1,
                "ask": ticker_price + 1,
                "last": ticker_price,
                "baseVolume": 100.0,
                "timestamp": int(datetime(2026, 6, 1).timestamp() * 1000),
            }
        )
        client.create_order = AsyncMock(return_value=create_mock_ccxt_order())
        client.amount_to_precision = Mock(side_effect=lambda s, a: f"{a:.4f}")
        client.price_to_precision = Mock(side_effect=lambda s, p: f"{p:.2f}")
        client.markets = markets
        service.exchange = client
        return service, client

    @pytest.mark.asyncio
    async def test_amount_rounded_to_precision(self):
        """Submitted amount is rounded to the market's precision."""
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.0001}, "cost": {"min": 1.0}}}}
        service, client = self._connected_service(markets)

        result = await service.place_market_order("BTC/USDT", OrderSide.BUY, 0.12345678)

        assert result is not None
        submitted_amount = client.create_order.call_args.args[3]
        assert submitted_amount == 0.1235  # rounded by amount_to_precision

    @pytest.mark.asyncio
    async def test_below_min_cost_rejected_locally(self):
        """Order below the market's minimum cost never reaches the exchange."""
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.0000001}, "cost": {"min": 10.0}}}}
        service, client = self._connected_service(markets, ticker_price=50000.0)

        # 0.0001 BTC * 50000 = $5 < $10 minimum cost
        result = await service.place_market_order("BTC/USDT", OrderSide.BUY, 0.0001)

        assert result is None
        client.create_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_below_min_amount_rejected_locally(self):
        """Order below the market's minimum amount never reaches the exchange."""
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.001}, "cost": {"min": 1.0}}}}
        service, client = self._connected_service(markets)

        result = await service.place_market_order("BTC/USDT", OrderSide.BUY, 0.0005)

        assert result is None
        client.create_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_amount_rounding_to_zero_rejected(self):
        """Amount that rounds to zero is rejected."""
        markets = {"BTC/USDT": {"limits": {}}}
        service, client = self._connected_service(markets)

        result = await service.place_market_order("BTC/USDT", OrderSide.BUY, 0.00000001)

        assert result is None
        client.create_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_limit_order_price_and_amount_normalized(self):
        """Limit orders get price precision applied and pass limits checks."""
        markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.0001}, "cost": {"min": 1.0}}}}
        service, client = self._connected_service(markets)

        result = await service.place_limit_order(
            "BTC/USDT", OrderSide.BUY, 0.12345678, 49999.987654
        )

        assert result is not None
        args = client.create_order.call_args.args
        assert args[3] == 0.1235       # amount rounded
        assert args[4] == 49999.99     # price rounded

    @pytest.mark.asyncio
    async def test_simulator_skips_preflight(self):
        """Dry-run fills are simulated and not subject to live order limits."""
        from app.services.exchange import SimulatedExchangeService

        service = SimulatedExchangeService(initial_balance=10000.0)
        service._connected = True
        client = AsyncMock()
        client.fetch_ticker = AsyncMock(
            return_value={
                "symbol": "BTC/USDT",
                "bid": 49999.0,
                "ask": 50001.0,
                "last": 50000.0,
                "baseVolume": 100.0,
                "timestamp": int(datetime(2026, 6, 1).timestamp() * 1000),
            }
        )
        service.exchange = client

        # Tiny order: would fail live min-cost checks, fills fine in dry-run
        order = await service.place_market_order("BTC/USDT", OrderSide.BUY, 0.00001)

        assert order is not None
        client.create_order.assert_not_awaited()
