"""API tests for WebSocket endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from datetime import datetime

from app.routers.websocket import router


@pytest.fixture
def app():
    """Create FastAPI test app."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


def create_mock_indicators(symbol="BTCUSDT", last_price=50000.0):
    """Helper to create mock market indicators."""
    indicators = Mock()
    indicators.symbol = symbol
    indicators.timestamp = datetime(2025, 1, 15, 10, 0, 0)
    indicators.last_price = last_price
    indicators.sentiment_score = 0.7
    indicators.risk_score = 0.3
    indicators.signal = "BUY"
    
    orderbook = Mock()
    orderbook.imbalance_ratio = 1.5
    orderbook.bid_volume = 1000.0
    orderbook.ask_volume = 800.0
    orderbook.has_liquidity_vacuum = False
    indicators.orderbook = orderbook
    
    volume_delta = Mock()
    volume_delta.buy_volume = 1200.0
    volume_delta.sell_volume = 900.0
    volume_delta.delta = 300.0
    volume_delta.normalized_delta = 0.25
    indicators.volume_delta = volume_delta
    
    spread = Mock()
    spread.spread = 10.0
    spread.spread_percent = 0.02
    spread.spread_widening = False
    indicators.spread = spread
    
    volatility = Mock()
    volatility.atr = 500.0
    volatility.atr_percent = 1.0
    volatility.atr_ratio = 1.2
    volatility.vol_regime = "NORMAL"
    indicators.volatility = volatility
    
    return indicators

class TestRESTSubscriptionEndpoints:
    """Tests for REST-based subscription endpoints."""

    def test_subscribe_symbol_success(self, client):
        """Subscribe to symbol via REST."""
        with patch("app.routers.websocket.ws_manager.subscribe_symbol", new_callable=AsyncMock, return_value=True):
            response = client.post("/ws/subscribe/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["symbol"] == "BTCUSDT"

    def test_subscribe_symbol_failure(self, client):
        """Subscribe to invalid symbol fails."""
        with patch("app.routers.websocket.ws_manager.subscribe_symbol", new_callable=AsyncMock, return_value=False):
            response = client.post("/ws/subscribe/INVALID")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_unsubscribe_symbol_success(self, client):
        """Unsubscribe from symbol via REST."""
        with patch("app.routers.websocket.ws_manager.unsubscribe_symbol", new_callable=AsyncMock, return_value=True):
            response = client.post("/ws/unsubscribe/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["symbol"] == "BTCUSDT"

    def test_unsubscribe_symbol_not_subscribed(self, client):
        """Unsubscribe from non-subscribed symbol."""
        with patch("app.routers.websocket.ws_manager.unsubscribe_symbol", new_callable=AsyncMock, return_value=False):
            response = client.post("/ws/unsubscribe/ETHUSDT")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestIndicatorEndpoints:
    """Tests for market indicator endpoints."""

    def test_get_indicators_for_symbol(self, client):
        """Get indicators for specific symbol."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTCUSDT"
        assert data["last_price"] == 50000.0
        assert data["sentiment_score"] == 0.7
        assert data["signal"] == "BUY"

    def test_get_indicators_not_available(self, client):
        """Returns error when indicators not available."""
        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=None):
            response = client.get("/ws/indicators/UNKNOWN")

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["symbol"] == "UNKNOWN"

    def test_indicators_response_includes_orderbook(self, client):
        """Indicators include orderbook data."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert "orderbook" in data
        assert data["orderbook"]["imbalance_ratio"] == 1.5

    def test_indicators_response_includes_volume_delta(self, client):
        """Indicators include volume delta."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert "volume_delta" in data
        assert data["volume_delta"]["delta"] == 300.0

    def test_indicators_response_includes_spread(self, client):
        """Indicators include spread data."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert "spread" in data
        assert data["spread"]["spread_percent"] == 0.02

    def test_indicators_response_includes_volatility(self, client):
        """Indicators include volatility data."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert "volatility" in data
        assert data["volatility"]["vol_regime"] == "NORMAL"

    def test_get_all_indicators(self, client):
        """Get indicators for all symbols."""
        mock_btc = create_mock_indicators(symbol="BTCUSDT", last_price=50000.0)
        mock_eth = create_mock_indicators(symbol="ETHUSDT", last_price=3000.0)
        
        all_indicators = {
            "BTCUSDT": mock_btc,
            "ETHUSDT": mock_eth,
        }

        with patch("app.routers.websocket.ws_manager.get_all_market_indicators", return_value=all_indicators):
            response = client.get("/ws/indicators")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert "BTCUSDT" in data["indicators"]
        assert "ETHUSDT" in data["indicators"]

    def test_get_all_indicators_empty(self, client):
        """Get all indicators when none available."""
        with patch("app.routers.websocket.ws_manager.get_all_market_indicators", return_value={}):
            response = client.get("/ws/indicators")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["indicators"] == {}


class TestWebSocketResponseShape:
    """Tests for response JSON shape validation."""

    def test_indicators_response_shape(self, client):
        """Indicators response has expected shape."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "timestamp" in data
        assert "last_price" in data
        assert "sentiment_score" in data
        assert "risk_score" in data
        assert "signal" in data

    def test_all_indicators_response_shape(self, client):
        """All indicators response has expected shape."""
        mock_indicators = {
            "BTCUSDT": create_mock_indicators(),
        }

        with patch("app.routers.websocket.ws_manager.get_all_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators")

        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "indicators" in data
        assert isinstance(data["indicators"], dict)


class TestWebSocketEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_numeric_values_are_serializable(self, client):
        """Numeric values in indicators are JSON serializable."""
        mock_indicators = create_mock_indicators()

        with patch("app.routers.websocket.ws_manager.get_market_indicators", return_value=mock_indicators):
            response = client.get("/ws/indicators/BTCUSDT")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["last_price"], (int, float))
        assert isinstance(data["sentiment_score"], (int, float))
        assert isinstance(data["risk_score"], (int, float))
