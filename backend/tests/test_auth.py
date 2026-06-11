"""Tests for API authentication and the loopback binding fail-safe."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import WebSocketDisconnect

from app.main import binding_failsafe_error
from app.routers.websocket import websocket_endpoint
from app.services.config import get_api_token


TOKEN = "test-secret-token"


class TestApiTokenResolution:
    """get_api_token: env var first, config fallback."""

    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", "from-env")
        assert get_api_token() == "from-env"

    def test_empty_when_unset(self, monkeypatch):
        monkeypatch.delenv("TRADINGBOT_API_TOKEN", raising=False)
        with patch("app.services.config.config_service.get", return_value=None):
            assert get_api_token() == ""


class TestHttpAuthMiddleware:
    """Bearer-token enforcement on /api routes."""

    @pytest.mark.asyncio
    async def test_request_without_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        response = await client.get("/api/config/strategies")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_request_with_wrong_token_rejected(self, client, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        response = await client.get(
            "/api/config/strategies",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_request_with_valid_token_served(self, client, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        response = await client.get(
            "/api/config/strategies",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_stays_open(self, client, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        response = await client.get("/api/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_no_token_configured_means_open(self, client, monkeypatch):
        monkeypatch.delenv("TRADINGBOT_API_TOKEN", raising=False)
        with patch("app.services.config.config_service.get", return_value=None):
            response = await client.get("/api/config/strategies")
        assert response.status_code == 200


class TestWebSocketAuth:
    """Token check before accepting WebSocket connections."""

    @pytest.mark.asyncio
    async def test_ws_refused_without_token(self, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        ws = AsyncMock()
        ws.query_params = {}

        with patch("app.routers.websocket.ws_manager") as manager:
            manager.connect_client = AsyncMock()
            await websocket_endpoint(ws)
            manager.connect_client.assert_not_awaited()

        ws.close.assert_awaited_once()
        assert ws.close.await_args.kwargs.get("code") == 1008

    @pytest.mark.asyncio
    async def test_ws_refused_with_wrong_token(self, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        ws = AsyncMock()
        ws.query_params = {"token": "wrong"}

        with patch("app.routers.websocket.ws_manager") as manager:
            manager.connect_client = AsyncMock()
            await websocket_endpoint(ws)
            manager.connect_client.assert_not_awaited()

        ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ws_accepted_with_valid_token(self, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_API_TOKEN", TOKEN)
        ws = AsyncMock()
        ws.query_params = {"token": TOKEN}
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        with patch("app.routers.websocket.ws_manager") as manager:
            manager.connect_client = AsyncMock()
            manager.disconnect_client = AsyncMock()
            await websocket_endpoint(ws)
            manager.connect_client.assert_awaited_once_with(ws)


class TestBindingFailsafe:
    """Refuse non-loopback binding without a token."""

    def test_loopback_without_token_ok(self):
        assert binding_failsafe_error("127.0.0.1", "") == ""
        assert binding_failsafe_error("localhost", "") == ""

    def test_public_without_token_blocked(self):
        error = binding_failsafe_error("0.0.0.0", "")
        assert "API token" in error

    def test_public_with_token_ok(self):
        assert binding_failsafe_error("0.0.0.0", TOKEN) == ""
