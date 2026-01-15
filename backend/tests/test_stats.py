"""Tests for stats endpoint."""

import pytest


@pytest.mark.asyncio
async def test_get_stats(client):
    """Test getting stats."""
    response = await client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()

    assert "total_bots" in data
    assert "running_bots" in data
    assert "total_pnl" in data
    assert "active_trades" in data


@pytest.mark.asyncio
async def test_stats_with_bots(client):
    """Test stats after creating bots."""
    # Create a bot
    await client.post("/api/bots", json={
        "name": "Stats Test Bot",
        "trading_pair": "BTC/USDT",
        "strategy": "mean_reversion",
        "budget": 1000.0
    })

    response = await client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_bots"] == 1
