"""Tests for health endpoint."""

import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health endpoint returns OK."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
