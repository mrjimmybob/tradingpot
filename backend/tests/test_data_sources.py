"""Tests for external data sources API."""

import pytest


@pytest.mark.asyncio
async def test_get_data_sources(client):
    """Test getting all data source configurations."""
    response = await client.get("/api/data-sources/sources")
    assert response.status_code == 200
    data = response.json()

    # Should have all 5 sources
    assert "news_api" in data
    assert "fear_greed" in data
    assert "onchain_metrics" in data
    assert "social_sentiment" in data
    assert "market_conditions" in data

    # All should be disabled by default
    for source in data.values():
        assert source["enabled"] == False


@pytest.mark.asyncio
async def test_enable_data_source(client):
    """Test enabling a data source."""
    response = await client.put(
        "/api/data-sources/sources/fear_greed",
        json={"enabled": True}
    )
    assert response.status_code == 200
    assert response.json()["enabled"] == True


@pytest.mark.asyncio
async def test_disable_data_source(client):
    """Test disabling a data source."""
    # First enable
    await client.put(
        "/api/data-sources/sources/market_conditions",
        json={"enabled": True}
    )

    # Then disable
    response = await client.put(
        "/api/data-sources/sources/market_conditions",
        json={"enabled": False}
    )
    assert response.status_code == 200
    assert response.json()["enabled"] == False


@pytest.mark.asyncio
async def test_bulk_enable_sources(client):
    """Test enabling all sources at once."""
    response = await client.post(
        "/api/data-sources/sources/bulk",
        json={"enabled": True}
    )
    assert response.status_code == 200
    data = response.json()

    for source in data.values():
        assert source["enabled"] == True


@pytest.mark.asyncio
async def test_bulk_disable_sources(client):
    """Test disabling all sources at once."""
    # First enable all
    await client.post("/api/data-sources/sources/bulk", json={"enabled": True})

    # Then disable all
    response = await client.post(
        "/api/data-sources/sources/bulk",
        json={"enabled": False}
    )
    assert response.status_code == 200
    data = response.json()

    for source in data.values():
        assert source["enabled"] == False


@pytest.mark.asyncio
async def test_get_aggregated_signals(client):
    """Test getting aggregated signals."""
    response = await client.get("/api/data-sources/signals")
    assert response.status_code == 200
    data = response.json()

    assert "overall_sentiment" in data
    assert "confidence" in data
    assert "signal" in data
    assert "contributing_sources" in data


@pytest.mark.asyncio
async def test_invalid_source_type(client):
    """Test updating invalid source type."""
    response = await client.put(
        "/api/data-sources/sources/invalid_source",
        json={"enabled": True}
    )
    assert response.status_code == 400
