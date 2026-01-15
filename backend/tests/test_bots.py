"""Tests for bot CRUD operations."""

import pytest


@pytest.mark.asyncio
async def test_create_bot(client):
    """Test creating a new bot."""
    bot_data = {
        "name": "Test Bot",
        "trading_pair": "BTC/USDT",
        "strategy": "dca_accumulator",
        "budget": 1000.0,
        "is_dry_run": True
    }

    response = await client.post("/api/bots", json=bot_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Bot"
    assert data["trading_pair"] == "BTC/USDT"
    assert data["status"] == "created"


@pytest.mark.asyncio
async def test_create_bot_whitespace_name_rejected(client):
    """Test that whitespace-only name is rejected (#168)."""
    bot_data = {
        "name": "   ",  # Whitespace only
        "trading_pair": "BTC/USDT",
        "strategy": "dca_accumulator",
        "budget": 1000.0
    }

    response = await client.post("/api/bots", json=bot_data)
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_get_bots_empty(client):
    """Test getting bots when none exist."""
    response = await client.get("/api/bots")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_bots_list(client):
    """Test getting list of bots."""
    # Create a bot first
    bot_data = {
        "name": "Test Bot",
        "trading_pair": "ETH/USDT",
        "strategy": "mean_reversion",
        "budget": 500.0
    }
    await client.post("/api/bots", json=bot_data)

    response = await client.get("/api/bots")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Test Bot"


@pytest.mark.asyncio
async def test_get_bot_by_id(client):
    """Test getting a specific bot."""
    # Create a bot
    bot_data = {
        "name": "Specific Bot",
        "trading_pair": "BTC/USDT",
        "strategy": "adaptive_grid",
        "budget": 2000.0
    }
    create_response = await client.post("/api/bots", json=bot_data)
    bot_id = create_response.json()["id"]

    response = await client.get(f"/api/bots/{bot_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Specific Bot"


@pytest.mark.asyncio
async def test_get_bot_not_found(client):
    """Test getting non-existent bot."""
    response = await client.get("/api/bots/9999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_bot(client):
    """Test updating a bot."""
    # Create a bot
    bot_data = {
        "name": "Original Name",
        "trading_pair": "BTC/USDT",
        "strategy": "dca_accumulator",
        "budget": 1000.0
    }
    create_response = await client.post("/api/bots", json=bot_data)
    bot_id = create_response.json()["id"]

    # Update it
    update_data = {"name": "Updated Name", "budget": 2000.0}
    response = await client.put(f"/api/bots/{bot_id}", json=update_data)
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Name"
    assert response.json()["budget"] == 2000.0


@pytest.mark.asyncio
async def test_delete_bot(client):
    """Test deleting a bot."""
    # Create a bot
    bot_data = {
        "name": "To Delete",
        "trading_pair": "BTC/USDT",
        "strategy": "mean_reversion",
        "budget": 500.0
    }
    create_response = await client.post("/api/bots", json=bot_data)
    bot_id = create_response.json()["id"]

    # Delete it
    response = await client.delete(f"/api/bots/{bot_id}")
    assert response.status_code == 204

    # Verify it's gone
    get_response = await client.get(f"/api/bots/{bot_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_filter_bots_by_status(client):
    """Test filtering bots by status."""
    # Create bots
    await client.post("/api/bots", json={
        "name": "Bot 1",
        "trading_pair": "BTC/USDT",
        "strategy": "mean_reversion",
        "budget": 1000.0
    })

    response = await client.get("/api/bots?status=created")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "created"


@pytest.mark.asyncio
async def test_copy_bot(client):
    """Test copying a bot."""
    # Create original bot
    bot_data = {
        "name": "Original",
        "trading_pair": "BTC/USDT",
        "strategy": "dca_accumulator",
        "budget": 1000.0
    }
    create_response = await client.post("/api/bots", json=bot_data)
    bot_id = create_response.json()["id"]

    # Copy it
    response = await client.post(f"/api/bots/{bot_id}/copy?new_name=Copy")
    assert response.status_code == 200
    assert response.json()["name"] == "Copy"
    assert response.json()["trading_pair"] == "BTC/USDT"
