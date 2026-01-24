"""Pytest configuration and fixtures."""

import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.models import Base, get_session


# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create a fresh test database for each test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture(scope="function")
async def client(test_db):
    """Create test client with test database."""

    async def override_get_session():
        yield test_db

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def sample_bot(test_db):
    """Create a sample bot for testing."""
    from app.models import Bot

    bot = Bot(
        name="Test Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True
    )
    test_db.add(bot)
    await test_db.flush()
    return bot


@pytest.fixture
async def sample_order(test_db, sample_bot):
    """Create a sample order for testing."""
    from app.models import Order, OrderType, OrderStatus

    order = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True
    )
    test_db.add(order)
    await test_db.flush()
    return order


@pytest.fixture
async def sample_trade(test_db, sample_bot, sample_order):
    """Create a sample trade for testing."""
    from app.services.accounting import TradeRecorderService
    from app.models import TradeSide

    trade_recorder = TradeRecorderService(test_db)
    trade = await trade_recorder.record_trade(
        order_id=sample_order.id,
        owner_id="test_owner",
        bot_id=sample_bot.id,
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.BUY,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.01,
        quote_amount=500.0,
        price=50000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.1,
    )
    await test_db.flush()
    return trade
