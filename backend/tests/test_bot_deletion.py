"""Regression tests for the bot deletion cascade.

Root cause being guarded against: ``StrategyPerformanceMetrics.bot`` was defined
with a bare ``backref="strategy_metrics"``. That produced a ``Bot.strategy_metrics``
collection with SQLAlchemy's DEFAULT relationship cascade ("save-update, merge"),
which on ``session.delete(bot)`` DISASSOCIATES children by emitting
``UPDATE strategy_performance_metrics SET bot_id=NULL``. Because ``bot_id`` is
``nullable=False`` the commit raised ``IntegrityError`` and the API returned 500
(which the frontend then surfaced as a JSON.parse error).

The fix declares an explicit ``Bot.strategy_metrics`` relationship with
``cascade="all, delete-orphan"`` (matching every other child), so the ORM DELETEs
the child rows with the parent. No schema migration is required — cascade is an
ORM-level directive, not DDL.

These tests prove every child table referencing ``Bot`` is removed on delete,
no orphans remain, the emitted SQL no longer nullifies the FK, and the API
returns a clean 204 (no JSON body) with correct repeat-delete behaviour.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import sessionmaker

from app.models import (
    Base, Bot, BotStatus, Order, OrderType, OrderStatus, Position, PositionSide,
    Trade, TradeSide, PnLSnapshot, WalletLedger, StrategyRotation, Alert,
    StrategyPerformanceMetrics,
)


# --- helpers ---------------------------------------------------------------

async def _make_bot(client) -> int:
    resp = await client.post("/api/bots", json={
        "name": "Delete Me",
        "trading_pair": "BTC/USDT",
        "strategy": "auto_mode",
        "budget": 1000.0,
        "is_dry_run": True,
    })
    assert resp.status_code == 201
    return resp.json()["id"]


async def _add_metrics(test_db, bot_id: int) -> None:
    test_db.add(StrategyPerformanceMetrics(
        bot_id=bot_id, strategy_name="mean_reversion",
        recent_pnl_pct=1.0, max_drawdown_pct=2.0, failure_count=1,
    ))
    await test_db.commit()


async def _add_order(test_db, bot_id: int) -> int:
    order = Order(
        bot_id=bot_id, order_type=OrderType.MARKET_BUY, trading_pair="BTC/USDT",
        amount=0.01, price=65000.0, status=OrderStatus.FILLED, strategy_used="auto_mode",
    )
    test_db.add(order)
    await test_db.commit()
    await test_db.refresh(order)
    return order.id


async def _add_position(test_db, bot_id: int) -> None:
    test_db.add(Position(
        bot_id=bot_id, trading_pair="BTC/USDT", side=PositionSide.LONG,
        entry_price=65000.0, current_price=65500.0, amount=0.01,
    ))
    await test_db.commit()


async def _add_history(test_db, bot_id: int, order_id: int) -> None:
    """Add the remaining historical child rows (trade, pnl, ledger, rotation, alert)."""
    test_db.add_all([
        Trade(
            order_id=order_id, owner_id="acct-1", bot_id=bot_id, exchange="simulated",
            trading_pair="BTC/USDT", side=TradeSide.BUY, base_asset="BTC",
            quote_asset="USDT", base_amount=0.01, quote_amount=650.0, price=65000.0,
            executed_at=datetime.utcnow(),
        ),
        PnLSnapshot(bot_id=bot_id, total_pnl=12.5),
        WalletLedger(
            owner_id="acct-1", bot_id=bot_id, asset="USDT", delta_amount=-650.0,
            reason="buy",
        ),
        StrategyRotation(bot_id=bot_id, from_strategy="mean_reversion", to_strategy="trend_following"),
        Alert(bot_id=bot_id, alert_type="info", message="hi"),
    ])
    await test_db.commit()


async def _count_all_children(test_db, bot_id: int) -> dict:
    """Return per-table counts of rows still referencing this bot_id."""
    counts = {}
    for model in (
        Order, Position, Trade, PnLSnapshot, WalletLedger, StrategyRotation,
        Alert, StrategyPerformanceMetrics,
    ):
        n = (await test_db.execute(
            select(func.count()).select_from(model).where(model.bot_id == bot_id)
        )).scalar()
        counts[model.__tablename__] = n
    return counts


# --- API-level deletion tests ---------------------------------------------

@pytest.mark.asyncio
async def test_delete_bot_no_children_succeeds(client):
    bot_id = await _make_bot(client)
    resp = await client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_bot_with_strategy_metrics_succeeds(client, test_db):
    """The exact failing case from the bug report."""
    bot_id = await _make_bot(client)
    await _add_metrics(test_db, bot_id)

    resp = await client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204

    remaining = (await test_db.execute(
        select(func.count()).select_from(StrategyPerformanceMetrics)
        .where(StrategyPerformanceMetrics.bot_id == bot_id)
    )).scalar()
    assert remaining == 0


@pytest.mark.asyncio
async def test_delete_bot_with_positions_succeeds(client, test_db):
    bot_id = await _make_bot(client)
    await _add_position(test_db, bot_id)
    assert (await client.delete(f"/api/bots/{bot_id}")).status_code == 204
    assert (await _count_all_children(test_db, bot_id))["positions"] == 0


@pytest.mark.asyncio
async def test_delete_bot_with_orders_succeeds(client, test_db):
    bot_id = await _make_bot(client)
    await _add_order(test_db, bot_id)
    assert (await client.delete(f"/api/bots/{bot_id}")).status_code == 204
    assert (await _count_all_children(test_db, bot_id))["orders"] == 0


@pytest.mark.asyncio
async def test_delete_bot_with_all_history_leaves_no_orphans(client, test_db):
    bot_id = await _make_bot(client)
    await _add_metrics(test_db, bot_id)
    order_id = await _add_order(test_db, bot_id)
    await _add_position(test_db, bot_id)
    await _add_history(test_db, bot_id, order_id)

    # Sanity: children exist before deletion.
    before = await _count_all_children(test_db, bot_id)
    assert all(v > 0 for v in before.values()), before

    resp = await client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204

    after = await _count_all_children(test_db, bot_id)
    assert all(v == 0 for v in after.values()), after
    # And the bot itself is gone.
    assert (await test_db.execute(select(Bot).where(Bot.id == bot_id))).scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_repeated_delete_returns_404(client):
    """First delete is 204; a second delete of the same id is a clean 404."""
    bot_id = await _make_bot(client)
    assert (await client.delete(f"/api/bots/{bot_id}")).status_code == 204
    second = await client.delete(f"/api/bots/{bot_id}")
    assert second.status_code == 404
    assert second.json()["detail"]  # valid JSON error body


@pytest.mark.asyncio
async def test_delete_returns_204_with_empty_body(client):
    """The documented contract: 204 No Content, empty body — nothing for the
    frontend to JSON.parse on success."""
    bot_id = await _make_bot(client)
    resp = await client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204
    assert resp.content == b""


# --- ORM-level proof: the emitted SQL no longer nullifies the FK -----------

def test_delete_emits_child_delete_not_fk_nullify():
    """Capture the literal SQL emitted on bot deletion and prove SQLAlchemy now
    DELETEs the strategy_performance_metrics row instead of issuing
    ``UPDATE strategy_performance_metrics SET bot_id=NULL``.

    Uses a synchronous in-memory SQLite engine so statements are captured
    deterministically via the ``before_cursor_execute`` event.
    """
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _rec):  # enforce FKs like production does
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)

    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    Session = sessionmaker(bind=engine)
    with Session() as session:
        bot = Bot(name="b", trading_pair="BTC/USDT", strategy="auto_mode",
                  budget=100.0, current_balance=100.0, status=BotStatus.STOPPED)
        session.add(bot)
        session.flush()
        session.add(StrategyPerformanceMetrics(
            bot_id=bot.id, strategy_name="mean_reversion"))
        session.commit()

        statements.clear()  # only care about statements from the delete
        session.delete(bot)
        session.commit()

    joined = " ".join(s.lower() for s in statements)
    # The bug signature must NOT appear.
    assert "update strategy_performance_metrics set bot_id" not in joined, statements
    # The correct behaviour: the child row is DELETEd.
    assert any(
        "delete from strategy_performance_metrics" in s.lower() for s in statements
    ), statements

    # And nothing is left behind.
    with Session() as session:
        assert session.scalar(
            select(func.count()).select_from(StrategyPerformanceMetrics)
        ) == 0
