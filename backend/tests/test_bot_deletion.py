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
    StrategyPerformanceMetrics, TaxLot, RealizedGain,
)


# --- helpers ---------------------------------------------------------------

async def _make_bot(fk_client) -> int:
    resp = await fk_client.post("/api/bots", json={
        "name": "Delete Me",
        "trading_pair": "BTC/USDT",
        "strategy": "auto_mode",
        "budget": 1000.0,
        "is_dry_run": True,
    })
    assert resp.status_code == 201
    return resp.json()["id"]


async def _add_metrics(fk_test_db, bot_id: int) -> None:
    fk_test_db.add(StrategyPerformanceMetrics(
        bot_id=bot_id, strategy_name="mean_reversion",
        recent_pnl_pct=1.0, max_drawdown_pct=2.0, failure_count=1,
    ))
    await fk_test_db.commit()


async def _add_order(fk_test_db, bot_id: int) -> int:
    order = Order(
        bot_id=bot_id, order_type=OrderType.MARKET_BUY, trading_pair="BTC/USDT",
        amount=0.01, price=65000.0, status=OrderStatus.FILLED, strategy_used="auto_mode",
    )
    fk_test_db.add(order)
    await fk_test_db.commit()
    await fk_test_db.refresh(order)
    return order.id


async def _add_position(fk_test_db, bot_id: int) -> None:
    fk_test_db.add(Position(
        bot_id=bot_id, trading_pair="BTC/USDT", side=PositionSide.LONG,
        entry_price=65000.0, current_price=65500.0, amount=0.01,
    ))
    await fk_test_db.commit()


async def _add_trade(fk_test_db, bot_id: int, order_id: int) -> int:
    trade = Trade(
        order_id=order_id, owner_id=str(bot_id), bot_id=bot_id, exchange="simulated",
        trading_pair="BTC/USDT", side=TradeSide.BUY, base_asset="BTC",
        quote_asset="USDT", base_amount=0.01, quote_amount=650.0, price=65000.0,
        executed_at=datetime.utcnow(),
    )
    fk_test_db.add(trade)
    await fk_test_db.commit()
    await fk_test_db.refresh(trade)
    return trade.id


async def _add_accounting(fk_test_db, bot_id: int, trade_id: int) -> None:
    """Add tax_lots + realized_gains referencing a trade — the production case
    that fails with FOREIGN KEY constraint failed once trades are cascade-deleted.

    realized_gains/tax_lots reference trades.id via NOT NULL FKs and have no
    relationship to Bot, so they are not reachable by the ORM bot→trades cascade.
    """
    lot = TaxLot(
        owner_id=str(bot_id), asset="BTC", quantity_acquired=0.01,
        quantity_remaining=0.0, unit_cost=65000.0, total_cost=650.0,
        purchase_trade_id=trade_id, purchase_date=datetime.utcnow(),
    )
    fk_test_db.add(lot)
    await fk_test_db.commit()
    await fk_test_db.refresh(lot)
    fk_test_db.add(RealizedGain(
        owner_id=str(bot_id), asset="BTC", quantity=0.01, proceeds=655.0,
        cost_basis=650.0, gain_loss=5.0, holding_period_days=1, is_long_term=False,
        purchase_trade_id=trade_id, sell_trade_id=trade_id, tax_lot_id=lot.id,
        purchase_date=datetime.utcnow(), sell_date=datetime.utcnow(),
    ))
    await fk_test_db.commit()


async def _add_history(fk_test_db, bot_id: int, order_id: int) -> None:
    """Add the remaining historical child rows (trade, pnl, ledger, rotation,
    alert) plus the accounting grandchildren (tax_lots, realized_gains)."""
    trade_id = await _add_trade(fk_test_db, bot_id, order_id)
    await _add_accounting(fk_test_db, bot_id, trade_id)
    fk_test_db.add_all([
        PnLSnapshot(bot_id=bot_id, total_pnl=12.5),
        WalletLedger(
            owner_id=str(bot_id), bot_id=bot_id, asset="USDT", delta_amount=-650.0,
            reason="buy", related_order_id=order_id, related_trade_id=trade_id,
        ),
        StrategyRotation(bot_id=bot_id, from_strategy="mean_reversion", to_strategy="trend_following"),
        Alert(bot_id=bot_id, alert_type="info", message="hi"),
    ])
    await fk_test_db.commit()


async def _count_accounting(fk_test_db, bot_id: int) -> dict:
    """Count tax_lots/realized_gains belonging to this bot (owner_id == bot id)."""
    return {
        "tax_lots": (await fk_test_db.execute(
            select(func.count()).select_from(TaxLot).where(TaxLot.owner_id == str(bot_id))
        )).scalar(),
        "realized_gains": (await fk_test_db.execute(
            select(func.count()).select_from(RealizedGain).where(RealizedGain.owner_id == str(bot_id))
        )).scalar(),
    }


async def _count_all_children(fk_test_db, bot_id: int) -> dict:
    """Return per-table counts of rows still referencing this bot_id."""
    counts = {}
    for model in (
        Order, Position, Trade, PnLSnapshot, WalletLedger, StrategyRotation,
        Alert, StrategyPerformanceMetrics,
    ):
        n = (await fk_test_db.execute(
            select(func.count()).select_from(model).where(model.bot_id == bot_id)
        )).scalar()
        counts[model.__tablename__] = n
    return counts


# --- API-level deletion tests ---------------------------------------------

@pytest.mark.asyncio
async def test_delete_bot_no_children_succeeds(fk_client):
    bot_id = await _make_bot(fk_client)
    resp = await fk_client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_delete_bot_with_strategy_metrics_succeeds(fk_client, fk_test_db):
    """The exact failing case from the bug report."""
    bot_id = await _make_bot(fk_client)
    await _add_metrics(fk_test_db, bot_id)

    resp = await fk_client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204

    remaining = (await fk_test_db.execute(
        select(func.count()).select_from(StrategyPerformanceMetrics)
        .where(StrategyPerformanceMetrics.bot_id == bot_id)
    )).scalar()
    assert remaining == 0


@pytest.mark.asyncio
async def test_delete_bot_with_positions_succeeds(fk_client, fk_test_db):
    bot_id = await _make_bot(fk_client)
    await _add_position(fk_test_db, bot_id)
    assert (await fk_client.delete(f"/api/bots/{bot_id}")).status_code == 204
    assert (await _count_all_children(fk_test_db, bot_id))["positions"] == 0


@pytest.mark.asyncio
async def test_delete_bot_with_orders_succeeds(fk_client, fk_test_db):
    bot_id = await _make_bot(fk_client)
    await _add_order(fk_test_db, bot_id)
    assert (await fk_client.delete(f"/api/bots/{bot_id}")).status_code == 204
    assert (await _count_all_children(fk_test_db, bot_id))["orders"] == 0


@pytest.mark.asyncio
async def test_delete_bot_with_tax_accounting_records_succeeds(fk_client, fk_test_db):
    """The exact production failure: tax_lots + realized_gains referencing the
    bot's trades caused 'FOREIGN KEY constraint failed' once trades cascade."""
    bot_id = await _make_bot(fk_client)
    order_id = await _add_order(fk_test_db, bot_id)
    trade_id = await _add_trade(fk_test_db, bot_id, order_id)
    await _add_accounting(fk_test_db, bot_id, trade_id)

    before = await _count_accounting(fk_test_db, bot_id)
    assert before == {"tax_lots": 1, "realized_gains": 1}

    resp = await fk_client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204

    after = await _count_accounting(fk_test_db, bot_id)
    assert after == {"tax_lots": 0, "realized_gains": 0}


@pytest.mark.asyncio
async def test_delete_bot_with_all_history_leaves_no_orphans(fk_client, fk_test_db):
    bot_id = await _make_bot(fk_client)
    await _add_metrics(fk_test_db, bot_id)
    order_id = await _add_order(fk_test_db, bot_id)
    await _add_position(fk_test_db, bot_id)
    await _add_history(fk_test_db, bot_id, order_id)

    # Sanity: children exist before deletion.
    before = await _count_all_children(fk_test_db, bot_id)
    assert all(v > 0 for v in before.values()), before
    assert (await _count_accounting(fk_test_db, bot_id)) == {"tax_lots": 1, "realized_gains": 1}

    resp = await fk_client.delete(f"/api/bots/{bot_id}")
    assert resp.status_code == 204

    after = await _count_all_children(fk_test_db, bot_id)
    assert all(v == 0 for v in after.values()), after
    # Accounting grandchildren (tax_lots, realized_gains) are gone too.
    assert (await _count_accounting(fk_test_db, bot_id)) == {"tax_lots": 0, "realized_gains": 0}
    # And the bot itself is gone.
    assert (await fk_test_db.execute(select(Bot).where(Bot.id == bot_id))).scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_repeated_delete_returns_404(fk_client):
    """First delete is 204; a second delete of the same id is a clean 404."""
    bot_id = await _make_bot(fk_client)
    assert (await fk_client.delete(f"/api/bots/{bot_id}")).status_code == 204
    second = await fk_client.delete(f"/api/bots/{bot_id}")
    assert second.status_code == 404
    assert second.json()["detail"]  # valid JSON error body


@pytest.mark.asyncio
async def test_delete_returns_204_with_empty_body(fk_client):
    """The documented contract: 204 No Content, empty body — nothing for the
    frontend to JSON.parse on success."""
    bot_id = await _make_bot(fk_client)
    resp = await fk_client.delete(f"/api/bots/{bot_id}")
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
