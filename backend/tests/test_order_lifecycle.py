"""Order-lifecycle hardening tests for C2, H2, H3.

Evidence that exchange/database desynchronization is eliminated and that orders
and positions are always recoverable after a failure:

- H3: position-closing sells are never blocked by the $10 software minimum.
- H2: locally-PENDING orders are resolved against the exchange (filled ->
      finalized with a real Trade + Position; canceled -> CANCELLED; open -> kept).
- C2: exchange fills with no local record are imported and finalized on recovery,
      idempotently. _recover_bot_orders performs both on startup.

The H2/C2 tests run through the real accounting/finalize path against a real
in-memory database, so a recovered order produces a real Trade and Position.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import select

import ccxt.async_support as ccxt

from app.models import Bot, BotStatus, Order, OrderType, OrderStatus, Position, Trade
from app.services.exchange import (
    ExchangeOrder,
    ExchangeService,
    OrderSide,
    SimulatedExchangeService,
)
from app.services.trading_engine import TradingEngine, TradeSignal


# ============================================================================
# Helpers
# ============================================================================


def _closed_order(oid, side="buy", amount=0.01, price=50000.0, status="closed"):
    cost = amount * price
    return ExchangeOrder(
        id=oid, symbol="BTC/USDT", side=side, type="market",
        amount=amount, price=price, cost=cost, fee=cost * 0.001,
        fee_currency="USDT", status=status, timestamp=datetime.utcnow(),
        filled=amount if status in ("closed", "filled", "partial") else 0.0,
        remaining=0.0,
    )


class RecoveryExchange:
    """Exchange stand-in for recovery tests."""

    def __init__(self, by_id=None, recent=None):
        self._by_id = by_id or {}
        self._recent = recent or []

    async def get_order(self, order_id, symbol):
        return self._by_id.get(order_id)

    async def get_recent_orders(self, symbol, limit=50):
        return [o for o in self._recent if o.symbol == symbol][-limit:]


async def _make_db_bot(test_db, strategy="dca_accumulator"):
    bot = Bot(
        name="b", trading_pair="BTC/USDT", strategy=strategy, strategy_params={},
        budget=100000.0, current_balance=100000.0, is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    await test_db.flush()
    return bot


async def _positions(test_db, bot_id):
    return (
        await test_db.execute(select(Position).where(Position.bot_id == bot_id))
    ).scalars().all()


async def _trades(test_db, bot_id):
    return (
        await test_db.execute(select(Trade).where(Trade.bot_id == bot_id))
    ).scalars().all()


# ============================================================================
# H3 - Position-closing sells are never blocked by the $10 minimum
# ============================================================================


def _mock_session():
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


async def _call_execute(engine, bot, signal, session, exchange, position=None):
    res = Mock()
    res.scalar_one_or_none = Mock(return_value=position)
    session.execute = AsyncMock(return_value=res)
    with patch.object(engine, "_finalize_filled_order", AsyncMock(return_value=True)), \
         patch("app.services.trading_engine.PortfolioRiskService") as PR, \
         patch("app.services.trading_engine.StrategyCapacityService") as SC, \
         patch("app.services.trading_engine.CSVExportService"):
        PR.return_value.check_portfolio_risk = AsyncMock(
            return_value=SimpleNamespace(
                ok=True, action="allow", adjusted_amount=None,
                violated_cap=None, details=None,
            )
        )
        SC.return_value.check_capacity_for_trade = AsyncMock(
            return_value=SimpleNamespace(ok=True, adjusted_amount=None, reason=None)
        )
        return await engine._execute_trade(bot, exchange, signal, 100.0, session)


def _exec_bot():
    return SimpleNamespace(
        id=1, name="b", trading_pair="BTC/USDT", strategy="trend_following",
        is_dry_run=True, current_balance=1000.0, exchange_fee=0.0,
    )


class TestExitMinimum:
    @pytest.mark.asyncio
    async def test_small_sell_is_not_blocked(self):
        engine = TradingEngine()
        bot = _exec_bot()
        exchange = Mock()
        exchange.place_market_order = AsyncMock(
            return_value=_closed_order("S1", side="sell", amount=0.05, price=100.0)
        )
        # $5 sell (< $10 floor) closing a 1.0 position must be allowed.
        signal = TradeSignal(action="sell", amount=5.0, order_type="market", reason="exit")
        order = await _call_execute(
            engine, bot, signal, _mock_session(), exchange,
            position=SimpleNamespace(amount=1.0),
        )
        assert order is not None
        exchange.place_market_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_small_buy_is_still_rejected(self):
        engine = TradingEngine()
        bot = _exec_bot()
        exchange = Mock()
        exchange.place_market_order = AsyncMock()
        signal = TradeSignal(action="buy", amount=5.0, order_type="market", reason="open")
        order = await _call_execute(engine, bot, signal, _mock_session(), exchange)
        assert order is None
        exchange.place_market_order.assert_not_awaited()


# ============================================================================
# H2 - Resolve locally-PENDING orders against the exchange
# ============================================================================


def _pending_order(bot_id, oid="X1", side="buy", amount=0.01, price=50000.0):
    return Order(
        bot_id=bot_id, exchange_order_id=oid,
        order_type=OrderType.MARKET_BUY if side == "buy" else OrderType.MARKET_SELL,
        trading_pair="BTC/USDT", amount=amount, price=price,
        status=OrderStatus.PENDING, strategy_used="dca_accumulator", is_simulated=True,
    )


class TestResolvePending:
    @pytest.mark.asyncio
    async def test_filled_pending_is_finalized(self, test_db):
        bot = await _make_db_bot(test_db)
        order = _pending_order(bot.id)
        test_db.add(order)
        await test_db.flush()

        exchange = RecoveryExchange(by_id={"X1": _closed_order("X1", amount=0.01)})
        engine = TradingEngine()
        resolved = await engine._resolve_pending_orders(bot.id, exchange, test_db)

        assert resolved == 1
        refreshed = (
            await test_db.execute(select(Order).where(Order.id == order.id))
        ).scalar_one()
        assert refreshed.status == OrderStatus.FILLED
        # A real Trade and Position were produced by the finalize path.
        positions = await _positions(test_db, bot.id)
        assert len(positions) == 1
        assert positions[0].amount == pytest.approx(0.01)
        assert len(await _trades(test_db, bot.id)) == 1

    @pytest.mark.asyncio
    async def test_canceled_pending_is_marked_cancelled(self, test_db):
        bot = await _make_db_bot(test_db)
        order = _pending_order(bot.id, oid="C1")
        test_db.add(order)
        await test_db.flush()

        exchange = RecoveryExchange(
            by_id={"C1": _closed_order("C1", status="canceled")}
        )
        engine = TradingEngine()
        resolved = await engine._resolve_pending_orders(bot.id, exchange, test_db)

        assert resolved == 1
        refreshed = (
            await test_db.execute(select(Order).where(Order.id == order.id))
        ).scalar_one()
        assert refreshed.status == OrderStatus.CANCELLED
        assert await _positions(test_db, bot.id) == []

    @pytest.mark.asyncio
    async def test_still_open_pending_is_left_untouched(self, test_db):
        bot = await _make_db_bot(test_db)
        order = _pending_order(bot.id, oid="O1")
        test_db.add(order)
        await test_db.flush()

        exchange = RecoveryExchange(by_id={"O1": _closed_order("O1", status="open")})
        engine = TradingEngine()
        resolved = await engine._resolve_pending_orders(bot.id, exchange, test_db)

        assert resolved == 0
        refreshed = (
            await test_db.execute(select(Order).where(Order.id == order.id))
        ).scalar_one()
        assert refreshed.status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_unknown_order_left_pending(self, test_db):
        # Exchange has no record (get_order -> None): do not touch the order.
        bot = await _make_db_bot(test_db)
        order = _pending_order(bot.id, oid="U1")
        test_db.add(order)
        await test_db.flush()

        engine = TradingEngine()
        resolved = await engine._resolve_pending_orders(bot.id, RecoveryExchange(), test_db)

        assert resolved == 0
        refreshed = (
            await test_db.execute(select(Order).where(Order.id == order.id))
        ).scalar_one()
        assert refreshed.status == OrderStatus.PENDING


# ============================================================================
# C2 - Import exchange fills missing from the database
# ============================================================================


class TestReconcileMissingFills:
    @pytest.mark.asyncio
    async def test_untracked_fill_is_imported(self, test_db):
        bot = await _make_db_bot(test_db)
        # Exchange knows a fill the DB never recorded (crash between fill/commit).
        exchange = RecoveryExchange(recent=[_closed_order("M1", amount=0.02)])
        engine = TradingEngine()

        imported = await engine._reconcile_orders_with_exchange(bot, exchange, test_db)

        assert imported == 1
        orders = (
            await test_db.execute(select(Order).where(Order.bot_id == bot.id))
        ).scalars().all()
        assert len(orders) == 1
        assert orders[0].exchange_order_id == "M1"
        assert orders[0].status == OrderStatus.FILLED
        positions = await _positions(test_db, bot.id)
        assert len(positions) == 1
        assert positions[0].amount == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_import_is_idempotent(self, test_db):
        bot = await _make_db_bot(test_db)
        exchange = RecoveryExchange(recent=[_closed_order("M2", amount=0.02)])
        engine = TradingEngine()

        first = await engine._reconcile_orders_with_exchange(bot, exchange, test_db)
        second = await engine._reconcile_orders_with_exchange(bot, exchange, test_db)

        assert first == 1
        assert second == 0  # already known, not duplicated
        orders = (
            await test_db.execute(select(Order).where(Order.bot_id == bot.id))
        ).scalars().all()
        assert len(orders) == 1

    @pytest.mark.asyncio
    async def test_known_fill_not_reimported(self, test_db):
        bot = await _make_db_bot(test_db)
        # A local order already records this exchange id.
        existing = Order(
            bot_id=bot.id, exchange_order_id="K1", order_type=OrderType.MARKET_BUY,
            trading_pair="BTC/USDT", amount=0.01, price=50000.0,
            status=OrderStatus.FILLED, strategy_used="dca_accumulator", is_simulated=True,
        )
        test_db.add(existing)
        await test_db.flush()

        exchange = RecoveryExchange(recent=[_closed_order("K1", amount=0.01)])
        engine = TradingEngine()
        imported = await engine._reconcile_orders_with_exchange(bot, exchange, test_db)

        assert imported == 0

    @pytest.mark.asyncio
    async def test_recover_bot_orders_does_both(self, test_db):
        bot = await _make_db_bot(test_db)
        pending = _pending_order(bot.id, oid="P1")
        test_db.add(pending)
        await test_db.flush()

        exchange = RecoveryExchange(
            by_id={"P1": _closed_order("P1", amount=0.01)},
            recent=[_closed_order("P1", amount=0.01), _closed_order("M3", amount=0.03)],
        )
        engine = TradingEngine()
        recovered = await engine._recover_bot_orders(bot, exchange, test_db)

        # P1 resolved from pending + M3 imported = 2 reconciled.
        assert recovered == 2
        orders = (
            await test_db.execute(select(Order).where(Order.bot_id == bot.id))
        ).scalars().all()
        statuses = {o.exchange_order_id: o.status for o in orders}
        assert statuses["P1"] == OrderStatus.FILLED
        assert statuses["M3"] == OrderStatus.FILLED
        # Two positions worth of base were recorded (0.01 + 0.03).
        total = sum(p.amount for p in await _positions(test_db, bot.id))
        assert total == pytest.approx(0.04)


# ============================================================================
# Exchange read-only recent-order access (used by recovery)
# ============================================================================


class TestExchangeRecentOrders:
    @pytest.mark.asyncio
    async def test_real_exchange_parses_closed_orders(self):
        rows = [
            {"id": "A1", "symbol": "BTC/USDT", "side": "buy", "type": "market",
             "amount": 0.01, "price": 50000.0, "cost": 500.0, "status": "closed",
             "filled": 0.01, "remaining": 0.0, "fee": {"cost": 0.5, "currency": "USDT"},
             "timestamp": int(datetime(2026, 1, 1).timestamp() * 1000)},
        ]
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.has = {"fetchClosedOrders": True}
        mock_exchange.fetch_closed_orders = AsyncMock(return_value=rows)
        with patch("app.services.exchange.ccxt") as mock_ccxt:
            mock_ccxt.mexc = Mock(return_value=mock_exchange)
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                orders = await service.get_recent_orders("BTC/USDT", limit=10)

        assert len(orders) == 1
        assert orders[0].id == "A1"
        assert orders[0].status == "closed"
        mock_exchange.fetch_closed_orders.assert_awaited_once_with("BTC/USDT", None, 10)

    @pytest.mark.asyncio
    async def test_real_exchange_unsupported_returns_empty(self):
        mock_exchange = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.has = {}  # neither fetchClosedOrders nor fetchOrders
        with patch("app.services.exchange.ccxt") as mock_ccxt:
            mock_ccxt.mexc = Mock(return_value=mock_exchange)
            with patch.object(ExchangeService, "_load_config", return_value={}):
                service = ExchangeService(exchange_id="mexc")
                await service.connect()
                assert await service.get_recent_orders("BTC/USDT") == []

    @pytest.mark.asyncio
    async def test_simulator_returns_filled_orders_by_symbol(self):
        sim = SimulatedExchangeService(initial_balance=10000.0)
        sim._connected = True
        client = AsyncMock()
        client.fetch_ticker = AsyncMock(return_value={
            "symbol": "BTC/USDT", "bid": 49999.0, "ask": 50001.0, "last": 50000.0,
            "baseVolume": 1.0, "timestamp": int(datetime(2026, 1, 1).timestamp() * 1000),
        })
        sim.exchange = client

        placed = await sim.place_market_order("BTC/USDT", OrderSide.BUY, 0.01)
        recent = await sim.get_recent_orders("BTC/USDT")

        assert any(o.id == placed.id for o in recent)
        assert await sim.get_recent_orders("ETH/USDT") == []
