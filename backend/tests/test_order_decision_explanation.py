"""Regression tests proving `_execute_trade` (and the TWAP/VWAP execution
paths) persist a summarized decision explanation onto the `Order` row
(add-strategy-decision-framework, Phase 0.6 - closes the Pillar 8
persistence gap: a historical trade should be explainable from the DB, not
only from the in-memory DiagnosticsStore).

Uses the same `_execute_trade` mocking pattern as
tests/test_order_lifecycle.py's H3 suite (mocked session,
`_finalize_filled_order` patched out) so this test focuses purely on the
new persistence behavior, not the full accounting pipeline.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.exchange import ExchangeOrder
from app.services.strategy_explain import ExplanationBuilder
from app.services.trading_engine import TradeSignal, TradingEngine


def _closed_order(oid="S1", side="buy", amount=0.01, price=50000.0):
    cost = amount * price
    return ExchangeOrder(
        id=oid, symbol="BTC/USDT", side=side, type="market",
        amount=amount, price=price, cost=cost, fee=cost * 0.001,
        fee_currency="USDT", status="closed", timestamp=datetime.utcnow(),
        filled=amount, remaining=0.0,
    )


def _exec_bot():
    return SimpleNamespace(
        id=1, name="b", trading_pair="BTC/USDT", strategy="trend_following",
        is_dry_run=True, current_balance=1000.0, exchange_fee=0.0,
    )


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
            return_value=SimpleNamespace(ok=True, action="allow", adjusted_amount=None, violated_cap=None, details=None)
        )
        SC.return_value.check_capacity_for_trade = AsyncMock(
            return_value=SimpleNamespace(ok=True, adjusted_amount=None, reason=None)
        )
        return await engine._execute_trade(bot, exchange, signal, 100.0, session)


class TestDecisionExplanationPersistedOnOrder:
    @pytest.mark.asyncio
    async def test_explanation_persisted_when_builder_present(self):
        engine = TradingEngine()
        bot = _exec_bot()

        builder = ExplanationBuilder("trend_following")
        builder.check("ema_cross", current=1.0, required="> 0", passed=True)
        builder.metric("ema_fast", 100.5)
        builder.state("LONG_OPEN")
        engine._explanations[bot.id] = builder

        exchange = Mock()
        exchange.place_market_order = AsyncMock(return_value=_closed_order())
        signal = TradeSignal(action="buy", amount=50.0, order_type="market", reason="EMA cross", is_accumulation=True)

        order = await _call_execute(engine, bot, signal, _mock_session(), exchange)

        assert order is not None
        assert order.decision_explanation is not None
        assert order.decision_explanation["strategy"] == "trend_following"
        assert order.decision_explanation["state"] == "LONG_OPEN"
        assert any(c["name"] == "ema_cross" for c in order.decision_explanation["checks"])
        assert order.decision_explanation["metrics"]["ema_fast"] == 100.5

    @pytest.mark.asyncio
    async def test_edge_management_category_extracted_when_present(self):
        engine = TradingEngine()
        bot = _exec_bot()

        builder = ExplanationBuilder("trend_following")
        builder.metric("edge_status_category", "B")
        engine._explanations[bot.id] = builder

        exchange = Mock()
        exchange.place_market_order = AsyncMock(return_value=_closed_order())
        signal = TradeSignal(action="buy", amount=50.0, order_type="market", reason="test", is_accumulation=True)

        order = await _call_execute(engine, bot, signal, _mock_session(), exchange)

        assert order.edge_management_category == "B"

    @pytest.mark.asyncio
    async def test_no_builder_leaves_explanation_null_not_error(self):
        """No strategy explanation was recorded for this bot this cycle
        (e.g. a signal constructed directly, bypassing _execute_strategy) -
        persistence must degrade to None, never raise."""
        engine = TradingEngine()
        bot = _exec_bot()
        # Deliberately do NOT populate engine._explanations[bot.id].

        exchange = Mock()
        exchange.place_market_order = AsyncMock(return_value=_closed_order())
        signal = TradeSignal(action="buy", amount=50.0, order_type="market", reason="test", is_accumulation=True)

        order = await _call_execute(engine, bot, signal, _mock_session(), exchange)

        assert order is not None
        assert order.decision_explanation is None
        assert order.edge_management_category is None

    @pytest.mark.asyncio
    async def test_persistence_never_blocks_trade_execution_on_error(self):
        """Observability must never affect a trading decision - a broken
        explanation builder must not prevent the order from executing."""
        engine = TradingEngine()
        bot = _exec_bot()

        class ExplodingBuilder:
            def to_dict(self):
                raise RuntimeError("boom")

        engine._explanations[bot.id] = ExplodingBuilder()

        exchange = Mock()
        exchange.place_market_order = AsyncMock(return_value=_closed_order())
        signal = TradeSignal(action="buy", amount=50.0, order_type="market", reason="test", is_accumulation=True)

        order = await _call_execute(engine, bot, signal, _mock_session(), exchange)

        assert order is not None
        assert order.decision_explanation is None

    def test_helper_is_pure_and_deterministic(self):
        engine = TradingEngine()
        builder = ExplanationBuilder("mean_reversion")
        builder.metric("bb_width", 0.02)
        engine._explanations[42] = builder

        r1 = engine._decision_explanation_for_order(42)
        r2 = engine._decision_explanation_for_order(42)
        assert r1[0]["strategy"] == r2[0]["strategy"] == "mean_reversion"
        assert r1[1] == r2[1] is None

    def test_unknown_bot_id_returns_none_none(self):
        engine = TradingEngine()
        assert engine._decision_explanation_for_order(99999) == (None, None)
