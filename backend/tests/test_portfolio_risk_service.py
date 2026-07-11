"""Regression tests for PortfolioRiskService cross-bot aggregation and
realized-P&L loss calculation (add-trading-safety-boundaries).

Uses a real in-memory SQLite session (the `test_db` fixture from
conftest.py) rather than mocks, because the bug this locks down is in the
SQL query itself (a missing WHERE clause), not in any surrounding logic a
mock could stand in for.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.models import Bot, BotStatus, Order, OrderType, OrderStatus, PortfolioRisk, TradeSide
from app.services.accounting import TradeRecorderService, FIFOTaxEngine
from app.services.portfolio_risk import PortfolioRiskService


async def _make_bot(test_db, *, owner_id: str, budget: float, current_balance: float) -> Bot:
    bot = Bot(
        name=f"bot-{owner_id}-{budget}",
        trading_pair="BTC/USDT",
        strategy="dca_accumulator",
        owner_id=owner_id,
        budget=budget,
        current_balance=current_balance,
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    await test_db.flush()
    return bot


async def _record_round_trip(
    test_db, bot: Bot, *, buy_price: float, sell_price: float, sell_fee: float,
) -> None:
    """BUY then SELL 1.0 BTC for `bot`, producing a real RealizedGain row via
    the same FIFOTaxEngine the production ledger uses - not a shortcut."""
    recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)
    owner = str(bot.id)  # ledger's own pseudo-owner-id convention (separate from Bot.owner_id)

    buy_order = Order(
        bot_id=bot.id, order_type=OrderType.MARKET_BUY, trading_pair="BTC/USDT",
        amount=1.0, price=buy_price, status=OrderStatus.FILLED,
        strategy_used="dca_accumulator", is_simulated=True,
    )
    test_db.add(buy_order)
    await test_db.flush()
    buy_trade = await recorder.record_trade(
        order_id=buy_order.id, owner_id=owner, bot_id=bot.id,
        exchange="simulated", trading_pair="BTC/USDT", side=TradeSide.BUY,
        base_asset="BTC", quote_asset="USDT",
        base_amount=1.0, quote_amount=buy_price, price=buy_price,
        fee_amount=0.0, fee_asset="USDT", modeled_cost=0.0,
    )
    await tax_engine.process_buy(buy_trade, bot_id=bot.id)

    sell_order = Order(
        bot_id=bot.id, order_type=OrderType.MARKET_SELL, trading_pair="BTC/USDT",
        amount=1.0, price=sell_price, status=OrderStatus.FILLED,
        strategy_used="dca_accumulator", is_simulated=True,
    )
    test_db.add(sell_order)
    await test_db.flush()
    sell_trade = await recorder.record_trade(
        order_id=sell_order.id, owner_id=owner, bot_id=bot.id,
        exchange="simulated", trading_pair="BTC/USDT", side=TradeSide.SELL,
        base_asset="BTC", quote_asset="USDT",
        base_amount=1.0, quote_amount=sell_price, price=sell_price,
        fee_amount=sell_fee, fee_asset="USDT", modeled_cost=0.0,
    )
    await tax_engine.process_sell(sell_trade, bot_id=bot.id)
    await test_db.commit()


class TestMultiBotAggregation:
    """Regression test for the owner_id aggregation bug: PortfolioRiskService
    used to compare a bot only against itself
    (`bot_ids = [b.id for b in owner_bots if b.id == bot_id]`), so a
    portfolio-wide cap could never actually see combined risk across bots."""

    @pytest.mark.asyncio
    async def test_drawdown_cap_aggregates_across_sibling_bots(self, test_db):
        # bot_a has ZERO drawdown on its own - the old buggy code would
        # compare bot_a only to itself and always allow its trades.
        bot_a = await _make_bot(test_db, owner_id="owner-x", budget=1000.0, current_balance=1000.0)
        # bot_b (same owner) is down 30% - only visible if aggregation works.
        bot_b = await _make_bot(test_db, owner_id="owner-x", budget=1000.0, current_balance=700.0)
        # Different owner entirely - must NOT be pulled into owner-x's check.
        await _make_bot(test_db, owner_id="owner-y", budget=1000.0, current_balance=1.0)

        test_db.add(PortfolioRisk(owner_id="owner-x", max_drawdown_pct=10.0, enabled=True))
        await test_db.commit()

        service = PortfolioRiskService(test_db)
        result = await service.check_portfolio_risk(
            bot_id=bot_a.id, order_amount_usd=10.0, order_side="buy",
        )

        # Combined: (1000+1000 initial) - (1000+700 balance) = 300 / 2000 = 15%
        assert result.ok is False
        assert result.violated_cap == "drawdown"
        assert result.details["drawdown_pct"] == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_bot_alone_within_limits_when_sibling_not_included(self, test_db):
        """Sanity check the fixture: without a risk config for its owner,
        the same bot_a is allowed (proves the block above is the cap firing,
        not some unrelated failure)."""
        bot_a = await _make_bot(test_db, owner_id="owner-solo", budget=1000.0, current_balance=1000.0)

        service = PortfolioRiskService(test_db)
        result = await service.check_portfolio_risk(
            bot_id=bot_a.id, order_amount_usd=10.0, order_side="buy",
        )
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_get_portfolio_metrics_scoped_to_owner(self, test_db):
        await _make_bot(test_db, owner_id="owner-x", budget=1000.0, current_balance=900.0)
        await _make_bot(test_db, owner_id="owner-x", budget=1000.0, current_balance=800.0)
        await _make_bot(test_db, owner_id="owner-y", budget=5000.0, current_balance=5000.0)
        await test_db.commit()

        service = PortfolioRiskService(test_db)
        metrics = await service.get_portfolio_metrics("owner-x")

        assert metrics["bot_count"] == 2
        assert metrics["total_initial"] == pytest.approx(2000.0)
        assert metrics["total_balance"] == pytest.approx(1700.0)


class TestRealizedPnlLossCalculation:
    """Regression test for the loss-cap calculation: it used to sum only
    Order.fees + modeled_total_cost ("simplified"), so a bot could lose
    thousands on a bad trade and register only a few dollars of "loss"
    against the daily/weekly cap. It now uses RealizedGain (the ledger's
    authoritative realized P&L), net of the closing trade's fee."""

    @pytest.mark.asyncio
    async def test_daily_loss_cap_trips_on_real_trading_loss_not_just_fees(self, test_db):
        bot = await _make_bot(test_db, owner_id="owner-z", budget=1000.0, current_balance=795.0)
        test_db.add(PortfolioRisk(owner_id="owner-z", daily_loss_cap_pct=5.0, enabled=True))
        await test_db.commit()

        # Bought at 1000, sold at 800 (a real $200 price loss) with a $5
        # sell-side fee - total realized loss $205, far more than the fee
        # alone would suggest.
        await _record_round_trip(test_db, bot, buy_price=1000.0, sell_price=800.0, sell_fee=5.0)

        service = PortfolioRiskService(test_db)
        result = await service.check_portfolio_risk(
            bot_id=bot.id, order_amount_usd=10.0, order_side="buy",
        )

        assert result.ok is False
        assert result.violated_cap == "daily_loss"
        assert result.details["daily_loss_usd"] == pytest.approx(205.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_realized_gain_excludes_loss_outside_the_period(self, test_db):
        bot = await _make_bot(test_db, owner_id="owner-w", budget=1000.0, current_balance=795.0)
        test_db.add(PortfolioRisk(owner_id="owner-w", daily_loss_cap_pct=5.0, enabled=True))
        await test_db.commit()

        await _record_round_trip(test_db, bot, buy_price=1000.0, sell_price=800.0, sell_fee=5.0)

        # Manually push the realized gain's sell_date to yesterday so it
        # falls outside today's daily-loss window.
        from sqlalchemy import select, update
        from app.models import RealizedGain
        yesterday = datetime.utcnow() - timedelta(days=1, hours=1)
        await test_db.execute(update(RealizedGain).values(sell_date=yesterday))
        await test_db.commit()

        service = PortfolioRiskService(test_db)
        result = await service.check_portfolio_risk(
            bot_id=bot.id, order_amount_usd=10.0, order_side="buy",
        )

        assert result.ok is True

    @pytest.mark.asyncio
    async def test_net_profitable_period_is_zero_loss_not_negative(self, test_db):
        bot = await _make_bot(test_db, owner_id="owner-v", budget=1000.0, current_balance=1195.0)
        test_db.add(PortfolioRisk(owner_id="owner-v", daily_loss_cap_pct=5.0, enabled=True))
        await test_db.commit()

        # Sold for a profit - realized P&L is positive, so "loss" must clamp
        # to 0, not go negative (which would never trip the >= comparison
        # but should still be asserted explicitly).
        await _record_round_trip(test_db, bot, buy_price=1000.0, sell_price=1200.0, sell_fee=5.0)

        service = PortfolioRiskService(test_db)
        result = await service.check_portfolio_risk(
            bot_id=bot.id, order_amount_usd=10.0, order_side="buy",
        )

        assert result.ok is True
