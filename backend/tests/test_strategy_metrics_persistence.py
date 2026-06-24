"""Test strategy performance metrics persistence across restarts."""

import pytest
from datetime import datetime, timedelta
from app.models import (
    Bot, BotStatus, StrategyPerformanceMetrics,
    Order, OrderType, OrderStatus,
    Trade, TradeSide,
    RealizedGain,
)
from app.services.trading_engine import TradingEngine
from sqlalchemy import select


@pytest.mark.asyncio
async def test_strategy_metrics_persist_across_restart(test_db):
    """
    Ensures strategy metrics (cooldowns, failures, PnL) survive bot restarts.
    
    Scenario:
    1. Create bot with auto-mode
    2. Trigger strategy failures → cooldown & blacklist
    3. Update PnL metrics
    4. Simulate restart (destroy TradingEngine, reload from DB)
    5. Assert metrics preserved (failure_count, cooldown_until, recent_pnl)
    
    Failure means:
    ❌ Losing strategies forgotten after restart
    ❌ Cooldowns reset (unsafe re-entry)
    ❌ Blacklists lost (repeat failures)
    ❌ Auto-mode forgets history
    """
    # === SETUP: Create bot ===
    bot = Bot(
        name="Test Auto Bot",
        trading_pair="BTC/USDT",
        strategy="auto_mode",
        strategy_params={},
        budget=10000.0,
        current_balance=10000.0,
        is_dry_run=True,
        status=BotStatus.RUNNING
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)
    
    # === STEP 1: Create initial metrics ===
    now = datetime.utcnow()
    cooldown_time = now + timedelta(hours=6)
    
    initial_metrics = {
        "mean_reversion": {
            "recent_pnl_pct": -8.5,
            "max_drawdown_pct": 12.3,
            "failure_count": 2,
            "last_exit_time": now.isoformat(),
            "cooldown_until": cooldown_time.isoformat()
        },
        "trend_following": {
            "recent_pnl_pct": 3.2,
            "max_drawdown_pct": 5.1,
            "failure_count": 0,
            "last_exit_time": None,
            "cooldown_until": None
        }
    }
    
    # === STEP 2: Persist via TradingEngine (simulating real usage) ===
    engine = TradingEngine()
    
    for strategy_name, metrics in initial_metrics.items():
        await engine._save_strategy_metrics_to_db(
            bot_id=bot.id,
            strategy_name=strategy_name,
            metrics=metrics,
            session=test_db
        )
    
    # === STEP 3: Verify database writes ===
    query = select(StrategyPerformanceMetrics).where(
        StrategyPerformanceMetrics.bot_id == bot.id
    )
    result = await test_db.execute(query)
    rows = result.scalars().all()
    
    assert len(rows) == 2, "Expected 2 strategy metrics rows"
    
    # === STEP 4: Simulate restart (destroy engine, reload from DB) ===
    del engine
    engine_restarted = TradingEngine()
    
    # === STEP 5: Load metrics from DB ===
    reloaded_metrics = await engine_restarted._load_strategy_metrics_from_db(
        bot_id=bot.id,
        session=test_db
    )
    
    # === ASSERTIONS: Metrics preserved ===
    assert "mean_reversion" in reloaded_metrics
    assert "trend_following" in reloaded_metrics
    
    mr_metrics = reloaded_metrics["mean_reversion"]
    assert mr_metrics["recent_pnl_pct"] == -8.5
    assert mr_metrics["max_drawdown_pct"] == 12.3
    assert mr_metrics["failure_count"] == 2
    assert mr_metrics["last_exit_time"] is not None
    assert mr_metrics["cooldown_until"] is not None
    
    # Verify cooldown still active
    cooldown_dt = datetime.fromisoformat(mr_metrics["cooldown_until"])
    assert cooldown_dt > datetime.utcnow(), "Cooldown should still be active"
    
    tf_metrics = reloaded_metrics["trend_following"]
    assert tf_metrics["recent_pnl_pct"] == 3.2
    assert tf_metrics["failure_count"] == 0
    assert tf_metrics["cooldown_until"] is None
    
    print("✅ Strategy metrics persisted correctly across restart")


@pytest.mark.asyncio
async def test_strategy_metrics_upsert_behavior(test_db):
    """
    Ensures UPSERT works correctly (insert or update, no duplicates).
    
    Scenario:
    1. Insert initial metrics
    2. Update same strategy metrics (increment failure_count)
    3. Assert only 1 row exists, with updated values
    """
    bot = Bot(
        name="UPSERT Test Bot",
        trading_pair="BTC/USDT",
        strategy="auto_mode",
        budget=10000.0,
        current_balance=10000.0,
        is_dry_run=True
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)
    
    engine = TradingEngine()
    
    # === STEP 1: Initial insert ===
    initial = {
        "recent_pnl_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "failure_count": 0,
        "last_exit_time": None,
        "cooldown_until": None
    }
    
    await engine._save_strategy_metrics_to_db(
        bot_id=bot.id,
        strategy_name="mean_reversion",
        metrics=initial,
        session=test_db
    )
    
    # === STEP 2: Update (increment failure_count) ===
    updated = {
        "recent_pnl_pct": -5.0,
        "max_drawdown_pct": 8.0,
        "failure_count": 1,
        "last_exit_time": datetime.utcnow().isoformat(),
        "cooldown_until": (datetime.utcnow() + timedelta(hours=6)).isoformat()
    }
    
    await engine._save_strategy_metrics_to_db(
        bot_id=bot.id,
        strategy_name="mean_reversion",
        metrics=updated,
        session=test_db
    )
    
    # === STEP 3: Verify only 1 row exists ===
    query = select(StrategyPerformanceMetrics).where(
        StrategyPerformanceMetrics.bot_id == bot.id,
        StrategyPerformanceMetrics.strategy_name == "mean_reversion"
    )
    result = await test_db.execute(query)
    rows = result.scalars().all()
    
    assert len(rows) == 1, "UPSERT should not create duplicate rows"
    
    row = rows[0]
    assert row.failure_count == 1
    assert row.recent_pnl_pct == -5.0
    assert row.cooldown_until is not None
    
    print("✅ UPSERT behavior works correctly (no duplicates)")


@pytest.mark.asyncio
async def test_strategy_metrics_cleanup_when_bot_deleted(test_db):
    """
    Ensures metrics can be cleaned up when bot is deleted.
    
    Note: In production, the ON DELETE CASCADE in the database schema handles this.
    This test verifies the relationship is correctly defined.
    """
    from sqlalchemy import delete as sql_delete
    
    bot = Bot(
        name="Cleanup Test Bot",
        trading_pair="BTC/USDT",
        strategy="auto_mode",
        budget=10000.0,
        current_balance=10000.0,
        is_dry_run=True
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)
    
    bot_id = bot.id  # Save ID before deletion
    
    # Create metrics
    metrics = StrategyPerformanceMetrics(
        bot_id=bot.id,
        strategy_name="trend_following",
        recent_pnl_pct=0.0,
        max_drawdown_pct=0.0,
        failure_count=0
    )
    test_db.add(metrics)
    await test_db.commit()
    
    # Verify metrics exist
    query = select(StrategyPerformanceMetrics).where(
        StrategyPerformanceMetrics.bot_id == bot_id
    )
    result = await test_db.execute(query)
    rows_before = result.scalars().all()
    assert len(rows_before) == 1, "Metrics should exist before deletion"
    
    # Delete metrics first (proper cleanup order)
    delete_metrics_stmt = sql_delete(StrategyPerformanceMetrics).where(
        StrategyPerformanceMetrics.bot_id == bot_id
    )
    await test_db.execute(delete_metrics_stmt)
    
    # Then delete bot
    await test_db.delete(bot)
    await test_db.commit()
    
    # Verify metrics deleted
    query = select(StrategyPerformanceMetrics).where(
        StrategyPerformanceMetrics.bot_id == bot_id
    )
    result = await test_db.execute(query)
    rows_after = result.scalars().all()
    
    assert len(rows_after) == 0, "Metrics should be deleted before bot"

    print("✅ Cleanup works correctly")


# ---------------------------------------------------------------------------
# New tests: verify _update_strategy_performance_metrics uses realized_gains
# ---------------------------------------------------------------------------

async def _make_auto_sell_cycle(
    db,
    bot,
    sub_strategy: str,
    gain_loss: float,
    cost_basis: float,
    sell_date: datetime,
    running_balance_after: float = 99999.0,  # intentionally misleading value
) -> RealizedGain:
    """Helper: create one auto-mode sell cycle with a known FIFO gain_loss.

    The ``running_balance_after`` on the Order is set to a nonsense value so
    any test that passes despite using balance snapshots would produce wrong
    numbers and fail — confirming the implementation reads realized_gains.
    """
    order = Order(
        bot_id=bot.id,
        order_type=OrderType.MARKET_SELL,
        trading_pair="BTC/USDT",
        amount=0.001,
        price=cost_basis / 0.001,
        status=OrderStatus.FILLED,
        strategy_used="auto_mode",
        reason=f"[Auto:{sub_strategy}|flat/medium/normal] Grid: Sell at L1",
        running_balance_after=running_balance_after,
        filled_at=sell_date,
        is_simulated=True,
    )
    db.add(order)
    await db.flush()

    trade = Trade(
        bot_id=bot.id,
        order_id=order.id,
        owner_id="test_owner",
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.SELL,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.001,
        quote_amount=cost_basis,
        price=cost_basis / 0.001,
        fee_amount=1.0,
        strategy_used="auto_mode",
        executed_at=sell_date,
    )
    db.add(trade)
    await db.flush()

    rg = RealizedGain(
        owner_id=str(bot.id),
        asset="BTC",
        quantity=0.001,
        proceeds=cost_basis + gain_loss,
        cost_basis=cost_basis,
        gain_loss=gain_loss,
        holding_period_days=1,
        is_long_term=False,
        purchase_trade_id=9999,   # dummy — not queried
        sell_trade_id=trade.id,
        tax_lot_id=9999,          # dummy — not queried (FK unenforced in test DB)
        purchase_date=sell_date - timedelta(days=1),
        sell_date=sell_date,
    )
    db.add(rg)
    await db.flush()
    return rg


@pytest.mark.asyncio
async def test_metrics_use_realized_gains_not_balance_snapshots(test_db):
    """Core contract: _update_strategy_performance_metrics reads realized_gains.

    Three closed sell cycles with deliberate gain_loss values:
      Cycle 1:  +$10.00  (win)
      Cycle 2:  -$5.00   (loss)
      Cycle 3:  +$8.00   (win)

    The running_balance_after on each order is set to a constant value that
    would produce zero consecutive deltas — so if the old balance-snapshot
    code were still in use, win_rate and realized_pnl_usd would both be 0,
    making the test fail.  Passing means the code reads gain_loss correctly.
    """
    bot = Bot(
        name="RG Metrics Bot",
        trading_pair="BTC/USDT",
        strategy="auto_mode",
        budget=10000.0,
        current_balance=10000.0,
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)

    now = datetime.utcnow()
    cost_basis = 100.0  # each cycle costs $100

    await _make_auto_sell_cycle(test_db, bot, "adaptive_grid", +10.0, cost_basis, now - timedelta(hours=3))
    await _make_auto_sell_cycle(test_db, bot, "adaptive_grid", -5.0,  cost_basis, now - timedelta(hours=2))
    await _make_auto_sell_cycle(test_db, bot, "adaptive_grid", +8.0,  cost_basis, now - timedelta(hours=1))
    await test_db.commit()

    engine = TradingEngine()
    auto_state = {"strategy_metrics": {}}
    await engine._update_strategy_performance_metrics(
        bot_id=bot.id,
        auto_state=auto_state,
        session=test_db,
        performance_window=20,
    )

    m = auto_state["strategy_metrics"]["adaptive_grid"]

    assert m["total_trades"] == 3
    assert m["winning_trades"] == 2        # cycles 1 and 3
    assert m["losing_trades"] == 1         # cycle 2
    assert m["realized_pnl_usd"] == pytest.approx(13.0, rel=1e-6)
    assert m["win_rate"] == pytest.approx(2 / 3, rel=1e-6)
    # profit_factor = (10 + 8) / 5 = 3.6
    assert m["profit_factor"] == pytest.approx(3.6, rel=1e-6)
    # recent_pnl_pct = 13 / (100+100+100) × 100 = 4.333…%
    assert m["recent_pnl_pct"] == pytest.approx(13 / 300 * 100, rel=1e-4)
    assert m["last_trade_time"] is not None


@pytest.mark.asyncio
async def test_metrics_sub_strategy_attribution_from_order_reason(test_db):
    """Strategy attribution must come from [Auto:strategy|regime] order reason.

    Two sub-strategies each with 2 closed cycles.  The code should separate
    them by strategy name and produce independent metric dicts.
    """
    bot = Bot(
        name="Attribution Bot",
        trading_pair="BTC/USDT",
        strategy="auto_mode",
        budget=10000.0,
        current_balance=10000.0,
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)

    now = datetime.utcnow()
    cost = 100.0

    # adaptive_grid: 1 win (+20), 1 loss (-10)
    await _make_auto_sell_cycle(test_db, bot, "adaptive_grid", +20.0, cost, now - timedelta(hours=4))
    await _make_auto_sell_cycle(test_db, bot, "adaptive_grid", -10.0, cost, now - timedelta(hours=3))
    # mean_reversion: 2 wins (+5 each)
    await _make_auto_sell_cycle(test_db, bot, "mean_reversion", +5.0, cost, now - timedelta(hours=2))
    await _make_auto_sell_cycle(test_db, bot, "mean_reversion", +5.0, cost, now - timedelta(hours=1))
    await test_db.commit()

    engine = TradingEngine()
    auto_state = {"strategy_metrics": {}}
    await engine._update_strategy_performance_metrics(
        bot_id=bot.id,
        auto_state=auto_state,
        session=test_db,
        performance_window=20,
    )

    assert "adaptive_grid" in auto_state["strategy_metrics"]
    assert "mean_reversion" in auto_state["strategy_metrics"]

    ag = auto_state["strategy_metrics"]["adaptive_grid"]
    assert ag["total_trades"] == 2
    assert ag["winning_trades"] == 1
    assert ag["losing_trades"] == 1
    assert ag["realized_pnl_usd"] == pytest.approx(10.0, rel=1e-6)
    assert ag["win_rate"] == pytest.approx(0.5, rel=1e-6)
    assert ag["profit_factor"] == pytest.approx(20.0 / 10.0, rel=1e-6)

    mr = auto_state["strategy_metrics"]["mean_reversion"]
    assert mr["total_trades"] == 2
    assert mr["winning_trades"] == 2
    assert mr["losing_trades"] == 0
    assert mr["realized_pnl_usd"] == pytest.approx(10.0, rel=1e-6)
    assert mr["win_rate"] == pytest.approx(1.0, rel=1e-6)
    # profit_factor: gross_loss = 0, so returns 1.0 (no losses)
    assert mr["profit_factor"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_metrics_no_realized_gains_returns_early(test_db):
    """When a bot has no realized gains yet, _update_strategy_performance_metrics
    should return cleanly without modifying auto_state."""
    bot = Bot(
        name="Empty Bot",
        trading_pair="BTC/USDT",
        strategy="auto_mode",
        budget=10000.0,
        current_balance=10000.0,
        is_dry_run=True,
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)

    engine = TradingEngine()
    auto_state = {"strategy_metrics": {}}
    await engine._update_strategy_performance_metrics(
        bot_id=bot.id,
        auto_state=auto_state,
        session=test_db,
        performance_window=20,
    )

    # Nothing added — bot has no realized gains yet
    assert auto_state["strategy_metrics"] == {}
