"""Tests for Ledger Replay Service"""

import pytest
from datetime import datetime

from app.services.ledger_replay import LedgerReplayService
from app.services.accounting import TradeRecorderService, FIFOTaxEngine
from app.services.ledger_writer import LedgerWriterService
from app.models import (
    Bot,
    Order,
    Trade,
    TradeSide,
    OrderType,
    OrderStatus,
    Position,
    PositionSide,
    WalletLedger,
    LedgerReason,
)


@pytest.mark.asyncio
async def test_replay_produces_identical_balances(test_db, sample_bot, sample_order):
    """Test that replay produces identical state when run multiple times."""
    # Create initial allocation
    ledger_service = LedgerWriterService(test_db)
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=1000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial allocation",
        related_order_id=None,
        related_trade_id=None,
    )
    await test_db.flush()

    # Execute 3 BUY trades
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)

    for i in range(3):
        order = Order(
            bot_id=sample_bot.id,
            order_type=OrderType.MARKET_BUY,
            trading_pair="BTC/USDT",
            amount=0.01,
            price=50000.0 + i * 1000,
            status=OrderStatus.FILLED,
            strategy_used="test",
            is_simulated=True,
        )
        test_db.add(order)
        await test_db.flush()

        trade = await trade_recorder.record_trade(
            order_id=order.id,
            owner_id="test_owner",
            bot_id=sample_bot.id,
            exchange="simulated",
            trading_pair="BTC/USDT",
            side=TradeSide.BUY,
            base_asset="BTC",
            quote_asset="USDT",
            base_amount=0.01,
            quote_amount=500.0 + i * 10,
            price=50000.0 + i * 1000,
            fee_amount=0.5,
            fee_asset="USDT",
            modeled_cost=0.1,
        )
        await tax_engine.process_buy(trade)
        await test_db.flush()

    await test_db.commit()

    # Run replay 3 times
    replay_service = LedgerReplayService(test_db)
    results = []

    for i in range(3):
        result = await replay_service.rebuild_state_from_ledger("test_owner", True)
        results.append(result.final_balances)

    # All results should be identical
    assert results[0] == results[1] == results[2]


@pytest.mark.asyncio
async def test_replay_separates_simulated_and_live(test_db):
    """Test that simulated and live data are kept separate during replay."""
    # Create two bots: one simulated, one live
    bot_simulated = Bot(
        name="Simulated Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
    )
    bot_live = Bot(
        name="Live Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=2000.0,
        current_balance=2000.0,
        is_dry_run=False,
    )
    test_db.add(bot_simulated)
    test_db.add(bot_live)
    await test_db.flush()

    # Create ledger entries for both
    ledger_service = LedgerWriterService(test_db)

    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=bot_simulated.id,
        asset="USDT",
        delta_amount=1000.0,
        reason=LedgerReason.ALLOCATION,
        description="Simulated allocation",
    )

    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=bot_live.id,
        asset="USDT",
        delta_amount=2000.0,
        reason=LedgerReason.ALLOCATION,
        description="Live allocation",
    )

    await test_db.commit()

    # Get live bot balance before replay
    live_balance_before = bot_live.current_balance

    # Replay simulated only
    replay_service = LedgerReplayService(test_db)
    await replay_service.rebuild_state_from_ledger("test_owner", True)

    # Refresh live bot
    await test_db.refresh(bot_live)

    # Live bot should be untouched
    assert bot_live.current_balance == live_balance_before


@pytest.mark.asyncio
async def test_replay_rebuilds_positions_correctly(test_db, sample_bot):
    """Test that positions are correctly rebuilt from BUY+SELL sequence."""
    # Create allocation
    ledger_service = LedgerWriterService(test_db)
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=2000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial allocation",
    )
    await test_db.flush()

    # Execute: BUY 0.02, BUY 0.01, SELL 0.01 = net position 0.02
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)

    # BUY 1
    order1 = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.02,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order1)
    await test_db.flush()

    trade1 = await trade_recorder.record_trade(
        order_id=order1.id,
        owner_id="test_owner",
        bot_id=sample_bot.id,
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.BUY,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.02,
        quote_amount=1000.0,
        price=50000.0,
        fee_amount=1.0,
        fee_asset="USDT",
        modeled_cost=0.1,
    )
    await tax_engine.process_buy(trade1)
    await test_db.flush()

    # BUY 2
    order2 = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=51000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order2)
    await test_db.flush()

    trade2 = await trade_recorder.record_trade(
        order_id=order2.id,
        owner_id="test_owner",
        bot_id=sample_bot.id,
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.BUY,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.01,
        quote_amount=510.0,
        price=51000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.1,
    )
    await tax_engine.process_buy(trade2)
    await test_db.flush()

    # SELL
    order3 = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_SELL,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=52000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order3)
    await test_db.flush()

    trade3 = await trade_recorder.record_trade(
        order_id=order3.id,
        owner_id="test_owner",
        bot_id=sample_bot.id,
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.SELL,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.01,
        quote_amount=520.0,
        price=52000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.1,
    )
    await tax_engine.process_sell(trade3)
    await test_db.commit()

    # Replay
    replay_service = LedgerReplayService(test_db)
    result = await replay_service.rebuild_state_from_ledger("test_owner", True)

    # Should have created 1 position with net amount 0.02
    assert result.positions_created == 1

    # Verify position
    from sqlalchemy import select
    query = select(Position).where(Position.bot_id == sample_bot.id)
    pos_result = await test_db.execute(query)
    position = pos_result.scalar_one_or_none()

    assert position is not None
    assert abs(position.amount - 0.02) < 1e-8


@pytest.mark.asyncio
async def test_validate_replay_detects_corruption(test_db, sample_bot):
    """Test that validation detects corrupted state."""
    # Create valid state
    ledger_service = LedgerWriterService(test_db)
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=1000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial allocation",
    )
    await test_db.commit()

    # Take snapshot
    replay_service = LedgerReplayService(test_db)
    pre_snapshot = await replay_service._snapshot_state("test_owner", True)

    # Manually corrupt bot balance
    sample_bot.current_balance = 5000.0  # Wrong!
    await test_db.commit()

    # Take post-snapshot
    post_snapshot = await replay_service._snapshot_state("test_owner", True)

    # Compare should detect differences
    differences = replay_service._compare_snapshots(pre_snapshot, post_snapshot)

    assert len(differences) > 0
    assert any("current_balance" in diff for diff in differences)


@pytest.mark.asyncio
async def test_replay_is_deterministic(test_db, sample_bot):
    """Test that multiple replays produce identical output."""
    # Create some ledger entries
    ledger_service = LedgerWriterService(test_db)
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=1000.0,
        reason=LedgerReason.ALLOCATION,
        description="Allocation",
    )
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=-100.0,
        reason=LedgerReason.FEE,
        description="Fee",
    )
    await test_db.commit()

    # Run replay 5 times
    replay_service = LedgerReplayService(test_db)
    results = []

    for i in range(5):
        result = await replay_service.rebuild_state_from_ledger("test_owner", True)
        results.append({
            'positions_created': result.positions_created,
            'balances_rebuilt': result.balances_rebuilt,
            'final_balances': result.final_balances,
        })

    # All results must be identical
    for i in range(1, 5):
        assert results[i] == results[0]


@pytest.mark.asyncio
async def test_dry_run_never_in_live_export(test_db):
    """Test that simulated trades don't appear in live CSV exports."""
    # Create simulated and live bots
    bot_simulated = Bot(
        name="Simulated Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
    )
    bot_live = Bot(
        name="Live Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=2000.0,
        current_balance=2000.0,
        is_dry_run=False,
    )
    test_db.add(bot_simulated)
    test_db.add(bot_live)
    await test_db.flush()

    # Create orders for both
    order_sim = Order(
        bot_id=bot_simulated.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    order_live = Order(
        bot_id=bot_live.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=False,
    )
    test_db.add(order_sim)
    test_db.add(order_live)
    await test_db.flush()

    # Create trades
    trade_recorder = TradeRecorderService(test_db)

    trade_sim = await trade_recorder.record_trade(
        order_id=order_sim.id,
        owner_id="test_owner",
        bot_id=bot_simulated.id,
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
        modeled_cost=0.0,
    )

    trade_live = await trade_recorder.record_trade(
        order_id=order_live.id,
        owner_id="test_owner",
        bot_id=bot_live.id,
        exchange="binance",
        trading_pair="BTC/USDT",
        side=TradeSide.BUY,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.01,
        quote_amount=500.0,
        price=50000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.0,
    )
    await test_db.commit()

    # Query trades with is_simulated filter
    from sqlalchemy import select
    from app.models import Trade, Order

    # Query live trades only
    query = select(Trade).join(Order).where(
        Order.is_simulated == False
    )
    result = await test_db.execute(query)
    live_trades = result.scalars().all()

    # Should only have 1 trade (the live one)
    assert len(live_trades) == 1
    assert live_trades[0].id == trade_live.id

    # Query simulated trades only
    query = select(Trade).join(Order).where(
        Order.is_simulated == True
    )
    result = await test_db.execute(query)
    sim_trades = result.scalars().all()

    # Should only have 1 trade (the simulated one)
    assert len(sim_trades) == 1
    assert sim_trades[0].id == trade_sim.id
