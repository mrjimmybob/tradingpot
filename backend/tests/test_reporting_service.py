"""Tests for Reporting Service

Tests all 12 reports with focus on:
- Data accuracy
- is_simulated separation
- Totals matching ledger reconstruction
"""

import pytest
from datetime import datetime, timedelta

from app.services.reporting_service import ReportingService
from app.services.accounting import TradeRecorderService, FIFOTaxEngine
from app.services.ledger_writer import LedgerWriterService
from app.models import (
    Bot,
    Order,
    Trade,
    TradeSide,
    OrderType,
    OrderStatus,
    WalletLedger,
    LedgerReason,
    Position,
    PositionSide,
)


@pytest.mark.asyncio
async def test_trade_history_report(test_db, sample_bot, sample_order):
    """Test trade history report returns all trades with realized P&L."""
    # Create 3 trades
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)

    # BUY
    trade1 = await trade_recorder.record_trade(
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
    await tax_engine.process_buy(trade1)
    await test_db.flush()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_trade_history(is_simulated=True)

    assert len(records) == 1
    assert records[0].trade_id == trade1.id
    assert records[0].bot_id == sample_bot.id
    assert records[0].symbol == "BTC/USDT"
    assert records[0].side == "BUY"
    assert records[0].quantity == 0.01
    assert records[0].is_simulated == True
    assert records[0].realized_pnl is None  # BUY doesn't have realized P&L


@pytest.mark.asyncio
async def test_live_vs_simulated_separation(test_db):
    """Test that live and simulated data are strictly separated."""
    # Create live and simulated bots
    bot_live = Bot(
        name="Live Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=False,
    )
    bot_sim = Bot(
        name="Simulated Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
    )
    test_db.add(bot_live)
    test_db.add(bot_sim)
    await test_db.flush()

    # Create orders
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
    order_sim = Order(
        bot_id=bot_sim.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order_live)
    test_db.add(order_sim)
    await test_db.flush()

    # Create trades
    trade_recorder = TradeRecorderService(test_db)

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

    trade_sim = await trade_recorder.record_trade(
        order_id=order_sim.id,
        owner_id="test_owner",
        bot_id=bot_sim.id,
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
    await test_db.commit()

    # Query reports with separation
    service = ReportingService(test_db)

    # Live report should only have live trade
    live_records = await service.get_trade_history(is_simulated=False)
    assert len(live_records) == 1
    assert live_records[0].trade_id == trade_live.id
    assert live_records[0].is_simulated == False

    # Simulated report should only have simulated trade
    sim_records = await service.get_trade_history(is_simulated=True)
    assert len(sim_records) == 1
    assert sim_records[0].trade_id == trade_sim.id
    assert sim_records[0].is_simulated == True


@pytest.mark.asyncio
async def test_order_lifecycle_report(test_db, sample_bot):
    """Test order lifecycle report."""
    # Create multiple orders with different statuses
    order1 = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
        filled_at=datetime.utcnow(),
    )
    order2 = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.LIMIT_SELL,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=55000.0,
        status=OrderStatus.PENDING,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order1)
    test_db.add(order2)
    await test_db.commit()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_order_lifecycle(is_simulated=True)

    assert len(records) == 2
    # Check FILLED order
    filled_order = next(r for r in records if r.status == 'FILLED')
    assert filled_order.filled_at is not None

    # Check PENDING order
    pending_order = next(r for r in records if r.status == 'PENDING')
    assert pending_order.filled_at is None


@pytest.mark.asyncio
async def test_realized_gains_report(test_db, sample_bot):
    """Test realized gains report."""
    ledger_service = LedgerWriterService(test_db)

    # Create allocation
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=2000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    # Execute BUY then SELL
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)

    # BUY
    order_buy = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order_buy)
    await test_db.flush()

    trade_buy = await trade_recorder.record_trade(
        order_id=order_buy.id,
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
        modeled_cost=0.0,
    )
    await tax_engine.process_buy(trade_buy)
    await test_db.flush()

    # SELL at higher price
    order_sell = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_SELL,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=55000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order_sell)
    await test_db.flush()

    trade_sell = await trade_recorder.record_trade(
        order_id=order_sell.id,
        owner_id="test_owner",
        bot_id=sample_bot.id,
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.SELL,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.01,
        quote_amount=550.0,
        price=55000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.0,
    )
    await tax_engine.process_sell(trade_sell)
    await test_db.commit()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_realized_gains(is_simulated=True)

    assert len(records) == 1
    assert records[0].asset == "BTC"
    assert records[0].quantity == 0.01
    assert records[0].gain_loss > 0  # Profit
    assert records[0].is_simulated == True


@pytest.mark.asyncio
async def test_balance_history_report(test_db, sample_bot):
    """Test balance history report aggregates correctly."""
    ledger_service = LedgerWriterService(test_db)

    # Create multiple ledger entries over time
    base_time = datetime.utcnow()

    for i in range(5):
        await ledger_service.write_entry(
            owner_id="test_owner",
            bot_id=sample_bot.id,
            asset="USDT",
            delta_amount=100.0 * (i + 1),
            reason=LedgerReason.ALLOCATION,
            description=f"Entry {i}",
        )
        await test_db.flush()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_balance_history(
        is_simulated=True,
        asset="USDT",
        time_bucket='hour',
    )

    assert len(records) > 0
    # Check balances are cumulative
    for record in records:
        assert record.asset == "USDT"
        assert record.balance > 0


@pytest.mark.asyncio
async def test_drawdown_report(test_db, sample_bot):
    """Test drawdown report calculates correctly."""
    ledger_service = LedgerWriterService(test_db)

    # Create equity curve with drawdown
    # Start at 1000, go to 1500 (peak), then 1200 (drawdown)
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=1000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=500.0,
        reason=LedgerReason.BUY,
        description="Profit",
    )
    await test_db.flush()

    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=-300.0,
        reason=LedgerReason.SELL,
        description="Loss",
    )
    await test_db.commit()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_drawdown(is_simulated=True, asset="USDT")

    assert len(records) == 3
    # Check that max drawdown was calculated
    max_dd = max(r.max_drawdown_pct for r in records)
    assert max_dd > 0


@pytest.mark.asyncio
async def test_strategy_performance_report(test_db, sample_bot):
    """Test strategy performance report aggregates by strategy."""
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)
    ledger_service = LedgerWriterService(test_db)

    # Create allocation
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=2000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    # Create trades for strategy
    order = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="momentum",
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
        quote_amount=500.0,
        price=50000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.0,
    )
    await tax_engine.process_buy(trade)
    await test_db.commit()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_strategy_performance(is_simulated=True)

    assert len(records) >= 1
    momentum_record = next((r for r in records if r.strategy == "momentum"), None)
    assert momentum_record is not None
    assert momentum_record.total_trades >= 1


@pytest.mark.asyncio
async def test_ledger_audit_report(test_db, sample_bot):
    """Test ledger audit report shows all entries with debit/credit."""
    ledger_service = LedgerWriterService(test_db)

    # Create debit and credit entries
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=1000.0,  # Credit
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=-500.0,  # Debit
        reason=LedgerReason.BUY,
        description="Purchase",
    )
    await test_db.commit()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_ledger_audit(is_simulated=True)

    assert len(records) == 2
    # Check credit entry
    credit_entry = next(r for r in records if r.credit is not None)
    assert credit_entry.credit == 1000.0
    assert credit_entry.debit is None

    # Check debit entry
    debit_entry = next(r for r in records if r.debit is not None)
    assert debit_entry.debit == 500.0
    assert debit_entry.credit is None


@pytest.mark.asyncio
async def test_cost_basis_report(test_db, sample_bot):
    """Test cost basis report shows open lots only."""
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)
    ledger_service = LedgerWriterService(test_db)

    # Create allocation
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=1000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    # BUY to create tax lot
    order = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.02,
        price=50000.0,
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
        base_amount=0.02,
        quote_amount=1000.0,
        price=50000.0,
        fee_amount=1.0,
        fee_asset="USDT",
        modeled_cost=0.0,
    )
    await tax_engine.process_buy(trade)
    await test_db.commit()

    # Get report
    service = ReportingService(test_db)
    records = await service.get_cost_basis(is_simulated=True)

    assert len(records) == 1
    assert records[0].asset == "BTC"
    assert records[0].quantity_remaining == 0.02
    assert records[0].unit_cost > 0


@pytest.mark.asyncio
async def test_totals_match_ledger_reconstruction(test_db, sample_bot):
    """Test that report totals match ledger-reconstructed values."""
    ledger_service = LedgerWriterService(test_db)

    # Create ledger entries
    entries = [
        (1000.0, LedgerReason.ALLOCATION),
        (-100.0, LedgerReason.FEE),
        (50.0, LedgerReason.BUY),
        (-200.0, LedgerReason.SELL),
    ]

    for delta, reason in entries:
        await ledger_service.write_entry(
            owner_id="test_owner",
            bot_id=sample_bot.id,
            asset="USDT",
            delta_amount=delta,
            reason=reason,
            description="Test",
        )
        await test_db.flush()

    await test_db.commit()

    # Reconstruct balance from ledger
    reconstructed = await ledger_service.reconstruct_balance(
        owner_id="test_owner",
        asset="USDT",
        bot_id=sample_bot.id,
    )

    # Get balance from report
    service = ReportingService(test_db)
    balance_records = await service.get_balance_history(
        is_simulated=True,
        asset="USDT",
        time_bucket='hour',
    )

    # Latest balance should match reconstructed
    if balance_records:
        latest_balance = balance_records[-1].balance
        assert abs(latest_balance - reconstructed) < 1e-8


# ============================================================================
# NEW ENDPOINT TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_trade_detail_report(test_db, sample_bot):
    """Test trade detail report returns full forensic trail."""
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)
    ledger_service = LedgerWriterService(test_db)

    # Create allocation
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=2000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    # Execute BUY then SELL
    # BUY
    order_buy = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_BUY,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=50000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order_buy)
    await test_db.flush()

    trade_buy = await trade_recorder.record_trade(
        order_id=order_buy.id,
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
    await tax_engine.process_buy(trade_buy)
    await test_db.flush()

    # SELL at higher price
    order_sell = Order(
        bot_id=sample_bot.id,
        order_type=OrderType.MARKET_SELL,
        trading_pair="BTC/USDT",
        amount=0.01,
        price=55000.0,
        status=OrderStatus.FILLED,
        strategy_used="test",
        is_simulated=True,
    )
    test_db.add(order_sell)
    await test_db.flush()

    trade_sell = await trade_recorder.record_trade(
        order_id=order_sell.id,
        owner_id="test_owner",
        bot_id=sample_bot.id,
        exchange="simulated",
        trading_pair="BTC/USDT",
        side=TradeSide.SELL,
        base_asset="BTC",
        quote_asset="USDT",
        base_amount=0.01,
        quote_amount=550.0,
        price=55000.0,
        fee_amount=0.5,
        fee_asset="USDT",
        modeled_cost=0.1,
    )
    await tax_engine.process_sell(trade_sell)
    await test_db.commit()

    # Get trade detail for SELL trade
    service = ReportingService(test_db)
    detail = await service.get_trade_detail(
        trade_id=trade_sell.id,
        is_simulated=True,
    )

    assert detail is not None
    assert detail.trade['id'] == trade_sell.id
    assert detail.order['order_id'] == order_sell.id
    assert len(detail.ledger_entries) > 0
    assert len(detail.tax_lots_consumed) == 1  # One lot consumed
    assert detail.realized_gain_loss is not None
    assert detail.realized_gain_loss > 0  # Profit
    assert detail.modeled_cost == 0.1
    assert detail.realized_cost > 0


@pytest.mark.asyncio
async def test_balance_drilldown_report(test_db, sample_bot):
    """Test balance drilldown report shows recent entries with classification."""
    ledger_service = LedgerWriterService(test_db)

    # Create various types of ledger entries
    entries = [
        (1000.0, LedgerReason.ALLOCATION, "funding"),
        (-100.0, LedgerReason.FEE, "fee"),
        (50.0, LedgerReason.BUY, "trade"),
        (-200.0, LedgerReason.SELL, "trade"),
        (10.0, LedgerReason.CORRECTION, "correction"),
    ]

    for delta, reason, expected_classification in entries:
        await ledger_service.write_entry(
            owner_id="test_owner",
            bot_id=sample_bot.id,
            asset="USDT",
            delta_amount=delta,
            reason=reason,
            description="Test",
        )
        await test_db.flush()

    await test_db.commit()

    # Get balance drilldown
    service = ReportingService(test_db)
    drilldown = await service.get_balance_drilldown(
        is_simulated=True,
        asset="USDT",
        limit=20,
    )

    assert drilldown.current_balance > 0
    assert len(drilldown.ledger_entries) == 5
    assert drilldown.cumulative_total == drilldown.current_balance

    # Check classifications
    fee_entries = [e for e in drilldown.ledger_entries if e.source_classification == "fee"]
    assert len(fee_entries) == 1

    trade_entries = [e for e in drilldown.ledger_entries if e.source_classification == "trade"]
    assert len(trade_entries) == 2


@pytest.mark.asyncio
async def test_risk_status_report(test_db, sample_bot):
    """Test risk status report shows bot and portfolio risk."""
    from app.models import Alert, StrategyRotation

    # Set bot initial state
    sample_bot.budget = 1000.0
    sample_bot.current_balance = 900.0  # 10% drawdown
    sample_bot.started_at = datetime.utcnow() - timedelta(days=2)
    await test_db.commit()

    # Create alert
    alert = Alert(
        bot_id=sample_bot.id,
        alert_type="STOP_LOSS",
        message="Stop loss triggered",
    )
    test_db.add(alert)
    await test_db.commit()

    # Get risk status
    service = ReportingService(test_db)
    risk_status = await service.get_risk_status(is_simulated=True)

    assert len(risk_status.bots) >= 1
    bot_risk = next((b for b in risk_status.bots if b.bot_id == sample_bot.id), None)
    assert bot_risk is not None
    assert bot_risk.drawdown_pct == 10.0
    assert bot_risk.kill_switch_state in ["active", "paused", "stopped"]
    assert bot_risk.last_risk_event is not None
    assert bot_risk.last_risk_event['type'] == "STOP_LOSS"

    # Check portfolio info
    assert 'total_exposure_pct' in risk_status.portfolio
    assert 'total_portfolio_value' in risk_status.portfolio


@pytest.mark.asyncio
async def test_equity_curve_report(test_db, sample_bot):
    """Test equity curve report with events."""
    from app.models import Alert, StrategyRotation

    ledger_service = LedgerWriterService(test_db)

    # Create equity curve
    for i in range(5):
        await ledger_service.write_entry(
            owner_id="test_owner",
            bot_id=sample_bot.id,
            asset="USDT",
            delta_amount=100.0 * (i + 1),
            reason=LedgerReason.ALLOCATION,
            description=f"Entry {i}",
        )
        await test_db.flush()

    # Create events
    rotation = StrategyRotation(
        bot_id=sample_bot.id,
        from_strategy="mean_reversion",
        to_strategy="trend_following",
        reason="market regime change",
    )
    test_db.add(rotation)

    alert = Alert(
        bot_id=sample_bot.id,
        alert_type="LARGE_LOSS",
        message="Large loss detected",
    )
    test_db.add(alert)
    await test_db.commit()

    # Get equity curve
    service = ReportingService(test_db)
    curve, events = await service.get_equity_curve(
        is_simulated=True,
        asset="USDT",
    )

    assert len(curve) == 5
    assert all(c.equity > 0 for c in curve)

    assert len(events) >= 2
    strategy_switch = next((e for e in events if e.event_type == "strategy_switch"), None)
    assert strategy_switch is not None

    large_loss = next((e for e in events if e.event_type == "large_loss"), None)
    assert large_loss is not None


@pytest.mark.asyncio
async def test_strategy_reason_report(test_db, sample_bot):
    """Test strategy reason report shows eligible and blocked strategies."""
    from app.models import StrategyRotation

    # Set bot strategy
    sample_bot.strategy = "mean_reversion"
    sample_bot.max_strategy_rotations = 3
    await test_db.commit()

    # Create a recent rotation (within cooldown)
    rotation = StrategyRotation(
        bot_id=sample_bot.id,
        from_strategy="mean_reversion",
        to_strategy="trend_following",
        reason="test",
        created_at=datetime.utcnow() - timedelta(minutes=30),  # Within 1h cooldown
    )
    test_db.add(rotation)
    await test_db.commit()

    # Get strategy reason
    service = ReportingService(test_db)
    reason = await service.get_strategy_reason(
        bot_id=sample_bot.id,
        is_simulated=True,
    )

    assert reason is not None
    assert reason.current_strategy == "mean_reversion"
    assert len(reason.eligible_strategies) > 0
    assert "mean_reversion" in reason.eligible_strategies

    # trend_following should be blocked due to cooldown
    blocked_tf = next(
        (b for b in reason.blocked_strategies if b.strategy_name == "trend_following"),
        None
    )
    assert blocked_tf is not None
    assert "cooldown" in blocked_tf.blocked_reason.lower()


@pytest.mark.asyncio
async def test_tax_summary_report(test_db, sample_bot):
    """Test tax summary report aggregates gains correctly."""
    trade_recorder = TradeRecorderService(test_db)
    tax_engine = FIFOTaxEngine(test_db)
    ledger_service = LedgerWriterService(test_db)

    # Create allocation
    await ledger_service.write_entry(
        owner_id="test_owner",
        bot_id=sample_bot.id,
        asset="USDT",
        delta_amount=5000.0,
        reason=LedgerReason.ALLOCATION,
        description="Initial",
    )
    await test_db.flush()

    # Execute multiple BUY/SELL pairs
    for i in range(2):
        # BUY
        order_buy = Order(
            bot_id=sample_bot.id,
            order_type=OrderType.MARKET_BUY,
            trading_pair="BTC/USDT",
            amount=0.01,
            price=50000.0,
            status=OrderStatus.FILLED,
            strategy_used="test",
            is_simulated=True,
        )
        test_db.add(order_buy)
        await test_db.flush()

        trade_buy = await trade_recorder.record_trade(
            order_id=order_buy.id,
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
            modeled_cost=0.0,
        )
        await tax_engine.process_buy(trade_buy)
        await test_db.flush()

        # SELL
        order_sell = Order(
            bot_id=sample_bot.id,
            order_type=OrderType.MARKET_SELL,
            trading_pair="BTC/USDT",
            amount=0.01,
            price=55000.0,
            status=OrderStatus.FILLED,
            strategy_used="test",
            is_simulated=True,
        )
        test_db.add(order_sell)
        await test_db.flush()

        trade_sell = await trade_recorder.record_trade(
            order_id=order_sell.id,
            owner_id="test_owner",
            bot_id=sample_bot.id,
            exchange="simulated",
            trading_pair="BTC/USDT",
            side=TradeSide.SELL,
            base_asset="BTC",
            quote_asset="USDT",
            base_amount=0.01,
            quote_amount=550.0,
            price=55000.0,
            fee_amount=0.5,
            fee_asset="USDT",
            modeled_cost=0.0,
        )
        await tax_engine.process_sell(trade_sell)
        await test_db.flush()

    await test_db.commit()

    # Get tax summary for current year
    service = ReportingService(test_db)
    summary = await service.get_tax_summary(
        year=datetime.utcnow().year,
        is_simulated=True,
    )

    assert summary.total_realized_gain > 0
    assert summary.lot_count == 2
    assert summary.trade_count == 2
    # Short-term vs long-term depends on holding period
    assert summary.short_term_gain + summary.long_term_gain == summary.total_realized_gain


@pytest.mark.asyncio
async def test_audit_log_report(test_db, sample_bot):
    """Test audit log report combines alerts and rotations."""
    from app.models import Alert, StrategyRotation

    # Create alerts with different severities
    alert1 = Alert(
        bot_id=sample_bot.id,
        alert_type="STOP_LOSS_ERROR",
        message="Stop loss triggered",
    )
    alert2 = Alert(
        bot_id=sample_bot.id,
        alert_type="WARNING_DRAWDOWN",
        message="High drawdown",
    )
    test_db.add(alert1)
    test_db.add(alert2)

    # Create strategy rotation
    rotation = StrategyRotation(
        bot_id=sample_bot.id,
        from_strategy="mean_reversion",
        to_strategy="trend_following",
        reason="market regime change",
    )
    test_db.add(rotation)
    await test_db.commit()

    # Get audit log
    service = ReportingService(test_db)
    records = await service.get_audit_log(is_simulated=True)

    assert len(records) >= 3

    # Check alert records
    alert_records = [r for r in records if r.source == "alerts_log"]
    assert len(alert_records) == 2

    error_alert = next((r for r in alert_records if r.severity == "error"), None)
    assert error_alert is not None

    warning_alert = next((r for r in alert_records if r.severity == "warning"), None)
    assert warning_alert is not None

    # Check rotation record
    rotation_records = [r for r in records if r.source == "strategy_rotations"]
    assert len(rotation_records) == 1
    assert rotation_records[0].severity == "info"

    # Test severity filtering
    error_only = await service.get_audit_log(is_simulated=True, severity="error")
    assert all(r.severity == "error" for r in error_only)


@pytest.mark.asyncio
async def test_new_endpoints_enforce_simulated_separation(test_db):
    """Test that new endpoints enforce is_simulated separation."""
    # Create live and simulated bots
    bot_live = Bot(
        name="Live Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=False,
    )
    bot_sim = Bot(
        name="Simulated Bot",
        trading_pair="BTC/USDT",
        strategy="test",
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
    )
    test_db.add(bot_live)
    test_db.add(bot_sim)
    await test_db.commit()

    service = ReportingService(test_db)

    # Test risk status separation
    live_risk = await service.get_risk_status(is_simulated=False)
    sim_risk = await service.get_risk_status(is_simulated=True)

    live_bot_ids = [b.bot_id for b in live_risk.bots]
    sim_bot_ids = [b.bot_id for b in sim_risk.bots]

    assert bot_live.id in live_bot_ids
    assert bot_live.id not in sim_bot_ids
    assert bot_sim.id in sim_bot_ids
    assert bot_sim.id not in live_bot_ids
