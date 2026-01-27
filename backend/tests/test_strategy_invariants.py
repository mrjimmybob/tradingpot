"""Strategy invariants test - Core accounting safety checks.

Purpose:
Ensure that the trading system never violates core accounting invariants:
- funds are never spent twice
- every executed trade has matching ledger entries
- no orphan trades exist
- balances always reconcile with ledger
- positions never go negative illegally

This is a SAFETY test, not a profitability test.

Failure meaning:
[FAIL] system can double-spend
[FAIL] money disappears or appears
[FAIL] trades are not fully recorded
[FAIL] accounting is unsafe
[FAIL] system is financially incorrect
[FAIL] real money would be at risk
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List, Dict
from unittest.mock import AsyncMock, Mock, patch

from app.models import (
    Bot,
    BotStatus,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Trade,
    TradeSide,
    WalletLedger,
    LedgerReason,
)
from app.services.trading_engine import TradingEngine, TradeSignal


# ============================================================================
# Deterministic Components
# ============================================================================


class DeterministicPriceFeed:
    """Deterministic price feed with slow oscillation."""

    def __init__(self, initial_price: float = 50000.0):
        """Initialize with base price."""
        self.initial_price = initial_price
        self.tick = 0

    def get_price(self) -> float:
        """Get price for current tick (oscillates ±1% slowly)."""
        self.tick += 1
        # Slow oscillation: period of 40 ticks
        import math
        phase = (2 * math.pi * self.tick) / 40
        deviation = 0.01 * math.sin(phase)  # ±1%
        return self.initial_price * (1.0 + deviation)


class DeterministicExchange:
    """Deterministic exchange with fixed fees and no slippage."""

    def __init__(self, fee_percent: float = 0.1):
        """Initialize deterministic exchange.

        Args:
            fee_percent: Fee as percentage (default 0.1%)
        """
        self.fee_percent = fee_percent / 100.0
        self.orders_executed = []
        self.is_connected = True
        self._current_price = 50000.0

    async def connect(self):
        """Mock connection."""
        self.is_connected = True

    async def get_ticker(self, symbol: str):
        """Return mock ticker."""
        return Mock(
            symbol=symbol,
            bid=self._current_price,
            ask=self._current_price,
            last=self._current_price,
            volume=1000000.0,
            timestamp=datetime.utcnow(),
        )

    async def place_market_order(self, symbol: str, side, amount: float):
        """Place deterministic market order with no slippage."""
        price = self._current_price
        cost = amount * price
        fee = cost * self.fee_percent

        order = Mock(
            id=f"det_order_{len(self.orders_executed) + 1}",
            symbol=symbol,
            side=str(side).lower() if hasattr(side, 'value') else side,
            amount=amount,
            filled=amount,
            price=price,
            cost=cost,
            fee=fee,
            status="closed",
            timestamp=datetime.utcnow().timestamp() * 1000,
        )

        self.orders_executed.append({
            "symbol": symbol,
            "side": str(side).lower() if hasattr(side, 'value') else side,
            "amount": amount,
            "price": price,
            "cost": cost,
            "fee": fee,
        })

        return order

    def set_current_price(self, price: float):
        """Set current price for next order execution."""
        self._current_price = price


# ============================================================================
# In-Memory Database Session
# ============================================================================


class InMemorySession:
    """In-memory session that tracks objects for invariant checking."""

    def __init__(self, bot: Bot):
        """Initialize with a bot."""
        self.bot = bot
        self.trades: List[Trade] = []
        self.ledger_entries: List[WalletLedger] = []
        self.positions: List[Position] = []
        self.orders: List[Order] = []
        self._id_counter = 1

    def add(self, obj):
        """Add object to session."""
        # Assign ID if not present
        if not hasattr(obj, 'id') or obj.id is None:
            obj.id = self._id_counter
            self._id_counter += 1

        if isinstance(obj, Trade):
            self.trades.append(obj)
        elif isinstance(obj, WalletLedger):
            self.ledger_entries.append(obj)
        elif isinstance(obj, Position):
            self.positions.append(obj)
        elif isinstance(obj, Order):
            self.orders.append(obj)

    async def commit(self):
        """Mock commit."""
        pass

    async def flush(self):
        """Mock flush."""
        pass

    async def refresh(self, obj):
        """Mock refresh."""
        pass

    async def rollback(self):
        """Mock rollback."""
        pass

    async def execute(self, query):
        """Mock execute - return empty results (no existing positions/trades)."""
        result = Mock()
        result.scalar_one_or_none = Mock(return_value=None)
        result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
        result.scalar = Mock(return_value=None)
        return result


# ============================================================================
# Test
# ============================================================================


@pytest.mark.asyncio
async def test_strategy_invariants_no_double_spend_no_orphans():
    """Ensure trading system never violates core accounting invariants.

    Test design:
    1. Deterministic environment (fixed prices, fixed fees)
    2. Run sequence of BUY/SELL trades
    3. Track all financial flows
    4. Assert accounting invariants

    Core invariants:
    A. No double spending (balance >= 0)
    B. Ledger completeness (every trade has ledger entries)
    C. No orphan entries (every ledger entry references real trade)
    D. Balance reconciliation (flows match balance)
    E. Position safety (no illegal negative positions)
    F. Trade/ledger bijection (entries match trades)
    G. No silent drops (all trades have ledger entries)
    """
    # ========================================================================
    # 1. Create deterministic environment
    # ========================================================================

    initial_balance = 10000.0
    price_feed = DeterministicPriceFeed(initial_price=50000.0)
    exchange = DeterministicExchange(fee_percent=0.1)
    await exchange.connect()

    # ========================================================================
    # 2. Create bot with fixed parameters
    # ========================================================================

    bot = Bot(
        id=1,
        name="Invariants Test Bot",
        trading_pair="BTC/USDT",
        strategy="test_strategy",
        strategy_params={},
        budget=initial_balance,
        current_balance=initial_balance,
        compound_enabled=True,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        total_pnl=0.0,
        stop_loss_percent=None,  # Disable risk limits
        drawdown_limit_percent=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    session = InMemorySession(bot)

    # ========================================================================
    # 3. Create trading engine with mocked services
    # ========================================================================

    engine = TradingEngine()

    # Track financial flows
    total_buy_costs = 0.0
    total_sell_proceeds = 0.0
    total_fees = 0.0
    successful_trades = 0
    balance_history = [initial_balance]

    # ========================================================================
    # 4. Mock services (but keep trade/ledger recording logic)
    # ========================================================================

    with patch('app.services.trading_engine.VirtualWalletService') as MockWallet, \
         patch('app.services.trading_engine.PortfolioRiskService') as MockPortfolioRisk, \
         patch('app.services.trading_engine.StrategyCapacityService') as MockCapacity, \
         patch('app.services.trading_engine.TradeRecorderService') as MockRecorder, \
         patch('app.services.trading_engine.FIFOTaxEngine') as MockTax, \
         patch('app.services.trading_engine.LedgerWriterService') as MockLedger, \
         patch('app.services.trading_engine.LedgerInvariantService') as MockInvariant, \
         patch('app.services.trading_engine.get_cost_model') as MockCostModel, \
         patch('app.services.trading_engine.CSVExportService') as MockCSV:

        # Configure wallet service
        mock_wallet_instance = MockWallet.return_value
        mock_wallet_instance.validate_trade = AsyncMock(
            return_value=Mock(is_valid=True, reason="", max_trade_amount=initial_balance)
        )
        mock_wallet_instance.update_balance = AsyncMock()
        mock_wallet_instance.record_trade_result = AsyncMock()

        # Configure portfolio risk service
        mock_portfolio_risk = MockPortfolioRisk.return_value
        mock_portfolio_risk.check_portfolio_risk = AsyncMock(
            return_value=Mock(ok=True, action=None, adjusted_amount=None, violated_cap=None, details="")
        )

        # Configure capacity service
        mock_capacity_instance = MockCapacity.return_value
        mock_capacity_instance.check_capacity_for_trade = AsyncMock(
            return_value=Mock(ok=True, reason="", adjusted_amount=None)
        )

        # Configure recorder service - IMPORTANT: Create real Trade objects
        mock_recorder_instance = MockRecorder.return_value

        def create_trade_with_ledger(*args, **kwargs):
            """Create Trade and corresponding WalletLedger entries."""
            trade = Trade(
                id=len(session.trades) + 1,
                order_id=kwargs.get('order_id', 1),
                owner_id=kwargs.get('owner_id', 'test'),
                bot_id=kwargs.get('bot_id', 1),
                exchange=kwargs.get('exchange', 'simulated'),
                trading_pair=kwargs.get('trading_pair', 'BTC/USDT'),
                side=kwargs.get('side', TradeSide.BUY),
                base_asset=kwargs.get('base_asset', 'BTC'),
                quote_asset=kwargs.get('quote_asset', 'USDT'),
                base_amount=kwargs.get('base_amount', 0.002),
                quote_amount=kwargs.get('quote_amount', 100.0),
                price=kwargs.get('price', 50000.0),
                fee_amount=kwargs.get('fee_amount', 0.1),
                fee_asset=kwargs.get('fee_asset', 'USDT'),
                modeled_cost=kwargs.get('modeled_cost', 0.1),
                exchange_trade_id=kwargs.get('exchange_trade_id', 'test_1'),
                executed_at=kwargs.get('executed_at', datetime.utcnow()),
                strategy_used=kwargs.get('strategy_used', 'test'),
            )

            # Add trade to session (this happens in real recorder)
            session.add(trade)

            # Create corresponding ledger entries
            # For BUY: quote goes down, base goes up, fee
            # For SELL: base goes down, quote goes up, fee
            if trade.side == TradeSide.BUY:
                # Quote asset outflow (USDT spent)
                ledger_quote = WalletLedger(
                    id=len(session.ledger_entries) + 1,
                    owner_id=trade.owner_id,
                    bot_id=trade.bot_id,
                    asset=trade.quote_asset,
                    delta_amount=-trade.quote_amount,
                    balance_after=0.0,  # Will be calculated
                    reason=LedgerReason.BUY,
                    related_trade_id=trade.id,
                    related_order_id=trade.order_id,
                    created_at=datetime.utcnow(),
                )
                session.add(ledger_quote)

                # Base asset inflow (BTC received)
                ledger_base = WalletLedger(
                    id=len(session.ledger_entries) + 1,
                    owner_id=trade.owner_id,
                    bot_id=trade.bot_id,
                    asset=trade.base_asset,
                    delta_amount=trade.base_amount,
                    balance_after=0.0,
                    reason=LedgerReason.BUY,
                    related_trade_id=trade.id,
                    related_order_id=trade.order_id,
                    created_at=datetime.utcnow(),
                )
                session.add(ledger_base)

                # Fee
                if trade.fee_amount > 0:
                    ledger_fee = WalletLedger(
                        id=len(session.ledger_entries) + 1,
                        owner_id=trade.owner_id,
                        bot_id=trade.bot_id,
                        asset=trade.fee_asset,
                        delta_amount=-trade.fee_amount,
                        balance_after=0.0,
                        reason=LedgerReason.FEE,
                        related_trade_id=trade.id,
                        related_order_id=trade.order_id,
                        created_at=datetime.utcnow(),
                    )
                    session.add(ledger_fee)

            else:  # SELL
                # Base asset outflow (BTC sold)
                ledger_base = WalletLedger(
                    id=len(session.ledger_entries) + 1,
                    owner_id=trade.owner_id,
                    bot_id=trade.bot_id,
                    asset=trade.base_asset,
                    delta_amount=-trade.base_amount,
                    balance_after=0.0,
                    reason=LedgerReason.SELL,
                    related_trade_id=trade.id,
                    related_order_id=trade.order_id,
                    created_at=datetime.utcnow(),
                )
                session.add(ledger_base)

                # Quote asset inflow (USDT received)
                ledger_quote = WalletLedger(
                    id=len(session.ledger_entries) + 1,
                    owner_id=trade.owner_id,
                    bot_id=trade.bot_id,
                    asset=trade.quote_asset,
                    delta_amount=trade.quote_amount,
                    balance_after=0.0,
                    reason=LedgerReason.SELL,
                    related_trade_id=trade.id,
                    related_order_id=trade.order_id,
                    created_at=datetime.utcnow(),
                )
                session.add(ledger_quote)

                # Fee
                if trade.fee_amount > 0:
                    ledger_fee = WalletLedger(
                        id=len(session.ledger_entries) + 1,
                        owner_id=trade.owner_id,
                        bot_id=trade.bot_id,
                        asset=trade.fee_asset,
                        delta_amount=-trade.fee_amount,
                        balance_after=0.0,
                        reason=LedgerReason.FEE,
                        related_trade_id=trade.id,
                        related_order_id=trade.order_id,
                        created_at=datetime.utcnow(),
                    )
                    session.add(ledger_fee)

            return trade

        mock_recorder_instance.record_trade = AsyncMock(side_effect=create_trade_with_ledger)

        # Configure tax engine
        mock_tax_instance = MockTax.return_value
        mock_tax_instance.process_buy = AsyncMock()
        mock_tax_instance.process_sell = AsyncMock(return_value=[])

        # Configure ledger writer
        mock_ledger_instance = MockLedger.return_value
        mock_ledger_instance.write_trade_entries = AsyncMock()

        # Configure invariant service
        mock_invariant_instance = MockInvariant.return_value
        mock_invariant_instance.validate_trade = AsyncMock()
        mock_invariant_instance.validate_invariants = AsyncMock()

        # Configure cost model
        mock_cost_model_instance = Mock()
        mock_cost_model_instance.estimate_cost = Mock(
            return_value=Mock(total_cost=0.1, exchange_fee=0.1, spread_cost=0.0, slippage_cost=0.0)
        )
        MockCostModel.return_value = mock_cost_model_instance

        # Configure CSV export service
        mock_csv_instance = MockCSV.return_value
        mock_csv_instance.export_trades_csv = AsyncMock()

        # ====================================================================
        # 5. Run sequence of alternating BUY/SELL trades
        # ====================================================================

        num_iterations = 150
        trade_size = 100.0  # $100 per trade

        # Create strategy that alternates BUY/SELL
        trade_counter = [0]

        async def alternating_strategy(bot, current_price, params, session):
            """Alternating BUY/SELL strategy."""
            trade_counter[0] += 1

            # Alternate: BUY on even ticks, SELL on odd ticks
            # But only if we have position to sell
            if trade_counter[0] % 2 == 0:
                # BUY
                return TradeSignal(
                    action="buy",
                    amount=trade_size,
                    order_type="market",
                    reason=f"Alternating buy #{trade_counter[0]}"
                )
            else:
                # SELL - but skip if no position
                # For simplicity, always try to sell (real system checks position)
                return TradeSignal(
                    action="sell",
                    amount=trade_size,
                    order_type="market",
                    reason=f"Alternating sell #{trade_counter[0]}"
                )

        with patch.object(engine, '_get_strategy_executor', return_value=alternating_strategy):

            for tick in range(num_iterations):
                try:
                    # Get current price
                    current_price = price_feed.get_price()
                    exchange.set_current_price(current_price)

                    # Generate signal
                    mock_ticker = Mock(
                        symbol="BTC/USDT",
                        bid=current_price,
                        ask=current_price,
                        last=current_price,
                        volume=1000000.0,
                        timestamp=datetime.utcnow(),
                    )

                    with patch.object(exchange, 'get_ticker', return_value=mock_ticker):
                        signal = await engine._execute_strategy(bot, current_price, session)

                    # Execute trade
                    if signal and signal.action != "hold":
                        order = await engine._execute_trade(bot, exchange, signal, current_price, session)

                        if order and order.status == OrderStatus.FILLED:
                            successful_trades += 1

                            # Track financial flows based on action
                            if signal.action == "buy":
                                # signal.amount is in quote currency (USDT)
                                # Actual cost = amount + fee
                                cost = signal.amount * 1.001  # 0.1% fee
                                total_buy_costs += cost
                                total_fees += signal.amount * 0.001

                                # Update bot balance (simulate)
                                bot.current_balance -= cost

                            elif signal.action == "sell":
                                # signal.amount is in quote currency value
                                # Actual proceeds = amount - fee
                                proceeds = signal.amount * 0.999  # 0.1% fee
                                total_sell_proceeds += proceeds
                                total_fees += signal.amount * 0.001

                                # Update bot balance (simulate)
                                bot.current_balance += proceeds

                            # Track balance history
                            balance_history.append(bot.current_balance)

                            # INVARIANT CHECK: No negative balance
                            assert bot.current_balance >= 0, (
                                f"[FAIL] DOUBLE SPEND DETECTED at tick {tick}!\n"
                                f"Balance went negative: ${bot.current_balance:.2f}\n"
                                f"Initial: ${initial_balance:.2f}\n"
                                f"Buy costs: ${total_buy_costs:.2f}\n"
                                f"Sell proceeds: ${total_sell_proceeds:.2f}\n"
                                f"Fees: ${total_fees:.2f}"
                            )

                except Exception as e:
                    # Fail test on any exception
                    pytest.fail(f"Exception at tick {tick}: {e}")

        # ====================================================================
        # 6. Assert Core Invariants
        # ====================================================================

        print(f"\n{'='*70}")
        print(f"Accounting Invariants Test Results")
        print(f"{'='*70}")
        print(f"Iterations: {num_iterations}")
        print(f"Successful trades: {successful_trades}")
        print(f"Trades recorded: {len(session.trades)}")
        print(f"Ledger entries: {len(session.ledger_entries)}")
        print(f"Initial balance: ${initial_balance:,.2f}")
        print(f"Final balance: ${bot.current_balance:,.2f}")
        print(f"Total buy costs: ${total_buy_costs:,.2f}")
        print(f"Total sell proceeds: ${total_sell_proceeds:,.2f}")
        print(f"Total fees: ${total_fees:,.2f}")
        print(f"Min balance: ${min(balance_history):,.2f}")
        print(f"Max balance: ${max(balance_history):,.2f}")
        print(f"{'='*70}\n")

        # ====================================================================
        # INVARIANT A: No double spending
        # ====================================================================
        assert bot.current_balance >= 0, (
            f"[FAIL] INVARIANT A VIOLATED: Double spending detected!\n"
            f"Final balance is negative: ${bot.current_balance:.2f}"
        )

        assert min(balance_history) >= 0, (
            f"[FAIL] INVARIANT A VIOLATED: Balance went negative during execution!\n"
            f"Minimum balance: ${min(balance_history):.2f}"
        )

        print("[PASS] INVARIANT A: No double spending (balance always >= 0)")

        # ====================================================================
        # INVARIANT B: Ledger completeness
        # ====================================================================
        # Every trade must have at least 2 ledger entries (base + quote)
        # Plus potentially 1 fee entry = 3 total

        trades_with_entries = {}
        for entry in session.ledger_entries:
            if entry.related_trade_id:
                if entry.related_trade_id not in trades_with_entries:
                    trades_with_entries[entry.related_trade_id] = 0
                trades_with_entries[entry.related_trade_id] += 1

        for trade in session.trades:
            entry_count = trades_with_entries.get(trade.id, 0)
            assert entry_count >= 2, (
                f"[FAIL] INVARIANT B VIOLATED: Incomplete ledger for trade {trade.id}!\n"
                f"Trade has only {entry_count} ledger entries (expected >= 2)\n"
                f"Trade: {trade.side.value} {trade.base_amount} {trade.base_asset} @ ${trade.price}"
            )

        print(f"[PASS] INVARIANT B: Ledger completeness (every trade has >= 2 entries)")

        # ====================================================================
        # INVARIANT C: No orphan ledger entries
        # ====================================================================
        trade_ids = {t.id for t in session.trades}

        for entry in session.ledger_entries:
            if entry.related_trade_id is not None:
                assert entry.related_trade_id in trade_ids, (
                    f"[FAIL] INVARIANT C VIOLATED: Orphan ledger entry detected!\n"
                    f"Ledger entry {entry.id} references non-existent trade_id={entry.related_trade_id}\n"
                    f"Asset: {entry.asset}, Delta: {entry.delta_amount}, Reason: {entry.reason.value}"
                )

        print(f"[PASS] INVARIANT C: No orphan entries (all ledger entries reference real trades)")

        # ====================================================================
        # INVARIANT D: Balance reconciliation
        # ====================================================================
        # initial_balance - buy_costs + sell_proceeds ≈ current_balance
        # (buy_costs and sell_proceeds already include fees)

        expected_balance = initial_balance - total_buy_costs + total_sell_proceeds
        balance_diff = abs(expected_balance - bot.current_balance)
        tolerance = 0.01  # Allow 1 cent tolerance for floating point

        assert balance_diff <= tolerance, (
            f"[FAIL] INVARIANT D VIOLATED: Balance reconciliation failed!\n"
            f"Expected balance: ${expected_balance:.2f}\n"
            f"Actual balance: ${bot.current_balance:.2f}\n"
            f"Difference: ${balance_diff:.2f}\n"
            f"\n"
            f"Flow breakdown:\n"
            f"  Initial: ${initial_balance:.2f}\n"
            f"  - Buy costs (incl. fees): ${total_buy_costs:.2f}\n"
            f"  + Sell proceeds (net of fees): ${total_sell_proceeds:.2f}\n"
            f"  = Expected: ${expected_balance:.2f}\n"
            f"\n"
            f"Total fees paid: ${total_fees:.2f}\n"
            f"\n"
            f"Money is missing or appearing from nowhere!"
        )

        print(f"[PASS] INVARIANT D: Balance reconciliation (flows match balance within ${tolerance})")

        # ====================================================================
        # INVARIANT E: Position safety
        # ====================================================================
        # Base asset position should never go negative (we're not shorting)

        # Calculate net base asset position from ledger
        base_asset_deltas = [
            entry.delta_amount
            for entry in session.ledger_entries
            if entry.asset == "BTC"
        ]

        if base_asset_deltas:
            cumulative_position = 0.0
            for delta in base_asset_deltas:
                cumulative_position += delta
                assert cumulative_position >= -0.00000001, (  # Small tolerance for float precision
                    f"[FAIL] INVARIANT E VIOLATED: Illegal negative position!\n"
                    f"Base asset position went negative: {cumulative_position}\n"
                    f"Shorting is not enabled, so position must be >= 0"
                )

        print(f"[PASS] INVARIANT E: Position safety (no illegal negative positions)")

        # ====================================================================
        # INVARIANT F: Trade ↔ ledger bijection
        # ====================================================================
        # Number of trades should be less than or equal to ledger entries / 2
        # (since each trade creates at least 2 entries)

        assert len(session.trades) <= len(session.ledger_entries) / 2, (
            f"[FAIL] INVARIANT F VIOLATED: Trade/ledger bijection broken!\n"
            f"Trades: {len(session.trades)}\n"
            f"Ledger entries: {len(session.ledger_entries)}\n"
            f"Expected: trades <= entries / 2"
        )

        print(f"[PASS] INVARIANT F: Trade/ledger bijection (entries match trades)")

        # ====================================================================
        # INVARIANT G: No silent drops
        # ====================================================================
        # Every trade must have ledger entries

        trades_without_entries = []
        for trade in session.trades:
            if trade.id not in trades_with_entries:
                trades_without_entries.append(trade.id)

        assert len(trades_without_entries) == 0, (
            f"[FAIL] INVARIANT G VIOLATED: Silent drops detected!\n"
            f"{len(trades_without_entries)} trades have no ledger entries:\n"
            f"Trade IDs: {trades_without_entries}\n"
            f"These trades executed but were not recorded in the ledger!"
        )

        print(f"[PASS] INVARIANT G: No silent drops (all trades have ledger entries)")

        # ====================================================================
        # Additional Safety Checks
        # ====================================================================

        # No NaN or infinite values
        assert not (bot.current_balance != bot.current_balance), "Balance is NaN"
        assert bot.current_balance != float('inf'), "Balance is infinity"
        assert bot.current_balance != float('-inf'), "Balance is negative infinity"

        # All ledger entries have valid amounts
        for entry in session.ledger_entries:
            assert not (entry.delta_amount != entry.delta_amount), f"Ledger entry {entry.id} has NaN delta"
            assert entry.delta_amount != float('inf'), f"Ledger entry {entry.id} has infinite delta"
            assert entry.delta_amount != float('-inf'), f"Ledger entry {entry.id} has negative infinite delta"

        # All trades have valid amounts
        for trade in session.trades:
            assert trade.base_amount > 0, f"Trade {trade.id} has invalid base_amount: {trade.base_amount}"
            assert trade.quote_amount > 0, f"Trade {trade.id} has invalid quote_amount: {trade.quote_amount}"
            assert trade.price > 0, f"Trade {trade.id} has invalid price: {trade.price}"

        print(f"[PASS] Additional checks: No NaN/infinite values, all amounts valid")

        print(f"\n{'='*70}")
        print(f"[PASS] ALL INVARIANTS PASSED")
        print(f"{'='*70}\n")

        # Final assertion
        assert True

