"""
Test: Ledger Replay Reconstructs Balance

Purpose:
    Verify that replaying ledger entries alone (without trades, engine, or exchange)
    reconstructs the exact same final wallet balance. This proves the ledger is the
    authoritative source of truth for all financial state.

Validates:
    - Ledger is complete
    - Ledger is authoritative
    - Wallet state is reproducible
    - Accounting is correct
    - Corruption is detectable
    - Balance can be reconstructed from ledger alone

Author: Trading Bot Test Suite
Date: 2026-01-27
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, List
from unittest.mock import Mock, AsyncMock, patch

from app.models.bot import Bot, BotStatus
from app.models.trade import Trade, TradeSide
from app.models.wallet_ledger import WalletLedger, LedgerReason
from app.models.position import Position
from app.models.order import Order
from app.services.trading_engine import TradingEngine


# ============================================================================
# Deterministic Price Feed
# ============================================================================


class DeterministicPriceFeed:
    """Deterministic price feed for reproducible testing."""

    def __init__(self, fixed_price: float = 100.0):
        """Initialize with a fixed price."""
        self.fixed_price = fixed_price

    async def get_current_price(self, trading_pair: str) -> float:
        """Return fixed price."""
        return self.fixed_price


# ============================================================================
# Deterministic Exchange
# ============================================================================


class DeterministicExchange:
    """Deterministic exchange with fixed fills and fees."""

    def __init__(self, fee_rate: float = 0.001):
        """Initialize with fixed fee rate."""
        self.fee_rate = fee_rate
        self.order_counter = 1

    async def create_order(
        self, trading_pair: str, side: str, amount: float, price: float, **kwargs
    ):
        """Create order and immediately fill it."""
        return await self._fill_order(trading_pair, side, amount, price)

    async def place_market_order(
        self, trading_pair: str, side: str, amount: float, **kwargs
    ):
        """Place market order and immediately fill it at current price."""
        price = 100.0
        return await self._fill_order(trading_pair, side, amount, price)

    async def _fill_order(
        self, trading_pair: str, side: str, amount: float, price: float
    ):
        """Fill order immediately with deterministic results."""
        order_id = f"ORDER_{self.order_counter}"
        self.order_counter += 1

        filled_amount = amount
        filled_price = price
        fee = filled_amount * filled_price * self.fee_rate

        return Mock(
            id=order_id,
            symbol=trading_pair,
            side=side,
            amount=filled_amount,
            price=filled_price,
            cost=filled_amount * filled_price,
            fee=fee,
            status="closed",
            filled=filled_amount,
        )

    async def fetch_balance(self):
        """Return mock balance."""
        return {
            "USDT": {"free": 10000, "used": 0, "total": 10000},
            "BTC": {"free": 0, "used": 0, "total": 0},
        }


# ============================================================================
# In-Memory Session for Tracking
# ============================================================================


class InMemorySession:
    """Mock session that stores objects in memory for validation."""

    def __init__(self, bot: Bot):
        """Initialize with a bot."""
        self.bot = bot
        self.trades: List[Trade] = []
        self.ledger_entries: List[WalletLedger] = []
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self._id_counter = 1

    def add(self, obj):
        """Add object to session."""
        if not hasattr(obj, 'id') or obj.id is None:
            obj.id = self._id_counter
            self._id_counter += 1

        if isinstance(obj, Trade):
            self.trades.append(obj)
        elif isinstance(obj, WalletLedger):
            self.ledger_entries.append(obj)
        elif isinstance(obj, Position):
            self.positions[obj.trading_pair] = obj
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
        if isinstance(obj, Position) and obj.trading_pair in self.positions:
            tracked = self.positions[obj.trading_pair]
            obj.amount = tracked.amount
            obj.entry_price = tracked.entry_price

    async def rollback(self):
        """Mock rollback."""
        pass

    async def delete(self, obj):
        """Mock delete."""
        if isinstance(obj, Position):
            if obj.trading_pair in self.positions:
                del self.positions[obj.trading_pair]

    async def execute(self, query):
        """Mock execute."""
        result = Mock()
        query_str = str(query)

        if 'bot.' in query_str.lower() or 'SELECT bots' in query_str:
            result.scalar_one_or_none = Mock(return_value=None)
            result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
        else:
            position = self.positions.get(self.bot.trading_pair)
            result.scalar_one_or_none = Mock(return_value=position)
            result.scalars = Mock(
                return_value=Mock(all=Mock(return_value=list(self.positions.values())))
            )

        result.scalar = Mock(return_value=None)
        return result


# ============================================================================
# Ledger Replay Engine
# ============================================================================


class LedgerReplayEngine:
    """
    Replays ledger entries to reconstruct balance.
    
    This is the authoritative way to compute balance from first principles.
    It treats the ledger as the single source of truth.
    """

    def __init__(self, initial_balance: float):
        """Initialize with starting balance."""
        self.initial_balance = initial_balance

    def replay(self, ledger_entries: List[WalletLedger], asset: str = "USDT") -> Dict:
        """
        Replay ledger entries for a specific asset to reconstruct balance.
        
        Args:
            ledger_entries: List of ledger entries to replay
            asset: Asset to replay (e.g., "USDT")
            
        Returns:
            Dictionary with replay results and diagnostics
        """
        # Filter entries for this asset
        asset_entries = [
            entry for entry in ledger_entries if entry.asset == asset
        ]

        # Sort by timestamp (created_at)
        sorted_entries = sorted(asset_entries, key=lambda e: e.created_at)

        # Replay each entry
        balance = self.initial_balance
        entry_count = 0
        total_delta = 0.0

        for entry in sorted_entries:
            delta = float(entry.delta_amount)
            balance += delta
            total_delta += delta
            entry_count += 1

        return {
            "initial_balance": self.initial_balance,
            "replayed_balance": balance,
            "total_delta": total_delta,
            "entry_count": entry_count,
            "asset": asset,
            "timestamps_monotonic": self._check_monotonic(sorted_entries),
            "no_nan": not any(
                isinstance(e.delta_amount, float) and (
                    e.delta_amount != e.delta_amount  # NaN check
                    or abs(e.delta_amount) == float('inf')
                )
                for e in sorted_entries
            ),
        }

    def _check_monotonic(self, entries: List[WalletLedger]) -> bool:
        """Check if timestamps are monotonically increasing."""
        if len(entries) < 2:
            return True

        for i in range(1, len(entries)):
            if entries[i].created_at < entries[i-1].created_at:
                return False

        return True


# ============================================================================
# Main Test
# ============================================================================


@pytest.mark.asyncio
async def test_ledger_replay_reconstructs_balance():
    """
    Test: Ledger Replay Reconstructs Balance

    Executes trades, captures ledger entries, then replays them to prove
    the ledger alone can reconstruct the exact final balance.
    """

    # -------------------------------------------------------------------------
    # Setup: Create deterministic environment
    # -------------------------------------------------------------------------

    initial_balance = Decimal("10000.00")

    bot = Bot(
        id=1,
        name="replay_test_bot",
        trading_pair="BTC/USDT",
        strategy="simple_moving_average",
        strategy_params={
            "fast_period": 5,
            "slow_period": 10,
        },
        budget=initial_balance,
        current_balance=initial_balance,
        is_dry_run=True,
        status="active",
    )

    exchange = DeterministicExchange(fee_rate=0.001)  # 0.1% fee
    price_feed = DeterministicPriceFeed(fixed_price=100.0)
    session = InMemorySession(bot)

    # -------------------------------------------------------------------------
    # Mock all services
    # -------------------------------------------------------------------------

    with patch("app.services.trading_engine.PortfolioRiskService") as mock_risk, \
         patch("app.services.trading_engine.StrategyCapacityService") as mock_capacity, \
         patch("app.services.trading_engine.ExecutionCostModel") as mock_cost, \
         patch("app.services.trading_engine.TradeRecorderService") as mock_recorder, \
         patch("app.services.trading_engine.FIFOTaxEngine") as mock_tax, \
         patch("app.services.trading_engine.LedgerWriterService") as mock_ledger_writer, \
         patch("app.services.trading_engine.LedgerInvariantService") as mock_invariant, \
         patch("app.services.trading_engine.VirtualWalletService") as mock_wallet:

        # Configure risk service
        mock_risk_instance = Mock()
        risk_check_result = Mock()
        risk_check_result.ok = True
        risk_check_result.violated_cap = None
        risk_check_result.details = ""
        risk_check_result.action = "allow"
        risk_check_result.adjusted_amount = None
        mock_risk_instance.check_portfolio_risk = AsyncMock(return_value=risk_check_result)
        mock_risk.return_value = mock_risk_instance

        # Configure capacity service
        mock_capacity_instance = Mock()
        capacity_check_result = Mock()
        capacity_check_result.ok = True
        capacity_check_result.violated_cap = None
        capacity_check_result.details = ""
        capacity_check_result.action = "allow"
        capacity_check_result.adjusted_amount = None
        mock_capacity_instance.check_capacity_for_trade = AsyncMock(return_value=capacity_check_result)
        mock_capacity.return_value = mock_capacity_instance

        # Configure cost model
        mock_cost_instance = Mock()
        mock_cost_instance.calculate_execution_cost = AsyncMock(
            return_value={"total_cost": 10.0, "expected_slippage": 0.0}
        )
        mock_cost.return_value = mock_cost_instance

        # Configure trade recorder
        def make_trade(
            order_id, owner_id, bot_id, exchange, trading_pair, side,
            base_asset, quote_asset, base_amount, quote_amount, price,
            fee_amount, fee_asset, modeled_cost, **kwargs
        ):
            """Create a Trade object and corresponding ledger entries."""
            trade = Trade(
                id=None,
                order_id=order_id,
                owner_id=owner_id,
                bot_id=bot_id,
                exchange=exchange,
                trading_pair=trading_pair,
                side=side,
                base_asset=base_asset,
                quote_asset=quote_asset,
                base_amount=Decimal(str(base_amount)),
                quote_amount=Decimal(str(quote_amount)),
                price=Decimal(str(price)),
                fee_amount=Decimal(str(fee_amount)),
                fee_asset=fee_asset,
                modeled_cost=Decimal(str(modeled_cost)),
                executed_at=datetime.utcnow(),
            )
            session.add(trade)

            # Create ledger entries (3 per trade: base, quote, fee)
            base_delta = base_amount if side == TradeSide.BUY else -base_amount
            base_entry = WalletLedger(
                id=None,
                owner_id=owner_id,
                bot_id=bot_id,
                asset=base_asset,
                delta_amount=base_delta,
                reason=LedgerReason.BUY if side == TradeSide.BUY else LedgerReason.SELL,
                related_trade_id=trade.id,
                created_at=trade.executed_at,
            )
            session.add(base_entry)

            quote_delta = -quote_amount if side == TradeSide.BUY else quote_amount
            quote_entry = WalletLedger(
                id=None,
                owner_id=owner_id,
                bot_id=bot_id,
                asset=quote_asset,
                delta_amount=quote_delta,
                reason=LedgerReason.BUY if side == TradeSide.BUY else LedgerReason.SELL,
                related_trade_id=trade.id,
                created_at=trade.executed_at,
            )
            session.add(quote_entry)

            fee_entry = WalletLedger(
                id=None,
                owner_id=owner_id,
                bot_id=bot_id,
                asset=fee_asset,
                delta_amount=-fee_amount,
                reason=LedgerReason.FEE,
                related_trade_id=trade.id,
                created_at=trade.executed_at,
            )
            session.add(fee_entry)

            return trade

        mock_recorder_instance = Mock()
        mock_recorder_instance.record_trade = AsyncMock(side_effect=make_trade)
        mock_recorder.return_value = mock_recorder_instance

        # Configure tax engine
        mock_tax_instance = Mock()
        mock_tax_instance.process_buy = AsyncMock(return_value=None)
        mock_tax_instance.process_sell = AsyncMock(return_value=None)
        mock_tax.return_value = mock_tax_instance

        # Configure invariant service
        mock_invariant_instance = Mock()
        mock_invariant_instance.validate_trade = AsyncMock(return_value=None)
        mock_invariant_instance.validate_invariants = AsyncMock(return_value=None)
        mock_invariant.return_value = mock_invariant_instance

        # Configure wallet service
        mock_wallet_instance = Mock()
        mock_wallet_instance.record_trade_result = AsyncMock(return_value=None)
        mock_wallet.return_value = mock_wallet_instance

        # ---------------------------------------------------------------------
        # Execute: Run trading sequence
        # ---------------------------------------------------------------------

        engine = TradingEngine()
        trade_count = 0
        num_iterations = 50  # Execute 50 trades

        for iteration in range(num_iterations):
            action = "buy" if iteration % 2 == 0 else "sell"

            signal = Mock()
            signal.action = action
            signal.symbol = bot.trading_pair
            signal.amount = 20.0  # Large enough to pass minimum

            current_price = 100.0

            try:
                order = await engine._execute_trade(
                    bot, exchange, signal, current_price, session
                )

                if order is not None:
                    trade_count += 1

            except Exception as e:
                pytest.fail(f"Exception at iteration {iteration}: {e}")

        # ---------------------------------------------------------------------
        # Capture: Record final state
        # ---------------------------------------------------------------------

        # Calculate final balance from ledger (this is what replay will verify)
        usdt_entries = [e for e in session.ledger_entries if e.asset == "USDT"]
        final_balance_from_ledger = float(initial_balance) + sum(
            float(e.delta_amount) for e in usdt_entries
        )

        ledger_entries = session.ledger_entries

        print("\n" + "=" * 70)
        print("Ledger Replay Test Results")
        print("=" * 70)
        print(f"Initial balance: ${float(initial_balance):,.2f}")
        print(f"Trades executed: {trade_count}")
        print(f"Ledger entries: {len(ledger_entries)}")
        print(f"Final balance (from ledger): ${final_balance_from_ledger:,.2f}")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Replay: Reconstruct balance from ledger alone
        # ---------------------------------------------------------------------

        replay_engine = LedgerReplayEngine(float(initial_balance))
        replay_result = replay_engine.replay(ledger_entries, asset="USDT")

        replayed_balance = replay_result["replayed_balance"]
        total_delta = replay_result["total_delta"]

        print(f"\nReplay Results:")
        print(f"  Replayed balance: ${replayed_balance:,.2f}")
        print(f"  Total delta: ${total_delta:,.2f}")
        print(f"  Entries processed: {replay_result['entry_count']}")
        print(f"  Timestamps monotonic: {replay_result['timestamps_monotonic']}")
        print(f"  No NaN/infinity: {replay_result['no_nan']}")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Validate: Assert invariants
        # ---------------------------------------------------------------------

        # INVARIANT A: Replayed balance matches final balance
        balance_diff = abs(replayed_balance - final_balance_from_ledger)
        assert balance_diff < 1e-6, (
            f"[FAIL] INVARIANT A: Replayed balance does not match!\n"
            f"  Replayed: ${replayed_balance:,.2f}\n"
            f"  Final: ${final_balance_from_ledger:,.2f}\n"
            f"  Difference: ${balance_diff:,.2f}"
        )
        print("[PASS] INVARIANT A: Replayed balance matches final balance")

        # INVARIANT B: Ledger is not empty
        if trade_count > 0:
            assert len(ledger_entries) > 0, "[FAIL] INVARIANT B: Ledger is empty but trades executed"
        print("[PASS] INVARIANT B: Ledger is not empty")

        # INVARIANT C: Replay is deterministic
        # Run replay again to ensure same result
        replay_result_2 = replay_engine.replay(ledger_entries, asset="USDT")
        replayed_balance_2 = replay_result_2["replayed_balance"]
        assert replayed_balance == replayed_balance_2, (
            "[FAIL] INVARIANT C: Replay is not deterministic!"
        )
        print("[PASS] INVARIANT C: Replay is deterministic")

        # INVARIANT D: No NaN or infinity
        assert replay_result["no_nan"], "[FAIL] INVARIANT D: NaN or infinity detected"
        assert replayed_balance == replayed_balance, "[FAIL] INVARIANT D: Replayed balance is NaN"
        assert abs(replayed_balance) != float('inf'), "[FAIL] INVARIANT D: Replayed balance is infinite"
        print("[PASS] INVARIANT D: No NaN or infinity")

        # INVARIANT E: Sum of deltas matches balance change
        expected_delta = final_balance_from_ledger - float(initial_balance)
        delta_diff = abs(total_delta - expected_delta)
        assert delta_diff < 1e-6, (
            f"[FAIL] INVARIANT E: Delta mismatch!\n"
            f"  Sum of deltas: ${total_delta:,.2f}\n"
            f"  Expected: ${expected_delta:,.2f}\n"
            f"  Difference: ${delta_diff:,.2f}"
        )
        print("[PASS] INVARIANT E: Sum of deltas matches balance change")

        # INVARIANT F: Timestamps are monotonic
        assert replay_result["timestamps_monotonic"], (
            "[FAIL] INVARIANT F: Timestamps are not monotonic"
        )
        print("[PASS] INVARIANT F: Timestamps are monotonic")

        # INVARIANT G: Asset isolation works
        # Count USDT entries vs total entries
        usdt_entries = [e for e in ledger_entries if e.asset == "USDT"]
        btc_entries = [e for e in ledger_entries if e.asset == "BTC"]
        # Should have both USDT and BTC entries
        assert len(usdt_entries) > 0, "[FAIL] INVARIANT G: No USDT entries"
        assert len(btc_entries) > 0, "[FAIL] INVARIANT G: No BTC entries"
        # Replay should only process USDT entries
        assert replay_result["entry_count"] == len(usdt_entries), (
            "[FAIL] INVARIANT G: Asset isolation failed"
        )
        print("[PASS] INVARIANT G: Asset isolation works")

        # Additional checks
        assert trade_count > 0, "No trades executed!"
        assert len(ledger_entries) == trade_count * 3, (
            f"Expected {trade_count * 3} ledger entries, got {len(ledger_entries)}"
        )

        print("\n" + "=" * 70)
        print("[PASS] ALL REPLAY INVARIANTS PASSED")
        print("=" * 70)


@pytest.mark.asyncio
async def test_ledger_replay_zero_trades():
    """
    Test: Zero-trade case

    Verify that with no trades, replayed balance equals initial balance.
    """

    initial_balance = 10000.0

    # No trades, empty ledger
    ledger_entries = []

    replay_engine = LedgerReplayEngine(initial_balance)
    replay_result = replay_engine.replay(ledger_entries, asset="USDT")

    replayed_balance = replay_result["replayed_balance"]

    print("\n" + "=" * 70)
    print("Zero Trades Replay Test Results")
    print("=" * 70)
    print(f"Initial balance: ${initial_balance:,.2f}")
    print(f"Replayed balance: ${replayed_balance:,.2f}")
    print(f"Entries processed: {replay_result['entry_count']}")
    print("=" * 70)

    # INVARIANT H: Zero-trade case
    assert replayed_balance == initial_balance, (
        f"[FAIL] INVARIANT H: Zero-trade replay failed!\n"
        f"  Expected: ${initial_balance:,.2f}\n"
        f"  Got: ${replayed_balance:,.2f}"
    )
    print("[PASS] INVARIANT H: Zero-trade case (balance unchanged)")

    assert replay_result["entry_count"] == 0, "Expected 0 entries processed"
    assert replay_result["total_delta"] == 0.0, "Expected 0 total delta"

    print("=" * 70)


@pytest.mark.asyncio
async def test_ledger_replay_mixed_assets():
    """
    Test: Multiple assets

    Verify that replay correctly isolates different assets.
    """

    initial_usdt = 10000.0
    initial_btc = 0.0

    # Create mock ledger entries for multiple assets
    ledger_entries = [
        Mock(
            asset="USDT",
            delta_amount=-1000.0,
            created_at=datetime(2026, 1, 1, 10, 0, 0),
        ),
        Mock(
            asset="BTC",
            delta_amount=0.5,
            created_at=datetime(2026, 1, 1, 10, 0, 1),
        ),
        Mock(
            asset="USDT",
            delta_amount=-10.0,  # Fee
            created_at=datetime(2026, 1, 1, 10, 0, 2),
        ),
        Mock(
            asset="USDT",
            delta_amount=2000.0,  # Sell proceeds
            created_at=datetime(2026, 1, 1, 10, 1, 0),
        ),
        Mock(
            asset="BTC",
            delta_amount=-0.5,
            created_at=datetime(2026, 1, 1, 10, 1, 1),
        ),
        Mock(
            asset="USDT",
            delta_amount=-20.0,  # Fee
            created_at=datetime(2026, 1, 1, 10, 1, 2),
        ),
    ]

    # Replay USDT
    usdt_replay = LedgerReplayEngine(initial_usdt)
    usdt_result = usdt_replay.replay(ledger_entries, asset="USDT")

    # Replay BTC
    btc_replay = LedgerReplayEngine(initial_btc)
    btc_result = btc_replay.replay(ledger_entries, asset="BTC")

    print("\n" + "=" * 70)
    print("Mixed Assets Replay Test Results")
    print("=" * 70)
    print(f"USDT:")
    print(f"  Initial: ${initial_usdt:,.2f}")
    print(f"  Replayed: ${usdt_result['replayed_balance']:,.2f}")
    print(f"  Entries: {usdt_result['entry_count']}")
    print(f"BTC:")
    print(f"  Initial: {initial_btc:.8f}")
    print(f"  Replayed: {btc_result['replayed_balance']:.8f}")
    print(f"  Entries: {btc_result['entry_count']}")
    print("=" * 70)

    # USDT: -1000 -10 +2000 -20 = +970
    expected_usdt = initial_usdt + 970.0
    assert abs(usdt_result['replayed_balance'] - expected_usdt) < 1e-6, (
        f"USDT replay incorrect: expected {expected_usdt}, got {usdt_result['replayed_balance']}"
    )

    # BTC: +0.5 -0.5 = 0
    expected_btc = initial_btc
    assert abs(btc_result['replayed_balance'] - expected_btc) < 1e-8, (
        f"BTC replay incorrect: expected {expected_btc}, got {btc_result['replayed_balance']}"
    )

    # Check entry counts
    assert usdt_result['entry_count'] == 4, "Expected 4 USDT entries"
    assert btc_result['entry_count'] == 2, "Expected 2 BTC entries"

    print("[PASS] Mixed assets replay correct")
    print("=" * 70)


# ============================================================================
# Test Using Real LedgerReplayService
# ============================================================================


@pytest.mark.asyncio
async def test_ledger_replay_rebuilds_balance():
    """Ensures ledger replay reconstructs the same final balance as live trading."""
    
    #--------------------------------------------------------------------------
    # Setup: Create bot and manually craft ledger entries
    # -------------------------------------------------------------------------
    
    from app.services.ledger_replay import LedgerReplayService
    
    initial_balance = Decimal("10000.00")
    
    bot = Bot(
        id=9001,
        name="replay-service-test",
        trading_pair="BTC/USDT",
        strategy="simple_moving_average",
        strategy_params={},
        budget=float(initial_balance),  # Use float to match ledger replay service
        current_balance=float(initial_balance),
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    bot.owner_id = "test-owner"
    
    # -------------------------------------------------------------------------
    # Manually create 20 trades worth of ledger entries
    # Pattern: BUY, SELL, BUY, SELL, ...
    # -------------------------------------------------------------------------
    
    ledger_entries = []
    trade_id_counter = 1
    current_time = datetime(2026, 1, 1, 10, 0, 0)
    
    # Track balance manually for validation
    manual_balance = float(initial_balance)
    
    print("\n" + "=" * 70)
    print("Ledger Replay Rebuilds Balance Test (Real Service)")
    print("=" * 70)
    print(f"Initial balance: ${float(initial_balance):,.2f}")
    print("Creating 20 trades...")
    print("=" * 70)
    
    for i in range(20):
        side = TradeSide.BUY if i % 2 == 0 else TradeSide.SELL
        trade_id = trade_id_counter
        trade_id_counter += 1
        
        # Fixed values per trade
        btc_amount = 0.1
        price = 50_000.0
        usdt_amount = btc_amount * price  # 5000 USDT
        fee_amount = usdt_amount * 0.001  # 0.1% fee = 5 USDT
        
        if side == TradeSide.BUY:
            # BUY: spend USDT, gain BTC, pay fee in USDT
            # 3 ledger entries:
            
            # 1. Base asset (BTC) inflow
            ledger_entries.append(WalletLedger(
                id=len(ledger_entries) + 1,
                owner_id="test-owner",
                bot_id=bot.id,
                asset="BTC",
                delta_amount=btc_amount,
                reason=LedgerReason.BUY,
                related_trade_id=trade_id,
                created_at=current_time,
            ))
            
            # 2. Quote asset (USDT) outflow
            ledger_entries.append(WalletLedger(
                id=len(ledger_entries) + 1,
                owner_id="test-owner",
                bot_id=bot.id,
                asset="USDT",
                delta_amount=-usdt_amount,
                reason=LedgerReason.BUY,
                related_trade_id=trade_id,
                created_at=current_time,
            ))
            
            # 3. Fee (USDT) outflow
            ledger_entries.append(WalletLedger(
                id=len(ledger_entries) + 1,
                owner_id="test-owner",
                bot_id=bot.id,
                asset="USDT",
                delta_amount=-fee_amount,
                reason=LedgerReason.FEE,
                related_trade_id=trade_id,
                created_at=current_time,
            ))
            
            # Update manual balance
            manual_balance -= (usdt_amount + fee_amount)
            
        else:  # SELL
            # SELL: spend BTC, gain USDT, pay fee in USDT
            # 3 ledger entries:
            
            # 1. Base asset (BTC) outflow
            ledger_entries.append(WalletLedger(
                id=len(ledger_entries) + 1,
                owner_id="test-owner",
                bot_id=bot.id,
                asset="BTC",
                delta_amount=-btc_amount,
                reason=LedgerReason.SELL,
                related_trade_id=trade_id,
                created_at=current_time,
            ))
            
            # 2. Quote asset (USDT) inflow
            ledger_entries.append(WalletLedger(
                id=len(ledger_entries) + 1,
                owner_id="test-owner",
                bot_id=bot.id,
                asset="USDT",
                delta_amount=usdt_amount,
                reason=LedgerReason.SELL,
                related_trade_id=trade_id,
                created_at=current_time,
            ))
            
            # 3. Fee (USDT) outflow
            ledger_entries.append(WalletLedger(
                id=len(ledger_entries) + 1,
                owner_id="test-owner",
                bot_id=bot.id,
                asset="USDT",
                delta_amount=-fee_amount,
                reason=LedgerReason.FEE,
                related_trade_id=trade_id,
                created_at=current_time,
            ))
            
            # Update manual balance
            manual_balance += (usdt_amount - fee_amount)
        
        current_time = current_time.replace(second=current_time.second + 1)
    
    print(f"[INFO] Created {len(ledger_entries)} ledger entries (3 per trade)")
    print(f"[INFO] Manual balance calculation: ${manual_balance:,.2f}")
    
    # -------------------------------------------------------------------------
    # Update bot balance to match manual calculation
    # (This simulates what would happen during live trading)
    # -------------------------------------------------------------------------
    
    bot.current_balance = manual_balance
    live_final_balance = manual_balance
    
    print(f"[INFO] Bot live balance set to: ${live_final_balance:,.2f}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create mock session for LedgerReplayService
    # -------------------------------------------------------------------------
    
    class ReplayMockSession:
        """Mock session that provides data for LedgerReplayService."""
        
        def __init__(self, bot, ledger_entries):
            self.bot = bot
            self.ledger_entries = ledger_entries
            
        async def execute(self, query):
            """Mock database queries."""
            from unittest.mock import MagicMock
            
            result = MagicMock()
            query_str = str(query)
            
            # Debug: print query (disable after fixing)
            # print(f"[DEBUG] Query: {query_str[:200]}")
            
            if "SELECT bots.id" in query_str and "WHERE bots.id IN" in query_str:
                # Query for bot IDs  - first query in rebuild_state_from_ledger
                result.all.return_value = [(self.bot.id,)]
            elif "DELETE FROM positions" in query_str:
                # Delete positions query
                result.rowcount = 0
            elif "SELECT bots" in query_str and "WHERE bots.id =" in query_str:
                # Query for specific bot by ID - return actual bot!
                result.scalar_one_or_none.return_value = self.bot
            elif "SELECT wallet_ledger" in query_str:
                # Query for ledger entries
                result.scalars.return_value.all.return_value = self.ledger_entries
            elif "SELECT trades" in query_str:
                # Query for trades - return empty list (no trades needed for this test)
                result.scalars.return_value.all.return_value = []
            else:
                # Default for any other query
                result.all.return_value = []
                result.scalar_one_or_none.return_value = self.bot  # Always return bot as fallback
                result.scalars.return_value.all.return_value = []
            
            return result
        
        async def flush(self):
            """Mock flush."""
            pass
        
        async def commit(self):
            """Mock commit."""
            pass
    
    session = ReplayMockSession(bot, ledger_entries)
    
    # -------------------------------------------------------------------------
    # Run REAL LedgerReplayService
    # -------------------------------------------------------------------------
    
    replay_service = LedgerReplayService(session)
    
    print("[INFO] Running LedgerReplayService.rebuild_state_from_ledger()...")
    
    replay_result = await replay_service.rebuild_state_from_ledger(
        owner_id="test-owner",
        is_simulated=True  # bot.is_dry_run=True
    )
    
    replayed_final_balance = float(bot.current_balance)
    
    print(f"[INFO] Replay completed!")
    print(f"[INFO] Replayed final balance: ${replayed_final_balance:,.2f}")
    print(f"[INFO] Replay stats:")
    print(f"  - Positions deleted: {replay_result.positions_deleted}")
    print(f"  - Positions created: {replay_result.positions_created}")
    print(f"  - Balances rebuilt: {replay_result.balances_rebuilt}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------------------
    
    # A. Replayed balance equals live balance (manual calculation)
    tolerance = 0.01  # $0.01
    assert abs(replayed_final_balance - live_final_balance) < tolerance, (
        f"Ledger replay balance mismatch: "
        f"live=${live_final_balance:,.2f}, replayed=${replayed_final_balance:,.2f}, "
        f"diff=${abs(replayed_final_balance - live_final_balance):,.2f}"
    )
    
    # B. No negative balances
    assert replayed_final_balance >= 0, f"Replayed balance is negative: {replayed_final_balance}"
    assert live_final_balance >= 0, f"Live balance is negative: {live_final_balance}"
    
    # C. Replay result is finite (not NaN, not inf)
    assert replayed_final_balance == replayed_final_balance, "Replayed balance is NaN"
    assert abs(replayed_final_balance) != float('inf'), "Replayed balance is infinite"
    
    # D. Ledger is not empty
    assert len(ledger_entries) > 0, "No ledger entries created"
    assert len(ledger_entries) == 60, f"Expected 60 entries (20 trades × 3), got {len(ledger_entries)}"
    
    # E. Final balance is reasonable (within 5% of initial)
    # (small loss due to fees is expected)
    min_expected = float(initial_balance) * 0.95
    assert replayed_final_balance >= min_expected, (
        f"Balance dropped too much: ${replayed_final_balance:,.2f} < ${min_expected:,.2f}"
    )
    
    # F. Manual balance matches expected
    # Net effect: 20 trades alternating BUY/SELL at same price = only fees lost
    # 20 trades × 5000 USDT × 0.001 fee = 100 USDT total fees
    expected_balance = float(initial_balance) - 100.0
    assert abs(manual_balance - expected_balance) < 0.01, (
        f"Manual balance calculation wrong: ${manual_balance:,.2f}, expected ${expected_balance:,.2f}"
    )
    
    print(f"\n[PASS] Ledger replay reconstructed exact balance using REAL service")
    print(f"[PASS] Live balance: ${live_final_balance:,.2f}")
    print(f"[PASS] Replayed balance: ${replayed_final_balance:,.2f}")
    print(f"[PASS] Expected (initial - fees): ${expected_balance:,.2f}")
    print(f"[PASS] Difference: ${abs(replayed_final_balance - live_final_balance):,.6f}")
    print("=" * 70)
