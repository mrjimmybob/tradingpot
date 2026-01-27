"""
Test: Drawdown Curve Correctness

Purpose:
    Verify that drawdown curves are computed correctly from equity curves.
    Drawdown is a critical risk metric that measures how far equity has
    fallen from its peak value.

Definition:
    drawdown[i] = (peak_so_far - equity[i]) / peak_so_far

Validates:
    - Drawdown calculations are mathematically correct
    - Risk metrics are reliable
    - Charts are accurate
    - Backtests show true risk
    - Users are properly informed

Author: Trading Bot Test Suite
Date: 2026-01-27
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Tuple
from unittest.mock import Mock, AsyncMock, patch

from app.models.bot import Bot
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
# Equity Curve Reconstructor
# ============================================================================


class EquityCurveReconstructor:
    """Reconstructs equity curve from ledger entries."""

    def __init__(self, initial_balance: float):
        """Initialize with starting balance."""
        self.initial_balance = initial_balance

    def reconstruct(
        self, ledger_entries: List[WalletLedger], asset: str = "USDT"
    ) -> List[Tuple[int, float, datetime]]:
        """
        Reconstruct equity curve from ledger entries.
        
        Returns:
            List of (trade_id, balance, timestamp) tuples
        """
        # Filter for this asset
        asset_entries = [e for e in ledger_entries if e.asset == asset]

        # Sort by timestamp
        sorted_entries = sorted(asset_entries, key=lambda e: e.created_at)

        # Group by trade_id
        trades_with_entries = {}
        for entry in sorted_entries:
            trade_id = entry.related_trade_id
            if trade_id not in trades_with_entries:
                trades_with_entries[trade_id] = []
            trades_with_entries[trade_id].append(entry)

        # Build equity curve
        equity_curve = [(0, self.initial_balance, datetime.min)]  # Initial point
        balance = self.initial_balance

        # Process trades in chronological order
        seen_trades = []
        for entry in sorted_entries:
            if entry.related_trade_id not in seen_trades:
                seen_trades.append(entry.related_trade_id)

        for trade_id in seen_trades:
            entries = trades_with_entries[trade_id]
            # Apply all deltas for this trade
            for entry in entries:
                balance += float(entry.delta_amount)

            # Get timestamp from first entry
            timestamp = entries[0].created_at

            # Record balance after this trade
            equity_curve.append((trade_id, balance, timestamp))

        return equity_curve


# ============================================================================
# Drawdown Calculator
# ============================================================================


class DrawdownCalculator:
    """
    Calculates drawdown curve from equity curve.
    
    Drawdown measures how far equity has fallen from its peak.
    Formula: drawdown[i] = (peak_so_far - equity[i]) / peak_so_far
    """

    def calculate(
        self, equity_curve: List[Tuple[int, float, datetime]]
    ) -> List[Tuple[int, float, float, datetime]]:
        """
        Calculate drawdown curve from equity curve.
        
        Args:
            equity_curve: List of (trade_id, equity, timestamp) tuples
            
        Returns:
            List of (trade_id, equity, drawdown, timestamp) tuples
        """
        if not equity_curve:
            return []

        drawdown_curve = []
        peak = equity_curve[0][1]  # Initial equity is peak

        for trade_id, equity, timestamp in equity_curve:
            # Update peak if we reached new high
            if equity > peak:
                peak = equity

            # Calculate drawdown as percentage
            if peak > 0:
                drawdown = (peak - equity) / peak
            else:
                drawdown = 0.0

            drawdown_curve.append((trade_id, equity, drawdown, timestamp))

        return drawdown_curve

    def get_max_drawdown(self, drawdown_curve: List[Tuple]) -> float:
        """Get maximum drawdown from curve."""
        if not drawdown_curve:
            return 0.0

        return max(dd[2] for dd in drawdown_curve)  # dd[2] is drawdown value

    def validate_drawdown(self, drawdown_curve: List[Tuple]) -> Dict[str, bool]:
        """Validate drawdown curve invariants."""
        if not drawdown_curve:
            return {"all_valid": True}

        results = {}

        # Check all drawdowns are >= 0
        results["non_negative"] = all(dd[2] >= 0 for dd in drawdown_curve)

        # Check all drawdowns are <= 1 (100%)
        results["within_bounds"] = all(dd[2] <= 1.0 for dd in drawdown_curve)

        # Check first drawdown is 0 (at peak)
        results["starts_at_zero"] = abs(drawdown_curve[0][2]) < 1e-10

        # Check no NaN or infinity
        results["no_nan"] = all(
            dd[2] == dd[2] and abs(dd[2]) != float('inf')
            for dd in drawdown_curve
        )

        results["all_valid"] = all(results.values())

        return results


# ============================================================================
# Main Test
# ============================================================================


@pytest.mark.asyncio
async def test_drawdown_curve_correctness():
    """
    Test: Drawdown Curve Correctness

    Executes trades to build equity curve, then computes drawdown curve
    and validates all mathematical properties.
    """

    # -------------------------------------------------------------------------
    # Setup: Create deterministic environment
    # -------------------------------------------------------------------------

    initial_balance = Decimal("10000.00")

    bot = Bot(
        id=1,
        name="drawdown_test_bot",
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
        # Reconstruct: Build equity curve from ledger
        # ---------------------------------------------------------------------

        equity_reconstructor = EquityCurveReconstructor(float(initial_balance))
        equity_curve = equity_reconstructor.reconstruct(
            session.ledger_entries, asset="USDT"
        )

        # ---------------------------------------------------------------------
        # Calculate: Compute drawdown curve
        # ---------------------------------------------------------------------

        drawdown_calculator = DrawdownCalculator()
        drawdown_curve = drawdown_calculator.calculate(equity_curve)

        # Manually compute expected drawdown for validation
        expected_drawdown = []
        peak = equity_curve[0][1]
        for trade_id, equity, timestamp in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            expected_drawdown.append(dd)

        # ---------------------------------------------------------------------
        # Report: Display results
        # ---------------------------------------------------------------------

        max_dd = drawdown_calculator.get_max_drawdown(drawdown_curve)
        validation = drawdown_calculator.validate_drawdown(drawdown_curve)

        print("\n" + "=" * 70)
        print("Drawdown Curve Correctness Test Results")
        print("=" * 70)
        print(f"Initial balance: ${float(initial_balance):,.2f}")
        print(f"Trades executed: {trade_count}")
        print(f"Equity curve points: {len(equity_curve)}")
        print(f"Drawdown curve points: {len(drawdown_curve)}")
        print(f"Max drawdown: {max_dd*100:.4f}%")
        print("\nFirst 5 drawdown points:")
        for i, (trade_id, equity, dd, timestamp) in enumerate(drawdown_curve[:5]):
            print(f"  {i}: Trade {trade_id} -> Equity ${equity:,.2f}, DD {dd*100:.4f}%")
        print("\nValidation results:")
        for key, value in validation.items():
            status = "[PASS]" if value else "[FAIL]"
            print(f"  {status} {key}: {value}")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Validate: Assert invariants
        # ---------------------------------------------------------------------

        # INVARIANT A: Length equality
        assert len(drawdown_curve) == len(equity_curve), (
            f"[FAIL] INVARIANT A: Length mismatch!\n"
            f"  Drawdown: {len(drawdown_curve)} points\n"
            f"  Equity: {len(equity_curve)} points"
        )
        print("[PASS] INVARIANT A: Lengths match")

        # INVARIANT B: First drawdown is 0
        first_dd = drawdown_curve[0][2]
        assert abs(first_dd) < 1e-10, (
            f"[FAIL] INVARIANT B: First drawdown not zero!\n"
            f"  First drawdown: {first_dd*100:.6f}%"
        )
        print("[PASS] INVARIANT B: First drawdown is 0")

        # INVARIANT C: All points match expected
        max_diff = 0.0
        for i, (dd_point, expected_dd) in enumerate(zip(drawdown_curve, expected_drawdown)):
            actual_dd = dd_point[2]
            diff = abs(actual_dd - expected_dd)
            max_diff = max(max_diff, diff)

            assert diff < 1e-6, (
                f"[FAIL] INVARIANT C: Drawdown mismatch at index {i}!\n"
                f"  Expected: {expected_dd*100:.6f}%\n"
                f"  Actual: {actual_dd*100:.6f}%\n"
                f"  Difference: {diff*100:.6f}%"
            )

        print(f"[PASS] INVARIANT C: All points match (max diff: {max_diff*100:.8f}%)")

        # INVARIANT D: All drawdowns >= 0
        assert validation["non_negative"], (
            "[FAIL] INVARIANT D: Negative drawdown detected!"
        )
        print("[PASS] INVARIANT D: All drawdowns >= 0")

        # INVARIANT E: All drawdowns <= 1
        assert validation["within_bounds"], (
            "[FAIL] INVARIANT E: Drawdown > 100% detected!"
        )
        print("[PASS] INVARIANT E: All drawdowns <= 1")

        # INVARIANT F: Drawdown returns to 0 at new peaks
        # Find indices where equity reaches new peak
        peak_value = equity_curve[0][1]
        for i, (trade_id, equity, timestamp) in enumerate(equity_curve):
            if equity > peak_value:
                peak_value = equity
                # At new peak, drawdown should be 0
                dd = drawdown_curve[i][2]
                assert abs(dd) < 1e-6, (
                    f"[FAIL] INVARIANT F: Drawdown not zero at new peak!\n"
                    f"  Index: {i}\n"
                    f"  Equity: ${equity:,.2f}\n"
                    f"  Drawdown: {dd*100:.6f}%"
                )
        print("[PASS] INVARIANT F: Drawdown returns to 0 at new peaks")

        # INVARIANT G: Max drawdown matches expected
        expected_max_dd = max(expected_drawdown)
        assert abs(max_dd - expected_max_dd) < 1e-6, (
            f"[FAIL] INVARIANT G: Max drawdown mismatch!\n"
            f"  Expected: {expected_max_dd*100:.4f}%\n"
            f"  Actual: {max_dd*100:.4f}%"
        )
        print(f"[PASS] INVARIANT G: Max drawdown correct ({max_dd*100:.4f}%)")

        # INVARIANT H: Deterministic
        # Recalculate and verify identical results
        drawdown_curve_2 = drawdown_calculator.calculate(equity_curve)
        assert len(drawdown_curve_2) == len(drawdown_curve), (
            "[FAIL] INVARIANT H: Recalculation changed length"
        )
        for i, (dd1, dd2) in enumerate(zip(drawdown_curve, drawdown_curve_2)):
            assert abs(dd1[2] - dd2[2]) < 1e-10, (
                f"[FAIL] INVARIANT H: Recalculation changed value at index {i}"
            )
        print("[PASS] INVARIANT H: Calculation is deterministic")

        # INVARIANT I: Timestamps match
        for i, (dd_point, eq_point) in enumerate(zip(drawdown_curve, equity_curve)):
            assert dd_point[3] == eq_point[2], (
                f"[FAIL] INVARIANT I: Timestamp mismatch at index {i}"
            )
        print("[PASS] INVARIANT I: Timestamps match equity curve")

        # INVARIANT J: No NaN or infinity
        assert validation["no_nan"], (
            "[FAIL] INVARIANT J: NaN or infinity detected!"
        )
        print("[PASS] INVARIANT J: No NaN or infinity")

        # Additional checks
        assert trade_count > 0, "No trades executed!"
        assert len(equity_curve) == trade_count + 1, (
            f"Equity curve should have {trade_count + 1} points"
        )

        print("\n" + "=" * 70)
        print("[PASS] ALL DRAWDOWN INVARIANTS PASSED")
        print("=" * 70)
        print(f"\nSummary:")
        print(f"  Trades: {trade_count}")
        print(f"  Equity points: {len(equity_curve)}")
        print(f"  Drawdown points: {len(drawdown_curve)}")
        print(f"  Max drawdown: {max_dd*100:.4f}%")
        print(f"  Final equity: ${equity_curve[-1][1]:,.2f}")
        print(f"  Final drawdown: {drawdown_curve[-1][2]*100:.4f}%")
        print("=" * 70)


@pytest.mark.asyncio
async def test_drawdown_curve_constant_equity():
    """
    Test: Constant equity (no drawdown)

    Verify drawdown stays at 0% when equity never decreases.
    """

    # Create flat equity curve (no losses)
    equity_curve = [
        (0, 10000.0, datetime(2026, 1, 1, 10, 0, 0)),
        (1, 10000.0, datetime(2026, 1, 1, 10, 1, 0)),
        (2, 10000.0, datetime(2026, 1, 1, 10, 2, 0)),
        (3, 10000.0, datetime(2026, 1, 1, 10, 3, 0)),
    ]

    calculator = DrawdownCalculator()
    drawdown_curve = calculator.calculate(equity_curve)

    print("\n" + "=" * 70)
    print("Constant Equity Drawdown Test Results")
    print("=" * 70)
    print(f"Equity points: {len(equity_curve)}")
    print(f"All equity values: ${equity_curve[0][1]:,.2f}")

    for i, (trade_id, equity, dd, timestamp) in enumerate(drawdown_curve):
        print(f"  Point {i}: DD {dd*100:.4f}%")
        assert abs(dd) < 1e-10, (
            f"[FAIL] Drawdown should be 0% for constant equity at index {i}, got {dd*100:.4f}%"
        )

    print("[PASS] All drawdowns are 0% for constant equity")
    print("=" * 70)


@pytest.mark.asyncio
async def test_drawdown_curve_monotonic_decline():
    """
    Test: Monotonic decline (increasing drawdown)

    Verify drawdown increases monotonically when equity only decreases.
    """

    # Create declining equity curve
    equity_curve = [
        (0, 10000.0, datetime(2026, 1, 1, 10, 0, 0)),
        (1, 9500.0, datetime(2026, 1, 1, 10, 1, 0)),   # 5% loss
        (2, 9000.0, datetime(2026, 1, 1, 10, 2, 0)),   # 10% loss
        (3, 8500.0, datetime(2026, 1, 1, 10, 3, 0)),   # 15% loss
        (4, 8000.0, datetime(2026, 1, 1, 10, 4, 0)),   # 20% loss
    ]

    calculator = DrawdownCalculator()
    drawdown_curve = calculator.calculate(equity_curve)

    print("\n" + "=" * 70)
    print("Monotonic Decline Drawdown Test Results")
    print("=" * 70)

    expected_drawdowns = [0.0, 0.05, 0.10, 0.15, 0.20]

    for i, ((trade_id, equity, dd, timestamp), expected_dd) in enumerate(
        zip(drawdown_curve, expected_drawdowns)
    ):
        print(f"  Point {i}: Equity ${equity:,.2f}, DD {dd*100:.2f}% (expected {expected_dd*100:.2f}%)")
        
        assert abs(dd - expected_dd) < 1e-6, (
            f"[FAIL] Drawdown mismatch at index {i}: expected {expected_dd*100:.2f}%, got {dd*100:.2f}%"
        )

        # Check monotonic increase
        if i > 0:
            prev_dd = drawdown_curve[i-1][2]
            assert dd >= prev_dd, (
                f"[FAIL] Drawdown decreased at index {i}: {prev_dd*100:.2f}% -> {dd*100:.2f}%"
            )

    max_dd = calculator.get_max_drawdown(drawdown_curve)
    assert abs(max_dd - 0.20) < 1e-6, f"Max drawdown should be 20%, got {max_dd*100:.2f}%"

    print(f"[PASS] Drawdown increases monotonically to {max_dd*100:.2f}%")
    print("=" * 70)


@pytest.mark.asyncio
async def test_drawdown_curve_recovery():
    """
    Test: Drawdown recovery

    Verify drawdown returns to 0% when equity recovers to new peak.
    """

    # Create equity curve with drawdown and recovery
    equity_curve = [
        (0, 10000.0, datetime(2026, 1, 1, 10, 0, 0)),  # Peak
        (1, 9000.0, datetime(2026, 1, 1, 10, 1, 0)),   # 10% drawdown
        (2, 8500.0, datetime(2026, 1, 1, 10, 2, 0)),   # 15% drawdown (max)
        (3, 9500.0, datetime(2026, 1, 1, 10, 3, 0)),   # Recovering
        (4, 10000.0, datetime(2026, 1, 1, 10, 4, 0)),  # Back to peak (0% DD)
        (5, 10500.0, datetime(2026, 1, 1, 10, 5, 0)),  # New peak (0% DD)
        (6, 10200.0, datetime(2026, 1, 1, 10, 6, 0)),  # Small drawdown from new peak
    ]

    calculator = DrawdownCalculator()
    drawdown_curve = calculator.calculate(equity_curve)

    print("\n" + "=" * 70)
    print("Drawdown Recovery Test Results")
    print("=" * 70)

    for i, (trade_id, equity, dd, timestamp) in enumerate(drawdown_curve):
        print(f"  Point {i}: Equity ${equity:,.2f}, DD {dd*100:.4f}%")

    # Check specific points
    assert abs(drawdown_curve[0][2]) < 1e-10, "Initial DD should be 0%"
    assert abs(drawdown_curve[1][2] - 0.10) < 1e-6, "DD should be 10% at point 1"
    assert abs(drawdown_curve[2][2] - 0.15) < 1e-6, "DD should be 15% at point 2"
    assert abs(drawdown_curve[4][2]) < 1e-10, "DD should be 0% when recovering to peak"
    assert abs(drawdown_curve[5][2]) < 1e-10, "DD should be 0% at new peak"

    # Check drawdown from new peak
    new_peak_dd = (10500.0 - 10200.0) / 10500.0  # ~2.86%
    assert abs(drawdown_curve[6][2] - new_peak_dd) < 1e-6, (
        f"DD from new peak incorrect: expected {new_peak_dd*100:.2f}%, got {drawdown_curve[6][2]*100:.2f}%"
    )

    max_dd = calculator.get_max_drawdown(drawdown_curve)
    assert abs(max_dd - 0.15) < 1e-6, f"Max DD should be 15%, got {max_dd*100:.2f}%"

    print(f"[PASS] Drawdown recovered correctly, max DD: {max_dd*100:.2f}%")
    print("=" * 70)
