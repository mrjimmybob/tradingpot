"""
Backtest determinism test - ensures reproducible results.

Verifies that running a backtest twice with identical inputs produces
identical outputs in every dimension:
- Final balance
- Number of trades
- Equity curve
- Trade sequence (side, amount, price)

This test ensures there is NO hidden randomness or state leakage.
"""

import pytest
import random
import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from types import SimpleNamespace

from app.models.bot import Bot, BotStatus
from app.models.order import Order, OrderStatus, OrderType
from app.models.trade import Trade, TradeSide
from app.models.position import Position, PositionSide
from app.services.trading_engine import TradingEngine, TradeSignal


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

FIXED_SEED = 42
INITIAL_BUDGET = 10000.0
TRADE_SIZE = 0.01  # BTC per trade
NUM_CANDLES = 100  # Number of historical data points


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Candle:
    """Historical price candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeRecord:
    """Record of a single trade for comparison."""
    iteration: int
    side: str
    amount: float
    price: float
    fee: float
    balance_after: float


@dataclass
class BacktestResult:
    """Complete backtest results for comparison."""
    final_balance: float
    trades: List[TradeRecord]
    equity_curve: List[float]
    num_trades: int


# ============================================================================
# DETERMINISTIC HISTORICAL DATA GENERATOR
# ============================================================================


def generate_fixed_historical_data(
    num_candles: int = NUM_CANDLES,
    base_price: float = 50000.0,
    start_time: datetime = None,
) -> List[Candle]:
    """
    Generate deterministic historical price data.

    Creates a predictable price sequence with no randomness.
    Prices oscillate in a sine wave pattern for variety.

    Args:
        num_candles: Number of candles to generate
        base_price: Base price level
        start_time: Starting timestamp

    Returns:
        List of fixed historical candles
    """
    if start_time is None:
        start_time = datetime(2025, 1, 1, 0, 0, 0)

    candles = []

    for i in range(num_candles):
        # Deterministic price oscillation using sine wave
        # This creates price movement without randomness
        t = i / 20.0  # Period of oscillation
        price_offset = math.sin(t) * 0.02  # ±2% oscillation
        base = base_price * (1.0 + price_offset)

        # Small deterministic spread for OHLC
        spread = base * 0.001  # 0.1% spread

        candle = Candle(
            timestamp=start_time + timedelta(hours=i),
            open=base,
            high=base + spread,
            low=base - spread,
            close=base + (spread * 0.5),  # Close slightly above base
            volume=100.0,  # Fixed volume
        )

        candles.append(candle)

    return candles


# ============================================================================
# DETERMINISTIC BACKTEST EXCHANGE
# ============================================================================


class DeterministicBacktestExchange:
    """
    Deterministic exchange for backtesting.

    - No slippage
    - No partial fills
    - No random failures
    - Executes at exact price provided
    - Fixed fee structure
    """

    def __init__(self, historical_data: List[Candle]):
        """
        Initialize backtest exchange.

        Args:
            historical_data: Fixed historical candles
        """
        self.historical_data = historical_data
        self.current_candle_index = 0
        self.order_counter = 0
        self.executed_orders: List[SimpleNamespace] = []
        self.is_connected = True

    async def connect(self):
        """Mock connect - always succeeds."""
        pass

    def get_current_price(self) -> float:
        """
        Get current price from historical data.

        Returns:
            Close price of current candle
        """
        if self.current_candle_index >= len(self.historical_data):
            # If we've exhausted historical data, return last price
            return self.historical_data[-1].close

        return self.historical_data[self.current_candle_index].close

    def advance_time(self) -> bool:
        """
        Move to next candle in historical data.

        Returns:
            True if advanced, False if at end
        """
        if self.current_candle_index < len(self.historical_data) - 1:
            self.current_candle_index += 1
            return True
        return False

    async def place_market_order(
        self,
        trading_pair: str,
        side,
        amount: float,
    ) -> SimpleNamespace:
        """
        Execute deterministic market order at current historical price.

        Args:
            trading_pair: Trading pair
            side: Order side
            amount: Amount to trade

        Returns:
            Order result
        """
        self.order_counter += 1
        order_id = f"backtest_order_{self.order_counter}"

        # Execute at current historical price (no slippage)
        execution_price = self.get_current_price()

        # Always 100% fill (no partial fills in backtest)
        filled_amount = amount

        # Fixed fee: 0.1%
        fee_rate = 0.001
        fee = execution_price * filled_amount * fee_rate

        # Get side string
        side_str = side.value if hasattr(side, 'value') else str(side).lower()

        order = SimpleNamespace(
            id=order_id,
            symbol=trading_pair,
            side=side_str,
            type="market",
            amount=amount,
            filled=filled_amount,
            remaining=0.0,
            price=execution_price,
            cost=execution_price * filled_amount,
            fee=fee,
            status="closed",
            timestamp=self.historical_data[self.current_candle_index].timestamp.isoformat(),
        )

        self.executed_orders.append(order)
        return order


# ============================================================================
# DETERMINISTIC BACKTEST PRICE FEED
# ============================================================================


class DeterministicBacktestPriceFeed:
    """
    Deterministic price feed that returns historical data.

    Synchronized with backtest exchange to return same prices.
    """

    def __init__(self, exchange: DeterministicBacktestExchange):
        """
        Initialize price feed.

        Args:
            exchange: Backtest exchange to sync with
        """
        self.exchange = exchange
        self.call_count = 0

    async def get_price(self, symbol: str) -> float:
        """
        Get current price from historical data.

        Args:
            symbol: Trading symbol

        Returns:
            Current historical price
        """
        self.call_count += 1
        return self.exchange.get_current_price()


# ============================================================================
# BACKTEST RUNNER
# ============================================================================


async def run_deterministic_backtest(
    bot: Bot,
    historical_data: List[Candle],
    trading_engine: TradingEngine,
    mock_session: AsyncMock,
    mock_services: Dict,
    trade_size: float,
) -> BacktestResult:
    """
    Run a complete deterministic backtest.

    Args:
        bot: Bot configuration
        historical_data: Fixed historical candles
        trading_engine: Trading engine instance
        mock_session: Mock database session
        mock_services: Mock service dependencies
        trade_size: Fixed trade size per order

    Returns:
        Complete backtest results
    """
    # Create deterministic exchange and price feed
    exchange = DeterministicBacktestExchange(historical_data)
    price_feed = DeterministicBacktestPriceFeed(exchange)

    # Track results
    trades: List[TradeRecord] = []
    equity_curve: List[float] = []

    # Record initial balance
    equity_curve.append(bot.current_balance)

    # Helper to create mock trades
    def create_mock_trade(trade_id: int, side: TradeSide, base_amount: float, price: float):
        """Create mock trade object."""
        trade = Mock(spec=Trade)
        trade.id = trade_id
        trade.bot_id = bot.id
        trade.side = side
        trade.base_amount = base_amount
        trade.quote_amount = base_amount * price
        trade.price = price
        trade.fee_amount = base_amount * price * 0.001
        trade.get_cost_basis_per_unit = Mock(return_value=price)
        return trade

    # Patch services for deterministic execution
    with patch(
        "app.services.trading_engine.PortfolioRiskService",
        return_value=mock_services["portfolio_risk"],
    ), patch(
        "app.services.trading_engine.StrategyCapacityService",
        return_value=mock_services["strategy_capacity"],
    ), patch(
        "app.services.trading_engine.TradeRecorderService",
        return_value=mock_services["trade_recorder"],
    ), patch(
        "app.services.trading_engine.FIFOTaxEngine",
        return_value=mock_services["tax_engine"],
    ), patch(
        "app.services.trading_engine.LedgerInvariantService",
        return_value=mock_services["invariant_validator"],
    ), patch(
        "app.services.trading_engine.VirtualWalletService",
        return_value=mock_services["wallet"],
    ), patch(
        "app.services.trading_engine.CSVExportService"
    ):

        # Run backtest through historical data
        for i in range(len(historical_data)):
            # Simple strategy: alternate buy/sell
            action = "buy" if i % 2 == 0 else "sell"
            side = TradeSide.BUY if action == "buy" else TradeSide.SELL

            # Get current price
            current_price = await price_feed.get_price(bot.trading_pair)

            # Create trade signal
            signal = TradeSignal(
                action=action,
                amount=trade_size * current_price,  # Amount in USDT
                order_type="market",
            )

            # Setup mock trade recorder
            mock_trade = create_mock_trade(i + 1, side, trade_size, current_price)
            mock_services["trade_recorder"].record_trade = AsyncMock(
                return_value=mock_trade
            )

            # Execute trade
            order = await trading_engine._execute_trade(
                bot,
                exchange,
                signal,
                current_price,
                mock_session,
            )

            # Record trade if successful
            if order is not None:
                trade_record = TradeRecord(
                    iteration=i,
                    side=action,
                    amount=trade_size,
                    price=current_price,
                    fee=trade_size * current_price * 0.001,
                    balance_after=bot.current_balance,
                )
                trades.append(trade_record)

            # Record equity curve
            equity_curve.append(bot.current_balance)

            # Advance to next candle
            exchange.advance_time()

    # Return complete results
    return BacktestResult(
        final_balance=bot.current_balance,
        trades=trades,
        equity_curve=equity_curve,
        num_trades=len(trades),
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def compare_equity_curves(
    curve1: List[float],
    curve2: List[float],
    tolerance: float = 0.01
) -> Tuple[bool, str]:
    """
    Compare two equity curves for equality.

    Args:
        curve1: First equity curve
        curve2: Second equity curve
        tolerance: Allowed floating-point difference

    Returns:
        (is_equal, reason_if_not_equal)
    """
    if len(curve1) != len(curve2):
        return False, f"Length mismatch: {len(curve1)} vs {len(curve2)}"

    for i, (v1, v2) in enumerate(zip(curve1, curve2)):
        # Check both are finite
        if not math.isfinite(v1) or not math.isfinite(v2):
            return False, f"Non-finite value at index {i}: {v1} vs {v2}"

        # Check equality within tolerance
        if abs(v1 - v2) > tolerance:
            return False, f"Mismatch at index {i}: {v1} vs {v2} (diff: {abs(v1 - v2)})"

    return True, ""


def compare_trade_sequences(
    trades1: List[TradeRecord],
    trades2: List[TradeRecord],
    tolerance: float = 0.01
) -> Tuple[bool, str]:
    """
    Compare two trade sequences for equality.

    Args:
        trades1: First trade sequence
        trades2: Second trade sequence
        tolerance: Allowed floating-point difference

    Returns:
        (is_equal, reason_if_not_equal)
    """
    if len(trades1) != len(trades2):
        return False, f"Trade count mismatch: {len(trades1)} vs {len(trades2)}"

    for i, (t1, t2) in enumerate(zip(trades1, trades2)):
        # Compare all fields
        if t1.iteration != t2.iteration:
            return False, f"Trade {i} iteration mismatch: {t1.iteration} vs {t2.iteration}"

        if t1.side != t2.side:
            return False, f"Trade {i} side mismatch: {t1.side} vs {t2.side}"

        if abs(t1.amount - t2.amount) > tolerance:
            return False, f"Trade {i} amount mismatch: {t1.amount} vs {t2.amount}"

        if abs(t1.price - t2.price) > tolerance:
            return False, f"Trade {i} price mismatch: {t1.price} vs {t2.price}"

        if abs(t1.fee - t2.fee) > tolerance:
            return False, f"Trade {i} fee mismatch: {t1.fee} vs {t2.fee}"

        if abs(t1.balance_after - t2.balance_after) > tolerance:
            return False, f"Trade {i} balance mismatch: {t1.balance_after} vs {t2.balance_after}"

    return True, ""


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def fixed_historical_data():
    """Generate fixed historical data for backtesting."""
    return generate_fixed_historical_data(
        num_candles=NUM_CANDLES,
        base_price=50000.0,
        start_time=datetime(2025, 1, 1, 0, 0, 0),
    )


@pytest.fixture
def backtest_bot():
    """
    Create bot for backtesting with deterministic parameters.

    All risk limits disabled to ensure pure determinism.
    """
    bot = Bot(
        id=1,
        name="Backtest Determinism Test Bot",
        trading_pair="BTC/USDT",
        strategy="simple_alternate",
        budget=INITIAL_BUDGET,
        current_balance=INITIAL_BUDGET,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,

        # Disable all risk limits for determinism
        stop_loss_percent=None,
        stop_loss_absolute=None,
        drawdown_limit_percent=None,
        drawdown_limit_absolute=None,
        daily_loss_limit=None,
        weekly_loss_limit=None,
        max_strategy_rotations=0,
        running_time_hours=None,

        created_at=datetime(2025, 1, 1, 0, 0, 0),
        started_at=datetime(2025, 1, 1, 0, 0, 0),
    )

    return bot


@pytest.fixture
def mock_session():
    """Create mock database session."""
    session = AsyncMock()
    session.add = Mock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.refresh = AsyncMock()

    # Track added objects
    session.added_objects = []

    def track_add(obj):
        session.added_objects.append(obj)

    session.add.side_effect = track_add

    # Mock query results
    mock_result = Mock()
    mock_result.scalar_one_or_none = Mock(return_value=None)
    mock_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
    session.execute.return_value = mock_result

    return session


@pytest.fixture
def trading_engine():
    """Create trading engine instance."""
    return TradingEngine()


@pytest.fixture
def mock_services():
    """
    Create deterministic mock services.

    All services configured to ALWAYS allow trades.
    NO randomness, NO chaos, NO fault injection.
    """
    services = {}

    # Portfolio Risk Service - always allows
    risk_check = Mock()
    risk_check.ok = True
    risk_check.action = "allow"
    risk_check.reason = None
    portfolio_risk_instance = AsyncMock()
    portfolio_risk_instance.check_portfolio_risk = AsyncMock(return_value=risk_check)
    services["portfolio_risk"] = portfolio_risk_instance

    # Strategy Capacity Service - always allows
    capacity_check = Mock()
    capacity_check.ok = True
    capacity_check.adjusted_amount = None
    capacity_check.reason = None
    strategy_capacity_instance = AsyncMock()
    strategy_capacity_instance.check_capacity_for_trade = AsyncMock(
        return_value=capacity_check
    )
    services["strategy_capacity"] = strategy_capacity_instance

    # Trade Recorder Service
    trade_recorder_instance = AsyncMock()
    services["trade_recorder"] = trade_recorder_instance

    # Tax Engine
    tax_engine_instance = AsyncMock()
    tax_engine_instance.process_buy = AsyncMock()
    tax_engine_instance.process_sell = AsyncMock(return_value=[])
    services["tax_engine"] = tax_engine_instance

    # Invariant Validator - always passes
    invariant_validator_instance = AsyncMock()
    invariant_validator_instance.validate_trade = AsyncMock()
    services["invariant_validator"] = invariant_validator_instance

    # Virtual Wallet Service - always allows
    wallet_instance = AsyncMock()
    wallet_instance.record_trade_result = AsyncMock(return_value=(True, "Success"))
    wallet_instance.validate_trade = AsyncMock(
        return_value=Mock(is_valid=True, reason="OK")
    )
    services["wallet"] = wallet_instance

    return services


# ============================================================================
# MAIN TEST
# ============================================================================


@pytest.mark.asyncio
async def test_backtest_determinism_same_data_same_result(
    fixed_historical_data,
    backtest_bot,
    mock_session,
    trading_engine,
    mock_services,
):
    """
    Verify that running backtest twice with identical inputs produces identical outputs.

    Tests complete reproducibility across:
    - Final balance
    - Number of trades
    - Equity curve values
    - Trade sequence (side, amount, price, fee)

    Setup:
    - Fixed random seed (42)
    - Fixed historical price data (100 candles)
    - Deterministic exchange (no slippage, no partial fills)
    - Deterministic price feed
    - No chaos/fault injection
    - No risk randomization
    - No timing-based behavior

    Assertions:
    - final_balance_run1 == final_balance_run2
    - num_trades_run1 == num_trades_run2
    - equity_curve_run1 == equity_curve_run2
    - trade_sequence_run1 == trade_sequence_run2
    - At least one trade occurred
    - No NaN or infinite balances

    Failure indicates:
    ⚠️ Backtest is NOT reproducible
    ⚠️ Hidden randomness exists
    ⚠️ State leakage between runs
    """
    # Set fixed random seed for reproducibility
    random.seed(FIXED_SEED)

    print(f"\n{'='*70}")
    print(f"Backtest Determinism Test")
    print(f"{'='*70}")
    print(f"Historical data: {len(fixed_historical_data)} candles")
    print(f"Initial balance: ${INITIAL_BUDGET:.2f}")
    print(f"Trade size: {TRADE_SIZE} BTC")
    print(f"Fixed seed: {FIXED_SEED}")
    print(f"{'='*70}\n")

    # ========================================================================
    # RUN #1: First backtest execution
    # ========================================================================

    print("Running backtest #1...")

    # Create fresh bot for run 1
    bot_run1 = Bot(
        id=1,
        name="Backtest Bot Run 1",
        trading_pair="BTC/USDT",
        strategy="simple_alternate",
        budget=INITIAL_BUDGET,
        current_balance=INITIAL_BUDGET,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        stop_loss_percent=None,
        stop_loss_absolute=None,
        drawdown_limit_percent=None,
        drawdown_limit_absolute=None,
        daily_loss_limit=None,
        weekly_loss_limit=None,
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        started_at=datetime(2025, 1, 1, 0, 0, 0),
    )

    result_run1 = await run_deterministic_backtest(
        bot=bot_run1,
        historical_data=fixed_historical_data,
        trading_engine=trading_engine,
        mock_session=mock_session,
        mock_services=mock_services,
        trade_size=TRADE_SIZE,
    )

    print(f"✓ Run 1 complete")
    print(f"  Final balance: ${result_run1.final_balance:.2f}")
    print(f"  Trades: {result_run1.num_trades}")
    print(f"  Equity curve length: {len(result_run1.equity_curve)}")

    # ========================================================================
    # RESET STATE COMPLETELY
    # ========================================================================

    print(f"\nResetting state...")

    # Reset random seed to same value
    random.seed(FIXED_SEED)

    # ========================================================================
    # RUN #2: Second backtest execution with identical setup
    # ========================================================================

    print("Running backtest #2...")

    # Create fresh bot for run 2 (identical configuration)
    bot_run2 = Bot(
        id=1,
        name="Backtest Bot Run 2",
        trading_pair="BTC/USDT",
        strategy="simple_alternate",
        budget=INITIAL_BUDGET,
        current_balance=INITIAL_BUDGET,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        stop_loss_percent=None,
        stop_loss_absolute=None,
        drawdown_limit_percent=None,
        drawdown_limit_absolute=None,
        daily_loss_limit=None,
        weekly_loss_limit=None,
        created_at=datetime(2025, 1, 1, 0, 0, 0),
        started_at=datetime(2025, 1, 1, 0, 0, 0),
    )

    result_run2 = await run_deterministic_backtest(
        bot=bot_run2,
        historical_data=fixed_historical_data,
        trading_engine=trading_engine,
        mock_session=mock_session,
        mock_services=mock_services,
        trade_size=TRADE_SIZE,
    )

    print(f"✓ Run 2 complete")
    print(f"  Final balance: ${result_run2.final_balance:.2f}")
    print(f"  Trades: {result_run2.num_trades}")
    print(f"  Equity curve length: {len(result_run2.equity_curve)}")

    # ========================================================================
    # ASSERTIONS: Compare results for exact equality
    # ========================================================================

    print(f"\n{'='*70}")
    print("Comparing results...")
    print(f"{'='*70}\n")

    # Assertion 1: At least one trade occurred
    assert result_run1.num_trades > 0, (
        "No trades occurred in run 1 - backtest may be broken"
    )
    assert result_run2.num_trades > 0, (
        "No trades occurred in run 2 - backtest may be broken"
    )
    print(f"✓ Both runs executed trades")

    # Assertion 2: Number of trades is identical
    assert result_run1.num_trades == result_run2.num_trades, (
        f"❌ Number of trades differs:\n"
        f"  Run 1: {result_run1.num_trades} trades\n"
        f"  Run 2: {result_run2.num_trades} trades\n"
        f"⚠️ This indicates hidden randomness or state leakage"
    )
    print(f"✓ Same number of trades: {result_run1.num_trades}")

    # Assertion 3: Final balance is identical
    balance_diff = abs(result_run1.final_balance - result_run2.final_balance)
    assert balance_diff < 0.01, (
        f"❌ Final balance differs:\n"
        f"  Run 1: ${result_run1.final_balance:.2f}\n"
        f"  Run 2: ${result_run2.final_balance:.2f}\n"
        f"  Difference: ${balance_diff:.2f}\n"
        f"⚠️ Backtest is NOT reproducible"
    )
    print(f"✓ Identical final balance: ${result_run1.final_balance:.2f}")

    # Assertion 4: No NaN or infinite balances
    for i, balance in enumerate(result_run1.equity_curve):
        assert math.isfinite(balance), (
            f"❌ Non-finite balance in run 1 at index {i}: {balance}"
        )
        assert balance >= 0, (
            f"❌ Negative balance in run 1 at index {i}: {balance}"
        )

    for i, balance in enumerate(result_run2.equity_curve):
        assert math.isfinite(balance), (
            f"❌ Non-finite balance in run 2 at index {i}: {balance}"
        )
        assert balance >= 0, (
            f"❌ Negative balance in run 2 at index {i}: {balance}"
        )
    print(f"✓ All balances finite and non-negative")

    # Assertion 5: Equity curves are identical
    curves_equal, curves_reason = compare_equity_curves(
        result_run1.equity_curve,
        result_run2.equity_curve,
        tolerance=0.01,
    )
    assert curves_equal, (
        f"❌ Equity curves differ:\n"
        f"  {curves_reason}\n"
        f"⚠️ This indicates calculation drift or state leakage"
    )
    print(f"✓ Identical equity curves ({len(result_run1.equity_curve)} points)")

    # Assertion 6: Trade sequences are identical
    trades_equal, trades_reason = compare_trade_sequences(
        result_run1.trades,
        result_run2.trades,
        tolerance=0.01,
    )
    assert trades_equal, (
        f"❌ Trade sequences differ:\n"
        f"  {trades_reason}\n"
        f"⚠️ This indicates hidden randomness in trade execution"
    )
    print(f"✓ Identical trade sequences (side, amount, price, fee)")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print("✅ BACKTEST DETERMINISM VERIFIED")
    print(f"{'='*70}")
    print("All checks passed:")
    print(f"  ✓ Same number of trades: {result_run1.num_trades}")
    print(f"  ✓ Same final balance: ${result_run1.final_balance:.2f}")
    print(f"  ✓ Identical equity curves: {len(result_run1.equity_curve)} points")
    print(f"  ✓ Identical trade sequences")
    print(f"  ✓ No NaN or infinite values")
    print(f"  ✓ No state leakage detected")
    print(f"  ✓ No hidden randomness detected")
    print(f"\n🎯 Backtest is fully reproducible and deterministic")
    print(f"{'='*70}\n")
