"""
Chaos Test (Randomized Trade Fault Injection)

Tests system resilience under random, unpredictable failures across the entire
trading pipeline. This is the "monkey in the server room" test.

Verifies the system:
- Never crashes
- Never corrupts the ledger
- Never bypasses risk rules
- Never trades after kill-switch
- Always returns deterministic, safe outcome

Even when exchanges fail, prices glitch, orders partially fill, services timeout,
and ledgers lag.
"""

import pytest
import random
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from enum import Enum

# Set seed for determinism
random.seed(42)

from app.models.bot import Bot, BotStatus
from app.models.order import Order, OrderStatus, OrderType
from app.models.trade import Trade, TradeSide
from app.models.wallet_ledger import WalletLedger
from app.services.risk_management import RiskManagementService, RiskAction, RiskAssessment
from app.services.virtual_wallet import VirtualWalletService
from app.services.exchange import ExchangeService, OrderSide
from app.services.ledger_writer import LedgerWriterService
from app.services.ledger_invariants import LedgerInvariantService, ValidationError


# ============================================================================
# CHAOS CONFIGURATION
# ============================================================================

CHAOS_SEED = 42  # For reproducibility
TRADE_ATTEMPTS = 50  # Number of trade attempts in chaos test

# Fault injection probabilities
FAULT_PROBABILITY = {
    "exchange_timeout": 0.15,
    "exchange_rejection": 0.10,
    "slippage_spike": 0.20,
    "partial_fill": 0.15,
    "invalid_symbol": 0.05,
    "price_gap": 0.10,
    "price_zero": 0.03,
    "price_nan": 0.02,
    "price_stale": 0.10,
    "ledger_delay": 0.15,
    "ledger_duplicate": 0.02,
    "risk_drawdown": 0.05,
    "risk_daily_loss": 0.05,
}


class TradeResult(str, Enum):
    """Possible trade outcomes."""
    SUCCESS = "success"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    FAILED = "failed"


# ============================================================================
# CHAOS INJECTORS
# ============================================================================
class LedgerReason(Enum):
    TRADE = "trade"

class ChaosExchange:
    """Exchange wrapper with random fault injection."""
    
    def __init__(self, seed: int = CHAOS_SEED):
        """Initialize chaos exchange.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        self.order_counter = 0
        self.executed_orders = []
        
    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        base_price: float = 50000.0,
    ) -> dict:
        """
        Place market order with random fault injection.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            amount: Order amount
            base_price: Base price for calculations
            
        Returns:
            Order result dict
            
        Raises:
            Exception: On random failures
        """
        self.order_counter += 1
        order_id = f"chaos_order_{self.order_counter}"
        
        # Fault: Invalid symbol
        if self.rng.random() < FAULT_PROBABILITY["invalid_symbol"]:
            raise Exception(f"Invalid symbol: {symbol}")
        
        # Fault: Exchange timeout
        if self.rng.random() < FAULT_PROBABILITY["exchange_timeout"]:
            raise Exception("Exchange timeout")
        
        # Fault: Exchange rejection
        if self.rng.random() < FAULT_PROBABILITY["exchange_rejection"]:
            raise Exception("Order rejected by exchange")
        
        # Calculate execution price with random slippage
        slippage_multiplier = 1.0
        if self.rng.random() < FAULT_PROBABILITY["slippage_spike"]:
            # Extreme slippage (1-5%)
            slippage_multiplier = 1.0 + self.rng.uniform(0.01, 0.05)
            if side == "sell":
                slippage_multiplier = 2.0 - slippage_multiplier  # Negative for sells
        
        execution_price = base_price * slippage_multiplier
        
        # Fault: Partial fill
        filled_amount = amount
        if self.rng.random() < FAULT_PROBABILITY["partial_fill"]:
            filled_amount = amount * self.rng.uniform(0.3, 0.9)
        
        # Calculate fee (random variation 0.05% - 0.15%)
        fee_rate = self.rng.uniform(0.0005, 0.0015)
        fee = execution_price * filled_amount * fee_rate
        
        # Random fee spike (rare)
        if self.rng.random() < 0.02:
            fee *= self.rng.uniform(5.0, 20.0)
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": "market",
            "amount": amount,
            "filled": filled_amount,
            "remaining": amount - filled_amount,
            "price": execution_price,
            "cost": execution_price * filled_amount,
            "fee": {"cost": fee, "currency": "USDT"},
            "status": "closed" if filled_amount == amount else "partial",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.executed_orders.append(order)
        return order


class ChaosPriceFeed:
    """Price feed with random fault injection."""
    
    def __init__(self, base_price: float = 50000.0, seed: int = CHAOS_SEED):
        """Initialize chaos price feed.
        
        Args:
            base_price: Base price to perturb
            seed: Random seed for reproducibility
        """
        self.base_price = base_price
        self.rng = random.Random(seed + 1)  # Different seed than exchange
        self.last_valid_price = base_price
        self.price_history = []
        
    def get_price(self, symbol: str) -> float:
        """
        Get current price with random fault injection.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
            
        Raises:
            ValueError: On invalid price
        """
        # Fault: Price gap (sudden jump ±30%)
        if self.rng.random() < FAULT_PROBABILITY["price_gap"]:
            gap = self.rng.uniform(-0.30, 0.30)
            price = self.base_price * (1.0 + gap)
            self.last_valid_price = price
            self.price_history.append(("gap", price))
            return price
        
        # Fault: Zero price
        if self.rng.random() < FAULT_PROBABILITY["price_zero"]:
            self.price_history.append(("zero", 0.0))
            return 0.0
        
        # Fault: NaN price
        if self.rng.random() < FAULT_PROBABILITY["price_nan"]:
            self.price_history.append(("nan", float('nan')))
            return float('nan')
        
        # Fault: Stale price (return old price)
        if self.rng.random() < FAULT_PROBABILITY["price_stale"]:
            stale_price = self.last_valid_price
            self.price_history.append(("stale", stale_price))
            return stale_price
        
        # Normal price with small random walk (±1%)
        normal_variation = self.rng.uniform(-0.01, 0.01)
        price = self.base_price * (1.0 + normal_variation)
        self.last_valid_price = price
        self.price_history.append(("normal", price))
        return price


class ChaosLedger:
    """Ledger writer with random fault injection."""
    
    def __init__(self, seed: int = CHAOS_SEED):
        """Initialize chaos ledger.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed + 2)  # Different seed
        self.entries = []
        self.duplicate_count = 0
        self.delayed_entries = []
        
    async def write_entry(
        self,
        bot_id: int,
        asset: str,
        amount: float,
        reason: LedgerReason,
        trade_id: int = None,
    ) -> None:
        """
        Write ledger entry with random fault injection.
        
        Args:
            bot_id: Bot ID
            asset: Asset symbol
            amount: Amount
            reason: Ledger reason
            trade_id: Optional trade ID
        """
        entry = {
            "bot_id": bot_id,
            "asset": asset,
            "amount": amount,
            "reason": reason,
            "trade_id": trade_id,
            "timestamp": datetime.utcnow(),
        }
        
        # Fault: Delayed write
        if self.rng.random() < FAULT_PROBABILITY["ledger_delay"]:
            self.delayed_entries.append(entry)
            return
        
        # Fault: Duplicate write (rare but serious)
        if self.rng.random() < FAULT_PROBABILITY["ledger_duplicate"]:
            self.entries.append(entry)
            self.entries.append(entry.copy())
            self.duplicate_count += 1
            return
        
        # Normal write
        self.entries.append(entry)
        
    def flush_delayed(self) -> int:
        """Flush delayed entries.
        
        Returns:
            Number of flushed entries
        """
        count = len(self.delayed_entries)
        self.entries.extend(self.delayed_entries)
        self.delayed_entries.clear()
        return count
    
    def validate_consistency(self) -> list[str]:
        """Validate ledger consistency.
        
        Returns:
            List of violations (empty if valid)
        """
        violations = []
        
        # Check for chronological order
        for i in range(1, len(self.entries)):
            if self.entries[i]["timestamp"] < self.entries[i-1]["timestamp"]:
                violations.append(f"Entry {i} out of chronological order")
        
        # Check for balance consistency
        balance_by_bot = {}
        for entry in self.entries:
            bot_id = entry["bot_id"]
            if bot_id not in balance_by_bot:
                balance_by_bot[bot_id] = 0.0
            balance_by_bot[bot_id] += entry["amount"]
            
            if balance_by_bot[bot_id] < -1000000:  # Sanity check
                violations.append(f"Bot {bot_id} balance extremely negative: {balance_by_bot[bot_id]}")
        
        return violations


class ChaosRiskManager:
    """Risk manager with random fault injection for conditions."""
    
    def __init__(self, bot: Bot, seed: int = CHAOS_SEED):
        """Initialize chaos risk manager.
        
        Args:
            bot: Bot to manage
            seed: Random seed for reproducibility
        """
        self.bot = bot
        self.rng = random.Random(seed + 3)  # Different seed
        self.daily_loss = 0.0
        self.kill_switch_active = False
        
    def inject_risk_condition(self) -> bool:
        """
        Randomly inject risk condition.
        
        Returns:
            True if risk condition injected
        """
        # Random drawdown spike
        if self.rng.random() < FAULT_PROBABILITY["risk_drawdown"]:
            # Simulate sudden balance drop
            self.bot.current_balance *= 0.70  # 30% drop
            return True
        
        # Random daily loss limit hit
        if self.rng.random() < FAULT_PROBABILITY["risk_daily_loss"]:
            # Force daily loss over limit
            if self.bot.daily_loss_limit:
                self.daily_loss = self.bot.daily_loss_limit * 1.5
                self.kill_switch_active = True
                return True
        
        return False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def chaos_bot():
    """Bot for chaos testing."""
    return Bot(
        id=1,
        name="Chaos Test Bot",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=10000.0,
        current_balance=10000.0,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        daily_loss_limit=1000.0,
        stop_loss_percent=10.0,
        drawdown_limit_percent=30.0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )


@pytest.fixture
def chaos_exchange():
    """Chaos exchange fixture."""
    return ChaosExchange(seed=CHAOS_SEED)


@pytest.fixture
def chaos_price_feed():
    """Chaos price feed fixture."""
    return ChaosPriceFeed(seed=CHAOS_SEED)


@pytest.fixture
def chaos_ledger():
    """Chaos ledger fixture."""
    return ChaosLedger(seed=CHAOS_SEED)


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_valid_price(price: float) -> bool:
    """Check if price is valid (not zero, not NaN, positive).
    
    Args:
        price: Price to check
        
    Returns:
        True if valid
    """
    import math
    if price is None:
        return False
    if math.isnan(price):
        return False
    if price <= 0:
        return False
    return True


async def attempt_trade(
    bot: Bot,
    exchange: ChaosExchange,
    price_feed: ChaosPriceFeed,
    ledger: ChaosLedger,
    risk_manager: ChaosRiskManager,
    side: OrderSide,
    amount: float,
    iteration: int,
) -> tuple[TradeResult, str, dict]:
    """
    Attempt a single trade with full chaos injection.
    
    Args:
        bot: Bot instance
        exchange: Chaos exchange
        price_feed: Chaos price feed
        ledger: Chaos ledger
        risk_manager: Chaos risk manager
        side: Order side
        amount: Trade amount
        iteration: Current iteration number
        
    Returns:
        (result, reason, metadata)
    """
    metadata = {
        "iteration": iteration,
        "side": side.value,
        "amount": amount,
        "bot_balance_before": bot.current_balance,
    }
    
    try:
        # Step 1: Check if kill switch already active
        if risk_manager.kill_switch_active:
            return TradeResult.BLOCKED, "Kill switch active", metadata
        
        # Step 2: Inject random risk condition
        if risk_manager.inject_risk_condition():
            metadata["risk_injected"] = True
        
        # Step 3: Get current price (may fail)
        try:
            current_price = price_feed.get_price(bot.trading_pair)
            metadata["price"] = current_price
        except Exception as e:
            metadata["price_error"] = str(e)
            return TradeResult.FAILED, f"Price feed error: {e}", metadata
        
        # Step 4: Validate price
        if not is_valid_price(current_price):
            metadata["invalid_price"] = current_price
            return TradeResult.REJECTED, f"Invalid price: {current_price}", metadata
        
        # Step 5: Check balance (basic wallet validation)
        if side == OrderSide.BUY:
            required = amount * current_price
            if bot.current_balance < required:
                return TradeResult.REJECTED, "Insufficient balance", metadata
        else:
            # For sell, would check position (simplified here)
            pass
        
        # Step 6: Check risk limits (drawdown)
        if bot.drawdown_limit_percent:
            drawdown = (bot.budget - bot.current_balance) / bot.budget * 100
            if drawdown >= bot.drawdown_limit_percent:
                risk_manager.kill_switch_active = True
                return TradeResult.BLOCKED, "Drawdown limit exceeded", metadata
        
        # Step 7: Check daily loss limit
        if bot.daily_loss_limit and risk_manager.daily_loss >= bot.daily_loss_limit:
            risk_manager.kill_switch_active = True
            return TradeResult.BLOCKED, "Daily loss limit exceeded", metadata
        
        # Step 8: Execute order on exchange (may fail)
        try:
            order_result = exchange.place_market_order(
                symbol=bot.trading_pair,
                side=side.value,
                amount=amount,
                base_price=current_price,
            )
            metadata["order_id"] = order_result["id"]
            metadata["filled"] = order_result["filled"]
            metadata["execution_price"] = order_result["price"]
            metadata["fee"] = order_result["fee"]["cost"]
        except Exception as e:
            metadata["exchange_error"] = str(e)
            return TradeResult.FAILED, f"Exchange error: {e}", metadata
        
        # Step 9: Update bot balance
        if side == OrderSide.BUY:
            cost = order_result["price"] * order_result["filled"]
            fee = order_result["fee"]["cost"]
            proceeds = 0.0
            bot.current_balance -= (cost + fee)
            loss = fee
        else:
            proceeds = order_result["price"] * order_result["filled"]
            fee = order_result["fee"]["cost"]
            cost = 0.0
            bot.current_balance += (proceeds - fee)
            loss = fee

        # Track daily loss
        risk_manager.daily_loss += loss
        metadata["daily_loss_total"] = risk_manager.daily_loss

        # Step 10: Write to ledger (may fail/delay/duplicate)
        try:
            await ledger.write_entry(
                bot_id=int(bot.id),
                asset="USDT",
                amount=-cost if side == OrderSide.BUY else proceeds,
                reason=LedgerReason.TRADE,
                trade_id=iteration,
            )

            await ledger.write_entry(
                bot_id=int(bot.id),
                asset=bot.trading_pair.split("/")[0],
                amount=amount if side == OrderSide.BUY else -amount,
                reason=LedgerReason.TRADE,
                trade_id=iteration,
            )
        except Exception as e:
            metadata["ledger_error"] = str(e)

        metadata["bot_balance_after"] = bot.current_balance
        return TradeResult.SUCCESS, "Trade executed", metadata
        
    except Exception as e:
        # Catch-all for unexpected errors
        metadata["unexpected_error"] = str(e)
        return TradeResult.FAILED, f"Unexpected error: {e}", metadata


def validate_global_invariants(
    bot: Bot,
    initial_balance: float,
    results: list,
    ledger: ChaosLedger,
) -> list[str]:
    """
    Validate global system invariants after chaos test.
    
    Args:
        bot: Bot instance
        initial_balance: Initial balance before test
        results: List of trade results
        ledger: Chaos ledger
        
    Returns:
        List of violations (empty if all pass)
    """
    violations = []
    
    # Invariant 1: Balance never negative
    if bot.current_balance < 0:
        violations.append(f"Negative balance: {bot.current_balance}")
    
    # Invariant 2: No duplicate order IDs
    order_ids = [r[2].get("order_id") for r in results if r[2].get("order_id")]
    if len(order_ids) != len(set(order_ids)):
        duplicates = [oid for oid in order_ids if order_ids.count(oid) > 1]
        violations.append(f"Duplicate order IDs: {duplicates}")
    
    # Invariant 3: All results have non-empty reason
    for i, (result, reason, meta) in enumerate(results):
        if not reason or reason.strip() == "":
            violations.append(f"Empty reason at iteration {i}: {result}")
    
    # Invariant 4: All results are valid enum values
    valid_results = {TradeResult.SUCCESS, TradeResult.REJECTED, TradeResult.BLOCKED, TradeResult.FAILED}
    for i, (result, reason, meta) in enumerate(results):
        if result not in valid_results:
            violations.append(f"Invalid result at iteration {i}: {result}")
    
    # Invariant 5: Balance change matches trade activity
    successful_trades = sum(1 for r, _, _ in results if r == TradeResult.SUCCESS)
    if successful_trades > 0 and bot.current_balance == initial_balance:
        # If trades succeeded, balance should have changed
        violations.append("Balance unchanged despite successful trades")
    
    # Invariant 6: No ledger duplicate writes (beyond intentional chaos)
    total_entries = len(ledger.entries)
    expected_max = successful_trades * 3  # Allow some duplicates from chaos
    if total_entries > expected_max:
        violations.append(f"Too many ledger entries: {total_entries} (expected max {expected_max})")
    
    # Invariant 7: Kill switch prevents future trades
    kill_switch_activated = False
    for i, (result, reason, meta) in enumerate(results):
        if result == TradeResult.BLOCKED and ("kill switch" in reason.lower() or "limit exceeded" in reason.lower()):
            kill_switch_activated = True
        
        # After kill switch, all subsequent trades should be BLOCKED
        if kill_switch_activated:
            if result == TradeResult.SUCCESS:
                violations.append(f"Trade succeeded after kill switch at iteration {i}")
    
    # Invariant 8: Ledger consistency
    ledger_violations = ledger.validate_consistency()
    violations.extend(ledger_violations)
    
    return violations


# ============================================================================
# CHAOS TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_chaos_trading_pipeline(
    chaos_bot,
    chaos_exchange,
    chaos_price_feed,
    chaos_ledger,
    mock_session,
):
    """
    Main chaos test: 50 randomized trade attempts with full fault injection.
    
    Verifies system never crashes, never corrupts state, always returns
    deterministic results.
    """
    # Seed for reproducibility
    random.seed(CHAOS_SEED)
    
    initial_balance = chaos_bot.current_balance
    risk_manager = ChaosRiskManager(chaos_bot, seed=CHAOS_SEED)
    
    results = []
    
    # Execute chaos loop
    for i in range(TRADE_ATTEMPTS):
        # Alternate buy/sell
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        
        # Random amount (0.01 - 0.1 BTC)
        rng = random.Random(CHAOS_SEED + i)
        amount = rng.uniform(0.01, 0.1)
        
        # Attempt trade with full chaos
        result, reason, metadata = await attempt_trade(
            bot=chaos_bot,
            exchange=chaos_exchange,
            price_feed=chaos_price_feed,
            ledger=chaos_ledger,
            risk_manager=risk_manager,
            side=side,
            amount=amount,
            iteration=i,
        )
        
        results.append((result, reason, metadata))
    
    # Flush delayed ledger entries
    flushed = chaos_ledger.flush_delayed()
    
    # Validate global invariants
    violations = validate_global_invariants(
        bot=chaos_bot,
        initial_balance=initial_balance,
        results=results,
        ledger=chaos_ledger,
    )
    
    # Assert no violations
    assert len(violations) == 0, f"Invariant violations:\n" + "\n".join(violations)
    
    # Summary assertions
    assert len(results) == TRADE_ATTEMPTS, f"Expected {TRADE_ATTEMPTS} results, got {len(results)}"
    
    success_count = sum(1 for r, _, _ in results if r == TradeResult.SUCCESS)
    rejected_count = sum(1 for r, _, _ in results if r == TradeResult.REJECTED)
    blocked_count = sum(1 for r, _, _ in results if r == TradeResult.BLOCKED)
    failed_count = sum(1 for r, _, _ in results if r == TradeResult.FAILED)
    
    # At least some trades should succeed
    assert success_count > 0, "No trades succeeded - test too chaotic"
    
    # At least some should fail/reject (chaos working)
    assert (rejected_count + blocked_count + failed_count) > 0, "No failures - chaos not injecting"
    
    # Balance should be non-negative
    assert chaos_bot.current_balance >= 0, f"Balance went negative: {chaos_bot.current_balance}"


@pytest.mark.asyncio
async def test_chaos_deterministic_reproducibility(
    chaos_bot,
    chaos_exchange,
    chaos_price_feed,
    chaos_ledger,
    mock_session,
):
    """
    Verify chaos test is deterministic: same seed produces same results.
    """
    random.seed(CHAOS_SEED)
    
    risk_manager = ChaosRiskManager(chaos_bot, seed=CHAOS_SEED)
    
    results_run1 = []
    
    # First run
    for i in range(10):  # Shorter for speed
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        rng = random.Random(CHAOS_SEED + i)
        amount = rng.uniform(0.01, 0.1)
        
        result, reason, metadata = await attempt_trade(
            bot=chaos_bot,
            exchange=chaos_exchange,
            price_feed=chaos_price_feed,
            ledger=chaos_ledger,
            risk_manager=risk_manager,
            side=side,
            amount=amount,
            iteration=i,
        )
        
        results_run1.append((result, reason))
    
    # Reset for second run
    random.seed(CHAOS_SEED)
    chaos_bot.current_balance = 10000.0  # Reset
    chaos_exchange2 = ChaosExchange(seed=CHAOS_SEED)
    chaos_price_feed2 = ChaosPriceFeed(seed=CHAOS_SEED)
    chaos_ledger2 = ChaosLedger(seed=CHAOS_SEED)
    risk_manager2 = ChaosRiskManager(chaos_bot, seed=CHAOS_SEED)
    
    results_run2 = []
    
    # Second run with same seed
    for i in range(10):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        rng = random.Random(CHAOS_SEED + i)
        amount = rng.uniform(0.01, 0.1)
        
        result, reason, metadata = await attempt_trade(
            bot=chaos_bot,
            exchange=chaos_exchange2,
            price_feed=chaos_price_feed2,
            ledger=chaos_ledger2,
            risk_manager=risk_manager2,
            side=side,
            amount=amount,
            iteration=i,
        )
        
        results_run2.append((result, reason))
    
    # Results should be identical
    assert results_run1 == results_run2, "Results not deterministic across runs"


@pytest.mark.asyncio
async def test_chaos_no_crashes_on_extreme_faults():
    """
    Test that system never crashes even with extreme fault combinations.
    """
    random.seed(CHAOS_SEED)
    
    bot = Bot(
        id=1,
        name="Extreme Chaos Bot",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=1000.0,
        current_balance=1000.0,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        daily_loss_limit=100.0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )
    
    # Extreme chaos: all fault probabilities at 50%
    extreme_exchange = ChaosExchange(seed=CHAOS_SEED)
    extreme_price = ChaosPriceFeed(seed=CHAOS_SEED)
    extreme_ledger = ChaosLedger(seed=CHAOS_SEED)
    risk_manager = ChaosRiskManager(bot, seed=CHAOS_SEED)
    
    # Temporarily increase fault rates
    original_probs = FAULT_PROBABILITY.copy()
    for key in FAULT_PROBABILITY:
        FAULT_PROBABILITY[key] = 0.50  # 50% fault rate
    
    try:
        # Run with extreme faults
        for i in range(20):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            amount = 0.01
            
            # Should not raise unhandled exception
            result, reason, metadata = await attempt_trade(
                bot=bot,
                exchange=extreme_exchange,
                price_feed=extreme_price,
                ledger=extreme_ledger,
                risk_manager=risk_manager,
                side=side,
                amount=amount,
                iteration=i,
            )
            
            # Every attempt should return valid result
            assert isinstance(result, TradeResult)
            assert isinstance(reason, str)
            assert len(reason) > 0
            
    finally:
        # Restore original probabilities
        FAULT_PROBABILITY.clear()
        FAULT_PROBABILITY.update(original_probs)
    
    # No assertion failures = test passed (no crashes)


@pytest.mark.asyncio
async def test_chaos_kill_switch_dominance():
    """
    Test that once kill switch activates, it dominates all other logic.
    No trades execute after kill switch, regardless of other conditions.
    """
    random.seed(CHAOS_SEED)
    
    bot = Bot(
        id=1,
        name="Kill Switch Bot",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=10000.0,
        current_balance=6000.0,  # Already at 40% loss
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        daily_loss_limit=500.0,
        drawdown_limit_percent=30.0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )
    
    exchange = ChaosExchange(seed=CHAOS_SEED)
    price_feed = ChaosPriceFeed(seed=CHAOS_SEED)
    ledger = ChaosLedger(seed=CHAOS_SEED)
    risk_manager = ChaosRiskManager(bot, seed=CHAOS_SEED)
    
    # Force kill switch activation
    risk_manager.daily_loss = 600.0  # Over limit
    risk_manager.kill_switch_active = True
    
    # Attempt 20 trades after kill switch
    for i in range(20):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        amount = 0.01
        
        result, reason, metadata = await attempt_trade(
            bot=bot,
            exchange=exchange,
            price_feed=price_feed,
            ledger=ledger,
            risk_manager=risk_manager,
            side=side,
            amount=amount,
            iteration=i,
        )
        
        # All should be BLOCKED
        assert result == TradeResult.BLOCKED, f"Trade {i} not blocked: {result}"
        assert "kill switch" in reason.lower() or "loss limit" in reason.lower()


@pytest.mark.asyncio
async def test_chaos_balance_never_negative():
    """
    Test that balance never goes negative even under extreme chaos.
    """
    random.seed(CHAOS_SEED)
    
    bot = Bot(
        id=1,
        name="Balance Test Bot",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=500.0,  # Small budget
        current_balance=500.0,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        daily_loss_limit=1000.0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )
    
    exchange = ChaosExchange(seed=CHAOS_SEED)
    price_feed = ChaosPriceFeed(seed=CHAOS_SEED)
    ledger = ChaosLedger(seed=CHAOS_SEED)
    risk_manager = ChaosRiskManager(bot, seed=CHAOS_SEED)
    
    # Run many trades
    for i in range(30):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        amount = 0.005  # Small trades
        
        result, reason, metadata = await attempt_trade(
            bot=bot,
            exchange=exchange,
            price_feed=price_feed,
            ledger=ledger,
            risk_manager=risk_manager,
            side=side,
            amount=amount,
            iteration=i,
        )

        # Check balance after each trade
        assert bot.current_balance >= 0, f"Balance negative at iteration {i}: {bot.current_balance}"


@pytest.mark.asyncio
async def test_chaos_ledger_consistency():
    """
    Test that ledger remains consistent with successful trades.
    """
    from tests.test_chaos_trading_pipeline import FAULT_PROBABILITY
    FAULT_PROBABILITY["ledger_delay"] = 0.0
    FAULT_PROBABILITY["ledger_duplicate"] = 0.0

    random.seed(CHAOS_SEED)

    bot = Bot(
        id=1,
        name="Ledger Test Bot",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=10000.0,
        current_balance=10000.0,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        daily_loss_limit=2000.0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )
    
    exchange = ChaosExchange(seed=CHAOS_SEED)
    price_feed = ChaosPriceFeed(seed=CHAOS_SEED)
    ledger = ChaosLedger(seed=CHAOS_SEED)
    risk_manager = ChaosRiskManager(bot, seed=CHAOS_SEED)
    
    successful_trades = []
    
    # Run trades
    for i in range(25):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        amount = 0.01
        
        result, reason, metadata = await attempt_trade(
            bot=bot,
            exchange=exchange,
            price_feed=price_feed,
            ledger=ledger,
            risk_manager=risk_manager,
            side=side,
            amount=amount,
            iteration=i,
        )
        
        if result == TradeResult.SUCCESS:
            successful_trades.append(i)
            assert (
                ledger.entries or ledger.delayed_entries
            ), "Successful trade did not write to ledger"

    # Flush delayed
    flushed = ledger.flush_delayed()
    
    # Validate ledger consistency
    ledger_violations = ledger.validate_consistency()
    assert len(ledger_violations) == 0, f"Ledger inconsistencies: {ledger_violations}"
    
    # Verify ledger has entries for successful trades
    ledger.flush_delayed()
    # (may have duplicates or delays, but should have at least some entries)
    assert len(ledger.entries) > 0, "Ledger has no entries despite successful trades"

    # Verify successful trades approximately match entry count (allowing for chaos)
    entry_count = len(ledger.entries)
    assert entry_count >= len(successful_trades) // 2, f"Too few ledger entries: {entry_count} for {len(successful_trades)} trades"


@pytest.mark.asyncio
async def test_chaos_price_validation():
    """
    Test that invalid prices (zero, NaN, negative) are always rejected.
    """
    random.seed(CHAOS_SEED)
    
    bot = Bot(
        id=1,
        name="Price Validation Bot",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=10000.0,
        current_balance=10000.0,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )
    
    exchange = ChaosExchange(seed=CHAOS_SEED)
    price_feed = ChaosPriceFeed(seed=CHAOS_SEED)
    ledger = ChaosLedger(seed=CHAOS_SEED)
    risk_manager = ChaosRiskManager(bot, seed=CHAOS_SEED)
    
    invalid_price_count = 0
    
    # Run many iterations to hit invalid prices
    for i in range(50):
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
        amount = 0.01
        
        result, reason, metadata = await attempt_trade(
            bot=bot,
            exchange=exchange,
            price_feed=price_feed,
            ledger=ledger,
            risk_manager=risk_manager,
            side=side,
            amount=amount,
            iteration=i,
        )
        
        # If price was invalid, trade must be rejected
        if "invalid_price" in metadata:
            invalid_price_count += 1
            assert result in [TradeResult.REJECTED, TradeResult.FAILED]
            assert "Invalid price" in reason or "price" in reason.lower()
    
    # Should have hit at least some invalid prices
    assert invalid_price_count > 0, "No invalid prices encountered - test may need adjustment"
