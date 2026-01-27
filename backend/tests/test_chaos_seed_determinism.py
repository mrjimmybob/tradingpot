"""
Chaos seed determinism test - ensures chaos testing is reproducible.

Verifies that:
1. Same seed produces identical chaos behavior (reproducible bugs)
2. Different seeds produce different chaos behavior (true randomness)

This is critical for debugging:
- Failed chaos tests can be replayed exactly
- Bugs found under stress are reproducible
- Failures are debuggable, not one-off flukes

If this test fails:
❌ Chaos is non-reproducible
❌ Bugs cannot be replayed
❌ Stress testing is unreliable
"""

import pytest
import random
import math
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

from app.models.bot import Bot, BotStatus
from app.services.exchange import OrderSide

# Import real chaos classes from chaos test
from tests.test_chaos_trading_pipeline import (
    ChaosExchange,
    ChaosPriceFeed,
    ChaosLedger,
    ChaosRiskManager,
    TradeResult,
    attempt_trade,
    LedgerReason,
)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

SEED_PRIMARY = 123  # Primary seed for determinism testing
SEED_DIFFERENT = 999  # Different seed to verify randomness
CHAOS_ITERATIONS = 150  # Number of trade attempts per run
INITIAL_BUDGET = 10000.0
TRADE_SIZE = 0.01  # BTC


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ChaosRunResult:
    """Complete results from a chaos test run."""
    trade_results: List[str]  # List of result types (SUCCESS, FAILED, etc.)
    final_balance: float
    ledger_entry_count: int
    kill_switch_active: bool
    exception_count: int
    successful_trades: int
    failed_trades: int
    blocked_trades: int
    rejected_trades: int


# ============================================================================
# CHAOS TEST RUNNER
# ============================================================================


async def run_chaos_test_with_seed(
    seed: int,
    iterations: int = CHAOS_ITERATIONS,
) -> ChaosRunResult:
    """
    Run a complete chaos test with a specific seed.

    Args:
        seed: Random seed for reproducibility
        iterations: Number of trade attempts

    Returns:
        Complete chaos run results
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Create bot with fixed configuration
    bot = Bot(
        id=1,
        name=f"Chaos Seed Test Bot (seed={seed})",
        trading_pair="BTC/USDT",
        strategy="grid",
        budget=INITIAL_BUDGET,
        current_balance=INITIAL_BUDGET,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        daily_loss_limit=1000.0,
        stop_loss_percent=10.0,
        drawdown_limit_percent=30.0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        started_at=datetime(2025, 1, 15, 10, 0, 0),
    )

    # Create chaos components with the specified seed
    exchange = ChaosExchange(seed=seed)
    price_feed = ChaosPriceFeed(seed=seed)
    ledger = ChaosLedger(seed=seed)
    risk_manager = ChaosRiskManager(bot, seed=seed)

    # Track results
    trade_results: List[str] = []
    exception_count = 0
    successful_trades = 0
    failed_trades = 0
    blocked_trades = 0
    rejected_trades = 0

    # Run chaos trading loop
    for i in range(iterations):
        # Alternate buy/sell
        side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL

        # Random amount (but deterministic with seed)
        rng = random.Random(seed + i)
        amount = rng.uniform(0.01, 0.1)

        try:
            # Attempt trade with full chaos
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

            # Record result
            trade_results.append(result.value)

            # Count by type
            if result == TradeResult.SUCCESS:
                successful_trades += 1
            elif result == TradeResult.FAILED:
                failed_trades += 1
            elif result == TradeResult.BLOCKED:
                blocked_trades += 1
            elif result == TradeResult.REJECTED:
                rejected_trades += 1

        except Exception as e:
            # Count exceptions
            exception_count += 1
            trade_results.append(f"EXCEPTION:{type(e).__name__}")

    # Flush delayed ledger entries
    ledger.flush_delayed()

    # Return complete results
    return ChaosRunResult(
        trade_results=trade_results,
        final_balance=bot.current_balance,
        ledger_entry_count=len(ledger.entries),
        kill_switch_active=risk_manager.kill_switch_active,
        exception_count=exception_count,
        successful_trades=successful_trades,
        failed_trades=failed_trades,
        blocked_trades=blocked_trades,
        rejected_trades=rejected_trades,
    )


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================


def compare_chaos_results(
    result1: ChaosRunResult,
    result2: ChaosRunResult,
    tolerance: float = 0.01,
) -> Tuple[bool, List[str]]:
    """
    Compare two chaos run results for equality.

    Args:
        result1: First run result
        result2: Second run result
        tolerance: Floating-point tolerance for balance comparison

    Returns:
        (is_equal, list_of_differences)
    """
    differences = []

    # Compare trade result sequences
    if result1.trade_results != result2.trade_results:
        # Find first mismatch
        for i, (r1, r2) in enumerate(zip(result1.trade_results, result2.trade_results)):
            if r1 != r2:
                differences.append(
                    f"Trade result mismatch at iteration {i}: '{r1}' vs '{r2}'"
                )
                break
        if len(result1.trade_results) != len(result2.trade_results):
            differences.append(
                f"Trade result count: {len(result1.trade_results)} vs {len(result2.trade_results)}"
            )

    # Compare final balance
    balance_diff = abs(result1.final_balance - result2.final_balance)
    if balance_diff > tolerance:
        differences.append(
            f"Final balance: ${result1.final_balance:.2f} vs ${result2.final_balance:.2f} "
            f"(diff: ${balance_diff:.2f})"
        )

    # Compare ledger entry count
    if result1.ledger_entry_count != result2.ledger_entry_count:
        differences.append(
            f"Ledger entries: {result1.ledger_entry_count} vs {result2.ledger_entry_count}"
        )

    # Compare kill switch state
    if result1.kill_switch_active != result2.kill_switch_active:
        differences.append(
            f"Kill switch: {result1.kill_switch_active} vs {result2.kill_switch_active}"
        )

    # Compare exception count
    if result1.exception_count != result2.exception_count:
        differences.append(
            f"Exceptions: {result1.exception_count} vs {result2.exception_count}"
        )

    # Compare trade counts
    if result1.successful_trades != result2.successful_trades:
        differences.append(
            f"Successful trades: {result1.successful_trades} vs {result2.successful_trades}"
        )

    if result1.failed_trades != result2.failed_trades:
        differences.append(
            f"Failed trades: {result1.failed_trades} vs {result2.failed_trades}"
        )

    if result1.blocked_trades != result2.blocked_trades:
        differences.append(
            f"Blocked trades: {result1.blocked_trades} vs {result2.blocked_trades}"
        )

    if result1.rejected_trades != result2.rejected_trades:
        differences.append(
            f"Rejected trades: {result1.rejected_trades} vs {result2.rejected_trades}"
        )

    is_equal = len(differences) == 0
    return is_equal, differences


def validate_chaos_invariants(result: ChaosRunResult) -> List[str]:
    """
    Validate chaos test invariants.

    Args:
        result: Chaos run result

    Returns:
        List of violations (empty if all valid)
    """
    violations = []

    # Invariant 1: At least one trade attempt occurred
    total_attempts = len(result.trade_results)
    if total_attempts == 0:
        violations.append("No trade attempts occurred")

    # Invariant 2: Balance is finite
    if not math.isfinite(result.final_balance):
        violations.append(f"Final balance is not finite: {result.final_balance}")

    # Invariant 3: Balance is non-negative
    if result.final_balance < 0:
        violations.append(f"Final balance is negative: ${result.final_balance:.2f}")

    # Invariant 4: Trade counts sum correctly
    total_categorized = (
        result.successful_trades +
        result.failed_trades +
        result.blocked_trades +
        result.rejected_trades
    )
    if total_categorized > total_attempts:
        violations.append(
            f"Trade counts exceed attempts: {total_categorized} > {total_attempts}"
        )

    # Invariant 5: All result strings are non-empty
    for i, res in enumerate(result.trade_results):
        if not res or res.strip() == "":
            violations.append(f"Empty result string at iteration {i}")

    return violations


# ============================================================================
# MAIN TEST
# ============================================================================


@pytest.mark.asyncio
async def test_chaos_seed_determinism():
    """
    Verify chaos testing is deterministic with same seed, random with different seeds.

    Test structure:
    1. Run chaos test with seed=123 → result1
    2. Reset state completely
    3. Run chaos test with seed=123 → result2
    4. Assert result1 == result2 (DETERMINISM)
    5. Run chaos test with seed=999 → result3
    6. Assert result1 != result3 (RANDOMNESS)

    Verifies:
    - Same seed produces identical behavior (reproducible bugs)
    - Different seeds produce different behavior (true chaos)
    - All chaos invariants hold
    - No NaN or infinite balances
    - Ledger consistency

    Ensures that chaos bugs can be debugged by replaying with same seed.
    """

    print(f"\n{'='*70}")
    print(f"Chaos Seed Determinism Test")
    print(f"{'='*70}")
    print(f"Primary seed: {SEED_PRIMARY}")
    print(f"Different seed: {SEED_DIFFERENT}")
    print(f"Iterations per run: {CHAOS_ITERATIONS}")
    print(f"Initial budget: ${INITIAL_BUDGET:.2f}")
    print(f"{'='*70}\n")

    # ========================================================================
    # RUN #1: Chaos test with seed=123
    # ========================================================================

    print(f"Running chaos test #1 (seed={SEED_PRIMARY})...")

    result_run1 = await run_chaos_test_with_seed(
        seed=SEED_PRIMARY,
        iterations=CHAOS_ITERATIONS,
    )

    print(f"✓ Run 1 complete")
    print(f"  Final balance: ${result_run1.final_balance:.2f}")
    print(f"  Successful trades: {result_run1.successful_trades}")
    print(f"  Failed trades: {result_run1.failed_trades}")
    print(f"  Blocked trades: {result_run1.blocked_trades}")
    print(f"  Rejected trades: {result_run1.rejected_trades}")
    print(f"  Ledger entries: {result_run1.ledger_entry_count}")
    print(f"  Kill switch: {result_run1.kill_switch_active}")
    print(f"  Exceptions: {result_run1.exception_count}")

    # Validate invariants for run 1
    violations_run1 = validate_chaos_invariants(result_run1)
    assert len(violations_run1) == 0, (
        f"❌ Chaos invariants violated in run 1:\n" +
        "\n".join(f"  - {v}" for v in violations_run1)
    )
    print(f"  ✓ All invariants valid")

    # ========================================================================
    # RESET STATE COMPLETELY
    # ========================================================================

    print(f"\nResetting state...")

    # Reset random module state
    random.seed(SEED_PRIMARY)

    # Note: Bot, exchange, ledger, etc. are recreated in run_chaos_test_with_seed
    # so state is automatically isolated

    # ========================================================================
    # RUN #2: Chaos test with same seed=123 (should be identical)
    # ========================================================================

    print(f"Running chaos test #2 (seed={SEED_PRIMARY}, same as run 1)...")

    result_run2 = await run_chaos_test_with_seed(
        seed=SEED_PRIMARY,
        iterations=CHAOS_ITERATIONS,
    )

    print(f"✓ Run 2 complete")
    print(f"  Final balance: ${result_run2.final_balance:.2f}")
    print(f"  Successful trades: {result_run2.successful_trades}")
    print(f"  Failed trades: {result_run2.failed_trades}")
    print(f"  Blocked trades: {result_run2.blocked_trades}")
    print(f"  Rejected trades: {result_run2.rejected_trades}")
    print(f"  Ledger entries: {result_run2.ledger_entry_count}")
    print(f"  Kill switch: {result_run2.kill_switch_active}")
    print(f"  Exceptions: {result_run2.exception_count}")

    # Validate invariants for run 2
    violations_run2 = validate_chaos_invariants(result_run2)
    assert len(violations_run2) == 0, (
        f"❌ Chaos invariants violated in run 2:\n" +
        "\n".join(f"  - {v}" for v in violations_run2)
    )
    print(f"  ✓ All invariants valid")

    # ========================================================================
    # ASSERTION: Run 1 and Run 2 should be IDENTICAL (same seed)
    # ========================================================================

    print(f"\n{'='*70}")
    print("Comparing run 1 vs run 2 (same seed - should be IDENTICAL)...")
    print(f"{'='*70}\n")

    is_equal, differences = compare_chaos_results(result_run1, result_run2)

    if not is_equal:
        print(f"❌ DETERMINISM FAILURE: Runs with same seed differ!")
        print(f"\nDifferences found:")
        for diff in differences:
            print(f"  - {diff}")
        pytest.fail(
            f"❌ Chaos test is NOT deterministic with same seed!\n"
            f"⚠️ This means chaos bugs CANNOT be reproduced\n"
            f"⚠️ Failures cannot be debugged\n"
            f"\nDifferences:\n" +
            "\n".join(f"  {d}" for d in differences)
        )

    print(f"✅ DETERMINISM VERIFIED")
    print(f"  ✓ Trade results identical: {len(result_run1.trade_results)} results")
    print(f"  ✓ Final balance identical: ${result_run1.final_balance:.2f}")
    print(f"  ✓ Ledger count identical: {result_run1.ledger_entry_count} entries")
    print(f"  ✓ Kill switch state identical: {result_run1.kill_switch_active}")
    print(f"  ✓ Exception count identical: {result_run1.exception_count}")
    print(f"  ✓ All trade counts identical")

    # ========================================================================
    # RUN #3: Chaos test with DIFFERENT seed=999 (should differ)
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"Running chaos test #3 (seed={SEED_DIFFERENT}, different from run 1)...")
    print(f"{'='*70}\n")

    result_run3 = await run_chaos_test_with_seed(
        seed=SEED_DIFFERENT,
        iterations=CHAOS_ITERATIONS,
    )

    print(f"✓ Run 3 complete")
    print(f"  Final balance: ${result_run3.final_balance:.2f}")
    print(f"  Successful trades: {result_run3.successful_trades}")
    print(f"  Failed trades: {result_run3.failed_trades}")
    print(f"  Blocked trades: {result_run3.blocked_trades}")
    print(f"  Rejected trades: {result_run3.rejected_trades}")
    print(f"  Ledger entries: {result_run3.ledger_entry_count}")
    print(f"  Kill switch: {result_run3.kill_switch_active}")
    print(f"  Exceptions: {result_run3.exception_count}")

    # Validate invariants for run 3
    violations_run3 = validate_chaos_invariants(result_run3)
    assert len(violations_run3) == 0, (
        f"❌ Chaos invariants violated in run 3:\n" +
        "\n".join(f"  - {v}" for v in violations_run3)
    )
    print(f"  ✓ All invariants valid")

    # ========================================================================
    # ASSERTION: Run 1 and Run 3 should DIFFER (different seeds)
    # ========================================================================

    print(f"\n{'='*70}")
    print("Comparing run 1 vs run 3 (different seeds - should DIFFER)...")
    print(f"{'='*70}\n")

    is_equal_diff, differences_diff = compare_chaos_results(result_run1, result_run3)

    if is_equal_diff:
        print(f"❌ RANDOMNESS FAILURE: Runs with different seeds are identical!")
        pytest.fail(
            f"❌ Chaos test is NOT random with different seeds!\n"
            f"⚠️ This means chaos is not actually testing different scenarios\n"
            f"⚠️ The random seed is not being used properly\n"
            f"\nRun 1 seed: {SEED_PRIMARY}\n"
            f"Run 3 seed: {SEED_DIFFERENT}\n"
            f"Both produced identical results, which should be impossible with chaos!"
        )

    print(f"✅ RANDOMNESS VERIFIED")
    print(f"\nDifferences between seed={SEED_PRIMARY} and seed={SEED_DIFFERENT}:")
    for diff in differences_diff[:5]:  # Show first 5 differences
        print(f"  - {diff}")
    if len(differences_diff) > 5:
        print(f"  ... and {len(differences_diff) - 5} more differences")

    # Verify at least one key metric differs
    key_differences = 0

    if result_run1.final_balance != result_run3.final_balance:
        key_differences += 1

    if result_run1.successful_trades != result_run3.successful_trades:
        key_differences += 1

    if result_run1.kill_switch_active != result_run3.kill_switch_active:
        key_differences += 1

    if result_run1.trade_results != result_run3.trade_results:
        key_differences += 1

    assert key_differences > 0, (
        f"❌ No key differences found between different seeds!\n"
        f"At least one of: final_balance, successful_trades, kill_switch, trade_results "
        f"should differ with different seeds"
    )

    print(f"  ✓ {key_differences} key metrics differ (as expected)")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print("✅ CHAOS SEED DETERMINISM FULLY VERIFIED")
    print(f"{'='*70}")
    print("\nDeterminism checks (same seed):")
    print(f"  ✓ Identical trade result sequences")
    print(f"  ✓ Identical final balances")
    print(f"  ✓ Identical ledger entry counts")
    print(f"  ✓ Identical kill switch states")
    print(f"  ✓ Identical exception counts")
    print(f"  ✓ Identical trade counts (success/fail/blocked/rejected)")
    print(f"\nRandomness checks (different seed):")
    print(f"  ✓ Different behavior observed ({key_differences} key differences)")
    print(f"  ✓ Chaos actually tests varied scenarios")
    print(f"\nInvariant checks:")
    print(f"  ✓ All balances finite and non-negative")
    print(f"  ✓ Trade counts consistent")
    print(f"  ✓ No empty result strings")
    print(f"\n🎯 Chaos testing is reproducible AND random")
    print(f"🎯 Bugs found in chaos tests CAN be debugged")
    print(f"🎯 Failures CAN be replayed with same seed")
    print(f"{'='*70}\n")


# ============================================================================
# ADDITIONAL TEST: Verify specific seed reproducibility
# ============================================================================


@pytest.mark.asyncio
async def test_chaos_specific_seed_replay():
    """
    Verify that a specific chaos seed can be replayed multiple times.

    This simulates the debugging workflow:
    1. Chaos test fails with seed X
    2. Developer reruns with seed X
    3. Should see exact same failure

    This test runs the same seed 3 times and verifies all 3 are identical.
    """
    print(f"\n{'='*70}")
    print(f"Chaos Seed Replay Test (3 runs with seed={SEED_PRIMARY})")
    print(f"{'='*70}\n")

    # Run chaos test 3 times with same seed
    results = []

    for run_num in range(1, 4):
        print(f"Running iteration {run_num}/3...")

        result = await run_chaos_test_with_seed(
            seed=SEED_PRIMARY,
            iterations=50,  # Shorter for faster testing
        )

        results.append(result)

        print(f"  Balance: ${result.final_balance:.2f}, "
              f"Trades: {result.successful_trades}S/{result.failed_trades}F/"
              f"{result.blocked_trades}B/{result.rejected_trades}R")

    # Compare all pairs
    print(f"\nComparing results...")

    # Run 1 vs Run 2
    is_equal_12, diff_12 = compare_chaos_results(results[0], results[1])
    assert is_equal_12, (
        f"❌ Run 1 vs Run 2 differ:\n" + "\n".join(diff_12)
    )
    print(f"  ✓ Run 1 == Run 2")

    # Run 2 vs Run 3
    is_equal_23, diff_23 = compare_chaos_results(results[1], results[2])
    assert is_equal_23, (
        f"❌ Run 2 vs Run 3 differ:\n" + "\n".join(diff_23)
    )
    print(f"  ✓ Run 2 == Run 3")

    # Run 1 vs Run 3
    is_equal_13, diff_13 = compare_chaos_results(results[0], results[2])
    assert is_equal_13, (
        f"❌ Run 1 vs Run 3 differ:\n" + "\n".join(diff_13)
    )
    print(f"  ✓ Run 1 == Run 3")

    print(f"\n✅ Same seed produces identical results across 3 runs")
    print(f"🎯 Specific failures CAN be replayed for debugging")
    print(f"{'='*70}\n")


# ============================================================================
# ADDITIONAL TEST: Verify seed isolation
# ============================================================================


@pytest.mark.asyncio
async def test_chaos_seed_isolation():
    """
    Verify that chaos components don't leak state between runs.

    Tests that each run is completely isolated even when run back-to-back
    without explicit state reset.
    """
    print(f"\n{'='*70}")
    print(f"Chaos Seed Isolation Test")
    print(f"{'='*70}\n")

    # Run with seed 100
    print(f"Running with seed=100...")
    result_100a = await run_chaos_test_with_seed(seed=100, iterations=50)
    print(f"  Balance: ${result_100a.final_balance:.2f}")

    # Immediately run with seed 200 (no explicit reset)
    print(f"Running with seed=200...")
    result_200 = await run_chaos_test_with_seed(seed=200, iterations=50)
    print(f"  Balance: ${result_200.final_balance:.2f}")

    # Run with seed 100 again (should match first run)
    print(f"Running with seed=100 again...")
    result_100b = await run_chaos_test_with_seed(seed=100, iterations=50)
    print(f"  Balance: ${result_100b.final_balance:.2f}")

    # Verify first and third runs are identical (same seed)
    is_equal, differences = compare_chaos_results(result_100a, result_100b)
    assert is_equal, (
        f"❌ Seed isolation failed - runs with seed=100 differ:\n" +
        "\n".join(differences) +
        "\n\n⚠️ This suggests state is leaking between runs"
    )

    print(f"\n✅ Seed isolation verified")
    print(f"  ✓ Seed=100 run 1 == Seed=100 run 2")
    print(f"  ✓ Intervening seed=200 run did not affect state")
    print(f"  ✓ No state leakage between chaos runs")
    print(f"{'='*70}\n")
