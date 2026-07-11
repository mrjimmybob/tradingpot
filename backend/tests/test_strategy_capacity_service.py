"""Tests for StrategyCapacityService (add-trading-safety-boundaries).

Before this file, `is_strategy_at_capacity` (auto-mode eligibility) and
`check_capacity_for_trade` (execution-time enforcement) had zero direct
test coverage - only incidental exercise via mocked usage in unrelated
integration tests. Uses a real in-memory SQLite session (the `test_db`
fixture from conftest.py) since the logic under test is the SQL
aggregation itself.
"""
from __future__ import annotations

import pytest

from app.models import Bot, BotStatus
from app.services.strategy_capacity import StrategyCapacityService, STRATEGY_CAPACITY_CONFIG


@pytest.fixture(autouse=True)
def _reset_capacity_config():
    """STRATEGY_CAPACITY_CONFIG is process-global mutable state (mutated via
    update_capacity_config) - isolate each test from it regardless of order."""
    original = {name: dict(cfg) for name, cfg in STRATEGY_CAPACITY_CONFIG.items()}
    yield
    STRATEGY_CAPACITY_CONFIG.clear()
    STRATEGY_CAPACITY_CONFIG.update(original)


async def _make_bot(test_db, *, strategy: str, current_balance: float, status=BotStatus.RUNNING) -> Bot:
    bot = Bot(
        name=f"bot-{strategy}-{current_balance}",
        trading_pair="BTC/USDT",
        strategy=strategy,
        budget=current_balance,
        current_balance=current_balance,
        is_dry_run=True,
        status=status,
    )
    test_db.add(bot)
    await test_db.flush()
    return bot


class TestIsStrategyAtCapacity:
    """Auto-mode eligibility gate - bot-count based capacity."""

    @pytest.mark.asyncio
    async def test_at_capacity_when_max_concurrent_bots_reached(self, test_db):
        await _make_bot(test_db, strategy="mean_reversion", current_balance=500.0)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        service.update_capacity_config("mean_reversion", max_concurrent_bots=1)

        is_at_capacity, reason = await service.is_strategy_at_capacity("mean_reversion")

        assert is_at_capacity is True
        assert "bot count" in reason

    @pytest.mark.asyncio
    async def test_not_at_capacity_below_limit(self, test_db):
        await _make_bot(test_db, strategy="mean_reversion", current_balance=500.0)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        service.update_capacity_config("mean_reversion", max_concurrent_bots=3)

        is_at_capacity, reason = await service.is_strategy_at_capacity("mean_reversion")

        assert is_at_capacity is False
        assert reason == "not at capacity"

    @pytest.mark.asyncio
    async def test_unlimited_by_default(self, test_db):
        """Default config (None = unlimited) preserves current behavior -
        many bots on one strategy must never trip the gate."""
        for _ in range(5):
            await _make_bot(test_db, strategy="dca_accumulator", current_balance=100.0)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        is_at_capacity, _ = await service.is_strategy_at_capacity("dca_accumulator")

        assert is_at_capacity is False

    @pytest.mark.asyncio
    async def test_stopped_bots_do_not_count_toward_capacity(self, test_db):
        await _make_bot(test_db, strategy="mean_reversion", current_balance=500.0, status=BotStatus.STOPPED)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        service.update_capacity_config("mean_reversion", max_concurrent_bots=1)

        is_at_capacity, _ = await service.is_strategy_at_capacity("mean_reversion")

        assert is_at_capacity is False


class TestCheckCapacityForTrade:
    """Execution-time enforcement - allocation-% based capacity."""

    @pytest.mark.asyncio
    async def test_order_resized_to_fit_remaining_allocation(self, test_db):
        bot_a = await _make_bot(test_db, strategy="mean_reversion", current_balance=500.0)
        await _make_bot(test_db, strategy="dca_accumulator", current_balance=500.0)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        service.update_capacity_config("mean_reversion", max_allocation_pct=60.0)

        # portfolio_total=1000, current mean_reversion allocation=500 (50%).
        # A $200 order would push it to 70% > 60% cap; remaining capacity to
        # the 60% ceiling is 600-500=$100, so it must resize to $100, not
        # block outright.
        result = await service.check_capacity_for_trade(bot_a.id, "mean_reversion", 200.0)

        assert result.ok is True
        assert result.adjusted_amount == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_order_blocked_when_already_at_allocation_cap(self, test_db):
        bot_a = await _make_bot(test_db, strategy="mean_reversion", current_balance=600.0)
        await _make_bot(test_db, strategy="dca_accumulator", current_balance=400.0)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        service.update_capacity_config("mean_reversion", max_allocation_pct=60.0)

        # portfolio_total=1000, mean_reversion allocation already at 600 (60%
        # = the cap) -> zero remaining capacity -> any additional buy blocks.
        result = await service.check_capacity_for_trade(bot_a.id, "mean_reversion", 50.0)

        assert result.ok is False
        assert result.adjusted_amount is None

    @pytest.mark.asyncio
    async def test_unlimited_allocation_never_blocks(self, test_db):
        bot_a = await _make_bot(test_db, strategy="mean_reversion", current_balance=999_000.0)
        await test_db.commit()

        service = StrategyCapacityService(test_db)
        result = await service.check_capacity_for_trade(bot_a.id, "mean_reversion", 500.0)

        assert result.ok is True
        assert result.adjusted_amount == pytest.approx(500.0)
