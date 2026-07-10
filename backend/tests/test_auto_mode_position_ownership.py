"""Regression tests for Auto Mode position ownership (fix-strategy-integrity).

Bug fixed: Auto Mode could open a position under one sub-strategy and let a
different sub-strategy manage/close it, because ownership only lived as a
mutable, bot-level "current strategy" pointer - never as a fact recorded on
the position itself. These tests lock down:

1. Opening a position persists owning_strategy/entry_reason on the Position row.
2. Strategy A opens; Strategy B can never be dispatched to manage/close it,
   even when B scores higher, as long as A remains entry-eligible.
3. If the in-memory pointer and the persisted position owner ever disagree
   (e.g. after a restart), the persisted fact wins (self-heal).
4. Auto Mode is free to re-score and reselect once the owning position closes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models import Bot, BotStatus, Position, PositionSide
from app.services.trading_engine import TradingEngine, TradeSignal


def _engine() -> TradingEngine:
    return TradingEngine()


def _session_with_empty_query_results() -> AsyncMock:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))
    return session


def _auto_mode_bot(bot_id: int) -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "auto_mode"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = 1000.0
    bot.budget = 1000.0
    return bot


# Regime under which BOTH mean_reversion (["trend_flat", "volatility_high"])
# and dip_recovery (["trend_down", "volatility_expanding", "volatility_high"])
# are simultaneously entry-eligible via the shared "volatility_high" tag - a
# real regime, not a mocked eligibility check.
_SHARED_ELIGIBLE_REGIME = {
    "trend_state": "flat",
    "volatility_state": "high",
    "volatility_direction": "stable",
    "liquidity_state": "normal",
    "persistence_bars": 0,
}


def _auto_state_holding(current_strategy: str, now: datetime) -> dict:
    return {
        "current_strategy": current_strategy,
        "last_switch_time": (now - timedelta(hours=1)).isoformat(),
        "last_bar_close_time": (now - timedelta(seconds=1)).isoformat(),
        "current_bar": {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        "bar_history": [],
        "current_regime": dict(_SHARED_ELIGIBLE_REGIME),
        "regime_change_count": 0,
        "strategy_metrics": {},
    }


def _force_dip_recovery_to_outscore_mean_reversion(engine: TradingEngine) -> None:
    """Deterministically make dip_recovery win the score race while leaving
    real eligibility (regime matching) untouched - isolates the ownership
    pinning logic from the numeric details of opportunity scoring."""
    def _fake_score(strategy_name, caps, metrics, **kwargs):
        final = {"mean_reversion": 1.0, "dip_recovery": 10.0}.get(strategy_name, 0.0)
        return {
            "opportunity": final, "performance": 0.0, "confidence": 0.0,
            "risk_penalty": 0.0, "final": final,
        }

    engine._score_strategy = _fake_score


class TestPositionOwnershipPersistence:
    @pytest.mark.asyncio
    async def test_opening_a_position_records_owner_and_reason(self, test_db):
        bot = Bot(
            name="b", trading_pair="BTC/USDT", strategy="mean_reversion",
            strategy_params={}, budget=1000.0, current_balance=1000.0,
            is_dry_run=True, status=BotStatus.RUNNING,
        )
        test_db.add(bot)
        await test_db.flush()

        engine = TradingEngine()
        await engine._open_or_add_position(
            bot.id, "BTC/USDT", 0.01, 50000.0, test_db,
            owning_strategy="mean_reversion",
            entry_reason="Mean Reversion: price below lower band",
        )
        await test_db.commit()

        from sqlalchemy import select
        result = await test_db.execute(select(Position).where(Position.bot_id == bot.id))
        position = result.scalar_one()
        assert position.owning_strategy == "mean_reversion"
        assert position.entry_reason == "Mean Reversion: price below lower band"

    @pytest.mark.asyncio
    async def test_adding_to_existing_position_does_not_overwrite_ownership(self, test_db):
        """A second buy into an already-open position must not let a stale/
        different owning_strategy argument overwrite the original owner -
        while a position is open, Auto only ever dispatches to its owner, so
        this also guards the invariant from the other direction."""
        bot = Bot(
            name="b", trading_pair="BTC/USDT", strategy="mean_reversion",
            strategy_params={}, budget=1000.0, current_balance=1000.0,
            is_dry_run=True, status=BotStatus.RUNNING,
        )
        test_db.add(bot)
        await test_db.flush()

        engine = TradingEngine()
        await engine._open_or_add_position(
            bot.id, "BTC/USDT", 0.01, 50000.0, test_db,
            owning_strategy="mean_reversion", entry_reason="first buy",
        )
        await test_db.flush()
        # Simulate a second buy call incorrectly tagged with a different
        # strategy - ownership must stay with the original owner.
        await engine._open_or_add_position(
            bot.id, "BTC/USDT", 0.01, 51000.0, test_db,
            owning_strategy="trend_following", entry_reason="should be ignored",
        )
        await test_db.commit()

        from sqlalchemy import select
        result = await test_db.execute(select(Position).where(Position.bot_id == bot.id))
        position = result.scalar_one()
        assert position.owning_strategy == "mean_reversion"
        assert position.entry_reason == "first buy"


class TestResolveOwningStrategy:
    def test_fixed_strategy_bot_owns_its_own_trades(self):
        engine = _engine()
        bot = MagicMock(strategy="trend_following")
        assert engine._resolve_owning_strategy(bot, None) == "trend_following"

    def test_auto_mode_bot_resolves_real_sub_strategy_from_reason(self):
        engine = _engine()
        bot = MagicMock(strategy="auto_mode")
        reason = "[Auto:mean_reversion|flat/high/normal] Mean Reversion: entering"
        assert engine._resolve_owning_strategy(bot, reason) == "mean_reversion"

    def test_auto_mode_bot_without_tagged_reason_falls_back_to_bot_strategy(self):
        engine = _engine()
        bot = MagicMock(strategy="auto_mode")
        assert engine._resolve_owning_strategy(bot, "plain untagged reason") == "auto_mode"


class TestStrategyACannotBeClosedByStrategyB:
    @pytest.mark.asyncio
    async def test_higher_scoring_strategy_cannot_take_over_an_open_position(self):
        """Strategy A (mean_reversion) opened a position and remains entry-
        eligible. Strategy B (dip_recovery) is also eligible and scores much
        higher. Auto Mode must keep dispatching to A - B must never run."""
        engine = _engine()
        bot = _auto_mode_bot(701)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        engine._save_auto_state(bot.id, _auto_state_holding("mean_reversion", now))
        _force_dip_recovery_to_outscore_mean_reversion(engine)

        owned_position = MagicMock()
        owned_position.amount = 0.01
        owned_position.owning_strategy = "mean_reversion"

        mean_reversion_signal = TradeSignal(action="hold", amount=0, reason="Mean Reversion: holding")
        dip_recovery_mock = AsyncMock(return_value=TradeSignal(action="sell", amount=0.01, reason="Dip Recovery: exit"))

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[owned_position])), \
             patch.object(engine, "_strategy_mean_reversion", new=AsyncMock(return_value=mean_reversion_signal)) as mr_mock, \
             patch.object(engine, "_strategy_dip_recovery", new=dip_recovery_mock):
            signal = await engine._strategy_auto(bot, 100.0, {}, session)

        mr_mock.assert_awaited_once()
        dip_recovery_mock.assert_not_awaited()
        assert "[Auto:mean_reversion|" in signal.reason

        state_after = engine._get_auto_state(bot.id)
        assert state_after["current_strategy"] == "mean_reversion", (
            "Auto switched ownership away from the strategy holding the open "
            "position to a higher-scoring competitor"
        )

    @pytest.mark.asyncio
    async def test_reselection_allowed_once_no_position_is_open(self):
        """Same regime/scores as above, but with NO open position - Auto must
        be free to switch to the higher-scoring strategy. Proves the pin is
        specifically about protecting open positions, not a permanent lock."""
        engine = _engine()
        bot = _auto_mode_bot(702)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        engine._save_auto_state(bot.id, _auto_state_holding("mean_reversion", now))
        _force_dip_recovery_to_outscore_mean_reversion(engine)

        dip_recovery_signal = TradeSignal(action="buy", amount=10.0, reason="Dip Recovery: confirmed reversal")

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])), \
             patch.object(engine, "_strategy_dip_recovery", new=AsyncMock(return_value=dip_recovery_signal)) as dr_mock:
            signal = await engine._strategy_auto(bot, 100.0, {}, session)

        dr_mock.assert_awaited_once()
        assert "[Auto:dip_recovery|" in signal.reason

        state_after = engine._get_auto_state(bot.id)
        assert state_after["current_strategy"] == "dip_recovery"


class TestOwnershipSelfHeal:
    @pytest.mark.asyncio
    async def test_persisted_owner_wins_over_stale_in_memory_pointer(self):
        """Simulates the in-memory pointer having drifted from (or never
        matched) the persisted position owner - e.g. after a restart where
        auto_state fell back to a default. The persisted Position.owning_
        strategy must win, and dispatch must go to the real owner, not the
        stale pointer."""
        engine = _engine()
        bot = _auto_mode_bot(703)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        # Stale/reset pointer - as if restart lost the real selection.
        stale_state = _auto_state_holding("dca_accumulator", now)
        engine._save_auto_state(bot.id, stale_state)

        owned_position = MagicMock()
        owned_position.amount = 0.01
        owned_position.owning_strategy = "mean_reversion"

        mean_reversion_signal = TradeSignal(action="hold", amount=0, reason="Mean Reversion: holding")
        dca_mock = AsyncMock(return_value=TradeSignal(action="buy", amount=10.0, reason="DCA: buy"))

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[owned_position])), \
             patch.object(engine, "_strategy_mean_reversion", new=AsyncMock(return_value=mean_reversion_signal)) as mr_mock, \
             patch.object(engine, "_strategy_dca", new=dca_mock):
            signal = await engine._strategy_auto(bot, 100.0, {}, session)

        dca_mock.assert_not_awaited()
        mr_mock.assert_awaited_once()
        assert "[Auto:mean_reversion|" in signal.reason

        state_after = engine._get_auto_state(bot.id)
        assert state_after["current_strategy"] == "mean_reversion", (
            "Self-heal did not correct the stale in-memory pointer to match "
            "the persisted position owner"
        )
