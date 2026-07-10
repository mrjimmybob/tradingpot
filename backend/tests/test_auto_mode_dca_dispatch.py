"""Regression tests for Auto Mode invoking DCA (dca_accumulator).

Root cause: _strategy_dca refused to run whenever bot.strategy == "auto_mode".
That check was meant to stop DCA's clock-driven, never-sell logic from
running unsupervised, but bot.strategy reads "auto_mode" for every
Auto-managed bot regardless of which sub-strategy Auto has actually selected
(Auto dispatches sub-strategies using the same bot object). So the guard
fired on every single Auto -> DCA call, including Auto's own designated
emergency fallback, and DCA silently no-opped every time Auto selected it.

Fix: _strategy_auto's dispatch now marks its own calls explicitly via
params["_invoked_by_auto"], and _strategy_dca's guard only blocks when
bot.strategy == "auto_mode" AND that marker is absent - i.e. a direct,
unsupervised call bypassing Auto's own eligibility/regime gate. bot.strategy
itself is never touched, so diagnostics keep showing the real bot
configuration and the real selected sub-strategy (via current_strategy /
the "[Auto:<strategy>|<regime>]" reason prefix), not a fake strategy name.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.trading_engine import TradingEngine


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
    bot.started_at = datetime.utcnow() - timedelta(hours=2)
    return bot


_NEUTRAL_REGIME = {
    "trend_state": "flat",
    "volatility_state": "medium",
    "volatility_direction": "stable",
    "liquidity_state": "normal",
    "persistence_bars": 0,
}

_DCA_PARAMS = {
    "regime_filter_enabled": False,  # isolate dispatch behaviour from regime gating
    "immediate_first_buy": True,
    "interval_minutes": 0,
}


# ---------------------------------------------------------------------------
# 1. Auto selects DCA -> DCA executes real logic
# ---------------------------------------------------------------------------

class TestAutoSelectsDca:
    @pytest.mark.asyncio
    async def test_auto_selected_dca_produces_a_real_buy(self):
        """When Auto's own state already has dca_accumulator selected, Auto
        Mode's dispatch must run DCA's real logic (a buy), not the old
        'Not intended for use inside auto_mode' hold."""
        engine = _engine()
        bot = _auto_mode_bot(801)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        auto_state = {
            "current_strategy": "dca_accumulator",
            "last_switch_time": (now - timedelta(hours=1)).isoformat(),
            "last_bar_close_time": (now - timedelta(seconds=1)).isoformat(),
            "current_bar": {"open": 64000, "high": 64000, "low": 64000, "close": 64000},
            "bar_history": [],
            "current_regime": dict(_NEUTRAL_REGIME),
            "regime_change_count": 0,
            "strategy_metrics": {},
        }
        engine._save_auto_state(bot.id, auto_state)

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])), \
             patch.object(engine, "_get_last_order", new=AsyncMock(return_value=None)), \
             patch.object(engine, "_get_order_count", new=AsyncMock(return_value=0)):
            signal = await engine._strategy_auto(bot, 64000.0, dict(_DCA_PARAMS), session)

        assert signal is not None
        assert signal.action == "buy", (
            f"Auto-selected DCA did not execute a real buy: {signal.reason}"
        )
        assert "Not intended" not in (signal.reason or "")
        assert "DCA buy" in (signal.reason or "")
        # Diagnostics show the real selected sub-strategy, not the bot's
        # top-level "auto_mode" configuration.
        assert "[Auto:dca_accumulator|" in signal.reason
        assert bot.strategy == "auto_mode", "bot.strategy must never be spoofed"


# ---------------------------------------------------------------------------
# 2. Emergency fallback to DCA works
# ---------------------------------------------------------------------------

class TestEmergencyFallbackToDca:
    @pytest.mark.asyncio
    async def test_fallback_to_dca_executes_when_everything_else_is_unavailable(self):
        """When every other strategy is regime-mismatched or in cooldown -
        including dca_accumulator's own cooldown metric - Auto's hard
        fallback (eligible_strategies = ["dca_accumulator"]) must still run
        DCA's real logic, not silently no-op."""
        engine = _engine()
        bot = _auto_mode_bot(802)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        future_cooldown = (now + timedelta(hours=1)).isoformat()
        all_strategies = [
            "trend_following", "volatility_breakout", "mean_reversion",
            "adaptive_grid", "dip_recovery", "dca_accumulator",
        ]
        # Cooldown everything, including DCA itself, to force the true
        # "no eligible strategies" fallback branch rather than DCA winning
        # on its own regime-eligibility merits.
        strategy_metrics = {name: {"cooldown_until": future_cooldown} for name in all_strategies}

        # trend_down / volatility_low regime: only dip_recovery would
        # otherwise match by regime, and it too is cooled down above.
        regime = {
            "trend_state": "down",
            "volatility_state": "low",
            "volatility_direction": "stable",
            "liquidity_state": "normal",
            "persistence_bars": 0,
        }
        auto_state = {
            "current_strategy": "adaptive_grid",
            "last_switch_time": (now - timedelta(hours=1)).isoformat(),
            "last_bar_close_time": (now - timedelta(seconds=1)).isoformat(),
            "current_bar": {"open": 64000, "high": 64000, "low": 64000, "close": 64000},
            "bar_history": [],
            "current_regime": regime,
            "regime_change_count": 0,
            "strategy_metrics": strategy_metrics,
        }
        engine._save_auto_state(bot.id, auto_state)

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])), \
             patch.object(engine, "_get_last_order", new=AsyncMock(return_value=None)), \
             patch.object(engine, "_get_order_count", new=AsyncMock(return_value=0)):
            signal = await engine._strategy_auto(bot, 64000.0, dict(_DCA_PARAMS), session)

        assert signal is not None
        assert signal.action == "buy", (
            f"Emergency fallback to DCA did not produce a real buy: {signal.reason}"
        )
        assert "DCA buy" in (signal.reason or "")

        state_after = engine._get_auto_state(bot.id)
        assert state_after["current_strategy"] == "dca_accumulator"


# ---------------------------------------------------------------------------
# 3. No recursion occurs
# ---------------------------------------------------------------------------

class TestNoRecursion:
    @pytest.mark.asyncio
    async def test_repeated_ticks_with_dca_selected_do_not_recurse_or_hang(self):
        """Auto dispatching to DCA over many consecutive ticks (DCA never
        sells, so the position - if any - never closes and Auto keeps
        calling it) must complete every call directly, with no runaway
        recursion or stack growth from the dispatch marker."""
        engine = _engine()
        bot = _auto_mode_bot(803)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])), \
             patch.object(engine, "_get_last_order", new=AsyncMock(return_value=None)), \
             patch.object(engine, "_get_order_count", new=AsyncMock(return_value=0)):
            for i in range(10):
                auto_state = engine._get_auto_state(bot.id) or {
                    "current_strategy": "dca_accumulator",
                    "last_switch_time": (now - timedelta(hours=1)).isoformat(),
                    "strategy_metrics": {},
                }
                auto_state["last_bar_close_time"] = (now - timedelta(seconds=1)).isoformat()
                auto_state["current_bar"] = {"open": 64000, "high": 64000, "low": 64000, "close": 64000}
                auto_state["bar_history"] = auto_state.get("bar_history", [])
                auto_state["current_regime"] = dict(_NEUTRAL_REGIME)
                auto_state["regime_change_count"] = 0
                engine._save_auto_state(bot.id, auto_state)

                # interval_minutes=0 with immediate_first_buy lets DCA buy on
                # every tick in this stub - the point is that _strategy_auto
                # never re-enters itself doing so, not DCA's own pacing.
                signal = await engine._strategy_auto(bot, 64000.0, dict(_DCA_PARAMS), session)

                assert signal is not None
                assert "Not intended" not in (signal.reason or "")

        # No exception, no hang, no stack overflow across 10 ticks - and the
        # dispatch marker never leaked into the bot's persisted identity.
        assert bot.strategy == "auto_mode"


# ---------------------------------------------------------------------------
# 4. Standalone DCA still works, and the original guard's real edge case
#    (a direct, unsupervised call on an auto_mode bot) is still blocked.
# ---------------------------------------------------------------------------

class TestStandaloneDcaUnchanged:
    @pytest.mark.asyncio
    async def test_standalone_dca_bot_still_buys(self):
        engine = _engine()
        bot = MagicMock()
        bot.id = 804
        bot.strategy = "dca_accumulator"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 1000.0
        bot.budget = 1000.0
        bot.started_at = datetime.utcnow() - timedelta(hours=2)
        session = _session_with_empty_query_results()

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])), \
             patch.object(engine, "_get_last_order", new=AsyncMock(return_value=None)), \
             patch.object(engine, "_get_order_count", new=AsyncMock(return_value=0)):
            signal = await engine._strategy_dca(bot, 64000.0, dict(_DCA_PARAMS), session)

        assert signal is not None
        assert signal.action == "buy"
        assert "DCA buy" in signal.reason

    @pytest.mark.asyncio
    async def test_direct_unsupervised_call_on_auto_mode_bot_still_blocked(self):
        """The original guard's real, still-valid edge case: something
        calling _strategy_dca directly on an auto_mode bot, bypassing
        Auto's own eligibility/regime supervision entirely, must still be
        refused."""
        engine = _engine()
        bot = _auto_mode_bot(805)
        session = _session_with_empty_query_results()

        signal = await engine._strategy_dca(bot, 64000.0, dict(_DCA_PARAMS), session)

        assert signal is not None
        assert signal.action == "hold"
        assert "Not intended" in signal.reason
