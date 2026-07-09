"""Regression tests for the 4 Auto Mode correctness fixes from the Auto Mode
investigation report (regime misclassification, force-exit-on-ineligibility,
eligibility-loss-as-failure, and capability/strategy metadata mismatches).

Fix 1 - Separate entry eligibility from exit management:
    _is_strategy_eligible() must gate NEW entries only. An already-open
    position must be managed by its own strategy's exit rules, never force-
    sold by Auto Mode just because the strategy is no longer entry-eligible.

Fix 2 - Eligibility loss is not strategy failure:
    Losing entry eligibility (regime rotation) must never increment
    failure_count / set a cooldown / count toward blacklisting.

Fix 3 - Fix market regime horizon:
    Trend classification must catch a slow, sustained move (that no single
    short slice looks like a trend) while still classifying genuine chop as
    flat and a fast pump/dump as up/down.

Fix 4 - Align Auto capability definitions with real strategy entry logic:
    _get_strategy_capabilities()'s allowed_regimes must match each
    strategy's own internal regime gate where one exists.
"""
from __future__ import annotations

import math
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
    return bot


def _vb_open_position_state(current_price: float = 64300.0) -> dict:
    """volatility_breakout's own per-bot state, as if it had already opened
    and is holding a long position well clear of its trailing stop."""
    return {
        "bars": [
            {"open": 64000, "high": 64050, "low": 63950, "close": 64000}
            for _ in range(25)
        ],
        "current_bar": None,
        "bb_width_history": [],
        "atr_history": [],
        "compression_active": False,
        "compression_bars": 0,
        "compression_start": None,
        "breakout_armed": False,
        "entry_price": 64000.0,
        "entry_atr": 200.0,
        "highest_price": 64500.0,
        "trailing_stop": 64100.0,  # well below current_price - no legitimate stop-out
        "bars_since_entry": 10,     # past failed_breakout_bars (default 3)
        "last_breakout_attempt": None,
    }


# A regime with no "volatility_expanding" tag - after Fix 4, volatility_breakout's
# capability entry is exactly ["volatility_expanding"], so this regime makes it
# entry-ineligible while a real open position (above) is being held.
_VB_INELIGIBLE_REGIME = {
    "trend_state": "flat",
    "volatility_state": "low",
    "volatility_direction": "stable",
    "liquidity_state": "normal",
    "persistence_bars": 0,
}


def _pinned_auto_state(now: datetime) -> dict:
    return {
        "current_strategy": "volatility_breakout",
        "last_switch_time": (now - timedelta(hours=1)).isoformat(),
        "last_bar_close_time": (now - timedelta(seconds=1)).isoformat(),
        "current_bar": {"open": 64000, "high": 64000, "low": 64000, "close": 64000},
        "bar_history": [],
        "current_regime": dict(_VB_INELIGIBLE_REGIME),
        "regime_change_count": 0,
        "strategy_metrics": {},
    }


# ---------------------------------------------------------------------------
# FIX 1 - entry eligibility must not force-close an open position
# ---------------------------------------------------------------------------

class TestForceExitSeparation:
    @pytest.mark.asyncio
    async def test_open_position_not_force_sold_when_strategy_becomes_ineligible(self):
        """Auto selects volatility_breakout, VB opens a position, the regime
        then changes so VB is no longer entry-eligible. Auto must NOT force
        sell - it must keep running VB so VB's own exit rules manage the
        position."""
        engine = _engine()
        bot = _auto_mode_bot(601)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        engine._save_auto_state(bot.id, _pinned_auto_state(now))
        engine._volatility_breakout_states = {bot.id: _vb_open_position_state()}

        fake_position = MagicMock()
        fake_position.amount = 0.01

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[fake_position])):
            signal = await engine._strategy_auto(bot, 64300.0, {}, session)

        assert signal is not None
        assert signal.action != "sell" or "FORCE EXIT" not in (signal.reason or ""), (
            f"Auto force-sold an open position on entry-ineligibility: {signal.reason}"
        )
        assert "FORCE EXIT" not in (signal.reason or "")
        assert "ineligible" not in (signal.reason or "").lower()

        state_after = engine._get_auto_state(bot.id)
        assert state_after["current_strategy"] == "volatility_breakout", (
            "Auto switched away from the strategy holding the open position instead "
            "of pinning it for exit management"
        )

    @pytest.mark.asyncio
    async def test_no_open_position_reallocates_without_force_sell(self):
        """When the current strategy becomes ineligible and there is NO open
        position, Auto is free to reallocate immediately - but it must do so
        by switching strategy, not by emitting a force-sell signal (there is
        nothing to sell)."""
        engine = _engine()
        bot = _auto_mode_bot(602)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        engine._save_auto_state(bot.id, _pinned_auto_state(now))

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            signal = await engine._strategy_auto(bot, 64300.0, {}, session)

        assert signal is not None
        assert "FORCE EXIT" not in (signal.reason or "")

        state_after = engine._get_auto_state(bot.id)
        assert state_after["current_strategy"] != "volatility_breakout", (
            "Auto should have reallocated away from the ineligible strategy "
            "since there was no position to protect"
        )


# ---------------------------------------------------------------------------
# FIX 2 - eligibility loss must not be recorded as a strategy failure
# ---------------------------------------------------------------------------

class TestEligibilityLossIsNotFailure:
    @pytest.mark.asyncio
    async def test_repeated_ineligibility_does_not_accumulate_failures(self):
        """A strategy can become ineligible on every tick (regime never
        favourable, position never closes) without ever accumulating
        failure_count, a cooldown, or moving toward blacklisting."""
        engine = _engine()
        bot = _auto_mode_bot(603)
        session = _session_with_empty_query_results()
        now = datetime.utcnow()

        fake_position = MagicMock()
        fake_position.amount = 0.01

        for _ in range(5):
            auto_state = engine._get_auto_state(bot.id) or {
                "current_strategy": "volatility_breakout",
                "last_switch_time": (now - timedelta(hours=1)).isoformat(),
                "strategy_metrics": {},
            }
            auto_state["last_bar_close_time"] = (now - timedelta(seconds=1)).isoformat()
            auto_state["current_bar"] = {"open": 64000, "high": 64000, "low": 64000, "close": 64000}
            auto_state["bar_history"] = auto_state.get("bar_history", [])
            auto_state["current_regime"] = dict(_VB_INELIGIBLE_REGIME)
            auto_state["regime_change_count"] = 0
            engine._save_auto_state(bot.id, auto_state)
            engine._volatility_breakout_states = {bot.id: _vb_open_position_state()}

            with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[fake_position])):
                await engine._strategy_auto(bot, 64300.0, {}, session)

            state_after = engine._get_auto_state(bot.id)
            vb_metrics = state_after.get("strategy_metrics", {}).get("volatility_breakout", {})
            assert vb_metrics.get("failure_count", 0) == 0
            assert vb_metrics.get("cooldown_until") is None

        final_metrics = engine._get_auto_state(bot.id).get("strategy_metrics", {}).get(
            "volatility_breakout", {}
        )
        assert final_metrics.get("failure_count", 0) == 0
        assert final_metrics.get("cooldown_until") is None


# ---------------------------------------------------------------------------
# FIX 3 - trend regime horizon: fast + medium slope
# ---------------------------------------------------------------------------

class TestTrendRegimeHorizon:
    def test_slow_sustained_decline_classifies_trend_down(self):
        """A grind that never moves more than ~0.5% over any 5-bar window
        (so the OLD fast-only classifier would call it 'flat') must still be
        recognised as trend_down once it has persisted over the medium
        window."""
        engine = _engine()
        start = 100.0
        # -0.05%/bar -> ~-3% cumulative over 60 bars; each 5-bar slice is
        # comfortably under the 0.5% fast threshold.
        prices = [start * (1 - 0.0005 * i) for i in range(60)]

        regime = engine._detect_market_regime(prices, None)
        assert regime["trend_state"] == "down", (
            f"Slow sustained decline was not classified trend_down: {regime}"
        )

    def test_slow_sustained_rise_classifies_trend_up(self):
        engine = _engine()
        start = 100.0
        prices = [start * (1 + 0.0005 * i) for i in range(60)]

        regime = engine._detect_market_regime(prices, None)
        assert regime["trend_state"] == "up", (
            f"Slow sustained rise was not classified trend_up: {regime}"
        )

    def test_sideways_chop_stays_trend_flat(self):
        """Oscillation that nets out near zero over the medium window must
        stay flat even though individual bars move up and down."""
        engine = _engine()
        prices = [100.0 + 0.3 * math.sin(i / 3.0) for i in range(60)]

        regime = engine._detect_market_regime(prices, None)
        assert regime["trend_state"] == "flat", (
            f"Sideways chop was misclassified: {regime}"
        )

    def test_fast_pump_classifies_trend_up(self):
        """A sharp move concentrated in the last few bars must still be
        caught immediately by the fast component (unchanged behaviour)."""
        engine = _engine()
        prices = [100.0] * 50 + [100.0 * (1 + 0.006 * i) for i in range(1, 11)]

        regime = engine._detect_market_regime(prices, None)
        assert regime["trend_state"] == "up", (
            f"Fast pump was not classified trend_up: {regime}"
        )

    def test_fast_dump_classifies_trend_down(self):
        engine = _engine()
        prices = [100.0] * 50 + [100.0 * (1 - 0.006 * i) for i in range(1, 11)]

        regime = engine._detect_market_regime(prices, None)
        assert regime["trend_state"] == "down", (
            f"Fast dump was not classified trend_down: {regime}"
        )

    def test_bar_based_variant_agrees_with_price_variant_on_slow_decline(self):
        """The Auto orchestrator's bar-based detector must catch the same
        slow, sustained decline as the price-only detector used by the
        individual strategies' own regime gates."""
        engine = _engine()
        start = 100.0
        closes = [start * (1 - 0.0005 * i) for i in range(60)]
        bars = [{"open": c, "high": c, "low": c, "close": c} for c in closes]

        regime = engine._detect_market_regime_bar_based(bars, None)
        assert regime["trend_state"] == "down", (
            f"Bar-based detector missed the slow sustained decline: {regime}"
        )


# ---------------------------------------------------------------------------
# FIX 4 - Auto capability table must match each strategy's own entry gate
# ---------------------------------------------------------------------------

class TestCapabilityAlignment:
    def test_volatility_breakout_matches_strategy_entry_regime(self):
        """_strategy_volatility_breakout only ever enters when
        allowed_regimes defaults to ["volatility_expanding"] (its own
        REGIME GATING section) - it never enters during compression, it only
        watches for one. The capability table must match, not declare it
        eligible for the opposite condition."""
        caps = _engine()._get_strategy_capabilities()
        assert caps["volatility_breakout"]["allowed_regimes"] == ["volatility_expanding"]

    def test_mean_reversion_matches_strategy_entry_regime(self):
        """_strategy_mean_reversion's own regime gate allows
        ["trend_flat", "volatility_high"] - the capability table used to be
        the stricter ["trend_flat"] only."""
        caps = _engine()._get_strategy_capabilities()
        assert caps["mean_reversion"]["allowed_regimes"] == ["trend_flat", "volatility_high"]

    def test_adaptive_grid_matches_strategy_entry_regime(self):
        """_strategy_grid's own regime gate only ever inspects trend_state /
        volatility_state tags (["trend_flat", "volatility_medium"]) - it
        never inspects volatility_direction, so a dangling
        "volatility_stable" entry in the capability table (a
        volatility_direction value) could never actually be acted on by the
        strategy itself."""
        caps = _engine()._get_strategy_capabilities()
        assert caps["adaptive_grid"]["allowed_regimes"] == ["trend_flat", "volatility_medium"]

    def test_funding_carry_still_matches_strategy_entry_regime(self):
        """Already correct before this fix - guard against regression."""
        caps = _engine()._get_strategy_capabilities()
        assert caps["funding_carry"]["allowed_regimes"] == ["trend_up", "trend_flat"]

    def test_capability_allowed_regimes_are_reachable_tags(self):
        """Every tag declared in allowed_regimes must be one _is_strategy_eligible
        can actually produce (trend_*, volatility_<state>, volatility_<direction>,
        liquidity_*, or the "all" sentinel) - guards against dead/unreachable
        capability entries like the old "volatility_stable" for adaptive_grid
        happening to be spelled right but never matchable."""
        reachable_trend = {"trend_up", "trend_down", "trend_flat"}
        reachable_vol_state = {"volatility_low", "volatility_medium", "volatility_high"}
        reachable_vol_direction = {
            "volatility_expanding", "volatility_contracting", "volatility_stable",
        }
        reachable_liquidity = {"liquidity_low", "liquidity_normal", "liquidity_high"}
        reachable = reachable_trend | reachable_vol_state | reachable_vol_direction | reachable_liquidity | {"all"}

        caps = _engine()._get_strategy_capabilities()
        for name, cap in caps.items():
            for tag in cap["allowed_regimes"]:
                assert tag in reachable, f"{name} declares unreachable tag {tag!r}"
