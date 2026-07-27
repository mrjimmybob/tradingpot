"""Regression tests for the Dip Recovery / Reversal Momentum strategy.

Covers the required scenarios end to end:
  1. Continuous decline never buys.
  2. Decline + confirmed recovery buys.
  3. Normal volatility noise never opens a false setup.
  4. High volatility expands the adaptive drop threshold.
  5. Low volatility contracts the adaptive drop threshold.
  6. Trailing exit activates and ratchets monotonically.
  7. An unresolved setup expires back to IDLE.
  8. Restart mid-TRACKING_DROP restores state via the generic persistence path.
  9. Restart mid-LONG_OPEN restores state via the generic persistence path.
  10. Auto Mode can select dip_recovery (capabilities + opportunity scoring).
  11. The explanation carries the exact calculated values.
  12. Existing strategies/dispatch are unchanged.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models import Bot, BotStatus
from app.services.trading_engine import (
    TradingEngine,
    _DipRecoveryState,
    validate_dip_recovery_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> TradingEngine:
    return TradingEngine()


def _bot(balance: float = 1000.0, bot_id: int = 1) -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "dip_recovery"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    return bot


def _no_positions(engine: TradingEngine) -> None:
    engine._get_bot_positions = AsyncMock(return_value=[])


# ATR is measured over `bar_interval_seconds` bars (fix-dip-recovery-bar-atr), so
# a test that drives the strategy in a tight loop under the real clock would
# never close a bar and would sit on the fee-coverage floor forever. Closing a
# bar on every call makes each bar's range the price change since the previous
# call - i.e. exactly the tick-to-tick semantics these tests were written
# against - so every existing assertion keeps its original meaning. This is the
# harness convention trend_following already documents ("some test harnesses set
# bar_interval_seconds=0").
_TICK_PARAMS = {"bar_interval_seconds": 0}


def _with_position(engine: TradingEngine, pos: MagicMock) -> None:
    engine._get_bot_positions = AsyncMock(return_value=[pos])


async def _feed(engine, bot, prices, params=None, session=None):
    """Drive the strategy across a sequence of prices, returning all signals."""
    session = session or AsyncMock()
    params = {**_TICK_PARAMS, **(params or {})}
    signals = []
    for p in prices:
        sig = await engine._strategy_dip_recovery(bot, p, params, session)
        signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# 1. Continuous decline never buys
# ---------------------------------------------------------------------------

class TestContinuousDeclineNeverBuys:
    @pytest.mark.asyncio
    async def test_no_buy_on_continuous_decline(self):
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        # Warm up flat, then decline monotonically for a long time - never a
        # single tick of recovery, so a BUY must never be produced.
        warmup = [100.0] * 20
        decline = [100.0 - i * 0.5 for i in range(1, 60)]  # steadily falling
        signals = await _feed(engine, bot, warmup + decline)

        assert all(s.action != "buy" for s in signals), (
            "A continuously falling price must never produce a BUY"
        )
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] in (_DipRecoveryState.TRACKING_DROP, _DipRecoveryState.IDLE)


# ---------------------------------------------------------------------------
# 2. Decline followed by valid recovery buys
# ---------------------------------------------------------------------------

class TestDeclineThenRecoveryBuys:
    @pytest.mark.asyncio
    async def test_buy_generated_after_confirmed_reversal(self):
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        warmup = [100.0, 100.05, 99.98, 100.02, 100.1, 99.95, 100.0, 100.03,
                  99.99, 100.01, 100.02, 99.98, 100.0, 100.01, 99.99]
        decline = [98.0, 96.0, 94.0, 92.0, 90.0, 89.5, 89.0]
        bounce = [89.3, 89.6, 90.0, 90.4, 90.8, 91.2, 91.6, 92.0]

        # Stop feeding once a BUY fires: in production a position now exists,
        # so continuing to drive ticks through a "no position" mock would
        # itself be an unrealistic setup (and the strategy defensively resets
        # to IDLE if it ever sees LONG_OPEN with no matching position).
        signals = []
        for p in warmup + decline + bounce:
            sig = await engine._strategy_dip_recovery(bot, p, _TICK_PARAMS, AsyncMock())
            signals.append(sig)
            if sig.action == "buy":
                break

        buys = [s for s in signals if s.action == "buy"]
        assert buys, "A confirmed reversal after a significant decline must produce a BUY"
        # Post Phase-3 migration the reason is mechanically derived from the
        # Decision Score's Evidence Items (Pillar 10 - never free-authored), so
        # the BUY carries the score/threshold instead of a hand-written phrase.
        assert buys[0].score is not None and buys[0].threshold is not None
        assert buys[0].score >= buys[0].threshold

        state = engine._dip_recovery_states[bot.id]
        assert state["state"] == _DipRecoveryState.LONG_OPEN
        assert state["entry_price"] is not None

    @pytest.mark.asyncio
    async def test_no_buy_while_still_below_recovery_threshold(self):
        """A small bounce that has not cleared the adaptive recovery threshold
        must stay in WAITING_REVERSAL, not buy."""
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        warmup = [100.0] * 15
        decline = [95.0, 90.0]  # sharp, clears drop threshold
        tiny_bounce = [90.01]  # negligible recovery, far below adaptive threshold

        signals = await _feed(engine, bot, warmup + decline + tiny_bounce)
        assert signals[-1].action == "hold"
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] in (_DipRecoveryState.WAITING_REVERSAL, _DipRecoveryState.TRACKING_DROP)


# ---------------------------------------------------------------------------
# 3. Normal volatility movement: no false setup
# ---------------------------------------------------------------------------

class TestNormalVolatilityNoFalseSetup:
    @pytest.mark.asyncio
    async def test_small_noise_never_arms_tracking(self):
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        # Realistic small noise (well under the 1.5% floor and under any
        # ATR-scaled threshold derived from this same noise).
        prices = []
        base = 100.0
        for i in range(80):
            base += 0.05 if i % 2 == 0 else -0.05
            prices.append(base)

        signals = await _feed(engine, bot, prices)
        assert all(s.action == "hold" for s in signals)
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] == _DipRecoveryState.IDLE, (
            "Normal-volatility noise must never arm TRACKING_DROP"
        )


# ---------------------------------------------------------------------------
# 4 & 5. Adaptive thresholds expand under high volatility, contract under low
# ---------------------------------------------------------------------------

class TestAdaptiveThresholds:
    @staticmethod
    def _alternating(base: float, amplitude: float, n: int) -> list:
        out = []
        for i in range(n):
            out.append(base + amplitude if i % 2 == 0 else base - amplitude)
        return out

    @pytest.mark.asyncio
    async def test_high_volatility_expands_drop_threshold(self):
        """With large tick-to-tick noise, a 2% single-tick decline must NOT be
        enough to arm TRACKING_DROP - the ATR-scaled threshold has expanded
        well past 2%."""
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        # amplitude=3 around 100 -> ATR proxy ~= 6 -> ATR% ~= 6% -> threshold
        # = max(1.5, 6% * 2.5) = 15%, far above the 2% move below.
        warmup = self._alternating(100.0, 3.0, 20)
        signals = await _feed(engine, bot, warmup)
        atr = engine._calc_price_atr_proxy(engine._get_price_history(bot.id), 14)
        assert atr / 100.0 * 100.0 > 2.0, "test setup must produce ATR% well above 2%"

        decline_tick = await engine._strategy_dip_recovery(bot, 98.0, _TICK_PARAMS, AsyncMock())
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] == _DipRecoveryState.IDLE, (
            "A 2% move must NOT arm tracking when volatility-scaled threshold is much larger"
        )
        assert decline_tick.action == "hold"

    @pytest.mark.asyncio
    async def test_low_volatility_contracts_drop_threshold(self):
        """With tiny tick-to-tick noise, the adaptive threshold collapses to
        the configured floor, so a modest 2% decline IS enough to arm
        TRACKING_DROP."""
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        # amplitude=0.02 around 100 -> ATR proxy tiny -> ATR%-driven term is
        # negligible -> threshold floors at min_drop_percent (default 1.5%).
        warmup = self._alternating(100.0, 0.02, 20)
        await _feed(engine, bot, warmup)
        atr = engine._calc_price_atr_proxy(engine._get_price_history(bot.id), 14)
        assert (atr / 100.0 * 100.0) * 2.5 < 1.5, "test setup must keep the ATR term below the floor"

        decline_tick = await engine._strategy_dip_recovery(bot, 98.0, _TICK_PARAMS, AsyncMock())
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] == _DipRecoveryState.TRACKING_DROP, (
            "A 2% decline must arm tracking once the floor (not an inflated ATR term) governs"
        )
        assert decline_tick.action == "hold"


# ---------------------------------------------------------------------------
# 6. Trailing exit activates correctly
# ---------------------------------------------------------------------------

class TestTrailingExit:
    @pytest.mark.asyncio
    async def test_trailing_stop_ratchets_and_exits_on_pullback(self):
        engine = _engine()
        bot = _bot()
        entry_price = 100.0
        atr = 1.0
        engine._price_histories = {bot.id: [entry_price] * 20}
        engine._dip_recovery_states = {
            bot.id: {
                **engine._dip_recovery_default_state(),
                "state": _DipRecoveryState.LONG_OPEN,
                "entry_price": entry_price,
                "entry_time": datetime.utcnow(),
                "entry_atr": atr,
                "highest_price_since_entry": entry_price,
                "trailing_stop": entry_price - atr * 1.5,
                "take_profit": entry_price + atr * 10.0,  # keep TP out of reach
                "emergency_stop": entry_price - atr * 20.0,  # keep emergency out of reach
            }
        }
        pos = MagicMock(amount=1.0, entry_price=entry_price)
        _with_position(engine, pos)

        seen_stops = []
        for p in [101.0, 102.0, 103.0]:
            sig = await engine._strategy_dip_recovery(bot, p, _TICK_PARAMS, AsyncMock())
            assert sig.action == "hold"
            seen_stops.append(engine._dip_recovery_states[bot.id]["trailing_stop"])

        # Monotonic tightening: each new high pushes the stop strictly upward.
        assert seen_stops == sorted(seen_stops)
        assert seen_stops[-1] == pytest.approx(103.0 - atr * 1.5)

        # A pullback through the ratcheted stop (not the original one) exits.
        sig = await engine._strategy_dip_recovery(bot, 101.4, _TICK_PARAMS, AsyncMock())
        assert sig.action == "sell"
        assert "trailing stop" in sig.reason.lower()
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] == _DipRecoveryState.COOLDOWN
        assert state["last_exit_was_loss"] is False  # 101.4 > 100 entry = win


# ---------------------------------------------------------------------------
# 7. Expired recovery setup resets
# ---------------------------------------------------------------------------

class TestSetupExpiry:
    @pytest.mark.asyncio
    async def test_stale_tracking_drop_resets_to_idle(self):
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        engine._price_histories = {bot.id: [100.0] * 20}
        engine._dip_recovery_states = {
            bot.id: {
                **engine._dip_recovery_default_state(),
                # Setup detection is bar-paced, so a warm bot needs bars, not
                # just ticks (fix-dip-recovery-setup-cadence).
                "dr_bars": [{"high": 100.0, "low": 100.0, "close": 100.0}] * 20,
                "state": _DipRecoveryState.TRACKING_DROP,
                "reference_high": 100.0,
                "reference_high_time": datetime.utcnow() - timedelta(minutes=500),
                "lowest_price": 95.0,
                "lowest_price_time": datetime.utcnow() - timedelta(minutes=300),
                "tracking_started_at": datetime.utcnow() - timedelta(minutes=500),
                "ticks_since_new_low": 5,
            }
        }

        # setup_expiry_minutes default is 240; 500 minutes elapsed must expire it.
        sig = await engine._strategy_dip_recovery(bot, 96.0, _TICK_PARAMS, AsyncMock())
        assert sig.action == "hold"
        assert "expired" in sig.reason.lower()
        state = engine._dip_recovery_states[bot.id]
        assert state["state"] == _DipRecoveryState.IDLE
        assert state["reference_high"] is None
        assert state["lowest_price"] is None


# ---------------------------------------------------------------------------
# 8 & 9. Restart restores state (TRACKING_DROP / LONG_OPEN)
# ---------------------------------------------------------------------------

class TestRestartRestoresState:
    @pytest.mark.asyncio
    async def test_restart_during_tracking_drop_restores_state(self, test_db):
        bot = Bot(
            name="dip-tracking", trading_pair="BTC/USDT", strategy="dip_recovery",
            strategy_params={}, budget=1000.0, current_balance=1000.0,
            is_dry_run=True, status=BotStatus.RUNNING,
        )
        test_db.add(bot)
        await test_db.flush()
        bot_id = bot.id

        engine = TradingEngine()
        entered_at = datetime(2026, 6, 20, 10, 0, 0)
        engine._dip_recovery_states = {
            bot_id: {
                **engine._dip_recovery_default_state(),
                "state": _DipRecoveryState.TRACKING_DROP,
                "reference_high": 105.0,
                "reference_high_time": entered_at,
                "lowest_price": 98.5,
                "lowest_price_time": entered_at + timedelta(minutes=5),
                "tracking_started_at": entered_at,
                "ticks_since_new_low": 3,
            }
        }

        await engine._save_bot_state(bot_id, test_db)
        await test_db.commit()

        # Simulate a full process restart: a fresh engine with empty state.
        engine2 = TradingEngine()
        refreshed = await test_db.get(Bot, bot_id)
        await engine2._restore_strategy_state(refreshed)

        restored = engine2._dip_recovery_states[bot_id]
        assert restored["state"] == _DipRecoveryState.TRACKING_DROP
        assert restored["reference_high"] == 105.0
        assert restored["lowest_price"] == 98.5
        assert restored["ticks_since_new_low"] == 3
        assert restored["tracking_started_at"] == entered_at  # datetime fidelity preserved

    @pytest.mark.asyncio
    async def test_restart_during_long_open_restores_state(self, test_db):
        bot = Bot(
            name="dip-long-open", trading_pair="BTC/USDT", strategy="dip_recovery",
            strategy_params={}, budget=1000.0, current_balance=1000.0,
            is_dry_run=True, status=BotStatus.RUNNING,
        )
        test_db.add(bot)
        await test_db.flush()
        bot_id = bot.id

        engine = TradingEngine()
        entry_time = datetime.utcnow() - timedelta(minutes=30)
        engine._dip_recovery_states = {
            bot_id: {
                **engine._dip_recovery_default_state(),
                "state": _DipRecoveryState.LONG_OPEN,
                "entry_price": 90.4,
                "entry_time": entry_time,
                "entry_atr": 0.9,
                "highest_price_since_entry": 92.0,
                "trailing_stop": 90.65,
                "take_profit": 93.1,
                "emergency_stop": 85.9,
            }
        }

        await engine._save_bot_state(bot_id, test_db)
        await test_db.commit()

        engine2 = TradingEngine()
        refreshed = await test_db.get(Bot, bot_id)
        await engine2._restore_strategy_state(refreshed)

        restored = engine2._dip_recovery_states[bot_id]
        assert restored["state"] == _DipRecoveryState.LONG_OPEN
        assert restored["entry_price"] == 90.4
        assert restored["entry_time"] == entry_time
        assert restored["highest_price_since_entry"] == 92.0
        assert restored["trailing_stop"] == 90.65

        # And the restored bot continues managing the SAME position (counters
        # continue rather than resetting) once the loop resumes.
        engine2._price_histories = {bot_id: [92.0] * 20}  # normally rebuilt by ticks after resume
        pos = MagicMock(amount=1.0, entry_price=90.4)
        _with_position(engine2, pos)
        sig = await engine2._strategy_dip_recovery(bot, 92.5, _TICK_PARAMS, AsyncMock())
        assert sig.action == "hold"
        assert engine2._dip_recovery_states[bot_id]["highest_price_since_entry"] == 92.5


# ---------------------------------------------------------------------------
# 10. Auto Mode can select Dip Recovery
# ---------------------------------------------------------------------------

class TestAutoModeIntegration:
    def test_capabilities_include_dip_recovery(self):
        engine = _engine()
        caps = engine._get_strategy_capabilities()
        assert "dip_recovery" in caps
        assert "trend_down" in caps["dip_recovery"]["allowed_regimes"]

    def test_eligible_in_downtrend_regime(self):
        engine = _engine()
        caps = engine._get_strategy_capabilities()["dip_recovery"]
        regime = {
            "trend_state": "down", "volatility_state": "medium",
            "volatility_direction": "stable", "liquidity_state": "normal",
        }
        eligible, reason = engine._is_strategy_eligible(
            "dip_recovery", caps, regime, {}, datetime.utcnow(), max_failures=3,
        )
        assert eligible, reason

    @staticmethod
    def _bars_decline_then_bounce() -> list:
        """Bars tracing a clear decline from a high, then an early bounce off
        the resulting low - the exact setup dip_recovery targets."""
        bars = []
        price = 100.0
        for _ in range(20):
            bars.append({"open": price, "high": price + 0.2, "low": price - 0.2, "close": price})
        for _ in range(10):
            price -= 0.8
            bars.append({"open": price + 0.8, "high": price + 0.8, "low": price - 0.1, "close": price})
        for _ in range(4):
            price += 0.3
            bars.append({"open": price - 0.3, "high": price + 0.1, "low": price - 0.3, "close": price})
        return bars

    def test_scores_high_for_a_forming_reversal(self):
        engine = _engine()
        bars = self._bars_decline_then_bounce()
        score = engine._compute_opportunity_score(
            "dip_recovery", bars, bars[-1]["close"],
            {"trend_state": "down", "volatility_direction": "expanding"},
        )
        assert score >= 6.0, f"Expected a high opportunity score for a forming reversal, got {score:.2f}"

    def test_scores_low_for_sideways_market(self):
        engine = _engine()
        bars = []
        price = 100.0
        for i in range(40):
            c = price + (0.1 if i % 2 == 0 else -0.1)
            bars.append({"open": price, "high": price + 0.1, "low": price - 0.1, "close": c})
        score = engine._compute_opportunity_score(
            "dip_recovery", bars, bars[-1]["close"],
            {"trend_state": "flat", "volatility_direction": "stable"},
        )
        assert score <= 4.0, f"Expected a low opportunity score for a sideways market, got {score:.2f}"

    def test_scores_low_while_still_actively_falling(self):
        """A decline in progress, with no bounce yet, should not outscore a
        confirmed-forming reversal - the strategy must not chase falling knives."""
        engine = _engine()
        falling_bars = []
        price = 100.0
        for _ in range(20):
            falling_bars.append({"open": price, "high": price + 0.2, "low": price - 0.2, "close": price})
        for _ in range(14):
            price -= 0.8
            falling_bars.append({"open": price + 0.8, "high": price + 0.8, "low": price - 0.1, "close": price})

        falling_score = engine._compute_opportunity_score(
            "dip_recovery", falling_bars, falling_bars[-1]["close"],
            {"trend_state": "down", "volatility_direction": "expanding"},
        )
        bouncing_score = engine._compute_opportunity_score(
            "dip_recovery", self._bars_decline_then_bounce(),
            self._bars_decline_then_bounce()[-1]["close"],
            {"trend_state": "down", "volatility_direction": "expanding"},
        )
        assert bouncing_score > falling_score, (
            f"A forming reversal ({bouncing_score:.2f}) should score higher than an "
            f"actively-falling market ({falling_score:.2f})"
        )

    def test_dip_recovery_executor_registered(self):
        engine = _engine()
        executor = engine._get_strategy_executor("dip_recovery")
        assert executor is not None
        assert executor.__func__ is TradingEngine._strategy_dip_recovery


# ---------------------------------------------------------------------------
# 11. Explanation contains exact calculated values
# ---------------------------------------------------------------------------

class TestExplanationExactValues:
    @pytest.mark.asyncio
    async def test_idle_explanation_reports_exact_drawdown_and_threshold(self):
        engine = _engine()
        bot = _bot()
        _no_positions(engine)

        # Flat warmup so ATR% is ~0 and the threshold floors at min_drop_percent.
        await _feed(engine, bot, [100.0] * 15)
        await engine._strategy_dip_recovery(bot, 99.0, _TICK_PARAMS, AsyncMock())

        exp = engine._explanations[bot.id].to_dict()
        assert exp["state"] == _DipRecoveryState.IDLE
        metrics = exp["metrics"]
        assert metrics["current_price"] == pytest.approx(99.0)
        assert metrics["reference_high"] == pytest.approx(100.0)
        assert metrics["drawdown_percent"] == pytest.approx(-1.0, abs=1e-6)
        assert metrics["drop_threshold_percent"] == pytest.approx(1.5, abs=1e-6)

        checks = {c["name"]: c for c in exp["checks"]}
        assert checks["Decline vs adaptive threshold"]["passed"] is False

    @pytest.mark.asyncio
    async def test_long_open_explanation_reports_exact_levels(self):
        engine = _engine()
        bot = _bot()
        entry_price = 100.0
        atr = 2.0
        engine._price_histories = {bot.id: [entry_price] * 20}
        engine._dip_recovery_states = {
            bot.id: {
                **engine._dip_recovery_default_state(),
                "state": _DipRecoveryState.LONG_OPEN,
                "entry_price": entry_price,
                "entry_time": datetime.utcnow(),
                "entry_atr": atr,
                "highest_price_since_entry": entry_price,
                "trailing_stop": entry_price - atr * 1.5,
                "take_profit": entry_price + atr * 3.0,
                "emergency_stop": entry_price - atr * 5.0,
            }
        }
        pos = MagicMock(amount=1.0, entry_price=entry_price)
        _with_position(engine, pos)

        await engine._strategy_dip_recovery(bot, 103.0, _TICK_PARAMS, AsyncMock())
        exp = engine._explanations[bot.id].to_dict()
        metrics = exp["metrics"]
        assert exp["state"] == _DipRecoveryState.LONG_OPEN
        assert metrics["entry_price"] == pytest.approx(100.0)
        assert metrics["highest_price"] == pytest.approx(103.0)
        assert metrics["trailing_stop"] == pytest.approx(103.0 - atr * 1.5)
        assert metrics["unrealized_pnl_percent"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 12. Existing strategies unchanged
# ---------------------------------------------------------------------------

class TestExistingStrategiesUnchanged:
    def test_original_strategies_still_dispatch_to_the_same_methods(self):
        engine = _engine()
        expected = {
            "dca_accumulator": TradingEngine._strategy_dca,
            "adaptive_grid": TradingEngine._strategy_grid,
            "mean_reversion": TradingEngine._strategy_mean_reversion,
            "trend_following": TradingEngine._strategy_trend_following,
            "volatility_breakout": TradingEngine._strategy_volatility_breakout,
            "auto_mode": TradingEngine._strategy_auto,
        }
        for name, unbound in expected.items():
            executor = engine._get_strategy_executor(name)
            assert executor.__func__ is unbound, f"{name} dispatch changed"

    def test_original_capabilities_untouched(self):
        engine = _engine()
        caps = engine._get_strategy_capabilities()
        # Pin the pre-existing entries exactly as they were before this change.
        assert caps["trend_following"] == {
            "allowed_regimes": ["trend_up", "volatility_expanding"],
            "priority": 4, "typical_holding_time": "long",
            "description": "Best for sustained uptrends with clear momentum",
        }
        assert caps["dca_accumulator"] == {
            "allowed_regimes": ["all"], "priority": 0, "typical_holding_time": "long",
            "description": "Safe default accumulator for all market conditions",
        }
        assert len(caps) == 6  # 5 remaining original + dip_recovery

    def test_persisted_state_attrs_still_include_all_originals(self):
        from app.services.trading_engine import _PERSISTED_STATE_ATTRS
        for original in (
            "_grid_states", "_mean_reversion_states", "_trend_states",
            "_volatility_breakout_states", "_twap_states",
            "_vwap_states", "_auto_states",
        ):
            assert original in _PERSISTED_STATE_ATTRS
        assert "_dip_recovery_states" in _PERSISTED_STATE_ATTRS


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestParamValidation:
    def test_defaults_are_valid(self):
        assert validate_dip_recovery_params({}) == []

    def test_emergency_stop_must_exceed_trailing_stop(self):
        errors = validate_dip_recovery_params({
            "trailing_stop_atr_multiplier": 3.0,
            "emergency_stop_atr_multiplier": 2.0,
        })
        assert errors

    def test_loss_cooldown_must_not_be_shorter_than_cooldown(self):
        errors = validate_dip_recovery_params({
            "cooldown_seconds": 600,
            "loss_cooldown_seconds": 100,
        })
        assert errors


# ---------------------------------------------------------------------------
# End-to-end API wiring: config.py STRATEGIES + bots.py validation
# ---------------------------------------------------------------------------

class TestApiIntegration:
    @pytest.mark.asyncio
    async def test_bot_creation_accepts_dip_recovery_via_api(self, client):
        bot_data = {
            "name": "Dip Recovery Bot",
            "trading_pair": "ETH/USDT",
            "strategy": "dip_recovery",
            "budget": 500.0,
            "is_dry_run": True,
        }
        response = await client.post("/api/bots", json=bot_data)
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["strategy"] == "dip_recovery"

    @pytest.mark.asyncio
    async def test_config_strategies_endpoint_lists_dip_recovery(self, client):
        response = await client.get("/api/config/strategies")
        assert response.status_code == 200
        names = [s["name"] for s in response.json()]
        assert "dip_recovery" in names
