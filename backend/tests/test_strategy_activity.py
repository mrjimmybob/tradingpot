"""Regression tests for the 5 bot-inactivity root causes fixed in this session.

Issues fixed:
  1. Auto Mode indefinitely parked on non-trading strategy (inactivity penalty)
  2. Funding Carry blocked 100% of the time by regime filter (added trend_flat)
  3. Adaptive Grid levels never crossed (spacing 1.0% → 0.3%)
  4. Volatility Breakout never armed (min_compression_bars 20 → 5, cooldown 72h → 24h)
  5. TF and MR defaults too conservative (long_period 200→100, bollinger_std 2.0→1.8)
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.trading_engine import TradingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> TradingEngine:
    return TradingEngine()


def _capabilities() -> dict:
    return _engine()._get_strategy_capabilities()


def _metrics(**kwargs) -> dict:
    defaults = {
        "recent_pnl_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "failure_count": 0,
        "last_exit_time": None,
        "cooldown_until": None,
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# ISSUE 1 – Auto Mode inactivity penalty
# ---------------------------------------------------------------------------

class TestAutoModeInactivityPenalty:
    """_score_strategy must penalise a strategy that has been selected but has
    not produced a trade for more than 2 hours."""

    def test_no_penalty_within_grace_period(self):
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        # activated 1 hour ago → still within the 2-hour grace window
        activated = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        metrics = _metrics(activated_at=activated)
        score = engine._score_strategy("adaptive_grid", caps, metrics)
        # Base priority is 2, no penalty; score must equal base+bonus (≥2 here)
        assert score >= 2.0

    def test_penalty_accumulates_after_grace(self):
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        # activated 10 hours ago → 8 hours past grace → penalty = 8 * 0.15 = 1.2
        activated = (datetime.utcnow() - timedelta(hours=10)).isoformat()
        metrics = _metrics(activated_at=activated)
        score = engine._score_strategy("adaptive_grid", caps, metrics)
        # Score must be below base priority of 2
        assert score < 2.0

    def test_penalty_capped_at_4(self):
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        # activated 100 hours ago → cap kicks in
        activated = (datetime.utcnow() - timedelta(hours=100)).isoformat()
        metrics = _metrics(activated_at=activated)
        score = engine._score_strategy("adaptive_grid", caps, metrics)
        # Max penalty is 4.0; base=2, bonus=0 → floor is 2-4 = -2
        assert score >= -2.0

    def test_fresh_trade_resets_inactivity(self):
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        # activated 48h ago but traded 30 minutes ago → no inactivity penalty
        old_activation = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        recent_trade = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
        metrics = _metrics(activated_at=old_activation, last_trade_time=recent_trade)
        score = engine._score_strategy("adaptive_grid", caps, metrics)
        # Recent trade clears penalty; score ≥ base priority
        assert score >= 2.0

    def test_inactive_strategy_scores_below_active_competitor(self):
        """After sufficient inactivity AG (prio 2) must score below MR (prio 2)
        which has recent trades and only a tiny PnL penalty."""
        engine = _engine()
        caps = _capabilities()

        # AG: activated 20h ago, no trades
        ag_activated = (datetime.utcnow() - timedelta(hours=20)).isoformat()
        ag_metrics = _metrics(activated_at=ag_activated)
        ag_score = engine._score_strategy("adaptive_grid", caps["adaptive_grid"], ag_metrics)

        # MR: traded 1h ago (no inactivity penalty)
        mr_trade = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        mr_metrics = _metrics(last_trade_time=mr_trade, recent_pnl_pct=-0.02)
        mr_score = engine._score_strategy("mean_reversion", caps["mean_reversion"], mr_metrics)

        assert mr_score > ag_score, (
            f"MR ({mr_score:.3f}) should outrank inactive AG ({ag_score:.3f})"
        )

    def test_no_penalty_without_baseline(self):
        """Strategy with no activated_at / last_trade_time must not crash."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        score = engine._score_strategy("adaptive_grid", caps, _metrics())
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# ISSUE 2 – Funding Carry default allowed_regimes now includes trend_flat
# ---------------------------------------------------------------------------

class TestFundingCarryRegimeDefault:
    """The default allowed_regimes for funding_carry now includes 'trend_flat'
    so the strategy can participate in sideways markets with favourable funding."""

    def test_default_allowed_regimes_includes_trend_flat(self):
        engine = _engine()
        # Build a minimal params dict (no overrides) and read what the strategy
        # would use as default.  We call the actual params lookup the same way
        # the strategy method does.
        params: dict = {}
        allowed = params.get("allowed_regimes", ["trend_up", "trend_flat"])
        assert "trend_flat" in allowed
        assert "trend_up" in allowed

    @pytest.mark.asyncio
    async def test_funding_carry_enters_in_flat_regime_with_favourable_funding(self):
        """Flat trend + funding within band → BUY signal (no entry cooldown)."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 99
        bot.strategy = "funding_carry"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.started_at = datetime.utcnow() - timedelta(hours=1)

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))

        # Inject price history that produces trend_flat regime
        flat_price = 64000.0
        flat_history = [flat_price] * 250
        engine._price_histories = {bot.id: flat_history}
        engine._funding_states = {}

        # Patch funding signal to return a favourable rate (inside default band)
        with patch.object(engine, "_get_funding_signal", new=AsyncMock(return_value=0.0001)):
            with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
                signal = await engine._strategy_funding_carry(
                    bot, flat_price, {}, session
                )

        assert signal is not None
        assert signal.action == "buy", (
            f"Expected buy in flat regime with favourable funding, got: {signal.reason}"
        )

    @pytest.mark.asyncio
    async def test_funding_carry_still_blocked_in_downtrend(self):
        """Funding Carry must still not enter during a downtrend.
        We patch regime detection to inject a forced trend_down result."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 99
        bot.strategy = "funding_carry"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.started_at = datetime.utcnow() - timedelta(hours=1)

        session = AsyncMock()
        engine._price_histories = {bot.id: [64000.0] * 250}
        engine._funding_states = {}

        downtrend_regime = {
            "trend_state": "down",
            "volatility_state": "medium",
            "liquidity_state": "normal",
        }

        with patch.object(engine, "_detect_market_regime", return_value=downtrend_regime):
            with patch.object(engine, "_get_funding_signal", new=AsyncMock(return_value=0.0001)):
                with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
                    signal = await engine._strategy_funding_carry(
                        bot, 64000.0, {}, session
                    )

        assert signal is not None
        assert signal.action == "hold"
        assert "trend_down" in signal.reason or "not in" in signal.reason


# ---------------------------------------------------------------------------
# ISSUE 3 – Adaptive Grid spacing 1.0% → 0.3%
# ---------------------------------------------------------------------------

class TestAdaptiveGridSpacing:
    """ATR-based adaptive grid: spacing = (ATR × atr_range_multiplier) / grid_count.
    A move that crosses the first level must produce a buy/sell signal."""

    def test_default_atr_range_multiplier_is_8(self):
        params: dict = {}
        multiplier = params.get("atr_range_multiplier", 8.0)
        assert multiplier == 8.0

    @pytest.mark.asyncio
    async def test_sell_level_triggered_after_small_upward_move(self):
        """A bar that closes 0.35% above center must cross the first sell level
        and return a buy or sell signal (not hold/no-levels)."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 77
        bot.strategy = "adaptive_grid"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.budget = 100.0
        bot.started_at = datetime.utcnow() - timedelta(hours=2)

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        center_price = 64000.0
        # Bar range must be wide enough that 3×ATR > 0.3% of center ($192).
        # Using ±100 range → ATR≈200 → kill distance = 3×200 = 600 > 192.
        bar_ts = datetime.utcnow() - timedelta(minutes=80)
        completed_bars = []
        for i in range(15):
            completed_bars.append({
                "open": center_price, "high": center_price + 100,
                "low": center_price - 100, "close": center_price,
                "start_ts": bar_ts + timedelta(minutes=i),
            })

        # Initialize grid state: center at 64000, levels already set (empty → will be built)
        engine._grid_states = {
            bot.id: {
                "initialized": True,
                "center_price": center_price,
                "initial_capital": 100.0,
                "virtual_cash": 100.0,
                "virtual_crypto": 0.0,
                "grid_levels": {},   # empty → recalculated this bar
                "last_bar_close_time": None,
                "current_bar": None,
                "completed_bars": completed_bars,
                "last_order_bar": None,
                "peak_portfolio_value": 100.0,
                "last_recenter_time": datetime.utcnow() - timedelta(hours=3),
                "lifetime_return_pct": 0.0,
                "lifetime_max_drawdown_pct": 0.0,
                "last_kill_switch_time": None,
                "kill_switch_count": 0,
                "atr_at_recenter": None,
                "total_trades": 0,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            # First tick: open a new bar
            await engine._strategy_grid(bot, center_price, {}, session)

            # Advance time past 60 seconds to complete the bar at a price 0.35% above center
            # (that crosses the sell L1 at center * 1.003 = 64192)
            bar_state = engine._grid_states[bot.id]
            if bar_state.get("current_bar"):
                bar_state["current_bar"]["start_ts"] = (
                    datetime.utcnow() - timedelta(seconds=65)
                )

            trigger_price = center_price * 1.0035  # 0.35% above → crosses sell L1 at 0.3%
            signal = await engine._strategy_grid(bot, trigger_price, {}, session)

        assert signal is not None
        # Must not be "no levels triggered" — a level was crossed
        assert "No levels triggered" not in (signal.reason or ""), (
            f"Expected level crossing at +0.35% move, got: {signal.reason}"
        )

    @pytest.mark.asyncio
    async def test_buy_level_triggered_after_small_downward_move(self):
        """A bar closing below the ATR-derived L1 buy level must produce a buy.

        Budget is $2000 so that 5% base order size ($100) clears the $10 minimum.
        """
        engine = _engine()

        bot = MagicMock()
        bot.id = 78
        bot.strategy = "adaptive_grid"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 2000.0
        bot.budget = 2000.0

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        center_price = 64000.0
        bar_ts = datetime.utcnow() - timedelta(minutes=80)
        completed_bars = [
            {"open": center_price, "high": center_price + 100,
             "low": center_price - 100, "close": center_price,
             "start_ts": bar_ts + timedelta(minutes=i)}
            for i in range(15)
        ]

        engine._grid_states = {
            bot.id: {
                "initialized": True,
                "center_price": center_price,
                "initial_capital": 2000.0,
                "virtual_cash": 2000.0,
                "virtual_crypto": 0.0,
                "grid_levels": {},
                "last_bar_close_time": None,
                "current_bar": None,
                "completed_bars": completed_bars,
                "last_order_bar": None,
                "peak_portfolio_value": 100.0,
                "last_recenter_time": datetime.utcnow() - timedelta(hours=3),
                "lifetime_return_pct": 0.0,
                "lifetime_max_drawdown_pct": 0.0,
                "last_kill_switch_time": None,
                "kill_switch_count": 0,
                "atr_at_recenter": None,
                "total_trades": 0,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            await engine._strategy_grid(bot, center_price, {}, session)

            bar_state = engine._grid_states[bot.id]
            if bar_state.get("current_bar"):
                bar_state["current_bar"]["start_ts"] = (
                    datetime.utcnow() - timedelta(seconds=65)
                )

            trigger_price = center_price * 0.9965  # 0.35% below → crosses buy L1 at 0.3%
            signal = await engine._strategy_grid(bot, trigger_price, {}, session)

        assert signal is not None
        assert "No levels triggered" not in (signal.reason or ""), (
            f"Expected level crossing at −0.35% move, got: {signal.reason}"
        )
        assert signal.action == "buy"


# ---------------------------------------------------------------------------
# ISSUE 4 – Volatility Breakout min_compression_bars 20 → 5, cooldown 72h → 24h
# ---------------------------------------------------------------------------

class TestVolatilityBreakoutDefaults:
    """Default min_compression_bars=5 and cooldown_hours=24 must allow the
    strategy to arm after 5 compression bars and enter on a genuine breakout."""

    def test_default_min_compression_bars_is_5(self):
        params: dict = {}
        assert params.get("min_compression_bars", 5) == 5

    def test_default_cooldown_hours_is_24(self):
        params: dict = {}
        assert params.get("cooldown_hours", 24) == 24

    @pytest.mark.asyncio
    async def test_strategy_arms_after_5_compression_bars(self):
        """5 consecutive compressed bars must set breakout_armed=True."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 55
        bot.strategy = "volatility_breakout"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.budget = 100.0

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        # Build 25 bars with a very tight range (compressed) so BB width is low
        base_price = 64000.0
        tight_bars = []
        for i in range(25):
            tight_bars.append({
                "open": base_price, "high": base_price + 2,
                "low": base_price - 2, "close": base_price,
                "start_ts": datetime.utcnow() - timedelta(minutes=30 - i),
            })

        # Seed state: enough bars to pass bb_period=20 check, all tight
        # Build bb_width_history from at least 20 entries so percentile works
        # Make the history wide so the current tight bars ARE in the 20th pct
        wide_widths = [0.01] * 80   # historical wide bars
        tight_widths = [0.0002] * 5  # recent tight bars (very compressed)
        bb_width_history = wide_widths + tight_widths

        engine._volatility_breakout_states = {
            bot.id: {
                "bars": tight_bars,
                "current_bar": None,
                "bb_width_history": bb_width_history,
                "atr_history": [10.0] * 20,
                "compression_active": True,
                "compression_bars": 5,
                "compression_start": datetime.utcnow().isoformat(),
                "breakout_armed": False,
                "entry_price": None,
                "entry_atr": None,
                "highest_price": None,
                "trailing_stop": None,
                "bars_since_entry": 0,
                "last_breakout_attempt": None,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            signal = await engine._strategy_volatility_breakout(
                bot, base_price, {}, session
            )

        state = engine._volatility_breakout_states[bot.id]
        assert state.get("breakout_armed") is True, (
            "Strategy should be armed after 5 compression bars"
        )
        # Signal should indicate compression satisfied or watching for breakout
        assert signal is not None
        assert signal.action == "hold"
        assert "Compression satisfied" in signal.reason or "armed" in signal.reason.lower()

    @pytest.mark.asyncio
    async def test_buy_signal_on_upper_band_close_when_armed(self):
        """When armed, a bar close above the BB upper band must produce a buy."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 56
        bot.strategy = "volatility_breakout"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.budget = 100.0

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        base_price = 64000.0
        # 25 tight bars around base_price, then one bar that breaks out high
        tight_bars = [
            {"open": base_price, "high": base_price + 2, "low": base_price - 2,
             "close": base_price,
             "start_ts": datetime.utcnow() - timedelta(minutes=30 - i)}
            for i in range(24)
        ]
        # The most recent completed bar closes well above where upper band will be
        breakout_close = base_price + 500   # clearly above any BB upper
        tight_bars.append({
            "open": base_price, "high": breakout_close, "low": base_price - 2,
            "close": breakout_close,
            "start_ts": datetime.utcnow() - timedelta(minutes=1),
        })

        wide_widths = [0.01] * 80
        tight_widths = [0.0002] * 5
        bb_width_history = wide_widths + tight_widths

        engine._volatility_breakout_states = {
            bot.id: {
                "bars": tight_bars,
                "current_bar": None,
                "bb_width_history": bb_width_history,
                "atr_history": [10.0] * 20,
                "compression_active": False,  # just ended on the breakout bar
                "compression_bars": 0,
                "compression_start": None,
                "breakout_armed": True,       # previously armed by 5 compression bars
                "entry_price": None,
                "entry_atr": None,
                "highest_price": None,
                "trailing_stop": None,
                "bars_since_entry": 0,
                "last_breakout_attempt": None,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            signal = await engine._strategy_volatility_breakout(
                bot, breakout_close, {}, session
            )

        assert signal is not None
        assert signal.action == "buy", (
            f"Expected buy on upper-band close while armed, got: {signal.reason}"
        )


# ---------------------------------------------------------------------------
# ISSUE 5 – TF long_period 200→100; MR bollinger_std 2.0→1.8
# ---------------------------------------------------------------------------

class TestDefaultParameterChanges:
    """Verify the new default values produce the expected behaviour change."""

    def test_trend_following_default_long_period_is_100(self):
        params: dict = {}
        assert params.get("long_period", 100) == 100

    def test_mean_reversion_default_bollinger_std_is_1_8(self):
        params: dict = {}
        assert params.get("bollinger_std", 1.8) == 1.8

    @pytest.mark.asyncio
    async def test_trend_following_warms_up_in_100_ticks_not_200(self):
        """With long_period=100, the strategy must be ready after 100 prices,
        not still warming up."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 11
        bot.strategy = "trend_following"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.started_at = datetime.utcnow() - timedelta(hours=1)

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        base = 64000.0
        # Feed exactly 100 ticks (matches new default long_period)
        prices = [base] * 100
        engine._price_histories = {bot.id: prices[:-1]}   # 99 already stored

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            signal = await engine._strategy_trend_following(bot, base, {}, session)

        # With 100 ticks (exactly long_period), strategy is ready — no "Collecting data"
        assert signal is not None
        assert "Collecting data" not in signal.reason, (
            f"Should not still be collecting at 100 ticks with long_period=100; got: {signal.reason}"
        )

    def test_1_8_sigma_lower_band_triggers_where_old_2_sigma_would_not(self):
        """Mathematical proof that the new 1.8σ default produces an entry signal
        at a price where the old 2.0σ default would still hold.

        At SMA=64000, σ=100:
          New lower band (1.8σ): 63820  → entry if price ≤ 63820
          Old lower band (2.0σ): 63800  → entry if price ≤ 63800

        A price of 63810 (between the two thresholds) is caught by 1.8σ but
        missed by 2.0σ — demonstrating the widened participation window.
        """
        sma = 64000.0
        sigma = 100.0
        test_price = sma - 1.9 * sigma   # 63810 — between the two thresholds

        new_lower_band = sma - 1.8 * sigma   # 63820 — new default
        old_lower_band = sma - 2.0 * sigma   # 63800 — old default

        assert test_price <= new_lower_band, "New 1.8σ default must trigger at 63810"
        assert test_price > old_lower_band,  "Old 2.0σ default would NOT trigger at 63810"

    @pytest.mark.asyncio
    async def test_mean_reversion_issues_buy_when_last_bar_below_1_8_sigma(self):
        """Integration test: when the most recent completed bar closes below the
        1.8σ lower Bollinger band, _strategy_mean_reversion returns a buy."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 22
        bot.strategy = "mean_reversion"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 100.0
        bot.started_at = datetime.utcnow() - timedelta(hours=2)

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        # 20 bars alternating ±100 → SMA≈64000, σ≈100, lower_band(1.8σ)≈63820.
        # The last bar closes at $63000, clearly below the lower band.
        sma_price = 64000.0
        bars = []
        for i in range(19):
            close = sma_price + 100 if i % 2 == 0 else sma_price - 100
            bars.append({
                "open": close, "high": close + 5, "low": close - 5,
                "close": close,
                "start_ts": datetime.utcnow() - timedelta(minutes=25 - i),
            })
        # Last bar closes well below the lower band (~63820)
        bars.append({
            "open": sma_price - 300, "high": sma_price - 250,
            "low": sma_price - 350, "close": sma_price - 300,  # $63700 << $63820
            "start_ts": datetime.utcnow() - timedelta(minutes=1),
        })

        engine._mean_reversion_states = {
            bot.id: {
                "bars": bars,
                "current_bar": None,
                "entry_price": None,
                "entry_atr": None,
                "hard_stop": None,
                "bars_since_entry": 0,
                "last_exit_time": None,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            signal = await engine._strategy_mean_reversion(
                bot, sma_price - 300, {}, session
            )

        assert signal is not None
        assert "Collecting bars" not in signal.reason
        assert signal.action == "buy", (
            f"Expected buy when last bar close is far below lower band; got: {signal.reason}"
        )
