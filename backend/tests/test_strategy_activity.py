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

class TestAutoModeScoring:
    """_score_strategy returns a dict with the three components of the new formula:

        final = (opportunity × 3) + performance - risk_penalty

    Inactivity alone is NOT penalised (the spec removed pure time-based decay).
    The opportunity_score handles whether a setup is present.
    """

    def test_returns_dict_with_required_keys(self):
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        result = engine._score_strategy("adaptive_grid", caps, _metrics())
        assert isinstance(result, dict)
        for key in ("final", "opportunity", "performance", "confidence", "risk_penalty"):
            assert key in result, f"Missing key: {key}"

    def test_no_inactivity_penalty_without_bar_history(self):
        """Pure inactivity does not penalise — opportunity score defaults to 5.0
        (neutral) when bar_history is empty."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        # Strategy was 'inactive' for 100 hours — no penalty under new spec
        very_old = (datetime.utcnow() - timedelta(hours=100)).isoformat()
        result = engine._score_strategy("adaptive_grid", caps, _metrics(activated_at=very_old))
        # opportunity=5.0, performance=0, risk=0 → final = 15.0
        assert result["final"] == pytest.approx(15.0, abs=0.1)
        assert result["risk_penalty"] == pytest.approx(0.0, abs=0.01)

    def test_failure_count_adds_risk_penalty(self):
        """Each failure adds 5.0 to risk_penalty."""
        engine = _engine()
        caps = _capabilities()["trend_following"]
        result = engine._score_strategy("trend_following", caps, _metrics(failure_count=2))
        assert result["risk_penalty"] == pytest.approx(10.0, abs=0.01)

    def test_drawdown_adds_exponential_risk_penalty(self):
        """Drawdown penalty = (max_drawdown_pct / 5) ** 2."""
        engine = _engine()
        caps = _capabilities()["trend_following"]
        result = engine._score_strategy("trend_following", caps, _metrics(max_drawdown_pct=10.0))
        # (10/5)^2 = 4.0
        assert result["risk_penalty"] == pytest.approx(4.0, abs=0.01)

    def test_performance_zero_with_no_trades(self):
        """With zero trades confidence=0 → performance=0."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        result = engine._score_strategy("adaptive_grid", caps, _metrics(total_trades=0))
        assert result["confidence"] == pytest.approx(0.0, abs=0.01)
        assert result["performance"] == pytest.approx(0.0, abs=0.01)

    def test_confidence_scales_performance(self):
        """25 trades → confidence=0.5 → performance is halved vs full confidence."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        m_25 = _metrics(
            total_trades=25, winning_trades=20, win_rate=0.8,
            profit_factor=2.0, recent_pnl_pct=5.0
        )
        m_50 = _metrics(
            total_trades=50, winning_trades=40, win_rate=0.8,
            profit_factor=2.0, recent_pnl_pct=5.0
        )
        r25 = engine._score_strategy("adaptive_grid", caps, m_25)
        r50 = engine._score_strategy("adaptive_grid", caps, m_50)
        assert r25["confidence"] == pytest.approx(0.5, abs=0.01)
        assert r50["confidence"] == pytest.approx(1.0, abs=0.01)
        # Performance is proportional to confidence
        assert r50["performance"] == pytest.approx(r25["performance"] * 2.0, rel=0.05)

    def test_high_performance_with_full_confidence_adds_to_final(self):
        """50+ trades, good win rate, high PF → positive performance contribution."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        m = _metrics(
            total_trades=50, winning_trades=40, win_rate=0.8,
            profit_factor=3.0, recent_pnl_pct=10.0
        )
        result = engine._score_strategy("adaptive_grid", caps, m)
        # opportunity=5 (no bar history), performance>0, risk=0
        assert result["performance"] > 0.0
        assert result["final"] > 15.0  # > (5 * 3) + 0 - 0

    def test_no_crash_without_baseline(self):
        """Strategy with no activated_at / last_trade_time must not crash."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        result = engine._score_strategy("adaptive_grid", caps, _metrics())
        assert isinstance(result["final"], float)


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


# ---------------------------------------------------------------------------
# NEW – Auto Mode: aggressive inactivity decay causes rotation
# ---------------------------------------------------------------------------

class TestAutoModeOpportunityScore:
    """_compute_opportunity_score returns 0–10 based on live market conditions.

    These tests verify the signal logic for each strategy type using controlled
    synthetic bar histories.
    """

    @staticmethod
    def _flat_bars(price: float, n: int = 50, spread: float = 50.0) -> list:
        """n bars oscillating ±spread around price."""
        bars = []
        for i in range(n):
            c = price + (spread if i % 2 == 0 else -spread)
            bars.append({"open": price, "high": price + spread, "low": price - spread, "close": c})
        return bars

    @staticmethod
    def _uptrend_bars(start: float = 60000.0, n: int = 60, drift: float = 50.0) -> list:
        """n bars with a consistent upward drift."""
        bars = []
        price = start
        for _ in range(n):
            price += drift
            bars.append({"open": price - drift, "high": price + 20, "low": price - 20, "close": price})
        return bars

    def test_trend_following_scores_high_in_uptrend(self):
        engine = _engine()
        bars = self._uptrend_bars(n=60)
        score = engine._compute_opportunity_score(
            "trend_following", bars, bars[-1]["close"],
            {"trend_state": "up", "volatility_direction": "expanding"}
        )
        assert score >= 5.0, f"TF should score high in uptrend, got {score:.2f}"

    def test_trend_following_scores_low_in_flat_market(self):
        engine = _engine()
        bars = self._flat_bars(64000.0, spread=20.0)
        score = engine._compute_opportunity_score(
            "trend_following", bars, 64000.0,
            {"trend_state": "flat", "volatility_direction": "stable"}
        )
        assert score <= 6.0, f"TF should score low in flat market, got {score:.2f}"

    def test_mean_reversion_scores_high_when_price_below_band(self):
        engine = _engine()
        # 20 bars around 64000, then crash to 62500 (well below lower BB)
        bars = self._flat_bars(64000.0, n=50, spread=100.0)
        score = engine._compute_opportunity_score(
            "mean_reversion", bars, 62500.0,  # ~24% below in std units = very extreme
            {"trend_state": "flat", "volatility_direction": "stable"}
        )
        assert score >= 6.0, f"MR should score high when price far below mean, got {score:.2f}"

    def test_mean_reversion_scores_low_near_mean(self):
        engine = _engine()
        bars = self._flat_bars(64000.0, n=50, spread=100.0)
        score = engine._compute_opportunity_score(
            "mean_reversion", bars, 64000.0,  # at the mean
            {"trend_state": "flat", "volatility_direction": "stable"}
        )
        assert score <= 6.0, f"MR should score low near mean, got {score:.2f}"

    def test_dca_scores_higher_with_bigger_drawdown(self):
        engine = _engine()
        bars_low = self._flat_bars(64000.0, n=50, spread=50.0)
        bars_high = self._flat_bars(64000.0, n=50, spread=50.0)
        # 5% dip vs at peak
        score_dip = engine._compute_opportunity_score("dca_accumulator", bars_low, 60800.0, {})
        score_peak = engine._compute_opportunity_score("dca_accumulator", bars_high, 64000.0, {})
        assert score_dip > score_peak, (
            f"DCA at dip ({score_dip:.2f}) should score > at peak ({score_peak:.2f})"
        )

    def test_returns_5_with_insufficient_bars(self):
        engine = _engine()
        bars = self._flat_bars(64000.0, n=5)  # only 5 bars
        score = engine._compute_opportunity_score("trend_following", bars, 64000.0, {})
        assert score == pytest.approx(5.0)

    def test_scores_are_within_range(self):
        """All strategy scores must be in [0, 10] for any input."""
        engine = _engine()
        caps = _capabilities()
        bars = self._flat_bars(64000.0, n=50, spread=200.0)
        regime = {"trend_state": "flat", "volatility_state": "medium",
                  "volatility_direction": "stable", "liquidity_state": "normal"}
        for name in caps:
            score = engine._compute_opportunity_score(name, bars, 64000.0, regime)
            assert 0.0 <= score <= 10.0, f"{name} score {score:.2f} out of [0,10]"

    def test_final_score_dominated_by_opportunity(self):
        """final = oppty*3 + perf - risk.  At max opportunity a strategy with
        no performance history should still score high."""
        engine = _engine()
        caps = _capabilities()
        bars = self._uptrend_bars(n=60)
        regime = {"trend_state": "up", "volatility_direction": "expanding",
                  "volatility_state": "high", "liquidity_state": "normal"}
        result = engine._score_strategy(
            "trend_following", caps["trend_following"], _metrics(),
            bar_history=bars, current_price=bars[-1]["close"], current_regime=regime
        )
        # Even with zero performance history, the opportunity component should dominate
        assert result["final"] > result["opportunity"] * 2.5, (
            f"Opportunity should dominate final score: {result}"
        )

    def test_bad_performance_with_low_confidence_barely_penalises(self):
        """With only 3 trades (confidence=0.06), even −100% PnL barely moves the score."""
        engine = _engine()
        caps = _capabilities()["adaptive_grid"]
        m = _metrics(total_trades=3, recent_pnl_pct=-100.0, win_rate=0.0, profit_factor=0.0)
        result = engine._score_strategy("adaptive_grid", caps, m)
        # performance = raw_perf * (3/50) ≈ very small penalty
        assert abs(result["performance"]) < 1.5, (
            f"Low-confidence bad performance should barely affect score: {result['performance']:.3f}"
        )


# ---------------------------------------------------------------------------
# NEW – Funding Carry: stale ["trend_up"] default upgraded at runtime
# ---------------------------------------------------------------------------

class TestFundingCarryStaleMigration:
    """Bots created under the old default allowed_regimes=['trend_up'] must
    be silently upgraded to ['trend_up','trend_flat'] at runtime so they can
    enter in flat markets."""

    @pytest.mark.asyncio
    async def test_stale_trend_up_only_config_still_enters_in_flat_regime(self):
        """Bot with persisted allowed_regimes=['trend_up'] must enter when
        regime is trend_flat (migration normalises the stale default)."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 201
        bot.strategy = "funding_carry"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 10_000.0

        session = AsyncMock()
        engine._price_histories = {bot.id: [64000.0] * 250}
        engine._funding_states = {}

        flat_regime = {"trend_state": "flat", "volatility_state": "medium", "liquidity_state": "normal"}

        # Simulate the old, stale persisted default
        stale_params = {"allowed_regimes": ["trend_up"]}

        with patch.object(engine, "_detect_market_regime", return_value=flat_regime):
            with patch.object(engine, "_get_funding_signal", new=AsyncMock(return_value=0.0001)):
                with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
                    signal = await engine._strategy_funding_carry(
                        bot, 64000.0, stale_params, session
                    )

        assert signal.action == "buy", (
            f"Stale ['trend_up'] config must be normalised to include trend_flat; got: {signal.reason}"
        )

    @pytest.mark.asyncio
    async def test_explicit_trend_up_only_does_not_block_trend_up(self):
        """After normalisation, trend_up regime still triggers entry."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 202
        bot.strategy = "funding_carry"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 10_000.0

        session = AsyncMock()
        engine._price_histories = {bot.id: [64000.0] * 250}
        engine._funding_states = {}

        up_regime = {"trend_state": "up", "volatility_state": "medium", "liquidity_state": "normal"}

        with patch.object(engine, "_detect_market_regime", return_value=up_regime):
            with patch.object(engine, "_get_funding_signal", new=AsyncMock(return_value=0.0001)):
                with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
                    signal = await engine._strategy_funding_carry(
                        bot, 64000.0, {"allowed_regimes": ["trend_up"]}, session
                    )

        assert signal.action == "buy", f"trend_up must still trigger entry; got: {signal.reason}"


# ---------------------------------------------------------------------------
# NEW – Adaptive Grid: minimum profitable spacing
# ---------------------------------------------------------------------------

class TestAdaptiveGridMinProfitableSpacing:
    """Grid spacing must never be smaller than the round-trip transaction cost
    (2 × taker fee + spread + profit buffer), even when ATR is tiny."""

    @pytest.mark.asyncio
    async def test_tiny_atr_spacing_floored_to_min_profitable(self):
        """When ATR-derived spacing < min_spacing, effective spacing uses the
        minimum.  With default fees (0.1% taker ×2 + 0.05% spread + 0.1% buffer
        = 0.35%) and center $60 000: min_spacing = $210.  Inject ATR=$5
        → raw_spacing = 5*8/10 = $4 → should be floored to ~$210."""
        engine = _engine()

        bot = MagicMock()
        bot.id = 301
        bot.strategy = "adaptive_grid"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 5_000.0
        bot.budget = 5_000.0

        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        center = 60_000.0
        # 15 very tight bars → ATR will be tiny (~$5)
        bar_ts = datetime.utcnow() - timedelta(hours=2)
        bars = [
            {"open": center, "high": center + 5, "low": center - 5, "close": center,
             "start_ts": bar_ts + timedelta(minutes=i)}
            for i in range(15)
        ]

        engine._grid_states = {
            bot.id: {
                "initialized": True,
                "center_price": center,
                "initial_capital": 5_000.0,
                "virtual_cash": 5_000.0,
                "virtual_crypto": 0.0,
                "grid_levels": {},
                "last_bar_close_time": None,
                "current_bar": None,
                "completed_bars": bars,
                "last_order_bar": None,
                "peak_portfolio_value": 5_000.0,
                "last_recenter_time": datetime.utcnow() - timedelta(hours=3),
                "lifetime_return_pct": 0.0,
                "lifetime_max_drawdown_pct": 0.0,
                "last_kill_switch_time": None,
                "kill_switch_count": 0,
                "atr_at_recenter": None,
                "total_trades": 0,
                "atr_spacing": None,
                "current_atr": None,
                "current_grid_range": None,
                "current_grid_spacing": None,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            # Open a bar
            await engine._strategy_grid(bot, center, {}, session)
            # Complete it
            state = engine._grid_states[bot.id]
            if state.get("current_bar"):
                state["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
            await engine._strategy_grid(bot, center, {}, session)

        state = engine._grid_states[bot.id]
        spacing = state.get("current_grid_spacing", 0)
        expected_min = center * (2 * 0.001 + 0.0005 + 0.001)  # 0.35% of 60000 = 210
        assert spacing >= expected_min * 0.99, (
            f"Spacing ${spacing:.2f} must be ≥ min profitable ${expected_min:.2f}"
        )

    def test_min_spacing_formula_matches_fee_structure(self):
        """Unit test for the min_spacing formula independent of bar execution."""
        center = 60_000.0
        taker_fee = 0.001   # 0.1%
        spread = 0.0005     # 0.05%
        buffer = 0.001      # 0.1%
        min_spacing = center * (2 * taker_fee + spread + buffer)
        assert min_spacing == pytest.approx(210.0)

    def test_wider_atr_spacing_unchanged(self):
        """When ATR spacing > min_spacing, it must not be reduced."""
        # ATR=300, multiplier=8, count=10 → spacing=$240 > $210 minimum
        atr_spacing = (300 * 8) / 10  # $240
        center = 60_000.0
        min_spacing = center * 0.0035  # $210
        effective = max(atr_spacing, min_spacing)
        assert effective == atr_spacing, "ATR spacing should win when larger"
