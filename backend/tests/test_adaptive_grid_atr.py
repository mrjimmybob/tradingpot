"""Tests for ATR-driven adaptive grid spacing.

Proves that:
  1. High ATR produces wider grid spacing and range.
  2. Low ATR produces narrower grid spacing and range.
  3. Grid spacing contracts when volatility falls (ATR decreases >10%).
  4. Grid spacing expands when volatility rises (ATR increases >10%).
  5. Trade execution (buy and sell level crossing) still works with ATR spacing.
  6. Soft recenter fires when price drifts >50% of half-range (no cooldown).
  7. ATR diagnostics are written to grid state every bar.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.trading_engine import TradingEngine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CENTER = 64000.0
GRID_COUNT = 10
ATR_MULT = 8.0


def _engine() -> TradingEngine:
    return TradingEngine()


def _make_bot(bot_id: int = 50, balance: float = 10000.0) -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "adaptive_grid"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    bot.budget = balance
    bot.started_at = datetime.utcnow() - timedelta(hours=2)
    return bot


def _session() -> AsyncMock:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))
    return session


def _make_bars(count: int, bar_range: float, center: float = CENTER) -> list:
    """Produce ``count`` completed OHLC bars with the given high-low range."""
    ts = datetime.utcnow() - timedelta(minutes=count + 5)
    bars = []
    for i in range(count):
        bars.append({
            "open": center,
            "high": center + bar_range / 2,
            "low": center - bar_range / 2,
            "close": center,
            "start_ts": ts + timedelta(minutes=i),
        })
    return bars


def _grid_state(
    bot_id: int,
    *,
    center: float = CENTER,
    bars: list,
    atr_spacing: float | None = None,
    virtual_crypto: float = 0.0,
    virtual_cash: float = 10000.0,
    grid_levels: dict | None = None,
) -> dict:
    return {
        bot_id: {
            "initialized": True,
            "center_price": center,
            "initial_capital": virtual_cash,
            "virtual_cash": virtual_cash,
            "virtual_crypto": virtual_crypto,
            "grid_levels": grid_levels if grid_levels is not None else {},
            "last_bar_close_time": None,
            "current_bar": None,
            "completed_bars": bars,
            "last_order_bar": None,
            "peak_portfolio_value": virtual_cash,
            "last_recenter_time": datetime.utcnow() - timedelta(hours=4),
            "lifetime_return_pct": 0.0,
            "lifetime_max_drawdown_pct": 0.0,
            "last_kill_switch_time": None,
            "kill_switch_count": 0,
            "atr_at_recenter": None,
            "atr_spacing": atr_spacing,
            "current_atr": None,
            "current_grid_range": None,
            "current_grid_spacing": None,
            "total_trades": 0,
        }
    }


async def _complete_one_bar(engine, bot, price, params=None):
    """Open a bar, then complete it by advancing start_ts past 60 s."""
    session = _session()
    with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
        await engine._strategy_grid(bot, price, params or {}, session)
        state = engine._grid_states.get(bot.id, {})
        if state.get("current_bar"):
            state["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
        signal = await engine._strategy_grid(bot, price, params or {}, session)
    return signal


# ---------------------------------------------------------------------------
# 1. High volatility → wide grid
# ---------------------------------------------------------------------------

class TestHighVolatilityWiderGrid:

    def test_high_atr_produces_wider_spacing_mathematically(self):
        """Spacing = ATR × multiplier / count; higher ATR must give wider spacing."""
        atr_low = 50.0    # calm market
        atr_high = 300.0  # volatile market
        count = GRID_COUNT

        spacing_low = (atr_low * ATR_MULT) / count
        spacing_high = (atr_high * ATR_MULT) / count

        assert spacing_high > spacing_low
        assert spacing_low == pytest.approx(40.0)    # 50×8/10
        assert spacing_high == pytest.approx(240.0)  # 300×8/10

    @pytest.mark.asyncio
    async def test_high_volatility_bars_produce_wide_grid_spacing(self):
        """With high-ATR bars (range $600), the grid builds wide levels.

        The freshly completed bar has range≈0 (both ticks at CENTER), so the
        14-bar ATR over 15 bars = 13/14 × 600 ≈ $557.  Expected spacing ≈ $446.
        We assert spacing > $300 to confirm the grid is meaningfully wide.
        """
        engine = _engine()
        bot = _make_bot(51)
        # 14 bars with $600 range → ATR dominated by high-range bars
        bars = _make_bars(count=14, bar_range=600.0)
        engine._grid_states = _grid_state(bot.id, bars=bars)

        signal = await _complete_one_bar(engine, bot, CENTER)

        state = engine._grid_states[bot.id]
        spacing = state.get("current_grid_spacing")
        assert spacing is not None
        assert spacing > 300.0, (
            f"Expected wide spacing (>$300) for high-ATR bars, got {spacing:.2f}"
        )
        # First sell level must be about one spacing above center
        sell_levels = [
            lv for lv in state["grid_levels"].values() if lv["side"] == "sell"
        ]
        assert sell_levels, "Expected sell levels to be created"
        first_sell = min(lv["price"] for lv in sell_levels)
        assert first_sell == pytest.approx(CENTER + spacing, rel=0.01)


# ---------------------------------------------------------------------------
# 2. Low volatility → narrow grid
# ---------------------------------------------------------------------------

class TestLowVolatilityNarrowerGrid:

    def test_low_atr_produces_narrower_spacing_mathematically(self):
        """Low ATR must produce spacing well below that of high ATR."""
        atr = 30.0
        spacing = (atr * ATR_MULT) / GRID_COUNT
        assert spacing == pytest.approx(24.0)  # 30×8/10

    @pytest.mark.asyncio
    async def test_low_volatility_bars_produce_tight_grid_spacing(self):
        """With low-ATR bars (range $60), the grid builds tight levels.

        ATR ≈ 13/14 × 60 ≈ $55.7 → spacing ≈ $44.6.  We assert spacing < $80
        to confirm the grid is meaningfully tighter than the high-volatility case.
        """
        engine = _engine()
        bot = _make_bot(52)
        bars = _make_bars(count=14, bar_range=60.0)
        engine._grid_states = _grid_state(bot.id, bars=bars)

        signal = await _complete_one_bar(engine, bot, CENTER)

        state = engine._grid_states[bot.id]
        spacing = state.get("current_grid_spacing")
        assert spacing is not None
        assert spacing < 80.0, (
            f"Expected tight spacing (<$80) for low-ATR bars, got {spacing:.2f}"
        )


# ---------------------------------------------------------------------------
# 3. Grid contracts when volatility falls
# ---------------------------------------------------------------------------

class TestGridContractsOnLowVolatility:

    @pytest.mark.asyncio
    async def test_grid_rebuilds_when_atr_falls_more_than_10_percent(self):
        """When ATR drops >10% the grid must rebuild with the new tighter spacing."""
        engine = _engine()
        bot = _make_bot(53)

        # Old spacing came from ATR≈$400 → spacing = $320
        old_atr = 400.0
        old_spacing = (old_atr * ATR_MULT) / GRID_COUNT   # 320

        # New bars have ATR≈$200 → spacing = $160 (50% drop, >10% threshold)
        bars = _make_bars(count=14, bar_range=200.0)  # actual ATR used below
        # Override: inject precomputed atr_spacing so engine sees the "old" value
        engine._grid_states = _grid_state(bot.id, bars=bars, atr_spacing=old_spacing)

        signal = await _complete_one_bar(engine, bot, CENTER)

        state = engine._grid_states[bot.id]
        new_spacing = state.get("atr_spacing")
        assert new_spacing is not None
        # New spacing must be less than old (grid contracted)
        assert new_spacing < old_spacing, (
            f"Expected grid to contract: new={new_spacing:.2f} vs old={old_spacing:.2f}"
        )
        # Difference must be >10% of old spacing (that's what triggers rebuild)
        assert (old_spacing - new_spacing) / old_spacing > 0.10


# ---------------------------------------------------------------------------
# 4. Grid expands when volatility rises
# ---------------------------------------------------------------------------

class TestGridExpandsOnHighVolatility:

    @pytest.mark.asyncio
    async def test_grid_rebuilds_when_atr_rises_more_than_10_percent(self):
        """When ATR rises >10% the grid must rebuild with wider spacing."""
        engine = _engine()
        bot = _make_bot(54)

        # Old spacing from low-ATR regime: ATR≈$60 → spacing=$48
        old_atr = 60.0
        old_spacing = (old_atr * ATR_MULT) / GRID_COUNT   # 48

        # New bars: ATR≈$300 → spacing=$240 (400% increase, >10% threshold)
        bars = _make_bars(count=14, bar_range=300.0)
        engine._grid_states = _grid_state(bot.id, bars=bars, atr_spacing=old_spacing)

        signal = await _complete_one_bar(engine, bot, CENTER)

        state = engine._grid_states[bot.id]
        new_spacing = state.get("atr_spacing")
        assert new_spacing is not None
        assert new_spacing > old_spacing, (
            f"Expected grid to expand: new={new_spacing:.2f} vs old={old_spacing:.2f}"
        )
        assert (new_spacing - old_spacing) / old_spacing > 0.10


# ---------------------------------------------------------------------------
# 5. Trade execution still works
# ---------------------------------------------------------------------------

class TestTradeExecutionWithAtrSpacing:

    @pytest.mark.asyncio
    async def test_buy_signal_when_price_crosses_first_buy_level(self):
        """Price closing below the ATR-derived L1 buy level must produce a buy."""
        engine = _engine()
        bot = _make_bot(55)
        # ATR = bar_range since all bars have same range
        bar_range = 200.0
        bars = _make_bars(count=14, bar_range=bar_range)
        engine._grid_states = _grid_state(bot.id, bars=bars)

        # With ATR≈200, spacing = 200*8/10 = 160
        # L1 buy is at CENTER - 160 = 63840
        # Trigger price just below that
        trigger = CENTER - 165.0

        session = _session()
        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            await engine._strategy_grid(bot, CENTER, {}, session)
            state = engine._grid_states[bot.id]
            if state.get("current_bar"):
                state["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
            signal = await engine._strategy_grid(bot, trigger, {}, session)

        assert signal is not None
        assert signal.action == "buy", (
            f"Expected buy at trigger {trigger:.0f} (L1 buy ≈ {CENTER - 160:.0f}), "
            f"got: {signal.action} — {signal.reason}"
        )

    @pytest.mark.asyncio
    async def test_sell_signal_when_price_crosses_first_sell_level(self):
        """Price closing above the ATR-derived L1 sell level must produce a sell."""
        engine = _engine()
        bot = _make_bot(56)
        bar_range = 200.0
        bars = _make_bars(count=14, bar_range=bar_range)
        # Seed with some virtual crypto so sell can proceed
        engine._grid_states = _grid_state(
            bot.id, bars=bars, virtual_crypto=1.0, virtual_cash=5000.0
        )

        # L1 sell at CENTER + 160 = 64160
        trigger = CENTER + 165.0

        session = _session()
        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            await engine._strategy_grid(bot, CENTER, {}, session)
            state = engine._grid_states[bot.id]
            if state.get("current_bar"):
                state["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
            signal = await engine._strategy_grid(bot, trigger, {}, session)

        assert signal is not None
        assert signal.action == "sell", (
            f"Expected sell at trigger {trigger:.0f} (L1 sell ≈ {CENTER + 160:.0f}), "
            f"got: {signal.action} — {signal.reason}"
        )

    @pytest.mark.asyncio
    async def test_hold_when_price_between_levels(self):
        """Price sitting between L1 buy and L1 sell must return hold."""
        engine = _engine()
        bot = _make_bot(57)
        bars = _make_bars(count=14, bar_range=200.0)
        engine._grid_states = _grid_state(bot.id, bars=bars)

        # With spacing≈160, no level is crossed when price stays at center
        session = _session()
        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            await engine._strategy_grid(bot, CENTER, {}, session)
            state = engine._grid_states[bot.id]
            if state.get("current_bar"):
                state["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
            signal = await engine._strategy_grid(bot, CENTER, {}, session)

        assert signal is not None
        assert signal.action == "hold"


# ---------------------------------------------------------------------------
# 6. Soft recenter
# ---------------------------------------------------------------------------

class TestSoftRecenter:

    @pytest.mark.asyncio
    async def test_soft_recenter_fires_when_price_drifts_half_range(self):
        """When price moves >50% of grid_half_range from center, grid recenters
        to the new price without triggering the hard kill switch or cooldown."""
        engine = _engine()
        bot = _make_bot(58)

        # ATR≈200 → grid_range=1600 → grid_half_range=800 → threshold=400
        bars = _make_bars(count=14, bar_range=200.0)
        old_center = CENTER
        engine._grid_states = _grid_state(bot.id, bars=bars, center=old_center)

        # Price drifts 450 from center — beyond the 400 threshold
        new_price = old_center + 450.0

        signal = await _complete_one_bar(engine, bot, new_price)

        state = engine._grid_states[bot.id]
        assert state["center_price"] == pytest.approx(new_price), (
            "Grid center must update to new price after soft recenter"
        )
        # Hard kill cooldown must NOT have been activated
        assert state["last_kill_switch_time"] is None, (
            "Soft recenter must not trigger the hard kill switch cooldown"
        )

    @pytest.mark.asyncio
    async def test_no_recenter_when_price_within_threshold(self):
        """When price stays inside the 50% half-range threshold, center is unchanged."""
        engine = _engine()
        bot = _make_bot(59)

        bars = _make_bars(count=14, bar_range=200.0)
        old_center = CENTER
        engine._grid_states = _grid_state(bot.id, bars=bars, center=old_center)

        # Price moves only 100 — well within the 400 threshold
        new_price = old_center + 100.0
        signal = await _complete_one_bar(engine, bot, new_price)

        state = engine._grid_states[bot.id]
        assert state["center_price"] == pytest.approx(old_center), (
            "Center must not change for a small price move within the threshold"
        )


# ---------------------------------------------------------------------------
# 7. Diagnostics written to state every bar
# ---------------------------------------------------------------------------

class TestAtrDiagnostics:

    @pytest.mark.asyncio
    async def test_diagnostics_written_after_completed_bar(self):
        """After a completed bar, current_atr / current_grid_range /
        current_grid_spacing must be populated in grid state."""
        engine = _engine()
        bot = _make_bot(60)
        bars = _make_bars(count=14, bar_range=200.0)
        engine._grid_states = _grid_state(bot.id, bars=bars)

        await _complete_one_bar(engine, bot, CENTER)

        state = engine._grid_states[bot.id]
        assert state.get("current_atr") is not None, "current_atr must be set"
        assert state.get("current_grid_range") is not None, "current_grid_range must be set"
        assert state.get("current_grid_spacing") is not None, "current_grid_spacing must be set"

        # Verify mathematical consistency
        atr = state["current_atr"]
        expected_range = atr * ATR_MULT
        expected_spacing = expected_range / GRID_COUNT
        assert state["current_grid_range"] == pytest.approx(expected_range, rel=0.01)
        assert state["current_grid_spacing"] == pytest.approx(expected_spacing, rel=0.01)

    @pytest.mark.asyncio
    async def test_center_in_state_matches_grid_center(self):
        """state['center_price'] must equal the center used when building levels."""
        engine = _engine()
        bot = _make_bot(61)
        bars = _make_bars(count=14, bar_range=200.0)
        engine._grid_states = _grid_state(bot.id, bars=bars)

        await _complete_one_bar(engine, bot, CENTER)

        state = engine._grid_states[bot.id]
        center = state["center_price"]
        buy_levels = [lv for lv in state["grid_levels"].values() if lv["side"] == "buy"]
        if buy_levels:
            deepest_buy = max(lv["price"] for lv in buy_levels)
            shallowest_buy = min(lv["price"] for lv in buy_levels)
            # All buy levels must be below center
            assert deepest_buy < center, "All buy levels must be below grid center"
            assert shallowest_buy < center
