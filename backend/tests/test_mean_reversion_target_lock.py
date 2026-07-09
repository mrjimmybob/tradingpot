"""Regression tests for the mean-reversion moving-target bug (Finding 1).

Root cause covered here:
``_strategy_mean_reversion`` locked ``entry_atr``/``hard_stop`` at entry (risk
never expands) but recomputed the profit target on every tick from the live
Bollinger SMA:

    exit_level = sma          # recomputed every call
    ...
    if last_bar_close >= exit_level: sell "mean reached"

Mean reversion only enters while price is already declining into the lower
band. As the decline continues, each new (lower) bar close drags the SMA down
with it, so the "target" silently follows price down. "Mean reached" then
fires the moment price stops falling — even if it is still below (or barely
above) the entry price — turning a take-profit exit into a moving-loss exit.
Forensic data showed this exit losing money 36/47 times (77%).

The fix locks ``target_price`` into strategy state at entry, exactly like
``entry_atr``/``hard_stop`` already are, and the "mean reached" exit now
compares against that frozen value instead of a freshly recomputed sma.

These tests:
1. Seed a position whose *locked* target sits above the current *live* SMA
   (simulating the SMA having fallen after entry) and prove price crossing
   the decayed live SMA does NOT trigger a sell.
2. Prove a sell still fires, using the ORIGINAL locked target, once price
   actually reaches it, even though the live SMA has moved further away.
3. Prove a legacy position (persisted before this fix, no ``target_price`` in
   its state) falls back gracefully instead of raising.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.trading_engine import TradingEngine


def _bar(close: float) -> dict:
    return {
        "open": close, "high": close, "low": close, "close": close,
        "start_ts": datetime.utcnow(),
    }


_PARAMS = {
    "bollinger_period": 3,
    "atr_period": 3,
    "bar_interval_seconds": 600,  # current bar never completes mid-test
    "max_hold_bars": 10,
    "exit_at_mean": True,
    "regime_filter_enabled": False,  # avoid regime helpers; force_exit stays False
}


def _seed(engine: TradingEngine, bot_id: int, bars: list, **state_overrides) -> None:
    if not hasattr(engine, "_mean_reversion_states"):
        engine._mean_reversion_states = {}
    state = {
        "bars": bars,
        "current_bar": None,
        "entry_price": 100.0,
        "entry_atr": 1.0,
        "target_price": 105.0,
        "hard_stop": 95.0,
        "bars_since_entry": 1,
        "last_exit_time": None,
    }
    state.update(state_overrides)
    engine._mean_reversion_states[bot_id] = state


@pytest.mark.asyncio
async def test_decayed_live_sma_does_not_trigger_sell():
    """SMA has fallen to ~96 after entry; price (103) crosses that decayed
    SMA but never reaches the original locked target (105) -> must HOLD.

    Under the pre-fix behaviour (exit_level recomputed as the live sma every
    tick), 103 >= 96 would have fired a "mean reached" sell here — a loss
    relative to the $100 entry once fees are included, despite being labelled
    a take-profit exit.
    """
    engine = TradingEngine()
    bot = SimpleNamespace(id=1, trading_pair="BTC/USDT")
    # closes [90, 95, 103] -> live sma = 96.0, last_bar_close = 103
    _seed(engine, bot.id, [_bar(90.0), _bar(95.0), _bar(103.0)])
    engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.01)])

    signal = await engine._strategy_mean_reversion(
        bot, current_price=103.0, params=dict(_PARAMS), session=None
    )

    assert signal is not None
    assert signal.action == "hold", (
        f"expected HOLD (target not reached), got {signal.action}: {signal.reason}"
    )


@pytest.mark.asyncio
async def test_locked_target_still_exits_when_actually_reached():
    """Price reaches the ORIGINAL locked target (105) even though the live
    SMA has since fallen further (to ~90) -> must SELL, and the reported
    target must be the locked value, not the decayed live SMA.
    """
    engine = TradingEngine()
    bot = SimpleNamespace(id=1, trading_pair="BTC/USDT")
    # closes [80, 85, 106] -> live sma = 90.33, last_bar_close = 106
    _seed(engine, bot.id, [_bar(80.0), _bar(85.0), _bar(106.0)])
    engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.01)])

    signal = await engine._strategy_mean_reversion(
        bot, current_price=106.0, params=dict(_PARAMS), session=None
    )

    assert signal is not None
    assert signal.action == "sell"
    assert "mean reached" in signal.reason.lower()
    assert "105.00" in signal.reason, (
        f"exit should report the locked target ($105.00), not the decayed live "
        f"sma (~$90.33): {signal.reason}"
    )


@pytest.mark.asyncio
async def test_legacy_state_without_target_price_falls_back_without_raising():
    """A position persisted before this fix has no ``target_price`` key.
    The strategy must fall back to the live sma for that tick, not crash.
    """
    engine = TradingEngine()
    bot = SimpleNamespace(id=1, trading_pair="BTC/USDT")
    _seed(engine, bot.id, [_bar(90.0), _bar(95.0), _bar(103.0)])
    del engine._mean_reversion_states[bot.id]["target_price"]
    engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.01)])

    signal = await engine._strategy_mean_reversion(
        bot, current_price=103.0, params=dict(_PARAMS), session=None
    )

    assert signal is not None
    # live sma = 96.0, last_bar_close = 103 >= 96 -> legacy fallback exits
    assert signal.action == "sell"


@pytest.mark.asyncio
async def test_hard_stop_fires_on_the_same_tick_it_is_reached_not_delayed():
    """Finding 4 guard: losses must not silently exceed the configured risk.

    The hard stop is locked at entry (target_price set far away so it cannot
    fire first). Price just above the stop must HOLD; price at-or-below the
    stop must exit immediately, on that same tick — not one tick later, which
    would silently let the realized loss run past the configured stop
    distance before anything reacts.
    """
    engine = TradingEngine()
    bot = SimpleNamespace(id=1, trading_pair="BTC/USDT")
    bars = [_bar(100.0), _bar(100.0), _bar(100.0)]  # live sma=100, target locked far above
    engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.01)])

    _seed(
        engine, bot.id, list(bars),
        target_price=200.0,  # unreachable this tick -> isolates the hard-stop path
        hard_stop=95.0,
        bars_since_entry=0,
    )
    just_above = await engine._strategy_mean_reversion(
        bot, current_price=95.01, params=dict(_PARAMS), session=None
    )
    assert just_above.action == "hold", "must not exit before price reaches the configured stop"

    _seed(
        engine, bot.id, list(bars),
        target_price=200.0,
        hard_stop=95.0,
        bars_since_entry=0,
    )
    at_stop = await engine._strategy_mean_reversion(
        bot, current_price=95.0, params=dict(_PARAMS), session=None
    )
    assert at_stop.action == "sell"
    assert "hard stop" in at_stop.reason.lower()
