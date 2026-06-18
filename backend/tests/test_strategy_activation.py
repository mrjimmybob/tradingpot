"""Regression tests for the strategy-activation audit fixes.

Covers the bugs that made the live fleet either pause or never trade:

  F1  Trend Following & Volatility Breakout crashed on the hold-with-position
      path via an f-string conditional format spec ({x:.2f if ...}) — the same
      ValueError class already fixed in Mean Reversion. A held position raised it
      every tick, tripping the failure breaker and PAUSING the bot. (Bots 2/8.)

  F2  Standalone Mean Reversion & Volatility Breakout detected market regime from
      a tick price-history buffer that ONLY trend_following/funding_carry ever
      populate. Empty -> _detect_market_regime returns neutral 'flat/medium'
      forever, which (a) disabled Mean Reversion's trend force-exit and (b)
      PERMANENTLY blocked Volatility Breakout's entry. Now both read their own
      bar closes. (Bots 4/9.)

  F4  Volatility Breakout's regime veto ran BEFORE compression tracking and
      early-returned, so compression_bars could never accumulate; and it demanded
      'expanding' volatility at the instant compression demands LOW volatility.
      Compression is now tracked unconditionally and a confirmed breakout is no
      longer vetoed.

  F3  Adaptive Grid's default allowed_regimes carried a dead 'volatility_normal'
      tag (the detector only emits low/medium/high) — now 'volatility_medium'.

Lives in backend/tests (run via pytest); bar-based strategies are driven with
bar_interval_seconds=0 so every call closes a bar.
"""
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models import BotStatus
from app.services.trading_engine import TradingEngine, TradeSignal


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_bot(strategy, params=None, balance=1000.0, budget=1000.0, bot_id=1):
    return SimpleNamespace(
        id=bot_id, name="vbot", trading_pair="BTC/USDT", strategy=strategy,
        strategy_params=params or {}, budget=budget, current_balance=balance,
        compound_enabled=False, is_dry_run=True, status=BotStatus.RUNNING,
        total_pnl=0.0,
    )


def engine_with_position(amount=0.001):
    """Engine whose position lookup reports one open long, with the tick
    price-history buffer deliberately EMPTY (the standalone-bot condition that
    triggered F2)."""
    engine = TradingEngine()
    engine._get_bot_positions = AsyncMock(return_value=[
        SimpleNamespace(amount=amount, trading_pair="BTC/USDT", entry_price=50000.0)
    ])
    engine._get_last_order = AsyncMock(return_value=None)
    engine._get_order_count = AsyncMock(return_value=0)
    # No _save_price_history caller has run -> empty shared tick buffer.
    engine._price_histories = {}
    return engine


def flat_engine():
    engine = TradingEngine()
    engine._get_bot_positions = AsyncMock(return_value=[])
    engine._get_last_order = AsyncMock(return_value=None)
    engine._get_order_count = AsyncMock(return_value=0)
    engine._price_histories = {}
    return engine


# --------------------------------------------------------------------------- #
# F1 - hold-with-position path must NOT raise (the pause bug)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_trend_following_hold_with_position_does_not_raise():
    """Drive Trend Following into a held position and assert the hold tick
    returns a valid signal instead of raising ValueError (Invalid format
    specifier), which previously paused Bots 2/8."""
    engine = engine_with_position()
    bot = make_bot("trend_following", balance=100.0, params={
        "short_period": 3, "long_period": 5, "atr_period": 3,
        "atr_multiplier": 2.0, "entry_confirmation_loops": 1,
        "exit_confirmation_loops": 2, "cooldown_seconds": 0,
    })
    # Seed a held position's state with a real (float) trailing stop — exactly
    # the value that made the old f-string spec raise.
    engine._trend_states = {bot.id: {
        "trailing_stop": 49000.0, "highest_price": 50500.0, "entry_atr": 100.0,
        "entry_time": None, "last_exit_time": None,
        "entry_confirmation_count": 0, "exit_confirmation_count": 0,
    }}
    executor = engine._get_strategy_executor("trend_following")
    # Pre-fill price history above the long EMA and above the trailing stop so we
    # land on the hold path (no new high, no stop hit, price > EMA(long)).
    engine._price_histories[bot.id] = [49800.0, 49850.0, 49900.0, 49950.0, 50000.0]
    sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
    assert isinstance(sig, TradeSignal)
    assert sig.action == "hold"
    assert "stop at" in sig.reason
    assert "$49000.00" in sig.reason  # formatted, not crashed


@pytest.mark.asyncio
async def test_trend_following_hold_with_none_stop_does_not_raise():
    engine = engine_with_position()
    bot = make_bot("trend_following", balance=100.0, params={
        "short_period": 3, "long_period": 5, "atr_period": 3,
        "exit_confirmation_loops": 5,
    })
    # trailing_stop None -> must render "N/A", not raise.
    engine._trend_states = {bot.id: {
        "trailing_stop": None, "highest_price": 50000.0, "entry_atr": 100.0,
        "entry_time": None, "last_exit_time": None,
        "entry_confirmation_count": 0, "exit_confirmation_count": 0,
    }}
    executor = engine._get_strategy_executor("trend_following")
    engine._price_histories[bot.id] = [49800.0, 49850.0, 49900.0, 49950.0, 50000.0]
    sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
    assert sig.action == "hold"
    assert "N/A" in sig.reason


@pytest.mark.asyncio
async def test_volatility_breakout_hold_with_position_does_not_raise():
    """Same crash class as F1 in Volatility Breakout's hold path."""
    engine = engine_with_position()
    bot = make_bot("volatility_breakout", balance=100.0, params={
        "bar_interval_seconds": 0, "bb_period": 5, "atr_period": 3,
        "failed_breakout_bars": 0,  # past the failed-breakout window
    })
    executor = engine._get_strategy_executor("volatility_breakout")
    # Build >= bb_period bars, then seed a held position with a float stop.
    for p in [50000.0, 50010.0, 49990.0, 50020.0, 50000.0, 50015.0]:
        await executor(bot, p, bot.strategy_params, SimpleNamespace())
    st = engine._volatility_breakout_states[bot.id]
    st.update({
        "entry_price": 50000.0, "entry_atr": 50.0, "highest_price": 50500.0,
        "trailing_stop": 49500.0, "bars_since_entry": 10,
    })
    sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
    assert sig.action == "hold"
    assert "$49500.00" in sig.reason


# --------------------------------------------------------------------------- #
# F2 - regime now derived from the strategy's own bars
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mean_reversion_force_exits_in_downtrend_from_bars():
    """A standalone Mean Reversion bot holding a position through a sustained
    DOWNTREND must FORCE-EXIT on the regime flip. A downtrend isolates the
    regime path: price stays below the SMA (so the 'mean reached' target never
    fires) and with no hard stop set the ONLY exit is the regime force-exit.

    Pre-fix the regime came from an empty tick buffer (always 'flat'), so
    force_exit_regime was never True and the bot just held; now it reads its own
    bar closes and correctly bails out of a trend."""
    engine = engine_with_position()
    bot = make_bot("mean_reversion", params={
        "bar_interval_seconds": 0, "bollinger_period": 20, "atr_period": 5,
    })
    executor = engine._get_strategy_executor("mean_reversion")
    # Sustained downtrend so the bar-based regime reads "down" (state default
    # leaves hard_stop/entry_price None -> no hard/time stop interferes). The
    # regime detector only resolves a trend once it has >= 50 closes (below that
    # ema_50 collapses to ema_20 and the trend comparison is always false), so
    # drive >50 bars.
    sig = None
    for i in range(70):
        price = 50000.0 - 200.0 * i  # sustained decline
        sig = await executor(bot, price, bot.strategy_params, SimpleNamespace())
    assert sig is not None
    assert sig.action == "sell", f"expected regime force-exit, got {sig.action}: {sig.reason}"
    assert "regime" in sig.reason.lower()


@pytest.mark.asyncio
async def test_volatility_breakout_not_permanently_regime_blocked():
    """Pre-fix a standalone Volatility Breakout bot was pinned to
    volatility_normal (empty tick buffer) and PERMANENTLY blocked, so it never
    accrued a single compression bar. Now compression is tracked unconditionally
    from its own bars — prove compression_bars can climb above zero."""
    engine = flat_engine()
    bot = make_bot("volatility_breakout", balance=100.0, params={
        "bar_interval_seconds": 0, "bb_period": 5, "atr_period": 3,
        "min_compression_bars": 3, "compression_percentile": 95,
        "cooldown_hours": 0,
    })
    executor = engine._get_strategy_executor("volatility_breakout")
    # >20 calm bars so bb_width_history fills and recent width is "compressed".
    for i in range(40):
        price = 50000.0 + (5.0 if i % 2 else -5.0)  # very tight range
        sig = await executor(bot, price, bot.strategy_params, SimpleNamespace())
        assert isinstance(sig, TradeSignal)
    st = engine._volatility_breakout_states[bot.id]
    assert st["compression_bars"] > 0, (
        "compression never tracked — regime veto still blocks compression"
    )


@pytest.mark.asyncio
async def test_volatility_breakout_completes_compression_then_breakout_buy():
    """Full activation path: build compression, then a breakout above the upper
    band must produce a valid executable BUY (regime filter ON)."""
    engine = flat_engine()
    bot = make_bot("volatility_breakout", balance=100.0, params={
        "bar_interval_seconds": 0, "bb_period": 5, "bb_std": 1.0, "atr_period": 3,
        "min_compression_bars": 1, "compression_percentile": 95,
        "cooldown_hours": 0, "regime_filter_enabled": True,
    })
    executor = engine._get_strategy_executor("volatility_breakout")
    actions = []
    buy_signal = None
    # 25 calm bars (fill width history + arm compression), then a sharp expansion
    # that pushes a bar close above the upper Bollinger band. The arming latch
    # must survive the wide breakout bar.
    prices = [50000.0 + (8.0 if i % 2 else -8.0) for i in range(25)]
    prices += [50000.0, 50300.0, 50800.0, 51600.0, 52800.0]  # accelerating breakout
    for p in prices:
        sig = await executor(bot, p, bot.strategy_params, SimpleNamespace())
        assert isinstance(sig, TradeSignal)
        actions.append(sig.action)
        if sig.action == "buy":
            buy_signal = sig
    assert "buy" in actions, f"Volatility Breakout never entered (actions={set(actions)})"
    # The breakout BUY must be an executable notional (>= the $10 minimum).
    from app.services.trading_engine import MIN_ORDER_USD
    assert buy_signal.amount >= MIN_ORDER_USD
    assert buy_signal.amount <= bot.current_balance + 1e-9


# --------------------------------------------------------------------------- #
# F3 - Adaptive Grid default regime tag is no longer the dead 'volatility_normal'
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grid_default_regime_allows_medium_volatility():
    """The detector emits volatility in {low,medium,high}; the grid default must
    reference 'volatility_medium', not the never-emitted 'volatility_normal'."""
    engine = flat_engine()
    bot = make_bot("adaptive_grid", params={"bar_interval_seconds": 0, "atr_period": 3})
    # Reach the regime block once (needs a couple of bars) and confirm a
    # medium-volatility tape is NOT rejected with a 'need flat/normal' reason.
    executor = engine._get_strategy_executor("adaptive_grid")
    seen_dead_tag = False
    for i, p in enumerate([50000.0 + 30.0 * math.sin(i / 3.0) for i in range(60)]):
        sig = await executor(bot, p, bot.strategy_params, SimpleNamespace())
        if sig and "volatility_normal" in sig.reason:
            seen_dead_tag = True
    assert not seen_dead_tag, "grid still surfaces the dead 'volatility_normal' tag"
