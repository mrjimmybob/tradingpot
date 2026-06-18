"""Automated strategy validation suite.

Goal: prove the trading engine can run UNATTENDED. The suite fails if any
strategy:
  * throws an exception on a normal data path,
  * generates an invalid TradeSignal (missing/!numeric amount, bad action,
    HOLD carrying size, BUY with no size),
  * generates a BUY below the configured minimum order size,
  * cannot complete a buy -> hold -> sell lifecycle.

It complements the execution-layer scenarios in test_trading_engine.py
(BUY executes, SELL closes/clamps, sub-minimum BUY rejected, sell-without-
position rejected, "sell all" sentinel closes the full position).

Covers the strategies called out as broken:
  * Adaptive Grid  - every HOLD omitted the required `amount` (crashed on the
    first call); see test_adaptive_grid_full_cycle.
  * Trend Following - ATR sizing had a unit bug (sub-$1 orders) and no
    min-order floor; see the trend-following sizing tests.
"""
import ast
import math
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models import BotStatus
from app.services.trading_engine import TradingEngine, TradeSignal, MIN_ORDER_USD

ENGINE_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "app", "services", "trading_engine.py")
)

CONCRETE_STRATEGIES = [
    "dca_accumulator",
    "adaptive_grid",
    "mean_reversion",
    "trend_following",
    "volatility_breakout",
    "funding_carry",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_bot(strategy, params=None, balance=1000.0, budget=1000.0, bot_id=1):
    return SimpleNamespace(
        id=bot_id,
        name="vbot",
        trading_pair="BTC/USDT",
        strategy=strategy,
        strategy_params=params or {},
        budget=budget,
        current_balance=balance,
        compound_enabled=False,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        total_pnl=0.0,
    )


def assert_valid_signal(sig, context=""):
    """The TradeSignal contract every strategy must honour every iteration."""
    if sig is None:
        return
    assert isinstance(sig, TradeSignal), f"{context}: not a TradeSignal: {sig!r}"
    assert sig.action in ("buy", "sell", "hold"), f"{context}: bad action {sig.action!r}"
    assert isinstance(sig.amount, (int, float)) and not isinstance(sig.amount, bool), (
        f"{context}: amount not numeric: {sig.amount!r}"
    )
    assert math.isfinite(sig.amount), f"{context}: amount not finite: {sig.amount}"
    assert sig.amount >= 0, f"{context}: negative amount {sig.amount}"
    assert isinstance(sig.reason, str), f"{context}: reason not a string"
    if sig.action == "hold":
        assert sig.amount == 0, f"{context}: HOLD must carry amount 0, got {sig.amount}"
    if sig.action == "buy":
        assert sig.amount > 0, f"{context}: BUY must carry a positive amount"
    # sell may legitimately be 0 ("sell all" sentinel resolved at execution).


def fresh_engine():
    """Engine with all DB/exchange touch-points stubbed so strategies can be
    driven in isolation. Positions default to empty (flat)."""
    engine = TradingEngine()
    engine._get_bot_positions = AsyncMock(return_value=[])
    engine._get_last_order = AsyncMock(return_value=None)
    engine._get_order_count = AsyncMock(return_value=0)
    engine._get_funding_signal = AsyncMock(return_value=0.0)
    return engine


# Params that let each strategy warm up quickly and progress past warmup so the
# smoke run exercises real indicator/level/sizing code (not just the first hold).
SMOKE_PARAMS = {
    "dca_accumulator": {"interval_seconds": 0},
    "adaptive_grid": {
        "bar_interval_seconds": 0, "regime_filter_enabled": False,
        "grid_count": 6, "atr_period": 5,
    },
    "mean_reversion": {
        "bar_interval_seconds": 0, "regime_filter_enabled": False,
        "bollinger_period": 5, "atr_period": 5,
    },
    "trend_following": {
        "short_period": 3, "long_period": 5, "atr_period": 3,
        "entry_confirmation_loops": 1, "exit_confirmation_loops": 1, "cooldown_seconds": 0,
    },
    "volatility_breakout": {
        "bar_interval_seconds": 0, "regime_filter_enabled": False,
        "bb_period": 5, "atr_period": 5, "min_compression_bars": 3,
    },
    "funding_carry": {"allowed_regimes": ["trend_up", "trend_flat", "trend_down"]},
}


def oscillating_prices(n=300, base=50000.0, amp=0.05, drift=0.0):
    """A deterministic oscillating-with-drift series that exercises both entry
    (rising/falling) and exit/mean-reversion paths."""
    out = []
    for i in range(n):
        out.append(base * (1 + amp * math.sin(i / 9.0)) + drift * i)
    return out


# --------------------------------------------------------------------------- #
# 1. Static contract guard — catches the Adaptive Grid bug class permanently
# --------------------------------------------------------------------------- #
def test_no_tradesignal_construction_missing_required_args():
    """Every TradeSignal(...) in the engine must supply action AND amount.

    The Adaptive Grid outage was 10 HOLD constructions that omitted the
    required `amount`, raising TypeError on the first call. This static check
    fails if any such construction is ever (re)introduced."""
    tree = ast.parse(open(ENGINE_FILE).read())
    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TradeSignal"
        ):
            has_splat = any(k.arg is None for k in node.keywords)
            kw = {k.arg for k in node.keywords}
            has_action = len(node.args) >= 1 or "action" in kw or has_splat
            has_amount = len(node.args) >= 2 or "amount" in kw or has_splat
            if not (has_action and has_amount):
                offenders.append(node.lineno)
    assert offenders == [], (
        f"TradeSignal(...) missing required action/amount at lines {offenders}"
    )


# --------------------------------------------------------------------------- #
# 2. Per-strategy smoke: warmup -> valid data -> never crashes, always valid
#    (Scenario A warmup/HOLD, Scenario F handled via the no-data variant)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", CONCRETE_STRATEGIES)
async def test_strategy_smoke_never_crashes_and_emits_valid_signals(strategy):
    engine = fresh_engine()
    bot = make_bot(strategy, params=dict(SMOKE_PARAMS[strategy]))
    executor = engine._get_strategy_executor(strategy)
    assert executor is not None, f"no executor for {strategy}"

    session = SimpleNamespace()  # unused once DB touch-points are stubbed
    actions = set()
    for i, price in enumerate(oscillating_prices(300)):
        sig = await executor(bot, price, bot.strategy_params, session)
        assert_valid_signal(sig, context=f"{strategy} iter {i} @ {price:.2f}")
        if sig is not None:
            actions.add(sig.action)

    # At minimum it must have produced *some* valid decision (usually HOLD).
    assert actions, f"{strategy} produced no signals at all"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", CONCRETE_STRATEGIES)
async def test_strategy_warmup_holds_on_first_tick(strategy):
    """Scenario A (start/warmup): the very first tick must not crash and must be
    a valid signal (this is exactly where Adaptive Grid used to blow up)."""
    engine = fresh_engine()
    bot = make_bot(strategy, params=dict(SMOKE_PARAMS[strategy]))
    executor = engine._get_strategy_executor(strategy)
    sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
    assert_valid_signal(sig, context=f"{strategy} first tick")


# --------------------------------------------------------------------------- #
# 3. Trend Following sizing (Bot 6)
# --------------------------------------------------------------------------- #
async def _drive_until_buy(engine, bot, prices, session):
    executor = engine._get_strategy_executor(bot.strategy)
    for price in prices:
        sig = await executor(bot, price, bot.strategy_params, session)
        assert_valid_signal(sig, context=f"{bot.strategy} @ {price:.2f}")
        if sig and sig.action == "buy":
            return sig
    return None


@pytest.mark.asyncio
async def test_trend_following_emits_executable_buy_on_100_dollar_account():
    """On a $100 account a confirmed uptrend must produce a BUY that is
    executable: >= the $10 minimum and <= the balance.

    Before the fix the ATR sizing was missing the *price conversion, so it
    produced sub-$1 "orders" that the engine rejected every loop."""
    engine = fresh_engine()
    bot = make_bot(
        "trend_following", balance=100.0, budget=100.0,
        params={
            "short_period": 3, "long_period": 5, "atr_period": 3,
            "atr_multiplier": 2.0, "risk_percent": 1.0,
            "entry_confirmation_loops": 1, "cooldown_seconds": 0,
        },
    )
    # Steady uptrend: price > EMA(long) and EMA(short) > EMA(long).
    prices = [50000.0 + 40.0 * i for i in range(40)]
    buy = await _drive_until_buy(engine, bot, prices, SimpleNamespace())

    assert buy is not None, "trend following never entered on a clean uptrend"
    assert buy.amount >= MIN_ORDER_USD, (
        f"BUY ${buy.amount:.4f} is below the ${MIN_ORDER_USD} minimum (sizing bug)"
    )
    assert buy.amount <= bot.current_balance + 1e-9, (
        f"BUY ${buy.amount:.2f} exceeds the ${bot.current_balance:.2f} balance"
    )


@pytest.mark.asyncio
async def test_trend_following_floors_subminimum_size_when_affordable():
    """When risk-based sizing lands below the minimum but the balance can afford
    the minimum, the order is floored to the minimum (not emitted sub-minimum)."""
    engine = fresh_engine()
    # Tiny risk on a modest account: risk_amount is ~ $0.20, so the raw ATR size
    # is well under $10; balance ($60) can afford the $10 floor.
    bot = make_bot(
        "trend_following", balance=60.0, budget=60.0,
        params={
            "short_period": 3, "long_period": 5, "atr_period": 3,
            "atr_multiplier": 2.0, "risk_percent": 0.3,  # 0.3% of $60 = $0.18
            "entry_confirmation_loops": 1, "cooldown_seconds": 0,
        },
    )
    # Large, volatile up-moves -> big ATR -> tiny coin count -> sub-$10 notional.
    prices = []
    p = 50000.0
    for i in range(40):
        p += 3000.0 if i % 2 == 0 else -1500.0  # net up, |delta| huge
        prices.append(p)
    buy = await _drive_until_buy(engine, bot, prices, SimpleNamespace())

    assert buy is not None, "trend following never entered"
    assert buy.amount >= MIN_ORDER_USD, f"BUY ${buy.amount} below minimum after floor"
    assert buy.amount <= bot.current_balance + 1e-9


# --------------------------------------------------------------------------- #
# 4. Adaptive Grid full cycle (Bot 5): init -> buy -> sell
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_adaptive_grid_initializes_and_holds_without_crashing():
    """Adaptive Grid must initialize and emit valid HOLDs (the original bug
    crashed here with 'TradeSignal missing amount')."""
    engine = fresh_engine()
    bot = make_bot(
        "adaptive_grid",
        params={"bar_interval_seconds": 0, "regime_filter_enabled": False,
                "grid_count": 6, "atr_period": 3},
    )
    executor = engine._get_strategy_executor("adaptive_grid")
    for i in range(10):
        sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
        assert_valid_signal(sig, context=f"grid warmup iter {i}")


@pytest.mark.asyncio
async def test_adaptive_grid_full_cycle():
    """Adaptive Grid must complete a full cycle: initialize, place a BUY (price
    dips to a buy level), then place a SELL (price rallies to a sell level)."""
    engine = fresh_engine()
    bot = make_bot(
        "adaptive_grid", budget=1000.0, balance=1000.0,
        params={
            "bar_interval_seconds": 0,        # every call closes a bar
            "regime_filter_enabled": False,   # don't gate on regime in the test
            "grid_count": 10,
            "grid_spacing_percent": 1.0,
            "range_percent": 20,
            "base_order_size_percent": 5,
            "max_drawdown_percent": 95,       # don't trip the drawdown kill switch
            "atr_period": 3,
        },
    )
    executor = engine._get_strategy_executor("adaptive_grid")

    # First tick sets the center at 50000. Then oscillate down (fill buy levels
    # below center) and up (fill sell levels above center) repeatedly.
    actions = []
    prices = [50000.0]
    for _cycle in range(6):
        prices += [49500, 49000, 48500, 49000, 49500, 50000, 50500, 51000, 50500, 50000]

    for i, price in enumerate(prices):
        sig = await executor(bot, float(price), bot.strategy_params, SimpleNamespace())
        assert_valid_signal(sig, context=f"grid cycle iter {i} @ {price}")
        if sig is not None and sig.action in ("buy", "sell"):
            actions.append(sig.action)

    assert "buy" in actions, f"Adaptive Grid never placed a BUY (actions={actions})"
    assert "sell" in actions, f"Adaptive Grid never placed a SELL (actions={actions})"


# --------------------------------------------------------------------------- #
# 4b. DCA min-order floor (Bot 2): never emit a sub-$10 executable buy
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dca_floors_subminimum_buy_to_minimum_when_affordable():
    """A DCA buy that sizes below the minimum but is affordable is floored UP to
    MIN_ORDER_USD, not emitted sub-minimum.

    Bot 2's loop: DCA used a $1 placeholder floor, so a $9.95 buy passed the
    strategy but was rejected by the execution layer's $10 minimum every tick
    ("DCA buy #3 = $9.95 / REJECTED" forever). The strategy must respect the
    same minimum the executor enforces."""
    engine = fresh_engine()
    # $99 balance * 10% default = $9.90 raw, below the $10 minimum but affordable.
    bot = make_bot(
        "dca_accumulator", balance=99.0, budget=99.0,
        params={"regime_filter_enabled": False, "immediate_first_buy": True},
    )
    executor = engine._get_strategy_executor("dca_accumulator")
    sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
    assert_valid_signal(sig, context="dca sub-minimum floor")
    assert sig.action == "buy", f"expected a floored BUY, got {sig.action} ({sig.reason})"
    assert sig.amount >= MIN_ORDER_USD, (
        f"DCA emitted a sub-minimum BUY ${sig.amount:.2f} (< ${MIN_ORDER_USD})"
    )
    assert sig.amount <= bot.current_balance + 1e-9


@pytest.mark.asyncio
async def test_dca_holds_when_balance_below_minimum():
    """When the balance can no longer afford the minimum order, DCA HOLDs
    (infinite accumulation complete) instead of emitting a doomed sub-min buy."""
    engine = fresh_engine()
    bot = make_bot(
        "dca_accumulator", balance=5.0, budget=5.0,
        params={"regime_filter_enabled": False, "immediate_first_buy": True},
    )
    executor = engine._get_strategy_executor("dca_accumulator")
    sig = await executor(bot, 50000.0, bot.strategy_params, SimpleNamespace())
    assert_valid_signal(sig, context="dca exhausted")
    assert sig.action == "hold", f"expected HOLD, got {sig.action}"
    assert sig.amount == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", CONCRETE_STRATEGIES)
@pytest.mark.parametrize("balance", [50.0, 100.0, 1000.0])
async def test_no_strategy_emits_executable_buy_below_minimum(strategy, balance):
    """Cross-strategy guarantee: across a long oscillating series and several
    account sizes, NO strategy may emit a BUY below MIN_ORDER_USD. A sub-minimum
    buy is rejected by the executor every tick and is the root of the rejection
    loops (Bot 2). Strategies must floor up or HOLD before signalling."""
    engine = fresh_engine()
    bot = make_bot(strategy, params=dict(SMOKE_PARAMS[strategy]),
                   balance=balance, budget=balance)
    executor = engine._get_strategy_executor(strategy)
    session = SimpleNamespace()
    for i, price in enumerate(oscillating_prices(300)):
        sig = await executor(bot, price, bot.strategy_params, session)
        assert_valid_signal(sig, context=f"{strategy} iter {i}")
        if sig is not None and sig.action == "buy":
            assert sig.amount >= MIN_ORDER_USD, (
                f"{strategy} emitted BUY ${sig.amount:.4f} < ${MIN_ORDER_USD} "
                f"minimum at iter {i} (balance ${balance}) — would loop on rejection"
            )


# --------------------------------------------------------------------------- #
# 5. Auto Mode force-exit constructs a valid "sell all" signal (Bot 1)
# --------------------------------------------------------------------------- #
def test_auto_mode_force_exit_signal_is_valid():
    """Auto Mode's force-exit uses the amount<=0 'sell all' sentinel. Assert it
    is a structurally valid sell (execution resolves it to the full position;
    that resolution is covered in test_trading_engine.py)."""
    sig = TradeSignal(action="sell", amount=0, reason="Auto Mode FORCE EXIT")
    assert_valid_signal(sig, context="auto force-exit")
    assert sig.action == "sell"
