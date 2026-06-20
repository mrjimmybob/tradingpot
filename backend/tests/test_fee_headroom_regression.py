"""Regression tests for fee-headroom bug in strategy position sizing.

Defect (confirmed in production, bot 7 / TestBot9-TF, 2026-06-18):
  Trend Following and Volatility Breakout size buy orders via an ATR-based
  formula that is capped to ``bot.current_balance`` when position_size >> balance
  (which happens whenever ATR is small relative to the balance).  The
  SimulatedExchangeService deducts ``cost + fee`` from its USDT balance, where
  fee = 0.1 % of cost.  A buy sized to 100 % of balance therefore needs
  balance * 1.001, which exceeds the available funds and the exchange returns
  None.  Five identical None returns in a row trip the repeated-rejection
  circuit breaker (MAX_CONSECUTIVE_REJECTIONS = 5) and pause the bot.

Fix: cap all strategy buy amounts at current_balance * _BUY_BALANCE_FRACTION
  (0.998) so the fee (0.1 %) is always covered.  Applies to:
    * _strategy_trend_following
    * _strategy_volatility_breakout
    * _strategy_dca (defensive cap)
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models import BotStatus
from app.services.trading_engine import (
    TradingEngine,
    _BUY_BALANCE_FRACTION,
    MIN_ORDER_USD,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_bot(strategy="trend_following", balance=99.97, budget=100.0, bot_id=1,
              params=None):
    """Minimal bot SimpleNamespace sufficient for strategy unit tests."""
    return SimpleNamespace(
        id=bot_id,
        name="TestBot",
        trading_pair="BTC/USDT",
        strategy=strategy,
        strategy_params=params or {},
        budget=budget,
        current_balance=balance,
        compound_enabled=True,
        is_dry_run=True,
        status=BotStatus.RUNNING,
        total_pnl=balance - budget,
    )


def _engine_no_position():
    """Engine that reports no open position (flat)."""
    engine = TradingEngine()
    engine._get_bot_positions = AsyncMock(return_value=[])
    engine._get_last_order = AsyncMock(return_value=None)
    engine._get_order_count = AsyncMock(return_value=0)
    return engine


# --------------------------------------------------------------------------- #
# Test: constant sanity
# --------------------------------------------------------------------------- #

def test_buy_balance_fraction_leaves_enough_for_fee():
    """_BUY_BALANCE_FRACTION must be strictly less than 1 / (1 + sim_fee_rate).

    The simulated exchange fee rate is 0.001 (0.1 %).  A buy of size X needs
    X * 1.001 in the wallet.  The fraction must satisfy:
        balance * fraction * 1.001 <= balance
    → fraction <= 1 / 1.001 ≈ 0.999001
    """
    SIM_FEE_RATE = 0.001
    balance = 100.0
    buy_amount = balance * _BUY_BALANCE_FRACTION
    total_needed = buy_amount * (1 + SIM_FEE_RATE)
    assert total_needed <= balance, (
        f"fee-adjusted total {total_needed:.4f} > balance {balance:.4f}; "
        f"_BUY_BALANCE_FRACTION={_BUY_BALANCE_FRACTION} is too high"
    )


# --------------------------------------------------------------------------- #
# Trend Following — fee headroom
# --------------------------------------------------------------------------- #

class TestTrendFollowingFeeHeadroom:
    """Position sizing must leave fee headroom when ATR is low (large cap)."""

    @pytest.mark.asyncio
    async def test_buy_amount_capped_with_fee_headroom_when_atr_low(self):
        """With flat prices the cap must be balance*_BUY_BALANCE_FRACTION, not
        balance.  This is the exact production failure mode (bot 7, 2026-06-18):
        ATR≈$1.03 → position_size≈$30,000 → capped to full balance → exchange
        rejected for insufficient funds (cost+fee > balance)."""
        balance = 99.97
        bot = _make_bot("trend_following", balance=balance, params={
            "short_period": 5,
            "long_period": 10,
            "atr_period": 3,
            "atr_multiplier": 2.0,
            "risk_percent": 1,
            "entry_confirmation_loops": 1,
            "exit_confirmation_loops": 2,
            "cooldown_seconds": 0,
        })
        engine = _engine_no_position()

        # Feed price history: long_period prices all identical, then a tiny
        # upward movement so entry conditions (price > EMA long, EMA short >
        # EMA long) are met.  Identical prices make ATR near zero →
        # position_size >> balance → cap is the only sizing constraint.
        base_price = 62_876.0
        prices = [base_price] * 10 + [base_price + 0.05]
        current_price = prices[-1]

        # Pre-seed price history (11 > long_period=10 so the strategy executes)
        engine._price_histories = {bot.id: list(prices[:-1])}

        signal = await engine._strategy_trend_following(
            bot, current_price,
            {"short_period": 5, "long_period": 10, "atr_period": 3,
             "atr_multiplier": 2.0, "risk_percent": 1,
             "entry_confirmation_loops": 1, "exit_confirmation_loops": 2,
             "cooldown_seconds": 0},
            AsyncMock(),
        )

        if signal and signal.action == "buy":
            max_allowed = balance * _BUY_BALANCE_FRACTION
            assert signal.amount <= max_allowed, (
                f"buy_amount ${signal.amount:.4f} exceeds fee-adjusted cap "
                f"${max_allowed:.4f} (balance={balance}); "
                f"this would be rejected by the simulated exchange and trip "
                f"the repeated-rejection circuit breaker"
            )
            # Also verify simulated exchange can fill it
            SIM_FEE = signal.amount * 0.001
            total_cost = signal.amount + SIM_FEE
            assert total_cost <= balance, (
                f"cost+fee ${total_cost:.4f} > balance ${balance:.4f}"
            )

    @pytest.mark.asyncio
    async def test_buy_amount_never_exceeds_fee_adjusted_balance(self):
        """Parameterised sweep: regardless of ATR level, the strategy must never
        emit a buy that the simulated exchange would reject for insufficient
        funds."""
        for atr_level_pct in [0.0, 0.001, 0.01, 0.1, 1.0]:
            balance = 99.97
            base_price = 62_876.0
            # Construct prices that produce the target ATR level
            prices = [base_price] * 10 + [base_price * (1 + atr_level_pct / 100)]
            current_price = prices[-1]

            bot = _make_bot("trend_following", balance=balance, params={
                "short_period": 5, "long_period": 10,
                "atr_period": 3, "atr_multiplier": 2.0,
                "risk_percent": 1, "entry_confirmation_loops": 1,
                "exit_confirmation_loops": 2, "cooldown_seconds": 0,
            })
            engine = _engine_no_position()
            engine._price_histories = {bot.id: list(prices[:-1])}

            signal = await engine._strategy_trend_following(
                bot, current_price,
                {"short_period": 5, "long_period": 10, "atr_period": 3,
                 "atr_multiplier": 2.0, "risk_percent": 1,
                 "entry_confirmation_loops": 1, "exit_confirmation_loops": 2,
                 "cooldown_seconds": 0},
                AsyncMock(),
            )

            if signal and signal.action == "buy":
                max_allowed = balance * _BUY_BALANCE_FRACTION
                assert signal.amount <= max_allowed, (
                    f"atr_pct={atr_level_pct}: buy_amount ${signal.amount:.4f} "
                    f"> fee-adjusted cap ${max_allowed:.4f}"
                )


# --------------------------------------------------------------------------- #
# Volatility Breakout — same defect pattern
# --------------------------------------------------------------------------- #

class TestVolatilityBreakoutFeeHeadroom:
    """Volatility Breakout uses identical cap logic; verify the same fix."""

    @pytest.mark.asyncio
    async def test_buy_amount_capped_with_fee_headroom(self):
        """Volatility Breakout emits a buy on a confirmed breakout.
        With low bar ATR the position_size >> balance; cap must be
        balance*_BUY_BALANCE_FRACTION, not balance."""
        balance = 99.97
        bot = _make_bot("volatility_breakout", balance=balance, params={
            "bar_interval_seconds": 0,
            "bb_period": 5,
            "bb_std": 2.0,
            "atr_period": 3,
            "atr_stop_multiplier": 2.0,
            "risk_percent": 1,
            "compression_percentile": 100,  # always compressed
            "min_compression_bars": 1,
            "cooldown_hours": 0,
            "regime_filter_enabled": False,
            "failed_breakout_bars": 0,
        })
        engine = _engine_no_position()

        base_price = 62_876.0
        # Build 6 bars: 5 nearly-identical (compressing BB), then a big spike
        # to arm the breakout.
        bars = [
            {"open": base_price, "high": base_price + 0.01,
             "low": base_price - 0.01, "close": base_price, "start_ts": None}
            for _ in range(5)
        ]
        breakout_price = base_price + 300  # well above upper BB

        state = {
            "bars": bars,
            "current_bar": None,
            "bb_width_history": [0.00001] * 20,  # compression percentile satisfied
            "atr_history": [0.5],
            "compression_active": True,
            "compression_bars": 5,
            "compression_start": None,
            "breakout_armed": True,
            "entry_price": None,
            "entry_atr": None,
            "highest_price": None,
            "trailing_stop": None,
            "bars_since_entry": 0,
            "last_breakout_attempt": None,
        }
        if not hasattr(engine, "_volatility_breakout_states"):
            engine._volatility_breakout_states = {}
        engine._volatility_breakout_states[bot.id] = state

        signal = await engine._strategy_volatility_breakout(
            bot, breakout_price,
            {"bar_interval_seconds": 0, "bb_period": 5, "bb_std": 2.0,
             "atr_period": 3, "atr_stop_multiplier": 2.0,
             "risk_percent": 1, "compression_percentile": 99,
             "min_compression_bars": 1, "cooldown_hours": 0,
             "regime_filter_enabled": False, "failed_breakout_bars": 0},
            AsyncMock(),
        )

        if signal and signal.action == "buy":
            max_allowed = balance * _BUY_BALANCE_FRACTION
            assert signal.amount <= max_allowed, (
                f"Volatility Breakout buy_amount ${signal.amount:.4f} exceeds "
                f"fee-adjusted cap ${max_allowed:.4f} (balance={balance})"
            )
            SIM_FEE = signal.amount * 0.001
            total_cost = signal.amount + SIM_FEE
            assert total_cost <= balance, (
                f"cost+fee ${total_cost:.4f} > balance ${balance:.4f}; "
                f"simulated exchange would reject this order"
            )


# --------------------------------------------------------------------------- #
# DCA — defensive cap
# --------------------------------------------------------------------------- #

class TestDCAFeeHeadroom:
    """DCA's defensive cap must also leave fee headroom."""

    @pytest.mark.asyncio
    async def test_dca_defensive_cap_uses_fee_fraction(self):
        """When amount_usd > balance the DCA defensive cap must limit to
        balance * _BUY_BALANCE_FRACTION so the exchange fee doesn't push the
        total over the available funds."""
        balance = 99.97
        bot = _make_bot("dca_accumulator", balance=balance, params={
            "amount_usd": 200.0,          # deliberately > balance
            "interval_minutes": 0,         # no wait between buys
            "immediate_first_buy": True,
            "regime_filter_enabled": False,
        })
        engine = _engine_no_position()

        signal = await engine._strategy_dca(
            bot, 62_876.0,
            {"amount_usd": 200.0, "interval_minutes": 0,
             "immediate_first_buy": True, "regime_filter_enabled": False},
            AsyncMock(),
        )

        assert signal is not None
        assert signal.action == "buy"
        max_allowed = balance * _BUY_BALANCE_FRACTION
        assert signal.amount <= max_allowed, (
            f"DCA buy_amount ${signal.amount:.4f} exceeds fee-adjusted cap "
            f"${max_allowed:.4f}; simulated exchange would reject (cost+fee > balance)"
        )


# --------------------------------------------------------------------------- #
# Simulated exchange — confirm a fee-headroom-sized buy succeeds
# --------------------------------------------------------------------------- #

class TestSimulatedExchangeAcceptsFeeHeadroomOrder:
    """End-to-end: a buy sized to balance*_BUY_BALANCE_FRACTION must fill."""

    @pytest.mark.asyncio
    async def test_fee_adjusted_buy_accepted(self):
        """The simulated exchange must accept an order sized with fee headroom."""
        from unittest.mock import patch, AsyncMock as AM
        from app.services.exchange import SimulatedExchangeService, OrderSide

        balance = 99.97
        sim = SimulatedExchangeService(initial_balance=balance)

        # Patch get_ticker so we don't need a live exchange
        price = 62_876.0
        fake_ticker = SimpleNamespace(last=price, bid=price - 0.5, ask=price + 0.5)

        with patch.object(sim, "get_ticker", new=AM(return_value=fake_ticker)):
            buy_usd = balance * _BUY_BALANCE_FRACTION
            buy_base = buy_usd / fake_ticker.ask  # same formula as _execute_trade

            order = await sim.place_market_order("BTC/USDT", OrderSide.BUY, buy_base)

        assert order is not None, (
            "SimulatedExchangeService rejected a fee-headroom-sized buy; "
            "the order was still over-budget despite _BUY_BALANCE_FRACTION"
        )

    @pytest.mark.asyncio
    async def test_full_balance_buy_rejected(self):
        """Confirm the pre-fix behaviour: a buy at exactly balance IS rejected.

        This test documents the root cause — it should PASS (i.e. return None)
        as long as the simulated exchange deducts cost+fee from the balance.
        If this test starts FAILING the exchange fee behaviour has changed and
        the headroom constant must be re-evaluated.
        """
        from unittest.mock import patch, AsyncMock as AM
        from app.services.exchange import SimulatedExchangeService, OrderSide

        balance = 99.97
        sim = SimulatedExchangeService(initial_balance=balance)

        price = 62_876.0
        fake_ticker = SimpleNamespace(last=price, bid=price - 0.5, ask=price + 0.5)

        with patch.object(sim, "get_ticker", new=AM(return_value=fake_ticker)):
            # Buy at exactly the full balance — fee pushes total over available.
            buy_base = balance / fake_ticker.ask
            order = await sim.place_market_order("BTC/USDT", OrderSide.BUY, buy_base)

        assert order is None, (
            "Expected SimulatedExchangeService to reject a 100%-balance buy "
            "(cost + 0.1% fee > balance), but it succeeded.  If the fee model "
            "changed, re-evaluate _BUY_BALANCE_FRACTION."
        )


# --------------------------------------------------------------------------- #
# Rejection reason propagation
# --------------------------------------------------------------------------- #

class TestRejectionReasonPropagation:
    """Exchange rejection reason must survive all the way to diagnostics."""

    @pytest.mark.asyncio
    async def test_last_order_error_set_on_insufficient_balance(self):
        """SimulatedExchangeService.last_order_error must be set when a BUY is
        rejected for insufficient funds, not silently swallowed."""
        from unittest.mock import patch, AsyncMock as AM
        from app.services.exchange import SimulatedExchangeService, OrderSide

        balance = 50.0
        sim = SimulatedExchangeService(initial_balance=balance)

        price = 62_876.0
        fake_ticker = SimpleNamespace(last=price, bid=price - 0.5, ask=price + 0.5)

        with patch.object(sim, "get_ticker", new=AM(return_value=fake_ticker)):
            # Order priced at the full balance — will be rejected by fee check.
            buy_base = balance / fake_ticker.ask
            order = await sim.place_market_order("BTC/USDT", OrderSide.BUY, buy_base)

        assert order is None, "Expected rejection but order succeeded"
        assert sim.last_order_error is not None, (
            "last_order_error must be set when the exchange rejects an order; "
            "without it the caller cannot surface the real reason to the user"
        )
        assert "Insufficient simulated balance" in sim.last_order_error, (
            f"Expected 'Insufficient simulated balance' in last_order_error, "
            f"got: {sim.last_order_error!r}"
        )

    @pytest.mark.asyncio
    async def test_last_order_error_cleared_on_success(self):
        """last_order_error must be None after a successful order so a stale
        error from a previous rejection cannot bleed into a later success."""
        from unittest.mock import patch, AsyncMock as AM
        from app.services.exchange import SimulatedExchangeService, OrderSide

        balance = 100.0
        sim = SimulatedExchangeService(initial_balance=balance)
        sim.last_order_error = "stale error from previous run"

        price = 62_876.0
        fake_ticker = SimpleNamespace(last=price, bid=price - 0.5, ask=price + 0.5)

        with patch.object(sim, "get_ticker", new=AM(return_value=fake_ticker)):
            buy_usd = balance * _BUY_BALANCE_FRACTION
            buy_base = buy_usd / fake_ticker.ask
            order = await sim.place_market_order("BTC/USDT", OrderSide.BUY, buy_base)

        assert order is not None, "Expected success but order was rejected"
        assert sim.last_order_error is None, (
            f"last_order_error should be None after a successful order, "
            f"got: {sim.last_order_error!r}"
        )

    @pytest.mark.asyncio
    async def test_rejection_reason_contains_amounts(self):
        """The rejection message must include cost and available balance figures
        so an operator can diagnose the mismatch from the UI without log diving."""
        from unittest.mock import patch, AsyncMock as AM
        from app.services.exchange import SimulatedExchangeService, OrderSide

        balance = 10.0
        sim = SimulatedExchangeService(initial_balance=balance)

        price = 62_876.0
        fake_ticker = SimpleNamespace(last=price, bid=price - 0.5, ask=price + 0.5)

        with patch.object(sim, "get_ticker", new=AM(return_value=fake_ticker)):
            buy_base = balance / fake_ticker.ask  # will be rejected
            await sim.place_market_order("BTC/USDT", OrderSide.BUY, buy_base)

        assert sim.last_order_error is not None
        # Must contain numeric context (dollar amounts) so the operator knows the gap
        assert "$" in sim.last_order_error or "need" in sim.last_order_error, (
            f"Rejection message lacks numeric context: {sim.last_order_error!r}"
        )
