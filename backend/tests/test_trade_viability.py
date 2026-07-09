"""Phase 2 execution-quality tests.

Covers the four required scenarios:
  a) Trade rejected when expected move cannot cover round-trip fees.
  b) Trade allowed when expected move clears round-trip fees.
  c) Exchange fee is read from bot.exchange_fee — never silently zero.
  d) Trend Following uses bar-based ATR, not microscopic tick ATR.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.trading_engine import TradingEngine, TradeSignal, BotStatus


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _engine() -> TradingEngine:
    return TradingEngine()


def _bot(
    balance: float = 1000.0,
    bot_id: int = 1,
    exchange_fee: float = 0.1,
) -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "mean_reversion"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    bot.exchange_fee = exchange_fee
    return bot


def _no_positions(engine: TradingEngine) -> None:
    engine._get_bot_positions = AsyncMock(return_value=[])


def _bar(close: float, spread: float = 10.0) -> dict:
    """Build an OHLC bar with the given close price and a fixed H-L spread.

    The H-L spread keeps bar ATR non-zero without affecting Bollinger Band
    calculations (which use only close prices).
    """
    return {
        "open": close,
        "high": close + spread / 2,
        "low": close - spread / 2,
        "close": close,
        "start_ts": datetime.utcnow(),
    }


def _mr_state(bars: list) -> dict:
    """Return a clean MR state dict with pre-built bars and no open position."""
    return {
        "bars": bars,
        "current_bar": None,
        "entry_price": None,
        "entry_atr": None,
        "hard_stop": None,
        "bars_since_entry": 0,
        "last_exit_time": None,
    }


def _tf_state(tf_bars: list) -> dict:
    """Return a TF state dict with pre-built bars and no open position."""
    return {
        "trailing_stop": None,
        "highest_price": None,
        "entry_atr": None,
        "entry_time": None,
        "last_exit_time": None,
        "entry_confirmation_count": 0,
        "exit_confirmation_count": 0,
        "tf_bars": tf_bars,
        "tf_current_bar": None,
    }


# Mean Reversion params: disable regime filter so we only test the viability
# logic; keep everything else at defaults.
_MR_PARAMS = {
    "period": 20,
    "std_mult": 1.8,
    "atr_period": 14,
    "order_size_percent": 0.95,
    "regime_filter_enabled": False,
}

# Trend Following params: short periods and 1-loop confirmation so the strategy
# can reach a BUY decision in a single call during tests.
_TF_PARAMS = {
    "short_period": 10,
    "long_period": 20,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "risk_percent": 1.0,
    "entry_confirmation_loops": 1,
    # bar_interval_seconds=0 ensures every call closes the in-progress bar so
    # we can test bar-count boundaries without real-time waiting.
    "bar_interval_seconds": 0,
}


# ---------------------------------------------------------------------------
# Test a: trade rejected when fees exceed expected profit
#
# Setup: 20 flat bars (std_dev ≈ 0) so the Bollinger Band collapses —
# lower_band ≈ SMA ≈ close, giving expected_move_pct ≈ 0 %.
# That is well below the 0.25 % fee hurdle, so the local MR pre-check must
# return HOLD before ever mutating entry state.
# ---------------------------------------------------------------------------

class TestTradeRejectedFeesTooHigh:
    @pytest.mark.asyncio
    async def test_mr_hold_when_band_too_narrow(self):
        """MR returns HOLD and never sets entry_price when band < fee hurdle."""
        engine = _engine()
        bot = _bot(exchange_fee=0.1)
        _no_positions(engine)
        session = AsyncMock()

        # 20 flat bars: std_dev = 0, lower_band = SMA = 64 000.
        # last_bar_close = 64 000 = lower_band  → entry condition is True,
        # but expected_move = 0 < 0.25 % → must be rejected.
        bars = [_bar(64_000.0) for _ in range(20)]
        engine._mean_reversion_states = {bot.id: _mr_state(bars)}

        signal = await engine._strategy_mean_reversion(
            bot, 64_000.0, _MR_PARAMS, session
        )

        assert signal.action == "hold"
        assert "narrow" in signal.reason.lower() or "narrow" in signal.reason.lower()
        # State must NOT show an entry — no DB position was opened.
        state_after = engine._mean_reversion_states[bot.id]
        assert state_after["entry_price"] is None, (
            "entry_price was mutated even though the trade should have been rejected"
        )


# ---------------------------------------------------------------------------
# Test b: trade allowed when expected profit clears costs
#
# Setup: 19 bars at $64 000 + 1 bar at $63 740 (the last / lowest bar).
# SMA ≈ $63 987, lower_band ≈ $63 885; last close ($63 740) is below the
# band, giving expected_move ≈ 0.39 % — above the 0.25 % fee hurdle.
# ---------------------------------------------------------------------------

class TestTradeAllowedProfitClears:
    @pytest.mark.asyncio
    async def test_mr_buy_when_band_wide_enough(self):
        """MR returns BUY with expected_move_pct set when band clears the fee threshold."""
        engine = _engine()
        bot = _bot(exchange_fee=0.1)
        _no_positions(engine)
        session = AsyncMock()

        bars = [_bar(64_000.0) for _ in range(19)] + [_bar(63_740.0)]
        engine._mean_reversion_states = {bot.id: _mr_state(bars)}

        signal = await engine._strategy_mean_reversion(
            bot, 63_740.0, _MR_PARAMS, session
        )

        assert signal.action == "buy", (
            f"Expected BUY but got {signal.action!r}: {signal.reason}"
        )
        assert signal.expected_move_pct is not None
        # expected_move must exceed the 0.25 % minimum viable threshold.
        assert signal.expected_move_pct > 0.0025, (
            f"expected_move_pct {signal.expected_move_pct:.4%} is below the 0.25 % fee threshold"
        )


# ---------------------------------------------------------------------------
# Test c: exchange fee is never silently zero
#
# The same MR setup from test b (expected_move ≈ 0.39 %) passes the gate at
# exchange_fee=0.1 % (min_viable = 0.25 %) but fails at exchange_fee=0.5 %
# (min_viable = 1.05 %).  This proves bot.exchange_fee is read — not assumed
# to be 0.0.
# ---------------------------------------------------------------------------

class TestExchangeFeeNotSilentlyZero:
    @pytest.mark.asyncio
    async def test_low_fee_allows_trade(self):
        """With fee=0.1 %, a 0.39 % expected move is above the 0.25 % threshold."""
        engine = _engine()
        bot = _bot(exchange_fee=0.1)
        _no_positions(engine)
        session = AsyncMock()

        bars = [_bar(64_000.0) for _ in range(19)] + [_bar(63_740.0)]
        engine._mean_reversion_states = {bot.id: _mr_state(bars)}

        signal = await engine._strategy_mean_reversion(
            bot, 63_740.0, _MR_PARAMS, session
        )
        assert signal.action == "buy"

    @pytest.mark.asyncio
    async def test_high_fee_rejects_same_trade(self):
        """With fee=0.5 %, the same 0.39 % expected move falls below the 1.05 % threshold."""
        engine = _engine()
        # Higher fee → min_viable = 2×0.005 + 0.0005 = 1.05 % >> 0.39 %
        bot = _bot(exchange_fee=0.5)
        _no_positions(engine)
        session = AsyncMock()

        bars = [_bar(64_000.0) for _ in range(19)] + [_bar(63_740.0)]
        engine._mean_reversion_states = {bot.id: _mr_state(bars)}

        signal = await engine._strategy_mean_reversion(
            bot, 63_740.0, _MR_PARAMS, session
        )
        assert signal.action == "hold", (
            "High-fee bot should have had the trade rejected, but got BUY"
        )
        assert "narrow" in signal.reason.lower()


# ---------------------------------------------------------------------------
# Test d: Trend Following uses bar-based ATR (not microscopic tick ATR)
#
# Two sub-tests:
#   d1 — warmup: when fewer than atr_period bars have completed, TF must hold.
#   d2 — ready:  when 14+ large-range bars are available, TF generates a BUY
#                with expected_move_pct well above the fee threshold.
# ---------------------------------------------------------------------------

def _tf_price_history() -> list:
    """Return a 20-tick price history that satisfies EMA conditions.

    Prices are flat at $64 000 for the first 10 ticks, then step up to $64 100.
    EMA(10) weights the recent plateau more heavily than EMA(20), so
    EMA_short > EMA_long.  The current_price ($64 200) is above EMA_long.
    """
    return [64_000.0] * 10 + [64_100.0] * 10


class TestTrendFollowingBarATR:
    @pytest.mark.asyncio
    async def test_d1_warmup_floor_prevents_microscopic_stop(self):
        """During bar ATR warmup (no completed bars), TF uses a fee-coverage floor ATR.

        The floor ensures expected_move_pct is above the 0.25 % fee hurdle even
        before any bar data has been collected.  This is the core invariant: TF
        never generates a trade where round-trip fees exceed the expected profit,
        regardless of whether bar ATR is ready or not.
        """
        engine = _engine()
        bot = _bot(exchange_fee=0.1)
        _no_positions(engine)
        session = AsyncMock()

        # Pre-populate EMA data but NO completed bars yet — forces floor ATR path.
        engine._price_histories = {bot.id: _tf_price_history()}
        engine._trend_states = {bot.id: _tf_state(tf_bars=[])}

        signal = await engine._strategy_trend_following(
            bot, 64_200.0, _TF_PARAMS, session
        )

        # Entry conditions are met (rising prices), so a BUY should be issued.
        # Floor ATR = price × (2 × fee + safety) ≈ $64 200 × 0.0025 = $160.5.
        # expected_move_pct = ($160.5 × 2) / $64 200 ≈ 0.5 % > 0.25 % threshold.
        assert signal.action == "buy", (
            f"Expected BUY with floor ATR during warmup, got {signal.action!r}: {signal.reason}"
        )
        assert signal.expected_move_pct is not None
        assert signal.expected_move_pct > 0.0025, (
            f"Floor ATR gave expected_move_pct {signal.expected_move_pct:.4%} — "
            f"still below the 0.25 % fee threshold (floor not applied correctly)"
        )

    @pytest.mark.asyncio
    async def test_d2_bar_atr_produces_viable_stop(self):
        """TF generates a BUY with expected_move_pct well above the fee hurdle.

        Pre-populate 14 completed bars each with a $200 H-L range.  The bar
        ATR will be ~$186 (the 15th bar that completes this tick has H-L ≈ $0,
        pulling the mean down slightly).  expected_move_pct ≈ 0.57 %, far above
        the 0.25 % fee threshold.  By contrast, tick ATR on this flat price
        series would be ~$5, giving expected_move_pct ≈ 0.016 % — well below
        the threshold and the root cause of the original bug.
        """
        engine = _engine()
        bot = _bot(exchange_fee=0.1)
        _no_positions(engine)
        session = AsyncMock()

        # 14 bars each with H-L = $200 (substantial real-candle volatility).
        large_bars = [
            {
                "high": 64_200.0,
                "low": 64_000.0,
                "close": 64_100.0,
                "start_ts": datetime.utcnow(),
            }
            for _ in range(14)
        ]

        engine._price_histories = {bot.id: _tf_price_history()}
        # Set entry_confirmation_count to 0 — with loops=1 in params, the first
        # conditions-met tick will increment to 1 and immediately trigger entry.
        engine._trend_states = {bot.id: _tf_state(tf_bars=large_bars)}

        signal = await engine._strategy_trend_following(
            bot, 64_200.0, _TF_PARAMS, session
        )

        assert signal.action == "buy", (
            f"Expected BUY after bar ATR warmup, got {signal.action!r}: {signal.reason}"
        )
        assert signal.expected_move_pct is not None
        # Bar ATR (~$186) × multiplier (2) / price ($64 200) ≈ 0.58 %,
        # which is above the 0.25 % fee minimum viable threshold.
        assert signal.expected_move_pct > 0.0025, (
            f"expected_move_pct {signal.expected_move_pct:.4%} is below the 0.25 % "
            f"fee threshold — stop may be microscopic (tick ATR bug not fixed)"
        )


# ---------------------------------------------------------------------------
# Shared helpers for _execute_trade integration tests (tests A, B, C)
#
# _execute_trade calls PortfolioRiskService and StrategyCapacityService;
# we patch them to always pass so the viability gate at Step 5.5 is the
# first check that can reject the signal.
# ---------------------------------------------------------------------------

def _passing_check(adjusted_amount=None):
    """Return a MagicMock that looks like a passing risk/capacity check result."""
    check = MagicMock()
    check.ok = True
    check.action = None
    check.adjusted_amount = adjusted_amount
    check.violated_cap = None
    check.details = ""
    check.reason = ""
    return check


def _execute_trade_bot(exchange_fee: float = 0.1, balance: float = 1_000.0) -> MagicMock:
    """Minimal bot mock for _execute_trade tests."""
    bot = MagicMock()
    bot.id = 999
    bot.trading_pair = "BTC/USDT"
    bot.strategy = "mean_reversion"
    bot.current_balance = balance
    bot.exchange_fee = exchange_fee
    bot.budget = balance
    bot.status = BotStatus.RUNNING
    bot.stop_loss_percent = None
    bot.stop_loss_absolute = None
    bot.drawdown_limit_percent = None
    bot.drawdown_limit_absolute = None
    bot.daily_loss_limit = None
    bot.weekly_loss_limit = None
    return bot


async def _run_execute_trade(
    signal: TradeSignal,
    exchange_fee: float = 0.1,
    current_price: float = 64_000.0,
):
    """Run _execute_trade with portfolio/capacity checks patched to pass."""
    engine = TradingEngine()
    engine._record_trade_outcome = AsyncMock()

    bot = _execute_trade_bot(exchange_fee=exchange_fee)
    exchange = MagicMock()
    session = AsyncMock()

    passing = _passing_check()

    with patch("app.services.trading_engine.PortfolioRiskService") as mock_prs, \
         patch("app.services.trading_engine.StrategyCapacityService") as mock_scs:
        mock_prs.return_value.check_portfolio_risk = AsyncMock(return_value=passing)
        mock_scs.return_value.check_capacity_for_trade = AsyncMock(return_value=passing)
        return await engine._execute_trade(bot, exchange, signal, current_price, session)


# ---------------------------------------------------------------------------
# Test A: _execute_trade rejects BUY when expected_move_pct is None
# ---------------------------------------------------------------------------

class TestExecuteTradeRejectsNoneExpectedMove:
    @pytest.mark.asyncio
    async def test_A_rejects_buy_when_expected_move_pct_is_none(self):
        """_execute_trade must reject a BUY signal carrying expected_move_pct=None.

        Fail-closed gate: a strategy that cannot estimate its expected move
        must not be allowed to reach the exchange. The gate must surface this
        as a diagnostic reason rather than silently proceeding.
        """
        signal = TradeSignal(
            action="buy",
            amount=500.0,
            order_type="market",
            reason="test signal — no expected move",
            expected_move_pct=None,  # ← the critical None
        )
        result = await _run_execute_trade(signal, exchange_fee=0.1)
        assert result is None, (
            "_execute_trade should return None (rejected) when expected_move_pct is None"
        )


# ---------------------------------------------------------------------------
# Test B: _execute_trade rejects BUY when expected_move_pct < fees + margin
# ---------------------------------------------------------------------------

class TestExecuteTradeRejectsLowExpectedMove:
    @pytest.mark.asyncio
    async def test_B_rejects_buy_when_expected_move_below_fees(self):
        """_execute_trade rejects when expected_move_pct cannot cover round-trip fees.

        At 0.1 % fee, round-trip = 0.2 % + 0.05 % safety = 0.25 % min.
        A signal with expected_move = 0.1 % must be rejected.
        """
        signal = TradeSignal(
            action="buy",
            amount=500.0,
            order_type="market",
            reason="test signal — move below fees",
            expected_move_pct=0.001,  # 0.1 % — below the 0.25 % threshold
        )
        result = await _run_execute_trade(signal, exchange_fee=0.1)
        assert result is None, (
            "_execute_trade should return None when expected_move_pct < round-trip fees + margin"
        )


# ---------------------------------------------------------------------------
# Test C: _execute_trade accepts BUY when expected_move_pct > required cost
# ---------------------------------------------------------------------------

class TestExecuteTradeAcceptsViableTrade:
    @pytest.mark.asyncio
    async def test_C_accepts_buy_when_expected_move_clears_fees(self):
        """_execute_trade proceeds when expected_move_pct comfortably exceeds fees.

        At 0.1 % fee, min_viable ≈ 0.25 %.  A signal with expected_move = 1.0 %
        must not be blocked by the viability gate (other guards may still apply,
        but the gate itself must pass).

        We verify that _execute_trade does NOT return None due to the gate.
        The signal may still be routed to the exchange mock (which returns None
        for an unimplemented order) — we only assert the gate did not block it.
        """
        sentinel_order = MagicMock()
        sentinel_order.id = "filled"

        exchange = MagicMock()
        exchange.place_market_order = AsyncMock(return_value=sentinel_order)

        engine = TradingEngine()
        engine._record_trade_outcome = AsyncMock()

        bot = _execute_trade_bot(exchange_fee=0.1)
        session = AsyncMock()

        signal = TradeSignal(
            action="buy",
            amount=500.0,
            order_type="market",
            reason="test signal — viable move",
            expected_move_pct=0.01,  # 1.0 % — well above the 0.25 % threshold
        )

        passing = _passing_check()
        with patch("app.services.trading_engine.PortfolioRiskService") as mock_prs, \
             patch("app.services.trading_engine.StrategyCapacityService") as mock_scs:
            mock_prs.return_value.check_portfolio_risk = AsyncMock(return_value=passing)
            mock_scs.return_value.check_capacity_for_trade = AsyncMock(return_value=passing)
            try:
                await engine._execute_trade(bot, exchange, signal, 64_000.0, session)
            except Exception:
                # Post-gate execution may fail due to minimal session/order mocking;
                # that is fine — we only care that the gate itself did not block.
                pass

        # Viability gate passed → execution reached the exchange before any failure.
        assert exchange.place_market_order.called, (
            "exchange.place_market_order was never called — viability gate may have "
            "incorrectly blocked a signal with expected_move_pct well above fees"
        )


# ---------------------------------------------------------------------------
# Test D: each strategy BUY either supplies expected_move_pct or is blocked
# ---------------------------------------------------------------------------

# ---- Adaptive Grid ----

class TestAdaptiveGridProvidesExpectedMove:
    @pytest.mark.asyncio
    async def test_D_grid_buy_carries_expected_move_pct(self):
        """Adaptive Grid BUY signals must include a numeric expected_move_pct.

        The grid's spacing is guaranteed to be >= min_spacing_usd (fee-covering),
        so expected_move_pct = grid_spacing / price will always be a non-None float.
        """
        engine = TradingEngine()
        engine._get_bot_positions = AsyncMock(return_value=[])
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        bot = MagicMock()
        bot.id = 901
        bot.strategy = "adaptive_grid"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 5_000.0
        bot.budget = 5_000.0
        bot.exchange_fee = 0.1

        center = 64_000.0
        # Pre-populate 15 bars so ATR is ready (range $200 each → spacing $160)
        bars = [
            {"open": center, "high": center + 100, "low": center - 100,
             "close": center, "start_ts": datetime.utcnow() - timedelta(minutes=20 - i)}
            for i in range(15)
        ]
        engine._grid_states = {
            bot.id: {
                "initialized": True, "center_price": center,
                "initial_capital": 5_000.0, "virtual_cash": 5_000.0,
                "virtual_crypto": 0.0, "grid_levels": {},
                "last_bar_close_time": None, "current_bar": None,
                "completed_bars": bars, "last_order_bar": None,
                "peak_portfolio_value": 5_000.0,
                "last_recenter_time": datetime.utcnow() - timedelta(hours=3),
                "lifetime_return_pct": 0.0, "lifetime_max_drawdown_pct": 0.0,
                "last_kill_switch_time": None, "kill_switch_count": 0,
                "atr_at_recenter": None, "total_trades": 0,
            }
        }

        # Tick at center to open a bar, then advance its timestamp so it closes
        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            await engine._strategy_grid(bot, center, {}, session)
            state = engine._grid_states[bot.id]
            if state.get("current_bar"):
                state["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
            # Trigger price 0.35% below center to cross the L1 buy level
            trigger = center * 0.9965
            signal = await engine._strategy_grid(bot, trigger, {}, session)

        assert signal is not None
        if signal.action == "buy":
            assert isinstance(signal.expected_move_pct, float), (
                "Grid BUY signal must carry a numeric expected_move_pct"
            )
            assert signal.expected_move_pct > 0.0, "expected_move_pct must be positive"


# ---- Volatility Breakout ----

class TestVolatilityBreakoutProvidesExpectedMove:
    @pytest.mark.asyncio
    async def test_D_vb_buy_carries_expected_move_pct(self):
        """Volatility Breakout BUY must include expected_move_pct = upper_gap / price."""
        engine = TradingEngine()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock(
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        ))

        bot = MagicMock()
        bot.id = 902
        bot.strategy = "volatility_breakout"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 1_000.0
        bot.budget = 1_000.0
        bot.exchange_fee = 0.1

        base_price = 64_000.0
        # 24 flat bars then 1 breakout bar (well above upper band)
        tight_bars = [
            {"open": base_price, "high": base_price + 2, "low": base_price - 2,
             "close": base_price,
             "start_ts": datetime.utcnow() - timedelta(minutes=30 - i)}
            for i in range(24)
        ]
        breakout_close = base_price + 1_000.0  # strong breakout: ~1.56% above upper band
        tight_bars.append({
            "open": base_price, "high": breakout_close, "low": base_price - 2,
            "close": breakout_close,
            "start_ts": datetime.utcnow() - timedelta(minutes=1),
        })

        engine._volatility_breakout_states = {
            bot.id: {
                "bars": tight_bars, "current_bar": None,
                "bb_width_history": [0.01] * 80 + [0.0002] * 5,
                "atr_history": [10.0] * 20,
                "compression_active": False, "compression_bars": 0,
                "compression_start": None, "breakout_armed": True,
                "entry_price": None, "entry_atr": None,
                "highest_price": None, "trailing_stop": None,
                "bars_since_entry": 0, "last_breakout_attempt": None,
            }
        }

        with patch.object(engine, "_get_bot_positions", new=AsyncMock(return_value=[])):
            signal = await engine._strategy_volatility_breakout(
                bot, breakout_close, {}, session
            )

        assert signal is not None
        if signal.action == "buy":
            assert isinstance(signal.expected_move_pct, float), (
                "VB BUY signal must carry a numeric expected_move_pct"
            )
            assert signal.expected_move_pct > 0.0025, (
                f"VB expected_move_pct {signal.expected_move_pct:.4%} is below the 0.25% fee threshold"
            )


# ---- Dip Recovery ----

class TestDipRecoveryProvidesExpectedMove:
    @pytest.mark.asyncio
    async def test_D_dip_recovery_buy_carries_expected_move_pct(self):
        """Dip Recovery BUY must include expected_move_pct = (atr × tp_mult) / price."""
        engine = TradingEngine()
        engine._get_bot_positions = AsyncMock(return_value=[])

        bot = MagicMock()
        bot.id = 903
        bot.strategy = "dip_recovery"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 1_000.0
        bot.budget = 1_000.0
        bot.exchange_fee = 0.1
        bot.started_at = datetime.utcnow() - timedelta(hours=2)

        # Drive through: warmup → decline → bounce to trigger BUY
        warmup = [100.0] * 15
        decline = [98.0, 96.0, 94.0, 92.0, 90.0, 89.5, 89.0]
        bounce = [89.3, 89.6, 90.0, 90.4, 90.8, 91.2, 91.6, 92.0]

        signal = None
        for p in warmup + decline + bounce:
            signal = await engine._strategy_dip_recovery(bot, p, {}, AsyncMock())
            if signal.action == "buy":
                break

        assert signal is not None
        if signal.action == "buy":
            assert isinstance(signal.expected_move_pct, float), (
                "Dip Recovery BUY must carry a numeric expected_move_pct"
            )
            assert signal.expected_move_pct > 0.0025, (
                f"DR expected_move_pct {signal.expected_move_pct:.4%} is below the 0.25% threshold"
            )


# ---- DCA — accumulation strategy with is_accumulation=True ----

class TestDCAHonestlyReportsNoExpectedMove:
    @pytest.mark.asyncio
    async def test_D_dca_buy_has_none_expected_move(self):
        """DCA has no profit-target model: expected_move_pct must stay None.

        DCA is an accumulation strategy: it sets is_accumulation=True on its
        BUY signals so the viability gate applies a fee sanity check instead of
        requiring a directional edge estimate. expected_move_pct remains None
        (no fake value), and the gate passes the trade through.
        """
        engine = TradingEngine()
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._get_last_order = AsyncMock(return_value=None)
        engine._get_order_count = AsyncMock(return_value=0)

        bot = MagicMock()
        bot.id = 904
        bot.strategy = "dca_accumulator"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 1_000.0
        bot.budget = 1_000.0
        bot.exchange_fee = 0.1
        bot.started_at = datetime.utcnow() - timedelta(hours=2)

        params = {
            "amount_usd": 100.0,
            "interval_minutes": 0,
            "immediate_first_buy": True,
            "regime_filter_enabled": False,
        }
        signal = await engine._strategy_dca(bot, 64_000.0, params, AsyncMock())

        assert signal is not None
        assert signal.action == "buy", "DCA should still generate a BUY signal"
        assert signal.expected_move_pct is None, (
            "DCA must NOT fake expected_move_pct — it has no profit-target model."
        )
        assert signal.is_accumulation is True, (
            "DCA must set is_accumulation=True so the gate uses fee-sanity check, "
            "not the directional edge check."
        )


# ---- Funding Carry — accumulation strategy with is_accumulation=True ----

class TestFundingCarryHonestlyReportsNoExpectedMove:
    @pytest.mark.asyncio
    async def test_D_funding_carry_buy_has_none_expected_move(self):
        """Funding Carry spots an opportunity via funding rate, not a price target.

        FC sets is_accumulation=True: it builds a spot long position when conditions
        are favourable, without a per-trade price-move target. expected_move_pct
        remains None (no fake value). The gate uses the accumulation fee check.
        """
        engine = TradingEngine()
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._get_funding_signal = AsyncMock(return_value=0.0001)  # in band

        bot = MagicMock()
        bot.id = 905
        bot.strategy = "funding_carry"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 1_000.0
        bot.budget = 1_000.0
        bot.exchange_fee = 0.1
        bot.started_at = datetime.utcnow() - timedelta(hours=2)

        # Pre-populate enough price history for a trend_up regime
        engine._price_histories = {bot.id: [100.0 * (1 + 0.001 * i) for i in range(100)]}

        params = {
            "min_funding_rate": -0.001,
            "max_funding_rate": 0.001,
            "allowed_regimes": ["trend_up", "trend_flat"],
            "regime_filter_enabled": False,
        }
        signal = await engine._strategy_funding_carry(bot, 110.0, params, AsyncMock())

        if signal is not None and signal.action == "buy":
            assert signal.expected_move_pct is None, (
                "Funding Carry must NOT fake expected_move_pct — it has no price target."
            )
            assert signal.is_accumulation is True, (
                "Funding Carry must set is_accumulation=True so the gate applies "
                "fee-sanity check instead of requiring a directional edge estimate."
            )


# ---------------------------------------------------------------------------
# Test E: recovery paper trading uses bot.exchange_fee (not hardcoded 0.1%)
# ---------------------------------------------------------------------------

class TestPaperTradeFeeUsesBot:
    @pytest.mark.asyncio
    async def test_E_paper_trade_fee_matches_bot_exchange_fee(self):
        """_process_paper_trade must compute fees from bot.exchange_fee.

        Set bot.exchange_fee = 0.5 % (5× the default). With a 10 % gross gain on
        $100 notional, the fee-inclusive P&L must be:
            gross = $10.00
            fees  = $100 × 0.005 × 2 = $1.00
            net   = $9.00
        If fees were still hardcoded to 0.1 %, net would be $9.80, exposing the bug.
        """
        engine = TradingEngine()
        bot = MagicMock()
        bot.id = 906
        bot.trading_pair = "BTC/USDT"
        bot.exchange_fee = 0.5  # 0.5% — 5× the default

        engine._recovery_states = {
            906: {
                "paper_trades": [],
                "consecutive_paper_wins": 0,
                "paper_position": {
                    "entry_price": 100.0,
                    "amount_usd": 100.0,
                    "trading_pair": "BTC/USDT",
                    "entered_at": datetime.utcnow().isoformat(),
                },
            }
        }

        sell_signal = TradeSignal(action="sell", amount=100.0)
        session = AsyncMock()

        # Price goes up 10 % → exit at 110.0
        await engine._process_paper_trade(bot, 906, sell_signal, 110.0, session)

        trade = engine._recovery_states[906]["paper_trades"][0]
        # expected: gross = 100 * (110/100 - 1) = 10; fees = 100 * 0.005 * 2 = 1; net = 9
        assert abs(trade["gain_loss_usd"] - 9.0) < 0.01, (
            f"Paper trade P&L was {trade['gain_loss_usd']:.4f}, expected ~9.00. "
            "Fees may still be hardcoded to 0.1% instead of reading bot.exchange_fee."
        )

    @pytest.mark.asyncio
    async def test_E_paper_trade_high_fee_reduces_win_rate(self):
        """A high exchange fee must turn a small-gain trade into a loss.

        With bot.exchange_fee = 2.0 % (4 % round-trip), a 1 % price move loses money.
        If fees were hardcoded to 0.1 %, the same trade would show a gain.
        """
        engine = TradingEngine()
        bot = MagicMock()
        bot.id = 907
        bot.trading_pair = "BTC/USDT"
        bot.exchange_fee = 2.0  # 2 % per side → 4 % round-trip

        engine._recovery_states = {
            907: {
                "paper_trades": [],
                "consecutive_paper_wins": 0,
                "paper_position": {
                    "entry_price": 100.0,
                    "amount_usd": 100.0,
                    "trading_pair": "BTC/USDT",
                    "entered_at": datetime.utcnow().isoformat(),
                },
            }
        }

        sell_signal = TradeSignal(action="sell", amount=100.0)
        session = AsyncMock()

        # Price up 1 %: gross gain = $1, round-trip fees at 2% = $4 → net = -$3
        await engine._process_paper_trade(bot, 907, sell_signal, 101.0, session)

        trade = engine._recovery_states[907]["paper_trades"][0]
        assert trade["win"] is False, (
            "A 1% gain with 2% per-side fees must be a LOSS. "
            "If this passes as a win, paper trade is using the hardcoded 0.1% fee."
        )
        assert trade["gain_loss_usd"] < 0, (
            f"gain_loss_usd={trade['gain_loss_usd']:.4f} should be negative at 2% fees"
        )


# ---------------------------------------------------------------------------
# Test F: Viability rejections are NOT failures — bot must not be paused
# ---------------------------------------------------------------------------

class TestViabilityGateDoesNotPauseBot:
    """Regression: 1000 consecutive viability rejections must not pause a bot.

    Root cause (fixed): _record_trade_outcome was called with "viability_gate_rejected"
    inside the gate, feeding the repeated-rejection circuit breaker. After 5
    consecutive identical keys the bot was paused. Viability rejections are valid
    no-trade decisions, not system failures. The fix removes the counter calls.
    """

    @pytest.mark.asyncio
    async def test_F_1000_viability_rejections_do_not_pause(self):
        """1000 consecutive viability rejections must leave _exec_rejections empty."""
        engine = TradingEngine()
        engine._record_trade_outcome = AsyncMock()

        pause_calls = []
        async def _fake_pause(bot_id, count, reason_text):
            pause_calls.append((bot_id, count, reason_text))
        engine._pause_bot_for_repeated_rejection = _fake_pause

        # expected_move_pct impossibly small — viability gate always blocks this
        signal = TradeSignal(
            action="buy",
            amount=100.0,
            order_type="market",
            expected_move_pct=0.000001,
        )
        bot = _execute_trade_bot(exchange_fee=0.1)
        passing = _passing_check()

        with patch("app.services.trading_engine.PortfolioRiskService") as mock_prs, \
             patch("app.services.trading_engine.StrategyCapacityService") as mock_scs:
            mock_prs.return_value.check_portfolio_risk = AsyncMock(return_value=passing)
            mock_scs.return_value.check_capacity_for_trade = AsyncMock(return_value=passing)
            for _ in range(1000):
                result = await engine._execute_trade(
                    bot, MagicMock(), signal, 64_000.0, AsyncMock()
                )
                assert result is None, "Viability gate must block this trade"

        # _record_trade_outcome must NOT have been called with viability_gate_rejected
        for call in engine._record_trade_outcome.await_args_list:
            key = call.args[1] if len(call.args) > 1 else call.kwargs.get("reason_key")
            assert key != "viability_gate_rejected", (
                "viability_gate_rejected must NOT feed the repeated-rejection counter."
            )
        assert pause_calls == [], (
            "Bot paused after viability rejections — gate must not increment failure counter."
        )

    @pytest.mark.asyncio
    async def test_F_accumulation_signal_passes_gate(self):
        """An accumulation signal (is_accumulation=True) with normal fee passes the gate."""
        sentinel_exchange = MagicMock()
        sentinel_exchange.place_market_order = AsyncMock(return_value=MagicMock(
            id="ord1", amount=0.0015, filled=0.0015,
            price=64000.0, cost=100.0, fee=0.1, fee_currency=None, status="closed",
        ))

        engine = TradingEngine()
        outcome_keys: list = []
        _orig = engine._record_trade_outcome
        async def _spy(b, key, *args, **kwargs):
            outcome_keys.append(key)
            return await _orig(b, key, *args, **kwargs)
        engine._record_trade_outcome = _spy

        bot = _execute_trade_bot(exchange_fee=0.1)
        signal = TradeSignal(
            action="buy",
            amount=100.0,
            order_type="market",
            is_accumulation=True,
        )

        passing = _passing_check()
        with patch("app.services.trading_engine.PortfolioRiskService") as mock_prs, \
             patch("app.services.trading_engine.StrategyCapacityService") as mock_scs:
            mock_prs.return_value.check_portfolio_risk = AsyncMock(return_value=passing)
            mock_scs.return_value.check_capacity_for_trade = AsyncMock(return_value=passing)
            try:
                await engine._execute_trade(
                    bot, sentinel_exchange, signal, 64_000.0, AsyncMock()
                )
            except Exception:
                pass  # Post-gate failures are fine; we only check the gate passed

        assert sentinel_exchange.place_market_order.called, (
            "Accumulation signal with 0.1% fee must pass viability gate and reach exchange."
        )
        assert "viability_gate_rejected" not in outcome_keys, (
            "Accumulation signal must not produce a viability_gate_rejected outcome key."
        )


# ---------------------------------------------------------------------------
# Test G: tf_bars KeyError — restored trend state missing the key
# ---------------------------------------------------------------------------

class TestTrendStateMissingTfBars:
    """Regression: _strategy_trend_following crashed with KeyError: 'tf_bars'
    when resuming from persisted state that predates the tf_bars key.

    Fix: _normalize_trend_state backfills missing keys before use.
    """

    @pytest.mark.asyncio
    async def test_G_missing_tf_bars_does_not_crash(self):
        engine = TradingEngine()
        engine._get_bot_positions = AsyncMock(return_value=[])

        bot = MagicMock()
        bot.id = 913
        bot.strategy = "trend_following"
        bot.trading_pair = "BTC/USDT"
        bot.current_balance = 1_000.0
        bot.budget = 1_000.0
        bot.exchange_fee = 0.1

        # Simulate old persisted state: tf_bars key is absent
        engine._trend_states = {
            913: {
                "trailing_stop": None,
                "highest_price": None,
                "entry_atr": None,
                "entry_time": None,
                "last_exit_time": None,
                "entry_confirmation_count": 0,
                "exit_confirmation_count": 0,
                # tf_bars intentionally missing (old persisted state)
                "tf_current_bar": None,
            }
        }

        params = {
            "short_period": 5,
            "long_period": 20,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "max_allocation_percent": 50,
        }

        # Must NOT raise KeyError
        try:
            signal = await engine._strategy_trend_following(
                bot, 64_000.0, params, AsyncMock()
            )
        except KeyError as e:
            pytest.fail(
                f"KeyError {e!r} raised — _normalize_trend_state must backfill 'tf_bars'."
            )

        # After normalization, tf_bars must exist
        assert "tf_bars" in engine._trend_states[913], (
            "_normalize_trend_state must add tf_bars to restored state."
        )
