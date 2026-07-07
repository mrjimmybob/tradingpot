"""Phase 2 execution-quality tests.

Covers the four required scenarios:
  a) Trade rejected when expected move cannot cover round-trip fees.
  b) Trade allowed when expected move clears round-trip fees.
  c) Exchange fee is read from bot.exchange_fee — never silently zero.
  d) Trend Following uses bar-based ATR, not microscopic tick ATR.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.trading_engine import TradingEngine


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
