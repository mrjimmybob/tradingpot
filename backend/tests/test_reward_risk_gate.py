"""Regression tests for the reward:risk viability check (Finding 2).

Root cause covered here:
The trade viability gate in ``_execute_trade`` (STEP 5.5) only asked "can
expected profit beat fees?" (``expected_move_pct > round_trip_fees +
margin``). It never asked "is the reward worth the risk?" Forensic review of
closed trades found average losses (~$0.062) running roughly 3x average wins
(~$0.022) — stops were sized independently of targets (e.g. mean_reversion's
ATR-based hard stop vs. its Bollinger-band profit target), so a strategy could
pass the fee check while still risking far more than it targeted to gain.

``evaluate_reward_risk`` (module-level, single implementation, called from the
gate) rejects a directional BUY when
``expected_move_pct / expected_risk_pct < _MIN_REWARD_RISK_RATIO``.
``expected_risk_pct`` is optional: strategies with no fixed price target
(trend_following, volatility_breakout) leave it ``None`` and are entirely
unaffected — the check is a no-op for them, preserving prior behavior exactly.

These tests cover:
1. ``evaluate_reward_risk`` in isolation (pass, reject, invalid risk/reward,
   None passthrough) — no bot/engine required.
2. The gate integration: good RR passes, bad RR rejects, the pre-existing fee
   check still runs (RR alone cannot rescue a sub-fee move), and strategies
   that don't supply ``expected_risk_pct`` are unaffected (preserved
   ``expected_move_pct``-only behavior).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.trading_engine import (
    BotStatus,
    TradeSignal,
    TradingEngine,
    evaluate_reward_risk,
    _MIN_REWARD_RISK_RATIO,
)


# ---------------------------------------------------------------------------
# Part 1: evaluate_reward_risk (pure function)
# ---------------------------------------------------------------------------

class TestEvaluateRewardRisk:
    def test_none_risk_is_a_noop_pass(self):
        """No fixed stop to measure risk against -> check does not apply."""
        ok, reason = evaluate_reward_risk(expected_move_pct=0.001, expected_risk_pct=None)
        assert ok is True
        assert reason == ""

    def test_good_ratio_passes(self):
        # Entry 62000, target 62300 (reward 300/62000), stop 61850 (risk 150/62000)
        # RR = 300/150 = 2.0 -> allow
        reward = 300 / 62000
        risk = 150 / 62000
        ok, reason = evaluate_reward_risk(reward, risk)
        assert ok is True
        assert reason == ""

    def test_bad_ratio_rejects(self):
        # Entry 62000, target 62100 (reward 100), stop 61700 (risk 300)
        # RR = 100/300 = 0.33 -> reject
        reward = 100 / 62000
        risk = 300 / 62000
        ok, reason = evaluate_reward_risk(reward, risk)
        assert ok is False
        assert "reward:risk" in reason
        assert "0.33" in reason

    def test_ratio_exactly_at_minimum_passes(self):
        risk = 0.01
        reward = risk * _MIN_REWARD_RISK_RATIO
        ok, _ = evaluate_reward_risk(reward, risk)
        assert ok is True

    def test_ratio_just_below_minimum_rejects(self):
        risk = 0.01
        reward = risk * (_MIN_REWARD_RISK_RATIO - 0.01)
        ok, _ = evaluate_reward_risk(reward, risk)
        assert ok is False

    def test_zero_or_negative_risk_rejects(self):
        ok, reason = evaluate_reward_risk(expected_move_pct=0.01, expected_risk_pct=0.0)
        assert ok is False
        assert "risk" in reason

        ok, reason = evaluate_reward_risk(expected_move_pct=0.01, expected_risk_pct=-0.005)
        assert ok is False

    def test_zero_or_negative_reward_rejects_even_with_positive_risk(self):
        ok, reason = evaluate_reward_risk(expected_move_pct=0.0, expected_risk_pct=0.01)
        assert ok is False
        assert "reward" in reason

        ok, reason = evaluate_reward_risk(expected_move_pct=-0.002, expected_risk_pct=0.01)
        assert ok is False

    def test_missing_reward_treated_as_zero_not_a_crash(self):
        ok, reason = evaluate_reward_risk(expected_move_pct=None, expected_risk_pct=0.01)
        assert ok is False


# ---------------------------------------------------------------------------
# Part 2: gate integration via _execute_trade
# ---------------------------------------------------------------------------

def _passing_check(adjusted_amount=None):
    check = MagicMock()
    check.ok = True
    check.action = None
    check.adjusted_amount = adjusted_amount
    check.violated_cap = None
    check.details = ""
    check.reason = ""
    return check


def _execute_trade_bot(exchange_fee: float = 0.1, balance: float = 1_000.0) -> MagicMock:
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


async def _run_execute_trade(signal: TradeSignal, exchange_fee: float = 0.1, current_price: float = 64_000.0):
    """Run _execute_trade with portfolio/capacity checks patched to pass and a
    market order that succeeds, so only the viability gate can reject.
    """
    engine = TradingEngine()
    engine._record_trade_outcome = AsyncMock()

    bot = _execute_trade_bot(exchange_fee=exchange_fee)
    exchange = MagicMock()
    sentinel_order = MagicMock()
    sentinel_order.id = "filled"
    exchange.place_market_order = AsyncMock(return_value=sentinel_order)
    session = AsyncMock()

    passing = _passing_check()
    with patch("app.services.trading_engine.PortfolioRiskService") as mock_prs, \
         patch("app.services.trading_engine.StrategyCapacityService") as mock_scs:
        mock_prs.return_value.check_portfolio_risk = AsyncMock(return_value=passing)
        mock_scs.return_value.check_capacity_for_trade = AsyncMock(return_value=passing)
        try:
            return await engine._execute_trade(bot, exchange, signal, current_price, session)
        except Exception:
            # Post-gate persistence may fail with minimal session mocking; the
            # gate itself already ran by the time place_market_order is called.
            return "reached_exchange" if exchange.place_market_order.called else None


class TestGateRejectsBadRewardRisk:
    @pytest.mark.asyncio
    async def test_good_rr_passes_gate(self):
        """Entry 62000 / target 62300 / stop 61850 -> RR=2.0, should reach the exchange."""
        signal = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="good RR", expected_move_pct=300 / 62000, expected_risk_pct=150 / 62000,
        )
        result = await _run_execute_trade(signal)
        assert result is not None, "a 2.0 RR signal should not be blocked by the reward:risk gate"

    @pytest.mark.asyncio
    async def test_bad_rr_rejects(self):
        """Entry 62000 / target 62100 / stop 61700 -> RR=0.33, must be rejected."""
        signal = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="bad RR", expected_move_pct=100 / 62000, expected_risk_pct=300 / 62000,
        )
        result = await _run_execute_trade(signal)
        assert result is None, "a 0.33 RR signal must be rejected by the reward:risk gate"

    @pytest.mark.asyncio
    async def test_fees_still_enforced_even_with_perfect_rr(self):
        """A tiny move with a proportionally tiny stop has a perfect 2:1 RR but
        still cannot clear round-trip fees — the pre-existing fee check must
        still block it. Proves the RR check was added, not substituted.
        """
        signal = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="RR ok but sub-fee", expected_move_pct=0.0005, expected_risk_pct=0.00025,
        )
        result = await _run_execute_trade(signal, exchange_fee=0.1)
        assert result is None, "sub-fee expected move must still be rejected regardless of RR"

    @pytest.mark.asyncio
    async def test_expected_move_pct_only_behavior_preserved(self):
        """A strategy that does not supply expected_risk_pct (e.g. trend_following,
        which has no fixed take-profit target) must behave exactly as before:
        only the fee-viability check applies, no RR check.
        """
        signal = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="no fixed stop", expected_move_pct=0.01, expected_risk_pct=None,
        )
        result = await _run_execute_trade(signal)
        assert result is not None, (
            "signals without expected_risk_pct must be unaffected by the RR gate"
        )
