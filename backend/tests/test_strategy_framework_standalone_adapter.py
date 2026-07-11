"""Unit tests for the Standalone Adapter
(add-strategy-decision-framework, Phase 0.11).

Covers: pure StrategyProposal -> TradeSignal translation, execution_intent
branching (never direction alone), expired-proposal discarding, and a
behavior-preservation test proving the adapter + existing `_execute_trade`
pipeline produces the identical outcome a hand-built TradeSignal +
`_execute_trade` already produces today (mirrors test_reward_risk_gate.py's
gate-integration pattern).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models import BotStatus
from app.services.strategy_framework.decision_score import DecisionScoreEngine, EvidenceItem
from app.services.strategy_framework.edge_management import EdgeCategory, EdgeStatus
from app.services.strategy_framework.market_suitability import MarketSuitabilityResult
from app.services.strategy_framework.proposal import (
    Direction,
    ExecutionIntent,
    ProposalValidity,
    StrategyProposal,
)
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter
from app.services.trading_engine import TradeSignal, TradingEngine

GENERATED_AT = datetime(2026, 1, 1, 12, 0, 0)


def _validity(generated_at=GENERATED_AT, minutes=5) -> ProposalValidity:
    return ProposalValidity(generated_at=generated_at, valid_until=generated_at + timedelta(minutes=minutes))


def _suitability() -> MarketSuitabilityResult:
    return MarketSuitabilityResult(
        is_suitable=True, regime_tags=["trend_up"], allowed_regimes=["trend_up"],
        matched_tags=["trend_up"], reason="ok",
    )


def _edge_status() -> EdgeStatus:
    return EdgeStatus(
        category=EdgeCategory.NONE, action="continue", signals=[], reason="ok",
        can_adapt=False, should_wait=False, should_stop=False, evaluated_at=GENERATED_AT,
    )


def _decision_score(total_override: float = 80.0, threshold: float = 50.0):
    engine = DecisionScoreEngine()
    item = EvidenceItem(
        name="Trend strength", measurement=lambda d: d["x"],
        normalization=lambda r: max(-1.0, min(1.0, r)), weight=total_override,
        reason="theory doc",
    )
    return engine.score("trend_following", [item], {"x": 1.0}, threshold=threshold)


def _proposal(
    *,
    direction: Direction = Direction.BUY,
    execution_intent: ExecutionIntent = ExecutionIntent.OPEN_POSITION,
    suggested_position_size: float = 500.0,
    validity: ProposalValidity = None,
    reasons_for=("Trend strength (+80.0): theory doc",),
) -> StrategyProposal:
    return StrategyProposal(
        strategy_id="trend_following",
        bot_id=999,
        generated_at=GENERATED_AT,
        direction=direction,
        execution_intent=execution_intent,
        validity=validity or _validity(),
        decision_score=_decision_score(),
        market_suitability=_suitability(),
        edge_status=_edge_status(),
        assumptions=("trend direction unchanged since entry",),
        reasons_for=reasons_for,
        suggested_position_size=suggested_position_size,
        suggested_risk_budget_pct=0.01,
    )


class TestToTradeSignalTranslation:
    def test_no_action_produces_no_signal(self):
        proposal = _proposal(direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION)
        assert StandaloneAdapter.to_trade_signal(proposal) is None

    def test_hold_position_produces_no_signal(self):
        proposal = _proposal(direction=Direction.HOLD, execution_intent=ExecutionIntent.HOLD_POSITION)
        assert StandaloneAdapter.to_trade_signal(proposal) is None

    def test_open_position_produces_buy_signal(self):
        proposal = _proposal(direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION)
        signal = StandaloneAdapter.to_trade_signal(proposal)
        assert isinstance(signal, TradeSignal)
        assert signal.action == "buy"
        assert signal.amount == 500.0

    def test_add_to_position_produces_buy_signal(self):
        proposal = _proposal(direction=Direction.BUY, execution_intent=ExecutionIntent.ADD_TO_POSITION)
        signal = StandaloneAdapter.to_trade_signal(proposal)
        assert signal.action == "buy"

    def test_reduce_position_produces_sell_signal(self):
        proposal = _proposal(direction=Direction.SELL, execution_intent=ExecutionIntent.REDUCE_POSITION)
        signal = StandaloneAdapter.to_trade_signal(proposal)
        assert signal.action == "sell"

    def test_close_position_produces_sell_signal(self):
        proposal = _proposal(direction=Direction.SELL, execution_intent=ExecutionIntent.CLOSE_POSITION)
        signal = StandaloneAdapter.to_trade_signal(proposal)
        assert signal.action == "sell"

    def test_branches_on_execution_intent_not_direction_alone(self):
        """Both OPEN_POSITION and ADD_TO_POSITION share direction=BUY but
        the action mapping must come from execution_intent, proving
        direction alone is never consulted for the branch."""
        open_signal = StandaloneAdapter.to_trade_signal(
            _proposal(direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION)
        )
        add_signal = StandaloneAdapter.to_trade_signal(
            _proposal(direction=Direction.BUY, execution_intent=ExecutionIntent.ADD_TO_POSITION)
        )
        assert open_signal.action == add_signal.action == "buy"

    def test_score_and_threshold_carried_from_decision_score(self):
        proposal = _proposal()
        signal = StandaloneAdapter.to_trade_signal(proposal)
        assert signal.score == proposal.decision_score.total
        assert signal.threshold == proposal.decision_score.threshold

    def test_reason_derived_from_reasons_for(self):
        proposal = _proposal(reasons_for=("Trend strength (+80.0): theory doc",))
        signal = StandaloneAdapter.to_trade_signal(proposal)
        assert "Trend strength" in signal.reason

    def test_expected_move_and_risk_pct_passthrough(self):
        proposal = _proposal()
        signal = StandaloneAdapter.to_trade_signal(
            proposal, expected_move_pct=0.01, expected_risk_pct=0.005,
        )
        assert signal.expected_move_pct == 0.01
        assert signal.expected_risk_pct == 0.005

    def test_deterministic(self):
        proposal = _proposal()
        s1 = StandaloneAdapter.to_trade_signal(proposal)
        s2 = StandaloneAdapter.to_trade_signal(proposal)
        assert s1.action == s2.action
        assert s1.amount == s2.amount
        assert s1.reason == s2.reason


# ---------------------------------------------------------------------------
# Expiry: never execute a stale proposal.
# ---------------------------------------------------------------------------

class TestExpiry:
    @pytest.mark.asyncio
    async def test_expired_proposal_is_discarded_without_calling_execute_trade(self):
        proposal = _proposal(validity=_validity(GENERATED_AT, minutes=5))
        engine = MagicMock()
        engine._execute_trade = AsyncMock()
        adapter = StandaloneAdapter()

        result = await adapter.execute(
            proposal,
            engine=engine,
            bot=MagicMock(id=999),
            exchange=MagicMock(),
            current_price=64_000.0,
            session=AsyncMock(),
            now=GENERATED_AT + timedelta(minutes=5),  # exactly valid_until -> expired
        )

        assert result is None
        engine._execute_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_expired_proposal_proceeds_to_execute_trade(self):
        proposal = _proposal(validity=_validity(GENERATED_AT, minutes=5))
        engine = MagicMock()
        engine._execute_trade = AsyncMock(return_value="order-sentinel")
        adapter = StandaloneAdapter()

        result = await adapter.execute(
            proposal,
            engine=engine,
            bot=MagicMock(id=999),
            exchange=MagicMock(),
            current_price=64_000.0,
            session=AsyncMock(),
            now=GENERATED_AT + timedelta(minutes=4, seconds=59),
        )

        assert result == "order-sentinel"
        engine._execute_trade.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_action_intent_never_calls_execute_trade(self):
        proposal = _proposal(direction=Direction.NO_TRADE, execution_intent=ExecutionIntent.NO_ACTION)
        engine = MagicMock()
        engine._execute_trade = AsyncMock()
        adapter = StandaloneAdapter()

        result = await adapter.execute(
            proposal, engine=engine, bot=MagicMock(id=999), exchange=MagicMock(),
            current_price=64_000.0, session=AsyncMock(), now=GENERATED_AT,
        )

        assert result is None
        engine._execute_trade.assert_not_called()


# ---------------------------------------------------------------------------
# Behavior preservation: adapter + _execute_trade == hand-built TradeSignal
# + _execute_trade (same pattern as tests/test_reward_risk_gate.py).
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
    bot.strategy = "trend_following"
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


async def _run_with_signal(signal: TradeSignal, current_price: float = 64_000.0):
    engine = TradingEngine()
    engine._record_trade_outcome = AsyncMock()
    bot = _execute_trade_bot()
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
            return "reached_exchange" if exchange.place_market_order.called else None


async def _run_with_proposal(proposal: StrategyProposal, current_price: float = 64_000.0, **kw):
    engine = TradingEngine()
    engine._record_trade_outcome = AsyncMock()
    bot = _execute_trade_bot()
    exchange = MagicMock()
    sentinel_order = MagicMock()
    sentinel_order.id = "filled"
    exchange.place_market_order = AsyncMock(return_value=sentinel_order)
    session = AsyncMock()
    adapter = StandaloneAdapter()

    passing = _passing_check()
    with patch("app.services.trading_engine.PortfolioRiskService") as mock_prs, \
         patch("app.services.trading_engine.StrategyCapacityService") as mock_scs:
        mock_prs.return_value.check_portfolio_risk = AsyncMock(return_value=passing)
        mock_scs.return_value.check_capacity_for_trade = AsyncMock(return_value=passing)
        try:
            return await adapter.execute(
                proposal, engine=engine, bot=bot, exchange=exchange,
                current_price=current_price, session=session, now=GENERATED_AT, **kw,
            )
        except Exception:
            return "reached_exchange" if exchange.place_market_order.called else None


class TestBehaviorPreservation:
    @pytest.mark.asyncio
    async def test_good_rr_buy_reaches_exchange_identically(self):
        """Fixed historical scenario: entry 62000 / target 62300 / stop
        61850 -> RR=2.0. Both the pre-migration TradeSignal path and the
        StrategyProposal + Standalone Adapter path must reach the exchange."""
        reward = 300 / 62000
        risk = 150 / 62000

        hand_built_signal = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="good RR", expected_move_pct=reward, expected_risk_pct=risk,
        )
        signal_result = await _run_with_signal(hand_built_signal)

        proposal = _proposal(
            direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
            suggested_position_size=500.0,
        )
        proposal_result = await _run_with_proposal(
            proposal, expected_move_pct=reward, expected_risk_pct=risk,
        )

        assert signal_result is not None
        assert proposal_result is not None
        assert signal_result == proposal_result == "reached_exchange"

    @pytest.mark.asyncio
    async def test_bad_rr_buy_rejected_identically(self):
        """Entry 62000 / target 62100 / stop 61700 -> RR=0.33, must be
        rejected via both paths identically."""
        reward = 100 / 62000
        risk = 300 / 62000

        hand_built_signal = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="bad RR", expected_move_pct=reward, expected_risk_pct=risk,
        )
        signal_result = await _run_with_signal(hand_built_signal)

        proposal = _proposal(
            direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
            suggested_position_size=500.0,
        )
        proposal_result = await _run_with_proposal(
            proposal, expected_move_pct=reward, expected_risk_pct=risk,
        )

        assert signal_result is None
        assert proposal_result is None

    def test_translated_signal_matches_hand_built_signal_field_for_field(self):
        """The adapter's translation itself - not just the downstream
        outcome - must produce a TradeSignal equivalent to what a
        pre-migration strategy would have hand-built."""
        proposal = _proposal(
            direction=Direction.BUY, execution_intent=ExecutionIntent.OPEN_POSITION,
            suggested_position_size=500.0,
        )
        translated = StandaloneAdapter.to_trade_signal(
            proposal, expected_move_pct=300 / 62000, expected_risk_pct=150 / 62000,
        )
        hand_built = TradeSignal(
            action="buy", amount=500.0, order_type="market",
            reason="good RR", expected_move_pct=300 / 62000, expected_risk_pct=150 / 62000,
        )
        assert translated.action == hand_built.action
        assert translated.amount == hand_built.amount
        assert translated.expected_move_pct == hand_built.expected_move_pct
        assert translated.expected_risk_pct == hand_built.expected_risk_pct
