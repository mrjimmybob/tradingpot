"""Phase 5 — Auto Mode runtime integration (single bot, no scheduler).

The committee runs inside the existing Auto bot's loop: evaluate all Alpha
strategies, collect StrategyProposals, run the committee, execute the winner
through the unchanged pipeline. Behind is_committee_enabled (OFF by default).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import sys
from app.services.trading_engine import (
    _COMMITTEE_ALPHA_STRATEGIES,
    TradeSignal,
    TradingEngine,
)

# app/services/__init__.py re-exports a `trading_engine` SINGLETON that shadows
# the submodule name, so `import ... as te` yields the instance. Grab the real
# module from sys.modules to monkeypatch its globals.
te = sys.modules["app.services.trading_engine"]
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter
from app.services.strategy_framework.decision_score import DecisionScoreResult, EvidenceReport
from app.services.strategy_framework.edge_management import EdgeCategory, EdgeStatus
from app.services.strategy_framework.market_suitability import MarketSuitabilityResult
from app.services.strategy_framework.proposal import (
    Direction, ExecutionIntent, ProposalValidity, StrategyProposal,
)

_GEN = datetime(2026, 1, 1, 12, 0, 0)
_NOW = datetime(2026, 1, 1, 12, 0, 30)


def _score(total):
    rep = EvidenceReport(strategy="t", contributions=[], total=total, threshold=0.0, approved=True)
    return DecisionScoreResult(total=total, threshold=0.0, approved=True,
                               contributions=[], evidence_report=rep)


def _proposal(strategy_id, total, size=100.0, intent=ExecutionIntent.OPEN_POSITION):
    return StrategyProposal(
        strategy_id=strategy_id, bot_id=1, generated_at=_GEN,
        direction=Direction.BUY, execution_intent=intent,
        validity=ProposalValidity(generated_at=_GEN, valid_until=_GEN + timedelta(hours=1)),
        decision_score=_score(total),
        market_suitability=MarketSuitabilityResult(True, [], ["all"], [], ""),
        edge_status=EdgeStatus(EdgeCategory.NONE, "", [], "", False, False, False),
        suggested_position_size=size, suggested_risk_budget_pct=1.0,
    )


def _signal_for(proposal):
    return StandaloneAdapter.to_trade_signal(proposal)  # carries _source_proposal


def _engine(monkeypatch):
    eng = TradingEngine()
    eng.clock = SimpleNamespace(now=lambda: _NOW)
    # Keep the committee's portfolio resolve out of the DB in unit tests.
    monkeypatch.setattr(te, "resolve_portfolio_constraints", AsyncMock(return_value=None))
    return eng


def _bot():
    return SimpleNamespace(id=1, strategy="auto_mode", strategy_params={})


# ---------------------------------------------------------------------------
# _source_proposal plumbing
# ---------------------------------------------------------------------------

class TestProposalCapture:
    def test_to_trade_signal_carries_source_proposal(self):
        p = _proposal("trend_following", 60.0)
        sig = StandaloneAdapter.to_trade_signal(p)
        assert sig._source_proposal is p

    def test_source_proposal_does_not_affect_equality(self):
        # Two signals from independently-built identical proposals stay equal;
        # _source_proposal is compare=False (Phase 2 byte-identical guarantee).
        a = StandaloneAdapter.to_trade_signal(_proposal("trend_following", 60.0))
        b = StandaloneAdapter.to_trade_signal(_proposal("trend_following", 60.0))
        assert a == b


# ---------------------------------------------------------------------------
# _committee_select
# ---------------------------------------------------------------------------

class TestCommitteeSelect:
    def _wire(self, eng, mapping):
        eng._get_strategy_executor = lambda name: mapping.get(name)

    def _exec_returning(self, sig):
        async def ex(bot, price, params, session):
            return sig
        return ex

    @pytest.mark.asyncio
    async def test_picks_highest_ranked_and_stamps_owner(self, monkeypatch):
        eng = _engine(monkeypatch)
        hi = _proposal("trend_following", 90.0)
        lo = _proposal("mean_reversion", 40.0)
        self._wire(eng, {
            "trend_following": self._exec_returning(_signal_for(hi)),
            "mean_reversion": self._exec_returning(_signal_for(lo)),
        })
        sig = await eng._committee_select(_bot(), 100.0,
                                          ("trend_following", "mean_reversion"), AsyncMock())
        assert sig is not None and sig.action == "buy"
        assert sig.reason.startswith("[Auto:trend_following|committee]")  # 90 > 40

    @pytest.mark.asyncio
    async def test_scales_amount_to_allocation(self, monkeypatch):
        eng = _engine(monkeypatch)
        # One proposal, but a portfolio cap trims the allocation to 250.
        from app.services.auto_committee import PortfolioConstraints
        monkeypatch.setattr(te, "resolve_portfolio_constraints",
                            AsyncMock(return_value=PortfolioConstraints(exposure_headroom_usd=250.0)))
        p = _proposal("trend_following", 80.0, size=1000.0)
        self._wire(eng, {"trend_following": self._exec_returning(_signal_for(p))})
        sig = await eng._committee_select(_bot(), 100.0, ("trend_following",), AsyncMock())
        assert sig.amount == 250.0

    @pytest.mark.asyncio
    async def test_returns_none_when_no_proposals(self, monkeypatch):
        eng = _engine(monkeypatch)
        # Executor returns a plain hold with no _source_proposal.
        self._wire(eng, {"trend_following": self._exec_returning(
            TradeSignal(action="hold", amount=0, reason="warmup"))})
        sig = await eng._committee_select(_bot(), 100.0, ("trend_following",), AsyncMock())
        assert sig is None

    @pytest.mark.asyncio
    async def test_a_failing_strategy_does_not_sink_the_cycle(self, monkeypatch):
        eng = _engine(monkeypatch)
        async def boom(bot, price, params, session):
            raise RuntimeError("strategy exploded")
        good = _proposal("mean_reversion", 55.0)
        self._wire(eng, {
            "trend_following": boom,
            "mean_reversion": self._exec_returning(_signal_for(good)),
        })
        sig = await eng._committee_select(_bot(), 100.0,
                                          ("trend_following", "mean_reversion"), AsyncMock())
        assert sig.reason.startswith("[Auto:mean_reversion|committee]")

    @pytest.mark.asyncio
    async def test_invoked_by_auto_flag_passed(self, monkeypatch):
        eng = _engine(monkeypatch)
        seen = {}
        async def ex(bot, price, params, session):
            seen["invoked_by_auto"] = params.get("_invoked_by_auto")
            return _signal_for(_proposal("trend_following", 70.0))
        self._wire(eng, {"trend_following": ex})
        await eng._committee_select(_bot(), 100.0, ("trend_following",), AsyncMock())
        assert seen["invoked_by_auto"] is True


# ---------------------------------------------------------------------------
# _strategy_auto_committee — ownership + flat entry
# ---------------------------------------------------------------------------

class TestAutoCommitteeDispatch:
    @pytest.mark.asyncio
    async def test_flat_runs_committee(self, monkeypatch):
        eng = _engine(monkeypatch)
        eng._get_bot_positions = AsyncMock(return_value=[])
        eng._committee_select = AsyncMock(return_value=TradeSignal(
            action="buy", amount=100.0, reason="[Auto:trend_following|committee] x"))
        sig = await eng._strategy_auto_committee(_bot(), 100.0, {}, AsyncMock())
        eng._committee_select.assert_awaited_once()
        assert sig.action == "buy"

    @pytest.mark.asyncio
    async def test_in_position_dispatches_only_owner(self, monkeypatch):
        eng = _engine(monkeypatch)
        eng._get_bot_positions = AsyncMock(return_value=[
            SimpleNamespace(owning_strategy="mean_reversion")])
        eng._committee_select = AsyncMock()  # must NOT be called
        exit_sig = TradeSignal(action="sell", amount=50.0, reason="take profit")
        async def owner_exec(bot, price, params, session):
            return exit_sig
        eng._get_strategy_executor = lambda name: owner_exec if name == "mean_reversion" else None
        sig = await eng._strategy_auto_committee(_bot(), 100.0, {}, AsyncMock())
        eng._committee_select.assert_not_called()
        assert sig.action == "sell"
        assert sig.reason.startswith("[Auto:mean_reversion|committee]")  # owner-stamped

    @pytest.mark.asyncio
    async def test_unowned_position_holds(self, monkeypatch):
        eng = _engine(monkeypatch)
        eng._get_bot_positions = AsyncMock(return_value=[
            SimpleNamespace(owning_strategy=None)])
        sig = await eng._strategy_auto_committee(_bot(), 100.0, {}, AsyncMock())
        assert sig.action == "hold"

    @pytest.mark.asyncio
    async def test_flat_no_selection_holds(self, monkeypatch):
        eng = _engine(monkeypatch)
        eng._get_bot_positions = AsyncMock(return_value=[])
        eng._committee_select = AsyncMock(return_value=None)
        sig = await eng._strategy_auto_committee(_bot(), 100.0, {}, AsyncMock())
        assert sig.action == "hold"


# ---------------------------------------------------------------------------
# Flag gating + Alpha set
# ---------------------------------------------------------------------------

class TestFlagAndAlphaSet:
    @pytest.mark.asyncio
    async def test_flag_on_delegates_to_committee(self, monkeypatch):
        eng = TradingEngine()
        monkeypatch.setattr(te, "is_committee_enabled", lambda bot: True)
        sentinel = TradeSignal(action="hold", amount=0, reason="committee ran")
        eng._strategy_auto_committee = AsyncMock(return_value=sentinel)
        out = await eng._strategy_auto(_bot(), 100.0, {}, AsyncMock())
        eng._strategy_auto_committee.assert_awaited_once()
        assert out is sentinel

    def test_dca_never_a_committee_candidate(self):
        assert "dca_accumulator" not in _COMMITTEE_ALPHA_STRATEGIES
        assert "dip_recovery" in _COMMITTEE_ALPHA_STRATEGIES
        assert set(_COMMITTEE_ALPHA_STRATEGIES) == {
            "trend_following", "mean_reversion", "volatility_breakout",
            "dip_recovery", "adaptive_grid"}
