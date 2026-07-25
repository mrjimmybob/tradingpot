"""Tests for the Auto Mode Investment Committee — Phase 0 (Committee Core).

Pure logic over synthetic StrategyProposal fixtures — no real strategy, no
portfolio service, no execution pipeline. Covers the Phase 0 acceptance
criteria and the Auto Certification Gate properties reachable at this phase:
immutable schemas, Comparison Contract field isolation, the reject/rank/
allocate/select steps, ranking determinism + strategy-identity-blindness,
multi-selection, no-proposal-silently-dropped, proposal immutability across a
cycle, and CommitteeDecision reproducibility.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta

import pytest

from app.services.auto_committee import (
    CommitteeDecision,
    RejectedProposal,
    SelectedAllocation,
    TrustAdjustment,
    read_comparison,
    run_committee,
)
from app.services.auto_committee.comparison import COMPARISON_CONTRACT_FIELDS, ComparisonView
from app.services.auto_committee.process import rank_key
from app.services.strategy_framework.decision_score import DecisionScoreResult, EvidenceReport
from app.services.strategy_framework.edge_management import EdgeCategory, EdgeStatus
from app.services.strategy_framework.market_suitability import MarketSuitabilityResult
from app.services.strategy_framework.proposal import (
    Direction,
    ExecutionIntent,
    ProposalValidity,
    StrategyProposal,
)

_GEN = datetime(2026, 1, 1, 12, 0, 0)
_NOW = datetime(2026, 1, 1, 12, 0, 30)  # 30s after generation, well within validity


def _score(total: float, threshold: float = 0.0) -> DecisionScoreResult:
    report = EvidenceReport(strategy="t", contributions=[], total=total,
                            threshold=threshold, approved=total >= threshold)
    return DecisionScoreResult(total=total, threshold=threshold,
                               approved=total >= threshold, contributions=[],
                               evidence_report=report)


def _proposal(*, strategy_id="trend_following", bot_id=1, gen=_GEN, total=50.0,
              threshold=0.0, intent=ExecutionIntent.OPEN_POSITION,
              direction=Direction.BUY, valid_minutes=60, size=100.0,
              edge_cat=EdgeCategory.NONE, is_suitable=True, risk_budget=1.0,
              edge_estimate=None) -> StrategyProposal:
    return StrategyProposal(
        strategy_id=strategy_id, bot_id=bot_id, generated_at=gen,
        direction=direction, execution_intent=intent,
        validity=ProposalValidity(generated_at=gen, valid_until=gen + timedelta(minutes=valid_minutes)),
        decision_score=_score(total, threshold),
        market_suitability=MarketSuitabilityResult(
            is_suitable=is_suitable, regime_tags=[], allowed_regimes=["all"],
            matched_tags=[], reason=""),
        edge_status=EdgeStatus(
            category=edge_cat, action="", signals=[], reason="",
            can_adapt=False, should_wait=False, should_stop=(edge_cat == EdgeCategory.C)),
        suggested_position_size=size, suggested_risk_budget_pct=risk_budget,
        expected_edge_estimate=edge_estimate,
    )


# ---------------------------------------------------------------------------
# 0.1 / 0.2 — schema immutability
# ---------------------------------------------------------------------------

class TestSchemasImmutable:
    def test_committee_decision_is_frozen_and_tuples(self):
        d = run_committee([_proposal()], now=_NOW)
        with pytest.raises(FrozenInstanceError):
            d.decision_id = "x"  # type: ignore[misc]
        assert isinstance(d.selected, tuple)
        assert isinstance(d.rejected, tuple)
        assert isinstance(d.proposals_considered, tuple)
        assert isinstance(d.ranking_snapshot, tuple)

    def test_selected_and_rejected_frozen(self):
        with pytest.raises(FrozenInstanceError):
            SelectedAllocation("p", 10.0, 1).allocated_size = 5.0  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            RejectedProposal("p", "expired", "r").rejection_step = "x"  # type: ignore[misc]

    def test_trust_adjustment_frozen_and_validated(self):
        ta = TrustAdjustment("p", "fear_greed", 1.2, _GEN)
        with pytest.raises(FrozenInstanceError):
            ta.adjustment = 2.0  # type: ignore[misc]
        with pytest.raises(ValueError):
            TrustAdjustment("", "src", 1.0, _GEN)
        with pytest.raises(ValueError):
            TrustAdjustment("p", "", 1.0, _GEN)


# ---------------------------------------------------------------------------
# 0.3 — Comparison Contract field isolation
# ---------------------------------------------------------------------------

class TestComparisonContractIsolation:
    def test_view_exposes_exactly_the_contract_fields(self):
        view = read_comparison(_proposal())
        exposed = {f for f in vars(view)}
        assert exposed == {"proposal_id", *COMPARISON_CONTRACT_FIELDS}

    def test_view_hides_all_strategy_internals(self):
        view = read_comparison(_proposal())
        for forbidden in ("explanation", "assumptions", "reasons_for",
                          "reasons_against", "adaptive_parameters_used",
                          "suggested_position_size", "decision_score",
                          "strategy_id", "bot_id"):
            assert not hasattr(view, forbidden), f"comparison view leaks {forbidden!r}"

    def test_rank_key_consumes_only_a_comparison_view(self):
        # rank_key's only parameter is a ComparisonView; it cannot be handed a
        # raw proposal's internals.
        import inspect
        params = list(inspect.signature(rank_key).parameters.values())
        assert len(params) == 1
        assert params[0].annotation in (ComparisonView, "ComparisonView")


# ---------------------------------------------------------------------------
# 0.4 — reject steps (2, 3, 4) with correct rejection_step
# ---------------------------------------------------------------------------

def _rejected_by_step(decision: CommitteeDecision) -> dict:
    return {r.proposal_id: r.rejection_step for r in decision.rejected}


class TestRejectionSteps:
    def test_expired_rejected_at_step_2(self):
        p = _proposal(valid_minutes=1)  # valid_until = 12:01, now = ...
        d = run_committee([p], now=_GEN + timedelta(minutes=5))
        assert _rejected_by_step(d)[p.proposal_id] == "expired"
        assert not d.selected

    def test_superseded_rejected_at_step_3(self):
        older = _proposal(bot_id=1, strategy_id="mean_reversion", gen=_GEN)
        newer = _proposal(bot_id=1, strategy_id="mean_reversion",
                          gen=_GEN + timedelta(seconds=1))
        d = run_committee([older, newer], now=_NOW + timedelta(seconds=1))
        assert _rejected_by_step(d)[older.proposal_id] == "superseded"
        assert newer.proposal_id in {s.proposal_id for s in d.selected}

    def test_edge_category_c_rejected_at_step_4(self):
        p = _proposal(edge_cat=EdgeCategory.C)
        d = run_committee([p], now=_NOW)
        assert _rejected_by_step(d)[p.proposal_id] == "edge_disqualified"

    def test_edge_category_a_and_b_are_eligible(self):
        pa = _proposal(strategy_id="s_a", edge_cat=EdgeCategory.A)
        pb = _proposal(strategy_id="s_b", edge_cat=EdgeCategory.B)
        d = run_committee([pa, pb], now=_NOW)
        selected = {s.proposal_id for s in d.selected}
        assert pa.proposal_id in selected and pb.proposal_id in selected

    def test_rejection_is_terminal_expired_not_reconsidered(self):
        # An expired proposal is rejected at step 2 and never appears selected,
        # even though its edge/suitability would otherwise pass.
        p = _proposal(valid_minutes=1, edge_cat=EdgeCategory.NONE)
        d = run_committee([p], now=_GEN + timedelta(minutes=10))
        assert _rejected_by_step(d)[p.proposal_id] == "expired"


# ---------------------------------------------------------------------------
# 0.5 — ranking determinism + strategy-identity-blindness
# ---------------------------------------------------------------------------

def _ranked_scores(decision, batch):
    by_id = {p.proposal_id: p for p in batch}
    return [by_id[pid].decision_score.total for pid in decision.ranking_snapshot]


class TestRankingDeterminismAndBlindness:
    def test_ranking_is_score_descending(self):
        batch = [_proposal(strategy_id="a", total=40.0),
                 _proposal(strategy_id="b", total=90.0),
                 _proposal(strategy_id="c", total=65.0)]
        d = run_committee(batch, now=_NOW)
        assert _ranked_scores(d, batch) == [90.0, 65.0, 40.0]

    def test_ranking_reproducible_across_runs(self):
        batch = [_proposal(strategy_id="a", total=40.0),
                 _proposal(strategy_id="b", total=90.0),
                 _proposal(strategy_id="c", total=65.0)]
        snaps = {tuple(run_committee(batch, now=_NOW).ranking_snapshot) for _ in range(20)}
        assert len(snaps) == 1

    def test_ranking_is_strategy_identity_blind(self):
        # Same Comparison Contract values, different strategy carrying each:
        # the ranked SCORE sequence must be identical regardless of identity.
        run1 = [_proposal(strategy_id="trend_following", total=90.0),
                _proposal(strategy_id="mean_reversion", total=40.0)]
        run2 = [_proposal(strategy_id="mean_reversion", total=90.0),
                _proposal(strategy_id="trend_following", total=40.0)]
        d1, d2 = run_committee(run1, now=_NOW), run_committee(run2, now=_NOW)
        assert _ranked_scores(d1, run1) == _ranked_scores(d2, run2) == [90.0, 40.0]

    def test_validated_edge_breaks_ties_above_score_only(self):
        # Equal decision_score: neither has a validated edge (both None here),
        # so ranking falls back deterministically to risk budget.
        batch = [_proposal(strategy_id="a", total=50.0, risk_budget=0.5),
                 _proposal(strategy_id="b", total=50.0, risk_budget=2.0)]
        d = run_committee(batch, now=_NOW)
        by_id = {p.proposal_id: p for p in batch}
        first = by_id[d.ranking_snapshot[0]]
        assert first.suggested_risk_budget_pct == 2.0

    def test_rank_key_uses_only_contract_values(self):
        # Two proposals differing ONLY by strategy identity produce equal keys.
        a = read_comparison(_proposal(strategy_id="x", total=70.0))
        b = read_comparison(_proposal(strategy_id="y", total=70.0))
        assert rank_key(a) == rank_key(b)


# ---------------------------------------------------------------------------
# 0.6 — selection supports zero / one / multiple
# ---------------------------------------------------------------------------

class TestSelectionArity:
    def test_zero_selected_when_all_non_actionable(self):
        p = _proposal(direction=Direction.NO_TRADE, intent=ExecutionIntent.NO_ACTION)
        d = run_committee([p], now=_NOW)
        assert d.selected == ()
        assert _rejected_by_step(d)[p.proposal_id] == "not_actionable"

    def test_exactly_one_selected(self):
        d = run_committee([_proposal(total=80.0)], now=_NOW)
        assert len(d.selected) == 1

    def test_multiple_selected_in_one_cycle(self):
        batch = [_proposal(strategy_id="a", total=90.0),
                 _proposal(strategy_id="b", total=70.0),
                 _proposal(strategy_id="c", total=50.0)]
        d = run_committee(batch, now=_NOW)
        assert len(d.selected) == 3
        # execution_priority follows ranking: 1,2,3 in descending score order.
        prio = {s.proposal_id: s.execution_priority for s in d.selected}
        by_id = {p.proposal_id: p for p in batch}
        ordered = sorted(prio, key=lambda pid: prio[pid])
        assert [by_id[pid].decision_score.total for pid in ordered] == [90.0, 70.0, 50.0]

    def test_selected_carries_allocated_size(self):
        d = run_committee([_proposal(size=250.0)], now=_NOW)
        assert d.selected[0].allocated_size == 250.0


# ---------------------------------------------------------------------------
# 0.7 — orchestrator invariants
# ---------------------------------------------------------------------------

class TestOrchestratorInvariants:
    def test_every_proposal_in_exactly_one_of_selected_or_rejected(self):
        batch = [
            _proposal(strategy_id="ok", total=80.0),
            _proposal(strategy_id="exp", valid_minutes=1),
            _proposal(strategy_id="edge", edge_cat=EdgeCategory.C),
            _proposal(strategy_id="hold", direction=Direction.NO_TRADE,
                      intent=ExecutionIntent.NO_ACTION),
        ]
        d = run_committee(batch, now=_GEN + timedelta(minutes=5))
        sel = {s.proposal_id for s in d.selected}
        rej = {r.proposal_id for r in d.rejected}
        assert sel.isdisjoint(rej)
        assert sel | rej == set(d.proposals_considered) == {p.proposal_id for p in batch}

    def test_proposals_considered_lists_full_candidate_set(self):
        batch = [_proposal(strategy_id="a"), _proposal(strategy_id="b")]
        d = run_committee(batch, now=_NOW)
        assert set(d.proposals_considered) == {p.proposal_id for p in batch}

    def test_trust_adjustments_recorded_for_surviving_proposals(self):
        p = _proposal()
        ta = TrustAdjustment(p.proposal_id, "fear_greed", 1.1, _GEN)
        d = run_committee([p], now=_NOW, trust_adjustments=[ta])
        assert d.trust_adjustments_applied == (f"fear_greed:{p.proposal_id}",)

    def test_portfolio_hard_block_rejects_actionable(self):
        from app.services.auto_committee import PortfolioConstraints
        p = _proposal()
        d = run_committee([p], now=_NOW,
                          portfolio=PortfolioConstraints(hard_block_reason="portfolio drawdown cap"))
        assert _rejected_by_step(d)[p.proposal_id] == "portfolio_risk"

    def test_shared_exposure_budget_trims_a_buy(self):
        from app.services.auto_committee import PortfolioConstraints
        p = _proposal(size=1000.0)
        d = run_committee([p], now=_NOW,
                          portfolio=PortfolioConstraints(exposure_headroom_usd=250.0))
        assert d.selected[0].allocated_size == 250.0


# ---------------------------------------------------------------------------
# Auto Certification Gate properties reachable in Phase 0
# ---------------------------------------------------------------------------

class TestCertificationProperties:
    def test_committee_never_mutates_a_proposal(self):
        # Snapshot every Comparison Contract value, run a full cycle mixing
        # selection/rejection/supersession, assert unchanged.
        older = _proposal(bot_id=1, strategy_id="mr", gen=_GEN, total=30.0)
        newer = _proposal(bot_id=1, strategy_id="mr", gen=_GEN + timedelta(seconds=1), total=88.0)
        winner = _proposal(bot_id=2, strategy_id="tf", total=95.0)
        dead = _proposal(bot_id=3, strategy_id="db", edge_cat=EdgeCategory.C)
        batch = [older, newer, winner, dead]
        before = [read_comparison(p) for p in batch]
        run_committee(batch, now=_NOW + timedelta(seconds=1))
        after = [read_comparison(p) for p in batch]
        assert before == after

    def test_committee_decision_reproducible(self):
        batch = [_proposal(strategy_id="a", total=90.0),
                 _proposal(strategy_id="b", total=40.0)]
        d1 = run_committee(batch, now=_NOW)
        d2 = run_committee(batch, now=_NOW)
        assert d1 == d2
        assert d1.decision_id == d2.decision_id

    def test_every_rejection_names_a_step_and_reason(self):
        batch = [_proposal(strategy_id="exp", valid_minutes=1),
                 _proposal(strategy_id="edge", edge_cat=EdgeCategory.C)]
        d = run_committee(batch, now=_GEN + timedelta(minutes=5))
        for r in d.rejected:
            assert r.rejection_step and r.rejection_reason

    def test_disagreement_does_not_stall_committee(self):
        # One BUY, one SELL (a REDUCE/CLOSE intent) on nominally the same asset:
        # both are actionable and get selected/ranked, not silently dropped.
        buy = _proposal(strategy_id="a", direction=Direction.BUY,
                        intent=ExecutionIntent.OPEN_POSITION, total=70.0)
        sell = _proposal(strategy_id="b", direction=Direction.SELL,
                         intent=ExecutionIntent.CLOSE_POSITION, total=80.0)
        d = run_committee([buy, sell], now=_NOW)
        assert len(d.selected) == 2  # resolved by ranking/allocation, not a freeze


# ---------------------------------------------------------------------------
# Phase 1 — portfolio risk wiring (step 5) evaluated across the whole decision
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.auto_committee import PortfolioConstraints, resolve_portfolio_constraints
from app.services.auto_committee.portfolio import _HEADROOM_PROBE_USD


def _alloc(decision):
    return {s.proposal_id: s.allocated_size for s in decision.selected}


class TestCombinedExposureAllocation:
    """The exposure cap is a single shared budget evaluated across the complete
    committee decision, deterministically and independent of execution order."""

    def test_combined_exposure_capped_across_the_decision(self):
        # Two buys, each $600, individually fit a $1000 budget but JOINTLY
        # ($1200) exceed it — the design's multi-selection sequencing case.
        hi = _proposal(strategy_id="hi", total=90.0, size=600.0)
        lo = _proposal(strategy_id="lo", total=50.0, size=600.0)
        d = run_committee([hi, lo], now=_NOW,
                          portfolio=PortfolioConstraints(exposure_headroom_usd=1000.0, min_order_usd=10.0))
        alloc = _alloc(d)
        # Higher rank gets its full 600; the lower-ranked gets the remaining 400.
        assert alloc[hi.proposal_id] == 600.0
        assert alloc[lo.proposal_id] == 400.0
        assert sum(v for v in alloc.values()) == pytest.approx(1000.0)  # never exceeds budget

    def test_lower_ranked_rejected_when_budget_exhausted(self):
        hi = _proposal(strategy_id="hi", total=90.0, size=1000.0)
        lo = _proposal(strategy_id="lo", total=50.0, size=1000.0)
        d = run_committee([hi, lo], now=_NOW,
                          portfolio=PortfolioConstraints(exposure_headroom_usd=1000.0, min_order_usd=10.0))
        assert _alloc(d)[hi.proposal_id] == 1000.0
        assert _rejected_by_step(d)[lo.proposal_id] == "portfolio_risk"

    def test_outcome_independent_of_batch_order(self):
        # Distinct ranks + a binding shared budget: shuffling the input batch
        # must not change which proposals are selected or their allocations.
        a = _proposal(strategy_id="a", total=90.0, size=600.0)
        b = _proposal(strategy_id="b", total=70.0, size=600.0)
        c = _proposal(strategy_id="c", total=50.0, size=600.0)
        pc = PortfolioConstraints(exposure_headroom_usd=1000.0, min_order_usd=10.0)
        results = []
        for batch in ([a, b, c], [c, a, b], [b, c, a], [c, b, a]):
            results.append(_alloc(run_committee(batch, now=_NOW, portfolio=pc)))
        assert all(r == results[0] for r in results)
        # a full (600), b trimmed to remaining (400), c rejected (0 left).
        assert results[0] == {a.proposal_id: 600.0, b.proposal_id: 400.0}

    def test_exact_tie_group_splits_proportionally(self):
        # Two buys with identical Comparison Contract values compete for a $600
        # budget: split proportionally to suggested size (identity-blind, order-
        # independent), never an arbitrary pick.
        x = _proposal(strategy_id="x", total=50.0, size=300.0, risk_budget=1.0)
        y = _proposal(strategy_id="y", total=50.0, size=100.0, risk_budget=1.0)
        d = run_committee([x, y], now=_NOW,
                          portfolio=PortfolioConstraints(exposure_headroom_usd=200.0, min_order_usd=10.0))
        alloc = _alloc(d)
        # 200 split 3:1 by suggested size -> 150 / 50.
        assert alloc[x.proposal_id] == pytest.approx(150.0)
        assert alloc[y.proposal_id] == pytest.approx(50.0)

    def test_sells_do_not_consume_buy_exposure_budget(self):
        buy = _proposal(strategy_id="buy", total=60.0, size=1000.0,
                        intent=ExecutionIntent.OPEN_POSITION)
        sell = _proposal(strategy_id="sell", total=90.0, size=1000.0,
                         direction=Direction.SELL, intent=ExecutionIntent.CLOSE_POSITION)
        d = run_committee([buy, sell], now=_NOW,
                          portfolio=PortfolioConstraints(exposure_headroom_usd=500.0, min_order_usd=10.0))
        alloc = _alloc(d)
        assert alloc[sell.proposal_id] == 1000.0   # sell unaffected by buy budget
        assert alloc[buy.proposal_id] == 500.0     # buy trimmed to the budget

    def test_capacity_block_and_cap(self):
        blocked = _proposal(strategy_id="blk", total=80.0, size=500.0)
        capped = _proposal(strategy_id="cap", total=60.0, size=500.0)
        pc = PortfolioConstraints(
            capacity_block={blocked.proposal_id: "trend_following at capacity"},
            capacity_cap={capped.proposal_id: 200.0},
        )
        d = run_committee([blocked, capped], now=_NOW, portfolio=pc)
        assert _rejected_by_step(d)[blocked.proposal_id] == "strategy_capacity"
        assert _alloc(d)[capped.proposal_id] == 200.0

    def test_none_headroom_is_unlimited(self):
        p = _proposal(size=10_000.0)
        d = run_committee([p], now=_NOW, portfolio=PortfolioConstraints(exposure_headroom_usd=None))
        assert _alloc(d)[p.proposal_id] == 10_000.0


class TestResolvePortfolioConstraints:
    """resolve_portfolio_constraints reuses the unchanged services; it reads
    back exactly what they compute (no reimplementation, no behavior drift)."""

    def _risk(self, monkeypatch, check_result):
        from app.services.auto_committee import portfolio as pmod
        inst = SimpleNamespace(check_portfolio_risk=AsyncMock(return_value=check_result))
        monkeypatch.setattr(pmod, "PortfolioRiskService", lambda session: inst)
        return inst

    def _capacity(self, monkeypatch, cap_result):
        from app.services.auto_committee import portfolio as pmod
        inst = SimpleNamespace(check_capacity_for_trade=AsyncMock(return_value=cap_result))
        monkeypatch.setattr(pmod, "StrategyCapacityService", lambda session: inst)
        return inst

    @pytest.mark.asyncio
    async def test_extracts_exposure_headroom_from_resize(self, monkeypatch):
        from app.services.portfolio_risk import PortfolioRiskCheck
        from app.services.strategy_capacity import CapacityCheck
        self._risk(monkeypatch, PortfolioRiskCheck(
            ok=True, violated_cap="exposure", details={}, action="resize", adjusted_amount=750.0))
        self._capacity(monkeypatch, CapacityCheck(ok=True, reason="", adjusted_amount=None))
        pc = await resolve_portfolio_constraints([_proposal(size=100.0)], session=object())
        assert pc.exposure_headroom_usd == 750.0
        assert pc.hard_block_reason is None

    @pytest.mark.asyncio
    async def test_hard_block_from_drawdown(self, monkeypatch):
        from app.services.portfolio_risk import PortfolioRiskCheck
        from app.services.strategy_capacity import CapacityCheck
        self._risk(monkeypatch, PortfolioRiskCheck(
            ok=False, violated_cap="drawdown", details={}, action="block", adjusted_amount=None))
        self._capacity(monkeypatch, CapacityCheck(ok=True, reason="", adjusted_amount=None))
        pc = await resolve_portfolio_constraints([_proposal()], session=object())
        assert pc.hard_block_reason == "portfolio drawdown cap"

    @pytest.mark.asyncio
    async def test_allow_means_unlimited_headroom(self, monkeypatch):
        from app.services.portfolio_risk import PortfolioRiskCheck
        from app.services.strategy_capacity import CapacityCheck
        self._risk(monkeypatch, PortfolioRiskCheck(
            ok=True, violated_cap=None, details={}, action="allow", adjusted_amount=_HEADROOM_PROBE_USD))
        self._capacity(monkeypatch, CapacityCheck(ok=True, reason="", adjusted_amount=None))
        pc = await resolve_portfolio_constraints([_proposal()], session=object())
        assert pc.exposure_headroom_usd is None

    @pytest.mark.asyncio
    async def test_capacity_block_recorded(self, monkeypatch):
        from app.services.portfolio_risk import PortfolioRiskCheck
        from app.services.strategy_capacity import CapacityCheck
        self._risk(monkeypatch, PortfolioRiskCheck(
            ok=True, violated_cap=None, details={}, action="allow", adjusted_amount=_HEADROOM_PROBE_USD))
        self._capacity(monkeypatch, CapacityCheck(ok=False, reason="at capacity", adjusted_amount=None))
        p = _proposal(intent=ExecutionIntent.OPEN_POSITION)
        pc = await resolve_portfolio_constraints([p], session=object())
        assert pc.capacity_block[p.proposal_id] == "at capacity"

    @pytest.mark.asyncio
    async def test_empty_batch_unconstrained(self):
        pc = await resolve_portfolio_constraints([], session=object())
        assert pc.hard_block_reason is None and pc.exposure_headroom_usd is None
