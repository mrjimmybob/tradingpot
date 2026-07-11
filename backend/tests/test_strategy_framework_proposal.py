"""Unit tests for Pillar 10's StrategyProposal schema
(add-strategy-decision-framework, Phase 0.10).

Covers: field completeness, deterministic proposal_id, direction/
execution_intent pairing enforcement, runtime immutability, validity
expiration, the Comparison Contract, expected_edge_estimate sourcing
enforcement, and mechanically-derived reasons.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import MappingProxyType

import pytest

from app.services.strategy_framework.decision_score import (
    DecisionScoreEngine,
    EvidenceItem,
)
from app.services.strategy_framework.edge_management import EdgeCategory, EdgeStatus
from app.services.strategy_framework.market_suitability import MarketSuitabilityResult
from app.services.strategy_framework.proposal import (
    Direction,
    EdgeEstimate,
    ExecutionIntent,
    InvalidProposalError,
    ProposalValidity,
    StrategyProposal,
    VALIDATED_EDGE_SOURCE,
    derive_reasons,
)

GENERATED_AT = datetime(2026, 1, 1, 12, 0, 0)


def _validity(generated_at: datetime = GENERATED_AT, minutes: int = 5) -> ProposalValidity:
    return ProposalValidity(generated_at=generated_at, valid_until=generated_at + timedelta(minutes=minutes))


def _suitability(is_suitable: bool = True) -> MarketSuitabilityResult:
    return MarketSuitabilityResult(
        is_suitable=is_suitable,
        regime_tags=["trend_up"],
        allowed_regimes=["trend_up"],
        matched_tags=["trend_up"] if is_suitable else [],
        reason="test",
    )


def _edge_status(category: EdgeCategory = EdgeCategory.NONE) -> EdgeStatus:
    return EdgeStatus(
        category=category,
        action="continue",
        signals=[],
        reason="test",
        can_adapt=False,
        should_wait=False,
        should_stop=False,
        evaluated_at=GENERATED_AT,
    )


def _decision_score():
    engine = DecisionScoreEngine()
    item = EvidenceItem(
        name="Trend strength",
        measurement=lambda d: d["slope"],
        normalization=lambda r: max(-1.0, min(1.0, r)),
        weight=80.0,
        reason="theory doc section 2",
    )
    return engine.score("trend_following", [item], {"slope": 1.0}, threshold=50.0)


def _make_proposal(**overrides) -> StrategyProposal:
    kwargs = dict(
        strategy_id="trend_following",
        bot_id=42,
        generated_at=GENERATED_AT,
        direction=Direction.BUY,
        execution_intent=ExecutionIntent.OPEN_POSITION,
        validity=_validity(),
        decision_score=_decision_score(),
        market_suitability=_suitability(),
        edge_status=_edge_status(),
        assumptions=("trend direction unchanged since entry",),
        suggested_position_size=100.0,
        suggested_risk_budget_pct=0.01,
    )
    kwargs.update(overrides)
    return StrategyProposal(**kwargs)


class TestConstruction:
    def test_valid_proposal_constructs(self):
        proposal = _make_proposal()
        assert proposal.strategy_id == "trend_following"
        assert proposal.direction == Direction.BUY

    def test_missing_required_field_is_typeerror(self):
        with pytest.raises(TypeError):
            StrategyProposal(strategy_id="x")  # missing everything else

    def test_empty_strategy_id_rejected(self):
        with pytest.raises(InvalidProposalError):
            _make_proposal(strategy_id="  ")

    def test_positional_construction_is_disallowed(self):
        """kw_only=True: even a fully-specified positional call must fail."""
        with pytest.raises(TypeError):
            StrategyProposal(
                "trend_following", 42, GENERATED_AT, Direction.BUY,
                ExecutionIntent.OPEN_POSITION, _validity(), _decision_score(),
                _suitability(), _edge_status(),
            )


class TestDeterministicProposalId:
    def test_same_inputs_produce_same_id(self):
        p1 = _make_proposal()
        p2 = _make_proposal()
        assert p1.proposal_id == p2.proposal_id
        assert p1.proposal_id != ""

    def test_different_bot_id_produces_different_id(self):
        p1 = _make_proposal(bot_id=1)
        p2 = _make_proposal(bot_id=2)
        assert p1.proposal_id != p2.proposal_id

    def test_different_timestamp_produces_different_id(self):
        p1 = _make_proposal()
        p2 = _make_proposal(
            generated_at=GENERATED_AT + timedelta(minutes=1),
            validity=_validity(GENERATED_AT + timedelta(minutes=1)),
        )
        assert p1.proposal_id != p2.proposal_id

    def test_id_is_not_a_random_uuid(self):
        # Deterministic hash output is stable and repeatable, unlike uuid4().
        proposal = _make_proposal()
        assert len(proposal.proposal_id) == 24
        assert all(c in "0123456789abcdef" for c in proposal.proposal_id)


class TestExecutionIntentPairing:
    @pytest.mark.parametrize(
        "direction,intent",
        [
            (Direction.NO_TRADE, ExecutionIntent.NO_ACTION),
            (Direction.HOLD, ExecutionIntent.HOLD_POSITION),
            (Direction.BUY, ExecutionIntent.OPEN_POSITION),
            (Direction.BUY, ExecutionIntent.ADD_TO_POSITION),
            (Direction.SELL, ExecutionIntent.REDUCE_POSITION),
            (Direction.SELL, ExecutionIntent.CLOSE_POSITION),
        ],
    )
    def test_all_valid_pairings_accepted(self, direction, intent):
        proposal = _make_proposal(direction=direction, execution_intent=intent)
        assert proposal.direction == direction
        assert proposal.execution_intent == intent

    @pytest.mark.parametrize(
        "direction,intent",
        [
            (Direction.NO_TRADE, ExecutionIntent.OPEN_POSITION),
            (Direction.BUY, ExecutionIntent.CLOSE_POSITION),
            (Direction.HOLD, ExecutionIntent.NO_ACTION),
            (Direction.SELL, ExecutionIntent.OPEN_POSITION),
        ],
    )
    def test_invalid_pairings_rejected(self, direction, intent):
        with pytest.raises(InvalidProposalError):
            _make_proposal(direction=direction, execution_intent=intent)


class TestValidityAndExpiration:
    def test_not_expired_before_valid_until(self):
        proposal = _make_proposal()
        assert proposal.is_expired(GENERATED_AT + timedelta(minutes=1)) is False

    def test_expired_at_or_after_valid_until(self):
        proposal = _make_proposal()
        assert proposal.is_expired(GENERATED_AT + timedelta(minutes=5)) is True
        assert proposal.is_expired(GENERATED_AT + timedelta(minutes=10)) is True

    def test_validity_generated_at_mismatch_rejected(self):
        with pytest.raises(InvalidProposalError):
            _make_proposal(validity=_validity(GENERATED_AT + timedelta(seconds=1)))

    def test_valid_until_before_generated_at_rejected(self):
        with pytest.raises(InvalidProposalError):
            ProposalValidity(generated_at=GENERATED_AT, valid_until=GENERATED_AT - timedelta(seconds=1))


class TestImmutability:
    def test_top_level_field_reassignment_raises(self):
        proposal = _make_proposal()
        with pytest.raises(Exception):
            proposal.direction = Direction.SELL

    def test_assumptions_is_a_tuple_not_a_list(self):
        proposal = _make_proposal(assumptions=["a", "b"])
        assert isinstance(proposal.assumptions, tuple)
        assert proposal.assumptions == ("a", "b")

    def test_adaptive_parameters_used_is_read_only(self):
        proposal = _make_proposal(adaptive_parameters_used={"atr_mult": 2.0})
        assert isinstance(proposal.adaptive_parameters_used, MappingProxyType)
        with pytest.raises(TypeError):
            proposal.adaptive_parameters_used["atr_mult"] = 99.0

    def test_explanation_is_read_only(self):
        proposal = _make_proposal(explanation={"strategy": "trend_following", "decision": "buy"})
        assert isinstance(proposal.explanation, MappingProxyType)
        with pytest.raises(TypeError):
            proposal.explanation["decision"] = "sell"

    def test_mutating_caller_supplied_list_after_construction_does_not_affect_proposal(self):
        source_assumptions = ["trend intact"]
        proposal = _make_proposal(assumptions=source_assumptions)
        source_assumptions.append("mutated after construction")
        assert proposal.assumptions == ("trend intact",)

    def test_empty_assumption_string_rejected(self):
        with pytest.raises(InvalidProposalError):
            _make_proposal(assumptions=("",))


class TestComparisonContract:
    def test_contains_only_documented_fields(self):
        proposal = _make_proposal()
        contract = proposal.comparison_contract()
        assert set(contract.keys()) == {
            "direction", "execution_intent", "decision_score_total",
            "decision_score_threshold", "suggested_risk_budget_pct",
            "expected_edge_estimate", "market_suitability_is_suitable",
            "edge_status_category", "validity_valid_until",
        }

    def test_never_exposes_raw_evidence_items(self):
        """The Comparison Contract must not leak strategy-internal
        Evidence Item detail - only the aggregate decision_score fields."""
        proposal = _make_proposal()
        contract = proposal.comparison_contract()
        assert "contributions" not in contract
        assert "decision_score" not in contract


class TestExpectedEdgeEstimateSourcing:
    def test_none_by_default(self):
        proposal = _make_proposal()
        assert proposal.expected_edge_estimate is None

    def test_valid_source_accepted(self):
        estimate = EdgeEstimate(
            expectancy=0.01, win_rate=0.55, profit_factor=1.4,
            sample_size=200, source=VALIDATED_EDGE_SOURCE,
        )
        proposal = _make_proposal(expected_edge_estimate=estimate)
        assert proposal.expected_edge_estimate is estimate

    def test_self_computed_source_rejected(self):
        """A strategy trying to self-report its own edge estimate is
        mechanically rejected - the sourcing rule cannot be bypassed."""
        with pytest.raises(InvalidProposalError):
            EdgeEstimate(
                expectancy=0.01, win_rate=0.55, profit_factor=1.4,
                sample_size=200, source="self_computed_rolling_window",
            )

    def test_invalid_win_rate_rejected(self):
        with pytest.raises(InvalidProposalError):
            EdgeEstimate(
                expectancy=0.01, win_rate=1.5, profit_factor=1.4,
                sample_size=200, source=VALIDATED_EDGE_SOURCE,
            )


class TestDeriveReasons:
    def test_derives_from_decision_score_contributions(self):
        engine = DecisionScoreEngine()
        items = [
            EvidenceItem(
                name="Trend strength", measurement=lambda d: 1.0,
                normalization=lambda r: r, weight=20.0, reason="theory doc",
            ),
            EvidenceItem(
                name="Nearby resistance", measurement=lambda d: -0.5,
                normalization=lambda r: r, weight=10.0, reason="theory doc",
            ),
        ]
        result = engine.score("trend_following", items, {}, threshold=10.0)
        reasons_for, reasons_against = derive_reasons(result)
        assert len(reasons_for) == 1
        assert "Trend strength" in reasons_for[0]
        assert len(reasons_against) == 1
        assert "Nearby resistance" in reasons_against[0]

    def test_reasons_are_usable_directly_on_a_proposal(self):
        engine = DecisionScoreEngine()
        items = [
            EvidenceItem(
                name="Trend strength", measurement=lambda d: 1.0,
                normalization=lambda r: r, weight=80.0, reason="theory doc",
            ),
        ]
        result = engine.score("trend_following", items, {}, threshold=10.0)
        reasons_for, reasons_against = derive_reasons(result)
        proposal = _make_proposal(decision_score=result, reasons_for=reasons_for, reasons_against=reasons_against)
        assert proposal.reasons_for == reasons_for
