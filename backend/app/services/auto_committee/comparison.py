"""The Comparison Contract reader (Committee Process input vocabulary).

`add-strategy-decision-framework` fixed the exact subset of `StrategyProposal`
fields a comparison layer may depend on; `add-auto-mode-investment-committee`
reuses it verbatim. This module is the ONLY place in the committee package
that touches a raw `StrategyProposal`: it extracts the Comparison Contract
fields into an immutable `ComparisonView`, and every ranking/tie-break/
rejection-condition function downstream operates on `ComparisonView`, never on
a proposal. That makes the design's "Auto never reads indicators" rule
structural rather than merely intended — comparison logic literally cannot see
`explanation`, Evidence Item contributions, `adaptive_parameters_used`,
`reasons_*`, `assumptions`, or any other strategy internal, because they are
not on the view at all.

`proposal_id` is carried as an identity handle so decisions can reference a
proposal, but it is NOT a ranking input — ranking is strategy-identity-blind
(see `process.rank`). The four fields the orchestrator needs for
identity/bookkeeping and allocation (`bot_id`, `strategy_id`, `generated_at`,
`suggested_position_size`) are read directly from the proposal by the
orchestrator, deliberately NOT surfaced on this view, so they can never leak
into a comparison decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from app.services.strategy_framework.edge_management import EdgeCategory
from app.services.strategy_framework.proposal import Direction, EdgeEstimate, ExecutionIntent

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from app.services.strategy_framework.proposal import StrategyProposal


# The Comparison Contract: the exact, only fields comparison logic may read.
COMPARISON_CONTRACT_FIELDS = (
    "direction",
    "execution_intent",
    "decision_score_total",
    "decision_score_threshold",
    "suggested_risk_budget_pct",
    "expected_edge_estimate",
    "market_suitability_is_suitable",
    "edge_status_category",
    "valid_until",
)


@dataclass(frozen=True)
class ComparisonView:
    """The immutable, comparison-only projection of a `StrategyProposal`.

    Exposes exactly the Comparison Contract fields (plus `proposal_id` as a
    non-ranking identity handle). No strategy internals are reachable from
    here.
    """

    proposal_id: str  # identity handle for references; NEVER a ranking input
    direction: Direction
    execution_intent: ExecutionIntent
    decision_score_total: float
    decision_score_threshold: float
    suggested_risk_budget_pct: Optional[float]
    expected_edge_estimate: Optional[EdgeEstimate]
    market_suitability_is_suitable: bool
    edge_status_category: EdgeCategory
    valid_until: datetime


def read_comparison(proposal: "StrategyProposal") -> ComparisonView:
    """Extract the Comparison Contract fields from a proposal — the single
    proposal-touching entry point for the committee package."""
    return ComparisonView(
        proposal_id=proposal.proposal_id,
        direction=proposal.direction,
        execution_intent=proposal.execution_intent,
        decision_score_total=proposal.decision_score.total,
        decision_score_threshold=proposal.decision_score.threshold,
        suggested_risk_budget_pct=proposal.suggested_risk_budget_pct,
        expected_edge_estimate=proposal.expected_edge_estimate,
        market_suitability_is_suitable=proposal.market_suitability.is_suitable,
        edge_status_category=proposal.edge_status.category,
        valid_until=proposal.validity.valid_until,
    )
