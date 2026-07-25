"""The `CommitteeDecision` schema — Auto's sole, immutable output.

Finalizes the shape `add-strategy-decision-framework` stubbed as a future
contract. A `CommitteeDecision` is the only object the execution pipeline
accepts from Auto (exactly as it accepts a single translated proposal from the
Standalone Adapter today). It references proposals by `proposal_id` and never
edits them: strategies own analysis, Auto owns selection, recorded separately.

Immutable once produced, for the same reason every `StrategyProposal` is: a
decision, once made, is a permanent audit record, not a draft. A new cycle
produces a new `CommitteeDecision`, never an edit to a prior one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class SelectedAllocation:
    """One proposal chosen for execution, with its committee-assigned size and
    priority. `allocated_size` is the committee's allocation (capped by the
    portfolio risk step when wired, Phase 1); `execution_priority` is 1-based,
    lower = executed first."""

    proposal_id: str
    allocated_size: Optional[float]
    execution_priority: int


@dataclass(frozen=True)
class RejectedProposal:
    """One considered proposal that was not selected, attributed to exactly one
    Committee Process step with a measurable reason (never an unmeasurable
    judgement — the same explainability discipline the framework requires of
    Evidence Items)."""

    proposal_id: str
    rejection_step: str
    rejection_reason: str


def _deterministic_decision_id(evaluated_at: datetime, proposal_ids: Sequence[str]) -> str:
    """Deterministic id from the cycle timestamp and the exact set of proposals
    considered — never a random UUID, so a replayed cycle against the same
    inputs reproduces the same id (design.md's determinism standard)."""
    canonical = evaluated_at.isoformat() + "|" + "|".join(sorted(proposal_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class CommitteeDecision:
    """The immutable result of one committee evaluation cycle."""

    decision_id: str = field(init=False, default="")
    evaluated_at: datetime = field(kw_only=True)
    proposals_considered: Tuple[str, ...] = field(kw_only=True, default=())
    selected: Tuple[SelectedAllocation, ...] = field(kw_only=True, default=())
    rejected: Tuple[RejectedProposal, ...] = field(kw_only=True, default=())
    trust_adjustments_applied: Tuple[str, ...] = field(kw_only=True, default=())
    ranking_snapshot: Tuple[str, ...] = field(kw_only=True, default=())

    def __post_init__(self) -> None:
        # Enforce immutable, hashable collections regardless of what was passed.
        object.__setattr__(self, "proposals_considered", tuple(self.proposals_considered))
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "trust_adjustments_applied", tuple(self.trust_adjustments_applied))
        object.__setattr__(self, "ranking_snapshot", tuple(self.ranking_snapshot))
        object.__setattr__(
            self,
            "decision_id",
            _deterministic_decision_id(self.evaluated_at, self.proposals_considered),
        )
