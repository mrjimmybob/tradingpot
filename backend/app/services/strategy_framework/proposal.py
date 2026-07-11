"""Pillar 10 — The Strategy Proposal Contract.

Every strategy SHALL return a ``StrategyProposal``, not execute a trade
directly. This is the permanent, FROZEN (per design.md's "Architecture
Freeze") contract between every strategy and everything downstream — the
Standalone Adapter today, and (not built here) a future Auto Mode
investment committee. See ``openspec/changes/add-strategy-decision-
framework/design.md``'s "The Strategy Proposal Contract" for the full
specification this module implements field-for-field.

This module is infrastructure only — no strategy constructs a
``StrategyProposal`` yet (Phase 0); that migration is Phase 1-6.

Immutability: ``StrategyProposal`` is a frozen dataclass, and every
collection field is converted to an immutable container
(``tuple``/``MappingProxyType``) in ``__post_init__`` — reassigning a field
raises (frozen dataclass) AND mutating a collection in place raises
(tuples/read-only mappings), so immutability is enforced at runtime, not
only by contract. The one deliberate exception is ``explanation``: it is
stored as an already-detached plain-dict snapshot (the output of
``ExplanationBuilder.to_dict()``), never a live reference to the mutable
``DecisionExplanation``/``ExplanationBuilder`` objects those still
legitimately mutate DURING a strategy's evaluation cycle — by the time a
``StrategyProposal`` is constructed, that cycle has finished and the
snapshot is wrapped in a read-only mapping like every other collection
field.

Determinism: ``proposal_id`` is a deterministic hash of
``(bot_id, strategy_id, generated_at)`` — never a random UUID — so
re-evaluating the same historical timestamp with the same state reproduces
the identical proposal, including its identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .decision_score import DecisionScoreResult
from .edge_management import EdgeStatus
from .market_suitability import MarketSuitabilityResult


class InvalidProposalError(ValueError):
    """Raised when a StrategyProposal (or one of its component values)
    would violate the frozen contract — an invalid direction/
    execution_intent pairing, a malformed assumption, or an
    ``expected_edge_estimate`` not sourced from validated tooling."""


class Direction(str, Enum):
    """WHICH WAY the market is expected to move. Extensible to SHORT/COVER
    later (add-short-selling-support) without changing execution_intent."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NO_TRADE = "no_trade"


class ExecutionIntent(str, Enum):
    """WHAT portfolio action is recommended — orthogonal to Direction. See
    design.md's "Execution Intent" pairing table for the valid
    direction/execution_intent combinations."""

    NO_ACTION = "no_action"
    OPEN_POSITION = "open_position"
    ADD_TO_POSITION = "add_to_position"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HOLD_POSITION = "hold_position"


# The ONLY valid (direction, execution_intent) pairings, per design.md's
# Execution Intent table. A strategy attempting any other pairing is
# rejected at construction — this is what makes "certification SHALL test
# that a strategy never produces an inconsistent pairing" a guarantee of
# the schema itself, not a hope enforced only by review.
_VALID_PAIRINGS = {
    (Direction.NO_TRADE, ExecutionIntent.NO_ACTION),
    (Direction.HOLD, ExecutionIntent.HOLD_POSITION),
    (Direction.BUY, ExecutionIntent.OPEN_POSITION),
    (Direction.BUY, ExecutionIntent.ADD_TO_POSITION),
    (Direction.SELL, ExecutionIntent.REDUCE_POSITION),
    (Direction.SELL, ExecutionIntent.CLOSE_POSITION),
}

# The only legitimate source string for a validated expected-edge estimate.
# Nothing in this repository can construct a passing EdgeEstimate under any
# other name — see "Expected Edge Requires Statistical Validation".
VALIDATED_EDGE_SOURCE = "add-strategy-validation-tooling"


@dataclass(frozen=True)
class ProposalValidity:
    """When a StrategyProposal stops being executable.

    ``valid_until`` is the deterministic OUTER bound (generated_at + the
    strategy's evaluation interval). The INNER, tighter bound —
    supersession by a newer proposal for the same bot — is enforced by
    whatever holds a sequence of proposals (e.g. the Standalone Adapter),
    not by this object itself, since a single ``ProposalValidity`` has no
    visibility into later proposals.
    """

    generated_at: datetime
    valid_until: datetime

    def __post_init__(self) -> None:
        if self.valid_until <= self.generated_at:
            raise InvalidProposalError(
                "ProposalValidity.valid_until must be strictly after generated_at "
                f"(got generated_at={self.generated_at!r}, valid_until={self.valid_until!r})"
            )

    def is_expired(self, now: datetime) -> bool:
        """Auto (and the Standalone Adapter) SHALL discard, never execute,
        a proposal for which this returns True."""
        return now >= self.valid_until


@dataclass(frozen=True)
class EdgeEstimate:
    """A statistically validated expectancy/win-rate/profit-factor estimate.

    SHALL be populated ONLY from ``add-strategy-validation-tooling``'s
    walk-forward/backtest results for this exact strategy configuration —
    NEVER computed live by the strategy from its own recent trades. This is
    mechanically enforced: ``source`` must equal ``VALIDATED_EDGE_SOURCE``
    or construction raises. Until that tooling exists (it does not yet —
    Tier 2, not implemented), no code in this repository can legitimately
    construct one; every ``StrategyProposal.expected_edge_estimate`` SHALL
    remain ``None``.
    """

    expectancy: float
    win_rate: float
    profit_factor: float
    sample_size: int
    source: str

    def __post_init__(self) -> None:
        if self.source != VALIDATED_EDGE_SOURCE:
            raise InvalidProposalError(
                f"EdgeEstimate.source must be {VALIDATED_EDGE_SOURCE!r} — "
                "expected_edge_estimate SHALL NEVER be self-computed by a "
                f"strategy (got source={self.source!r}). See design.md's "
                "'Expected Edge Requires Statistical Validation'."
            )
        if self.sample_size < 1:
            raise InvalidProposalError("EdgeEstimate.sample_size must be >= 1")
        if not (0.0 <= self.win_rate <= 1.0):
            raise InvalidProposalError("EdgeEstimate.win_rate must be within [0, 1]")


def derive_reasons(
    decision_score: DecisionScoreResult,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Mechanically derive (reasons_for, reasons_against) from a
    DecisionScoreResult's Evidence Item contributions.

    ``StrategyProposal.reasons_for``/``reasons_against`` SHALL be derived
    this way, never freely authored strings (design.md: "this avoids
    reintroducing subjectivity through the back door") — a strategy calls
    this helper rather than writing its own reason text.
    """
    reasons_for = tuple(
        f"{c.name} (+{c.contribution:.1f}): {c.reason}"
        for c in decision_score.contributions
        if c.contribution >= 0
    )
    reasons_against = tuple(
        f"{c.name} ({c.contribution:.1f}): {c.reason}"
        for c in decision_score.contributions
        if c.contribution < 0
    )
    return reasons_for, reasons_against


def _deterministic_proposal_id(bot_id: int, strategy_id: str, generated_at: datetime) -> str:
    """Deterministic identifier derived from (bot_id, strategy_id,
    generated_at) — NOT a random UUID, so a replayed evaluation against the
    same historical timestamp always reproduces the same proposal_id."""
    canonical = f"{bot_id}:{strategy_id}:{generated_at.isoformat()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True, kw_only=True)
class StrategyProposal:
    """A strategy's recommendation for one evaluation cycle — never an
    order. See design.md's "Recommendation, not an order": the existing
    execution pipeline (portfolio risk, capacity, cost, viability) remains
    the sole authority on whether and how much actually executes.

    Keyword-only by design (``kw_only=True``): with ~18 fields, a
    strategy constructing a proposal MUST name every argument — there is
    no positional ordering to memorize or get wrong.
    """

    proposal_id: str = field(init=False, default="")

    # Required — no defaults. Omitting any of these is a TypeError at
    # construction, not a silently-accepted None caught later.
    strategy_id: str
    bot_id: int
    generated_at: datetime
    direction: Direction
    execution_intent: ExecutionIntent
    validity: ProposalValidity
    decision_score: DecisionScoreResult
    market_suitability: MarketSuitabilityResult
    edge_status: EdgeStatus

    # Optional / defaulted.
    assumptions: Tuple[str, ...] = ()
    reasons_for: Tuple[str, ...] = ()
    reasons_against: Tuple[str, ...] = ()
    adaptive_parameters_used: Mapping[str, float] = field(default_factory=dict)
    explanation: Mapping[str, Any] = field(default_factory=dict)

    suggested_position_size: Optional[float] = None
    suggested_risk_budget_pct: Optional[float] = None
    expected_holding_horizon: Optional[str] = None
    expected_edge_estimate: Optional[EdgeEstimate] = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
            raise InvalidProposalError("StrategyProposal.strategy_id must be a non-empty string")

        if self.validity.generated_at != self.generated_at:
            raise InvalidProposalError(
                "StrategyProposal.validity.generated_at must match "
                "StrategyProposal.generated_at exactly"
            )

        pairing = (self.direction, self.execution_intent)
        if pairing not in _VALID_PAIRINGS:
            raise InvalidProposalError(
                f"invalid direction/execution_intent pairing: "
                f"{self.direction.value} + {self.execution_intent.value} — "
                "see design.md's Execution Intent pairing table for the "
                "only valid combinations"
            )

        for label, values in (
            ("assumptions", self.assumptions),
            ("reasons_for", self.reasons_for),
            ("reasons_against", self.reasons_against),
        ):
            for entry in values:
                if not isinstance(entry, str) or not entry.strip():
                    raise InvalidProposalError(f"{label} entries must be non-empty strings")

        # Enforce runtime immutability on every collection field.
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "reasons_for", tuple(self.reasons_for))
        object.__setattr__(self, "reasons_against", tuple(self.reasons_against))
        object.__setattr__(
            self, "adaptive_parameters_used", MappingProxyType(dict(self.adaptive_parameters_used))
        )
        object.__setattr__(self, "explanation", MappingProxyType(dict(self.explanation)))

        object.__setattr__(
            self,
            "proposal_id",
            _deterministic_proposal_id(self.bot_id, self.strategy_id, self.generated_at),
        )

    def is_expired(self, now: datetime) -> bool:
        return self.validity.is_expired(now)

    def comparison_contract(self) -> Dict[str, Any]:
        """The ONLY subset of this proposal a cross-strategy comparison
        layer (a future Auto Mode) is allowed to depend on — see design.md's
        "The Comparison Contract". Never a strategy-specific indicator
        value."""
        return {
            "direction": self.direction,
            "execution_intent": self.execution_intent,
            "decision_score_total": self.decision_score.total,
            "decision_score_threshold": self.decision_score.threshold,
            "suggested_risk_budget_pct": self.suggested_risk_budget_pct,
            "expected_edge_estimate": self.expected_edge_estimate,
            "market_suitability_is_suitable": self.market_suitability.is_suitable,
            "edge_status_category": self.edge_status.category,
            "validity_valid_until": self.validity.valid_until,
        }
