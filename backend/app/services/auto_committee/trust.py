"""The `TrustAdjustment` schema (External Trust Layer — Auto-owned).

Per `design.md`'s "External Trust Layer": external information belongs
exclusively to Auto, never to strategies. When future sources (Fear & Greed,
news, macro, funding, exchange health, ...) are built — none are built here —
they produce `TrustAdjustment` records that influence RANKING ONLY (Committee
Process step 6/7), and NEVER rewrite any `StrategyProposal` field.

Phase 0 defines the schema only; no source implementations exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

try:
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from app.services.strategy_framework.proposal import StrategyProposal


@dataclass(frozen=True)
class TrustAdjustment:
    """One external trust adjustment referencing a proposal by id.

    Immutable, like every audit record in this architecture. `adjustment` is a
    ranking-only multiplier/delta (its exact interpretation is fixed when the
    ranking wiring consumes it, Phase 3) — it is never applied to a proposal
    field.
    """

    proposal_id: str
    source: str
    adjustment: float
    generated_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, str) or not self.proposal_id.strip():
            raise ValueError("TrustAdjustment.proposal_id must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("TrustAdjustment.source must be a non-empty string")


class TrustProvider(Protocol):
    """Where `TrustAdjustment` records come from for a committee cycle.

    A future external source (Fear & Greed, news, macro, funding, exchange
    health, …) implements this to return adjustments keyed to proposals by id.
    None are built by this change. Keyed by `proposal_id` only — a provider
    never receives a `StrategyProposal`, so it cannot read strategy internals.
    """

    async def get_adjustments(
        self, proposal_ids: Sequence[str]
    ) -> Sequence[TrustAdjustment]:  # pragma: no cover - interface only
        ...


class NeutralTrustProvider:
    """Production default: no external trust source exists, so no adjustments
    are produced. With this provider the committee's ranking is unaffected —
    behaviour-identical to a cycle with no trust layer (Phase 2)."""

    async def get_adjustments(self, proposal_ids: Sequence[str]) -> Tuple[TrustAdjustment, ...]:
        return ()


async def resolve_trust_adjustments(
    proposals: Sequence["StrategyProposal"],
    *,
    provider: Optional[TrustProvider] = None,
) -> Tuple[TrustAdjustment, ...]:
    """Resolve the cycle's trust adjustments from a provider (default: none).

    Mirrors `resolve_portfolio_constraints`: the async, stateful source access
    happens here, so the ten-step `run_committee` stays a pure, deterministic
    function that consumes the resolved records. The default `NeutralTrustProvider`
    yields nothing, so production behaviour is unchanged from Phase 2.
    """
    provider = provider or NeutralTrustProvider()
    proposal_ids = [p.proposal_id for p in proposals]
    return tuple(await provider.get_adjustments(proposal_ids))
