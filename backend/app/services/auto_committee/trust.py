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
