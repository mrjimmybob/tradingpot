"""The ranking-adjustment policy seam (Committee Process step 6→7).

The committee owns a fixed, Comparison-Contract-only base ranking (see
`process.rank_key`). The External Trust Layer may only *adjust* that ranking —
and this change deliberately does NOT decide how. The mathematics by which a
future trust source modifies ranking (a Decision-Score multiplier, an additive
delta, a regime-conditioned weighting, anything else) is a calibration problem
for a later change, not part of this architecture.

So the committee consumes an EFFECTIVE RANKING VALUE handed to it by a
`RankingAdjustmentPolicy`; it never embeds the trust mathematics itself. This
change ships exactly one policy — `NeutralRankingPolicy` — which returns the
committee's base ranking value unchanged, so with no trust source the outcome
is behaviour-identical to Phase 2. A future change can supply a different
policy (and real `TrustAdjustment` records) to alter ranking WITHOUT modifying
the committee orchestrator, the Committee Process steps, or the
`StrategyProposal` / `CommitteeDecision` contracts.

Policy contract:
- `effective_ranking_value(base_value, adjustments)` returns an orderable value
  used ONLY to order proposals (higher sorts first, like the base key). Within
  a single cycle a policy MUST return mutually-comparable values (equal values
  are treated as a rank tie). It receives the base ranking value and the trust
  adjustments for one proposal — never a `StrategyProposal`, so it cannot read
  or mutate proposal fields.
"""
from __future__ import annotations

from typing import Any, Sequence

from .trust import TrustAdjustment

try:  # Protocol is nice-to-have for typing; not required at runtime.
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore


class RankingAdjustmentPolicy(Protocol):
    """How the External Trust Layer turns a base ranking value + a proposal's
    trust adjustments into the effective ranking value the committee orders by.
    Swappable by future work; this change embeds no such mathematics."""

    def effective_ranking_value(
        self, base_value: Any, adjustments: Sequence[TrustAdjustment]
    ) -> Any:  # pragma: no cover - interface only
        ...


class NeutralRankingPolicy:
    """Production default: trust adjustments do NOT alter ranking.

    Returns the committee's base ranking value unchanged, ignoring any
    adjustments, so a cycle with no trust source ranks exactly as it did in
    Phase 2. This is the only policy this change ships; it contains no trust
    mathematics of any kind.
    """

    def effective_ranking_value(
        self, base_value: Any, adjustments: Sequence[TrustAdjustment]
    ) -> Any:
        return base_value


# The single shipped policy and the committee's default.
NEUTRAL_RANKING_POLICY = NeutralRankingPolicy()
