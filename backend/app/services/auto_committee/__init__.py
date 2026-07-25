"""Auto Mode Investment Committee (add-auto-mode-investment-committee).

The portfolio-level decision layer sitting between strategies (which produce
immutable ``StrategyProposal`` objects) and the existing, unchanged execution
pipeline. Auto never re-derives a trading decision from market data — it only
selects among, allocates capital to, and rejects proposals strategies have
already made, from the Comparison Contract alone.

Phase 0 (this package's initial slice): the Committee Core as pure logic —
the Comparison Contract reader, the ``CommitteeDecision``/``TrustAdjustment``
schemas, the reject/rank/allocate/select step functions, and the ten-step
orchestrator — with zero wiring to real strategies, portfolio services, or the
execution pipeline.
"""
from .comparison import ComparisonView, read_comparison
from .decision import CommitteeDecision, RejectedProposal, SelectedAllocation
from .execution import execute_committee_decision
from .flag import is_committee_enabled
from .portfolio import PortfolioConstraints, resolve_portfolio_constraints
from .process import run_committee
from .ranking import NEUTRAL_RANKING_POLICY, NeutralRankingPolicy, RankingAdjustmentPolicy
from .trust import (
    NeutralTrustProvider,
    TrustAdjustment,
    TrustProvider,
    resolve_trust_adjustments,
)

__all__ = [
    "ComparisonView",
    "read_comparison",
    "CommitteeDecision",
    "RejectedProposal",
    "SelectedAllocation",
    "TrustAdjustment",
    "TrustProvider",
    "NeutralTrustProvider",
    "resolve_trust_adjustments",
    "RankingAdjustmentPolicy",
    "NeutralRankingPolicy",
    "NEUTRAL_RANKING_POLICY",
    "PortfolioConstraints",
    "resolve_portfolio_constraints",
    "run_committee",
    "execute_committee_decision",
    "is_committee_enabled",
]
