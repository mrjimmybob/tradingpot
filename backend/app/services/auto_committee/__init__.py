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
from .process import run_committee
from .trust import TrustAdjustment

__all__ = [
    "ComparisonView",
    "read_comparison",
    "CommitteeDecision",
    "RejectedProposal",
    "SelectedAllocation",
    "TrustAdjustment",
    "run_committee",
]
