"""The ten-step Committee Process, as pure functions over a proposal batch.

Every step is deterministic and reads only the Comparison Contract (via
`ComparisonView`) plus identity/bookkeeping fields the orchestrator reads
directly (`bot_id`, `strategy_id`, `generated_at`, `suggested_position_size`)
— never a strategy indicator. Ranking (step 7) is strategy-identity-blind: its
key is built from Comparison Contract *values* only.

Phase 0 scope: steps 1-4 and 7-10 are implemented; step 5 (portfolio risk) and
step 6 (external trust) are injected stubs, defaulting to no-ops, so Phases 1
and 3 wire the real services without touching this orchestrator's shape.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.services.strategy_framework.edge_management import EdgeCategory
from app.services.strategy_framework.proposal import ExecutionIntent, StrategyProposal

from .comparison import ComparisonView, read_comparison
from .decision import CommitteeDecision, RejectedProposal, SelectedAllocation
from .trust import TrustAdjustment

# execution_intent values that produce a real order (mirrors the Standalone
# Adapter's own split). A proposal whose intent is neither of the no-order
# intents is "actionable" and can be selected for execution.
_ORDER_INTENTS = frozenset({
    ExecutionIntent.OPEN_POSITION,
    ExecutionIntent.ADD_TO_POSITION,
    ExecutionIntent.REDUCE_POSITION,
    ExecutionIntent.CLOSE_POSITION,
})


# ---------------------------------------------------------------------------
# Step 7 — ranking key (Comparison Contract values only; identity-blind)
# ---------------------------------------------------------------------------

def rank_key(view: ComparisonView) -> Tuple[float, float, float, float]:
    """Deterministic, strategy-identity-blind ranking key (higher sorts first).

    Documented tie-breaking policy (design.md Open Questions, chosen here):
    rank by `decision_score.total`, then — only when a *validated*
    `expected_edge_estimate` is present — its expectancy, then
    `suggested_risk_budget_pct`. Every component is a Comparison Contract
    value; strategy identity and `proposal_id` are never consulted. Exact ties
    (equal on every component) are genuinely indistinguishable by the contract
    and are resolved downstream by SPLIT ALLOCATION at selection, not by an
    arbitrary identity-based pick.
    """
    edge = view.expected_edge_estimate
    has_validated_edge = 1.0 if edge is not None else 0.0
    edge_expectancy = edge.expectancy if edge is not None else 0.0
    risk_budget = view.suggested_risk_budget_pct or 0.0
    return (view.decision_score_total, has_validated_edge, edge_expectancy, risk_budget)


# ---------------------------------------------------------------------------
# The orchestrator (steps 1-10)
# ---------------------------------------------------------------------------

# Step 5 stub signature: given a proposal and its tentative allocation, return
# (ok, capped_size, reason). ok=False rejects (portfolio_risk); a smaller
# capped_size resizes. Default no-op passes everything unchanged.
PortfolioRiskCheck = Callable[[StrategyProposal, Optional[float]], Tuple[bool, Optional[float], str]]


def _default_portfolio_risk_check(
    proposal: StrategyProposal, tentative_size: Optional[float]
) -> Tuple[bool, Optional[float], str]:
    return True, tentative_size, ""


def run_committee(
    proposals: Sequence[StrategyProposal],
    *,
    now: datetime,
    portfolio_risk_check: Optional[PortfolioRiskCheck] = None,
    trust_adjustments: Optional[Sequence[TrustAdjustment]] = None,
) -> CommitteeDecision:
    """Run the full ten-step Committee Process over one batch of (Alpha)
    proposals and produce an immutable `CommitteeDecision`.

    `proposals` is assumed already filtered to Alpha strategies (Allocation
    strategies never enter the committee — see design.md "Strategy
    Categories"); this function does not re-filter by category.
    """
    portfolio_risk_check = portfolio_risk_check or _default_portfolio_risk_check
    trust_adjustments = list(trust_adjustments or [])

    # Step 1 — collect.
    by_id: Dict[str, StrategyProposal] = {p.proposal_id: p for p in proposals}
    considered: List[str] = [p.proposal_id for p in proposals]
    views: Dict[str, ComparisonView] = {pid: read_comparison(by_id[pid]) for pid in by_id}

    rejected: List[RejectedProposal] = []
    survivors: List[str] = list(by_id.keys())

    # Step 2 — reject expired (pure timestamp comparison on validity.valid_until).
    kept: List[str] = []
    for pid in survivors:
        if now >= views[pid].valid_until:
            rejected.append(RejectedProposal(
                pid, "expired",
                f"validity.valid_until {views[pid].valid_until.isoformat()} <= now {now.isoformat()}",
            ))
        else:
            kept.append(pid)
    survivors = kept

    # Step 3 — reject superseded (newest proposal per (bot, strategy) wins; the
    # older one's assumptions are presumed stale). Identity-based bookkeeping,
    # not ranking.
    newest: Dict[Tuple[int, str], str] = {}
    for pid in survivors:
        p = by_id[pid]
        key = (p.bot_id, p.strategy_id)
        cur = newest.get(key)
        if cur is None or by_id[pid].generated_at > by_id[cur].generated_at:
            newest[key] = pid
    kept = []
    for pid in survivors:
        p = by_id[pid]
        winner = newest[(p.bot_id, p.strategy_id)]
        if pid != winner:
            rejected.append(RejectedProposal(
                pid, "superseded",
                f"superseded by newer proposal {winner} for the same bot/strategy",
            ))
        else:
            kept.append(pid)
    survivors = kept

    # Step 4 — reject edge-disqualified (edge_status.category == C). Reads a
    # field; does not diagnose the strategy.
    kept = []
    for pid in survivors:
        if views[pid].edge_status_category == EdgeCategory.C:
            rejected.append(RejectedProposal(
                pid, "edge_disqualified",
                "edge_status.category == C (edge disappeared)",
            ))
        else:
            kept.append(pid)
    survivors = kept

    # Step 5 — apply portfolio risk constraints (injected; no-op by default in
    # Phase 0). Produces the tentative allocated size for each survivor.
    tentative_size: Dict[str, Optional[float]] = {}
    kept = []
    for pid in survivors:
        p = by_id[pid]
        ok, capped, reason = portfolio_risk_check(p, p.suggested_position_size)
        if not ok:
            rejected.append(RejectedProposal(
                pid, "portfolio_risk", reason or "blocked by portfolio risk/capacity",
            ))
        else:
            tentative_size[pid] = capped
            kept.append(pid)
    survivors = kept

    # Step 6 — apply external trust adjustments (injected; empty by default).
    # Phase 0 records which adjustments reference surviving proposals for audit;
    # their effect on ranking is wired in Phase 3.
    survivor_set = set(survivors)
    trust_applied = [
        f"{ta.source}:{ta.proposal_id}" for ta in trust_adjustments
        if ta.proposal_id in survivor_set
    ]

    # Step 7 — rank (Comparison Contract values only; stable sort keeps exact
    # ties in input order, which is reproducible and identity-blind).
    ranked = sorted(survivors, key=lambda pid: rank_key(views[pid]), reverse=True)
    ranking_snapshot = list(ranked)

    # Steps 8 & 9 — allocate capital and select. Actionable proposals (their
    # execution_intent produces an order) are selected in ranked order, each
    # carrying its step-5 allocation and a 1-based execution priority. A
    # survivor whose intent produces no order is recorded rejected
    # ("not_actionable"), so every considered proposal lands in exactly one of
    # selected/rejected. Selecting zero, one, or many are all valid outcomes.
    selected: List[SelectedAllocation] = []
    priority = 1
    for pid in ranked:
        view = views[pid]
        if view.execution_intent in _ORDER_INTENTS:
            selected.append(SelectedAllocation(
                proposal_id=pid,
                allocated_size=tentative_size.get(pid),
                execution_priority=priority,
            ))
            priority += 1
        else:
            rejected.append(RejectedProposal(
                pid, "not_actionable",
                f"execution_intent {view.execution_intent.value} produces no order",
            ))

    # Deterministic ordering of the rejected list (identity-based sort is fine
    # here — this is audit ordering, not ranking).
    rejected.sort(key=lambda r: (r.rejection_step, r.proposal_id))

    # Defensive invariant: no proposal silently dropped (Auto Certification
    # Gate — every considered proposal is in exactly one of selected/rejected).
    accounted = {s.proposal_id for s in selected} | {r.proposal_id for r in rejected}
    assert accounted == set(considered), (
        "committee invariant violated: some proposal is neither selected nor rejected"
    )

    return CommitteeDecision(
        evaluated_at=now,
        proposals_considered=tuple(considered),
        selected=tuple(selected),
        rejected=tuple(rejected),
        trust_adjustments_applied=tuple(trust_applied),
        ranking_snapshot=tuple(ranking_snapshot),
    )
