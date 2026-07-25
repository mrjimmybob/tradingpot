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
from typing import Dict, List, Optional, Sequence, Tuple

from app.services.strategy_framework.edge_management import EdgeCategory
from app.services.strategy_framework.proposal import ExecutionIntent, StrategyProposal

from .comparison import ComparisonView, read_comparison
from .decision import CommitteeDecision, RejectedProposal, SelectedAllocation
from .portfolio import BUY_INTENTS, PortfolioConstraints
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

def run_committee(
    proposals: Sequence[StrategyProposal],
    *,
    now: datetime,
    portfolio: Optional[PortfolioConstraints] = None,
    trust_adjustments: Optional[Sequence[TrustAdjustment]] = None,
) -> CommitteeDecision:
    """Run the full ten-step Committee Process over one batch of (Alpha)
    proposals and produce an immutable `CommitteeDecision`.

    Pure and deterministic: given the same proposals, `now`, and `portfolio`
    constraints, it always produces the same decision — the async, stateful
    portfolio-service calls are resolved beforehand into `portfolio` (see
    `portfolio.resolve_portfolio_constraints`). `portfolio=None` means no
    constraints (unconstrained), used by Phase 0 pure-logic tests.

    `proposals` is assumed already filtered to Alpha strategies (Allocation
    strategies never enter the committee — see design.md "Strategy
    Categories"); this function does not re-filter by category.
    """
    portfolio = portfolio or PortfolioConstraints()
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

    # Step 5 — apply portfolio risk constraints (Phase 1, from the pre-resolved
    # `portfolio` snapshot). The order-independent hard blocks (loss/drawdown)
    # reject every actionable order; per-strategy capacity blocks/caps apply per
    # proposal. The SHARED max-total-exposure budget is NOT applied per-proposal
    # here — it is a portfolio-wide budget evaluated across the whole decision at
    # allocation (steps 8-9), so combined exposure is correct rather than each
    # proposal seeing the same pre-cycle exposure in isolation.
    tentative_size: Dict[str, Optional[float]] = {}
    kept = []
    for pid in survivors:
        actionable = views[pid].execution_intent in _ORDER_INTENTS
        if actionable and portfolio.hard_block_reason:
            rejected.append(RejectedProposal(pid, "portfolio_risk", portfolio.hard_block_reason))
            continue
        if pid in portfolio.capacity_block:
            rejected.append(RejectedProposal(pid, "strategy_capacity", portfolio.capacity_block[pid]))
            continue
        size = by_id[pid].suggested_position_size
        if pid in portfolio.capacity_cap:
            size = portfolio.capacity_cap[pid]
        tentative_size[pid] = size
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

    # Step 8 — allocate capital across the COMPLETE committee decision. The
    # max-total-exposure cap is a single shared budget that all selected buys
    # draw from together, consumed in deterministic ranking order (higher rank =
    # first claim). When a rank-tie group cannot be fully funded, the remaining
    # budget is split PROPORTIONALLY to each proposal's suggested size — a
    # symmetric, batch-order-independent, strategy-identity-blind rule (never an
    # arbitrary pick, never a new/invented allocation, never optimisation). This
    # is what makes the committee outcome independent of execution order.
    final_size: Dict[str, Optional[float]] = {}
    exposure_rejected: Dict[str, str] = {}
    remaining = portfolio.exposure_headroom_usd  # None == unlimited
    idx = 0
    while idx < len(ranked):
        end = idx
        key = rank_key(views[ranked[idx]])
        while end < len(ranked) and rank_key(views[ranked[end]]) == key:
            end += 1
        group = ranked[idx:end]
        # Non-buy actionable orders (sells/closes) do not draw the buy-exposure
        # budget; they pass through with their tentative size.
        for pid in group:
            if views[pid].execution_intent in _ORDER_INTENTS and views[pid].execution_intent not in BUY_INTENTS:
                final_size[pid] = tentative_size.get(pid)
        buys = [pid for pid in group if views[pid].execution_intent in BUY_INTENTS]
        if remaining is None:
            for pid in buys:
                final_size[pid] = tentative_size.get(pid)
        else:
            wants = {pid: (tentative_size.get(pid) or 0.0) for pid in buys}
            total_want = sum(wants.values())
            if total_want <= remaining:
                for pid in buys:
                    final_size[pid] = wants[pid]
                remaining -= total_want
            else:
                for pid in buys:
                    share = remaining * (wants[pid] / total_want) if total_want > 0 else 0.0
                    if share < portfolio.min_order_usd:
                        exposure_rejected[pid] = (
                            f"combined exposure budget exhausted: allocated share "
                            f"${share:.2f} < min order ${portfolio.min_order_usd:.2f}"
                        )
                    else:
                        final_size[pid] = share
                remaining = 0.0
        idx = end

    # Step 9 — select actionable survivors in rank order with their final
    # allocation; every non-selected considered proposal is recorded rejected,
    # so nothing is silently dropped. Selecting zero, one, or many are all valid.
    selected: List[SelectedAllocation] = []
    priority = 1
    for pid in ranked:
        view = views[pid]
        if view.execution_intent not in _ORDER_INTENTS:
            rejected.append(RejectedProposal(
                pid, "not_actionable",
                f"execution_intent {view.execution_intent.value} produces no order",
            ))
        elif pid in exposure_rejected:
            rejected.append(RejectedProposal(pid, "portfolio_risk", exposure_rejected[pid]))
        else:
            selected.append(SelectedAllocation(
                proposal_id=pid,
                allocated_size=final_size.get(pid),
                execution_priority=priority,
            ))
            priority += 1

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
