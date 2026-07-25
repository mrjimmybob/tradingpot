"""Phase 1 — wire Committee Process step 5 to the existing portfolio services.

The committee evaluates portfolio constraints across the COMPLETE decision, not
independently per proposal: the max-total-exposure cap is a single shared
budget that all selected buys draw from together. To keep the ten-step process
a pure, deterministic function (design.md Migration Plan), the async, stateful
service calls happen HERE — resolving current portfolio state once per cycle
into an immutable `PortfolioConstraints` value — and the pure orchestrator
(`process.run_committee`) then consumes that value deterministically.

Nothing here re-implements portfolio risk logic: it calls the unchanged
`PortfolioRiskService` / `StrategyCapacityService` and reads back exactly the
numbers those services compute (the loss/drawdown block verdict, the service's
own `remaining_capacity`, and per-strategy capacity verdicts). Auto only ever
accepts, rejects, or trims a strategy's own suggested size — it never invents
an allocation or optimises a portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from app.services.portfolio_risk import PortfolioRiskService
from app.services.strategy_capacity import StrategyCapacityService
from app.services.strategy_framework.proposal import ExecutionIntent

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.services.strategy_framework.proposal import StrategyProposal

# Mirrors the execution pipeline's MIN_ORDER_USD (trading_engine). A buy trimmed
# below this by the shared exposure budget cannot execute, so it is rejected
# rather than emitted as an un-executable order.
DEFAULT_MIN_ORDER_USD = 10.0

# execution_intents that add long exposure (draw from the shared exposure
# budget). Matches the Standalone Adapter's buy mapping; REDUCE/CLOSE are sells
# and do not consume buy-side headroom.
BUY_INTENTS = frozenset({ExecutionIntent.OPEN_POSITION, ExecutionIntent.ADD_TO_POSITION})

# A deliberately huge probe order used ONLY to make the unchanged
# PortfolioRiskService reveal its own `remaining_capacity` (the exact shared
# exposure headroom) instead of re-deriving that formula here.
_HEADROOM_PROBE_USD = 1e12


@dataclass(frozen=True)
class PortfolioConstraints:
    """Immutable snapshot of the portfolio constraints for one committee cycle.

    - `hard_block_reason`: set when a loss/drawdown cap blocks ALL orders this
      cycle (order-amount-independent); every actionable proposal is rejected.
    - `exposure_headroom_usd`: the single shared max-total-exposure budget all
      selected buys draw from together; `None` means the exposure cap is
      disabled/non-binding (unlimited).
    - `capacity_block` / `capacity_cap`: per-proposal per-strategy capacity
      verdicts (hard block reason, or a resized cap) — capacity is a
      per-strategy limit, legitimately evaluated per proposal.
    """

    hard_block_reason: Optional[str] = None
    exposure_headroom_usd: Optional[float] = None
    capacity_block: Mapping[str, str] = field(default_factory=dict)
    capacity_cap: Mapping[str, float] = field(default_factory=dict)
    min_order_usd: float = DEFAULT_MIN_ORDER_USD

    def __post_init__(self) -> None:
        object.__setattr__(self, "capacity_block", MappingProxyType(dict(self.capacity_block)))
        object.__setattr__(self, "capacity_cap", MappingProxyType(dict(self.capacity_cap)))


async def resolve_portfolio_constraints(
    proposals: Sequence["StrategyProposal"],
    *,
    session: "AsyncSession",
    min_order_usd: float = DEFAULT_MIN_ORDER_USD,
) -> PortfolioConstraints:
    """Resolve the cycle's portfolio constraints by calling the existing,
    unchanged services once. Assumes the proposals form one owner's portfolio
    (Auto Mode is a per-owner portfolio decision); the portfolio-level call
    uses the first proposal's bot as the representative owner handle.
    """
    if not proposals:
        return PortfolioConstraints(min_order_usd=min_order_usd)

    risk = PortfolioRiskService(session)
    capacity = StrategyCapacityService(session)

    # --- Portfolio-level: hard block (loss/drawdown) and shared exposure
    # headroom, via one probe that makes the service reveal remaining_capacity.
    rep_bot_id = proposals[0].bot_id
    probe = await risk.check_portfolio_risk(rep_bot_id, _HEADROOM_PROBE_USD, "buy")

    hard_block_reason: Optional[str] = None
    exposure_headroom_usd: Optional[float] = None
    if probe.action == "block" and probe.violated_cap in (
        "daily_loss", "weekly_loss", "drawdown", "bot_not_found",
    ):
        hard_block_reason = f"portfolio {probe.violated_cap} cap"
    elif probe.action == "block" and probe.violated_cap == "exposure":
        exposure_headroom_usd = 0.0
    elif probe.action == "resize":
        # adjusted_amount == the service's own remaining_capacity, unless the
        # probe itself was the binding amount (headroom >= probe → unlimited).
        adj = probe.adjusted_amount
        exposure_headroom_usd = None if (adj is None or adj >= _HEADROOM_PROBE_USD) else adj
    # action == "allow" → exposure cap disabled/non-binding → None (unlimited).

    # --- Per-strategy capacity, for buys only (mirrors _execute_trade STEP 4,
    # which checks capacity only for buys).
    capacity_block: dict = {}
    capacity_cap: dict = {}
    for p in proposals:
        if p.execution_intent not in BUY_INTENTS:
            continue
        size = p.suggested_position_size or 0.0
        cap = await capacity.check_capacity_for_trade(p.bot_id, p.strategy_id, size)
        if not cap.ok:
            capacity_block[p.proposal_id] = cap.reason or "strategy capacity limit"
        elif cap.adjusted_amount is not None and cap.adjusted_amount < size:
            capacity_cap[p.proposal_id] = cap.adjusted_amount

    return PortfolioConstraints(
        hard_block_reason=hard_block_reason,
        exposure_headroom_usd=exposure_headroom_usd,
        capacity_block=capacity_block,
        capacity_cap=capacity_cap,
        min_order_usd=min_order_usd,
    )
