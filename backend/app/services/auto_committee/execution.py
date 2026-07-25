"""Phase 2 — Committee Process step 10: submit selected proposals to the
existing, unchanged execution pipeline.

Auto Mode decides WHAT executes; the execution pipeline (`_execute_trade`)
decides HOW and remains strategy-agnostic. This module does not duplicate or
bypass any execution logic: for each `CommitteeDecision.selected` entry, in
`execution_priority` order, it reuses the Standalone Adapter's exact
proposal→signal translation and calls the same `_execute_trade` the Standalone
path calls. The ONLY difference from the Standalone path is that the order
amount is the committee's `allocated_size` (which, for a single unconstrained
Alpha strategy, equals the proposal's own `suggested_position_size` — hence
byte-identical behaviour in the degenerate one-strategy case).

Proposals are never modified; the committee references them by id, and the
amount override lives on a freshly-translated `TradeSignal`, not on any
`StrategyProposal`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable, List, Mapping, Optional

from app.services.strategy_framework.standalone_adapter import StandaloneAdapter

from .decision import CommitteeDecision

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.models import Bot, Order
    from app.services.exchange import ExchangeService
    from app.services.strategy_framework.proposal import StrategyProposal
    from app.services.trading_engine import TradingEngine

logger = logging.getLogger(__name__)


async def execute_committee_decision(
    decision: CommitteeDecision,
    proposals_by_id: Mapping[str, "StrategyProposal"],
    *,
    engine: "TradingEngine",
    exchange: "ExchangeService",
    session: "AsyncSession",
    bot_for: Callable[["StrategyProposal"], "Bot"],
    price_for: Callable[["StrategyProposal"], float],
    now: Optional[datetime] = None,
) -> List["Order"]:
    """Execute each selected proposal through the unchanged `_execute_trade`
    pipeline, in `execution_priority` order, mirroring the Standalone Adapter.

    Returns the list of `Order`s `_execute_trade` produced (skipping any that
    produced no order — an expired selection or a pipeline-declined trade),
    matching the Standalone Adapter's own "None is a valid, non-error outcome".
    """
    now = now or datetime.utcnow()
    orders: List["Order"] = []

    for sel in sorted(decision.selected, key=lambda s: s.execution_priority):
        proposal = proposals_by_id[sel.proposal_id]

        # Mirror StandaloneAdapter.execute exactly: discard an expired proposal
        # before touching the pipeline (a correct, non-error outcome).
        if proposal.is_expired(now):
            logger.info(
                "Auto committee: discarding expired selected proposal %s "
                "(strategy=%s, valid_until=%s, now=%s)",
                proposal.proposal_id, proposal.strategy_id,
                proposal.validity.valid_until, now,
            )
            continue

        # Same translation the Standalone Adapter performs (direction/reason/
        # score/threshold from the proposal; Alpha strategies never accumulate —
        # is_accumulation stays False, as every Alpha strategy's own
        # to_trade_signal call already does; only DCA, an Allocation strategy
        # excluded from the committee, sets it True).
        signal = StandaloneAdapter.to_trade_signal(proposal)
        if signal is None:
            # A selected proposal always carries an order-producing intent, so
            # this is unreachable; guard without inventing behaviour.
            continue

        # The ONLY committee-introduced difference: scale the amount to the
        # committee's allocation (== suggested_position_size when unconstrained).
        if sel.allocated_size is not None:
            signal.amount = sel.allocated_size

        bot = bot_for(proposal)
        price = price_for(proposal)
        order = await engine._execute_trade(bot, exchange, signal, price, session)
        if order is not None:
            orders.append(order)

    return orders
