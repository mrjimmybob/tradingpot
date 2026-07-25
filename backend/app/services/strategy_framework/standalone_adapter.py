"""The Standalone Adapter — translates a StrategyProposal into the
existing, UNCHANGED ``_execute_trade`` pipeline.

Per design.md's "Standalone execution flow (Auto Mode disabled — today's
behavior, preserved)": this is a thin translator, NOT Auto Mode — no
ranking/comparison logic, exactly one proposal in, one order-or-nothing
out. It:

  (a) discards the proposal if ``now >= validity.valid_until`` (Auto and
      the Standalone Adapter SHALL NEVER execute a stale proposal),
  (b) branches on ``execution_intent`` (never ``direction`` alone) to
      determine the action,
  (c) extracts order intent (direction/suggested_position_size) from the
      proposal and feeds it into the EXISTING, unchanged
      ``TradingEngine._execute_trade`` pipeline.

The adapter never rewrites the proposal (Proposal Immutability) — it only
reads it to decide whether, and how, to call the unchanged execution
pipeline. This module is infrastructure only — no strategy produces a
``StrategyProposal`` for it to consume yet (Phase 0); that migration is
Phase 1-6.

``expected_move_pct``/``expected_risk_pct``/``is_accumulation`` are NOT
``StrategyProposal`` fields (see design.md's Decisions: these per-trade
figures live inside ``decision_score``'s Evidence Report, not as top-level
proposal fields) — a Phase 1-6 strategy's own translation call supplies
them explicitly to ``to_trade_signal()`` so the existing viability gate
keeps working exactly as it does today. Leaving them unsupplied preserves
today's fee-only viability behavior for accumulation-style signals.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .proposal import Direction, ExecutionIntent, StrategyProposal

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models import Bot, Order
    from app.services.exchange import ExchangeService
    from app.services.trading_engine import TradeSignal, TradingEngine

logger = logging.getLogger(__name__)

# execution_intent values that produce NO order — a strategy is
# recommending "hold position" or "do nothing", never a trade.
_NO_ORDER_INTENTS = frozenset({ExecutionIntent.NO_ACTION, ExecutionIntent.HOLD_POSITION})

# execution_intent -> TradeSignal.action. Direction alone is intentionally
# NOT consulted here (design.md: "branches on execution_intent, not
# direction alone") — this map is the single source of truth for the
# translation.
_INTENT_TO_ACTION = {
    ExecutionIntent.OPEN_POSITION: "buy",
    ExecutionIntent.ADD_TO_POSITION: "buy",
    ExecutionIntent.REDUCE_POSITION: "sell",
    ExecutionIntent.CLOSE_POSITION: "sell",
}


class StandaloneAdapter:
    """Thin, stateless translator from StrategyProposal to the existing
    execution pipeline. Holds no per-call state — every method is a pure
    function of its arguments (aside from the deliberate side effect of
    ``execute()`` calling ``_execute_trade``)."""

    @staticmethod
    def to_trade_signal(
        proposal: StrategyProposal,
        *,
        expected_move_pct: Optional[float] = None,
        expected_risk_pct: Optional[float] = None,
        is_accumulation: bool = False,
    ) -> Optional["TradeSignal"]:
        """Pure translation: StrategyProposal -> TradeSignal, or None for
        an execution_intent that produces no order. Exposed separately
        from ``execute()`` so the translation itself is deterministically
        unit-testable without a running engine/session.
        """
        if proposal.execution_intent in _NO_ORDER_INTENTS:
            return None

        action = _INTENT_TO_ACTION.get(proposal.execution_intent)
        if action is None:
            # Structurally unreachable: StrategyProposal.__post_init__
            # already rejects any (direction, execution_intent) pairing
            # not in the documented table, and every intent in that table
            # is covered by _NO_ORDER_INTENTS or _INTENT_TO_ACTION above.
            raise ValueError(
                f"unhandled execution_intent {proposal.execution_intent!r} — "
                "this should be unreachable given StrategyProposal's own "
                "pairing validation"
            )

        reason = "; ".join(proposal.reasons_for) or "; ".join(proposal.reasons_against) or proposal.strategy_id

        # Local import: avoids a module-level import cycle (trading_engine
        # imports nothing from strategy_framework today, but keeping this
        # import lazy means this package never risks becoming a hard
        # dependency of trading_engine's own import order).
        from app.services.trading_engine import TradeSignal

        return TradeSignal(
            action=action,
            amount=proposal.suggested_position_size or 0.0,
            reason=reason,
            score=proposal.decision_score.total,
            threshold=proposal.decision_score.threshold,
            expected_move_pct=expected_move_pct,
            expected_risk_pct=expected_risk_pct,
            is_accumulation=is_accumulation,
            # Carry the source proposal so the Auto committee (Phase 5) can
            # collect it from a strategy that returns a TradeSignal. compare=False
            # on the field keeps TradeSignal equality byte-identical.
            _source_proposal=proposal,
        )

    async def execute(
        self,
        proposal: StrategyProposal,
        *,
        engine: "TradingEngine",
        bot: "Bot",
        exchange: "ExchangeService",
        current_price: float,
        session: "AsyncSession",
        now: Optional[datetime] = None,
        expected_move_pct: Optional[float] = None,
        expected_risk_pct: Optional[float] = None,
        is_accumulation: bool = False,
    ) -> Optional["Order"]:
        """Execute (or discard) exactly one proposal via the existing,
        unchanged ``_execute_trade`` pipeline.

        Returns ``None`` without calling ``_execute_trade`` at all when the
        proposal is expired or its execution_intent requires no action —
        in both cases this is a correct, non-error outcome, not a failure.
        """
        now = now or datetime.utcnow()

        if proposal.is_expired(now):
            logger.info(
                "Standalone Adapter: discarding expired proposal %s "
                "(strategy=%s, bot=%s, valid_until=%s, now=%s)",
                proposal.proposal_id, proposal.strategy_id, bot.id,
                proposal.validity.valid_until, now,
            )
            return None

        signal = self.to_trade_signal(
            proposal,
            expected_move_pct=expected_move_pct,
            expected_risk_pct=expected_risk_pct,
            is_accumulation=is_accumulation,
        )
        if signal is None:
            logger.debug(
                "Standalone Adapter: proposal %s (strategy=%s) has "
                "execution_intent=%s - no order",
                proposal.proposal_id, proposal.strategy_id, proposal.execution_intent.value,
            )
            return None

        return await engine._execute_trade(bot, exchange, signal, current_price, session)
