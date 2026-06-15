"""In-memory per-bot "decision status" — the engine's latest thinking.

A lightweight, NON-persisted snapshot of what a bot's strategy is currently
doing (evaluating, warming up, holding, entering, paused by risk, ...). It exists
purely for observability: the bot detail UI reads it so an operator can tell at a
glance why a bot is or is not trading, without reading server logs.

Design constraints (intentional):
  * Only the MOST RECENT status per bot is kept — this is not an audit log.
  * State lives in process memory only; nothing is written to the database.
  * Updating is cheap and side-effect free so it can run on every loop tick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DecisionState:
    """Canonical state labels (also the strings shown in the UI).

    Kept as constants so the engine, the API and the tests all agree on the
    exact wording instead of sprinkling string literals around.
    """

    EVALUATING = "Evaluating market"
    WAITING_FOR_DATA = "Waiting for data"
    WARMING_UP = "Warming up indicators"
    HOLD = "Hold"
    BUY_SIGNAL = "Buy signal detected"
    SELL_SIGNAL = "Sell signal detected"
    ENTERING_POSITION = "Entering position"
    EXITING_POSITION = "Exiting position"
    COOLDOWN = "Cooldown active"
    RISK_LIMIT = "Risk limit reached"
    PAUSED = "Paused"


@dataclass
class DecisionStatus:
    """A single bot's most recent decision snapshot."""

    bot_id: int
    state: str
    reason: str = ""
    symbol: Optional[str] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "state": self.state,
            "reason": self.reason,
            "symbol": self.symbol,
            "score": self.score,
            "threshold": self.threshold,
            "updated_at": self.updated_at.isoformat(),
        }


# Hold-reason keyword → state mapping. Ordered: the first substring found wins,
# so more specific conditions are listed before generic ones.
_HOLD_REASON_STATES = (
    ("cooldown", DecisionState.COOLDOWN),
    ("kill switch", DecisionState.RISK_LIMIT),
    ("unavailable", DecisionState.WAITING_FOR_DATA),  # e.g. funding-rate feed down
    ("collecting", DecisionState.WARMING_UP),
    ("warmup", DecisionState.WARMING_UP),
    ("warming", DecisionState.WARMING_UP),
    ("insufficient data", DecisionState.WARMING_UP),
    ("starting new bar", DecisionState.WARMING_UP),
    ("paused", DecisionState.PAUSED),
    ("regime", DecisionState.PAUSED),
)


def derive_state_from_signal(signal) -> str:
    """Map a strategy ``TradeSignal`` to a user-facing decision state.

    ``signal`` is the engine's ``TradeSignal`` (or ``None`` when the strategy
    returned nothing). We read ``action``/``reason`` defensively so this never
    raises inside the hot loop.
    """
    if signal is None:
        return DecisionState.EVALUATING

    action = (getattr(signal, "action", "") or "").lower()
    if action == "buy":
        return DecisionState.BUY_SIGNAL
    if action == "sell":
        return DecisionState.SELL_SIGNAL

    reason = (getattr(signal, "reason", "") or "").lower()
    for keyword, state in _HOLD_REASON_STATES:
        if keyword in reason:
            return state
    return DecisionState.HOLD


class DecisionStatusStore:
    """Thread-unsafe (single asyncio loop) in-memory store of latest statuses."""

    def __init__(self) -> None:
        self._statuses: Dict[int, DecisionStatus] = {}

    def update(
        self,
        bot_id: int,
        state: str,
        reason: str = "",
        symbol: Optional[str] = None,
        score: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Record the latest decision for ``bot_id``.

        Returns ``True`` when the state label changed from the previous snapshot
        (a transition), so callers can log transitions at INFO without spamming
        a line on every identical tick.
        """
        previous = self._statuses.get(bot_id)
        changed = previous is None or previous.state != state
        self._statuses[bot_id] = DecisionStatus(
            bot_id=bot_id,
            state=state,
            reason=reason,
            symbol=symbol if symbol is not None else (previous.symbol if previous else None),
            score=score,
            threshold=threshold,
            updated_at=datetime.utcnow(),
        )
        return changed

    def update_from_signal(
        self, bot_id: int, signal, symbol: Optional[str] = None
    ) -> bool:
        """Convenience: derive the state/reason/score from a ``TradeSignal``."""
        state = derive_state_from_signal(signal)
        reason = (getattr(signal, "reason", "") if signal else "") or ""
        return self.update(
            bot_id,
            state,
            reason=reason,
            symbol=symbol,
            score=getattr(signal, "score", None) if signal else None,
            threshold=getattr(signal, "threshold", None) if signal else None,
        )

    def get(self, bot_id: int) -> Optional[DecisionStatus]:
        return self._statuses.get(bot_id)

    def clear(self, bot_id: int) -> None:
        self._statuses.pop(bot_id, None)


# Global singleton shared by the trading engine and the API router.
decision_status_store = DecisionStatusStore()
