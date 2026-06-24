"""In-memory per-bot strategy diagnostics — observe-only operability data.

A sibling of ``decision_status.py``: a lightweight, NON-persisted, side-effect
free store that the trading loop WRITES to and the bot-detail UI READS. Where
``decision_status`` keeps only the single latest "what is the bot thinking right
now" snapshot, this accumulates the counters/last-events an operator needs to
answer, within seconds and without reading logs:

  * What has this bot been doing?  (evaluations, signals)
  * What is the strategy actually thinking?  (top decision reasons)
  * Why is it not trading?  (blocked-trade counters)
  * Is execution failing?  (execution outcomes)
  * Is the data feed healthy?  (market-data failures)
  * Why is it paused?  (explicit pause reason)

Hard rules (this module observes behavior, it never controls it):
  * Nothing here influences trading, risk, execution or sizing.
  * Every public method is total and exception-safe: a bug in diagnostics must
    NEVER propagate into the trading loop. Methods swallow and log their own
    errors so call sites stay clean one-liners.
  * In-memory only; counters may reset on process restart.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical blocked-trade categories (stable keys shared by engine, API, tests).
BLOCK_RISK_MANAGER = "risk_manager"
BLOCK_MIN_ORDER_SIZE = "min_order_size"
BLOCK_INSUFFICIENT_BALANCE = "insufficient_balance"
BLOCK_COOLDOWN = "cooldown"
BLOCK_POSITION_LIMITS = "position_limits"
BLOCK_EXCHANGE_VALIDATION = "exchange_validation"
BLOCK_OTHER = "other"

_BLOCK_CATEGORIES = (
    BLOCK_RISK_MANAGER,
    BLOCK_MIN_ORDER_SIZE,
    BLOCK_INSUFFICIENT_BALANCE,
    BLOCK_COOLDOWN,
    BLOCK_POSITION_LIMITS,
    BLOCK_EXCHANGE_VALIDATION,
    BLOCK_OTHER,
)

# Market-data failure kinds.
DATA_TICKER = "ticker"
DATA_WEBSOCKET = "websocket"
DATA_UNAVAILABLE = "unavailable"

# Bound on distinct decision reasons retained per bot (cardinality guard).
_MAX_REASONS = 100
# Hourly evaluation buckets retained for the rolling 24h count.
_MAX_EVAL_BUCKETS = 26

# Collapse volatile tokens (prices, sizes, percentages) so that, e.g.,
# "Holding position, stop at $49000.00" and "...$48000.00" count as ONE reason.
_NUM_RE = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*%?")


def _normalize_reason(reason: str) -> str:
    """Reduce a free-text strategy reason to a stable, low-cardinality bucket."""
    if not reason:
        return "(no reason)"
    collapsed = _NUM_RE.sub("#", reason).strip()
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed[:120] if collapsed else "(no reason)"


@dataclass
class BotDiagnostics:
    """Accumulated observe-only diagnostics for a single bot."""

    bot_id: int

    # --- Evaluation statistics ---
    total_evaluations: int = 0
    runtime_evaluations: int = 0
    runtime_started_at: Optional[datetime] = None
    last_evaluation_at: Optional[datetime] = None
    # hour-bucket (isoformat hour) -> count, for the rolling 24h figure
    eval_buckets: Dict[str, int] = field(default_factory=dict)

    # --- Signal statistics ---
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    last_signal_action: Optional[str] = None
    last_signal_reason: Optional[str] = None
    last_signal_at: Optional[datetime] = None

    # --- Decision reasons (normalized -> count) ---
    reason_counts: Dict[str, int] = field(default_factory=dict)

    # --- Blocked trades (category -> count) ---
    blocked: Dict[str, int] = field(default_factory=lambda: {c: 0 for c in _BLOCK_CATEGORIES})
    last_block_category: Optional[str] = None
    last_block_reason: Optional[str] = None
    last_block_at: Optional[datetime] = None

    # --- Execution outcomes ---
    successful_buys: int = 0
    successful_sells: int = 0
    failed_buys: int = 0
    failed_sells: int = 0
    last_exec_failure_reason: Optional[str] = None
    last_exec_failure_at: Optional[datetime] = None

    # --- Market-data health ---
    ticker_failures: int = 0
    websocket_failures: int = 0
    data_unavailable: int = 0
    last_data_failure_reason: Optional[str] = None
    last_data_failure_at: Optional[datetime] = None

    # --- Pause diagnostics ---
    pause_reason: Optional[str] = None
    paused_at: Optional[datetime] = None

    # --- Recovery mode diagnostics ---
    recovery_is_active: bool = False
    recovery_reason: Optional[str] = None
    recovery_started_at: Optional[datetime] = None
    recovery_trade_count: int = 0
    recovery_win_count: int = 0
    recovery_loss_count: int = 0
    recovery_pnl_usd: float = 0.0
    recovery_consecutive_wins: int = 0
    last_recovery_trade_at: Optional[datetime] = None
    # Capped at 50 most-recent events to bound memory use.
    recovery_events: List[dict] = field(default_factory=list)

    def _bump_eval_bucket(self, now: datetime) -> None:
        key = now.replace(minute=0, second=0, microsecond=0).isoformat()
        self.eval_buckets[key] = self.eval_buckets.get(key, 0) + 1
        if len(self.eval_buckets) > _MAX_EVAL_BUCKETS:
            for old in sorted(self.eval_buckets)[:-_MAX_EVAL_BUCKETS]:
                self.eval_buckets.pop(old, None)

    def evaluations_last_24h(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.utcnow()
        cutoff = now - timedelta(hours=24)
        total = 0
        for key, count in self.eval_buckets.items():
            try:
                if datetime.fromisoformat(key) >= cutoff:
                    total += count
            except ValueError:
                continue
        return total

    def top_reasons(self, limit: int = 10) -> List[dict]:
        ordered = sorted(self.reason_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [{"reason": r, "count": c} for r, c in ordered[:limit]]

    def to_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "evaluations": {
                "total": self.total_evaluations,
                "runtime": self.runtime_evaluations,
                "last_24h": self.evaluations_last_24h(),
                "last_evaluation_at": _iso(self.last_evaluation_at),
                "runtime_started_at": _iso(self.runtime_started_at),
            },
            "signals": {
                "buy": self.buy_signals,
                "sell": self.sell_signals,
                "hold": self.hold_signals,
                "last_action": self.last_signal_action,
                "last_reason": self.last_signal_reason,
                "last_at": _iso(self.last_signal_at),
            },
            "top_reasons": self.top_reasons(),
            "blocked": {
                **{c: self.blocked.get(c, 0) for c in _BLOCK_CATEGORIES},
                "last_category": self.last_block_category,
                "last_reason": self.last_block_reason,
                "last_at": _iso(self.last_block_at),
            },
            "execution": {
                "successful_buys": self.successful_buys,
                "successful_sells": self.successful_sells,
                "failed_buys": self.failed_buys,
                "failed_sells": self.failed_sells,
                "last_failure_reason": self.last_exec_failure_reason,
                "last_failure_at": _iso(self.last_exec_failure_at),
            },
            "market_data": {
                "ticker_failures": self.ticker_failures,
                "websocket_failures": self.websocket_failures,
                "data_unavailable": self.data_unavailable,
                "last_failure_reason": self.last_data_failure_reason,
                "last_failure_at": _iso(self.last_data_failure_at),
            },
            "pause": {
                "reason": self.pause_reason,
                "paused_at": _iso(self.paused_at),
            },
            "recovery": {
                "is_active": self.recovery_is_active,
                "reason": self.recovery_reason,
                "started_at": _iso(self.recovery_started_at),
                "trade_count": self.recovery_trade_count,
                "win_count": self.recovery_win_count,
                "loss_count": self.recovery_loss_count,
                "pnl_usd": self.recovery_pnl_usd,
                "consecutive_wins": self.recovery_consecutive_wins,
                "last_trade_at": _iso(self.last_recovery_trade_at),
                "events": list(self.recovery_events),
            },
        }


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _guard(method):
    """Wrap a store method so a diagnostics bug can never reach the trading loop."""
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - observability must never throw
            logger.warning("diagnostics %s failed (non-fatal): %s", method.__name__, exc)
            return None
    wrapper.__name__ = method.__name__
    return wrapper


class DiagnosticsStore:
    """Thread-unsafe (single asyncio loop) in-memory diagnostics store."""

    def __init__(self) -> None:
        self._diags: Dict[int, BotDiagnostics] = {}

    def _get_or_create(self, bot_id: int) -> BotDiagnostics:
        d = self._diags.get(bot_id)
        if d is None:
            d = BotDiagnostics(bot_id=bot_id)
            self._diags[bot_id] = d
        return d

    @_guard
    def start_runtime(self, bot_id: int) -> None:
        """Mark a fresh run: reset the per-runtime evaluation counter and clear
        any stale pause reason. Lifetime totals are preserved."""
        d = self._get_or_create(bot_id)
        d.runtime_evaluations = 0
        d.runtime_started_at = datetime.utcnow()
        d.pause_reason = None
        d.paused_at = None

    @_guard
    def record_evaluation(self, bot_id: int) -> None:
        d = self._get_or_create(bot_id)
        now = datetime.utcnow()
        d.total_evaluations += 1
        d.runtime_evaluations += 1
        d.last_evaluation_at = now
        d._bump_eval_bucket(now)

    @_guard
    def record_signal(self, bot_id: int, signal) -> None:
        """Count a strategy signal and bucket its decision reason.

        ``signal`` is a ``TradeSignal`` (or ``None``); read defensively."""
        d = self._get_or_create(bot_id)
        action = (getattr(signal, "action", "") or "").lower() if signal else "hold"
        reason = (getattr(signal, "reason", "") if signal else "") or ""
        if action == "buy":
            d.buy_signals += 1
        elif action == "sell":
            d.sell_signals += 1
        else:
            action = "hold"
            d.hold_signals += 1
        d.last_signal_action = action
        d.last_signal_reason = reason
        d.last_signal_at = datetime.utcnow()

        key = _normalize_reason(reason)
        if key in d.reason_counts or len(d.reason_counts) < _MAX_REASONS:
            d.reason_counts[key] = d.reason_counts.get(key, 0) + 1

    @_guard
    def record_blocked(self, bot_id: int, category: str, reason: str = "") -> None:
        d = self._get_or_create(bot_id)
        if category not in d.blocked:
            category = BLOCK_OTHER
        d.blocked[category] += 1
        d.last_block_category = category
        d.last_block_reason = reason
        d.last_block_at = datetime.utcnow()

    @_guard
    def record_execution(self, bot_id: int, action: str, success: bool, reason: str = "") -> None:
        d = self._get_or_create(bot_id)
        action = (action or "").lower()
        if success:
            if action == "sell":
                d.successful_sells += 1
            else:
                d.successful_buys += 1
        else:
            if action == "sell":
                d.failed_sells += 1
            else:
                d.failed_buys += 1
            d.last_exec_failure_reason = reason
            d.last_exec_failure_at = datetime.utcnow()

    @_guard
    def record_data_failure(self, bot_id: int, kind: str, reason: str = "") -> None:
        d = self._get_or_create(bot_id)
        if kind == DATA_WEBSOCKET:
            d.websocket_failures += 1
        elif kind == DATA_UNAVAILABLE:
            d.data_unavailable += 1
        else:
            d.ticker_failures += 1
        d.last_data_failure_reason = reason
        d.last_data_failure_at = datetime.utcnow()

    @_guard
    def record_pause(self, bot_id: int, reason: str) -> None:
        d = self._get_or_create(bot_id)
        d.pause_reason = reason
        d.paused_at = datetime.utcnow()

    @_guard
    def record_recovery_entered(self, bot_id: int, reason: str) -> None:
        d = self._get_or_create(bot_id)
        d.recovery_is_active = True
        d.recovery_reason = reason
        d.recovery_started_at = datetime.utcnow()
        d.recovery_trade_count = 0
        d.recovery_win_count = 0
        d.recovery_loss_count = 0
        d.recovery_pnl_usd = 0.0
        d.recovery_consecutive_wins = 0
        d.recovery_events.append({
            "type": "entered",
            "at": datetime.utcnow().isoformat(),
            "reason": reason,
        })
        if len(d.recovery_events) > 50:
            d.recovery_events = d.recovery_events[-50:]

    @_guard
    def record_paper_trade(
        self,
        bot_id: int,
        gain_loss_usd: float,
        win: bool,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
    ) -> None:
        d = self._get_or_create(bot_id)
        d.recovery_trade_count += 1
        d.recovery_pnl_usd += gain_loss_usd
        d.last_recovery_trade_at = datetime.utcnow()
        if win:
            d.recovery_win_count += 1
            d.recovery_consecutive_wins += 1
        else:
            d.recovery_loss_count += 1
            d.recovery_consecutive_wins = 0
        d.recovery_events.append({
            "type": "paper_win" if win else "paper_loss",
            "at": datetime.utcnow().isoformat(),
            "gain_loss_usd": round(gain_loss_usd, 4),
            "entry_price": entry_price,
            "exit_price": exit_price,
        })
        if len(d.recovery_events) > 50:
            d.recovery_events = d.recovery_events[-50:]

    @_guard
    def record_recovery_exited(self, bot_id: int, reason: str) -> None:
        d = self._get_or_create(bot_id)
        d.recovery_is_active = False
        d.recovery_events.append({
            "type": "exited",
            "at": datetime.utcnow().isoformat(),
            "reason": reason,
        })
        if len(d.recovery_events) > 50:
            d.recovery_events = d.recovery_events[-50:]

    def get(self, bot_id: int) -> Optional[BotDiagnostics]:
        return self._diags.get(bot_id)

    @_guard
    def clear(self, bot_id: int) -> None:
        self._diags.pop(bot_id, None)


# Global singleton shared by the trading engine and the API router.
diagnostics_store = DiagnosticsStore()
