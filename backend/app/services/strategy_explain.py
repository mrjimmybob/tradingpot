"""Structured, numeric decision explanations for strategy evaluations.

Every strategy evaluation produces a ``DecisionExplanation``: the final
BUY/SELL/HOLD decision, the human reason, and — crucially — the exact list of
numeric *checks* that participated in that decision (current value, the value
required, and whether it passed), plus a free-form ``metrics`` bag of named
numbers the strategy computed. The backend COMPUTES this; the frontend only
RENDERS it. No strategy logic is ever duplicated in React.

This module is observe-only and totally exception-safe by contract: a bug while
recording an explanation must NEVER affect a trading decision. Every builder
method swallows and logs its own error and returns ``self`` so call sites stay
clean one-liners that can be chained.

Design notes:
  * ``check(name, current, required, passed)`` is the atomic unit an operator
    reads to answer "which condition failed, and by how much".
  * ``required`` is a short human string ("< 30.0", ">= 300 sec", "Price <= Lower
    Band") so the renderer needs no per-strategy knowledge.
  * ``metric(name, value)`` records a participating number that is not itself a
    pass/fail gate (ATR, grid spacing, EMA distance, ...). The renderer shows
    these as a flat key/value grid.
  * ``candidates`` / ``selected`` carry Auto Mode's full per-strategy scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _coerce(value: Any) -> Any:
    """Make a value JSON-friendly and stable for display.

    Floats are rounded to 6 dp to avoid noisy trailing digits; bools/ints/str
    pass through; anything else is stringified so the API never fails to encode.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return round(value, 6)
        except (ValueError, OverflowError):
            return value
    if value is None or isinstance(value, str):
        return value
    return str(value)


@dataclass
class DecisionCheck:
    """One numeric gate that participated in a strategy decision."""

    name: str
    current: Any
    required: str
    passed: bool
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "current": _coerce(self.current),
            "required": self.required,
            "passed": bool(self.passed),
            "detail": self.detail,
        }


@dataclass
class DecisionExplanation:
    """The full structured explanation for ONE strategy evaluation."""

    strategy: str
    decision: str = "hold"
    reason: str = ""
    checks: List[DecisionCheck] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Auto Mode scoring rows + the chosen strategy (empty for single strategies).
    candidates: List[dict] = field(default_factory=list)
    selected: Optional[str] = None
    evaluated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "decision": self.decision,
            "reason": self.reason,
            "checks": [c.to_dict() for c in self.checks],
            "metrics": {k: _coerce(v) for k, v in self.metrics.items()},
            "candidates": list(self.candidates),
            "selected": self.selected,
            "evaluated_at": self.evaluated_at,
        }


class ExplanationBuilder:
    """Accumulates a strategy's structured decision explanation for ONE cycle.

    Strategies add ``check``/``metric`` calls at the points where the numbers are
    computed; the engine then ``finalize``s the decision+reason from the returned
    ``TradeSignal`` and records the result to the diagnostics store. Every method
    is exception-safe and chainable.
    """

    def __init__(self, strategy: str) -> None:
        self.exp = DecisionExplanation(strategy=strategy or "unknown")

    def check(
        self,
        name: str,
        current: Any,
        required: str,
        passed: bool,
        detail: Optional[str] = None,
    ) -> "ExplanationBuilder":
        try:
            self.exp.checks.append(
                DecisionCheck(
                    name=str(name),
                    current=current,
                    required=str(required),
                    passed=bool(passed),
                    detail=detail,
                )
            )
        except Exception as exc:  # noqa: BLE001 - observability must never throw
            logger.debug("explain.check(%s) failed: %s", name, exc)
        return self

    def metric(self, name: str, value: Any) -> "ExplanationBuilder":
        try:
            self.exp.metrics[str(name)] = value
        except Exception as exc:  # noqa: BLE001
            logger.debug("explain.metric(%s) failed: %s", name, exc)
        return self

    def update(self, values: Dict[str, Any]) -> "ExplanationBuilder":
        """Record several metrics at once."""
        try:
            for k, v in (values or {}).items():
                self.exp.metrics[str(k)] = v
        except Exception as exc:  # noqa: BLE001
            logger.debug("explain.update failed: %s", exc)
        return self

    def candidate(self, row: Dict[str, Any]) -> "ExplanationBuilder":
        """Append one Auto Mode scoring row."""
        try:
            self.exp.candidates.append(dict(row))
        except Exception as exc:  # noqa: BLE001
            logger.debug("explain.candidate failed: %s", exc)
        return self

    def select(self, name: Optional[str]) -> "ExplanationBuilder":
        self.exp.selected = name
        return self

    def summary(self, decision: Optional[str] = None, reason: Optional[str] = None) -> "ExplanationBuilder":
        """Set the decision/reason explicitly (overrides the signal-derived ones)."""
        try:
            if decision is not None:
                self.exp.decision = str(decision).lower()
            if reason is not None:
                self.exp.reason = str(reason)
        except Exception as exc:  # noqa: BLE001
            logger.debug("explain.summary failed: %s", exc)
        return self

    def finalize(self, signal: Any = None) -> "ExplanationBuilder":
        """Adopt decision + reason from the returned TradeSignal (if not already set)."""
        try:
            if signal is not None:
                action = (getattr(signal, "action", None) or "hold")
                self.exp.decision = str(action).lower()
                sig_reason = getattr(signal, "reason", "") or ""
                # Strategy-provided summary reason wins; otherwise use the signal's.
                if not self.exp.reason:
                    self.exp.reason = sig_reason
        except Exception as exc:  # noqa: BLE001
            logger.debug("explain.finalize failed: %s", exc)
        return self

    def to_dict(self) -> dict:
        try:
            self.exp.evaluated_at = datetime.utcnow().isoformat()
            return self.exp.to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.debug("explain.to_dict failed: %s", exc)
            return {
                "strategy": getattr(self.exp, "strategy", "unknown"),
                "decision": "hold",
                "reason": "",
                "checks": [],
                "metrics": {},
                "candidates": [],
                "selected": None,
                "evaluated_at": None,
            }
