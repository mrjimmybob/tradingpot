"""Pillar 8 persistence gap closure — summarize a live DecisionExplanation
for storage on the Order record that executed it.

The audit behind ``add-strategy-decision-framework`` found ``Explanation
Builder``/``DecisionExplanation`` (``app/services/strategy_explain.py``)
already covers most of pillar 8, but the full structured explanation lives
only in the in-memory, current-state-only ``DiagnosticsStore`` — a
historical trade can only be explained by its terse ``Order.reason``
string, not the full evidence that produced it. This module provides the
pure summarization function; ``trading_engine.py`` calls it at each of the
three real order-creation sites (market/limit, TWAP, VWAP — NOT the
recovery-import path, which records an untracked exchange fill with no
strategy decision behind it) to populate ``Order.decision_explanation``.

Bounded, not unbounded: ``checks``/``metrics`` are capped so a
pathological strategy can never grow a single Order row without limit.

Forward-compatible convention for Phase 1-6: once a strategy is migrated
to use ``DecisionScoreEngine``/``StrategyEdgeManager``, it records its
Evidence Report and edge-management category via
``self._explain(bot.id).metric("evidence_report", result.evidence_report.to_dict())``
and ``.metric("edge_status_category", status.category.value)`` — this
module reads those two well-known metric keys, if present, into dedicated
top-level fields (``evidence_report``, and
``extract_edge_management_category``'s return value) so they are
queryable without unpacking the generic ``metrics`` bag. Until a strategy
is migrated, both are simply absent/None — expected, not an error.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_MAX_PERSISTED_CHECKS = 25
_MAX_PERSISTED_METRICS = 40

# Well-known metric keys a migrated (Phase 1-6) strategy populates. See the
# module docstring's "Forward-compatible convention" above.
_EVIDENCE_REPORT_METRIC_KEY = "evidence_report"
_EDGE_CATEGORY_METRIC_KEY = "edge_status_category"

_VALID_EDGE_CATEGORIES = frozenset({"A", "B", "C"})


def summarize_decision_explanation(explanation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Bound and shape a ``DecisionExplanation.to_dict()`` payload for
    storage on an ``Order`` row. ``None`` in, ``None`` out (a strategy that
    was never instrumented this cycle has nothing to persist)."""
    if not explanation:
        return None

    checks = explanation.get("checks") or []
    metrics = explanation.get("metrics") or {}

    return {
        "strategy": explanation.get("strategy"),
        "decision": explanation.get("decision"),
        "reason": explanation.get("reason"),
        "state": explanation.get("state"),
        "checks": list(checks[:_MAX_PERSISTED_CHECKS]),
        "metrics": dict(list(metrics.items())[:_MAX_PERSISTED_METRICS]),
        "evaluated_at": explanation.get("evaluated_at"),
    }


def extract_edge_management_category(explanation: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pull the active Strategy Edge Management category (A/B/C) out of an
    explanation's metrics, if a migrated strategy recorded one this cycle.
    Returns None for any strategy not yet migrated to StrategyEdgeManager,
    or when no degradation was classified (category NONE)."""
    if not explanation:
        return None
    metrics = explanation.get("metrics") or {}
    category = metrics.get(_EDGE_CATEGORY_METRIC_KEY)
    if category in _VALID_EDGE_CATEGORIES:
        return category
    return None
