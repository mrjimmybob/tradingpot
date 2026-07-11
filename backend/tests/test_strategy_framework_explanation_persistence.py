"""Unit tests for the Order/Trade decision-explanation persistence helper
(add-strategy-decision-framework, Phase 0.6).
"""
from __future__ import annotations

from app.services.strategy_framework.explanation_persistence import (
    extract_edge_management_category,
    summarize_decision_explanation,
)


class TestSummarizeDecisionExplanation:
    def test_none_in_none_out(self):
        assert summarize_decision_explanation(None) is None

    def test_empty_dict_treated_as_none(self):
        assert summarize_decision_explanation({}) is None

    def test_copies_core_fields(self):
        explanation = {
            "strategy": "trend_following",
            "decision": "buy",
            "reason": "EMA cross confirmed",
            "state": "LONG_OPEN",
            "checks": [{"name": "ema_cross", "current": 1.0, "required": "> 0", "passed": True}],
            "metrics": {"ema_fast": 100.5},
            "evaluated_at": "2026-01-01T00:00:00",
        }
        summary = summarize_decision_explanation(explanation)
        assert summary["strategy"] == "trend_following"
        assert summary["decision"] == "buy"
        assert summary["reason"] == "EMA cross confirmed"
        assert summary["state"] == "LONG_OPEN"
        assert summary["checks"] == explanation["checks"]
        assert summary["metrics"] == {"ema_fast": 100.5}
        assert summary["evaluated_at"] == "2026-01-01T00:00:00"

    def test_checks_are_bounded(self):
        explanation = {
            "strategy": "x",
            "checks": [{"name": f"check_{i}"} for i in range(100)],
            "metrics": {},
        }
        summary = summarize_decision_explanation(explanation)
        assert len(summary["checks"]) == 25

    def test_metrics_are_bounded(self):
        explanation = {
            "strategy": "x",
            "checks": [],
            "metrics": {f"m{i}": i for i in range(100)},
        }
        summary = summarize_decision_explanation(explanation)
        assert len(summary["metrics"]) == 40

    def test_missing_checks_and_metrics_default_to_empty(self):
        summary = summarize_decision_explanation({"strategy": "x", "decision": "hold"})
        assert summary["checks"] == []
        assert summary["metrics"] == {}


class TestExtractEdgeManagementCategory:
    def test_none_explanation_returns_none(self):
        assert extract_edge_management_category(None) is None

    def test_no_category_metric_returns_none(self):
        assert extract_edge_management_category({"metrics": {}}) is None

    def test_unmigrated_strategy_has_no_category(self):
        """A strategy not yet wired to StrategyEdgeManager (all of them,
        as of Phase 0) has no edge_status_category metric at all."""
        explanation = {"strategy": "trend_following", "metrics": {"ema_fast": 1.0}}
        assert extract_edge_management_category(explanation) is None

    def test_valid_category_extracted(self):
        for category in ("A", "B", "C"):
            explanation = {"metrics": {"edge_status_category": category}}
            assert extract_edge_management_category(explanation) == category

    def test_none_category_string_not_extracted(self):
        """EdgeCategory.NONE ('none') is not one of A/B/C - correctly
        treated as 'no active degradation category', not surfaced."""
        explanation = {"metrics": {"edge_status_category": "none"}}
        assert extract_edge_management_category(explanation) is None

    def test_garbage_value_ignored(self):
        explanation = {"metrics": {"edge_status_category": "not_a_real_category"}}
        assert extract_edge_management_category(explanation) is None
