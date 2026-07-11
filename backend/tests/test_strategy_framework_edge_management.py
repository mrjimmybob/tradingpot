"""Unit tests for Pillar 7's shared StrategyEdgeManager
(add-strategy-decision-framework, Phase 0.5).

Covers: insufficient-sample handling, the three degradation categories
classifying correctly from synthetic episodes, the never-force-closes
structural guarantee, and determinism.
"""
from __future__ import annotations

import inspect

import pytest

from app.services.strategy_framework.edge_management import (
    EdgeCategory,
    EdgeStatus,
    StrategyEdgeManager,
)

BOT_ID = 1
STRATEGY = "trend_following"


def _manager(**overrides) -> StrategyEdgeManager:
    kwargs = dict(min_sample_size=10, outcome_window=50, expectancy_floor=0.0, win_rate_floor=0.35)
    kwargs.update(overrides)
    return StrategyEdgeManager(**kwargs)


def _feed_losing_streak(mgr: StrategyEdgeManager, n: int, *, pnl: float = -5.0) -> None:
    for _ in range(n):
        mgr.record_trade_outcome(BOT_ID, STRATEGY, pnl=pnl, win=False)


def _feed_winning_streak(mgr: StrategyEdgeManager, n: int, *, pnl: float = 5.0) -> None:
    for _ in range(n):
        mgr.record_trade_outcome(BOT_ID, STRATEGY, pnl=pnl, win=True)


class TestInsufficientSample:
    def test_below_min_sample_returns_none_category(self):
        mgr = _manager(min_sample_size=10)
        _feed_losing_streak(mgr, 3)
        status = mgr.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=False)
        assert status.category == EdgeCategory.NONE
        assert status.can_adapt is False
        assert status.should_wait is False
        assert status.should_stop is False
        assert "insufficient" in status.action.lower()


class TestNoDegradation:
    def test_profitable_strategy_classifies_as_none(self):
        mgr = _manager(min_sample_size=10)
        _feed_winning_streak(mgr, 15, pnl=10.0)
        status = mgr.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=False)
        assert status.category == EdgeCategory.NONE
        assert "No degradation" in status.action
        explanation = status.explain()
        assert "win_rate" in explanation


class TestCategoryA:
    def test_degradation_with_regime_mismatch_classifies_category_a(self):
        """Synthetic episode: losing streak + regime outside suitable range
        -> Category A (temporary market mismatch)."""
        mgr = _manager(min_sample_size=10, expectancy_floor=0.0, win_rate_floor=0.35)
        _feed_losing_streak(mgr, 15)
        status = mgr.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=True)
        assert status.category == EdgeCategory.A
        assert status.should_wait is True
        assert status.can_adapt is False
        assert status.should_stop is False
        assert "Never a permanent stop" in status.action

    def test_category_a_takes_priority_over_parameter_evidence(self):
        """Regime mismatch is checked before parameter-mismatch evidence -
        matches design.md's A-then-B-then-C ordering (A/B responses happen
        before a C conclusion is reached)."""
        mgr = _manager(min_sample_size=10)
        _feed_losing_streak(mgr, 15)
        status = mgr.evaluate(
            BOT_ID, STRATEGY,
            regime_outside_suitable_range=True,
            parameter_mismatch_evidence="stop multiplier stale",
        )
        assert status.category == EdgeCategory.A


class TestCategoryB:
    def test_degradation_with_parameter_evidence_classifies_category_b(self):
        mgr = _manager(min_sample_size=10)
        _feed_losing_streak(mgr, 15)
        status = mgr.evaluate(
            BOT_ID, STRATEGY,
            regime_outside_suitable_range=False,
            parameter_mismatch_evidence="ATR multiplier calibrated for a 40% lower volatility regime",
        )
        assert status.category == EdgeCategory.B
        assert status.can_adapt is True
        assert status.should_wait is False
        assert status.should_stop is False
        assert "AdaptiveParameterResolver" in status.action
        assert "ATR multiplier" in status.reason


class TestCategoryC:
    def test_degradation_with_no_explanation_classifies_category_c(self):
        """No regime mismatch, no parameter-mismatch evidence, but real,
        sustained degradation -> Category C (edge disappeared)."""
        mgr = _manager(min_sample_size=10)
        _feed_losing_streak(mgr, 15)
        status = mgr.evaluate(
            BOT_ID, STRATEGY,
            regime_outside_suitable_range=False,
            parameter_mismatch_evidence=None,
        )
        assert status.category == EdgeCategory.C
        assert status.should_stop is True
        assert status.can_adapt is False
        assert status.should_wait is False
        assert "human-reviewed re-certification" in status.action


class TestNeverForceClosesPosition:
    def test_public_api_has_no_position_closing_capability(self):
        """Structural guarantee: the class has no method that could close,
        exit, sell, or liquidate a position - not just a documented rule."""
        forbidden_substrings = ("close", "exit", "sell", "liquidat")
        public_methods = [
            name for name, _ in inspect.getmembers(StrategyEdgeManager, predicate=inspect.isfunction)
            if not name.startswith("_")
        ]
        for name in public_methods:
            lowered = name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), (
                f"StrategyEdgeManager.{name} looks like it could force-close a "
                "position - this manager must only classify, never act on positions"
            )

    def test_category_c_status_carries_no_execution_side_effect(self):
        """Even a Category C classification is purely informational - the
        caller decides what to do; evaluate() itself performs no action."""
        mgr = _manager(min_sample_size=10)
        _feed_losing_streak(mgr, 15)
        status = mgr.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=False)
        assert isinstance(status, EdgeStatus)
        assert status.category == EdgeCategory.C
        # No exception, no external call - the return value is the only effect.


class TestDecisionScoreTrend:
    def test_declining_decision_score_contributes_to_degradation(self):
        mgr = _manager(min_sample_size=10, decision_score_trend_window=5, decision_score_trend_epsilon=1.0)
        # Otherwise-healthy trade outcomes (would classify NONE on P&L alone)...
        _feed_winning_streak(mgr, 12, pnl=1.0)
        # ...but Decision Score has been trending down significantly.
        for score in [80, 78, 76, 75, 74]:
            mgr.record_decision_score(BOT_ID, STRATEGY, score)
        for score in [50, 48, 45, 40, 35]:
            mgr.record_decision_score(BOT_ID, STRATEGY, score)
        status = mgr.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=False)
        assert status.category != EdgeCategory.NONE
        assert any("decision_score_trend_delta" in s.name for s in status.signals)


class TestDeterminism:
    def test_same_evidence_produces_same_classification(self):
        mgr1 = _manager(min_sample_size=10)
        mgr2 = _manager(min_sample_size=10)
        _feed_losing_streak(mgr1, 15)
        _feed_losing_streak(mgr2, 15)
        s1 = mgr1.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=True)
        s2 = mgr2.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=True)
        assert s1.category == s2.category
        assert s1.reason == s2.reason
        assert [sig.value for sig in s1.signals] == [sig.value for sig in s2.signals]


class TestReset:
    def test_reset_clears_tracked_history(self):
        mgr = _manager(min_sample_size=5)
        _feed_losing_streak(mgr, 10)
        mgr.reset(BOT_ID, STRATEGY)
        status = mgr.evaluate(BOT_ID, STRATEGY, regime_outside_suitable_range=False)
        assert status.category == EdgeCategory.NONE
        assert "insufficient" in status.action.lower()


class TestConstructorValidation:
    def test_min_sample_size_must_be_positive(self):
        with pytest.raises(ValueError):
            StrategyEdgeManager(min_sample_size=0)

    def test_outcome_window_must_be_at_least_min_sample_size(self):
        with pytest.raises(ValueError):
            StrategyEdgeManager(min_sample_size=20, outcome_window=5)
