"""Unit tests for Pillar 3's shared DecisionScoreEngine
(add-strategy-decision-framework, Phase 0.2).

Covers: EvidenceItem construction validation (rejecting subjective/
unmeasurable factors), scoring math, determinism enforcement, threshold
approval, and the Evidence Report format.
"""
from __future__ import annotations

import pytest

from app.services.strategy_framework.decision_score import (
    DecisionScoreEngine,
    EvidenceItem,
    EvidenceReport,
    InvalidEvidenceItemError,
    NonDeterministicEvidenceItemError,
)


def _trend_item(weight: float = 20.0) -> EvidenceItem:
    return EvidenceItem(
        name="Trend strength",
        measurement=lambda data: data["ema_slope_pct"],
        normalization=lambda raw: max(-1.0, min(1.0, raw / 2.0)),
        weight=weight,
        reason="Theory doc section 2: trend continuation edge requires a rising EMA slope",
    )


def _volume_item(weight: float = 15.0) -> EvidenceItem:
    return EvidenceItem(
        name="Volume confirmation",
        measurement=lambda data: data["volume_zscore"],
        normalization=lambda raw: max(-1.0, min(1.0, raw / 3.0)),
        weight=weight,
        reason="Theory doc section 3: breakouts without volume confirmation are false-breakout prone",
    )


class TestEvidenceItemValidation:
    def test_valid_item_constructs(self):
        item = _trend_item()
        assert item.name == "Trend strength"

    def test_empty_name_rejected(self):
        with pytest.raises(InvalidEvidenceItemError):
            EvidenceItem(
                name="  ",
                measurement=lambda d: 1.0,
                normalization=lambda r: r,
                weight=10.0,
                reason="some reason",
            )

    def test_non_callable_measurement_rejected(self):
        """A fixed/subjective value ('looks bullish' encoded as a constant)
        cannot be expressed as a deterministic Measurement — rejected."""
        with pytest.raises(InvalidEvidenceItemError):
            EvidenceItem(
                name="Looks bullish",
                measurement="bullish",  # not callable
                normalization=lambda r: r,
                weight=10.0,
                reason="gut feeling",
            )

    def test_non_callable_normalization_rejected(self):
        with pytest.raises(InvalidEvidenceItemError):
            EvidenceItem(
                name="Bad factor",
                measurement=lambda d: 1.0,
                normalization=0.5,  # not callable
                weight=10.0,
                reason="some reason",
            )

    def test_zero_or_negative_weight_rejected(self):
        with pytest.raises(InvalidEvidenceItemError):
            EvidenceItem(
                name="Zero weight",
                measurement=lambda d: 1.0,
                normalization=lambda r: r,
                weight=0.0,
                reason="some reason",
            )
        with pytest.raises(InvalidEvidenceItemError):
            EvidenceItem(
                name="Negative weight",
                measurement=lambda d: 1.0,
                normalization=lambda r: r,
                weight=-5.0,
                reason="some reason",
            )

    def test_missing_reason_rejected(self):
        """No documented reason it contributes to edge -> not traceable to a
        Theory document -> rejected (design.md pillar 3)."""
        with pytest.raises(InvalidEvidenceItemError):
            EvidenceItem(
                name="Mystery factor",
                measurement=lambda d: 1.0,
                normalization=lambda r: r,
                weight=10.0,
                reason="",
            )

    def test_item_is_immutable(self):
        item = _trend_item()
        with pytest.raises(Exception):
            item.weight = 999.0  # frozen dataclass


class TestDecisionScoreEngineScoring:
    def setup_method(self):
        self.engine = DecisionScoreEngine()

    def test_empty_items_rejected(self):
        with pytest.raises(InvalidEvidenceItemError):
            self.engine.score("trend_following", [], {}, threshold=50.0)

    def test_single_condition_alone_can_be_insufficient(self):
        """Mirrors spec.md's 'Single-condition trigger is insufficient'
        scenario: one weak factor alone should not clear a real threshold."""
        items = [_trend_item(weight=20.0)]
        data = {"ema_slope_pct": 0.2}  # normalized = 0.1 -> contribution = 2.0
        result = self.engine.score("trend_following", items, data, threshold=50.0)
        assert result.total == pytest.approx(2.0)
        assert result.approved is False

    def test_high_evidence_trade_scores_above_threshold(self):
        """Mirrors spec.md's 'High-evidence trade scores above threshold'
        scenario: multiple aligned factors together clear the bar."""
        items = [_trend_item(weight=50.0), _volume_item(weight=50.0)]
        data = {"ema_slope_pct": 2.0, "volume_zscore": 3.0}  # both normalize to 1.0
        result = self.engine.score("trend_following", items, data, threshold=75.0)
        assert result.total == pytest.approx(100.0)
        assert result.approved is True

    def test_negative_evidence_reduces_score(self):
        items = [
            _trend_item(weight=50.0),
            EvidenceItem(
                name="Nearby resistance",
                measurement=lambda d: -0.4,
                normalization=lambda r: r,
                weight=20.0,
                reason="Theory doc: resistance proximity increases false-breakout risk",
            ),
        ]
        data = {"ema_slope_pct": 2.0}
        result = self.engine.score("trend_following", items, data, threshold=50.0)
        # trend contributes +50, resistance contributes -8
        assert result.total == pytest.approx(42.0)
        assert result.approved is False

    def test_normalization_is_clamped_defensively(self):
        """Even if a Normalization forgets to clamp, the engine clamps to
        [-1, 1] itself — a defensive backstop on the bounded-range contract."""
        item = EvidenceItem(
            name="Unclamped",
            measurement=lambda d: 100.0,
            normalization=lambda r: r,  # returns 100.0, way outside [-1, 1]
            weight=10.0,
            reason="Theory doc: deliberately unclamped to test the engine's backstop",
        )
        result = self.engine.score("test_strategy", [item], {}, threshold=1.0)
        assert result.total == pytest.approx(10.0)  # clamped to 1.0 * weight 10

    def test_result_is_deterministic(self):
        items = [_trend_item(), _volume_item()]
        data = {"ema_slope_pct": 1.0, "volume_zscore": 1.5}
        r1 = self.engine.score("trend_following", items, data, threshold=50.0)
        r2 = self.engine.score("trend_following", items, data, threshold=50.0)
        assert r1.total == r2.total
        assert r1.approved == r2.approved
        assert [c.contribution for c in r1.contributions] == [c.contribution for c in r2.contributions]

    def test_non_deterministic_measurement_rejected_at_score_time(self):
        state = {"calls": 0}

        def flaky_measurement(data):
            state["calls"] += 1
            return float(state["calls"])  # different every call

        item = EvidenceItem(
            name="Flaky",
            measurement=flaky_measurement,
            normalization=lambda r: r,
            weight=10.0,
            reason="deliberately non-deterministic for the test",
        )
        with pytest.raises(NonDeterministicEvidenceItemError):
            self.engine.score("test_strategy", [item], {}, threshold=1.0)

    def test_non_deterministic_normalization_rejected_at_score_time(self):
        state = {"calls": 0}

        def flaky_normalization(raw):
            state["calls"] += 1
            return float(state["calls"])

        item = EvidenceItem(
            name="Flaky norm",
            measurement=lambda d: 1.0,
            normalization=flaky_normalization,
            weight=10.0,
            reason="deliberately non-deterministic for the test",
        )
        with pytest.raises(NonDeterministicEvidenceItemError):
            self.engine.score("test_strategy", [item], {}, threshold=1.0)

    def test_non_numeric_measurement_rejected(self):
        item = EvidenceItem(
            name="Non-numeric",
            measurement=lambda d: "bullish",
            normalization=lambda r: r,
            weight=10.0,
            reason="deliberately non-numeric for the test",
        )
        with pytest.raises(InvalidEvidenceItemError):
            self.engine.score("test_strategy", [item], {}, threshold=1.0)


class TestEvidenceReport:
    def test_report_matches_design_doc_format(self):
        engine = DecisionScoreEngine()
        items = [
            _trend_item(weight=20.0),
            _volume_item(weight=15.0),
        ]
        data = {"ema_slope_pct": 1.8, "volume_zscore": 3.0}
        result = engine.score("trend_following", items, data, threshold=25.0)
        rendered = result.evidence_report.render()
        assert rendered.startswith("Evidence\n")
        assert "Trend strength" in rendered
        assert "Volume confirmation" in rendered
        assert "Total Decision Score:" in rendered
        assert "Minimum required:" in rendered
        assert rendered.strip().endswith("Trade approved.")

    def test_report_shows_rejected_when_below_threshold(self):
        engine = DecisionScoreEngine()
        items = [_trend_item(weight=20.0)]
        data = {"ema_slope_pct": 0.1}
        result = engine.score("trend_following", items, data, threshold=90.0)
        rendered = result.evidence_report.render()
        assert rendered.strip().endswith("Trade rejected.")

    def test_report_to_dict_is_json_friendly(self):
        engine = DecisionScoreEngine()
        result = engine.score("trend_following", [_trend_item()], {"ema_slope_pct": 1.0}, threshold=1.0)
        d = result.evidence_report.to_dict()
        assert d["strategy"] == "trend_following"
        assert isinstance(d["contributions"], list)
        assert d["contributions"][0]["name"] == "Trend strength"

    def test_every_executed_trade_can_produce_a_complete_report(self):
        """Mirrors spec.md's 'Every executed trade produces a complete
        Evidence Report' scenario."""
        engine = DecisionScoreEngine()
        items = [_trend_item(), _volume_item()]
        data = {"ema_slope_pct": 1.5, "volume_zscore": 2.0}
        result = engine.score("trend_following", items, data, threshold=10.0)
        report = result.evidence_report
        assert isinstance(report, EvidenceReport)
        assert len(report.contributions) == len(items)
        assert report.total == result.total
        assert report.threshold == 10.0
        assert report.approved == result.approved
