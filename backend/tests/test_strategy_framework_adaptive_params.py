"""Unit tests for Pillar 4's shared AdaptiveParameterResolver
(add-strategy-decision-framework, Phase 0.3).
"""
from __future__ import annotations

import pytest

from app.services.strategy_framework.adaptive_params import (
    AdaptedParameter,
    AdaptiveParameterResolver,
)


class TestAtrPercentileScaledMultiplier:
    def test_scales_with_percentile(self):
        result = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
            name="stop_atr_multiplier",
            atr_percentile=1.5,
            base_multiplier=2.0,
            min_multiplier=1.0,
            max_multiplier=4.0,
        )
        assert isinstance(result, AdaptedParameter)
        assert result.value == pytest.approx(3.0)
        assert "atr_percentile" in result.formula

    def test_clamped_to_max(self):
        result = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
            name="stop_atr_multiplier",
            atr_percentile=10.0,
            base_multiplier=2.0,
            min_multiplier=1.0,
            max_multiplier=4.0,
        )
        assert result.value == 4.0

    def test_clamped_to_min(self):
        result = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
            name="stop_atr_multiplier",
            atr_percentile=0.01,
            base_multiplier=2.0,
            min_multiplier=1.0,
            max_multiplier=4.0,
        )
        assert result.value == 1.0

    def test_negative_percentile_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
                name="x", atr_percentile=-1.0, base_multiplier=2.0,
                min_multiplier=1.0, max_multiplier=4.0,
            )

    def test_invalid_bounds_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveParameterResolver.atr_percentile_scaled_multiplier(
                name="x", atr_percentile=1.0, base_multiplier=2.0,
                min_multiplier=5.0, max_multiplier=1.0,
            )

    def test_deterministic(self):
        kwargs = dict(name="x", atr_percentile=1.2, base_multiplier=2.5, min_multiplier=1.0, max_multiplier=5.0)
        r1 = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(**kwargs)
        r2 = AdaptiveParameterResolver.atr_percentile_scaled_multiplier(**kwargs)
        assert r1 == r2


class TestRegimeScaledLookback:
    def test_applies_regime_multiplier(self):
        result = AdaptiveParameterResolver.regime_scaled_lookback(
            name="lookback",
            regime="trend_up",
            base_lookback=20,
            regime_multipliers={"trend_up": 1.5, "trend_down": 0.5},
            min_lookback=5,
            max_lookback=60,
        )
        assert result.value == 30.0

    def test_unmapped_regime_defaults_to_1x(self):
        result = AdaptiveParameterResolver.regime_scaled_lookback(
            name="lookback",
            regime="trend_flat",
            base_lookback=20,
            regime_multipliers={"trend_up": 1.5},
            min_lookback=5,
            max_lookback=60,
        )
        assert result.value == 20.0

    def test_clamped_within_bounds(self):
        result = AdaptiveParameterResolver.regime_scaled_lookback(
            name="lookback",
            regime="trend_up",
            base_lookback=20,
            regime_multipliers={"trend_up": 10.0},
            min_lookback=5,
            max_lookback=60,
        )
        assert result.value == 60.0

    def test_invalid_bounds_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveParameterResolver.regime_scaled_lookback(
                name="x", regime="trend_up", base_lookback=20,
                regime_multipliers={}, min_lookback=60, max_lookback=5,
            )


class TestAdaptForEdgeCategoryB:
    def test_applies_adjustment_with_citation(self):
        result = AdaptiveParameterResolver.adapt_for_edge_category_b(
            name="stop_atr_multiplier",
            current_value=2.0,
            degradation_signal="ATR percentile 1.8x calibration baseline",
            adjustment_factor=1.25,
        )
        assert result.value == pytest.approx(2.5)
        assert "Category B" in result.formula
        assert "ATR percentile" in result.formula

    def test_missing_degradation_signal_rejected(self):
        """No cited evidence -> a guess, not a justified adaptation -> rejected."""
        with pytest.raises(ValueError):
            AdaptiveParameterResolver.adapt_for_edge_category_b(
                name="x", current_value=2.0, degradation_signal="",
                adjustment_factor=1.1,
            )

    def test_bounds_applied(self):
        result = AdaptiveParameterResolver.adapt_for_edge_category_b(
            name="x", current_value=2.0, degradation_signal="evidence",
            adjustment_factor=5.0, min_value=0.5, max_value=3.0,
        )
        assert result.value == 3.0

    def test_invalid_bounds_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveParameterResolver.adapt_for_edge_category_b(
                name="x", current_value=2.0, degradation_signal="evidence",
                adjustment_factor=1.0, min_value=5.0, max_value=1.0,
            )
