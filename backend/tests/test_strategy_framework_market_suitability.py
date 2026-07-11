"""Unit tests for Pillar 2's shared MarketSuitabilityGate
(add-strategy-decision-framework, Phase 0.1).

Pure-function tests only — no bot/session/DB required, matching how
``evaluate_reward_risk`` (test_reward_risk_gate.py) is tested in isolation.
"""
from __future__ import annotations

from app.services.strategy_framework.market_suitability import (
    ALL_REGIMES,
    MarketSuitabilityGate,
    MarketSuitabilityResult,
)


def _regime(trend="flat", volatility="medium", direction="stable", liquidity="normal") -> dict:
    return {
        "trend_state": trend,
        "volatility_state": volatility,
        "volatility_direction": direction,
        "liquidity_state": liquidity,
    }


class TestRegimeTags:
    def test_builds_all_four_tag_families(self):
        tags = MarketSuitabilityGate.regime_tags(
            _regime(trend="up", volatility="high", direction="expanding", liquidity="low")
        )
        assert tags == [
            "trend_up",
            "volatility_high",
            "volatility_expanding",
            "liquidity_low",
        ]

    def test_missing_keys_fall_back_to_neutral_defaults(self):
        # Matches _detect_market_regime_bar_based's own pre-warm-up neutral regime.
        tags = MarketSuitabilityGate.regime_tags({})
        assert tags == [
            "trend_flat",
            "volatility_medium",
            "volatility_stable",
            "liquidity_normal",
        ]

    def test_deterministic(self):
        regime = _regime(trend="down", volatility="low")
        assert MarketSuitabilityGate.regime_tags(regime) == MarketSuitabilityGate.regime_tags(regime)


class TestEvaluate:
    def setup_method(self):
        self.gate = MarketSuitabilityGate()

    def test_no_allowed_regimes_is_unsuitable(self):
        result = self.gate.evaluate(_regime(), [])
        assert isinstance(result, MarketSuitabilityResult)
        assert result.is_suitable is False
        assert result.matched_tags == []

    def test_all_sentinel_is_always_suitable(self):
        result = self.gate.evaluate(_regime(trend="down", volatility="high"), [ALL_REGIMES])
        assert result.is_suitable is True
        assert result.matched_tags == result.regime_tags

    def test_matching_tag_is_suitable(self):
        result = self.gate.evaluate(
            _regime(trend="flat", volatility="medium"),
            ["trend_flat", "volatility_high"],
        )
        assert result.is_suitable is True
        assert "trend_flat" in result.matched_tags

    def test_no_matching_tag_is_unsuitable(self):
        result = self.gate.evaluate(
            _regime(trend="up", volatility="low", direction="stable", liquidity="normal"),
            ["trend_down", "volatility_high"],
        )
        assert result.is_suitable is False
        assert result.matched_tags == []
        assert "outside allowed" in result.reason

    def test_volatility_direction_tag_can_match(self):
        # volatility_breakout's own convention: allowed_regimes=["volatility_expanding"]
        # matches the DIRECTION tag, not the LEVEL tag.
        result = self.gate.evaluate(
            _regime(trend="flat", volatility="low", direction="expanding"),
            ["volatility_expanding"],
        )
        assert result.is_suitable is True
        assert result.matched_tags == ["volatility_expanding"]

    def test_result_is_deterministic_for_same_inputs(self):
        regime = _regime(trend="up")
        allowed = ["trend_up"]
        r1 = self.gate.evaluate(regime, allowed)
        r2 = self.gate.evaluate(regime, allowed)
        assert r1 == r2


class TestGate:
    def test_gate_returns_plain_bool(self):
        gate = MarketSuitabilityGate()
        assert gate.gate(_regime(trend="flat"), ["trend_flat"]) is True
        assert gate.gate(_regime(trend="up"), ["trend_flat"]) is False

    def test_gate_blocks_unsuitable_regime_even_with_other_matches_absent(self):
        gate = MarketSuitabilityGate()
        # dip_recovery-style capability declaration.
        allowed = ["trend_down", "volatility_expanding", "volatility_high"]
        assert gate.gate(_regime(trend="up", volatility="low", direction="stable"), allowed) is False
