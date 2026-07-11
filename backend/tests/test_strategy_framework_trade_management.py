"""Unit tests for Pillar 6's shared TradeManagementMonitor
(add-strategy-decision-framework, Phase 0.4).
"""
from __future__ import annotations

from app.services.strategy_framework.trade_management import (
    TradeManagementMonitor,
    TradeManagementReport,
)


def _regime(trend="flat", volatility="medium") -> dict:
    return {
        "trend_state": trend,
        "volatility_state": volatility,
        "volatility_direction": "stable",
        "liquidity_state": "normal",
    }


class TestTradeManagementMonitor:
    def setup_method(self):
        self.monitor = TradeManagementMonitor()

    def test_intact_thesis_no_hooks_fired(self):
        report = self.monitor.evaluate(
            current_regime=_regime(trend="up"),
            allowed_regimes=["trend_up"],
        )
        assert isinstance(report, TradeManagementReport)
        assert report.thesis_intact is True
        assert report.should_exit is False
        assert report.volatility_changed is False
        assert report.stop_should_tighten is False
        assert report.take_partial_profit is False
        assert report.reasons == []

    def test_thesis_invalidated_signals_exit(self):
        """Mirrors spec.md's 'Thesis invalidated after entry' scenario:
        conditions that justified entry change materially -> the exit
        evaluation reflects it."""
        report = self.monitor.evaluate(
            current_regime=_regime(trend="down"),
            allowed_regimes=["trend_up"],
        )
        assert report.thesis_intact is False
        assert report.should_exit is True
        assert any("thesis" in r for r in report.reasons)

    def test_volatility_hook_fires(self):
        report = self.monitor.evaluate(
            current_regime=_regime(trend="up"),
            allowed_regimes=["trend_up"],
            volatility_check=lambda: (True, "ATR doubled since entry"),
        )
        assert report.volatility_changed is True
        assert any("ATR doubled" in r for r in report.reasons)

    def test_stop_tighten_hook_fires(self):
        report = self.monitor.evaluate(
            current_regime=_regime(trend="up"),
            allowed_regimes=["trend_up"],
            stop_tighten_check=lambda: (True, "price extended 2x ATR from entry"),
        )
        assert report.stop_should_tighten is True

    def test_partial_profit_hook_fires(self):
        report = self.monitor.evaluate(
            current_regime=_regime(trend="up"),
            allowed_regimes=["trend_up"],
            partial_profit_check=lambda: (True, "reached first target"),
        )
        assert report.take_partial_profit is True

    def test_missing_hooks_treated_as_not_fired_not_error(self):
        report = self.monitor.evaluate(
            current_regime=_regime(trend="up"),
            allowed_regimes=["trend_up"],
        )
        assert report.volatility_changed is False
        assert report.stop_should_tighten is False
        assert report.take_partial_profit is False

    def test_all_four_checks_can_fire_together(self):
        report = self.monitor.evaluate(
            current_regime=_regime(trend="down"),
            allowed_regimes=["trend_up"],
            volatility_check=lambda: (True, "vol up"),
            stop_tighten_check=lambda: (True, "extended"),
            partial_profit_check=lambda: (True, "target hit"),
        )
        assert report.thesis_intact is False
        assert report.volatility_changed is True
        assert report.stop_should_tighten is True
        assert report.take_partial_profit is True
        assert len(report.reasons) == 4

    def test_deterministic_for_same_inputs_and_hooks(self):
        kwargs = dict(
            current_regime=_regime(trend="up"),
            allowed_regimes=["trend_up"],
            volatility_check=lambda: (False, ""),
        )
        r1 = self.monitor.evaluate(**kwargs)
        r2 = self.monitor.evaluate(**kwargs)
        assert r1 == r2
