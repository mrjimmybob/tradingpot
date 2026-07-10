"""Regression tests: funding_carry is fully removed (fix-strategy-integrity).

funding_carry cannot generate funding yield on a spot-only bot - it was an
unmanaged directional long strategy mislabeled as a carry trade. It has been
deleted completely: no catalog entry, no dispatch, no capability entry, no
data. These tests lock down that removal.
"""
import pytest

from app.routers.bots import validate_strategy_params, _STRATEGY_PARAM_VALIDATORS
from app.routers.config import STRATEGIES
from app.services.trading_engine import TradingEngine, _ALPHA_STRATEGIES


def test_not_in_strategy_catalog():
    """GET /strategies (backed by STRATEGIES) must not list funding_carry."""
    assert "funding_carry" not in {s.name for s in STRATEGIES}


def test_not_in_alpha_strategies_rotation_set():
    assert "funding_carry" not in _ALPHA_STRATEGIES


def test_not_in_param_validators():
    assert "funding_carry" not in _STRATEGY_PARAM_VALIDATORS


def test_bot_creation_rejects_it():
    """POST /bots validation must reject funding_carry as an unknown strategy."""
    errors = validate_strategy_params("funding_carry", {})
    assert errors, "expected a validation error for an unknown strategy"
    assert any("funding_carry" in e and "Unknown strategy" in e for e in errors)


def test_no_dispatch_executor():
    """Auto Mode (and direct dispatch) must not be able to resolve an executor."""
    engine = TradingEngine()
    assert engine._get_strategy_executor("funding_carry") is None


def test_not_in_auto_mode_capability_table():
    """Auto Mode's regime-based eligibility table must not carry a stale entry
    that could make funding_carry scoreable/selectable again."""
    engine = TradingEngine()
    caps = engine._get_strategy_capabilities()
    assert "funding_carry" not in caps


def test_no_funding_carry_strategy_method_left_on_engine():
    """No dead code path: the implementation method itself must be gone."""
    assert not hasattr(TradingEngine, "_strategy_funding_carry")


def test_no_funding_diagnostic_module_left():
    """The funding-rate diagnostic module had no other caller once
    _strategy_funding_carry was removed - it must be gone too, not orphaned."""
    with pytest.raises(ModuleNotFoundError):
        import app.services.funding_diagnostic  # noqa: F401


def test_report_renders_unrecognized_strategy_name_without_raising():
    """Reports must not special-case or crash on an unrecognized strategy_used
    value (defensive regression - funding_carry data itself has been purged
    from the database per the operator's explicit decision, not kept around
    to test against)."""
    from app.models.trade import Trade, TradeSide

    trade = Trade(
        id=1, order_id=1, owner_id="1", bot_id=1, exchange="simulated",
        trading_pair="BTC/USDT", side=TradeSide.BUY, base_asset="BTC",
        quote_asset="USDT", base_amount=0.001, quote_amount=50.0, price=50000.0,
        strategy_used="some_removed_or_unrecognized_strategy",
    )
    rendered = trade.to_dict()
    assert rendered["strategy_used"] == "some_removed_or_unrecognized_strategy"
