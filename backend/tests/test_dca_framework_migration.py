"""Tests for dca_accumulator's migration to the Strategy Decision Framework
(add-strategy-decision-framework, Phase 6).

Phase 6.3 — Market Suitability (Pillar 2) re-scoped away from market timing.
A classic DCA is DIRECTION-AGNOSTIC: it accumulates fixed chunks on schedule
no matter the market conditions. The only structural condition that halts
accumulation is invalidation of the long-term investment thesis
(thesis_invalidated) - never a short/medium-term price-direction forecast.
Execution-feasibility and portfolio-constraint suitability are enforced by
existing downstream mechanisms (MIN_ORDER_USD floor, fee-adjusted cap, the
execution pipeline's PortfolioRiskService), not by this gate.

These tests are the INVERSE of the other five strategies' suitability tests:
where they assert "no trade in a disallowed regime", DCA asserts "a scheduled
buy still fires in a downtrend, absent an execution/portfolio/thesis problem".
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routers.config import STRATEGIES
from app.services.strategy_framework.edge_management import (
    DCA_THESIS_INVALIDATION_CONDITIONS,
    DcaEdgeManager,
    EdgeCategory,
)
from app.services.trading_engine import _BUY_BALANCE_FRACTION, MIN_ORDER_USD, TradingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bot(balance: float = 100_000.0, budget: float = 100_000.0, bot_id: int = 1):
    return SimpleNamespace(
        id=bot_id,
        name="DcaBot",
        trading_pair="BTC/USDT",
        strategy="dca_accumulator",
        strategy_params={},
        budget=budget,
        current_balance=balance,
        started_at=None,
        status=None,
    )


def _engine(*, regime_state: str = "down"):
    """Engine with no prior orders. `regime_state` is what the (non-classic)
    regime overlay would see IF it were consulted - default 'down' so that any
    accidental direction-gating would surface as a paused/hold."""
    engine = TradingEngine()
    engine._get_last_order = AsyncMock(return_value=None)
    engine._get_order_count = AsyncMock(return_value=0)
    engine._get_price_history = MagicMock(return_value=[
        {"timestamp": "2026-01-01T00:00:00", "price": 100.0 - i} for i in range(60)
    ])
    engine._detect_market_regime = MagicMock(return_value={"trend_state": regime_state})
    return engine


def _dca_config_defaults() -> dict:
    info = next(s for s in STRATEGIES if s.name == "dca_accumulator")
    return {k: v.get("default") for k, v in info.parameters.items()}


# ---------------------------------------------------------------------------
# 1. Direction-agnosticism (the inverse suitability test)
# ---------------------------------------------------------------------------

class TestDcaDirectionAgnostic:
    @pytest.mark.asyncio
    async def test_trend_down_does_not_block_scheduled_buy(self):
        """A pure downtrend, with no execution/portfolio/thesis problem, must
        NOT block a scheduled buy. This is the whole point of a classic DCA and
        the inverse of the other five strategies' suitability gates."""
        engine = _engine(regime_state="down")
        bot = _bot()
        sig = await engine._strategy_dca(bot, 50_000.0, {}, AsyncMock())
        assert sig.action == "buy", (
            f"classic DCA must buy through a downtrend; got {sig.action!r} "
            f"({getattr(sig, 'reason', None)!r})"
        )
        assert "regime" not in (sig.reason or "").lower()

    @pytest.mark.asyncio
    async def test_regime_detector_not_consulted_by_default(self):
        """With the overlay off (default), the regime detector must never be
        consulted - direction is not an input to a classic DCA's decision."""
        engine = _engine(regime_state="down")
        bot = _bot()
        await engine._strategy_dca(bot, 50_000.0, {}, AsyncMock())
        engine._detect_market_regime.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Thesis-invalidation structural stop (the ONLY halt on suitability grounds)
# ---------------------------------------------------------------------------

class TestThesisInvalidationStop:
    @pytest.mark.asyncio
    async def test_thesis_invalidated_halts_accumulation(self):
        engine = _engine(regime_state="up")  # even a great regime is irrelevant
        bot = _bot()
        sig = await engine._strategy_dca(
            bot, 50_000.0, {"thesis_invalidated": True}, AsyncMock()
        )
        assert sig.action == "hold"
        assert "thesis" in (sig.reason or "").lower()

    @pytest.mark.asyncio
    async def test_thesis_valid_emits_positive_check(self):
        """Pillar 8: a passing suitability gate is explained, not only failures."""
        engine = _engine(regime_state="down")
        bot = _bot()
        await engine._strategy_dca(bot, 50_000.0, {}, AsyncMock())
        checks = engine._explain(bot.id).exp.checks
        thesis_checks = [c for c in checks if "thesis" in c.name.lower()]
        assert thesis_checks, "expected a 'Long-term thesis valid' check"
        assert all(c.passed for c in thesis_checks)


# ---------------------------------------------------------------------------
# 3. Non-classic market-timing overlay (explicit opt-in, off by default)
# ---------------------------------------------------------------------------

class TestNonClassicRegimeOverlay:
    def test_overlay_defaults_off_in_config(self):
        defaults = _dca_config_defaults()
        assert defaults["regime_filter_enabled"] is False
        assert "thesis_invalidated" in defaults
        assert defaults["thesis_invalidated"] is False

    @pytest.mark.asyncio
    async def test_overlay_opt_in_still_pauses_in_downtrend(self):
        """The market-timing overlay is preserved for operators who explicitly
        opt in - regime_filter_enabled=True in a disallowed regime pauses."""
        engine = _engine(regime_state="down")
        bot = _bot()
        sig = await engine._strategy_dca(
            bot, 50_000.0,
            {"regime_filter_enabled": True, "allowed_regimes": ["trend_up", "trend_flat"]},
            AsyncMock(),
        )
        assert sig.action == "hold"
        assert "regime" in (sig.reason or "").lower()

    @pytest.mark.asyncio
    async def test_overlay_opt_in_allows_permitted_regime(self):
        engine = _engine(regime_state="up")
        bot = _bot()
        sig = await engine._strategy_dca(
            bot, 50_000.0,
            {"regime_filter_enabled": True, "allowed_regimes": ["trend_up", "trend_flat"]},
            AsyncMock(),
        )
        assert sig.action == "buy"


# ---------------------------------------------------------------------------
# 4. Portfolio/budget floor still halts (regression - the (b) category)
# ---------------------------------------------------------------------------

class TestBudgetExhaustionStop:
    @pytest.mark.asyncio
    async def test_balance_below_min_order_halts(self):
        engine = _engine(regime_state="down")
        bot = _bot(balance=MIN_ORDER_USD - 1.0)
        sig = await engine._strategy_dca(bot, 50_000.0, {}, AsyncMock())
        assert sig.action == "hold"
        assert "accumulation complete" in (sig.reason or "").lower()


# ---------------------------------------------------------------------------
# 5. Strategy Edge Management (Pillar 7) — DcaEdgeManager classifier
# ---------------------------------------------------------------------------

class TestDcaEdgeManagerClassification:
    """A classic DCA fails ONLY when its accumulation thesis is objectively
    invalidated - never because the market falls. These tests pin the
    classification axis: C = thesis invalidation only; A/B = operational; and
    there is deliberately NO price/performance input that could trip C."""

    @pytest.mark.parametrize("condition", DCA_THESIS_INVALIDATION_CONDITIONS)
    def test_each_thesis_condition_is_category_C(self, condition):
        status = DcaEdgeManager().evaluate(thesis_invalidation={condition: True})
        assert status.category is EdgeCategory.C
        assert status.should_stop is True
        assert status.can_adapt is False and status.should_wait is False
        assert condition in status.reason

    def test_operational_pause_is_category_A(self):
        status = DcaEdgeManager().evaluate(operational_pause="budget temporarily exhausted")
        assert status.category is EdgeCategory.A
        assert status.should_wait is True
        assert status.should_stop is False  # A is never a permanent stop

    def test_operational_adaptation_is_category_B(self):
        status = DcaEdgeManager().evaluate(operational_adaptation="chunk floored to $10")
        assert status.category is EdgeCategory.B
        assert status.can_adapt is True
        assert status.should_stop is False

    def test_no_conditions_is_healthy_none(self):
        status = DcaEdgeManager().evaluate()
        assert status.category is EdgeCategory.NONE
        assert not status.should_stop and not status.should_wait

    def test_unrecognised_condition_cannot_invalidate_thesis(self):
        """A price/performance/trend-flavoured condition passed by mistake must
        be ignored - only the objective, whitelisted conditions invalidate."""
        status = DcaEdgeManager().evaluate(
            thesis_invalidation={"price_fell": True, "drawdown_deep": True, "trend_down": True}
        )
        assert status.category is EdgeCategory.NONE
        assert status.should_stop is False

    def test_thesis_invalidation_takes_precedence(self):
        status = DcaEdgeManager().evaluate(
            thesis_invalidation={"asset_delisted": True},
            operational_pause="budget exhausted",
            operational_adaptation="chunk floored",
        )
        assert status.category is EdgeCategory.C

    def test_classifier_has_no_price_or_performance_input(self):
        """Structural guarantee: DcaEdgeManager.evaluate exposes no pnl/price/
        drawdown/win-rate parameter, so drawdown cannot be an input at all."""
        params = set(inspect.signature(DcaEdgeManager.evaluate).parameters) - {"self"}
        assert params == {"thesis_invalidation", "operational_pause",
                          "operational_adaptation", "now"}
        forbidden = ("pnl", "price", "drawdown", "return", "win", "profit", "loss")
        assert not any(any(f in p for f in forbidden) for p in params)


class TestDcaEdgeWiring:
    """Edge management wired into _strategy_dca: C halts, drawdown never does,
    and the operational A/B paths are classified where DCA actually hits them."""

    @pytest.mark.asyncio
    async def test_drawdown_regime_never_trips_category_C(self):
        """The whole point: a falling market (down regime, low price) keeps
        accumulating and is classified NONE - never C."""
        engine = _engine(regime_state="down")
        bot = _bot()
        sig = await engine._strategy_dca(bot, 100.0, {}, AsyncMock())  # deeply "down" price
        assert sig.action == "buy"
        assert engine._explain(bot.id).exp.metrics["edge_status_category"] == EdgeCategory.NONE.value

    @pytest.mark.asyncio
    @pytest.mark.parametrize("condition", DCA_THESIS_INVALIDATION_CONDITIONS)
    async def test_each_thesis_condition_halts_accumulation(self, condition):
        engine = _engine(regime_state="up")
        bot = _bot()
        sig = await engine._strategy_dca(bot, 50_000.0, {condition: True}, AsyncMock())
        assert sig.action == "hold"
        assert "thesis invalidated" in (sig.reason or "").lower()
        assert engine._explain(bot.id).exp.metrics["edge_status_category"] == EdgeCategory.C.value

    @pytest.mark.asyncio
    async def test_legacy_thesis_invalidated_param_maps_to_operator_invalidated(self):
        engine = _engine(regime_state="up")
        bot = _bot()
        sig = await engine._strategy_dca(bot, 50_000.0, {"thesis_invalidated": True}, AsyncMock())
        assert sig.action == "hold"
        assert engine._explain(bot.id).exp.metrics["edge_status_category"] == EdgeCategory.C.value

    @pytest.mark.asyncio
    async def test_budget_exhaustion_classified_category_A(self):
        engine = _engine(regime_state="down")
        bot = _bot(balance=MIN_ORDER_USD - 1.0)
        sig = await engine._strategy_dca(bot, 50_000.0, {}, AsyncMock())
        assert sig.action == "hold"
        assert engine._explain(bot.id).exp.metrics["edge_status_category"] == EdgeCategory.A.value

    @pytest.mark.asyncio
    async def test_chunk_floor_classified_category_B(self):
        # balance 50 * 10% = $5 chunk, below the $10 minimum but affordable ->
        # floored to $10 (operational adaptation, Category B).
        engine = _engine(regime_state="down")
        bot = _bot(balance=50.0)
        sig = await engine._strategy_dca(bot, 50_000.0, {}, AsyncMock())
        assert sig.action == "buy"
        assert sig.amount == MIN_ORDER_USD
        assert engine._explain(bot.id).exp.metrics["edge_status_category"] == EdgeCategory.B.value


# ---------------------------------------------------------------------------
# 6. Position Sizing (Pillar 5) — flat, schedule-driven, deterministic
# ---------------------------------------------------------------------------

class TestDcaFlatSizing:
    """Classic DCA deploys capital consistently, not by conviction. Sizing must
    be flat, deterministic, and independent of market state - no Decision-Score
    weighting, no market timing through sizing, no price-based adaptation. The
    only legitimate adjuster is objective portfolio governance, enforced
    downstream by the pipeline (PortfolioRiskService / StrategyCapacityService),
    not re-implemented here."""

    @pytest.mark.asyncio
    async def test_flat_percentage_of_balance(self):
        engine = _engine(regime_state="down")
        bot = _bot(balance=1_000.0)
        sig = await engine._strategy_dca(bot, 50_000.0, {"amount_percent": 10}, AsyncMock())
        assert sig.action == "buy"
        assert sig.amount == pytest.approx(100.0)  # 10% of 1000, well below cap

    @pytest.mark.asyncio
    async def test_fixed_usd_overrides_percent_deterministically(self):
        engine = _engine(regime_state="down")
        bot = _bot(balance=1_000.0)
        sig = await engine._strategy_dca(
            bot, 50_000.0, {"amount_percent": 10, "amount_usd": 250}, AsyncMock()
        )
        assert sig.amount == pytest.approx(250.0)

    @pytest.mark.asyncio
    async def test_sizing_independent_of_price_and_regime(self):
        """The core 'no market-timing through sizing' guarantee: identical
        balance + params must yield an identical chunk regardless of price or
        (overlay-off) regime state."""
        amounts = []
        for price, regime in [(10.0, "down"), (50_000.0, "up"), (123_456.0, "flat")]:
            engine = _engine(regime_state=regime)
            bot = _bot(balance=1_000.0)
            sig = await engine._strategy_dca(bot, price, {"amount_percent": 10}, AsyncMock())
            amounts.append(sig.amount)
        assert len(set(amounts)) == 1, f"sizing varied with market state: {amounts}"
        assert amounts[0] == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_deterministic_across_repeated_calls(self):
        engine = _engine(regime_state="down")
        bot = _bot(balance=777.0)
        a = await engine._strategy_dca(bot, 50_000.0, {"amount_percent": 7}, AsyncMock())
        b = await engine._strategy_dca(bot, 40_000.0, {"amount_percent": 7}, AsyncMock())
        assert a.amount == b.amount == pytest.approx(777.0 * 0.07)

    @pytest.mark.asyncio
    async def test_never_exceeds_fee_adjusted_budget(self):
        """Even a 100% chunk is capped to the fee/spread-adjusted balance so a
        buy can never exceed available funds (strategy-layer budget respect;
        portfolio-level caps are enforced downstream)."""
        engine = _engine(regime_state="down")
        bot = _bot(balance=1_000.0)
        sig = await engine._strategy_dca(bot, 50_000.0, {"amount_percent": 100}, AsyncMock())
        assert sig.amount <= 1_000.0 * _BUY_BALANCE_FRACTION + 1e-9

    def test_strategy_source_has_no_decision_score_sizing(self):
        """Structural guarantee: _strategy_dca contains no Decision-Score /
        conviction-weighted sizing path at all."""
        src = inspect.getsource(TradingEngine._strategy_dca)
        lowered = src.lower()
        assert "decision_score" not in lowered
        assert "decisionscore" not in lowered
