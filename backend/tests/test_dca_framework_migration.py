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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routers.config import STRATEGIES
from app.services.trading_engine import MIN_ORDER_USD, TradingEngine


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
