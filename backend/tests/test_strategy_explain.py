"""Tests for structured, numeric strategy decision explanations.

Covers three layers:
  1. ExplanationBuilder (pure unit + exception-safety contract).
  2. DiagnosticsStore.record_explanation + to_dict exposure.
  3. Engine integration: _execute_strategy builds and records an explanation, and
     real strategies populate exact numeric checks/metrics (trend warmup,
     mean-reversion entry).
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from app.models import Bot, BotStatus
from app.services.diagnostics import DiagnosticsStore, diagnostics_store
from app.services.strategy_explain import (
    DecisionCheck,
    DecisionExplanation,
    ExplanationBuilder,
)
from app.services.trading_engine import TradingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bot(strategy: str, bot_id: int, **kw) -> Bot:
    b = Bot(
        name=f"explain-{bot_id}",
        trading_pair="BTC/USDT",
        strategy=strategy,
        strategy_params=kw.get("strategy_params", {}),
        budget=1000.0,
        current_balance=kw.get("current_balance", 1000.0),
        is_dry_run=True,
        status=BotStatus.RUNNING,
        strategy_state=None,
    )
    b.id = bot_id
    return b


# ---------------------------------------------------------------------------
# 1. ExplanationBuilder
# ---------------------------------------------------------------------------


class TestExplanationBuilder:
    def test_check_metric_and_finalize_from_signal(self):
        import types

        b = ExplanationBuilder("mean_reversion")
        b.check("RSI", 34.8, "< 30.0", False).metric("atr", 218.4)
        b.finalize(types.SimpleNamespace(action="HOLD", reason="Waiting for lower band"))
        d = b.to_dict()

        assert d["strategy"] == "mean_reversion"
        assert d["decision"] == "hold"  # normalized lower-case
        assert d["reason"] == "Waiting for lower band"
        assert d["checks"] == [
            {"name": "RSI", "current": 34.8, "required": "< 30.0", "passed": False, "detail": None}
        ]
        assert d["metrics"]["atr"] == 218.4
        assert d["evaluated_at"] is not None

    def test_explicit_summary_overrides_signal_reason(self):
        import types

        b = ExplanationBuilder("grid")
        b.summary(reason="Price has not moved one spacing from the center.")
        b.finalize(types.SimpleNamespace(action="hold", reason="Grid: No levels triggered"))
        d = b.to_dict()
        # The explicit summary reason wins over the terse signal reason.
        assert d["reason"] == "Price has not moved one spacing from the center."
        assert d["decision"] == "hold"

    def test_float_rounding_and_non_numeric_coercion(self):
        b = ExplanationBuilder("x")
        b.metric("noisy", 1.0 / 3.0)
        b.check("obj", object(), ">0", True)  # object stringified, never raises
        d = b.to_dict()
        assert d["metrics"]["noisy"] == round(1.0 / 3.0, 6)
        assert isinstance(d["checks"][0]["current"], str)

    def test_state_and_next_trade(self):
        b = ExplanationBuilder("mean_reversion")
        b.state("WAITING_LOWER_BAND")
        b.next_trade(
            current=60514.0, current_label="Current price",
            target=60076.0, target_label="Lower band", distance=438.0,
            status="Needs another 438.00 points lower",
        )
        d = b.to_dict()
        assert d["state"] == "WAITING_LOWER_BAND"
        assert d["next_trade"]["current"] == 60514.0
        assert d["next_trade"]["target"] == 60076.0
        assert d["next_trade"]["distance"] == 438.0
        assert "438" in d["next_trade"]["status"]

    def test_state_and_next_trade_default_none(self):
        b = ExplanationBuilder("x")
        d = b.to_dict()
        assert d["state"] is None
        assert d["next_trade"] is None

    def test_candidates_and_select(self):
        b = ExplanationBuilder("auto_mode")
        b.candidate({"strategy": "trend_following", "final": 7.8})
        b.candidate({"strategy": "mean_reversion", "final": 2.9})
        b.select("trend_following")
        d = b.to_dict()
        assert d["selected"] == "trend_following"
        assert [c["strategy"] for c in d["candidates"]] == ["trend_following", "mean_reversion"]

    def test_builder_is_exception_safe(self):
        """A bug inside a builder call must never propagate (observability rule)."""
        b = ExplanationBuilder("x")

        class Boom:
            def __str__(self):
                raise RuntimeError("boom")

        # Coercing Boom would raise inside to_dict's str() — must be swallowed and
        # still return a well-formed dict.
        b.metric("bad", Boom())
        d = b.to_dict()
        assert d["strategy"] == "x"
        assert "checks" in d and "metrics" in d


# ---------------------------------------------------------------------------
# 2. DiagnosticsStore wiring
# ---------------------------------------------------------------------------


class TestDiagnosticsExplanation:
    def test_record_and_expose_explanation(self):
        store = DiagnosticsStore()
        payload = {"strategy": "grid", "decision": "hold", "checks": [], "metrics": {}}
        store.record_explanation(42, payload)
        assert store.get(42).to_dict()["explanation"] == payload

    def test_explanation_defaults_none(self):
        store = DiagnosticsStore()
        store.record_evaluation(7)
        assert store.get(7).to_dict()["explanation"] is None


# ---------------------------------------------------------------------------
# 3. Engine integration
# ---------------------------------------------------------------------------


class TestEngineExplanationIntegration:
    @pytest.mark.asyncio
    async def test_execute_strategy_records_explanation(self):
        """_execute_strategy must build a fresh explanation and record it with the
        strategy name, decision, and the strategy's own numeric checks."""
        engine = TradingEngine()
        bot = _bot("trend_following", 7001, strategy_params={"long_period": 100})
        session = AsyncMock()

        signal = await engine._execute_strategy(bot, 50000.0, session)

        assert signal.action == "hold"  # warming up
        exp = diagnostics_store.get(7001).last_explanation
        assert exp is not None
        assert exp["strategy"] == "trend_following"
        assert exp["decision"] == "hold"
        assert exp["metrics"]["current_price"] == 50000.0
        # The exact warmup gate, with numbers.
        assert any(c["name"] == "Data collected" and not c["passed"] for c in exp["checks"])

    @pytest.mark.asyncio
    async def test_mean_reversion_entry_checks_have_real_numbers(self, monkeypatch):
        """With a seeded bar window and price at the lower band, the explanation
        must expose the Bollinger bands and the lower-band-touch entry gate with
        the actual values that drove the decision."""
        engine = TradingEngine()
        bot = _bot(
            "mean_reversion", 7002,
            strategy_params={"regime_filter_enabled": False, "bollinger_period": 20},
        )

        base = datetime.utcnow() - timedelta(seconds=4000)
        bars = [
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "start_ts": base}
            for _ in range(20)
        ]
        engine._mean_reversion_states = {
            7002: {
                "bars": bars,
                "current_bar": {
                    "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                    "start_ts": datetime.utcnow(),
                },
                "entry_price": None,
                "entry_atr": None,
                "hard_stop": None,
                "bars_since_entry": 0,
                "last_exit_time": None,
            }
        }

        async def _no_positions(_bot_id, _session):
            return []

        monkeypatch.setattr(engine, "_get_bot_positions", _no_positions)
        session = AsyncMock()

        signal = await engine._execute_strategy(bot, 95.0, session)

        exp = diagnostics_store.get(7002).last_explanation
        assert exp["strategy"] == "mean_reversion"
        # Bands are exposed with the exact computed values (flat series → bands==SMA).
        assert exp["metrics"]["lower_bb"] == 100.0
        assert exp["metrics"]["middle_bb_sma"] == 100.0
        assert exp["metrics"]["current_price"] == 95.0
        # The entry gate is present with current-vs-required.
        touch = next(c for c in exp["checks"] if c["name"] == "Lower band touch")
        assert touch["required"] == "<= 100.00"
        # bar close (100) <= lower band (100) → entry fires.
        assert touch["passed"] is True
        assert signal.action == "buy"
        assert exp["decision"] == "buy"
        # Machine-readable state + next-trade preview come from the strategy.
        assert exp["state"] in {"ENTRY_ARMED", "WAITING_LOWER_BAND"}
        assert exp["next_trade"]["target"] == 100.0  # the lower band
        assert exp["next_trade"]["current_label"] == "Current price"

    @pytest.mark.asyncio
    async def test_trend_following_warmup_state(self):
        """A warming-up trend bot reports WARMING_UP with the data-collected gate."""
        engine = TradingEngine()
        bot = _bot("trend_following", 7003, strategy_params={"long_period": 100})
        session = AsyncMock()
        await engine._execute_strategy(bot, 50000.0, session)
        exp = diagnostics_store.get(7003).last_explanation
        assert exp["state"] == "WARMING_UP"
