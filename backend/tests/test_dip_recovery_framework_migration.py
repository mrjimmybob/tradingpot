"""Tests for dip_recovery's migration to the Strategy Decision Framework
(add-strategy-decision-framework, Phase 3).

Covers the full decision flow the migration adds on top of dip_recovery's
existing state machine: Market Suitability Gate (Pillar 2, via the price-only
_detect_market_regime - see the audit for why) -> Evidence-Based Decision
Score over the formerly-diagnostic-only opportunity signals (Pillar 3) ->
Strategy Edge Management (Pillar 7) -> Decision-Score-weighted sizing
(Pillar 5) -> StrategyProposal (Pillar 10) -> Standalone Adapter.

The pre-migration "never buy on the way down" safety preconditions are
preserved and still covered by test_dip_recovery_strategy.py; this file covers
the framework layer.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.trading_engine import TradingEngine, TradeSignal, _DipRecoveryState
from app.services.strategy_framework.edge_management import EdgeCategory, StrategyEdgeManager
from app.services.strategy_framework.proposal import Direction, ExecutionIntent, StrategyProposal
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A warmup, a real decline, then a confirmed bounce off the low - the same
# shape test_dip_recovery_strategy.py uses to reach a BUY.
_WARMUP = [100.0, 100.05, 99.98, 100.02, 100.1, 99.95, 100.0, 100.03,
           99.99, 100.01, 100.02, 99.98, 100.0, 100.01, 99.99]
_DECLINE = [98.0, 96.0, 94.0, 92.0, 90.0, 89.5, 89.0]
_BOUNCE = [89.3, 89.6, 90.0, 90.4, 90.8, 91.2, 91.6, 92.0]
_ENTRY_PRICES = _WARMUP + _DECLINE + _BOUNCE


def _bot(balance: float = 100_000.0, bot_id: int = 1):
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "dip_recovery"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    bot.exchange_fee = 0.1
    bot.strategy_params = {}
    return bot


def _capture_proposals(monkeypatch) -> list:
    captured: list = []
    real_fn = StandaloneAdapter.to_trade_signal

    def spy(proposal, **kwargs):
        captured.append(proposal)
        return real_fn(proposal, **kwargs)

    monkeypatch.setattr(StandaloneAdapter, "to_trade_signal", staticmethod(spy))
    return captured


async def _drive_to_buy(engine, bot, monkeypatch, *, params=None):
    """Feed the decline->recovery series until a BUY fires. Returns
    (buy_signal_or_None, captured_proposals)."""
    engine._get_bot_positions = AsyncMock(return_value=[])
    captured = _capture_proposals(monkeypatch)
    params = params or {}
    buy = None
    for p in _ENTRY_PRICES:
        sig = await engine._strategy_dip_recovery(bot, p, params, AsyncMock())
        if sig.action == "buy":
            buy = sig
            break
    return buy, captured


def _long_open_state(entry=90.0, atr=1.0):
    return {
        "state": _DipRecoveryState.LONG_OPEN,
        "reference_high": 100.0, "reference_high_time": None,
        "lowest_price": 89.0, "lowest_price_time": None,
        "tracking_started_at": None, "ticks_since_new_low": 5,
        "entry_price": entry, "entry_time": None, "entry_atr": atr,
        "highest_price_since_entry": entry,
        "trailing_stop": entry - atr * 1.5,
        "take_profit": entry + atr * 3.0,
        "emergency_stop": entry - atr * 5.0,
        "cooldown_until": None, "last_exit_was_loss": None,
    }


# ---------------------------------------------------------------------------
# 1. Evidence generation
# ---------------------------------------------------------------------------

class TestEvidenceGeneration:
    @pytest.mark.asyncio
    async def test_entry_names_all_four_evidence_factors(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        buy, captured = await _drive_to_buy(engine, bot, monkeypatch)
        assert buy is not None and buy.action == "buy"
        prop = next(p for p in captured if p.direction == Direction.BUY)
        names = [c.name for c in prop.decision_score.contributions]
        assert names == [
            "Decline depth", "Recovery strength",
            "Reversal momentum (EMA slope)", "Base stability (no new low)",
        ]

    @pytest.mark.asyncio
    async def test_explanation_contains_decision_and_regime_metrics(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _, captured = await _drive_to_buy(engine, bot, monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        m = prop.explanation["metrics"]
        assert "decision_score_total" in m
        assert "edge_status_category" in m
        assert "market_suitable" in m


# ---------------------------------------------------------------------------
# 2. Decision Score
# ---------------------------------------------------------------------------

class TestDecisionScoreCalculation:
    @pytest.mark.asyncio
    async def test_decision_score_deterministic(self, monkeypatch):
        totals = []
        for _ in range(2):
            engine = TradingEngine()
            _, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
            prop = next(p for p in captured if p.direction == Direction.BUY)
            totals.append(prop.decision_score.total)
        assert totals[0] == totals[1]

    @pytest.mark.asyncio
    async def test_no_single_factor_reaches_the_threshold(self, monkeypatch):
        """Structural single-factor insufficiency: no individual Evidence
        Item's maximum contribution (its weight) reaches the entry threshold,
        so a trade always requires multiple factors to agree."""
        engine = TradingEngine()
        _, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        threshold = prop.decision_score.threshold
        assert all(c.weight < threshold for c in prop.decision_score.contributions)

    @pytest.mark.asyncio
    async def test_unreachable_threshold_blocks_entry(self, monkeypatch):
        """The Decision Score genuinely gates entry (not just the
        preconditions): a threshold above the maximum achievable score refuses
        the trade even on a fully confirmed reversal."""
        engine = TradingEngine()
        bot = _bot()
        buy, captured = await _drive_to_buy(
            engine, bot, monkeypatch, params={"decision_score_threshold": 101.0},
        )
        assert buy is None
        assert any(
            p.direction == Direction.NO_TRADE and not p.decision_score.approved
            for p in captured
        )


# ---------------------------------------------------------------------------
# 3. Market suitability (Pillar 2)
# ---------------------------------------------------------------------------

class TestMarketSuitability:
    @pytest.mark.asyncio
    async def test_unsuitable_regime_blocks_entry_despite_confirmed_reversal(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        # A declining market reads trend_down; declaring only trend_up allowed
        # makes it unsuitable even though the reversal itself is confirmed.
        buy, captured = await _drive_to_buy(
            engine, bot, monkeypatch, params={"allowed_regimes": ["trend_up"]},
        )
        assert buy is None
        blocked = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert blocked and blocked[-1].market_suitability.is_suitable is False

    @pytest.mark.asyncio
    async def test_suitable_regime_allows_entry(self, monkeypatch):
        engine = TradingEngine()
        buy, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
        assert buy is not None
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert prop.market_suitability.is_suitable is True
        # Documented consequence: the price-only detector emits no
        # volatility_direction, so "volatility_expanding" never appears.
        assert "volatility_expanding" not in prop.market_suitability.regime_tags


# ---------------------------------------------------------------------------
# 4. Strategy Edge Management (Pillar 7)
# ---------------------------------------------------------------------------

def _seed_edge(engine, bot, *, pnl, win, n=25):
    mgr = StrategyEdgeManager()
    for _ in range(n):
        mgr.record_trade_outcome(bot.id, "dip_recovery", pnl=pnl, win=win)
    engine._dip_recovery_edge_manager = mgr
    return mgr


class TestStrategyEdgeManagement:
    @pytest.mark.asyncio
    async def test_category_c_blocks_new_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _seed_edge(engine, bot, pnl=-50.0, win=False)
        buy, captured = await _drive_to_buy(engine, bot, monkeypatch)
        assert buy is None
        blocked = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert blocked and blocked[-1].edge_status.category == EdgeCategory.C

    @pytest.mark.asyncio
    async def test_healthy_history_allows_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _seed_edge(engine, bot, pnl=+50.0, win=True)
        buy, _ = await _drive_to_buy(engine, bot, monkeypatch)
        assert buy is not None

    @pytest.mark.asyncio
    async def test_edge_management_never_force_closes_open_position(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _seed_edge(engine, bot, pnl=-50.0, win=False)
        engine._price_histories = {bot.id: [90.0] * 20}
        engine._dip_recovery_states = {bot.id: _long_open_state(entry=90.0, atr=1.0)}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=1.0)])
        # Price between trailing stop and take-profit -> no exit trigger.
        sig = await engine._strategy_dip_recovery(bot, 90.2, {}, AsyncMock())
        assert sig.action == "hold"  # NOT force-closed despite Category C


# ---------------------------------------------------------------------------
# 5. StrategyProposal generation (Pillar 10)
# ---------------------------------------------------------------------------

class TestStrategyProposalGeneration:
    @pytest.mark.asyncio
    async def test_buy_proposal_is_well_formed(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
        buy = next(p for p in captured if p.direction == Direction.BUY)
        assert isinstance(buy, StrategyProposal)
        assert buy.execution_intent == ExecutionIntent.OPEN_POSITION
        assert buy.suggested_position_size and buy.suggested_position_size > 0
        assert "decision_score_size_multiplier" in buy.adaptive_parameters_used
        assert len(buy.assumptions) >= 1
        assert buy.expected_edge_estimate is None

    @pytest.mark.asyncio
    async def test_all_proposals_have_valid_intent_pairings(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
        for p in captured:
            assert (p.direction, p.execution_intent) in {
                (Direction.BUY, ExecutionIntent.OPEN_POSITION),
                (Direction.SELL, ExecutionIntent.CLOSE_POSITION),
                (Direction.HOLD, ExecutionIntent.HOLD_POSITION),
                (Direction.NO_TRADE, ExecutionIntent.NO_ACTION),
            }


# ---------------------------------------------------------------------------
# 6. Standalone Adapter compatibility
# ---------------------------------------------------------------------------

class TestStandaloneAdapterCompatibility:
    @pytest.mark.asyncio
    async def test_buy_signal_matches_adapter_translation(self, monkeypatch):
        engine = TradingEngine()
        buy, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert isinstance(buy, TradeSignal)
        assert buy.amount == prop.suggested_position_size
        assert buy.score == prop.decision_score.total
        assert buy.expected_move_pct is not None
        assert buy.expected_risk_pct is not None

    @pytest.mark.asyncio
    async def test_expired_proposal_would_be_discarded(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_to_buy(engine, _bot(), monkeypatch)
        buy = next(p for p in captured if p.direction == Direction.BUY)
        assert buy.is_expired(buy.validity.valid_until) is True
        assert buy.is_expired(buy.validity.generated_at) is False


# ---------------------------------------------------------------------------
# 7. Exit paths + Pillar 8
# ---------------------------------------------------------------------------

class TestExitPaths:
    @pytest.mark.asyncio
    async def test_take_profit_exit_produces_close_sell(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        engine._price_histories = {bot.id: [90.0] * 20}
        engine._dip_recovery_states = {bot.id: _long_open_state(entry=90.0, atr=1.0)}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=1.0)])
        captured = _capture_proposals(monkeypatch)
        # Price at/above take_profit (90 + 3*1 = 93).
        sig = await engine._strategy_dip_recovery(bot, 93.5, {}, AsyncMock())
        assert sig.action == "sell"
        prop = captured[-1]
        assert prop.direction == Direction.SELL
        assert prop.execution_intent == ExecutionIntent.CLOSE_POSITION

    @pytest.mark.asyncio
    async def test_emergency_stop_check_is_surfaced(self, monkeypatch):
        """Pillar 8 gap the audit found: the emergency stop is now a visible
        diagnostic check, not just a silent trigger."""
        engine = TradingEngine()
        bot = _bot()
        engine._price_histories = {bot.id: [90.0] * 20}
        engine._dip_recovery_states = {bot.id: _long_open_state(entry=90.0, atr=1.0)}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=1.0)])
        captured = _capture_proposals(monkeypatch)
        # Hold (price between stops) so the HOLD proposal carries the checks.
        await engine._strategy_dip_recovery(bot, 90.2, {}, AsyncMock())
        exp = captured[-1].explanation
        assert any(c["name"] == "Emergency stop hit" for c in exp["checks"])
