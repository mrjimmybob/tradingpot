"""Tests for mean_reversion's migration to the Strategy Decision Framework
(add-strategy-decision-framework, Phase 4).

mean_reversion already had the reference Pillar 2 (regime gate) and Pillar 6
(four-way exits); Phase 4 adds the Evidence-Based Decision Score (Pillar 3),
Decision-Score-weighted sizing (Pillar 5), Strategy Edge Management (Pillar 7),
the missing Pillar 8 checks, and the StrategyProposal/Standalone-Adapter
interface (Pillar 10). The pre-existing regime/exit behaviour stays covered by
the other suites; this file covers the framework layer.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.trading_engine import TradingEngine, TradeSignal
from app.services.strategy_framework.edge_management import EdgeCategory, StrategyEdgeManager
from app.services.strategy_framework.proposal import Direction, ExecutionIntent, StrategyProposal
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bot(balance: float = 100_000.0, bot_id: int = 1):
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "mean_reversion"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    bot.exchange_fee = 0.1
    bot.strategy_params = {}
    return bot


def _oversold_bars(base=64_000.0, amp=400.0, dip=62_800.0, n=19):
    """A flat, symmetric oscillating range (-> trend_flat regime) with a final
    completed bar dipping well below the ~1.8σ lower Bollinger band."""
    bars = []
    for i in range(n):
        close = base + amp if i % 2 == 0 else base - amp
        bars.append({
            "open": close, "high": close + 5, "low": close - 5, "close": close,
            "start_ts": datetime.utcnow() - timedelta(minutes=25 - i),
        })
    bars.append({
        "open": dip, "high": dip + 5, "low": dip - 5, "close": dip,
        "start_ts": datetime.utcnow() - timedelta(minutes=1),
    })
    return bars


def _mr_state(bars, **overrides):
    state = {
        "bars": bars, "current_bar": None,
        "entry_price": None, "entry_atr": None, "target_price": None,
        "hard_stop": None, "bars_since_entry": 0, "last_exit_time": None,
    }
    state.update(overrides)
    return state


def _capture_proposals(monkeypatch) -> list:
    captured: list = []
    real_fn = StandaloneAdapter.to_trade_signal

    def spy(proposal, **kwargs):
        captured.append(proposal)
        return real_fn(proposal, **kwargs)

    monkeypatch.setattr(StandaloneAdapter, "to_trade_signal", staticmethod(spy))
    return captured


async def _drive_entry(engine, bot, monkeypatch, *, params=None, dip=62_800.0):
    engine._mean_reversion_states = {bot.id: _mr_state(_oversold_bars(dip=dip))}
    engine._get_bot_positions = AsyncMock(return_value=[])
    captured = _capture_proposals(monkeypatch)
    sig = await engine._strategy_mean_reversion(bot, dip, params or {}, AsyncMock())
    return sig, captured


# ---------------------------------------------------------------------------
# 1. Evidence generation
# ---------------------------------------------------------------------------

class TestEvidenceGeneration:
    @pytest.mark.asyncio
    async def test_entry_names_all_three_evidence_factors(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_entry(engine, _bot(), monkeypatch)
        assert sig.action == "buy"
        prop = next(p for p in captured if p.direction == Direction.BUY)
        names = [c.name for c in prop.decision_score.contributions]
        assert names == [
            "Reversion target distance", "Oversold penetration", "Band width adequacy",
        ]

    @pytest.mark.asyncio
    async def test_explanation_contains_decision_metrics(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_entry(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        m = prop.explanation["metrics"]
        assert "decision_score_total" in m
        assert "edge_status_category" in m


# ---------------------------------------------------------------------------
# 2. Decision Score
# ---------------------------------------------------------------------------

class TestDecisionScoreCalculation:
    @pytest.mark.asyncio
    async def test_decision_score_deterministic(self, monkeypatch):
        totals = []
        for _ in range(2):
            engine = TradingEngine()
            _, captured = await _drive_entry(engine, _bot(), monkeypatch)
            prop = next(p for p in captured if p.direction == Direction.BUY)
            totals.append(prop.decision_score.total)
        assert totals[0] == totals[1]

    @pytest.mark.asyncio
    async def test_no_single_factor_reaches_the_threshold(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_entry(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        threshold = prop.decision_score.threshold
        assert all(c.weight < threshold for c in prop.decision_score.contributions)

    @pytest.mark.asyncio
    async def test_unreachable_threshold_blocks_entry(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_entry(
            engine, _bot(), monkeypatch, params={"decision_score_threshold": 101.0},
        )
        assert sig.action == "hold"
        assert any(
            p.direction == Direction.NO_TRADE and not p.decision_score.approved
            for p in captured
        )


# ---------------------------------------------------------------------------
# 3. Market suitability (Pillar 2)
# ---------------------------------------------------------------------------

class TestMarketSuitability:
    @pytest.mark.asyncio
    async def test_unsuitable_regime_blocks_entry(self, monkeypatch):
        engine = TradingEngine()
        # A flat range reads trend_flat; declaring only trend_up allowed makes
        # it unsuitable even at a textbook oversold band touch.
        sig, captured = await _drive_entry(
            engine, _bot(), monkeypatch, params={"allowed_regimes": ["trend_up"]},
        )
        assert sig.action == "hold"
        blocked = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert blocked and blocked[-1].market_suitability.is_suitable is False

    @pytest.mark.asyncio
    async def test_suitable_regime_allows_entry(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_entry(engine, _bot(), monkeypatch)
        assert sig.action == "buy"
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert prop.market_suitability.is_suitable is True


# ---------------------------------------------------------------------------
# 4. Strategy Edge Management (Pillar 7)
# ---------------------------------------------------------------------------

def _seed_edge(engine, bot, *, pnl, win, n=25):
    mgr = StrategyEdgeManager()
    for _ in range(n):
        mgr.record_trade_outcome(bot.id, "mean_reversion", pnl=pnl, win=win)
    engine._mean_reversion_edge_manager = mgr
    return mgr


class TestStrategyEdgeManagement:
    @pytest.mark.asyncio
    async def test_category_c_blocks_new_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _seed_edge(engine, bot, pnl=-50.0, win=False)
        sig, captured = await _drive_entry(engine, bot, monkeypatch)
        assert sig.action == "hold"
        blocked = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert blocked and blocked[-1].edge_status.category == EdgeCategory.C

    @pytest.mark.asyncio
    async def test_healthy_history_allows_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _seed_edge(engine, bot, pnl=+50.0, win=True)
        sig, _ = await _drive_entry(engine, bot, monkeypatch)
        assert sig.action == "buy"

    @pytest.mark.asyncio
    async def test_edge_management_never_force_closes_open_position(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        _seed_edge(engine, bot, pnl=-50.0, win=False)
        bars = _oversold_bars(dip=63_950.0)  # near the mean: no exit trigger
        engine._mean_reversion_states = {bot.id: _mr_state(
            bars, entry_price=63_900.0, entry_atr=100.0, target_price=64_500.0,
            hard_stop=63_500.0, bars_since_entry=1,
        )}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=1.0)])
        sig = await engine._strategy_mean_reversion(bot, 63_950.0, {}, AsyncMock())
        assert sig.action == "hold"  # NOT force-closed despite Category C


# ---------------------------------------------------------------------------
# 5. StrategyProposal generation (Pillar 10)
# ---------------------------------------------------------------------------

class TestStrategyProposalGeneration:
    @pytest.mark.asyncio
    async def test_buy_proposal_is_well_formed(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_entry(engine, _bot(), monkeypatch)
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
        _, captured = await _drive_entry(engine, _bot(), monkeypatch)
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
        sig, captured = await _drive_entry(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert isinstance(sig, TradeSignal)
        assert sig.amount == prop.suggested_position_size
        assert sig.score == prop.decision_score.total
        assert sig.expected_move_pct is not None
        assert sig.expected_risk_pct is not None

    @pytest.mark.asyncio
    async def test_expired_proposal_would_be_discarded(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_entry(engine, _bot(), monkeypatch)
        buy = next(p for p in captured if p.direction == Direction.BUY)
        assert buy.is_expired(buy.validity.valid_until) is True
        assert buy.is_expired(buy.validity.generated_at) is False


# ---------------------------------------------------------------------------
# 7. Exit paths (Pillar 6 preserved, now proposals)
# ---------------------------------------------------------------------------

class TestExitPaths:
    @pytest.mark.asyncio
    async def test_target_reached_produces_close_sell(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        # Flat bars so the last close is at/above the locked target.
        bars = _oversold_bars(dip=64_000.0)
        engine._mean_reversion_states = {bot.id: _mr_state(
            bars, entry_price=63_500.0, entry_atr=100.0, target_price=63_900.0,
            hard_stop=63_100.0, bars_since_entry=1,
        )}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=1.0)])
        captured = _capture_proposals(monkeypatch)
        sig = await engine._strategy_mean_reversion(bot, 64_000.0, {}, AsyncMock())
        assert sig.action == "sell"
        prop = captured[-1]
        assert prop.direction == Direction.SELL
        assert prop.execution_intent == ExecutionIntent.CLOSE_POSITION

    @pytest.mark.asyncio
    async def test_hard_stop_produces_close_sell(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        bars = _oversold_bars(dip=63_000.0)
        engine._mean_reversion_states = {bot.id: _mr_state(
            bars, entry_price=63_500.0, entry_atr=100.0, target_price=64_500.0,
            hard_stop=63_100.0, bars_since_entry=1,
        )}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=1.0)])
        captured = _capture_proposals(monkeypatch)
        # current price below the hard stop -> stop exit.
        sig = await engine._strategy_mean_reversion(bot, 63_000.0, {}, AsyncMock())
        assert sig.action == "sell"
        assert captured[-1].execution_intent == ExecutionIntent.CLOSE_POSITION
