"""Phase 5 migration tests: adaptive_grid on the Strategy Decision Framework.

Covers the eight areas the sibling phases' suites cover, adapted to the grid's
virtual-inventory model (not a single position):

  1. Evidence generation (Pillar 3)          5. StrategyProposal generation (Pillar 10)
  2. Decision Score (Pillar 3)                6. Standalone Adapter compatibility
  3. Market suitability (Pillar 2)            7. Kill-switch diagnostics (Pillar 8)
  4. Strategy Edge Management (Pillar 7)      8. Regressions (mechanical grid preserved)

The grid mutates virtual state as part of deciding, so several tests also
assert virtual-wallet invariants (a blocked crossing must leave the level
unfilled and the wallet untouched — no buy/sell inventory desync).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.trading_engine import TradingEngine, TradeSignal
from app.services.strategy_framework.edge_management import EdgeCategory, StrategyEdgeManager
from app.services.strategy_framework.proposal import Direction, ExecutionIntent, StrategyProposal
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter


CENTER = 64_000.0
GRID_COUNT = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bot(bot_id: int = 70, balance: float = 10_000.0) -> MagicMock:
    bot = MagicMock()
    bot.id = bot_id
    bot.strategy = "adaptive_grid"
    bot.trading_pair = "BTC/USDT"
    bot.current_balance = balance
    bot.budget = balance
    bot.exchange_fee = 0.1
    bot.started_at = datetime.utcnow() - timedelta(hours=2)
    return bot


def _session() -> AsyncMock:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))
    return session


def _bars(count: int, bar_range: float, center: float = CENTER) -> list:
    ts = datetime.utcnow() - timedelta(minutes=count + 5)
    return [
        {
            "open": center, "high": center + bar_range / 2,
            "low": center - bar_range / 2, "close": center,
            "start_ts": ts + timedelta(minutes=i),
        }
        for i in range(count)
    ]


def _state(
    bot_id: int, *, bars: list, center: float = CENTER,
    virtual_crypto: float = 0.0, virtual_cash: float = 10_000.0, **overrides,
) -> dict:
    s = {
        "initialized": True,
        "center_price": center,
        "initial_capital": virtual_cash,
        "virtual_cash": virtual_cash,
        "virtual_crypto": virtual_crypto,
        "grid_levels": {},
        "last_bar_close_time": None,
        "current_bar": None,
        "completed_bars": bars,
        "last_order_bar": None,
        "peak_portfolio_value": virtual_cash,
        "last_recenter_time": datetime.utcnow() - timedelta(hours=4),
        "lifetime_return_pct": 0.0,
        "lifetime_max_drawdown_pct": 0.0,
        "last_kill_switch_time": None,
        "kill_switch_count": 0,
        "last_kill_reason": None,
        "atr_at_recenter": None,
        "atr_spacing": None,
        "current_atr": None,
        "current_grid_range": None,
        "current_grid_spacing": None,
        "total_trades": 0,
    }
    s.update(overrides)
    return {bot_id: s}


def _capture(monkeypatch) -> list:
    captured: list = []
    real = StandaloneAdapter.to_trade_signal

    def spy(proposal, **kwargs):
        captured.append(proposal)
        return real(proposal, **kwargs)

    monkeypatch.setattr(StandaloneAdapter, "to_trade_signal", staticmethod(spy))
    return captured


async def _drive(engine, bot, open_price, trigger_price, params=None):
    """Open a bar at open_price, then complete it at trigger_price."""
    session = _session()
    engine._get_bot_positions = AsyncMock(return_value=[])
    await engine._strategy_grid(bot, open_price, params or {}, session)
    st = engine._grid_states[bot.id]
    if st.get("current_bar"):
        st["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
    return await engine._strategy_grid(bot, trigger_price, params or {}, session)


async def _drive_buy(engine, bot, monkeypatch, *, params=None, bar_range=200.0):
    """Cross the L1 buy level (0.5% below center clears the min-spacing floor)."""
    engine._grid_states = _state(bot.id, bars=_bars(14, bar_range))
    captured = _capture(monkeypatch)
    trigger = CENTER * 0.995
    sig = await _drive(engine, bot, CENTER, trigger, params)
    return sig, captured


async def _drive_sell(engine, bot, monkeypatch, *, params=None, virtual_crypto=1.0,
                      virtual_cash=5_000.0, bar_range=200.0):
    engine._grid_states = _state(
        bot.id, bars=_bars(14, bar_range),
        virtual_crypto=virtual_crypto, virtual_cash=virtual_cash,
    )
    captured = _capture(monkeypatch)
    trigger = CENTER * 1.005
    sig = await _drive(engine, bot, CENTER, trigger, params)
    return sig, captured


# ---------------------------------------------------------------------------
# 1. Evidence generation
# ---------------------------------------------------------------------------

class TestEvidenceGeneration:
    @pytest.mark.asyncio
    async def test_fill_names_all_three_evidence_factors(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_buy(engine, _bot(), monkeypatch)
        assert sig.action == "buy"
        prop = next(p for p in captured if p.direction == Direction.BUY)
        names = [c.name for c in prop.decision_score.contributions]
        assert names == [
            "Range-bound conviction", "Post-fee spacing margin", "Volatility adequacy",
        ]

    @pytest.mark.asyncio
    async def test_no_evidence_factor_references_depth(self, monkeypatch):
        # Depth is the grid's intended convex payoff (rewarded by the depth
        # multiplier), NOT a quality penalty — no Evidence Item may cite it.
        engine = TradingEngine()
        _, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        for c in prop.decision_score.contributions:
            assert "depth" not in c.name.lower()
            assert "depth" not in c.reason.lower()

    @pytest.mark.asyncio
    async def test_explanation_contains_decision_metrics(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        m = prop.explanation["metrics"]
        assert "decision_score_total" in m
        assert "decision_score_threshold" in m
        assert "decision_score_size_multiplier" in m
        assert "edge_status_category" in m


# ---------------------------------------------------------------------------
# 2. Decision Score
# ---------------------------------------------------------------------------

class TestDecisionScore:
    @pytest.mark.asyncio
    async def test_decision_score_deterministic(self, monkeypatch):
        totals = []
        for _ in range(2):
            engine = TradingEngine()
            _, captured = await _drive_buy(engine, _bot(), monkeypatch)
            prop = next(p for p in captured if p.direction == Direction.BUY)
            totals.append(prop.decision_score.total)
        assert totals[0] == totals[1]

    @pytest.mark.asyncio
    async def test_no_single_factor_reaches_a_full_threshold(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        # No one Evidence Item's weight is the whole 100 points: the score is
        # multi-factor by construction.
        assert all(c.weight < 100.0 for c in prop.decision_score.contributions)
        assert sum(c.weight for c in prop.decision_score.contributions) == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_default_threshold_is_zero_and_does_not_suppress_a_normal_fill(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert prop.decision_score.threshold == 0.0
        assert prop.decision_score.approved
        assert sig.action == "buy"

    @pytest.mark.asyncio
    async def test_unreachable_threshold_blocks_fill_without_mutating_state(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        sig, captured = await _drive_buy(
            engine, bot, monkeypatch, params={"decision_score_threshold": 101.0},
        )
        assert sig.action == "hold"
        assert any(
            p.direction == Direction.NO_TRADE and not p.decision_score.approved
            for p in captured
        )
        # The blocked crossing must leave the wallet untouched (no desync).
        st = engine._grid_states[bot.id]
        assert st["virtual_cash"] == 10_000.0
        assert st["virtual_crypto"] == 0.0
        assert not any(lv["filled"] for lv in st["grid_levels"].values())


# ---------------------------------------------------------------------------
# 3. Market suitability (Pillar 2)
# ---------------------------------------------------------------------------

class TestMarketSuitability:
    @pytest.mark.asyncio
    async def test_unsuitable_regime_blocks_fill(self, monkeypatch):
        engine = TradingEngine()
        # A flat oscillating range reads trend_flat; allowing only trend_up
        # makes it unsuitable, so even a crossed level does not fill.
        sig, captured = await _drive_buy(
            engine, _bot(), monkeypatch, params={"allowed_regimes": ["trend_up"]},
        )
        assert sig.action == "hold"
        blocked = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert blocked
        assert not blocked[-1].market_suitability.is_suitable

    @pytest.mark.asyncio
    async def test_suitable_regime_allows_fill(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_buy(engine, _bot(), monkeypatch)
        assert sig.action == "buy"
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert prop.market_suitability.is_suitable

    @pytest.mark.asyncio
    async def test_regime_filter_disabled_maps_to_all(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_buy(
            engine, _bot(), monkeypatch, params={"regime_filter_enabled": False},
        )
        assert sig.action == "buy"
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert prop.market_suitability.is_suitable
        assert prop.market_suitability.allowed_regimes == ["all"]


# ---------------------------------------------------------------------------
# 4. Strategy Edge Management (Pillar 7)
# ---------------------------------------------------------------------------

class TestEdgeManagement:
    @pytest.mark.asyncio
    async def test_category_c_blocks_new_fill_without_liquidating_inventory(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        # Seed a degraded edge with NO parameter/regime explanation -> Category C.
        mgr = StrategyEdgeManager()
        for _ in range(mgr.outcome_window):
            mgr.record_trade_outcome(bot.id, "adaptive_grid", pnl=-1.0, win=False)
        engine._grid_edge_manager = mgr
        sig, captured = await _drive_sell(
            engine, bot, monkeypatch, virtual_crypto=5.0,
        )
        assert sig.action == "hold"
        blocked = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert blocked
        assert blocked[-1].edge_status.category == EdgeCategory.C
        # Edge Management NEVER force-closes: the virtual inventory is intact.
        st = engine._grid_states[bot.id]
        assert st["virtual_crypto"] == 5.0

    @pytest.mark.asyncio
    async def test_range_escape_kill_classifies_category_b(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        # Pre-seed enough losses that the kill's own re-evaluation can classify;
        # a range-escape kill supplies parameter-mismatch evidence -> Category B.
        mgr = StrategyEdgeManager()
        for _ in range(mgr.min_sample_size):
            mgr.record_trade_outcome(bot.id, "adaptive_grid", pnl=-1.0, win=False)
        engine._grid_edge_manager = mgr
        # Tight bars -> tiny ATR -> a modest move exceeds 3x ATR (range escape)
        # while staying inside the min-spacing-floored soft-recenter band, so
        # the range-escape kill fires (not a soft recenter).
        engine._grid_states = _state(bot.id, bars=_bars(14, 4.0))
        captured = _capture(monkeypatch)
        sig = await _drive(engine, bot, CENTER, CENTER * 1.002)
        st = engine._grid_states[bot.id]
        assert st["last_kill_reason"] == "range_escape"
        kill_props = [p for p in captured if p.direction == Direction.NO_TRADE]
        assert kill_props
        assert kill_props[-1].edge_status.category == EdgeCategory.B

    @pytest.mark.asyncio
    async def test_sell_fill_records_a_win_outcome(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        sig, _ = await _drive_sell(engine, bot, monkeypatch)
        assert sig.action == "sell"
        outcomes = engine._grid_edge_manager._outcomes[(bot.id, "adaptive_grid")]
        assert any(o.win for o in outcomes)


# ---------------------------------------------------------------------------
# 5. StrategyProposal generation (Pillar 10)
# ---------------------------------------------------------------------------

class TestProposalGeneration:
    @pytest.mark.asyncio
    async def test_buy_fill_is_add_to_position(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        # Grid entries are incremental inventory adds, not fresh OPEN_POSITIONs.
        assert prop.execution_intent == ExecutionIntent.ADD_TO_POSITION
        assert prop.suggested_position_size == sig.amount
        assert prop.validity.valid_until > prop.generated_at

    @pytest.mark.asyncio
    async def test_partial_sell_is_reduce_position(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_sell(engine, _bot(), monkeypatch, virtual_crypto=100.0)
        prop = next(p for p in captured if p.direction == Direction.SELL)
        assert prop.execution_intent == ExecutionIntent.REDUCE_POSITION

    @pytest.mark.asyncio
    async def test_emptying_sell_is_close_position(self, monkeypatch):
        # Two-phase: discover the order size with ample inventory (REDUCE), then
        # reconstruct with exactly that much crypto so the sell empties it -> CLOSE.
        trigger = CENTER * 1.005
        engine = TradingEngine()
        bot = _bot()
        sig1, cap1 = await _drive_sell(engine, bot, monkeypatch, virtual_crypto=100.0)
        prop1 = next(p for p in cap1 if p.direction == Direction.SELL)
        assert prop1.execution_intent == ExecutionIntent.REDUCE_POSITION
        exact_crypto = sig1.amount / trigger

        engine2 = TradingEngine()
        bot2 = _bot(bot_id=71)
        sig2, cap2 = await _drive_sell(
            engine2, bot2, monkeypatch, virtual_crypto=exact_crypto,
        )
        prop2 = next(p for p in cap2 if p.direction == Direction.SELL)
        assert prop2.execution_intent == ExecutionIntent.CLOSE_POSITION

    @pytest.mark.asyncio
    async def test_no_level_crossed_is_hold_position(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        engine._grid_states = _state(bot.id, bars=_bars(14, 200.0))
        captured = _capture(monkeypatch)
        # Stay at center: no unfilled level is crossed.
        sig = await _drive(engine, bot, CENTER, CENTER)
        assert sig.action == "hold"
        holds = [p for p in captured
                 if p.direction == Direction.HOLD
                 and p.execution_intent == ExecutionIntent.HOLD_POSITION]
        assert holds

    @pytest.mark.asyncio
    async def test_all_grid_proposals_have_valid_intent_pairings(self, monkeypatch):
        # The frozen StrategyProposal rejects invalid (direction, intent) pairs
        # at construction, so merely constructing every branch's proposal proves
        # the pairing. Exercise buy, sell, hold and a no-trade branch.
        engine = TradingEngine()
        seen = []
        _, c1 = await _drive_buy(engine, _bot(bot_id=80), monkeypatch)
        seen += c1
        _, c2 = await _drive_sell(engine, _bot(bot_id=81), monkeypatch)
        seen += c2
        _, c3 = await _drive_buy(
            engine, _bot(bot_id=82), monkeypatch,
            params={"decision_score_threshold": 101.0},
        )
        seen += c3
        assert seen
        for p in seen:
            assert isinstance(p, StrategyProposal)  # constructed => pairing valid
        pairs = {(p.direction, p.execution_intent) for p in seen}
        assert (Direction.BUY, ExecutionIntent.ADD_TO_POSITION) in pairs
        assert (Direction.NO_TRADE, ExecutionIntent.NO_ACTION) in pairs


# ---------------------------------------------------------------------------
# 6. Standalone Adapter compatibility
# ---------------------------------------------------------------------------

class TestStandaloneAdapterCompatibility:
    @pytest.mark.asyncio
    async def test_buy_signal_matches_adapter_translation(self, monkeypatch):
        engine = TradingEngine()
        sig, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        translated = StandaloneAdapter.to_trade_signal(
            prop, expected_move_pct=sig.expected_move_pct,
        )
        assert translated.action == sig.action == "buy"
        assert translated.amount == sig.amount

    @pytest.mark.asyncio
    async def test_buy_carries_positive_expected_move(self, monkeypatch):
        engine = TradingEngine()
        sig, _ = await _drive_buy(engine, _bot(), monkeypatch)
        assert isinstance(sig.expected_move_pct, float)
        assert sig.expected_move_pct > 0.0

    @pytest.mark.asyncio
    async def test_hold_position_proposal_produces_no_order(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        engine._grid_states = _state(bot.id, bars=_bars(14, 200.0))
        _capture(monkeypatch)
        sig = await _drive(engine, bot, CENTER, CENTER)
        # HOLD_POSITION is a no-order intent: adapter returns None, strategy
        # falls back to a hold TradeSignal.
        assert sig.action == "hold"

    @pytest.mark.asyncio
    async def test_expired_proposal_is_discarded(self, monkeypatch):
        engine = TradingEngine()
        _, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        assert prop.is_expired(prop.validity.valid_until + timedelta(seconds=1))
        assert not prop.is_expired(prop.generated_at)


# ---------------------------------------------------------------------------
# 7. Kill-switch diagnostics (Pillar 8 — the audit's single biggest gap)
# ---------------------------------------------------------------------------

class TestKillSwitchDiagnostics:
    @pytest.mark.asyncio
    async def test_range_escape_kill_is_explained(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        engine._grid_states = _state(bot.id, bars=_bars(14, 4.0))
        engine._get_bot_positions = AsyncMock(return_value=[])
        session = _session()
        await engine._strategy_grid(bot, CENTER, {}, session)
        st = engine._grid_states[bot.id]
        st["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
        await engine._strategy_grid(bot, CENTER * 1.002, {}, session)
        exp = engine._explain(bot.id).exp
        assert exp.state == "KILL_SWITCH_RANGE_ESCAPE"
        assert any("kill switch" in c.name.lower() for c in exp.checks)
        assert st["last_kill_reason"] == "range_escape"

    @pytest.mark.asyncio
    async def test_drawdown_kill_is_explained(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        # A big virtual-crypto position plus a wide grid (so no range-escape
        # fires first) and a price crash produces a >15% portfolio drawdown.
        bars = _bars(14, 2_000.0)  # wide ATR -> wide grid, no range-escape
        engine._grid_states = _state(
            bot.id, bars=bars, virtual_crypto=1.0, virtual_cash=100.0,
            initial_capital=CENTER + 100.0, peak_portfolio_value=CENTER + 100.0,
        )
        engine._get_bot_positions = AsyncMock(return_value=[])
        session = _session()
        crashed = CENTER * 0.80  # 20% down on a fully-invested virtual wallet
        await engine._strategy_grid(bot, crashed, {}, session)
        st = engine._grid_states[bot.id]
        st["current_bar"]["start_ts"] = datetime.utcnow() - timedelta(seconds=65)
        await engine._strategy_grid(bot, crashed, {}, session)
        st = engine._grid_states[bot.id]
        exp = engine._explain(bot.id).exp
        assert st["last_kill_reason"] == "drawdown"
        assert exp.state == "KILL_SWITCH_DRAWDOWN"
        assert any("drawdown" in c.name.lower() for c in exp.checks)


# ---------------------------------------------------------------------------
# 8. Regressions — the mechanical grid is preserved by default
# ---------------------------------------------------------------------------

class TestRegressions:
    @pytest.mark.asyncio
    async def test_buy_still_fires_on_a_downward_crossing(self, monkeypatch):
        engine = TradingEngine()
        sig, _ = await _drive_buy(engine, _bot(), monkeypatch)
        assert sig.action == "buy"
        assert sig.amount > 0

    @pytest.mark.asyncio
    async def test_sell_still_fires_on_an_upward_crossing(self, monkeypatch):
        engine = TradingEngine()
        sig, _ = await _drive_sell(engine, _bot(), monkeypatch)
        assert sig.action == "sell"

    @pytest.mark.asyncio
    async def test_score_sizing_preserves_depth_multiplier_relationship(self, monkeypatch):
        # Sizing = base * depth_multiplier ** (depth-1) * score_multiplier. The
        # adaptive_parameters_used record must carry both multipliers so the
        # depth (convex) component is preserved distinct from the score component.
        engine = TradingEngine()
        _, captured = await _drive_buy(engine, _bot(), monkeypatch)
        prop = next(p for p in captured if p.direction == Direction.BUY)
        used = prop.adaptive_parameters_used
        assert "depth_size_multiplier" in used
        assert "decision_score_size_multiplier" in used
        assert 0.5 <= used["decision_score_size_multiplier"] <= 1.5

    @pytest.mark.asyncio
    async def test_diagnostic_state_still_written_each_bar(self, monkeypatch):
        engine = TradingEngine()
        bot = _bot()
        await _drive_buy(engine, bot, monkeypatch)
        st = engine._grid_states[bot.id]
        assert st["current_atr"] is not None
        assert st["current_grid_range"] is not None
        assert st["current_grid_spacing"] is not None
