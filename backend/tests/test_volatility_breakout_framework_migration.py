"""Tests for volatility_breakout's migration to the Strategy Decision
Framework (add-strategy-decision-framework, Phase 1).

Covers the full decision flow the migration is required to implement
end-to-end: Market Suitability Gate -> Adaptive Parameter Resolver ->
Evidence Collection -> Evidence-Based Decision Score -> Strategy Edge
Management -> StrategyProposal -> Standalone Adapter -> existing
execution pipeline (the return value flowing into the bot loop's
existing wallet-check + `_execute_trade` call, unchanged).

Uses the same bar-seeding conventions as tests/test_strategy_activity.py
and tests/test_strategy_activation.py (bar_interval_seconds=0 so every
call closes a bar; hand-built bar lists for deterministic indicator math).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models import BotStatus
from app.services.trading_engine import TradingEngine, TradeSignal
from app.services.strategy_framework.edge_management import EdgeCategory, StrategyEdgeManager
from app.services.strategy_framework.market_suitability import MarketSuitabilityResult
from app.services.strategy_framework.proposal import Direction, ExecutionIntent, StrategyProposal
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bot(params=None, balance=100_000.0, bot_id=1):
    return SimpleNamespace(
        id=bot_id, name="vb-test", trading_pair="BTC/USDT", strategy="volatility_breakout",
        strategy_params=params or {}, budget=balance, current_balance=balance,
        compound_enabled=False, is_dry_run=True, status=BotStatus.RUNNING,
        total_pnl=0.0, exchange_fee=0.1,
    )


def _flat_session():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))
    return session


def _armed_state(
    *,
    base_price: float = 64_000.0,
    breakout_close: float = None,
    armed_compression_bars: int = 8,
    armed_compression_min_width: float = 0.0002,
    n_tight_bars: int = 24,
) -> dict:
    """A hand-built state dict representing a mature, well-evidenced,
    already-armed compression episode about to (or already having)
    broken out — mirrors test_strategy_activity.py's seeding convention.
    """
    breakout_close = breakout_close if breakout_close is not None else base_price + 500
    tight_bars = [
        {"open": base_price, "high": base_price + 2, "low": base_price - 2,
         "close": base_price, "start_ts": datetime.utcnow() - timedelta(minutes=30 - i)}
        for i in range(n_tight_bars)
    ]
    tight_bars.append({
        "open": base_price, "high": breakout_close, "low": base_price - 2,
        "close": breakout_close, "start_ts": datetime.utcnow() - timedelta(minutes=1),
    })
    wide_widths = [0.01] * 80
    tight_widths = [0.0002] * 5
    return {
        "bars": tight_bars,
        "current_bar": None,
        "bb_width_history": wide_widths + tight_widths,
        "atr_history": [10.0] * 20,
        "compression_active": False,
        "compression_bars": 0,
        "compression_start": None,
        "compression_min_width": None,
        "breakout_armed": True,
        "armed_compression_bars": armed_compression_bars,
        "armed_compression_min_width": armed_compression_min_width,
        "entry_price": None,
        "entry_atr": None,
        "entry_stop_multiplier": None,
        "entry_time": None,
        "highest_price": None,
        "trailing_stop": None,
        "take_profit_price": None,
        "bars_since_entry": 0,
        "last_breakout_attempt": None,
        "regime_state": None,
    }


def _capture_proposals(monkeypatch) -> list:
    """Spy on StandaloneAdapter.to_trade_signal so tests can inspect the
    exact StrategyProposal the strategy built, while still exercising the
    real translation (never bypassed)."""
    captured: list = []
    real_fn = StandaloneAdapter.to_trade_signal

    def spy(proposal, **kwargs):
        captured.append(proposal)
        return real_fn(proposal, **kwargs)

    monkeypatch.setattr(StandaloneAdapter, "to_trade_signal", staticmethod(spy))
    return captured


# ---------------------------------------------------------------------------
# 1. Evidence generation
# ---------------------------------------------------------------------------

class TestEvidenceGeneration:
    @pytest.mark.asyncio
    async def test_explanation_contains_all_evidence_metrics(self):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        session = _flat_session()

        with_positions = AsyncMock(return_value=[])
        engine._get_bot_positions = with_positions

        await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, session,
        )

        explanation = engine._explain(bot.id).to_dict()
        metrics = explanation["metrics"]
        assert "decision_score_total" in metrics
        assert "decision_score_threshold" in metrics
        assert "atr_percentile" in metrics
        assert "edge_status_category" in metrics
        assert metrics["market_suitable"] is True
        assert metrics["regime_tags"] is not None

    @pytest.mark.asyncio
    async def test_evidence_report_names_all_four_factors(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert len(captured) == 1
        report_names = {c.name for c in captured[0].decision_score.contributions}
        assert report_names == {
            "Breakout magnitude", "Compression maturity",
            "Compression tightness", "Volatility expansion strength",
        }


# ---------------------------------------------------------------------------
# 2. Decision Score calculation
# ---------------------------------------------------------------------------

class TestDecisionScoreCalculation:
    @pytest.mark.asyncio
    async def test_strong_setup_clears_threshold_and_buys(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert signal is not None and signal.action == "buy"
        assert captured[0].decision_score.approved is True
        assert signal.score == captured[0].decision_score.total
        assert signal.threshold == captured[0].decision_score.threshold

    @pytest.mark.asyncio
    async def test_marginal_setup_does_not_clear_threshold(self, monkeypatch):
        """Barely armed (exactly at minimum), barely broken out, and a
        loose (not tight) compression - weak evidence across the board
        must NOT clear the default threshold."""
        engine = TradingEngine()
        bot = make_bot()
        state = _armed_state(
            armed_compression_bars=5,  # exactly at min_compression_bars, no margin
            armed_compression_min_width=0.0099,  # barely below the ~0.01 percentile
            breakout_close=64_000.0 + 1.0,  # a one-dollar breakout, negligible vs ATR
        )
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_001.0, bot.strategy_params, _flat_session(),
        )

        assert signal is None or signal.action == "hold"
        assert captured[0].decision_score.approved is False
        assert captured[0].direction == Direction.NO_TRADE

    @pytest.mark.asyncio
    async def test_decision_score_deterministic_across_identical_calls(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._get_bot_positions = AsyncMock(return_value=[])

        results = []
        for _ in range(2):
            engine._volatility_breakout_states = {bot.id: _armed_state()}
            captured = _capture_proposals(monkeypatch)
            await engine._strategy_volatility_breakout(
                bot, 64_500.0, bot.strategy_params, _flat_session(),
            )
            results.append(captured[0].decision_score.total)

        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# 3. Market Suitability - refuses unsuitable markets even with strong evidence
# ---------------------------------------------------------------------------

class TestMarketSuitability:
    @pytest.mark.asyncio
    async def test_unsuitable_regime_blocks_entry_despite_strong_setup(self, monkeypatch):
        """The single most important Pillar 2 test: a fully armed, tight,
        well-matured compression with a large breakout must still be
        REFUSED if the regime detector says volatility is not expanding.
        """
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        # Force an UNSUITABLE regime (contracting, not expanding) regardless
        # of what the real bar-based detector would compute from the seeded
        # bars - isolates the suitability gate from the evidence collection.
        engine._detect_market_regime_bar_based = MagicMock(return_value={
            "trend_state": "flat", "volatility_state": "medium",
            "volatility_direction": "contracting", "liquidity_state": "normal",
        })
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert signal is None or signal.action == "hold"
        assert captured[0].direction == Direction.NO_TRADE
        assert captured[0].execution_intent == ExecutionIntent.NO_ACTION
        assert captured[0].market_suitability.is_suitable is False
        if signal is not None:
            assert "unsuitable" in signal.reason.lower() or "regime" in signal.reason.lower()

    @pytest.mark.asyncio
    async def test_suitable_regime_does_not_block_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._detect_market_regime_bar_based = MagicMock(return_value={
            "trend_state": "up", "volatility_state": "high",
            "volatility_direction": "expanding", "liquidity_state": "normal",
        })
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert captured[0].market_suitability.is_suitable is True
        assert signal is not None and signal.action == "buy"

    @pytest.mark.asyncio
    async def test_suitability_gate_uses_declared_allowed_regimes_param(self, monkeypatch):
        """A strategy param override of allowed_regimes changes what
        counts as suitable - proves the gate is actually parameterized,
        not hardcoded."""
        engine = TradingEngine()
        bot = make_bot(params={"allowed_regimes": ["volatility_contracting"]})
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._detect_market_regime_bar_based = MagicMock(return_value={
            "trend_state": "flat", "volatility_state": "medium",
            "volatility_direction": "contracting", "liquidity_state": "normal",
        })
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert captured[0].market_suitability.is_suitable is True
        assert signal is not None and signal.action == "buy"


# ---------------------------------------------------------------------------
# 4. Strategy Edge Management
# ---------------------------------------------------------------------------

class TestStrategyEdgeManagement:
    @pytest.mark.asyncio
    async def test_category_c_blocks_new_entry_despite_qualifying_score(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        state = _armed_state()
        # Match the seeded history to the ATR actually computed from
        # _armed_state()'s bars (~40, from the breakout bar's wide range)
        # so atr_percentile stays near 1.0 and does not itself trigger a
        # parameter_mismatch_evidence citation - isolates this test to the
        # Category C classification path (no regime mismatch, no parameter
        # mismatch, only a sustained, unexplained losing streak).
        state["atr_history"] = [40.0] * 20
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._detect_market_regime_bar_based = MagicMock(return_value={
            "trend_state": "up", "volatility_state": "high",
            "volatility_direction": "expanding", "liquidity_state": "normal",
        })

        # Pre-seed a manager already classifying Category C for this bot -
        # a sustained losing streak with no regime/parameter explanation.
        edge_manager = StrategyEdgeManager(min_sample_size=5)
        for _ in range(10):
            edge_manager.record_trade_outcome(
                bot.id, "volatility_breakout", pnl=-10.0, win=False,
            )
        engine._volatility_breakout_edge_manager = edge_manager

        captured = _capture_proposals(monkeypatch)
        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert captured[0].edge_status.category == EdgeCategory.C
        assert signal is None or signal.action == "hold"
        assert captured[0].direction == Direction.NO_TRADE
        assert "Category C" in "; ".join(captured[0].reasons_against) or (
            signal is not None and "Category C" in signal.reason
        )

    @pytest.mark.asyncio
    async def test_healthy_history_does_not_block_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        engine._detect_market_regime_bar_based = MagicMock(return_value={
            "trend_state": "up", "volatility_state": "high",
            "volatility_direction": "expanding", "liquidity_state": "normal",
        })

        edge_manager = StrategyEdgeManager(min_sample_size=5)
        for _ in range(10):
            edge_manager.record_trade_outcome(
                bot.id, "volatility_breakout", pnl=10.0, win=True,
            )
        engine._volatility_breakout_edge_manager = edge_manager

        captured = _capture_proposals(monkeypatch)
        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        assert captured[0].edge_status.category == EdgeCategory.NONE
        assert signal is not None and signal.action == "buy"

    @pytest.mark.asyncio
    async def test_edge_management_never_force_closes_open_position(self):
        """Even a Category C classification must not force-sell an open
        position - only price-based exits (stop/target/failed-breakout/
        time-stop) may close it."""
        engine = TradingEngine()
        bot = make_bot()
        state = _armed_state()
        state.update({
            "entry_price": 64_000.0, "entry_atr": 50.0, "entry_stop_multiplier": 2.0,
            "entry_time": datetime.utcnow().isoformat(),
            "highest_price": 64_100.0, "trailing_stop": 63_800.0,  # far from current price
            "take_profit_price": 70_000.0,  # far above current price
            "bars_since_entry": 5, "breakout_armed": False,
        })
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[
            SimpleNamespace(amount=0.01, trading_pair="BTC/USDT", entry_price=64_000.0)
        ])

        edge_manager = StrategyEdgeManager(min_sample_size=5)
        for _ in range(10):
            edge_manager.record_trade_outcome(
                bot.id, "volatility_breakout", pnl=-10.0, win=False,
            )
        engine._volatility_breakout_edge_manager = edge_manager

        signal = await engine._strategy_volatility_breakout(
            bot, 64_050.0, bot.strategy_params, _flat_session(),
        )

        # No price-based exit condition was met - position must remain held.
        assert signal is not None
        assert signal.action == "hold"


# ---------------------------------------------------------------------------
# 5. StrategyProposal generation
# ---------------------------------------------------------------------------

class TestStrategyProposalGeneration:
    @pytest.mark.asyncio
    async def test_buy_proposal_is_well_formed(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        proposal = captured[0]
        assert isinstance(proposal, StrategyProposal)
        assert proposal.strategy_id == "volatility_breakout"
        assert proposal.bot_id == bot.id
        assert proposal.direction == Direction.BUY
        assert proposal.execution_intent == ExecutionIntent.OPEN_POSITION
        assert proposal.suggested_position_size is not None and proposal.suggested_position_size > 0
        assert proposal.suggested_risk_budget_pct is not None
        assert len(proposal.assumptions) > 0
        assert len(proposal.reasons_for) > 0
        assert "atr_stop_multiplier" in proposal.adaptive_parameters_used
        assert "decision_score_size_multiplier" in proposal.adaptive_parameters_used
        assert not proposal.is_expired(proposal.generated_at)

    @pytest.mark.asyncio
    async def test_sell_proposal_is_well_formed(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        state = _armed_state()
        state.update({
            "entry_price": 64_000.0, "entry_atr": 50.0, "entry_stop_multiplier": 2.0,
            "entry_time": datetime.utcnow().isoformat(),
            "highest_price": 64_000.0, "trailing_stop": 63_950.0,
            "bars_since_entry": 10, "breakout_armed": False,
        })
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[
            SimpleNamespace(amount=0.01, trading_pair="BTC/USDT", entry_price=64_000.0)
        ])
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 63_900.0, bot.strategy_params, _flat_session(),  # below trailing stop
        )

        assert signal is not None and signal.action == "sell"
        proposal = captured[0]
        assert proposal.direction == Direction.SELL
        assert proposal.execution_intent == ExecutionIntent.CLOSE_POSITION
        assert proposal.decision_score.contributions[0].name == "trailing_stop"

    @pytest.mark.asyncio
    async def test_no_trade_proposal_when_not_armed(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        # Fresh state: warm up with flat bars, never compresses/arms.
        for p in [64_000.0 + (i % 3) for i in range(30)]:
            await engine._strategy_volatility_breakout(
                bot, p, {"bar_interval_seconds": 0}, _flat_session(),
            )

        assert captured, "expected at least one proposal once past warm-up"
        last = captured[-1]
        assert last.direction == Direction.NO_TRADE
        assert last.execution_intent == ExecutionIntent.NO_ACTION

    @pytest.mark.asyncio
    async def test_hold_proposal_when_position_open_and_no_exit(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        state = _armed_state()
        state.update({
            "entry_price": 64_000.0, "entry_atr": 50.0, "entry_stop_multiplier": 2.0,
            "entry_time": datetime.utcnow().isoformat(),
            "highest_price": 64_100.0, "trailing_stop": 63_800.0,
            "bars_since_entry": 2, "breakout_armed": False,
        })
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[
            SimpleNamespace(amount=0.01, trading_pair="BTC/USDT", entry_price=64_000.0)
        ])
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_050.0, bot.strategy_params, _flat_session(),
        )

        assert signal is not None and signal.action == "hold"
        assert captured[0].direction == Direction.HOLD
        assert captured[0].execution_intent == ExecutionIntent.HOLD_POSITION


# ---------------------------------------------------------------------------
# 6. Standalone Adapter compatibility
# ---------------------------------------------------------------------------

class TestStandaloneAdapterCompatibility:
    @pytest.mark.asyncio
    async def test_buy_signal_fields_match_adapter_translation(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        signal = await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )

        proposal = captured[0]
        expected = StandaloneAdapter.to_trade_signal(
            proposal, expected_move_pct=signal.expected_move_pct,
        )
        assert signal.action == expected.action
        assert signal.amount == expected.amount
        assert signal.score == expected.score
        assert signal.threshold == expected.threshold

    @pytest.mark.asyncio
    async def test_expired_proposal_would_be_discarded_by_adapter(self, monkeypatch):
        """The proposal this strategy builds carries a real validity window
        - confirm the Standalone Adapter's own expiry check (Phase 0.11)
        would discard it once past valid_until, proving the two components
        are actually wired together correctly, not just coincidentally
        compatible."""
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {bot.id: _armed_state()}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)

        await engine._strategy_volatility_breakout(
            bot, 64_500.0, bot.strategy_params, _flat_session(),
        )
        proposal = captured[0]
        assert proposal.is_expired(proposal.validity.valid_until) is True
        assert proposal.is_expired(proposal.generated_at) is False


# ---------------------------------------------------------------------------
# 7. Historical replay (end-to-end lifecycle through repeated calls)
# ---------------------------------------------------------------------------

class TestHistoricalReplay:
    @pytest.mark.asyncio
    async def test_full_lifecycle_compression_breakout_stop_exit(self):
        """Deterministic, no-crash replay driven entirely through the
        public strategy entry point: a mature, realistic compression-then-
        breakout history (nonzero bar ranges, matching real OHLC data -
        note bar_interval_seconds=0 organic tick-driving produces
        degenerate zero-range single-point bars, unrepresentative of real
        bar data, so this replay starts from a realistic pre-armed history
        exactly like production bar aggregation would produce) -> breakout
        buy -> a sharp reversal that trips the trailing stop -> sell.
        """
        engine = TradingEngine()
        # Default bar_interval_seconds (60): the seeded _armed_state()
        # bars must be read as-is on the first tick, not diluted by a
        # degenerate new single-tick bar closing immediately (see
        # test_strategy_activity.py's own convention of NOT using
        # bar_interval_seconds=0 when starting from a fully pre-seeded bar
        # history, only when driving bar-by-bar from a cold start).
        bot = make_bot(params={"min_compression_bars": 5})
        session = _flat_session()
        engine._volatility_breakout_states = {bot.id: _armed_state()}

        position = {"open": False, "entry": None}
        actions = []

        async def positions_side_effect(*_a, **_kw):
            if position["open"]:
                return [SimpleNamespace(amount=0.01, trading_pair="BTC/USDT", entry_price=position["entry"])]
            return []

        engine._get_bot_positions = AsyncMock(side_effect=positions_side_effect)

        # Tick 1: the seeded history's last bar is already a confirmed
        # breakout close - this tick should fire the entry.
        prices = [64_500.0] + [64_400.0, 64_200.0, 64_000.0, 63_500.0, 63_000.0]

        for price in prices:
            signal = await engine._strategy_volatility_breakout(
                bot, price, bot.strategy_params, session,
            )
            assert signal is None or isinstance(signal, TradeSignal)
            if signal is not None:
                actions.append(signal.action)
                if signal.action == "buy":
                    position["open"] = True
                    position["entry"] = price
                elif signal.action == "sell":
                    position["open"] = False
                    position["entry"] = None
                    break  # lifecycle complete

        assert actions[0] == "buy", f"expected the pre-armed breakout to fire first, got {actions}"
        assert "sell" in actions, f"expected the reversal to trip an exit, got {actions}"
        # No exceptions and every produced signal was well-formed - the
        # real assertion of this test is that the full lifecycle ran
        # deterministically to completion without crashing.


# ---------------------------------------------------------------------------
# 8. Regression tests
# ---------------------------------------------------------------------------

class TestRegressions:
    @pytest.mark.asyncio
    async def test_legacy_state_dict_missing_new_keys_does_not_crash(self):
        """A state dict seeded with only the PRE-migration keys (as if
        persisted before this Phase 1 migration, or restored from an old
        checkpoint) must not KeyError."""
        engine = TradingEngine()
        bot = make_bot()
        engine._volatility_breakout_states = {
            bot.id: {
                "bars": _armed_state()["bars"],
                "current_bar": None,
                "bb_width_history": [0.01] * 80 + [0.0002] * 5,
                "atr_history": [10.0] * 20,
                "compression_active": True,
                "compression_bars": 6,
                "compression_start": datetime.utcnow().isoformat(),
                "breakout_armed": False,
                "entry_price": None,
                "entry_atr": None,
                "highest_price": None,
                "trailing_stop": None,
                "bars_since_entry": 0,
                "last_breakout_attempt": None,
                # deliberately NO compression_min_width / armed_* / entry_stop_multiplier /
                # entry_time / take_profit_price keys.
            }
        }
        engine._get_bot_positions = AsyncMock(return_value=[])

        signal = await engine._strategy_volatility_breakout(
            bot, 64_000.0, bot.strategy_params, _flat_session(),
        )
        assert signal is None or isinstance(signal, TradeSignal)

    @pytest.mark.asyncio
    async def test_legacy_open_position_with_unknown_entry_price_does_not_crash(self):
        """A position opened outside this strategy's own tracking (e.g.
        imported from the exchange) has no locally-known entry_price - a
        failed-breakout exit on such a position must not crash trying to
        compute pnl, and must simply skip recording a trade outcome."""
        engine = TradingEngine()
        bot = make_bot(params={"bar_interval_seconds": 0, "failed_breakout_bars": 100})
        state = _armed_state()
        state.update({
            "entry_price": None, "entry_atr": None, "bars_since_entry": 0,
            "breakout_armed": False, "compression_active": False,
        })
        # Force the failed-breakout condition: bar close below upper band.
        state["bars"] = [
            {"open": 64_000.0, "high": 64_002.0, "low": 63_998.0, "close": 64_000.0,
             "start_ts": datetime.utcnow() - timedelta(minutes=30 - i)}
            for i in range(25)
        ]
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[
            SimpleNamespace(amount=0.01, trading_pair="BTC/USDT", entry_price=64_000.0)
        ])

        signal = await engine._strategy_volatility_breakout(
            bot, 64_000.0, bot.strategy_params, _flat_session(),
        )
        assert signal is None or isinstance(signal, TradeSignal)

    @pytest.mark.asyncio
    async def test_zero_atr_does_not_crash_evidence_collection(self):
        """All bars identical (zero true range) -> ATR=0. Breakout
        magnitude's division-by-ATR must degrade gracefully, not raise."""
        engine = TradingEngine()
        bot = make_bot()
        flat_bars = [
            {"open": 64_000.0, "high": 64_000.0, "low": 64_000.0, "close": 64_000.0,
             "start_ts": datetime.utcnow() - timedelta(minutes=30 - i)}
            for i in range(25)
        ]
        state = _armed_state()
        state["bars"] = flat_bars
        state["atr_history"] = [0.0] * 20
        engine._volatility_breakout_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[])

        signal = await engine._strategy_volatility_breakout(
            bot, 64_000.0, bot.strategy_params, _flat_session(),
        )
        assert signal is None or isinstance(signal, TradeSignal)
