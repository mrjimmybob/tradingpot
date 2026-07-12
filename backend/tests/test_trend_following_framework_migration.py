"""Tests for trend_following's migration to the Strategy Decision Framework
(add-strategy-decision-framework, Phase 2).

Covers the full decision flow the migration is required to implement
end-to-end: Market Suitability Gate (built from scratch - the audit's most
acute finding was ZERO internal regime awareness) -> Adaptive Parameter
Resolver -> Evidence Collection -> Evidence-Based Decision Score -> Strategy
Edge Management -> StrategyProposal -> Standalone Adapter -> existing
execution pipeline (unchanged).

Uses the same bar-seeding conventions as the Phase 1 volatility_breakout
migration tests (bar_interval_seconds=0 so every call closes a bar;
hand-built series for deterministic indicator math).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models import BotStatus
from app.services.trading_engine import TradingEngine, TradeSignal
from app.services.strategy_framework.edge_management import EdgeCategory, StrategyEdgeManager
from app.services.strategy_framework.proposal import Direction, ExecutionIntent, StrategyProposal
from app.services.strategy_framework.standalone_adapter import StandaloneAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_PARAMS = {
    "short_period": 10,
    "long_period": 20,
    "atr_period": 14,
    "atr_multiplier": 2.0,
    "risk_percent": 1.0,
    "entry_confirmation_loops": 1,
    "exit_confirmation_loops": 1,
    "cooldown_seconds": 0,
    "bar_interval_seconds": 0,  # every call closes a bar
    "decision_score_threshold": 40.0,
    # default allowed_regimes (["trend_up", "volatility_expanding"]) unless a
    # test overrides it to exercise the suitability gate.
}


def make_bot(params=None, balance=100_000.0, bot_id=1):
    p = dict(_BASE_PARAMS)
    if params:
        p.update(params)
    return SimpleNamespace(
        id=bot_id, name="tf-test", trading_pair="BTC/USDT", strategy="trend_following",
        strategy_params=p, budget=balance, current_balance=balance,
        compound_enabled=False, is_dry_run=True, status=BotStatus.RUNNING,
        total_pnl=0.0, exchange_fee=0.1,
    )


def _uptrend_bars(n=60, base=60_000.0, step_pct=0.004):
    """Rising bar closes -> a clear trend_up regime with real H-L volatility."""
    bars = []
    price = base
    for _ in range(n):
        bars.append({
            "high": price * 1.0015, "low": price * 0.9985,
            "close": price, "start_ts": None,
        })
        price *= (1 + step_pct)
    return bars


def _downtrend_bars(n=60, base=60_000.0, step_pct=0.004):
    bars = []
    price = base
    for _ in range(n):
        bars.append({
            "high": price * 1.0015, "low": price * 0.9985,
            "close": price, "start_ts": None,
        })
        price *= (1 - step_pct)
    return bars


def _uptrend_ticks(n=30, base=60_000.0, step_pct=0.006):
    ticks = []
    price = base
    for _ in range(n):
        ticks.append(price)
        price *= (1 + step_pct)
    return ticks, price  # also return the next price to drive the entry tick


def _fresh_state(*, tf_bars, position_open=False, current_price=None,
                 trailing_stop=None, highest_price=None, ema_ref=None):
    """A trend state dict (position or flat). start_ts on the in-progress bar
    is None so the first call with bar_interval_seconds=0 closes it."""
    state = {
        "trailing_stop": trailing_stop,
        "highest_price": highest_price,
        "entry_atr": 100.0 if position_open else None,
        "entry_stop_multiplier": 2.0 if position_open else None,
        "entry_price": (current_price if position_open else None),
        "entry_time": None,
        "last_exit_time": None,
        "entry_confirmation_count": 0,
        "exit_confirmation_count": 0,
        "tf_bars": list(tf_bars),
        "tf_current_bar": None,
        "tf_atr_history": [],
        "regime_state": None,
    }
    return state


def _capture_proposals(monkeypatch) -> list:
    """Spy on StandaloneAdapter.to_trade_signal so tests can inspect the exact
    StrategyProposal the strategy built, while still exercising the real
    translation (never bypassed)."""
    captured: list = []
    real_fn = StandaloneAdapter.to_trade_signal

    def spy(proposal, **kwargs):
        captured.append(proposal)
        return real_fn(proposal, **kwargs)

    monkeypatch.setattr(StandaloneAdapter, "to_trade_signal", staticmethod(spy))
    return captured


async def _drive_entry(engine, bot, monkeypatch, *, params=None):
    """Seed a strong up-trend and drive one entry evaluation. Returns
    (signal, captured_proposals)."""
    ticks, next_price = _uptrend_ticks()
    engine._price_histories = {bot.id: list(ticks)}
    engine._trend_states = {bot.id: _fresh_state(tf_bars=_uptrend_bars())}
    engine._get_bot_positions = AsyncMock(return_value=[])
    captured = _capture_proposals(monkeypatch)
    session = AsyncMock()
    signal = await engine._strategy_trend_following(
        bot, next_price, bot.strategy_params, session,
    )
    return signal, captured, next_price


# ---------------------------------------------------------------------------
# 1. Evidence generation
# ---------------------------------------------------------------------------

class TestEvidenceGeneration:
    @pytest.mark.asyncio
    async def test_entry_names_all_four_evidence_factors(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)

        assert signal is not None and signal.action == "buy"
        buy = next(p for p in captured if p.direction == Direction.BUY)
        names = [c.name for c in buy.decision_score.contributions]
        assert names == [
            "Trend strength", "Price participation",
            "Confirmation persistence", "Volatility-normalized trend",
        ]

    @pytest.mark.asyncio
    async def test_explanation_contains_decision_metrics(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)

        buy = next(p for p in captured if p.direction == Direction.BUY)
        exp = buy.explanation
        assert "decision_score_total" in exp["metrics"]
        assert "edge_status_category" in exp["metrics"]
        assert exp["metrics"]["ema_fast"] > exp["metrics"]["ema_slow"]  # up-trend


# ---------------------------------------------------------------------------
# 2. Decision Score
# ---------------------------------------------------------------------------

class TestDecisionScoreCalculation:
    @pytest.mark.asyncio
    async def test_decision_score_deterministic_across_identical_calls(self, monkeypatch):
        totals = []
        for _ in range(2):
            engine = TradingEngine()
            bot = make_bot()
            _, captured, _ = await _drive_entry(engine, bot, monkeypatch)
            buy = next(p for p in captured if p.direction == Direction.BUY)
            totals.append(buy.decision_score.total)
        assert totals[0] == totals[1]

    @pytest.mark.asyncio
    async def test_single_factor_alone_is_insufficient(self, monkeypatch):
        """A near-flat market where only the confirmation-persistence factor
        is meaningfully positive (trend strength / participation ~ 0) must NOT
        clear the threshold - a single factor cannot carry a trade."""
        engine = TradingEngine()
        bot = make_bot()
        # Flat ticks: price barely above a flat EMA -> trend_strength ~ 0,
        # participation ~ 0, but confirmation counts to 1.
        flat = [60_000.0] * 30
        engine._price_histories = {bot.id: flat}
        engine._trend_states = {bot.id: _fresh_state(tf_bars=_uptrend_bars())}
        engine._get_bot_positions = AsyncMock(return_value=[])
        captured = _capture_proposals(monkeypatch)
        # A hair above the flat EMA so base_trend_ok can hold, but the
        # magnitude evidence is negligible.
        signal = await engine._strategy_trend_following(
            bot, 60_000.01, bot.strategy_params, AsyncMock(),
        )
        assert signal.action == "hold"
        prop = captured[-1]
        assert prop.decision_score.total < 40.0
        assert prop.direction == Direction.NO_TRADE


# ---------------------------------------------------------------------------
# 3. Market suitability (Pillar 2 - built from scratch)
# ---------------------------------------------------------------------------

class TestMarketSuitability:
    @pytest.mark.asyncio
    async def test_unsuitable_regime_blocks_entry_despite_strong_setup(self, monkeypatch):
        """A strong, high-score up-trend setup is still refused when the
        declared allowed_regimes excludes the current regime."""
        engine = TradingEngine()
        bot = make_bot(params={"allowed_regimes": ["trend_down"]})
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)

        assert signal.action == "hold"
        prop = captured[-1]
        assert prop.direction == Direction.NO_TRADE
        assert prop.market_suitability.is_suitable is False

    @pytest.mark.asyncio
    async def test_suitable_regime_allows_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()  # default allowed_regimes include trend_up
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)
        assert signal.action == "buy"
        buy = next(p for p in captured if p.direction == Direction.BUY)
        assert buy.market_suitability.is_suitable is True


# ---------------------------------------------------------------------------
# 4. Strategy Edge Management (Pillar 7)
# ---------------------------------------------------------------------------

def _seed_category_c(engine, bot):
    """Seed a losing history that classifies Category C (edge gone) when the
    regime is suitable and no parameter mismatch is cited."""
    mgr = StrategyEdgeManager()
    for _ in range(25):
        mgr.record_trade_outcome(bot.id, "trend_following", pnl=-50.0, win=False)
    engine._trend_following_edge_manager = mgr
    return mgr


def _seed_healthy(engine, bot):
    mgr = StrategyEdgeManager()
    for _ in range(25):
        mgr.record_trade_outcome(bot.id, "trend_following", pnl=+50.0, win=True)
    engine._trend_following_edge_manager = mgr
    return mgr


class TestStrategyEdgeManagement:
    @pytest.mark.asyncio
    async def test_category_c_blocks_new_entry_despite_qualifying_score(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        _seed_category_c(engine, bot)
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)

        prop = captured[-1]
        assert prop.edge_status.category == EdgeCategory.C
        assert signal.action == "hold"
        assert prop.direction == Direction.NO_TRADE

    @pytest.mark.asyncio
    async def test_healthy_history_does_not_block_entry(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        _seed_healthy(engine, bot)
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)
        assert signal.action == "buy"

    @pytest.mark.asyncio
    async def test_edge_management_never_force_closes_open_position(self, monkeypatch):
        """Even with a Category C classification, an OPEN position with no
        price-based exit trigger must HOLD - never be force-closed."""
        engine = TradingEngine()
        bot = make_bot()
        _seed_category_c(engine, bot)
        ticks, next_price = _uptrend_ticks()
        engine._price_histories = {bot.id: list(ticks)}
        # Position open, trailing stop well below price, price above slow EMA
        # (no trend break) -> no exit trigger.
        state = _fresh_state(
            tf_bars=_uptrend_bars(), position_open=True, current_price=next_price,
            trailing_stop=next_price * 0.90, highest_price=next_price * 1.05,
        )
        engine._trend_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.5)])
        signal = await engine._strategy_trend_following(
            bot, next_price, bot.strategy_params, AsyncMock(),
        )
        assert signal.action == "hold"  # NOT sell


# ---------------------------------------------------------------------------
# 5. StrategyProposal generation (Pillar 10)
# ---------------------------------------------------------------------------

class TestStrategyProposalGeneration:
    @pytest.mark.asyncio
    async def test_buy_proposal_is_well_formed(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        _, captured, _ = await _drive_entry(engine, bot, monkeypatch)
        buy = next(p for p in captured if p.direction == Direction.BUY)

        assert isinstance(buy, StrategyProposal)
        assert buy.execution_intent == ExecutionIntent.OPEN_POSITION
        assert buy.suggested_position_size and buy.suggested_position_size > 0
        assert "atr_multiplier" in buy.adaptive_parameters_used
        assert "decision_score_size_multiplier" in buy.adaptive_parameters_used
        assert len(buy.assumptions) >= 1
        # expected_edge_estimate SHALL be None (never self-computed).
        assert buy.expected_edge_estimate is None

    @pytest.mark.asyncio
    async def test_every_proposal_has_consistent_intent_pairing(self, monkeypatch):
        # Constructing any off-table (direction, intent) pairing raises in
        # StrategyProposal.__post_init__, so a captured proposal existing at
        # all proves the pairing is valid. Assert the ones this strategy emits.
        engine = TradingEngine()
        bot = make_bot()
        _, captured, _ = await _drive_entry(engine, bot, monkeypatch)
        for p in captured:
            assert (p.direction, p.execution_intent) in {
                (Direction.BUY, ExecutionIntent.OPEN_POSITION),
                (Direction.SELL, ExecutionIntent.CLOSE_POSITION),
                (Direction.HOLD, ExecutionIntent.HOLD_POSITION),
                (Direction.NO_TRADE, ExecutionIntent.NO_ACTION),
            }


# ---------------------------------------------------------------------------
# 6. Standalone Adapter compatibility (Pillar 10/11)
# ---------------------------------------------------------------------------

class TestStandaloneAdapterCompatibility:
    @pytest.mark.asyncio
    async def test_buy_signal_matches_adapter_translation(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        signal, captured, _ = await _drive_entry(engine, bot, monkeypatch)
        buy = next(p for p in captured if p.direction == Direction.BUY)

        assert isinstance(signal, TradeSignal)
        assert signal.action == "buy"
        assert signal.amount == buy.suggested_position_size
        assert signal.score == buy.decision_score.total
        assert signal.threshold == buy.decision_score.threshold
        assert signal.expected_move_pct is not None

    @pytest.mark.asyncio
    async def test_expired_proposal_would_be_discarded_by_adapter(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        _, captured, _ = await _drive_entry(engine, bot, monkeypatch)
        buy = next(p for p in captured if p.direction == Direction.BUY)
        # A proposal is expired at/after its own valid_until.
        assert buy.is_expired(buy.validity.valid_until) is True
        assert buy.is_expired(buy.validity.generated_at) is False


# ---------------------------------------------------------------------------
# 7. Exit paths + regressions
# ---------------------------------------------------------------------------

class TestExitPaths:
    @pytest.mark.asyncio
    async def test_trailing_stop_exit_produces_close_sell(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()
        ticks, next_price = _uptrend_ticks()
        engine._price_histories = {bot.id: list(ticks)}
        # Price at/below the restored trailing stop, and NOT a new high -> stop hit.
        state = _fresh_state(
            tf_bars=_uptrend_bars(), position_open=True, current_price=next_price,
            trailing_stop=next_price * 1.001, highest_price=next_price * 1.10,
        )
        engine._trend_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.5)])
        captured = _capture_proposals(monkeypatch)
        signal = await engine._strategy_trend_following(
            bot, next_price, bot.strategy_params, AsyncMock(),
        )
        assert signal.action == "sell"
        assert "trailing stop" in signal.reason.lower()
        prop = captured[-1]
        assert prop.direction == Direction.SELL
        assert prop.execution_intent == ExecutionIntent.CLOSE_POSITION

    @pytest.mark.asyncio
    async def test_confirmed_trend_break_produces_close_sell(self, monkeypatch):
        engine = TradingEngine()
        bot = make_bot()  # exit_confirmation_loops=1 -> confirms in one tick
        # Downtrend ticks so price is below the slow EMA (trend break).
        down = [70_000.0 * (0.99 ** i) for i in range(30)]
        low_price = down[-1] * 0.99
        engine._price_histories = {bot.id: down}
        state = _fresh_state(
            tf_bars=_downtrend_bars(), position_open=True, current_price=low_price,
            trailing_stop=low_price * 0.80, highest_price=low_price * 1.20,
        )
        engine._trend_states = {bot.id: state}
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.5)])
        captured = _capture_proposals(monkeypatch)
        signal = await engine._strategy_trend_following(
            bot, low_price, bot.strategy_params, AsyncMock(),
        )
        assert signal.action == "sell"
        assert "trend break" in signal.reason.lower()
        prop = captured[-1]
        assert prop.direction == Direction.SELL

    @pytest.mark.asyncio
    async def test_warmup_returns_hold_collecting_data(self):
        engine = TradingEngine()
        bot = make_bot(params={"long_period": 100})
        signal = await engine._strategy_trend_following(
            bot, 60_000.0, bot.strategy_params, AsyncMock(),
        )
        assert signal.action == "hold"
        assert "Collecting data" in signal.reason
