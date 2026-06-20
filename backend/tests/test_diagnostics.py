"""Tests for the observe-only Strategy Diagnostics feature.

Covers the validation checklist from the diagnostics spec:
  1. diagnostics are populated for every strategy
  2. a HOLD-only strategy accumulates HOLD counters
  3. BUY and SELL signals are counted
  4. pause reasons are visible (store + API)
  5. execution failures are visible (store + engine call site + API)
  6. blocked trades are visible (store + engine call site + API)
  7. market-data failures are visible (store + API)
  8. diagnostics never crash a bot (the guard swallows internal errors)
  9. diagnostics do not change trading behavior (observe-only, no signal mutation)
"""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.models import Bot, BotStatus
from app.services.diagnostics import (
    DiagnosticsStore,
    diagnostics_store,
    BotDiagnostics,
    _normalize_reason,
    BLOCK_RISK_MANAGER,
    BLOCK_MIN_ORDER_SIZE,
    BLOCK_INSUFFICIENT_BALANCE,
    BLOCK_POSITION_LIMITS,
    BLOCK_EXCHANGE_VALIDATION,
    BLOCK_OTHER,
    DATA_TICKER,
    DATA_WEBSOCKET,
    DATA_UNAVAILABLE,
)
from app.services.trading_engine import TradingEngine, TradeSignal


def _sig(action, reason=""):
    return TradeSignal(action=action, amount=0, reason=reason)


# --------------------------------------------------------------------------- #
# 2 & 3: signal counting
# --------------------------------------------------------------------------- #
def test_hold_only_accumulates_hold_counter():
    store = DiagnosticsStore()
    for _ in range(25):
        store.record_signal(1, _sig("hold", "RSI neutral"))
    d = store.get(1)
    assert d.hold_signals == 25
    assert d.buy_signals == 0
    assert d.sell_signals == 0
    assert d.last_signal_action == "hold"


def test_buy_and_sell_signals_counted():
    store = DiagnosticsStore()
    store.record_signal(1, _sig("buy", "EMA crossover up"))
    store.record_signal(1, _sig("buy", "EMA crossover up"))
    store.record_signal(1, _sig("sell", "trailing stop hit"))
    store.record_signal(1, _sig("hold", "holding"))
    d = store.get(1)
    assert d.buy_signals == 2
    assert d.sell_signals == 1
    assert d.hold_signals == 1
    assert d.last_signal_action == "hold"


def test_none_signal_counts_as_hold():
    store = DiagnosticsStore()
    store.record_signal(1, None)
    assert store.get(1).hold_signals == 1


def test_reason_normalization_collapses_numbers():
    assert _normalize_reason("Holding position, stop at $49000.00") == \
        _normalize_reason("Holding position, stop at $48000.50")
    store = DiagnosticsStore()
    store.record_signal(1, _sig("hold", "Holding position, stop at $49000.00"))
    store.record_signal(1, _sig("hold", "Holding position, stop at $48000.50"))
    top = store.get(1).top_reasons()
    assert len(top) == 1
    assert top[0]["count"] == 2


def test_top_reasons_ordered_by_count():
    store = DiagnosticsStore()
    for _ in range(5):
        store.record_signal(1, _sig("hold", "Collecting bars"))
    for _ in range(2):
        store.record_signal(1, _sig("hold", "Trend not confirmed"))
    top = store.get(1).top_reasons()
    assert top[0]["reason"] == "Collecting bars"
    assert top[0]["count"] == 5
    assert top[1]["count"] == 2


# --------------------------------------------------------------------------- #
# Evaluation statistics
# --------------------------------------------------------------------------- #
def test_evaluations_counted_and_24h():
    store = DiagnosticsStore()
    for _ in range(10):
        store.record_evaluation(1)
    d = store.get(1)
    assert d.total_evaluations == 10
    assert d.runtime_evaluations == 10
    assert d.evaluations_last_24h() == 10
    assert d.last_evaluation_at is not None


def test_start_runtime_resets_runtime_not_total_and_clears_pause():
    store = DiagnosticsStore()
    for _ in range(5):
        store.record_evaluation(1)
    store.record_pause(1, "manual")
    store.start_runtime(1)
    for _ in range(3):
        store.record_evaluation(1)
    d = store.get(1)
    assert d.total_evaluations == 8     # lifetime preserved
    assert d.runtime_evaluations == 3   # runtime reset
    assert d.pause_reason is None       # pause cleared on fresh run


# --------------------------------------------------------------------------- #
# 6: blocked trades
# --------------------------------------------------------------------------- #
def test_blocked_categories_counted():
    store = DiagnosticsStore()
    store.record_blocked(1, BLOCK_RISK_MANAGER, "portfolio cap")
    store.record_blocked(1, BLOCK_MIN_ORDER_SIZE, "below min")
    store.record_blocked(1, BLOCK_INSUFFICIENT_BALANCE, "no funds")
    store.record_blocked(1, BLOCK_POSITION_LIMITS, "capacity")
    store.record_blocked(1, BLOCK_EXCHANGE_VALIDATION, "rejected")
    store.record_blocked(1, BLOCK_OTHER, "misc")
    d = store.get(1).to_dict()["blocked"]
    assert d[BLOCK_RISK_MANAGER] == 1
    assert d[BLOCK_MIN_ORDER_SIZE] == 1
    assert d[BLOCK_INSUFFICIENT_BALANCE] == 1
    assert d[BLOCK_POSITION_LIMITS] == 1
    assert d[BLOCK_EXCHANGE_VALIDATION] == 1
    assert d[BLOCK_OTHER] == 1
    assert d["last_category"] == BLOCK_OTHER
    assert d["last_reason"] == "misc"


def test_unknown_block_category_falls_back_to_other():
    store = DiagnosticsStore()
    store.record_blocked(1, "totally-made-up", "x")
    assert store.get(1).blocked[BLOCK_OTHER] == 1


# --------------------------------------------------------------------------- #
# 5: execution outcomes
# --------------------------------------------------------------------------- #
def test_execution_outcomes_counted():
    store = DiagnosticsStore()
    store.record_execution(1, "buy", success=True)
    store.record_execution(1, "sell", success=True)
    store.record_execution(1, "buy", success=False, reason="exchange rejected")
    store.record_execution(1, "sell", success=False, reason="invariant failed")
    d = store.get(1)
    assert d.successful_buys == 1
    assert d.successful_sells == 1
    assert d.failed_buys == 1
    assert d.failed_sells == 1
    assert d.last_exec_failure_reason == "invariant failed"
    assert d.last_exec_failure_at is not None


# --------------------------------------------------------------------------- #
# 7: market-data failures
# --------------------------------------------------------------------------- #
def test_market_data_failures_counted():
    store = DiagnosticsStore()
    store.record_data_failure(1, DATA_TICKER, "ticker down")
    store.record_data_failure(1, DATA_WEBSOCKET, "ws drop")
    store.record_data_failure(1, DATA_UNAVAILABLE, "no data")
    d = store.get(1)
    assert d.ticker_failures == 1
    assert d.websocket_failures == 1
    assert d.data_unavailable == 1
    assert d.last_data_failure_reason == "no data"


# --------------------------------------------------------------------------- #
# 4: pause reason
# --------------------------------------------------------------------------- #
def test_pause_reason_recorded():
    store = DiagnosticsStore()
    store.record_pause(1, "Failure circuit breaker after 10 errors")
    d = store.get(1).to_dict()
    assert d["pause"]["reason"] == "Failure circuit breaker after 10 errors"
    assert d["pause"]["paused_at"] is not None


# --------------------------------------------------------------------------- #
# 8: diagnostics never crash a bot (guard swallows internal errors)
# --------------------------------------------------------------------------- #
def test_guard_swallows_internal_errors():
    store = DiagnosticsStore()

    class Boom:
        @property
        def action(self):
            raise RuntimeError("boom")

    # record_signal reads signal.action; the guard must swallow the RuntimeError.
    result = store.record_signal(1, Boom())
    assert result is None  # no exception propagated

    # Other entry points must also be total even on bad input.
    assert store.record_blocked(1, None) is None
    assert store.record_execution(1, None, success=True) is None


def test_empty_diagnostics_to_dict_shape():
    d = BotDiagnostics(bot_id=7).to_dict()
    assert d["bot_id"] == 7
    assert d["evaluations"]["total"] == 0
    assert d["signals"]["hold"] == 0
    assert set(d["blocked"]).issuperset(
        {BLOCK_RISK_MANAGER, BLOCK_MIN_ORDER_SIZE, BLOCK_OTHER}
    )
    assert d["top_reasons"] == []
    assert d["pause"]["reason"] is None


# --------------------------------------------------------------------------- #
# 9: observe-only — record_signal must not mutate the signal and returns None
# --------------------------------------------------------------------------- #
def test_record_signal_does_not_mutate_signal():
    store = DiagnosticsStore()
    sig = _sig("buy", "EMA crossover")
    before = (sig.action, sig.amount, sig.reason)
    ret = store.record_signal(1, sig)
    assert ret is None
    assert (sig.action, sig.amount, sig.reason) == before


# --------------------------------------------------------------------------- #
# 1: diagnostics populate for EVERY strategy (real strategy output is counted)
# --------------------------------------------------------------------------- #
ALL_STRATEGIES = [
    "dca_accumulator", "adaptive_grid", "mean_reversion",
    "trend_following", "volatility_breakout", "funding_carry", "auto_mode",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
async def test_diagnostics_populated_for_every_strategy(strategy):
    """Drive each real strategy executor and confirm its emitted signal is
    counted by the diagnostics store exactly as the trading loop does."""
    engine = TradingEngine()
    engine._get_bot_positions = AsyncMock(return_value=[])
    engine._get_last_order = AsyncMock(return_value=None)
    engine._get_order_count = AsyncMock(return_value=0)
    engine._price_histories = {}

    bot = SimpleNamespace(
        id=4242, name="diag", trading_pair="BTC/USDT", strategy=strategy,
        strategy_params={}, budget=1000.0, current_balance=1000.0,
        compound_enabled=False, is_dry_run=True, status=BotStatus.RUNNING,
        total_pnl=0.0,
    )
    # funding_carry / auto_mode read a registered exchange; give them a sim one.
    engine._exchange_services[bot.id] = engine._make_simulated_exchange(bot.budget)

    # Generic empty-result session for strategies that touch the DB (auto_mode).
    empty_result = Mock()
    empty_result.scalars = Mock(return_value=Mock(all=Mock(return_value=[])))
    empty_result.scalar_one_or_none = Mock(return_value=None)
    empty_result.scalar = Mock(return_value=0)
    empty_result.all = Mock(return_value=[])
    session = SimpleNamespace(execute=AsyncMock(return_value=empty_result))

    store = DiagnosticsStore()
    # Mirror the loop: record an evaluation, then the produced signal.
    signal = await engine._execute_strategy(bot, 50000.0, session)
    store.record_evaluation(bot.id)
    store.record_signal(bot.id, signal)

    d = store.get(bot.id)
    assert d is not None, f"no diagnostics recorded for {strategy}"
    assert d.total_evaluations == 1
    total_signals = d.buy_signals + d.sell_signals + d.hold_signals
    assert total_signals == 1, f"{strategy} signal not counted"
    assert len(d.top_reasons()) >= 1


# --------------------------------------------------------------------------- #
# 5/6 engine wiring: a real engine call site records into the GLOBAL store
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_execute_trade_sell_without_position_records_block():
    """A real _execute_trade call path (sell with no open position) must record a
    blocked trade in the shared diagnostics_store — proving the engine is wired
    to diagnostics, not just the store in isolation."""
    diagnostics_store.clear(99001)
    engine = TradingEngine()

    bot = SimpleNamespace(id=99001, trading_pair="BTC/USDT", strategy="trend_following",
                          is_dry_run=True)
    signal = TradeSignal(action="sell", amount=100.0, order_type="market")

    # session.execute(...).scalar_one_or_none() -> None  (no position)
    session = SimpleNamespace()
    result_obj = SimpleNamespace(scalar_one_or_none=lambda: None)
    session.execute = AsyncMock(return_value=result_obj)

    order = await engine._execute_trade(bot, SimpleNamespace(), signal, 50000.0, session)
    assert order is None  # behavior unchanged: still rejected
    d = diagnostics_store.get(99001)
    assert d is not None
    assert d.blocked[BLOCK_OTHER] >= 1
    assert d.last_block_reason == "cannot sell without open position"


# --------------------------------------------------------------------------- #
# API endpoint — diagnostics are visible over HTTP without reading logs
# --------------------------------------------------------------------------- #
async def _create_bot(test_db, strategy="trend_following", status=BotStatus.RUNNING):
    bot = Bot(
        name="diag-api", trading_pair="BTC/USDT", strategy=strategy,
        strategy_params={}, budget=1000.0, current_balance=1000.0,
        is_dry_run=True, status=status,
    )
    test_db.add(bot)
    await test_db.commit()
    await test_db.refresh(bot)
    return bot


@pytest.mark.asyncio
async def test_diagnostics_endpoint_404_for_unknown_bot(client):
    resp = await client.get("/api/bots/123456/diagnostics")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_diagnostics_endpoint_returns_zeroed_shape(client, test_db):
    bot = await _create_bot(test_db)
    diagnostics_store.clear(bot.id)
    resp = await client.get(f"/api/bots/{bot.id}/diagnostics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["evaluations"]["total"] == 0
    assert body["signals"]["hold"] == 0
    assert body["current_activity"]  # always a human-readable line
    assert body["paused_reason"] is None


@pytest.mark.asyncio
async def test_diagnostics_endpoint_surfaces_all_recorded(client, test_db):
    bot = await _create_bot(test_db)
    diagnostics_store.clear(bot.id)
    diagnostics_store.record_evaluation(bot.id)
    diagnostics_store.record_signal(bot.id, _sig("hold", "Waiting for EMA crossover"))
    diagnostics_store.record_signal(bot.id, _sig("buy", "EMA crossover up"))
    diagnostics_store.record_blocked(bot.id, BLOCK_MIN_ORDER_SIZE, "order $5 < $10 min")
    diagnostics_store.record_execution(bot.id, "buy", success=False, reason="exchange rejected")
    diagnostics_store.record_data_failure(bot.id, DATA_UNAVAILABLE, "feed down")

    resp = await client.get(f"/api/bots/{bot.id}/diagnostics")
    body = resp.json()
    assert body["signals"]["hold"] == 1
    assert body["signals"]["buy"] == 1
    assert body["blocked"][BLOCK_MIN_ORDER_SIZE] == 1
    assert body["execution"]["failed_buys"] == 1
    assert body["execution"]["last_failure_reason"] == "exchange rejected"
    assert body["market_data"]["data_unavailable"] == 1
    reasons = {r["reason"] for r in body["top_reasons"]}
    assert "Waiting for EMA crossover" in reasons


@pytest.mark.asyncio
async def test_diagnostics_endpoint_shows_pause_reason(client, test_db):
    bot = await _create_bot(test_db, status=BotStatus.PAUSED)
    diagnostics_store.clear(bot.id)
    diagnostics_store.record_pause(bot.id, "Failure circuit breaker after 10 errors")
    resp = await client.get(f"/api/bots/{bot.id}/diagnostics")
    body = resp.json()
    assert body["paused_reason"] == "Failure circuit breaker after 10 errors"
    assert "Paused" in body["current_activity"]
    assert "circuit breaker" in body["current_activity"]


@pytest.mark.asyncio
async def test_diagnostics_endpoint_paused_without_recorded_reason(client, test_db):
    """A bot paused before diagnostics captured a reason still gets a non-empty
    explanation rather than a blank field."""
    bot = await _create_bot(test_db, status=BotStatus.PAUSED)
    diagnostics_store.clear(bot.id)
    resp = await client.get(f"/api/bots/{bot.id}/diagnostics")
    body = resp.json()
    assert body["paused_reason"]  # non-empty fallback


# --------------------------------------------------------------------------- #
# P1: lifecycle state vs decision state must never collide on the UI surfaces.
# Reproduces the TestBot8-FC report (RUNNING bot, funding-carry regime HOLD).
# --------------------------------------------------------------------------- #
from app.services.decision_status import decision_status_store, DecisionState


@pytest.mark.asyncio
async def test_running_regime_hold_is_not_displayed_as_paused(client, test_db):
    """A RUNNING bot holding on a regime filter must show a DECISION of
    'Waiting for market regime' and a Current Activity that NEVER begins with
    'Paused'. (The exact bug: lifecycle RUNNING but UI said PAUSED.)"""
    bot = await _create_bot(test_db, strategy="funding_carry", status=BotStatus.RUNNING)
    diagnostics_store.clear(bot.id)
    decision_status_store.clear(bot.id)

    # Exactly the production signal funding_carry emits on a regime miss.
    sig = _sig("hold", "Funding Carry: regime trend_flat not in ['trend_up']")
    decision_status_store.update_from_signal(bot.id, sig, symbol="BTC/USDT")
    diagnostics_store.record_signal(bot.id, sig)

    # Decision-status endpoint: state is the regime decision, NOT lifecycle Paused.
    ds = (await client.get(f"/api/bots/{bot.id}/decision-status")).json()
    assert ds["state"] == DecisionState.WAITING_FOR_REGIME
    assert ds["state"] != DecisionState.PAUSED

    # Diagnostics endpoint: lifecycle stays running; activity must not say Paused.
    diag = (await client.get(f"/api/bots/{bot.id}/diagnostics")).json()
    assert diag["status"] == "running"
    assert diag["paused_reason"] is None
    assert not diag["current_activity"].startswith("Paused")
    assert "Waiting for market regime" in diag["current_activity"]


@pytest.mark.asyncio
async def test_lifecycle_and_decision_are_independent(client, test_db):
    """When the bot IS paused, lifecycle shows paused and Current Activity leads
    with 'Paused:' — proving the two concepts render independently and correctly
    in both directions."""
    bot = await _create_bot(test_db, strategy="funding_carry", status=BotStatus.PAUSED)
    diagnostics_store.clear(bot.id)
    decision_status_store.clear(bot.id)
    diagnostics_store.record_pause(bot.id, "Risk limit reached")

    diag = (await client.get(f"/api/bots/{bot.id}/diagnostics")).json()
    assert diag["status"] == "paused"
    assert diag["paused_reason"] == "Risk limit reached"
    assert diag["current_activity"].startswith("Paused:")


@pytest.mark.asyncio
async def test_current_activity_never_leads_with_paused_while_running(client, test_db):
    """Defense-in-depth: even if a decision state somehow carried the lifecycle
    'Paused' label, a RUNNING bot's Current Activity must not begin with
    'Paused'."""
    bot = await _create_bot(test_db, status=BotStatus.RUNNING)
    diagnostics_store.clear(bot.id)
    decision_status_store.clear(bot.id)
    # Force the (now-impossible via mapping) lifecycle label onto the decision.
    decision_status_store.update(bot.id, DecisionState.PAUSED, reason="should be coerced")

    diag = (await client.get(f"/api/bots/{bot.id}/diagnostics")).json()
    assert diag["status"] == "running"
    assert not diag["current_activity"].startswith("Paused")
