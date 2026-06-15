"""Tests for the in-memory bot decision-status feature.

Covers the three pieces of the "Decision Status" panel:
  1. The in-memory store (update/change-detection/get/clear).
  2. Deriving a user-facing state from a strategy TradeSignal.
  3. The GET /api/bots/{id}/decision-status endpoint (live + empty + 404).
"""

import pytest

from app.services.decision_status import (
    DecisionState,
    DecisionStatusStore,
    decision_status_store,
    derive_state_from_signal,
)
from app.services.trading_engine import TradeSignal


# --- Store -----------------------------------------------------------------

def test_update_and_get_roundtrip():
    store = DecisionStatusStore()
    store.update(1, DecisionState.HOLD, reason="nothing to do", symbol="BTC/USDT")
    snap = store.get(1)
    assert snap is not None
    assert snap.state == DecisionState.HOLD
    assert snap.reason == "nothing to do"
    assert snap.symbol == "BTC/USDT"
    assert snap.updated_at is not None


def test_update_returns_true_only_on_state_change():
    store = DecisionStatusStore()
    assert store.update(1, DecisionState.EVALUATING) is True   # first ever
    assert store.update(1, DecisionState.EVALUATING) is False  # same state
    assert store.update(1, DecisionState.HOLD) is True         # transition


def test_symbol_is_retained_across_updates_when_omitted():
    """A later update that does not pass a symbol keeps the previous one, so the
    panel never loses the trading pair between ticks."""
    store = DecisionStatusStore()
    store.update(1, DecisionState.EVALUATING, symbol="ETH/USDT")
    store.update(1, DecisionState.HOLD)  # no symbol
    assert store.get(1).symbol == "ETH/USDT"


def test_clear_removes_status():
    store = DecisionStatusStore()
    store.update(2, DecisionState.HOLD)
    store.clear(2)
    assert store.get(2) is None


def test_only_latest_status_is_kept():
    """It is not an audit log: a second update replaces the first."""
    store = DecisionStatusStore()
    store.update(1, DecisionState.WARMING_UP, reason="collecting")
    store.update(1, DecisionState.BUY_SIGNAL, reason="go")
    snap = store.get(1)
    assert snap.state == DecisionState.BUY_SIGNAL
    assert snap.reason == "go"


# --- State derivation ------------------------------------------------------

def test_derive_buy_and_sell():
    assert derive_state_from_signal(
        TradeSignal(action="buy", amount=10)
    ) == DecisionState.BUY_SIGNAL
    assert derive_state_from_signal(
        TradeSignal(action="sell", amount=10)
    ) == DecisionState.SELL_SIGNAL


def test_derive_none_is_evaluating():
    assert derive_state_from_signal(None) == DecisionState.EVALUATING


@pytest.mark.parametrize("reason,expected", [
    # Warming up — collecting initial history (every strategy uses "Collecting").
    ("Mean Reversion: Collecting bars (5/20)", DecisionState.WARMING_UP),
    ("Trend Following: Collecting data (10/200)", DecisionState.WARMING_UP),
    ("Volatility Breakout: Collecting bars (3/20)", DecisionState.WARMING_UP),
    # Waiting for data — an external feed is down.
    ("Funding Carry: funding-rate data unavailable (holding)", DecisionState.WAITING_FOR_DATA),
    # Cooldown / risk.
    ("Grid: Cooldown after kill (30min remaining)", DecisionState.COOLDOWN),
    ("Grid: Kill switch (drawdown 16.0%)", DecisionState.RISK_LIMIT),
    # Paused by regime filter.
    ("DCA: Paused (regime=trend_down)", DecisionState.PAUSED),
    ("Mean Reversion: Waiting for suitable regime (current: trend_up)", DecisionState.PAUSED),
    # Working and holding — waiting for an entry condition, NOT warming up.
    ("DCA: Next buy in 12.0 min", DecisionState.HOLD),
    ("Trend Following: Waiting for EMA crossover (short <= long)", DecisionState.HOLD),
    ("Mean Reversion: Holding, target $101, stop $99, bars 3/10", DecisionState.HOLD),
])
def test_derive_hold_reasons_map_to_states(reason, expected):
    assert derive_state_from_signal(
        TradeSignal(action="hold", amount=0, reason=reason)
    ) == expected


def test_five_required_categories_are_distinct():
    """The operator must be able to tell these five apart without reading logs."""
    states = {
        derive_state_from_signal(TradeSignal("hold", 0, reason="Waiting for EMA crossover")),
        derive_state_from_signal(TradeSignal("hold", 0, reason="Collecting bars (1/20)")),
        derive_state_from_signal(TradeSignal("hold", 0, reason="funding data unavailable")),
        derive_state_from_signal(TradeSignal("hold", 0, reason="Paused (regime=trend_down)")),
        derive_state_from_signal(TradeSignal("hold", 0, reason="Kill switch (drawdown)")),
    }
    assert states == {
        DecisionState.HOLD,
        DecisionState.WARMING_UP,
        DecisionState.WAITING_FOR_DATA,
        DecisionState.PAUSED,
        DecisionState.RISK_LIMIT,
    }


def test_update_from_signal_carries_score_and_threshold():
    store = DecisionStatusStore()
    sig = TradeSignal(action="buy", amount=5, reason="conviction", score=0.8, threshold=0.5)
    store.update_from_signal(7, sig, symbol="BTC/USDT")
    snap = store.get(7)
    assert snap.state == DecisionState.BUY_SIGNAL
    assert snap.score == 0.8
    assert snap.threshold == 0.5
    assert snap.symbol == "BTC/USDT"


# --- API endpoint ----------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_global_store():
    """Keep the process-global store from leaking between API tests."""
    yield
    decision_status_store._statuses.clear()


async def _create_bot(client) -> int:
    resp = await client.post("/api/bots", json={
        "name": "Status Bot",
        "trading_pair": "BTC/USDT",
        "strategy": "dca_accumulator",
        "budget": 1000.0,
        "is_dry_run": True,
    })
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.mark.asyncio
async def test_decision_status_endpoint_empty(client):
    """With no live snapshot the endpoint returns state=null, not a 500."""
    bot_id = await _create_bot(client)
    resp = await client.get(f"/api/bots/{bot_id}/decision-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["bot_id"] == bot_id
    assert data["state"] is None
    assert data["symbol"] == "BTC/USDT"


@pytest.mark.asyncio
async def test_decision_status_endpoint_reflects_store(client):
    bot_id = await _create_bot(client)
    decision_status_store.update(
        bot_id, DecisionState.WARMING_UP,
        reason="Collecting bars (3/20)", symbol="BTC/USDT",
    )
    resp = await client.get(f"/api/bots/{bot_id}/decision-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == DecisionState.WARMING_UP
    assert data["reason"] == "Collecting bars (3/20)"
    assert data["updated_at"] is not None


@pytest.mark.asyncio
async def test_decision_status_endpoint_404_for_unknown_bot(client):
    resp = await client.get("/api/bots/999999/decision-status")
    assert resp.status_code == 404


async def _set_status(test_db, bot_id: int, status) -> None:
    from sqlalchemy import select
    from app.models import Bot
    bot = (await test_db.execute(select(Bot).where(Bot.id == bot_id))).scalar_one()
    bot.status = status
    await test_db.commit()


@pytest.mark.asyncio
async def test_running_bot_never_blank_without_snapshot(client, test_db):
    """A bot left RUNNING in the DB with no live in-memory snapshot (e.g. a
    fresh process / ghost-RUNNING after a failed resume) must still report a
    meaningful state, never a blank panel."""
    from app.models import BotStatus
    bot_id = await _create_bot(client)
    await _set_status(test_db, bot_id, BotStatus.RUNNING)
    assert decision_status_store.get(bot_id) is None  # no live snapshot

    resp = await client.get(f"/api/bots/{bot_id}/decision-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == DecisionState.EVALUATING  # non-blank, meaningful
    assert data["symbol"] == "BTC/USDT"


@pytest.mark.asyncio
async def test_paused_bot_without_snapshot_reports_paused(client, test_db):
    from app.models import BotStatus
    bot_id = await _create_bot(client)
    await _set_status(test_db, bot_id, BotStatus.PAUSED)

    resp = await client.get(f"/api/bots/{bot_id}/decision-status")
    assert resp.json()["state"] == DecisionState.PAUSED


@pytest.mark.asyncio
async def test_stopped_bot_without_snapshot_is_idle(client, test_db):
    """A non-running bot legitimately has no decision state (state=null)."""
    from app.models import BotStatus
    bot_id = await _create_bot(client)
    await _set_status(test_db, bot_id, BotStatus.STOPPED)

    resp = await client.get(f"/api/bots/{bot_id}/decision-status")
    assert resp.json()["state"] is None


@pytest.mark.asyncio
async def test_live_snapshot_takes_precedence_over_db_fallback(client, test_db):
    """When a real snapshot exists it wins over the DB-status fallback."""
    from app.models import BotStatus
    bot_id = await _create_bot(client)
    await _set_status(test_db, bot_id, BotStatus.RUNNING)
    decision_status_store.update(
        bot_id, DecisionState.HOLD, reason="Waiting for EMA crossover",
        symbol="BTC/USDT",
    )
    resp = await client.get(f"/api/bots/{bot_id}/decision-status")
    data = resp.json()
    assert data["state"] == DecisionState.HOLD
    assert data["reason"] == "Waiting for EMA crossover"
