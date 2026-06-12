"""Production-hardening tests for C1, C3/H1, M4, M5.

Provides deterministic evidence that:
- C1: per-position stop-loss is enforced on every loop iteration (even on hold).
- C3/M5: all strategy runtime state (incl. datetimes) round-trips through the
  dedicated Bot.strategy_state column, and runtime keys are kept out of config.
- H1: a resumed bot keeps its trailing stop active immediately (no warmup gap).
- M4: the DB URL is resolved by precedence and SQLite gets WAL + busy_timeout.
"""

import sqlite3
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import select

from app.models import Bot, BotStatus
from app.models.database import (
    DEFAULT_DATABASE_URL,
    _apply_sqlite_pragmas,
    _resolve_database_url,
)
from app.services.risk_management import RiskAction, RiskAssessment
from app.services.trading_engine import (
    TradingEngine,
    TradeSignal,
    _from_jsonable,
    _to_jsonable,
)


# ============================================================================
# M4 - Database URL resolution + SQLite hardening
# ============================================================================


class TestDatabaseHardening:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("TRADINGBOT_DATABASE_URL", "sqlite+aiosqlite:///env.db")
        assert _resolve_database_url() == "sqlite+aiosqlite:///env.db"

    def test_config_used_when_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRADINGBOT_DATABASE_URL", raising=False)
        cfg = tmp_path / "config.yaml"
        cfg.write_text("database:\n  url: sqlite+aiosqlite:///cfg.db\n")
        assert _resolve_database_url(str(cfg)) == "sqlite+aiosqlite:///cfg.db"

    def test_default_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRADINGBOT_DATABASE_URL", raising=False)
        assert _resolve_database_url(str(tmp_path / "nope.yaml")) == DEFAULT_DATABASE_URL

    def test_malformed_config_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TRADINGBOT_DATABASE_URL", raising=False)
        cfg = tmp_path / "config.yaml"
        cfg.write_text("this: : not: valid: yaml: [")
        assert _resolve_database_url(str(cfg)) == DEFAULT_DATABASE_URL

    def test_pragmas_enable_wal_and_busy_timeout(self, tmp_path):
        db = tmp_path / "wal.db"
        conn = sqlite3.connect(str(db))
        try:
            _apply_sqlite_pragmas(conn)
            cur = conn.cursor()
            assert cur.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            assert cur.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
            assert cur.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        finally:
            conn.close()


# ============================================================================
# C3/M5 - State serialization and persistence
# ============================================================================


class TestStateSerialization:
    def test_datetime_roundtrip_is_json_safe(self):
        import json

        dt = datetime(2026, 6, 11, 12, 30, 45)
        state = {
            "trailing_stop": 95.5,
            "entry_time": dt,
            "last_exit_time": None,
            "nested": {"t": dt},
            "series": [dt, 1, "x"],
        }
        encoded = _to_jsonable(state)
        json.dumps(encoded)  # must not raise

        restored = _from_jsonable(encoded)
        assert restored["entry_time"] == dt
        assert restored["nested"]["t"] == dt
        assert restored["series"][0] == dt
        assert restored["last_exit_time"] is None
        assert restored["trailing_stop"] == 95.5


def _make_bot(test_db, strategy, params):
    bot = Bot(
        name="b",
        trading_pair="BTC/USDT",
        strategy=strategy,
        strategy_params=params,
        budget=1000.0,
        current_balance=1000.0,
        is_dry_run=True,
        status=BotStatus.RUNNING,
    )
    test_db.add(bot)
    return bot


class TestStatePersistence:
    @pytest.mark.asyncio
    async def test_trend_state_roundtrips_through_db(self, test_db):
        # Config carries a legacy runtime key that must be stripped (M5).
        bot = _make_bot(
            test_db, "trend_following",
            {"short_period": 50, "_grid_state": {"stale": 1}},
        )
        await test_db.flush()

        dt = datetime(2026, 6, 11, 9, 0, 0)
        engine = TradingEngine()
        engine._trend_states = {bot.id: {
            "trailing_stop": 95.0, "highest_price": 100.0, "entry_atr": 2.0,
            "entry_time": dt, "last_exit_time": None,
            "entry_confirmation_count": 0, "exit_confirmation_count": 0,
        }}
        engine._price_histories = {bot.id: [100.0 + i for i in range(60)]}

        await engine._save_bot_state(bot.id, test_db)
        await test_db.commit()

        refreshed = (
            await test_db.execute(select(Bot).where(Bot.id == bot.id))
        ).scalar_one()
        assert refreshed.strategy_state is not None
        # M5: runtime state lives in strategy_state, NOT strategy_params.
        assert "_grid_state" not in refreshed.strategy_params
        assert refreshed.strategy_params.get("short_period") == 50

        # Simulate a restart: a fresh engine restores from the DB column.
        engine2 = TradingEngine()
        await engine2._restore_strategy_state(refreshed)
        st = engine2._trend_states[bot.id]
        assert st["trailing_stop"] == 95.0
        assert st["entry_atr"] == 2.0
        assert st["entry_time"] == dt  # datetime fidelity preserved
        assert engine2._price_histories[bot.id] == [100.0 + i for i in range(60)]

    @pytest.mark.asyncio
    async def test_funding_cooldown_persisted(self, test_db):
        bot = _make_bot(test_db, "funding_carry", {})
        await test_db.flush()

        dt = datetime(2026, 6, 11, 8, 0, 0)
        engine = TradingEngine()
        engine._funding_states = {bot.id: {"last_exit_time": dt}}

        await engine._save_bot_state(bot.id, test_db)
        await test_db.commit()

        refreshed = (
            await test_db.execute(select(Bot).where(Bot.id == bot.id))
        ).scalar_one()
        engine2 = TradingEngine()
        await engine2._restore_strategy_state(refreshed)
        assert engine2._funding_states[bot.id]["last_exit_time"] == dt

    @pytest.mark.asyncio
    async def test_legacy_params_state_restored(self):
        # Bot saved by an older build: state embedded in strategy_params.
        bot = SimpleNamespace(
            id=7,
            strategy_state=None,
            strategy_params={"_grid_state": {"x": 1}, "_price_history": [1.0, 2.0, 3.0]},
        )
        engine = TradingEngine()
        await engine._restore_strategy_state(bot)
        assert engine._grid_states[7] == {"x": 1}
        assert engine._price_histories[7] == [1.0, 2.0, 3.0]


# ============================================================================
# H1 - Resumed bot keeps trailing stop active (no warmup gap)
# ============================================================================


def _trend_bot():
    return Bot(
        id=1, name="t", trading_pair="BTC/USDT", strategy="trend_following",
        strategy_params={}, budget=1000.0, current_balance=1000.0,
        is_dry_run=True, status=BotStatus.RUNNING,
    )


class TestResumeManagesPosition:
    @pytest.mark.asyncio
    async def test_without_restore_position_unmanaged_until_warmup(self):
        # Negative control: the defect's symptom. With no restored history a
        # resumed trend bot holding a position returns "collecting data".
        engine = TradingEngine()
        bot = _trend_bot()
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.1)])

        signal = await engine._strategy_trend_following(bot, 98.0, bot.strategy_params, None)
        assert signal.action == "hold"
        assert "Collecting data" in signal.reason

    @pytest.mark.asyncio
    async def test_restored_trailing_stop_exits_immediately(self):
        engine = TradingEngine()
        bot = _trend_bot()
        dt = datetime.utcnow()
        bot.strategy_state = _to_jsonable({
            "_trend_states": {
                "trailing_stop": 99.0, "highest_price": 110.0, "entry_atr": 2.0,
                "entry_time": dt, "last_exit_time": None,
                "entry_confirmation_count": 0, "exit_confirmation_count": 0,
            },
            "_price_histories": [100.0] * 250,
        })
        await engine._restore_strategy_state(bot)
        engine._get_bot_positions = AsyncMock(return_value=[SimpleNamespace(amount=0.1)])

        # Price below the restored trailing stop -> exit immediately, no warmup.
        signal = await engine._strategy_trend_following(bot, 98.0, bot.strategy_params, None)
        assert signal.action == "sell"
        assert "trailing stop" in signal.reason.lower()


# ============================================================================
# C1 - Stop-loss enforced every loop iteration, even on hold
# ============================================================================


class _FakeSessionCtx:
    def __init__(self, session):
        self._s = session

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *args):
        return False


class TestStopLossEnforcedOnHold:
    @pytest.mark.asyncio
    async def test_check_positions_runs_on_hold_iteration(self):
        engine = TradingEngine()
        bot_id = 1
        bot = SimpleNamespace(
            id=bot_id, status=BotStatus.RUNNING, is_dry_run=True,
            trading_pair="BTC/USDT", name="x", total_pnl=0.0,
        )

        session = Mock()
        result = Mock()
        result.scalar_one_or_none = Mock(return_value=bot)
        session.execute = AsyncMock(return_value=result)

        exchange = Mock()
        exchange.get_ticker = AsyncMock(return_value=SimpleNamespace(last=100.0))
        engine._exchange_services[bot_id] = exchange

        # Strategy holds (no trade) - the defect was that stop-loss was skipped.
        engine._execute_strategy = AsyncMock(
            return_value=TradeSignal(action="hold", amount=0)
        )
        engine._check_positions_stop_loss = AsyncMock()
        engine._resolve_pending_orders = AsyncMock(return_value=0)
        engine._save_bot_state = AsyncMock()
        # End the loop after exactly one iteration.
        async def _snapshot(*_a, **_k):
            engine._stop_flags[bot_id] = True
        engine._take_pnl_snapshot = AsyncMock(side_effect=_snapshot)

        engine._stop_flags[bot_id] = False

        risk_mgr = Mock()
        risk_mgr.full_risk_check = AsyncMock(
            return_value=RiskAssessment(action=RiskAction.CONTINUE, reason="", details={})
        )

        with patch("app.services.trading_engine.async_session_maker",
                   return_value=_FakeSessionCtx(session)), \
             patch("app.services.trading_engine.RiskManagementService", return_value=risk_mgr), \
             patch("app.services.trading_engine.VirtualWalletService", return_value=Mock()), \
             patch("app.services.trading_engine.asyncio.sleep", new=AsyncMock()):
            await engine._run_bot_loop(bot_id)

        engine._check_positions_stop_loss.assert_awaited_once()
        # H-1: state is checkpointed in-loop (not only on graceful shutdown).
        engine._save_bot_state.assert_awaited()
