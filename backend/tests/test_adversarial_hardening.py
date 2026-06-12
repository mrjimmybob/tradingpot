"""Tests for the adversarial-review fixes: CR-1..3, H-1..3, M-1, M-2.

- CR-1: budget enforcement counts capital deployed in open positions.
- CR-2: dry-run simulator balances persist and restore across a restart.
- CR-3: FIFO realized P&L (not average-cost) drives the wallet P&L.
- H-1:  strategy/simulator state is checkpointed in the loop (not only on shutdown).
- H-2:  ledger consistency tolerance scales with balance magnitude.
- H-3:  strategy rotation never lands on an execution algo / auto_mode.
- M-1:  relative SQLite paths are anchored to an absolute location.
- M-2:  online DB backups are produced and are valid copies.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select

from app.models import Bot, BotStatus, Position, PositionSide
from app.models.database import _normalize_sqlite_url
from app.services.db_backup import DatabaseBackupService, _sqlite_file_path
from app.services.exchange import SimulatedExchangeService
from app.services.ledger_invariants import LedgerInvariantService
from app.services.trading_engine import TradingEngine, _ALPHA_STRATEGIES
from app.services.virtual_wallet import VirtualWalletService


async def _make_bot(test_db, budget=10000.0, balance=10000.0, compound=True):
    bot = Bot(
        name="b", trading_pair="BTC/USDT", strategy="funding_carry", strategy_params={},
        budget=budget, current_balance=balance, compound_enabled=compound,
        is_dry_run=True, status=BotStatus.RUNNING, total_pnl=0.0,
    )
    test_db.add(bot)
    await test_db.flush()
    return bot


# ============================================================================
# CR-1 - Budget counts deployed capital
# ============================================================================


class TestBudgetVsDeployedCapital:
    @pytest.mark.asyncio
    async def test_open_position_reduces_available(self, test_db):
        bot = await _make_bot(test_db, budget=10000.0, balance=10000.0)
        test_db.add(Position(
            bot_id=bot.id, trading_pair="BTC/USDT", side=PositionSide.LONG,
            amount=0.1, entry_price=50000.0, current_price=50000.0,
        ))
        await test_db.flush()

        wallet = VirtualWalletService(test_db)
        status = await wallet.get_wallet_status(bot.id)
        # 10000 budget - (0.1 * 50000 = 5000 deployed) = 5000 available.
        assert status.available_for_trade == pytest.approx(5000.0)

    @pytest.mark.asyncio
    async def test_buy_exceeding_remaining_budget_rejected(self, test_db):
        bot = await _make_bot(test_db, budget=10000.0, balance=10000.0)
        test_db.add(Position(
            bot_id=bot.id, trading_pair="BTC/USDT", side=PositionSide.LONG,
            amount=0.1, entry_price=50000.0, current_price=50000.0,
        ))
        await test_db.flush()

        wallet = VirtualWalletService(test_db)
        too_big = await wallet.validate_trade(bot.id, 6000.0)   # > 5000 remaining
        ok = await wallet.validate_trade(bot.id, 4000.0)        # <= 5000 remaining
        assert too_big.is_valid is False
        assert ok.is_valid is True

    @pytest.mark.asyncio
    async def test_no_positions_unchanged(self, test_db):
        bot = await _make_bot(test_db, budget=10000.0, balance=10000.0)
        wallet = VirtualWalletService(test_db)
        status = await wallet.get_wallet_status(bot.id)
        assert status.available_for_trade == pytest.approx(10000.0)


# ============================================================================
# CR-2 - Simulator durability
# ============================================================================


class TestSimulatorDurability:
    def test_export_import_roundtrip(self):
        sim = SimulatedExchangeService(initial_balance=10000.0)
        sim._simulated_balance = {"USDT": 5000.0, "BTC": 0.1}
        sim._order_counter = 7

        state = sim.export_state()

        fresh = SimulatedExchangeService(initial_balance=10000.0)
        fresh.import_state(state)
        assert fresh._simulated_balance == {"USDT": 5000.0, "BTC": 0.1}
        assert fresh._order_counter == 7

    def test_collect_bot_state_includes_sim_state(self):
        engine = TradingEngine()
        sim = SimulatedExchangeService(initial_balance=10000.0)
        sim._simulated_balance = {"USDT": 1234.0}
        engine._exchange_services[5] = sim

        state = engine._collect_bot_state(5)
        assert "_sim_state" in state
        assert state["_sim_state"]["balances"]["USDT"] == 1234.0

    def test_import_ignores_garbage(self):
        sim = SimulatedExchangeService(initial_balance=10000.0)
        before = dict(sim._simulated_balance)
        sim.import_state(None)
        sim.import_state({"balances": {}})  # empty -> keep
        assert sim._simulated_balance == before


# ============================================================================
# CR-3 - FIFO realized P&L drives the wallet
# ============================================================================


class TestRealizedPnlSource:
    @pytest.mark.asyncio
    async def test_uses_provided_fifo_pnl(self, test_db):
        bot = await _make_bot(test_db)
        test_db.add(Position(
            bot_id=bot.id, trading_pair="BTC/USDT", side=PositionSide.LONG,
            amount=0.1, entry_price=48000.0, current_price=48000.0,
        ))
        await test_db.flush()

        engine = TradingEngine()
        wallet = AsyncMock()
        await engine._close_or_reduce_position(
            bot.id, "BTC/USDT", 0.1, 50000.0, test_db, wallet, realized_pnl=123.45
        )
        # FIFO value is authoritative, NOT average-cost (which would be 200.0).
        wallet.record_trade_result.assert_awaited_once_with(bot.id, 123.45, 0)

    @pytest.mark.asyncio
    async def test_falls_back_to_average_cost(self, test_db):
        bot = await _make_bot(test_db)
        test_db.add(Position(
            bot_id=bot.id, trading_pair="BTC/USDT", side=PositionSide.LONG,
            amount=0.1, entry_price=48000.0, current_price=48000.0,
        ))
        await test_db.flush()

        engine = TradingEngine()
        wallet = AsyncMock()
        await engine._close_or_reduce_position(
            bot.id, "BTC/USDT", 0.1, 50000.0, test_db, wallet  # no realized_pnl
        )
        # (50000 - 48000) * 0.1 = 200.0
        wallet.record_trade_result.assert_awaited_once_with(bot.id, 200.0, 0)


# ============================================================================
# H-2 - Scaled consistency tolerance
# ============================================================================


class TestConsistencyTolerance:
    def test_tolerance_scales_with_magnitude(self):
        svc = LedgerInvariantService(Mock())
        assert svc._consistency_tolerance(0.0, 0.0) == pytest.approx(1e-6)
        # 1e9 * 1e-9 = 1.0 dominates the 1e-6 floor.
        assert svc._consistency_tolerance(1e9, 1e9) == pytest.approx(1.0)


# ============================================================================
# H-3 - Safe rotation
# ============================================================================


class TestRotation:
    @pytest.mark.asyncio
    async def test_never_rotates_into_execution_algo(self):
        engine = TradingEngine()
        for s in _ALPHA_STRATEGIES:
            nxt = await engine._get_next_strategy(s)
            assert nxt in _ALPHA_STRATEGIES
            assert nxt not in ("twap", "vwap", "auto_mode")

    @pytest.mark.asyncio
    async def test_cycles_and_handles_unknown(self):
        engine = TradingEngine()
        assert await engine._get_next_strategy(_ALPHA_STRATEGIES[-1]) == _ALPHA_STRATEGIES[0]
        assert await engine._get_next_strategy("twap") == _ALPHA_STRATEGIES[0]


# ============================================================================
# M-1 - Absolute SQLite path
# ============================================================================


class TestDbPathNormalization:
    def test_relative_becomes_absolute(self):
        out = _normalize_sqlite_url("sqlite+aiosqlite:///./tradingbot.db")
        path = out.split(":///", 1)[1]
        assert Path(path).is_absolute()
        assert path.endswith("/tradingbot.db")

    def test_memory_and_non_sqlite_unchanged(self):
        mem = "sqlite+aiosqlite:///:memory:"
        pg = "postgresql+asyncpg://u:p@h/db"
        assert _normalize_sqlite_url(mem) == mem
        assert _normalize_sqlite_url(pg) == pg


# ============================================================================
# M-2 - Database backup
# ============================================================================


class TestDatabaseBackup:
    @pytest.mark.asyncio
    async def test_backup_once_creates_valid_copy(self, tmp_path):
        db = tmp_path / "t.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE x (a INTEGER)")
        conn.execute("INSERT INTO x VALUES (42)")
        conn.commit()
        conn.close()

        svc = DatabaseBackupService(
            db_path=str(db), backup_dir=str(tmp_path / "backups"), retention=3
        )
        dest = await svc.backup_once()

        assert dest is not None and dest.exists()
        check = sqlite3.connect(str(dest))
        assert check.execute("SELECT a FROM x").fetchone()[0] == 42
        check.close()

    @pytest.mark.asyncio
    async def test_backup_noop_for_missing_db(self, tmp_path):
        svc = DatabaseBackupService(
            db_path=str(tmp_path / "nope.db"), backup_dir=str(tmp_path / "b")
        )
        assert await svc.backup_once() is None

    def test_sqlite_file_path_helper(self):
        assert _sqlite_file_path("sqlite+aiosqlite:///:memory:") is None
        assert _sqlite_file_path("postgresql://x") is None
        assert _sqlite_file_path("sqlite+aiosqlite:///D:/a/b.db") == "D:/a/b.db"
