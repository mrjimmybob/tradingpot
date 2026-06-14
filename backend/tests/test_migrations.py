"""Regression tests for the database migration runner.

These reproduce and lock down the production deploy failure where
``migrations/run_migrations.py`` ran (deploy stops the service and migrates
BEFORE the app's ``init_db`` ever creates the core ORM tables) and a migration's
``ALTER TABLE orders``/``ALTER TABLE bots`` hit ``no such table``.

The runner must be self-sufficient: it creates the ORM baseline schema first,
so it succeeds against an empty / partially-initialised / existing database
WITHOUT the application having started. We drive the real entrypoint as a
subprocess (the way the deploy invokes it), pointing it at a temp DB via
``TRADINGBOT_DATABASE_URL``.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
RUNNER = BACKEND_DIR / "migrations" / "run_migrations.py"


def _run_migrations(db_path: Path) -> subprocess.CompletedProcess:
    """Invoke the migration runner against an isolated SQLite file."""
    # Inherit the full environment (interpreter/library paths) and only point
    # the DB URL at the isolated temp file.
    env = dict(os.environ)
    env["TRADINGBOT_DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    return subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
    )


def _tables(db_path: Path) -> set:
    con = sqlite3.connect(str(db_path))
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def _columns(db_path: Path, table: str) -> set:
    con = sqlite3.connect(str(db_path))
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def test_fresh_empty_database_migrates_without_app_start(tmp_path):
    """The exact production scenario: migrate a brand-new DB, no app start."""
    db = tmp_path / "fresh.db"
    result = _run_migrations(db)

    assert result.returncode == 0, f"migration failed:\n{result.stderr}"
    tables = _tables(db)
    # Core ORM tables the SQL migrations depend on must exist...
    assert {"orders", "bots"} <= tables
    # ...the accounting tables the SQL migrations add must exist...
    assert {"wallet_ledger", "trades", "tax_lots", "realized_gains"} <= tables
    # ...and the ALTERs that used to fail must have applied.
    assert "reason" in _columns(db, "orders")
    assert "strategy_state" in _columns(db, "bots")


def test_migration_is_idempotent(tmp_path):
    """Running twice (e.g. redeploys) must keep succeeding."""
    db = tmp_path / "idem.db"
    assert _run_migrations(db).returncode == 0
    second = _run_migrations(db)
    assert second.returncode == 0, f"second run failed:\n{second.stderr}"


def test_realistic_partial_database(tmp_path):
    """A DB with a correct subset of tables (older schema) migrates cleanly."""
    db = tmp_path / "partial.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE bots (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()

    result = _run_migrations(db)
    assert result.returncode == 0, f"partial migration failed:\n{result.stderr}"
    assert "reason" in _columns(db, "orders")
    assert {"wallet_ledger", "trades"} <= _tables(db)


def test_existing_data_is_preserved(tmp_path):
    """Migrations are additive: existing rows survive a re-migration."""
    db = tmp_path / "data.db"
    assert _run_migrations(db).returncode == 0

    con = sqlite3.connect(str(db))
    con.execute(
        "INSERT INTO bots (name, trading_pair, strategy, budget, current_balance, "
        "is_dry_run, status) VALUES "
        "('keepme', 'BTC/USDT', 'dca_accumulator', 100, 100, 1, 'STOPPED')"
    )
    con.commit()
    con.close()

    assert _run_migrations(db).returncode == 0
    con = sqlite3.connect(str(db))
    try:
        n = con.execute("SELECT COUNT(*) FROM bots WHERE name='keepme'").fetchone()[0]
    finally:
        con.close()
    assert n == 1


def test_bot_and_simulator_state_survive_migration(tmp_path):
    """Bot row + persisted dry-run simulator state (bots.strategy_state JSON)
    survive a re-migration intact — the deploy-restart / rollback data-safety
    guarantee."""
    import json

    db = tmp_path / "state.db"
    assert _run_migrations(db).returncode == 0

    sim_state = {"_sim_state": {"balances": {"USDT": 99.98, "BTC": 0.0003135}},
                 "trailing_stop": 63000.0}
    con = sqlite3.connect(str(db))
    con.execute(
        "INSERT INTO bots (name, trading_pair, strategy, budget, current_balance, "
        "is_dry_run, status, strategy_state) VALUES "
        "('simbot', 'BTC/USDT', 'auto_mode', 100, 99.98, 1, 'RUNNING', ?)",
        (json.dumps(sim_state),),
    )
    con.commit()
    con.close()

    assert _run_migrations(db).returncode == 0  # e.g. a redeploy

    con = sqlite3.connect(str(db))
    try:
        row = con.execute(
            "SELECT current_balance, status, strategy_state FROM bots WHERE name='simbot'"
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    assert row[0] == 99.98               # balance intact
    assert row[1] == "RUNNING"           # status intact (resumes after restart)
    assert json.loads(row[2]) == sim_state  # dry-run simulator state intact
