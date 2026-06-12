"""Database configuration and session management."""

import os
from pathlib import Path

import yaml
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./tradingbot.db"


def _resolve_database_url(config_path: str | None = None) -> str:
    """Resolve the database URL.

    Precedence: TRADINGBOT_DATABASE_URL env var -> config.yaml `database.url`
    -> built-in default. The config file is read directly (not via the services
    layer) to avoid an import cycle at module-load time.
    """
    env_url = os.environ.get("TRADINGBOT_DATABASE_URL")
    if env_url:
        return env_url

    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent.parent / "config.yaml")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        url = (data.get("database") or {}).get("url")
        if url:
            return url
    except FileNotFoundError:
        pass
    except Exception:
        # Never let a malformed config file prevent startup; fall back to default.
        pass

    return DEFAULT_DATABASE_URL


def _normalize_sqlite_url(url: str) -> str:
    """Anchor a relative SQLite file path to the backend dir (M-1).

    A relative URL like ``sqlite+aiosqlite:///./tradingbot.db`` resolves against
    the process CWD, so starting the server from a different directory silently
    opens a different (often empty) database. Rewrite relative SQLite paths to an
    absolute path anchored at the backend directory. Non-SQLite and in-memory
    URLs are returned unchanged.
    """
    if not url.startswith("sqlite") or ":memory:" in url:
        return url
    marker = ":///"
    if marker not in url:
        return url
    scheme, path = url.split(marker, 1)
    p = Path(path)
    if not p.is_absolute():
        backend_dir = Path(__file__).resolve().parent.parent.parent
        p = (backend_dir / path).resolve()
    return f"{scheme}:///{p.as_posix()}"


def _apply_sqlite_pragmas(dbapi_connection) -> None:
    """Apply durability/concurrency PRAGMAs to a raw SQLite connection.

    - WAL journaling lets readers and a writer proceed concurrently, which
      matters when several bot loops commit at once (avoids "database is
      locked" errors that would otherwise surface as skipped trades).
    - busy_timeout makes a contended writer wait instead of failing immediately.
    - synchronous=NORMAL is the safe, durable pairing with WAL.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


DATABASE_URL = _normalize_sqlite_url(_resolve_database_url())

_is_sqlite = DATABASE_URL.startswith("sqlite")
_is_memory = ":memory:" in DATABASE_URL

# A generous busy timeout at the driver level complements the PRAGMA above.
_connect_args = {"timeout": 30} if _is_sqlite else {}

engine = create_async_engine(DATABASE_URL, echo=False, connect_args=_connect_args)

# WAL is file-only; in-memory databases neither need nor support it.
if _is_sqlite and not _is_memory:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        _apply_sqlite_pragmas(dbapi_connection)


async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


async def get_session() -> AsyncSession:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        yield session


async def init_db():
    """Initialize the database, creating all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
