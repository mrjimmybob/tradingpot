"""SQLite database backup service (M-2).

A 30-day run on a single SQLite file with no backups means corruption or a
disk-full event loses everything. This provides an online (consistent) backup
using SQLite's backup API, a startup backup, and a periodic background backup
with simple retention. No-ops cleanly for in-memory or non-SQLite databases.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from ..models.database import DATABASE_URL
from .config import config_service

logger = logging.getLogger(__name__)


def _sqlite_file_path(url: str) -> Optional[str]:
    """Extract the on-disk path from a SQLite URL, or None if not applicable."""
    if not url.startswith("sqlite") or ":memory:" in url:
        return None
    marker = ":///"
    if marker not in url:
        return None
    return url.split(marker, 1)[1]


class DatabaseBackupService:
    """Creates consistent SQLite backups on a schedule with retention."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        backup_dir: Optional[str] = None,
        retention: int = 7,
    ):
        self.db_path = db_path if db_path is not None else _sqlite_file_path(DATABASE_URL)
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        elif self.db_path:
            self.backup_dir = Path(self.db_path).resolve().parent / "backups"
        else:
            self.backup_dir = None
        self.retention = max(1, retention)
        self._task: Optional[asyncio.Task] = None
        self._stop = False

    async def backup_once(self) -> Optional[Path]:
        """Write one consistent backup; returns the path, or None if skipped."""
        if not self.db_path or not Path(self.db_path).exists() or self.backup_dir is None:
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = self.backup_dir / f"tradingbot-{ts}.db"
        # SQLite online backup: consistent even while the DB is in use.
        async with aiosqlite.connect(self.db_path) as src, \
                aiosqlite.connect(str(dest)) as dst:
            await src.backup(dst)
        self._prune()
        logger.info(f"Database backup written to {dest}")
        return dest

    def _prune(self) -> None:
        if self.backup_dir is None:
            return
        backups = sorted(self.backup_dir.glob("tradingbot-*.db"))
        for old in backups[:-self.retention]:
            try:
                old.unlink()
            except Exception as e:
                logger.warning(f"Could not prune old backup {old}: {e}")

    async def _run(self, interval: float) -> None:
        while not self._stop:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            if self._stop:
                break
            try:
                await self.backup_once()
            except Exception as e:
                logger.error(f"Scheduled DB backup failed: {e}")

    def start(self) -> None:
        """Start the periodic backup task (no-op for non-file databases)."""
        if not self.db_path:
            logger.info("DB backup disabled (non-file database)")
            return
        interval = float(config_service.get("trading.backup_interval_seconds") or 3600)
        self._stop = False
        self._task = asyncio.create_task(self._run(interval))
        logger.info(f"DB backup task started (every {interval:.0f}s, keep {self.retention})")

    async def stop(self) -> None:
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None


# Global instance used by the application lifespan.
db_backup_service = DatabaseBackupService()
