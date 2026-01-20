"""Database migration runner for accounting tables.

This script applies the accounting-grade ledger migration to the database.

Usage:
    python migrations/run_migrations.py

IMPORTANT: Backup your database before running migrations!
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.models import async_session_maker, Base, engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def apply_sql_migration(migration_file: Path):
    """Apply a SQL migration file.

    Args:
        migration_file: Path to SQL migration file
    """
    logger.info(f"Applying migration: {migration_file.name}")

    # Read migration SQL
    with open(migration_file, 'r') as f:
        sql = f.read()

    # Split into individual statements (handle multi-line)
    statements = []
    current_statement = []

    for line in sql.split('\n'):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('--'):
            continue

        current_statement.append(line)

        # Check if statement is complete
        if line.endswith(';'):
            statements.append(' '.join(current_statement))
            current_statement = []

    # Execute statements
    async with async_session_maker() as session:
        for i, statement in enumerate(statements):
            try:
                await session.execute(text(statement))
                await session.commit()
                logger.info(f"  Executed statement {i + 1}/{len(statements)}")
            except Exception as e:
                # Some statements may fail if columns already exist (idempotent)
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    logger.warning(f"  Statement {i + 1} already applied (skipping): {e}")
                else:
                    logger.error(f"  Failed to execute statement {i + 1}: {e}")
                    logger.error(f"  Statement: {statement[:100]}...")
                    raise

    logger.info(f"Migration {migration_file.name} completed successfully")


async def create_tables_from_models():
    """Create all tables from SQLAlchemy models.

    This is an alternative to SQL migrations using the ORM.
    """
    logger.info("Creating tables from SQLAlchemy models...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Tables created successfully")


async def verify_migration():
    """Verify that migration was applied successfully."""
    logger.info("Verifying migration...")

    required_tables = ['wallet_ledger', 'trades', 'tax_lots', 'realized_gains']

    async with async_session_maker() as session:
        result = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        existing_tables = [row[0] for row in result.fetchall()]

        missing_tables = [t for t in required_tables if t not in existing_tables]

        if missing_tables:
            logger.error(f"Migration verification FAILED: Missing tables: {missing_tables}")
            return False

        logger.info("Migration verification PASSED: All tables exist")

        # Check row counts
        for table in required_tables:
            result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            logger.info(f"  {table}: {count} rows")

    return True


async def main():
    """Run database migrations."""
    logger.info("=" * 70)
    logger.info("DATABASE MIGRATION: Accounting-Grade Ledger")
    logger.info("=" * 70)

    try:
        # Method 1: Apply SQL migration
        migration_file = Path(__file__).parent / "001_add_accounting_tables.sql"
        if migration_file.exists():
            await apply_sql_migration(migration_file)
        else:
            logger.warning(f"SQL migration file not found: {migration_file}")
            logger.info("Using ORM-based table creation instead...")
            await create_tables_from_models()

        # Verify migration
        success = await verify_migration()

        if success:
            logger.info("=" * 70)
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info("\nNext steps:")
            logger.info("1. Restart any running trading bots")
            logger.info("2. Test trade execution to verify ledger entries")
            logger.info("3. Export CSV files using the CSVExportService")
            logger.info("4. Run balance reconstruction validation")
        else:
            logger.error("MIGRATION FAILED - please review errors above")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Migration failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
