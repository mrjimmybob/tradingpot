# Database Migrations

This directory contains database migrations for the trading bot system.

## Overview

The accounting-grade ledger system requires new database tables:
- `wallet_ledger` - Append-only ledger for all balance changes
- `trades` - Execution event records
- `tax_lots` - FIFO cost basis tracking
- `realized_gains` - Tax reporting records

## Running Migrations

### Method 1: Python Script (Recommended)

```bash
cd backend
python migrations/run_migrations.py
```

This script:
- Applies the SQL migration
- Verifies all tables were created
- Provides status information

### Method 2: Direct SQL

```bash
cd backend
sqlite3 tradingbot.db < migrations/001_add_accounting_tables.sql
```

### Method 3: ORM-based (Development)

```python
from app.models import Base, engine

async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
```

## Migration Files

- `001_add_accounting_tables.sql` - SQL migration script
- `run_migrations.py` - Python migration runner
- `README.md` - This file

## Verification

After running migrations, verify tables exist:

```sql
SELECT name FROM sqlite_master
WHERE type='table'
AND name IN ('wallet_ledger', 'trades', 'tax_lots', 'realized_gains');
```

Expected output: 4 rows

## Rollback

⚠️ **IMPORTANT**: These migrations are designed to be ADDITIVE ONLY.
- No existing data is modified
- No existing columns are removed
- Rollback is not provided (append-only design)

If you need to rollback:
1. Restore from backup
2. Or manually DROP the new tables (NOT RECOMMENDED in production)

## Backward Compatibility

✅ All migrations are backward compatible:
- Existing bots continue working
- New tables start empty
- No breaking API changes
- Historical data is preserved

## Next Steps

After migration:
1. Restart trading bots
2. Test trade execution
3. Verify ledger entries are created
4. Check tax lot generation
5. Export CSV files for validation

## Support

For issues or questions:
- Check logs in `backend/logs/`
- Review migration errors
- Verify database permissions
- Ensure SQLite version >= 3.8.0
