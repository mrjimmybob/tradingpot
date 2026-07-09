-- Migration 005: Add exchange_fee column to bots table.
-- exchange_fee stores the taker fee percentage (e.g. 0.1 = 0.1%) used by the
-- execution viability gate. Default 0.1% matches the simulated exchange rate
-- and is a common real-exchange taker fee, so existing bots migrate safely.
ALTER TABLE bots ADD COLUMN exchange_fee REAL NOT NULL DEFAULT 0.1;
