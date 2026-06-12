-- Migration: Add per-strategy runtime state column to bots
-- Version: 003
-- Date: 2026-06-11
-- Description: Persist strategy runtime state (trailing stops, locked entry ATR,
--   cooldowns, price history, ...) separately from the user-facing
--   strategy_params config. Restored on resume so restarts are deterministic
--   and open positions keep their stop state across a restart.

-- SQLite supports ADD COLUMN; JSON has TEXT affinity. Existing rows get NULL,
-- which the engine treats as "no saved state" (falls back to legacy params).
ALTER TABLE bots ADD COLUMN strategy_state JSON;
