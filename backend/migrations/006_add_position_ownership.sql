-- Migration: Add position ownership tracking to positions
-- Version: 006
-- Date: 2026-07-10
-- Description: Auto Mode could open a position under one sub-strategy and let
--   a different sub-strategy manage/close it, because ownership only lived as
--   a mutable, bot-level "current strategy" pointer, never as a fact recorded
--   on the position itself. This persists the owning strategy, its entry
--   reason, and whatever minimal strategy state it needs for exit decisions,
--   directly on the Position row so dispatch can be pinned to it while the
--   position stays open.

-- SQLite supports ADD COLUMN; JSON has TEXT affinity. Existing open positions
-- get NULL, which is treated as "unowned" (see auto-mode-position-ownership
-- spec) rather than an error.
ALTER TABLE positions ADD COLUMN owning_strategy VARCHAR(50);
ALTER TABLE positions ADD COLUMN entry_reason VARCHAR(500);
ALTER TABLE positions ADD COLUMN entry_strategy_state JSON;
