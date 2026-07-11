-- Migration 007: Add owner_id column to bots table.
-- Cross-bot grouping key for PortfolioRiskService (add-trading-safety-boundaries).
-- Every existing bot backfills to 'default' - a single-operator deployment's
-- bots all share one owner, so portfolio-wide caps now aggregate across all
-- of them instead of silently comparing a bot to itself (the bug this
-- migration fixes the root cause of). Explicit multi-tenant support (a real
-- owner/user model) is a separate, future concern.
ALTER TABLE bots ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'default';
