-- Migration: Add rolling trade statistics to strategy_performance_metrics
-- Version: 004
-- Date: 2026-06-24
-- Description: Extends auto-mode learning with per-strategy win/loss/PnL stats
--   so scoring uses actual trade outcomes rather than balance-change proxies.

ALTER TABLE strategy_performance_metrics ADD COLUMN total_trades INTEGER NOT NULL DEFAULT 0;
ALTER TABLE strategy_performance_metrics ADD COLUMN winning_trades INTEGER NOT NULL DEFAULT 0;
ALTER TABLE strategy_performance_metrics ADD COLUMN losing_trades INTEGER NOT NULL DEFAULT 0;
ALTER TABLE strategy_performance_metrics ADD COLUMN realized_pnl_usd REAL NOT NULL DEFAULT 0.0;
ALTER TABLE strategy_performance_metrics ADD COLUMN profit_factor REAL NOT NULL DEFAULT 0.0;
ALTER TABLE strategy_performance_metrics ADD COLUMN win_rate REAL NOT NULL DEFAULT 0.0;
ALTER TABLE strategy_performance_metrics ADD COLUMN last_trade_time TIMESTAMP;
