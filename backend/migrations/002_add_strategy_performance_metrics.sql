-- Migration: Add strategy performance metrics table
-- Version: 002
-- Date: 2026-01-27
-- Description: Persistent storage for auto-mode strategy learning

-- Strategy Performance Metrics Table
-- Stores per-bot, per-strategy performance metrics to survive restarts
-- Used by auto-mode to remember cooldowns, blacklists, and PnL history
CREATE TABLE IF NOT EXISTS strategy_performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id INTEGER NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    
    -- Performance metrics
    recent_pnl_pct REAL NOT NULL DEFAULT 0.0,
    max_drawdown_pct REAL NOT NULL DEFAULT 0.0,
    
    -- Failure tracking
    failure_count INTEGER NOT NULL DEFAULT 0,
    
    -- Timing
    last_exit_time TIMESTAMP,
    cooldown_until TIMESTAMP,
    
    -- Metadata
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE,
    UNIQUE (bot_id, strategy_name)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_strategy_perf_bot ON strategy_performance_metrics(bot_id);
CREATE INDEX IF NOT EXISTS idx_strategy_perf_strategy ON strategy_performance_metrics(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_perf_cooldown ON strategy_performance_metrics(cooldown_until);
