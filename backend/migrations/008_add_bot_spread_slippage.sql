-- Migration 008: Add market_spread_pct and slippage_pct columns to bots table.
-- Closes the live/backtest execution-cost-model parity gap
-- (add-trading-safety-boundaries): live trading previously hardcoded these
-- to 0.0 with no way to configure them, while the backtest CLI already
-- supports non-zero --spread-pct/--slippage-pct. Default 0.0 preserves
-- current live behavior exactly; operators can now set realistic values.
ALTER TABLE bots ADD COLUMN market_spread_pct REAL NOT NULL DEFAULT 0.0;
ALTER TABLE bots ADD COLUMN slippage_pct REAL NOT NULL DEFAULT 0.0;
