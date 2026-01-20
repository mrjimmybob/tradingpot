-- Migration: Add accounting-grade ledger tables
-- Version: 001
-- Date: 2026-01-20
-- Description: Adds wallet_ledger, trades, tax_lots, and realized_gains tables

-- CRITICAL: These tables are the single source of truth for all financial transactions
-- SQLite is authoritative, CSV files are derived exports only

-- =====================================================================
-- 1. WALLET LEDGER (Append-only balance changes)
-- =====================================================================
CREATE TABLE IF NOT EXISTS wallet_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id VARCHAR(100) NOT NULL,
    bot_id INTEGER,
    asset VARCHAR(10) NOT NULL,
    delta_amount REAL NOT NULL,
    balance_after REAL,
    reason VARCHAR(50) NOT NULL,
    description VARCHAR(500),
    related_order_id INTEGER,
    related_trade_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (bot_id) REFERENCES bots(id),
    FOREIGN KEY (related_order_id) REFERENCES orders(id)
);

CREATE INDEX IF NOT EXISTS idx_wallet_ledger_owner ON wallet_ledger(owner_id);
CREATE INDEX IF NOT EXISTS idx_wallet_ledger_bot ON wallet_ledger(bot_id);
CREATE INDEX IF NOT EXISTS idx_wallet_ledger_asset ON wallet_ledger(asset);
CREATE INDEX IF NOT EXISTS idx_wallet_ledger_reason ON wallet_ledger(reason);
CREATE INDEX IF NOT EXISTS idx_wallet_ledger_created ON wallet_ledger(created_at);
CREATE INDEX IF NOT EXISTS idx_wallet_ledger_order ON wallet_ledger(related_order_id);
CREATE INDEX IF NOT EXISTS idx_wallet_ledger_trade ON wallet_ledger(related_trade_id);

-- =====================================================================
-- 2. TRADES (Execution events)
-- =====================================================================
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    owner_id VARCHAR(100) NOT NULL,
    bot_id INTEGER NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    base_asset VARCHAR(10) NOT NULL,
    quote_asset VARCHAR(10) NOT NULL,
    base_amount REAL NOT NULL,
    quote_amount REAL NOT NULL,
    price REAL NOT NULL,
    fee_amount REAL DEFAULT 0.0,
    fee_asset VARCHAR(10),
    modeled_cost REAL DEFAULT 0.0,
    exchange_trade_id VARCHAR(100),
    executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    strategy_used VARCHAR(50),

    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (bot_id) REFERENCES bots(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_order ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_owner ON trades(owner_id);
CREATE INDEX IF NOT EXISTS idx_trades_bot ON trades(bot_id);
CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(trading_pair);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side);
CREATE INDEX IF NOT EXISTS idx_trades_executed ON trades(executed_at);

-- =====================================================================
-- 3. TAX LOTS (FIFO cost basis)
-- =====================================================================
CREATE TABLE IF NOT EXISTS tax_lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id VARCHAR(100) NOT NULL,
    asset VARCHAR(10) NOT NULL,
    quantity_acquired REAL NOT NULL,
    quantity_remaining REAL NOT NULL,
    unit_cost REAL NOT NULL,
    total_cost REAL NOT NULL,
    purchase_trade_id INTEGER NOT NULL,
    purchase_date TIMESTAMP NOT NULL,
    is_fully_consumed BOOLEAN DEFAULT 0,
    consumed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (purchase_trade_id) REFERENCES trades(id)
);

CREATE INDEX IF NOT EXISTS idx_tax_lots_owner ON tax_lots(owner_id);
CREATE INDEX IF NOT EXISTS idx_tax_lots_asset ON tax_lots(asset);
CREATE INDEX IF NOT EXISTS idx_tax_lots_consumed ON tax_lots(is_fully_consumed);
CREATE INDEX IF NOT EXISTS idx_tax_lots_purchase_date ON tax_lots(purchase_date);
CREATE INDEX IF NOT EXISTS idx_tax_lots_purchase_trade ON tax_lots(purchase_trade_id);

-- =====================================================================
-- 4. REALIZED GAINS (Tax reporting)
-- =====================================================================
CREATE TABLE IF NOT EXISTS realized_gains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id VARCHAR(100) NOT NULL,
    asset VARCHAR(10) NOT NULL,
    quantity REAL NOT NULL,
    proceeds REAL NOT NULL,
    cost_basis REAL NOT NULL,
    gain_loss REAL NOT NULL,
    holding_period_days INTEGER NOT NULL,
    is_long_term BOOLEAN NOT NULL,
    purchase_trade_id INTEGER NOT NULL,
    sell_trade_id INTEGER NOT NULL,
    tax_lot_id INTEGER NOT NULL,
    purchase_date TIMESTAMP NOT NULL,
    sell_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (purchase_trade_id) REFERENCES trades(id),
    FOREIGN KEY (sell_trade_id) REFERENCES trades(id),
    FOREIGN KEY (tax_lot_id) REFERENCES tax_lots(id)
);

CREATE INDEX IF NOT EXISTS idx_realized_gains_owner ON realized_gains(owner_id);
CREATE INDEX IF NOT EXISTS idx_realized_gains_asset ON realized_gains(asset);
CREATE INDEX IF NOT EXISTS idx_realized_gains_gain_loss ON realized_gains(gain_loss);
CREATE INDEX IF NOT EXISTS idx_realized_gains_sell_date ON realized_gains(sell_date);
CREATE INDEX IF NOT EXISTS idx_realized_gains_sell_trade ON realized_gains(sell_trade_id);

-- =====================================================================
-- 5. UPDATE ORDERS TABLE (Add new fields)
-- =====================================================================
-- Note: SQLite doesn't support ALTER TABLE ADD COLUMN IF NOT EXISTS
-- These columns may already exist if added previously

-- Add reason column for tracking trade/rejection reasons
ALTER TABLE orders ADD COLUMN reason VARCHAR(500);

-- Validation query to check if migration was successful
-- Run this to verify all tables were created:
-- SELECT name FROM sqlite_master WHERE type='table' AND name IN ('wallet_ledger', 'trades', 'tax_lots', 'realized_gains');
