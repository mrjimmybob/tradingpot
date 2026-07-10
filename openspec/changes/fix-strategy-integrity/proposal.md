# Change: Fix strategy integrity (remove invalid funding_carry, enforce Auto Mode position ownership)

## Why
An expectancy audit found two correctness defects that invalidate performance statistics and mislabel risk:

1. `funding_carry` claims to be a funding-rate carry trade, but this bot only trades spot — there is no
   opposite perpetual leg to collect funding against. It is actually an unmanaged directional long strategy
   wearing a carry-trade label, which misrepresents its risk to anyone reading strategy performance.
2. Auto Mode can open a position under one sub-strategy and let a different sub-strategy close it. Ownership
   of an open position is tracked only as a mutable, bot-level "current strategy" pointer
   (`auto_state["current_strategy"]` in `TradingEngine._strategy_auto`), not as a fact recorded against the
   position itself. When the owning strategy is still entry-eligible but a competitor scores higher, Auto
   Mode reassigns the pointer with no open-position check, and the new strategy's executor then manages a
   position it never opened. This makes per-strategy win rate, expectancy, and drawdown numbers meaningless.

## What Changes
- **BREAKING**: Remove `funding_carry` from the strategy catalog, Auto Mode selection, capability tables,
  parameter validation, and API exposure. Historical rows referencing `funding_carry` are left untouched and
  must remain readable in reports/UI.
- **BREAKING**: Add a persisted `owning_strategy` (+ `entry_reason` + minimal exit-relevant strategy state) to
  the `Position` record, written at position-open time via an additive DB migration.
- Auto Mode's strategy-switch logic is changed so that while a position is open, dispatch is pinned to the
  position's persisted `owning_strategy` — closing the gap where a still-eligible-but-lower-scoring strategy
  could be swapped out for a higher-scoring one mid-trade. Re-scoring/reselection only happens once the
  owning position is fully closed. This must hold across live mode and service restarts, and binds the
  design of the not-yet-built backtester (`add-historical-backtesting`) to reuse the same field rather than
  a parallel mechanism.

## Impact
- Affected specs: `strategy-catalog` (new), `auto-mode-position-ownership` (new)
- Affected code:
  - `backend/app/routers/config.py` (STRATEGIES catalog)
  - `backend/app/routers/bots.py` (param validators)
  - `backend/app/services/trading_engine.py` (`_ALPHA_STRATEGIES`, `_strategy_funding_carry`, dispatch map,
    `_get_strategy_capabilities`, `_strategy_auto` switch logic, position-open call sites)
  - `backend/app/services/funding_diagnostic.py` (removed — only consumer was `_strategy_funding_carry`)
  - `backend/app/models/position.py` (new columns) + `backend/migrations/006_add_position_ownership.sql`
  - `backend/tests/test_funding_carry_strategy.py`, `test_funding_diagnostic.py` (removed) and every test file
    that uses `funding_carry` as a fixture strategy name (updated to a still-valid strategy)
  - `FUNDING_CARRY.md` (removed), `DEPLOYMENT.md` (example updated)
- Out of scope: `backend/app/services/trading_engine.py.bak_20260619_010315` is stray, untracked clutter not
  loaded by the app; left untouched by this change.
