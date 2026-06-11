# Change: Add trading safety boundaries (cost modeling, portfolio caps, capacity limits)

## Why
Current safety controls are per-bot and fee-only. We need a consistent execution cost model and higher-level
risk boundaries across bots and strategies to prevent hidden risk build-up.

## What Changes
- Add execution cost modeling (fees, spread, slippage/impact) and apply it to wallet/PnL/risk checks.
- Add portfolio-level risk caps across all bots (loss caps, exposure caps) with enforcement in the trade path.
- Add strategy capacity limits (per-strategy allocation and concurrent bot caps) with enforcement in auto-mode and order sizing.

## Impact
- Affected specs: execution-cost-modeling, portfolio-risk-caps, strategy-capacity-limits
- Affected code: backend/app/services/trading_engine.py, backend/app/services/exchange.py,
  backend/app/services/risk_management.py, backend/app/models/order.py, backend/app/models/bot.py,
  backend/app/routers/bots.py, backend/app/routers/config.py, frontend/src/pages/*
