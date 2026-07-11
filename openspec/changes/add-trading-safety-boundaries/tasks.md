## 1. Execution Cost Modeling
- [x] 1.1 Add execution cost model service and configuration (spread, slippage, impact, fee overrides). `backend/app/services/execution_cost_model.py`.
- [x] 1.2 Extend order storage to record modeled and realized costs. `backend/app/models/order.py`: `modeled_exchange_fee`, `modeled_spread_cost`, `modeled_slippage_cost`, `modeled_total_cost`, `realized_total_cost`.
- [x] 1.3 Apply cost model in the simulated/backtest execution path. `backend/app/backtesting/execution_model.py`.
- [x] 1.4 Apply cost model in the live execution path. Added `Bot.market_spread_pct`/`Bot.slippage_pct` columns (migration `008_add_bot_spread_slippage.sql`, default 0.0 - preserves current behavior exactly), and wired both live call sites (`_execute_trade`'s cost estimate and `_cost_estimate_for` used by order recovery/reconciliation) to read them instead of the hardcoded `0.0` literals. Not exposed via a bot-creation/edit API endpoint yet - `exchange_fee` (the existing analogous column) has no such endpoint either, so this matches established precedent rather than being a new gap.
- [x] 1.5 Replaced the daily/weekly loss calculation's "sum fees + modeled costs" approximation with real realized trading P&L: `_calculate_portfolio_period_loss` now joins `RealizedGain` (the ledger's authoritative realized-P&L record) to the closing `Trade` via its real `bot_id` FK, netting out the closing trade's fee (which `RealizedGain.gain_loss` doesn't already include - see the docstring on `_calculate_portfolio_period_loss` for exactly why). The drawdown check was already correct (real balance delta) and untouched.

## 2. Portfolio-Level Risk Caps
- [x] 2.1 Schema (`PortfolioRisk` model) and enforcement call site wired into the trade path. `backend/app/services/portfolio_risk.py`, called from `trading_engine.py`.
- [x] 2.2 Added `Bot.owner_id` (migration `007_add_bot_owner_id.sql`, backfills every existing bot to `'default'` - a single-operator deployment's bots all share one owner, so caps now aggregate correctly out of the box with zero behavior change for anyone who hasn't configured caps).
- [x] 2.3 `PortfolioRiskService.check_portfolio_risk` now queries `Bot.owner_id == owner_id` for real, and `bot_ids` is every bot for that owner (not just the one being checked).
- [x] 2.4 Same fix automatically covers the exposure cap's `Position` query, which already used the real `bot_id` FK and just needed a correct `bot_ids` list. Also fixed `get_portfolio_metrics`, which had the identical missing-filter bug (returned metrics for every bot in the database regardless of the requested `owner_id`).
- [x] 2.5 **Decision**: kept `enabled=False` as the column default. Auto-flipping every deployment's caps on (even with no threshold values set, which would be a no-op) was judged too easy to confuse with "caps are now silently protecting me" without an operator ever having chosen values - the new frontend form (4.3) makes opting in one click away with sane, visible defaults instead. Enabling is now a deliberate, informed action rather than a silent flip - matches the original design.md migration plan's own step 3 ("Enable portfolio caps... once UI configuration is in place").

## 3. Strategy Capacity Limits
- [x] 3.1 Enforced in auto-mode eligibility. `trading_engine.py`, `StrategyCapacityService.is_strategy_at_capacity`.
- [x] 3.2 Enforced at order time. `trading_engine.py`, `check_capacity_for_trade`.

## 4. API Schemas and UI
- [x] 4.1 Backend CRUD API for portfolio risk config. `backend/app/routers/portfolio.py` (`GET/POST/DELETE /portfolio/risk/{owner_id}`, `GET /portfolio/metrics/{owner_id}`).
- [x] 4.2 Read-only risk display. `frontend/src/components/RiskSafetyPanel.tsx` - was built but never mounted anywhere; now mounted on the Settings page alongside 4.3.
- [x] 4.3 Added `frontend/src/components/PortfolioRiskSettings.tsx` - a form for the 4 cap percentages + enabled toggle, wired to `GET`/`POST /portfolio/risk`. Mounted on `Settings.tsx` together with `RiskSafetyPanel`.
- [x] 4.4 **Decision**: this deployment has no multi-user/account model, so the UI hardcodes `owner_id = "default"` (matching `Bot.owner_id`'s backfill value from 2.2) with no owner picker - there is exactly one portfolio to configure today. A real owner picker is deferred until an actual user/account model exists; adding one now would be speculative UI for a concept that doesn't exist yet.

## 5. Tests
- [x] 5.1 CRUD endpoint tests. `backend/tests/test_portfolio_api.py`.
- [x] 5.2 Multi-bot aggregation tests: `backend/tests/test_portfolio_risk_service.py` (`TestMultiBotAggregation`, 3 tests) - a bot with zero drawdown of its own is correctly blocked once its sibling bot's drawdown is aggregated in; a bot under a different owner is correctly excluded; `get_portfolio_metrics` is scoped to the requested owner. Verified these fail against the pre-fix code (reverted the fix, confirmed 3 of 6 new tests failed with the exact old bug symptoms, restored the fix).
- [x] 5.3 Realized-loss tests: `backend/tests/test_portfolio_risk_service.py` (`TestRealizedPnlLossCalculation`, 3 tests) - a $205 realized trading loss (price loss + sell fee) correctly trips a 5% daily cap on a $1000 bot; a loss outside the daily window is correctly excluded; a net-profitable period correctly clamps to zero loss rather than going negative.
- [x] 5.4 Strategy capacity tests: `backend/tests/test_strategy_capacity_service.py` (7 tests, previously zero direct coverage) - bot-count capacity gate (`is_strategy_at_capacity`) and allocation-% capacity enforcement including the resize-vs-block branches (`check_capacity_for_trade`).

## Status Summary
All 20 items done. `python -m pytest tests/` (994 tests, was 981 before this
session's work) and `python -m pytest tests/test_migrations.py` (5 tests,
covering the two new migrations) both pass. Frontend: `tsc --noEmit`, `vite
build`, and `vitest run` (21 tests) all pass.
