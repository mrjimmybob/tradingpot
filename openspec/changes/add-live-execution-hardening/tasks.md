# Tasks: add-live-execution-hardening

## 1. Order pre-flight (exchange service)

- [x] 1.1 Add pre-flight to `ExchangeService.place_market_order`/`place_limit_order`: `amount_to_precision`/`price_to_precision` rounding, market min-amount and min-cost checks with clear local rejection; log full raw ccxt response on success
- [x] 1.2 Unit tests: rounding applied, below-min-cost rejected without calling create_order, raw response logged, simulator unaffected

## 2. Fill accuracy (trading engine)

- [x] 2.1 Market/limit path `_execute_trade`: record `fee_asset` from `exchange_order.fee_currency` (fallback quote); use `filled` amount and exchange `cost` for trade/ledger; use `filled` for position updates
- [x] 2.2 Apply the same to TWAP/VWAP execution recording paths
- [x] 2.3 Unit tests: partial fill updates position with filled amount; fee currency recorded from exchange response

## 3. Balance reconciliation

- [x] 3.1 Add `_reconcile_live_account` to the trading engine: throttled (default 300s, `trading.reconciliation_interval_seconds`), aggregates running live bots' virtual cash (quote) and open positions (base per asset), compares against exchange balances, warns + creates `Alert` (type `balance_reconciliation`) on shortfall > tolerance (default 1%)
- [x] 3.2 Hook into the live bot loop (no-op for dry-run bots, no-op when interval not elapsed)
- [x] 3.3 Unit tests: sufficient balances → no alert; quote shortfall → alert with amounts; base shortfall → alert; dry-run bots excluded

## 4. Verification

- [x] 4.1 Full backend suite green (baseline 457)
- [x] 4.2 Validate change with openspec --strict
