# Change: Add live execution hardening (order pre-flight, fill accuracy, balance reconciliation)

## Why

The live order path has never been exercised against a real exchange, and code review found concrete defects that will surface on first contact with real money:

1. **No precision or minimum-size handling**: the engine computes raw amounts like `0.000808172760181522` BTC and submits them unmodified. Exchanges reject orders with wrong amount precision or below minimum notional — small-budget bots are the most likely to hit both.
2. **Fee currency is assumed, not read**: trades are recorded with `fee_asset=quote_asset` even though the exchange reports the actual fee currency (MEXC spot buys often charge fees in the base asset). Live P&L and the ledger would silently drift from exchange reality.
3. **Partial fills corrupt position state**: trade/ledger recording uses the actual `filled` amount, but position updates use the *requested* amount — a partial fill desynchronizes positions from the ledger.
4. **No audit trail of raw exchange responses**: when the first live orders behave unexpectedly there is nothing to diagnose from.
5. **No balance reconciliation**: nothing ever compares what the bots think they hold against what the exchange account actually holds, so external withdrawals, fee drift, or accounting bugs would go unnoticed while trading continues.

This is the gate between "dry-run validated" and "small live-money test" in the agreed rollout plan.

## What Changes

- **Order pre-flight (live only)**: before submitting, round the amount/price to the exchange's precision rules (`amount_to_precision`/`price_to_precision`) and reject orders below the market's minimum amount or minimum cost with a clear log message. Dry-run (`SimulatedExchangeService`) overrides order placement and is unaffected.
- **Raw response logging (live only)**: log the complete ccxt order response for every live order placement.
- **Fill accuracy**: record trades and update positions from the actual `filled` amount and exchange-reported `cost`; record the exchange-reported `fee_currency` (fallback: quote asset). Applies to the market/limit path and the TWAP/VWAP execution paths.
- **Balance reconciliation (live only)**: periodically (default every 300s) compare the exchange account's actual balances against the aggregate expectations of all running live bots (virtual cash in quote currency, open position amounts in base currencies). On insufficiency beyond a tolerance (default 1%), write a warning log and create an `Alert` (visible in the existing alerts UI/API). Reconciliation never blocks trading in this version — it alerts.

Out of scope: pending limit-order lifecycle polling (no strategy currently emits limit orders), slippage/fill-realism modeling (pending `add-trading-safety-boundaries`), auto-pause on reconciliation failure.

## Impact

- Affected specs: `live-order-execution` (new), `balance-reconciliation` (new)
- Affected code:
  - `backend/app/services/exchange.py` (pre-flight precision/limits, raw response logging)
  - `backend/app/services/trading_engine.py` (fee currency, filled amounts, TWAP/VWAP recording, reconciliation hook in bot loop)
  - `backend/app/models/alert.py` (no schema change; new `alert_type` value `balance_reconciliation`)
- Affected tests: new unit tests for pre-flight rejection/rounding, fee-currency recording, partial-fill position updates, reconciliation alerting.
