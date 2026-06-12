# Funding Carry Strategy & Funding-Rate Diagnostic

This document describes the funding-rate extension: a **read-only diagnostic**
(Phase 1) and the **`funding_carry` strategy** (Phase 2). It was deliberately
built as a minimal extension of the existing single-symbol spot architecture —
no new architectural layer, no multi-leg hedging, no basis/market-neutral
trading, and no redesign of the trading engine.

## Scope & non-goals

`funding_carry` is a **long-only spot** strategy. It uses perpetual **funding
rates as a signal**, not as a yield source. It does **not**:

- short, hold a perpetual position, or hedge;
- run a delta-neutral basis trade (long spot + short perp);
- attempt latency-sensitive arbitrage.

True funding *harvesting* requires a perpetual leg, which is intentionally out
of scope. What this strategy harvests instead is **better entries**: it only
buys spot when funding indicates the perp market is neither over-crowded-long
(squeeze risk) nor in a falling-knife regime, **and** the price trend is
favourable.

## Phase 1 — Funding-Rate Diagnostic

Purpose: validate whether a funding edge plausibly exists on the exchange before
trading.

- **Service:** `app/services/funding_diagnostic.py`
  - `compute_funding_stats(rates, interval_hours)` — pure statistics (mean,
    median, min/max, stdev, positive/negative share, annualised mean).
  - `FundingRateDiagnostic.analyze(...)` — fetches funding history via the
    existing `ExchangeService`, estimates round-trip cost via the existing
    `ExecutionCostModel`, and reports net funding, breakeven periods, the share
    of profitable windows, best/worst periods, and a viability verdict.
- **CLI:** `python -m scripts.run_funding_diagnostic --symbols BTC/USDT ETH/USDT`
  - Uses the public market-data API (no credentials), like dry-run mode.
  - Places no trades.

Spot pairs carry no funding, so the diagnostic maps each spot symbol to its
linear perpetual (`BTC/USDT` → `BTC/USDT:USDT`) via
`ExchangeService.to_swap_symbol`.

### Exchange data availability (MEXC)

`ccxt` reports MEXC supports both `fetchFundingRate` and
`fetchFundingRateHistory`, and live runs return ~200+ windows of 8h funding for
majors. **However**, observed funding on BTC/ETH is very small — on the order of
0.0003–0.0006% per 8h window (~0.4–0.7% annualised), with round-trip spot fees
near 0.2%. Net funding over a few-period hold is therefore **negative**, and the
diagnostic returns "NO CLEAR EDGE". This is exactly why `funding_carry` treats
funding as a *filter*, not a *yield*.

## Phase 2 — `funding_carry` Strategy

Implemented as `TradingEngine._strategy_funding_carry` and registered in
`_get_strategy_executor` alongside the other strategies. It reuses the existing
tick-based regime detector (`_detect_market_regime`), price-history helpers,
per-bot state pattern, risk/validation pipeline, execution path, and logging.

**Entry (all required):**
1. Mean funding over `funding_lookback_periods` is inside the favourable band
   `[min_funding_rate, max_funding_rate]`.
2. Market regime `trend_state` is in `allowed_regimes`.
3. No open position, not in cooldown, sufficient balance.

**Exit (any):** funding leaves the band, or the regime is no longer favourable.

Funding is fetched via `ExchangeService.get_funding_rate_history` and cached
per-bot for `funding_refresh_seconds` to avoid hitting the API on every 1s loop.
If funding data is unavailable, the strategy **holds** (never trades on missing
data).

### Configuration

Per-bot `strategy_params` (registered in `app/routers/config.py`, validated by
`validate_funding_carry_params`). No operational values are hardcoded.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_funding_rate` | `-0.0005` | Lower bound of favourable funding band (fraction per interval) |
| `max_funding_rate` | `0.0005` | Upper bound of favourable funding band |
| `funding_lookback_periods` | `3` | Funding windows to average (3 ≈ 1 day at 8h) |
| `allowed_regimes` | `["trend_up"]` | Favourable trend regimes for entry |
| `max_allocation_percent` | `20` | Max % of balance per position |
| `cooldown_seconds` | `300` | Wait after exit before re-entry |
| `funding_refresh_seconds` | `300` | Funding-rate cache TTL |

## Files

**Added:** `app/services/funding_diagnostic.py`,
`scripts/run_funding_diagnostic.py`, `tests/test_funding_diagnostic.py`,
`tests/test_funding_carry_strategy.py`, `FUNDING_CARRY.md`.

**Modified:** `app/services/exchange.py` (FundingRate + funding methods),
`app/services/trading_engine.py` (strategy + validation + registration),
`app/routers/config.py` (strategy registration), `README.md`.

## Testing

```bash
cd backend
pytest tests/test_funding_diagnostic.py tests/test_funding_carry_strategy.py
```
