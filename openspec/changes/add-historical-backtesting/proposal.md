# Change: Add historical backtesting foundation

## Why
Live dry-run is not validation: it only shows what a strategy does going forward, slowly, on whatever
market regime happens to occur next. There is no way today to answer "how would this strategy have
performed across the last five years of BTC/SOL data" or "is this strategy's expectancy actually positive
before we let it touch real capital." This change builds the measuring system — a deterministic historical
backtester that replays real candle data through the exact same strategy decision code used live, with a
realistic (fee/spread/slippage, no-lookahead) execution model. It does not change, tune, or validate any
strategy's profitability; it only makes profitability measurable.

## What Changes
- Add `backend/app/backtesting/`: a backtest engine that feeds historical candles through
  `TradingEngine._get_strategy_executor(name)` — the same dispatch used by live trading — so strategy alpha
  logic is reused, not duplicated. Tracks full performance metrics, per-trade export, and a buy-and-hold
  benchmark. Deterministic: identical (data, strategy, parameters, date range) always produces identical
  output.
- Add `CsvHistoricalDataProvider` implementing a `HistoricalDataProvider` interface: dynamically discovers
  exchanges/symbols/timeframes under `data/backtest/` (no hardcoded names), parses the
  `Unix,Date,Symbol,Open,High,Low,Close,Volume <ASSET>,Volume <ASSET>,tradecount` CSV format with dynamic
  volume-column detection, merges and validates multi-file directories (duplicate/missing timestamp
  detection), and resamples 1m candles into 5m/15m/1h/4h/1d.
- Add a backtest execution model: default 0.1%-per-side fee, configurable spread and slippage, strict
  no-lookahead sequencing (decision made on candles `0..N`, any resulting order fills at candle `N+1`'s
  open).
- Introduce a single, minimal seam required to replay strategy decision code deterministically against
  historical timestamps instead of the wall clock (see design.md) — this is the only change to
  `trading_engine.py` itself, and it changes no strategy behavior in live mode.
- Report (not fix): remaining places where strategy indicator math depends on tick cadence rather than
  candle boundaries (see design.md "Tick vs. Candle Audit"). Per instructions, these are not rewritten in
  this change.

## Impact
- Affected specs: `historical-backtesting` (new), `csv-historical-data-provider` (new),
  `backtest-execution-model` (new)
- Affected code (new): `backend/app/backtesting/` (engine, portfolio, metrics, CSV provider, resampler,
  execution model), `data/backtest/` (already present, now has a consumer)
- Affected code (existing, minimally touched): `backend/app/services/trading_engine.py` — no logic changes;
  the backtester patches the module-level `datetime` symbol it already imports (`from datetime import
  datetime`) from outside the module for the duration of a backtest run. No strategy method is edited.
- Depends on: `fix-strategy-integrity` (companion proposal) for the `owning_strategy` field on `Position`,
  which the backtester's simulated position bookkeeping reuses rather than inventing a parallel ownership
  mechanism, and for `funding_carry` being absent from the strategies the backtester needs to support.
- Non-goals: rewriting strategy indicator math (EMA/ATR/Bollinger tick-vs-candle semantics), adding new
  strategies, tuning parameters, downloading data from exchanges, or claiming any strategy is profitable.
