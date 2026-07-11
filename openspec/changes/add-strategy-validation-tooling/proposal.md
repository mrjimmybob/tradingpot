# Change: Add strategy validation tooling (parameter optimization, walk-forward testing, regime-conditioned reporting)

## Why
Every strategy default in this codebase was set once, by hand, and never
empirically validated: backtests only ever cover one fixed date range, there
is no parameter search, and no out-of-sample check exists anywhere in the
repo (verified: zero hits for walk-forward/hyperopt/grid-search-style
tooling; `openspec/changes/add-historical-backtesting/design.md` explicitly
names parameter tuning as out of scope for that work). This is why defaults
drift from their own docstrings (`volatility_breakout`'s
`min_compression_bars` docstring says 20, the code ships 5) and why
"`dca_accumulator` is profitable" cannot currently be distinguished from "the
backtest window happened to be a supercycle bull market." Mature reference
bots (freqtrade's hyperopt, intelligent-trading-bot's rolling train/predict
pipeline) treat this as a first-class capability, not an afterthought.

## What Changes
- Add a parameter-sweep optimizer that wraps the existing
  `BacktestEngine.run_candles` (no changes to the engine itself) to search a
  strategy's parameter space and rank results by a configurable objective
  (default: expectancy subject to a max-drawdown constraint).
- Add a walk-forward harness that splits a historical range into rolling
  train/validate windows, runs the optimizer on each train window, and
  scores the resulting parameters on the held-out validate window only —
  flagging parameter sets that look good in-sample but fail out-of-sample.
- Add regime-conditioned backtest reporting: bucket an existing
  `BacktestResult`'s trades/equity curve by the regime detector's
  classification at each trade's entry, and report per-regime expectancy —
  turning `_strategy_auto`'s hardcoded priority/scoring table into something
  checkable against real per-regime performance.

## Impact
- Affected specs: parameter-optimization, walk-forward-validation,
  regime-performance-reporting
- Affected code (additive; no changes to existing engine internals):
  new modules alongside `backend/app/backtesting/engine.py` and `run.py`.
- Depends on: `fix-regime-detection-consistency` (regime-conditioned
  reporting needs a trustworthy regime classification to bucket by).
