# Change: Add strategy validation tooling (measurement and explanation only)

## Why
Every strategy default in this codebase was set once, by hand, and never
empirically validated: backtests only ever cover one fixed date range, and no
out-of-sample check exists anywhere in the repo. This is why
"`dca_accumulator` is profitable" cannot currently be distinguished from "the
backtest window happened to be a bull market", and why
`add-strategy-decision-framework`'s `expected_edge_estimate` is always `None`
(it may only be populated from validated results, and no tool produces them
yet). The gap is objective measurement — not tuning.

## Premise (scope boundary — binding)
The purpose of this change is **to measure strategies objectively and explain
the results — not to optimise strategies automatically.**
- The tool MUST NOT search a parameter space, tune parameters, or select
  "better" parameters.
- The tool MUST NOT write `strategy_params` back to any bot or config, ever.
- Measurement is kept strictly separate from optimisation.
- Any optimisation or parameter search is a **future, separate change** and is
  explicitly out of scope here.

## What Changes
- Add **out-of-sample (walk-forward) measurement of a FIXED strategy
  configuration**: split a historical range into rolling windows and measure
  the strategy's *given* parameters on each held-out window, exposing whether
  its edge is stable out-of-sample. No parameter search of any kind — the
  same fixed params are measured on every window.
- Add **regime-conditioned reporting**: bucket a `BacktestResult`'s trades and
  equity curve by the canonical regime classification at each trade's entry
  and report per-regime expectancy/win-rate/drawdown — so per-regime claims
  are checkable against data.
- Add a **validated measurement record** (expectancy, win rate, profit factor,
  sample size) produced only from these out-of-sample measurements — the
  shape `add-strategy-decision-framework`'s `EdgeEstimate` already defines and
  gates behind `VALIDATED_EDGE_SOURCE`. This change *produces and reports* that
  record; wiring it into live proposals is left to a separate change.
- Everything is a **read-only report a human reviews**. The tool prints/writes
  a report; it changes no strategy, no engine, and no production config.

## Impact
- Affected specs: strategy-validation-measurement (new; the measurement/
  optimisation boundary + validated measurement record), walk-forward-validation
  (measurement of a fixed config), regime-performance-reporting.
- Removed from scope vs the original proposal: parameter-optimization
  (parameter search / drawdown-constrained objective ranking) — deferred to a
  future, separate change per the Premise above.
- Affected code (additive; no changes to existing engine internals): new
  read-only modules alongside `backend/app/backtesting/engine.py`/`run.py`,
  calling `BacktestEngine.run_candles` unmodified.
- Depends on: `fix-regime-detection-consistency` (regime-conditioned reporting
  needs a trustworthy regime classification to bucket by).
