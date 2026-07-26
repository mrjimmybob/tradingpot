# Change: Measure accumulating strategies against benchmarks (return and drawdown)

## Why
The 2020-2026 out-of-sample baseline (`STRATEGY_BASELINE_2020_2026.md`) could
say **nothing** about three of the six strategies. `dca_accumulator`,
`adaptive_grid`, and `dip_recovery` closed zero round trips in all 13 windows at
both 4h and 1d, so expectancy, win rate, and profit factor — every metric the
validation tooling reports — were undefined for them, and
`add-strategy-validation-tooling` correctly refused to produce a validated
record rather than print a misleading zero.

They were not inactive. `adaptive_grid` returned **-1.64%** in a window where
buy-and-hold returned **+26.89%**, and its per-window drawdowns ranged from
2.88% to 60.74%. The measurement instrument was simply wrong for them, for two
distinct reasons:

1. `dca_accumulator` never sells by design — it is the project's reference
   accumulation strategy, and a never-sell strategy has no round trips to count.
2. `adaptive_grid` and `dip_recovery` scale out partially and rarely flatten
   completely, and the backtest portfolio only records a closed `TradeRecord` on
   a **full** close (`backend/app/backtesting/portfolio.py:65-73`, a documented
   simplification). Their realised P&L is real but invisible to the trade
   counter.

Either way, these strategies can only be judged on their **equity curve against
a benchmark**, not on closed trades. Without that, half this project's
strategies remain unmeasured — the exact gap the validation tooling was built to
close.

## Premise (scope boundary — binding, inherited)
This change inherits `strategy-validation-measurement`'s binding premise in
full: **measure objectively and explain — never optimise.**
- MUST NOT search a parameter space, tune parameters, or select a "better"
  parameter set or a "better" strategy.
- MUST NOT write `strategy_params` (or any parameter) anywhere.
- MUST NOT modify `BacktestEngine`, `BacktestPortfolio`, or any strategy.
- Benchmark comparison is **measurement, not selection**: the report compares
  what was measured and never recommends a strategy. Rows are never sorted by
  result.

## What Changes
- Add **benchmark equity curves** computed as pure, deterministic functions over
  the same candles a measurement ran on, paying the same fee model the strategy
  paid:
  - **buy-and-hold** — full capital deployed at the first candle, held. Answers
    "could I have just held?"
  - **periodic DCA** — capital deployed in equal instalments at a fixed cadence,
    never sold. The fair reference for an accumulator, and the project's own
    stated long-term benchmark for accumulation strategies.
- Add **benchmark-relative measures** per window: terminal return, max drawdown,
  return per unit of drawdown, excess return versus each benchmark, and the
  drawdown difference versus each benchmark.
- Add **realised exposure** (beta of the strategy's equity returns to the
  asset's), so a strategy that was merely under-deployed is not read as a bad
  one — the single most likely misreading of an accumulator versus buy-and-hold.
- Report these for **every** strategy, and make them the primary reading for
  strategies that close no round trips — replacing "0 trades, no record" with an
  actual measurement.
- Re-run the 2020-2026 baseline and record what the three previously unmeasured
  strategies actually did.

## Impact
- Affected specs: **benchmark-relative-measurement** (new capability).
- Affected code (additive only): new module(s) under
  `backend/app/backtesting/validation/`, plus report/CLI wiring. No change to
  `engine.py`, `portfolio.py`, `results.py`, or any strategy.
- Affected docs: `STRATEGY_BASELINE_2020_2026.md` gains the measurements it
  currently records as a gap.
- Depends on: `add-strategy-validation-tooling` (archived; supplies
  `Measurement`, `WalkForwardMeasurement`, and the read-only boundary this
  change is bound by).
