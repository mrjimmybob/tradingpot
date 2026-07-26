## Context
`add-strategy-validation-tooling` measures strategies by closed round trips:
expectancy, win rate, profit factor, and a validated `EdgeEstimate` built from
them. That instrument is undefined for a strategy that never closes a round
trip, and three of six strategies never do. The equity curve those runs already
produce (`BacktestResult.equity_curve`, marked to market on every candle) holds
the answer; nothing yet reads it against a reference.

The engine already computes one benchmark — `buy_and_hold_return_pct`, a single
scalar (`engine.py:267`). It is not enough: a scalar return cannot be compared
on drawdown, and buy-and-hold alone is an unfair reference for a strategy that
deploys capital gradually.

## Goals / Non-Goals
- Goals:
  - Make every strategy measurable, including those with zero closed trades.
  - Compare on **both** return and drawdown, against benchmarks that pay the
    same costs the strategy paid.
  - Report exposure, so under-deployment is never mistaken for poor selection.
- Non-Goals:
  - Optimisation, parameter search, or strategy selection of any kind.
  - Changing `BacktestEngine`, `BacktestPortfolio`, `results.py`, or a strategy.
  - Risk-free-rate-based measures (Sharpe/Sortino). They require a rate series
    and an annualisation convention this project has not settled, and would add
    assumptions to a tool whose value is having few.
  - Fixing the partial-exit `TradeRecord` simplification in `portfolio.py`. That
    is a real limitation, but changing the portfolio would alter every existing
    measurement and invalidate the baseline just recorded. It is called out in
    the report instead, and belongs to its own change.

## Decisions
- Decision: benchmarks are **pure functions over the candle series**, not
  strategies run through the engine.
  - Why: deterministic, trivially testable, and free of strategy dispatch, DB
    setup, and warm-up. A benchmark whose value depended on the engine's replay
    would be as hard to trust as the thing it benchmarks.
- Decision: the benchmark set is **buy-and-hold** and **periodic DCA**.
  - Why: buy-and-hold answers "could I have just held?" — the question any
    strategy must beat to justify its complexity. Periodic DCA answers it
    *fairly for an accumulator*: it deploys on the same gradual schedule, so the
    comparison isolates timing and sizing from mere exposure. It is also already
    this project's stated long-term accumulation benchmark, so measuring against
    it makes the existing intent concrete rather than introducing a new idea.
- Decision: benchmarks **pay the strategy's own fee model** — buy-and-hold pays
  one entry fee, periodic DCA pays one per instalment.
  - Why: a cost-free benchmark is not a benchmark, it is a handicap. Comparing a
    fee-paying strategy against a frictionless ideal would understate every
    strategy by exactly the fees it could not have avoided.
- Decision: exposure is reported as **realised beta** of the strategy's
  per-candle equity returns to the asset's, not as a true cash/base split.
  - Why: the portfolio's equity curve records only total equity
    (`portfolio.py:94`), so the split is not recoverable post-hoc, and
    reconstructing it would mean modifying the engine — which this change
    forbids. For a long-only spot strategy, beta to the asset is a faithful
    proxy for average fractional deployment, and it is computable from data
    already recorded.
- Decision: report **return per unit of max drawdown**, not a Sharpe-family
  ratio.
  - Why: it needs no risk-free rate and no annualisation convention, and
    drawdown is the risk measure this codebase already computes everywhere.
- Decision: benchmark-relative measures are reported for **every** strategy, not
  only zero-trade ones.
  - Why: a strategy that closes trades can still lose to buy-and-hold, and that
    is worth knowing. Restricting the measure to the strategies that embarrassed
    the old instrument would be choosing where to look.

## Risks / Trade-offs
- A benchmark comparison invites ranking. Mitigation: the same discipline the
  existing summary already enforces — declaration order, never sorted by result,
  with the "comparison, not a ranking" statement carried alongside, and a test
  asserting it.
- Periodic DCA has a cadence parameter, and a different cadence yields a
  different benchmark. Mitigation: the cadence is reported with every result and
  defaults to a single documented value; it is a stated property of the
  benchmark, not something searched over. **Tuning the cadence to flatter a
  strategy would be optimisation and is forbidden.**
- Beta is a proxy, not measured exposure, and is unreliable when a window has
  very few candles or near-zero asset variance. Mitigation: report it as a
  proxy, state the limitation, and suppress it where the denominator is
  degenerate rather than printing a meaningless number.
- Excess return over a benchmark is not skill when exposure differs.
  Mitigation: exposure is printed on the same row, and the limitation says so.

## Migration Plan
Additive and read-only; nothing to roll back beyond not running the CLI. Existing
measurements are unchanged because no existing module is modified.

## Open Questions
- None blocking. The periodic-DCA cadence default is a run-time parameter, to be
  documented rather than settled here.
