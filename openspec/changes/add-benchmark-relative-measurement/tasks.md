## Premise (binding for every task)
Inherited from `strategy-validation-measurement`: measure objectively and
explain — never optimise. No task below may search a parameter space, tune a
strategy or benchmark parameter, rank or recommend a strategy, or write any
parameter anywhere. No task may modify `BacktestEngine`, `BacktestPortfolio`,
`results.py`, or any strategy.

## 1. Benchmark Curves
- [x] 1.1 New module `app/backtesting/validation/benchmarks.py`: pure,
      deterministic functions building benchmark equity curves from a candle
      series and an execution-cost model — no engine, no strategy dispatch, no
      replay loop.
- [x] 1.2 `buy_and_hold_curve()`: full capital deployed at the first candle's
      open, held to the last candle, paying one entry fee under the same fee
      model the measurement used.
- [x] 1.3 `periodic_dca_curve()`: capital deployed in equal instalments at a
      fixed cadence, never sold, each instalment paying the fee model. Cadence
      is an explicit, reported parameter with a documented default — never
      searched or tuned.
- [x] 1.4 Tests: hand-computable curves over a synthetic series (flat price,
      monotonic rise, monotonic fall); determinism; fees actually reduce the
      benchmark; a cadence longer than the span still deploys at least once;
      instalments sum to the starting balance.

## 2. Benchmark-Relative Measures
- [x] 2.1 Compute, from a `Measurement`'s recorded equity curve, the strategy's
      terminal return, max drawdown, and return per unit of max drawdown —
      reusing the engine's own drawdown computation, never a second
      implementation.
- [x] 2.2 Compute excess return and drawdown difference against each benchmark.
- [x] 2.3 Compute realised exposure as the beta of the strategy's per-candle
      equity returns to the asset's returns; report it as unavailable (not as a
      number) when the basis is degenerate — too few points, or zero asset
      variance.
- [x] 2.4 Tests: a strategy that closes zero round trips is fully measured; a
      strategy holding 100% of the time betas to ~1.0 and one holding nothing
      betas to ~0.0; a strategy identical to a benchmark shows ~zero excess;
      degenerate bases yield unavailable rather than a value.

## 3. Reporting And CLI
- [x] 3.1 A benchmark-relative report section per strategy: strategy vs each
      benchmark on return AND drawdown, with exposure and the benchmark
      parameters shown alongside.
- [x] 3.2 Limitations travel with the numbers: exposure is a proxy not a
      measured split; excess return is not skill when exposure differs; the
      partial-exit `TradeRecord` simplification in `portfolio.py:65-73` means
      closed-trade counts understate scale-out strategies; benchmark costs are
      modelled, not observed.
- [x] 3.3 Wire into the validation CLI for every strategy, and make it the
      primary reading for strategies that closed no round trips (replacing "0
      trades, no record" with an actual measurement).
- [x] 3.4 Cross-strategy summary keeps declaration order and the "comparison,
      not a ranking" statement, extended with the benchmark-relative columns.
- [x] 3.5 Tests: report contents; a zero-round-trip strategy is measured rather
      than dismissed; ordering never depends on results; the boundary guard
      tests (which walk every module in the package) still pass for the new
      modules.

## 4. Re-Run And Record The Baseline
- [x] 4.1 Re-run the 2020-2026 baseline with benchmark-relative measures for all
      six strategies at 4h and 1d.
- [x] 4.2 Update `STRATEGY_BASELINE_2020_2026.md` with what
      `dca_accumulator`, `adaptive_grid`, and `dip_recovery` actually did —
      closing the gap that document currently records as open — with sample-size
      and exposure limitations stated. No parameters are changed as a result.

### Discoveries made during implementation (material to how the tasks read)
- **`estimate_exposure` needed a near-zero variance guard, not `variance > 0`.**
  A constant-drift price series has *identical* per-candle returns, so its return
  variance is pure float noise (~1e-17) — genuinely positive, and meaningless.
  Dividing a near-zero covariance by it produced a beta of **-4.9e10** in a test
  fixture. Now guarded by a minimum return standard deviation
  (`MIN_RETURN_STDDEV_FOR_EXPOSURE`), with a regression test. The first test
  fixtures also had to be changed to vary their returns; a perfectly smooth
  series tested nothing, because real price data always varies.
- **`dip_recovery` never opens a position at all** with its shipped defaults —
  return +0.00%, drawdown 0.00%, exposure 0.00 in all 13 windows at both
  resolutions. The closed-trade instrument could not distinguish "never traded"
  from "traded constantly without flattening"; exposure makes it unmistakable.
  Its "beats buy-and-hold in 4 of 13 windows" is what holding cash does in a
  falling market, not skill. Flagged for investigation; **no parameter changed.**
- **`dca_accumulator` behaves like buy-and-hold** (exposure 0.97-1.00 at 4h),
  deploying fully and early over a 180d window rather than averaging in. It beats
  the weekly-DCA benchmark largely *because* it deploys faster — an exposure
  difference, not entry timing. Notable for a strategy whose stated role is being
  the project's clean accumulation reference.
- **Most strategies are barely deployed** (exposure ≤ 0.15 for `mean_reversion`,
  `trend_following`, `volatility_breakout`). This explains why all six strategies
  beat buy-and-hold in roughly the same 4 of 13 windows: those are the down
  windows, where being in cash wins regardless of what the strategy did.
- **Benchmarks are rebuilt per window**, not once over the whole range, so each
  window stays an independent sample; a range-spanning benchmark would carry
  information from outside the window into its comparison.

## Explicitly Out of Scope (separate, future changes)
- Fixing the partial-exit `TradeRecord` simplification in `portfolio.py`.
  Changing the portfolio would alter every existing measurement and invalidate
  the baseline just recorded; it is reported as a limitation here.
- Risk-free-rate measures (Sharpe/Sortino) — they need a rate series and an
  annualisation convention this project has not settled.
- Any optimisation, parameter search, or strategy selection.
- Wiring anything produced here into runtime decisions.
