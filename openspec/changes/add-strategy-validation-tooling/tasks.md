## 1. Parameter Optimization
- [ ] 1.1 Define a parameter-space spec format per strategy (ranges/choices
      per param, using each strategy's existing `params.get(...)` defaults
      as the search space's center).
- [ ] 1.2 Implement grid and random search over `BacktestEngine.run_candles`,
      with no changes to the engine itself.
- [ ] 1.3 Implement the default objective: expectancy subject to a
      configurable max-drawdown constraint.
- [ ] 1.4 CLI entry point (e.g. `python -m app.backtesting.optimize`)
      printing top-N parameter sets with their metrics.
- [ ] 1.5 Tests: deterministic small-grid search over a synthetic candle
      fixture with a known best parameter set.

## 2. Walk-Forward Validation
- [ ] 2.1 Implement rolling train/validate window splitting over a
      historical date range.
- [ ] 2.2 Run the optimizer (Section 1) on each train window; score the
      chosen parameters on the held-out validate window only.
- [ ] 2.3 Report in-sample vs out-of-sample metrics side by side per window,
      flagging windows where out-of-sample expectancy is materially worse.
- [ ] 2.4 Tests: a synthetic fixture where a parameter set is deliberately
      overfit to noise in the train window, and the harness must flag it.

## 3. Regime-Conditioned Reporting
- [ ] 3.1 Classify each trade's entry regime using the canonical detector
      (post `fix-regime-detection-consistency`).
- [ ] 3.2 Bucket `BacktestResult.trades`/`equity_curve` by regime and
      compute per-regime expectancy/win-rate/drawdown.
- [ ] 3.3 CLI output or report file summarizing per-regime performance per
      strategy.
- [ ] 3.4 Feed findings back into `_strategy_auto`'s scoring table
      (`trading_engine.py` `_get_strategy_capabilities`) as documented,
      data-backed `allowed_regimes`/priority values instead of guesses —
      track as a follow-up task, not a blocker for this change's
      completion.
- [ ] 3.5 Tests: regime bucketing correctness against a synthetic fixture
      with known regime transitions.

## 4. Baseline Validation Run
- [ ] 4.1 Once Sections 1-3 are usable, re-run all 6 strategies through
      walk-forward + regime-conditioned reporting across the full available
      history (2020-2026) and record the results as the first real,
      validated baseline this project has ever had.
