## Premise (binding for every task)
Measure objectively and explain — never optimise. No task below may search a
parameter space, tune parameters, pick a "better" parameter set, or write
`strategy_params` anywhere. Optimisation is a future, separate change.

## 1. Measurement Boundary (measurement ≠ optimisation)
- [x] 1.1 New read-only package `app/backtesting/validation/` that wraps
      `BacktestEngine.run_candles` unmodified. It takes a strategy + a FIXED
      parameter dict + a date range and returns measurements only; it exposes
      no parameter-search or parameter-writeback API.
      Implemented as `app/backtesting/validation/measurement.py`:
      `FixedConfig` (strategy + trading pair + params + starting balance),
      `MeasurementSpan` (inclusive ms date range, `None` = unbounded),
      `Measurement` (immutable projection of `BacktestResult` + provenance),
      `measure_fixed_config()` and `select_candles()`. `FixedConfig`
      deep-copies the operator's dict at construction and exposes it only via
      `MappingProxyType`; `params_for_run()` hands the engine a *fresh* deep
      copy per run, so no strategy mutation can leak across windows. Every
      `Measurement` carries a `params_fingerprint` (sha256 of canonical JSON),
      making "identical parameters on every window" mechanically provable
      rather than merely asserted in prose.
- [x] 1.2 Test: a structural/guard test asserting the package contains no
      parameter-search path and never writes `strategy_params` (no DB/bot/config
      write, no "try N param sets and rank" loop) — the measurement/optimisation
      separation is enforced, not just documented.
      `backend/tests/test_validation_measurement_boundary.py` (35 tests). Six
      AST detectors run against *every* module discovered by walking the
      package directory (so a module added later cannot be added *around* the
      guard): no write through an object to `strategy_params`/`params`, no
      `setattr(..., "strategy_params", ...)`, no import of any persistence
      layer (`sqlalchemy`, `app.models`, `app.core.config`, …), no file opened
      for writing, no definition or argument using optimisation vocabulary
      (`optimi*`/`tune`/`sweep`/`param_grid`/`candidate`/…), and only
      `engine.run_candles` ever called on the engine. A `TestTheGuardsActually
      Bite` meta-suite runs each detector against deliberately-offending
      synthetic source, so a silently-broken detector fails the suite instead
      of reporting a boundary it stopped checking. Behavioural half proves the
      same from outside: the operator's dict and candle list come back
      byte-identical, repeated measurement is deterministic, and every metric
      equals what `BacktestEngine` itself reported (no recomputation).

## 2. Walk-Forward Measurement (fixed config, out-of-sample)
- [ ] 2.1 Split a historical date range into rolling windows; measure the
      SAME operator-supplied fixed parameters on each window (no train-time
      search — the window split only makes the samples independent).
- [ ] 2.2 Report per-window metrics (expectancy, win rate, profit factor,
      drawdown, trade count) side by side so consistency (or its absence)
      across time is visible; state the sample-size limitation honestly.
- [ ] 2.3 CLI entry point (read-only; prints/writes a report, changes nothing).
- [ ] 2.4 Tests: deterministic measurement over a synthetic candle fixture with
      a known result; a fixture whose edge is present in one window and absent
      in others is correctly shown as inconsistent (NOT "fixed" — just shown).

## 3. Regime-Conditioned Reporting (measurement)
- [ ] 3.1 Classify each trade's entry regime using the canonical detector
      (post `fix-regime-detection-consistency`).
- [ ] 3.2 Bucket `BacktestResult.trades`/`equity_curve` by regime and compute
      per-regime expectancy/win-rate/drawdown.
- [ ] 3.3 CLI output / report summarising per-regime performance per strategy
      (read-only).
- [ ] 3.4 Tests: regime bucketing correctness against a synthetic fixture with
      known regime transitions.

## 4. Validated Measurement Record
- [ ] 4.1 Aggregate the out-of-sample measurements (Section 2) into the
      framework's `EdgeEstimate` shape (expectancy, win_rate, profit_factor,
      sample_size), constructed with `VALIDATED_EDGE_SOURCE` — produced and
      REPORTED only. Do NOT wire it into any live `StrategyProposal`
      (`expected_edge_estimate` stays `None` at runtime until a separate change
      wires it).
- [ ] 4.2 Tests: the record is constructible/valid for a measured strategy; a
      test confirms no runtime proposal path is modified by this change.

## 5. Baseline Measurement Run
- [ ] 5.1 Once Sections 1-4 are usable, measure all 6 strategies via
      walk-forward + regime-conditioned reporting across the full available
      history (2020-2026) and record the results as the first objective,
      out-of-sample baseline this project has — a report a human reviews, with
      its sample-size limitations stated. No parameters are changed as a result.

## Explicitly Out of Scope (deferred to a future, separate change)
- Parameter optimisation / search / sweep of any kind, and any objective that
  ranks parameter sets (grid/random/latin-hypercube, drawdown-constrained
  ranking, etc.).
- Writing optimiser- or measurement-chosen parameters back into `strategy_params`
  for any bot or config.
- Wiring the validated `EdgeEstimate` into live `StrategyProposal.expected_edge_
  estimate` at runtime (this change only produces/reports the record).
