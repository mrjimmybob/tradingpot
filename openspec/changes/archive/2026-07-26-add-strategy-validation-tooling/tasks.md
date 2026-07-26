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
- [x] 2.1 Split a historical date range into rolling windows; measure the
      SAME operator-supplied fixed parameters on each window (no train-time
      search — the window split only makes the samples independent).
      `app/backtesting/validation/walk_forward.py`: `plan_windows()` tiles a
      bounded span into successive windows (inclusive spans, so window k+1
      starts exactly 1ms after window k ends; the final window is clipped to
      the requested end so the windows cover the range with no overhang).
      `step_ms` defaults to `window_ms` — non-overlapping, i.e. genuinely
      independent samples. A smaller step is allowed but reported as
      non-independent; a step *larger* than the window is rejected outright
      because it would leave unmeasured gaps in the requested range.
      `run_walk_forward()` measures one `FixedConfig` on every window.
- [x] 2.2 Report per-window metrics (expectancy, win rate, profit factor,
      drawdown, trade count) side by side so consistency (or its absence)
      across time is visible; state the sample-size limitation honestly.
      `format_walk_forward_report()` emits the per-window table *and* its
      limitations from one function, so the numbers are hard to copy anywhere
      without the caveats. Consistency is reported as
      `not_assessable`/`consistently_positive`/`consistently_negative`/`mixed`
      rather than a boolean — see the design note below. Limitations cover:
      < 5 windows, < 30 trades, windows that produced no trades, windows that
      could not be measured at all, window overlap, cold-start warm-up,
      modelled-not-observed fees, unrealised mark-to-market in Return%/MaxDD%
      when a window has no closed round trips, and an explicit "this is a
      measurement, not a profitability claim".
- [x] 2.3 CLI entry point (read-only; prints/writes a report, changes nothing).
      `python -m app.backtesting.validation.cli` (also runnable as
      `python -m backend.app.backtesting.validation.cli` from the repo root),
      mirroring `app/backtesting/run.py`'s flags plus `--window-days` /
      `--step-days`. Loads the whole range once via the engine's own loader so
      the timeframe-resampling fallback behaves identically to a hand-run
      backtest, then windows it in memory. Prints only — a test snapshots the
      data root's file sizes and mtimes before and after a run and asserts
      they are unchanged.
- [x] 2.4 Tests: deterministic measurement over a synthetic candle fixture with
      a known result; a fixture whose edge is present in one window and absent
      in others is correctly shown as inconsistent (NOT "fixed" — just shown).
      `backend/tests/test_validation_walk_forward.py` (35) and
      `backend/tests/test_validation_cli.py` (18). The inconsistent-edge case
      is driven by a scripted stand-in for `BacktestEngine` returning
      exactly-known per-window results: the claims under test are about
      aggregation and reporting ("every window used identical parameters", "a
      sign flip is reported, not repaired"), and only known per-window numbers
      can prove them. Coaxing a real strategy into a chosen win/loss pattern
      would test the strategy, not this layer, and would break whenever the
      strategy changed. Real-engine tests then cover the integration
      end-to-end: determinism run-to-run, windows partitioning the candle
      series exactly once, and a single full-range window reproducing a
      directly-run backtest metric for metric.

### Design notes for Section 2 (decisions made during implementation)
- **Consistency is not a boolean.** The first implementation returned
  `is_edge_consistent: bool`, which called an all-losing strategy
  "inconsistent" — it is in fact perfectly consistent, and perfectly bad —
  while a bare `True` would have read as good news for either sign. Replaced
  with a four-valued `consistency` plus a narrowly-named
  `has_positive_edge_in_every_traded_window`.
- **Windows are measured cold; there is no train/warm-up span.** design.md
  left window sizing open and mentioned a "train" span. Since nothing is
  trained, a separate train span would only serve as indicator warm-up — and
  excluding its trades would mean recomputing metrics outside the engine,
  which risks drifting from what `BacktestEngine` itself reports. Each window
  is therefore measured cold and the warm-up cost is stated as a limitation
  instead of being silently absorbed. `--window-days` defaults to 180 so
  warm-up is a small fraction of each window.
- **Windows that cannot be measured are recorded, not dropped.** A skipped
  window (fewer than 2 candles in range) appears in the report and in the
  limitations; silently dropping them would shrink the denominator and present
  a partial run as a complete one.

## 3. Regime-Conditioned Reporting (measurement)
- [x] 3.1 Classify each trade's entry regime using the canonical detector
      (post `fix-regime-detection-consistency`).
      `app/backtesting/validation/regime.py`: `build_regime_timeline()` walks
      the candle series forward once, driving
      `TradingEngine._detect_market_regime_bar_based` — the exact bound method
      `_strategy_auto` calls — the same way `_strategy_auto` drives it: the
      same 100-bar rolling window, the same 20-bar minimum before any regime
      is reported, and the previous regime carried forward so the 3-bar
      persistence hysteresis sees an identical input sequence. A test asserts
      the default detector *is* that function (`detector.__func__ is
      TradingEngine._detect_market_regime_bar_based`), so a second ad hoc
      classifier cannot quietly replace it.
      Labels are causal: the regime at candle *i* uses candles 0..i only,
      proved by a test that truncates the series and asserts no already-
      assigned label changes. Candles before the 20-bar minimum are
      `UNCLASSIFIED`, never "flat" — labelling them with a real regime would
      invent data.
- [x] 3.2 Bucket `BacktestResult.trades`/`equity_curve` by regime and compute
      per-regime expectancy/win-rate/drawdown.
      `bucket_trades_by_regime()` groups trades by the regime at each trade's
      ENTRY and routes every trade metric through the engine's own
      `compute_result`, so no number in this report has a second definition
      that could drift. Buckets exist for every regime the *market* was in,
      not only those traded, so "never entered during a downtrend" shows as an
      empty row rather than a missing one; each bucket carries its candle
      exposure so a row's weight is visible.
      Per-regime drawdown is computed per contiguous stretch of that regime
      and the worst taken — concatenating disjoint stretches would carry a
      peak from March into a trough in September. Each stretch is anchored on
      the equity mark immediately preceding it so a drop occurring on the move
      *into* a regime is attributed to it instead of falling into the crack
      between two stretches; the anchor is one point, not a running peak, so
      peaks still cannot leak between disjoint stretches. Both properties have
      dedicated tests.
- [x] 3.3 CLI output / report summarising per-regime performance per strategy
      (read-only).
      Two tables after the walk-forward report: a trend-only rollup (keeps
      per-bucket samples large enough to read) and the full
      trend/volatility/liquidity regime (what `_is_strategy_eligible` actually
      gates on). Trades are pooled across windows because a per-window *and*
      per-regime split would leave a handful of trades per cell; pooling is
      only sound while windows are disjoint, so an overlapping run states
      plainly that trades are counted more than once. `--skip-regime-report`
      opts out (labelling costs ~one detector pass per candle).
- [x] 3.4 Tests: regime bucketing correctness against a synthetic fixture with
      known regime transitions.
      `backend/tests/test_validation_regime.py` (23 tests). Bucketing is
      proved against a synthetic series with deliberately constructed
      transitions (warm-up → "up" → "down") driven by a scripted detector, so
      a failure points at this module rather than at a change in the live
      detector's thresholds — the detector's own behaviour is covered by its
      tests. Real-engine tests then confirm every trade of an actual
      measurement lands in exactly one bucket and the per-bucket P&L
      reconstructs the measurement's own total.

## 4. Validated Measurement Record
- [x] 4.1 Aggregate the out-of-sample measurements (Section 2) into the
      framework's `EdgeEstimate` shape (expectancy, win_rate, profit_factor,
      sample_size), constructed with `VALIDATED_EDGE_SOURCE` — produced and
      REPORTED only. Do NOT wire it into any live `StrategyProposal`
      (`expected_edge_estimate` stays `None` at runtime until a separate change
      wires it).
      `app/backtesting/validation/edge_record.py`:
      `build_validated_edge_record()` pools the walk-forward windows' trades
      and routes them through the engine's own `compute_result`, so expectancy,
      win rate and profit factor are the same quantities computed the same way
      as everywhere else in this tooling. `win_rate` is converted from
      `BacktestResult`'s percentage to the fraction `EdgeEstimate` validates.
      `ValidatedEdgeRecord` wraps the estimate with the provenance a human
      needs (configuration fingerprint, range, window counts, consistency) and
      the caveats the four-number contract has no room for.
      **The record is refused, not caveated, when it would mislead**
      (`edge_record_blockers()`): a single measured window, trades confined to
      a single window (in-sample by another name), overlapping windows (would
      pool shared trades twice), no trades, or windows that did not share one
      configuration. Producing a "validated" number from one window would
      launder a single backtest into a stamp of approval — the exact failure
      this change exists to prevent.
- [x] 4.2 Tests: the record is constructible/valid for a measured strategy; a
      test confirms no runtime proposal path is modified by this change.
      `backend/tests/test_validation_edge_record.py` (21 tests). Pooled
      arithmetic is hand-checked against scripted per-window results; each
      blocker has its own test; and `TestRuntimeProposalsAreUnaffected` walks
      the AST of **every module under `app/`** asserting that no code outside
      `app/backtesting/validation/` constructs an `EdgeEstimate` and that no
      call passes a non-`None` `expected_edge_estimate` (a pass-through of an
      already-`None` field, as the Auto committee's read-only view does, is
      correctly distinguished from populating one). Also confirms the schema
      still defaults the field to `None` and that the framework still rejects a
      forged source marker — the gate that makes this record legitimate.

## 5. Baseline Measurement Run
- [x] 5.1 Once Sections 1-4 are usable, measure all 6 strategies via
      walk-forward + regime-conditioned reporting across the full available
      history (2020-2026) and record the results as the first objective,
      out-of-sample baseline this project has — a report a human reviews, with
      its sample-size limitations stated. No parameters are changed as a result.
      `app/backtesting/validation/baseline.py` + `--strategy all` on the CLI:
      loads the candle series once and measures all six concrete strategies
      over identical windows and execution model, then prints a cross-strategy
      summary. `auto_mode` is excluded (it selects among the six, so measuring
      it would double-count whichever it picked). No parameters are supplied,
      so each strategy falls back to its own internal defaults — which is
      precisely what "never empirically validated" referred to.
      The summary is a **comparison, not a ranking**: rows stay in declaration
      order and are never sorted by result, with a test asserting that even
      when the last-measured strategy has the best numbers. A sorted table
      would be a recommendation wearing a table's clothes, and these sample
      sizes would not support one.
      **Results recorded in `STRATEGY_BASELINE_2020_2026.md`** (repo root), with
      full raw reports at `baseline-raw-4h.txt` / `baseline-raw-1d.txt`
      alongside this change. Run over `binance/BTCUSDT`, 2020-01-01 →
      2026-01-01, 13 × 180d non-overlapping windows, at both 4h and 1d.
      Headline findings: (1) **every** strategy that closed trades came back
      `mixed` — no strategy's edge is stable across the 13 windows at either
      resolution; (2) the headline number depends heavily on timeframe —
      `trend_following` measures −26.76/trade at 1d and +24.43 at 4h, changing
      sign on identical data and parameters; (3) `dca_accumulator`,
      `adaptive_grid` and `dip_recovery` close zero round trips (they scale in
      and out without flattening), so expectancy is not a meaningful measure
      for them and the tooling refuses a record rather than printing a
      misleading zero; (4) per-regime, `mean_reversion` and `trend_following`
      both earn in up/flat and lose in down, while `volatility_breakout`
      inverts that. **No parameter was changed as a result.**

### Design notes for Section 5
- **Zero closed trades is not inactivity.** The baseline surfaced that the
  round-trip counter under-reports strategies that scale in and out without
  fully flattening: `adaptive_grid` returned −1.64% in a window where
  buy-and-hold returned +26.89%, while showing 0 trades. The summary now says
  this explicitly and points at the Return%/MaxDD% columns, and the
  walk-forward limitations flag unrealised mark-to-market wherever it occurs.
  Measuring these three strategies properly needs return/drawdown-vs-benchmark
  measures the tooling does not yet provide — recorded as a gap, not silently
  papered over.
- **Both 4h and 1d are reported**, because the disagreement between them is one
  of the findings rather than an inconvenience to be resolved by picking one.

## Explicitly Out of Scope (deferred to a future, separate change)
- Parameter optimisation / search / sweep of any kind, and any objective that
  ranks parameter sets (grid/random/latin-hypercube, drawdown-constrained
  ranking, etc.).
- Writing optimiser- or measurement-chosen parameters back into `strategy_params`
  for any bot or config.
- Wiring the validated `EdgeEstimate` into live `StrategyProposal.expected_edge_
  estimate` at runtime (this change only produces/reports the record).
