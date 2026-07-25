## Context
`BacktestEngine.run_candles` (`backend/app/backtesting/engine.py`) already
returns a clean `BacktestResult` (expectancy, drawdown, win rate, equity
curve, trade list — `backend/app/backtesting/results.py`) for a single
strategy/param/date-range combination, with a proven no-lookahead guarantee
(`test_decision_never_sees_a_future_candle`,
`backend/tests/test_backtesting_engine.py`). The missing piece is not a search
loop — it is **objective, out-of-sample measurement and explanation** of a
strategy as it is actually configured. `add-strategy-decision-framework`
defined an `EdgeEstimate` (expectancy/win_rate/profit_factor/sample_size,
constructible only with `source == VALIDATED_EDGE_SOURCE`) precisely so that a
measurement tool — this one — can produce validated numbers a strategy may not
self-compute.

## Premise (binding)
Measure, don't optimise. This tool measures a **fixed** strategy configuration
and explains the result. It performs **no parameter search, no tuning, no
selection of a "better" parameter set, and never writes parameters back**
anywhere. Optimisation is a separate, future change. Every requirement and
task below is written to keep measurement and optimisation strictly apart.

## Goals / Non-Goals
- Goals:
  - Measure a strategy's *given* parameters out-of-sample (rolling windows) so
    a single lucky window can't masquerade as an edge.
  - Explain the result: per-window and per-regime breakdowns, plus honest
    sample-size/limitation notes, not a single headline number.
  - Produce the validated `EdgeEstimate`-shaped record from those measurements
    (report only; wiring into live proposals is a separate change).
- Non-Goals (explicit):
  - **Parameter optimisation / search / tuning of any kind** — deferred to a
    future, separate change.
  - Writing `strategy_params` to any bot or config.
  - Machine-learning signal generation; multi-symbol/portfolio backtesting.
  - Changing `BacktestEngine`, any strategy, or any production trading path.

## Decisions
- Decision: the tool is a **read-only measurement wrapper** around
  `BacktestEngine.run_candles`, run from a CLI, that never mutates engine,
  strategy, or config state.
  - Why: preserves the engine's proven no-lookahead / no-logic-change
    guarantees; keeps the measurement/optimisation boundary structurally
    obvious (there is simply no code path that changes a parameter).
- Decision: walk-forward here means **out-of-sample measurement of one fixed
  config**, not train-then-optimise-then-validate. Each rolling window runs
  the SAME operator-supplied parameters; the "train" window is not used to
  pick anything — it exists only so windows are independent and the report can
  show consistency (or its absence) across time.
  - Why: this is the honest measurement the premise asks for. Introducing a
    train-time search would be optimisation, which is out of scope.
- Decision: the validated record is the framework's `EdgeEstimate` shape,
  constructed with `VALIDATED_EDGE_SOURCE`, aggregated across the out-of-sample
  windows (not a single in-sample fit).
  - Why: it is the one contract already designed for "validated, not
    self-computed" numbers; producing it here is measurement, and it makes the
    tool's output directly consumable by a later, separate wiring change.
- Decision: regime-conditioned reporting reuses the canonical regime detector
  (`fix-regime-detection-consistency`), classifying regime at each trade's
  entry timestamp from the already-loaded candle series — a post-hoc bucketing
  of existing `BacktestResult.trades`/`equity_curve`.
  - Why: no coupling to replay-time state; it is pure measurement of results
    already produced.

## Risks / Trade-offs
- With only ~6 years of history (2020-2026), the number of independent
  out-of-sample windows is limited — the report SHALL state this and avoid
  overstating statistical confidence, never presenting a small-sample result
  as a strong edge.
- Long measurement runs must stay observable: reuse the CSV-loader's
  progress-reporting pattern (`backend/app/backtesting/progress.py`).
- Temptation to "just also rank a few parameter variants" — explicitly
  forbidden here; that is the future optimisation change. A test asserts the
  tool exposes no parameter-search / param-writeback path.

## Migration Plan
1. Ship the measurement boundary + walk-forward (fixed-config OOS) first.
2. Add regime-conditioned reporting (after `fix-regime-detection-consistency`).
3. Produce the validated `EdgeEstimate` record from the OOS measurements.
4. Run a baseline measurement of all six strategies as the first validated
   record this project has. None of this touches production trading paths — no
   rollback needed beyond not running the CLI.

## Open Questions
- Default rolling-window sizes (train/validate span, step) for the baseline
  run — pick sizes that yield at least a few independent windows over
  2020-2026 and document them; not a contract, a run-time parameter.
- Report format (stdout table vs a written artifact) — implementation detail;
  must in all cases be read-only.
