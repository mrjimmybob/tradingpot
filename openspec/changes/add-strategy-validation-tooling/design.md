## Context
`BacktestEngine.run_candles` (`backend/app/backtesting/engine.py`) already
returns a clean `BacktestResult` (expectancy, drawdown, win rate, equity
curve, trade list — `backend/app/backtesting/results.py`) for a single
strategy/param/date-range combination, with a proven no-lookahead guarantee
(`test_decision_never_sees_a_future_candle`,
`backend/tests/test_backtesting_engine.py:183-212`). The missing piece for
optimization/walk-forward is purely an outer loop and an objective function —
nothing in the engine needs to change.

## Goals / Non-Goals
- Goals:
  - Let a strategy's parameter space be searched against real history and
    ranked by an explicit, configurable objective.
  - Make in-sample overfitting visible via out-of-sample (walk-forward)
    scoring, instead of trusting a single full-period backtest.
  - Make it possible to answer "is this strategy actually good in a
    trend_down regime, or only in trend_up" from data already being
    collected.
- Non-Goals:
  - Machine-learning-based signal generation (the intelligent-trading-bot
    approach) — a much bigger bet, not proposed here.
  - Multi-symbol/portfolio-level backtesting — still out of scope; each
    optimizer/walk-forward run stays single-symbol, matching the existing
    engine.
  - Automatically deploying optimizer-chosen parameters to production bots —
    this tooling produces a report a human reviews; it does not write
    `strategy_params` back into the database.

## Decisions
- Decision: implement the optimizer as a thin CLI wrapper
  (`app/backtesting/optimize.py`) that calls `BacktestEngine.run_candles` in
  a loop over a param grid/random sample, never modifying `engine.py`.
  - Why: preserves the no-lookahead / no-strategy-logic-change guarantees
    already proven for the engine; the optimizer is purely a caller.
- Decision: walk-forward splits are simple fixed-size rolling windows
  (e.g. N months train, M months validate, step forward by M), not
  anchored/expanding windows initially.
  - Why: simplest thing that catches "overfit to one window"; anchored/
    expanding windows can be added later if rolling windows prove
    insufficient.
- Decision: regime-conditioned reporting reuses whichever regime detector
  `fix-regime-detection-consistency` leaves as canonical, classifying regime
  at each trade's entry timestamp from the already-loaded candle series (a
  post-hoc bucketing of existing `BacktestResult.trades` /
  `equity_curve`, no coupling to replay-time state needed).

## Risks / Trade-offs
- Grid search over multiple params is combinatorial; cap default grid size
  and prefer random/latin-hypercube sampling for large spaces rather than a
  full grid, to keep a full optimizer run tractable. Reuse the CSV-loader's
  progress-reporting pattern (`backend/app/backtesting/progress.py`) so a
  long optimizer/walk-forward run stays observable, not silent.
- Walk-forward with only ~6 years of history (2020-2026) limits how many
  independent out-of-sample windows are possible — document this limitation
  rather than overstating statistical confidence.

## Migration Plan
1. Ship the optimizer first (useful standalone).
2. Ship walk-forward on top of it once the optimizer is stable.
3. Ship regime-conditioned reporting last, after
   `fix-regime-detection-consistency` lands.
4. None of this touches production trading paths — no rollback plan needed
   beyond not running the new CLI commands.

## Open Questions
- Default optimization objective: raw expectancy, Sharpe-like risk-adjusted
  return, or expectancy constrained by a max-drawdown ceiling? Recommend the
  latter (constrained expectancy) so the optimizer can't just find a
  high-return/high-ruin-risk parameter set — confirm before implementation.
