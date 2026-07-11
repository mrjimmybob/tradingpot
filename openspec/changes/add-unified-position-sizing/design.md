## Context
Position sizing today has three different conventions in one file: fixed-$
or fixed-%-of-balance (`dca`), fixed-%-of-capital with a convex depth
multiplier (`grid`), fixed-%-of-balance (`mean_reversion`), and
risk-%-of-balance/ATR-scaled (`trend_following`, `volatility_breakout`,
`dip_recovery` — three near-identical copies of the same formula). None of
them consult portfolio-level state when sizing; only the portfolio risk
service (called after signal generation) can shrink an order after the
fact — sizing and risk are sequential and one-directional, not integrated.

## Goals / Non-Goals
- Goals:
  - One sizing function, one convention, used by all 6 strategies.
  - Sizing scales with volatility (ATR) so order size reflects current risk,
    not just a flat % of balance.
  - No strategy's entry/exit *decision* logic changes — this only changes
    how much a "buy"/"sell" signal that was already decided gets sized.
- Non-Goals:
  - Kelly-criterion or other edge-proportional sizing (would require a
    trustworthy live win-rate/edge estimate per strategy — a reasonable
    future step once `add-strategy-validation-tooling`'s regime-conditioned
    reporting has produced real per-strategy edge numbers, but not
    attempted here).
  - Changing DCA's "never sells" behavior or any other strategy's exit
    logic.

## Decisions
- Decision: base the shared formula on the existing
  `trend_following`/`volatility_breakout`/`dip_recovery` convention
  (`risk_percent × balance / (ATR × multiplier)`) since it's already proven
  across 3 strategies, rather than inventing a new formula.
  - Why: minimizes new, unvalidated surface area; the goal here is
    consistency, not a novel sizing model.
- Decision: for `dca`/`grid`/`mean_reversion`, treat their current fixed-%
  parameter as a *ceiling* on the new ATR-scaled size, not a full
  replacement — so behavior in a "normal" volatility regime stays close to
  today's, and the difference only shows up in unusually calm or violent
  markets. This limits how much backtest results shift purely from the
  sizing refactor.
- Decision: this change does not touch DCA's philosophy (regular scheduled
  buying) — only how much each scheduled buy is for.

## Risks / Trade-offs
- Changing sizing changes backtest results for all 6 strategies even with
  identical entry/exit logic — must be re-validated (ideally via
  `add-strategy-validation-tooling`), not just assumed to be strictly
  better.
- Grid's convex depth-multiplier sizing is intentionally *not* flat — the
  unification must preserve "bigger orders at deeper discount levels" as a
  multiplier on top of the new base size, not discard it.

## Migration Plan
1. Extract `trend_following`'s sizing formula into a shared function; add
   tests proving it's behavior-identical for the 3 strategies that already
   use it.
2. Apply the shared function (with the ceiling decision above) to
   `mean_reversion`, then `grid`, then `dca`, one at a time with
   before/after backtests each.
3. Re-run the Tier 2 validation tooling on all 6 strategies once done.

## Open Questions
- Should `risk_percent`'s default differ per strategy (e.g. DCA's
  schedule-driven buys probably shouldn't risk as much per order as a
  discretionary breakout entry)? Needs a decision before step 2 of the
  migration plan.
