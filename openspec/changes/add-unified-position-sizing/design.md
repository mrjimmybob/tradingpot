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
  - **Sizing scales with the trade's Evidence-Based Decision Score**
    (`add-strategy-decision-framework` Pillar 5, "Decision-Score-Weighted
    Position Sizing"). A marginal-score trade — one that only just clears
    its strategy's configured threshold — SHALL size smaller than a
    materially-higher-score trade, all else (volatility, risk_percent)
    equal. This closes the "⚠ Blocked on revision" note this proposal
    previously carried (see `proposal.md`): `add-strategy-decision-
    framework`'s Phase 0 has now shipped the Decision Score input this
    change was waiting on — `DecisionScoreResult` (`backend/app/services/
    strategy_framework/decision_score.py`), whose `.total` (0-100) and
    `.threshold` fields are exactly what the shared sizing function
    consumes (see Decisions below).
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
  - Defining the exact Decision-Score-to-size-multiplier curve (linear vs.
    stepped vs. another deterministic function). Structurally required
    (see Decisions), but the specific formula/calibration is left to this
    change's own implementation, informed by
    `add-strategy-validation-tooling` — matching how
    `add-strategy-decision-framework` itself leaves Evidence Item weights
    and thresholds as certification-phase, not design-phase, decisions.

## Decisions
- Decision: base the shared formula on the existing
  `trend_following`/`volatility_breakout`/`dip_recovery` convention
  (`risk_percent × balance / (ATR × multiplier)`) since it's already proven
  across 3 strategies, rather than inventing a new formula.
  - Why: minimizes new, unvalidated surface area; the goal here is
    consistency, not a novel sizing model.
- Decision: the shared sizing function's signature takes a
  `decision_score: DecisionScoreResult` parameter (from
  `app/services/strategy_framework/decision_score.py`, Phase 0 of
  `add-strategy-decision-framework`, already implemented) and applies a
  deterministic `size_multiplier(decision_score)` on top of the existing
  ATR/risk-percent formula: `size = base_risk_size *
  size_multiplier(decision_score.total, decision_score.threshold)`. A
  higher score does NOT automatically mean maximum size — the multiplier
  function itself must be deterministic and reproducible (same standard
  Pillar 3 already holds `DecisionScoreEngine` to), not a separate,
  unaccountable scaling knob.
  - Why: this is the specific, required coupling
    `add-strategy-decision-framework`'s Pillar 5 mandates — sizing based on
    Decision Score, portfolio exposure, and risk budget, never Decision
    Score alone and never a flat percentage regardless of the evidence
    behind the trade. Passing the full `DecisionScoreResult` (not just the
    bare `.total` float) also gives the sizing function access to
    `.threshold` for margin-aware scaling (e.g. a trade at 51/75 threshold
    sizes differently than one at 95/75) without needing a second shared
    lookup.
  - Alternatives considered: sizing only off `decision_score.total` in
    isolation (no threshold context) — rejected because "marginal-score
    trade sized down" (the framework's own required scenario) needs to
    know how close to the threshold a score is, not just its absolute
    value, which varies by strategy since thresholds are
    per-strategy-configurable.
- Decision: for `dca`/`grid`/`mean_reversion`, treat their current fixed-%
  parameter as a *ceiling* on the new ATR-scaled-and-Decision-Score-scaled
  size, not a full replacement — so behavior in a "normal" volatility
  regime with a middling score stays close to today's, and the difference
  only shows up at the extremes (unusually calm/violent markets, or
  unusually weak/strong evidence). This limits how much backtest results
  shift purely from the sizing refactor.
- Decision: this change does not touch DCA's philosophy (regular scheduled
  buying) — only how much each scheduled buy is for. DCA's own Decision
  Score input depends on `add-strategy-decision-framework`'s Phase 6
  (`dca_accumulator`'s migration, which itself has a blocking design
  decision — see that change's tasks.md 6.1) landing first; until then,
  DCA's scheduled buys size using the pre-existing ATR/risk-percent formula
  only, with Decision-Score scaling applied once Phase 6 resolves what a
  Decision Score even means for a schedule-driven accumulator.

## Risks / Trade-offs
- Changing sizing changes backtest results for all 6 strategies even with
  identical entry/exit logic — must be re-validated (ideally via
  `add-strategy-validation-tooling`), not just assumed to be strictly
  better. Adding Decision-Score scaling on top compounds this — before/
  after comparisons must isolate "ATR scaling changed" from "Decision
  Score scaling changed" where possible, not attribute all drift to one.
- Grid's convex depth-multiplier sizing is intentionally *not* flat — the
  unification must preserve "bigger orders at deeper discount levels" as a
  multiplier on top of the new base size, not discard it. This is now a
  THIRD multiplier stacked with the ATR scaling and the Decision Score
  scaling — the implementation must keep these three composable and
  independently testable (e.g. `size = base * atr_multiplier *
  decision_score_multiplier * depth_multiplier`), not fused into one
  opaque formula that can't isolate which factor caused a given size.
- A strategy that has not yet migrated to `StrategyProposal`/
  `DecisionScoreEngine` (i.e. any strategy before its own
  `add-strategy-decision-framework` Phase 1-6 lands) has no
  `DecisionScoreResult` to pass in. Mitigation: sequence each strategy's
  sizing migration AFTER that same strategy's Phase 1-6 migration, never
  before — this is now an explicit ordering constraint, not just a
  suggestion (see Migration Plan).

## Migration Plan
0. **Precondition, per strategy**: do not apply the shared sizing function
   to a strategy until that strategy has completed its own
   `add-strategy-decision-framework` Phase 1-6 migration and has a real
   `DecisionScoreResult` to size from — sizing cannot take a Decision Score
   input from a strategy that does not yet compute one.
1. Extract `trend_following`'s sizing formula into a shared function
   (signature: `size(base_risk_inputs, decision_score: DecisionScoreResult,
   ...) -> float`); add tests proving it's behavior-identical to today's
   ATR-only formula when `size_multiplier(decision_score)` is held at 1.0,
   for the 3 strategies that already use the ATR convention.
2. Apply the shared function (with the ceiling decision above) to
   `mean_reversion`, then `grid`, then `dca`, one at a time with
   before/after backtests each, in each case only after that strategy's own
   Phase 1-6 migration has landed (precondition 0).
3. Re-run the Tier 2 validation tooling on all 6 strategies once done.

## Open Questions
- Should `risk_percent`'s default differ per strategy (e.g. DCA's
  schedule-driven buys probably shouldn't risk as much per order as a
  discretionary breakout entry)? Needs a decision before step 2 of the
  migration plan.
- The exact `size_multiplier(decision_score)` curve (e.g. linear from
  threshold→100 mapping to 0.5x→1.5x, or a steeper/stepped function) is
  intentionally left open (see Non-Goals) — an implementation-time,
  per-strategy-validated decision, not fixed in this design.
