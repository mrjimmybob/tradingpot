## Phase ordering rationale

Phases are ordered by (1) urgency — strategies with **zero enforced market
suitability** go first, since "trades in inappropriate markets" was the
user's first-named symptom and is the most acute, active risk — then (2)
implementation complexity ascending, with `dca_accumulator` deliberately
last because it requires a conceptual design decision (does "never sell"
remain correct) before mechanical remediation can even be scoped, not
because it's less important. See `audits/README.md` for the full
per-strategy pillar table.

Each strategy phase (1-6) is its own reviewable unit of work and should be
backtested before/after independently — do not batch multiple strategies'
remediation into one implementation pass.

**Terminology note**: "confidence" and "failure detection" were replaced
during design review with "Evidence-Based Decision Score" and "Strategy
Edge Management" respectively (see `design.md`'s revision note). All tasks
below use the current terminology.

**Architecture note**: this revision adds Pillar 10, the Strategy Proposal
Interface — every strategy's Phase 1-6 migration now also changes its
return type from `TradeSignal` to `StrategyProposal` (see `design.md`'s
"The Strategy Proposal Contract"). This is why Phase 0 now also builds the
`StrategyProposal` schema and Standalone Adapter (0.10/0.11 below), and
each phase's task list ends with a proposal-migration task before
certification.

**Architecture Freeze**: as of `design.md`'s final architecture revision,
the `StrategyProposal` contract (including `execution_intent`, `validity`,
Market Assumptions, Proposal Immutability, the Comparison Contract, the
External Trust Layer, and the future `CommitteeDecision`/`TrustAdjustment`
shapes) is frozen. Every task below implements against that fixed
contract. No task in Phase 0-6 may add a new top-level `StrategyProposal`
field, change `direction`/`execution_intent` semantics, or weaken
immutability. If implementation or backtesting surfaces a concrete
limitation the frozen contract cannot express, stop and raise a new,
separate OpenSpec change with the specific evidence — do not amend the
contract ad hoc inside a phase. See `design.md`'s "Architecture Freeze"
section.

## Phase 0 — Shared Framework Infrastructure

**Status: ✓ Complete (12/12).** Implemented in
`backend/app/services/strategy_framework/` (`market_suitability.py`,
`decision_score.py`, `adaptive_params.py`, `trade_management.py`,
`edge_management.py`, `proposal.py`, `standalone_adapter.py`,
`explanation_persistence.py`, `future_auto_mode_contract.py`), plus the
`Order.decision_explanation`/`edge_management_category` columns
(`backend/app/models/order.py`, migration `009_add_order_decision_
explanation.sql`) and revisions to `add-unified-position-sizing/
{proposal,design}.md`. 137 new backend tests added (994 -> 1131 passing,
zero regressions) across `backend/tests/test_strategy_framework_*.py` and
`test_order_decision_explanation.py`. No strategy code touched; nothing
wired in yet — infrastructure only, per this phase's own acceptance
criteria.

- [x] 0.1 Build `app/services/strategy_framework/market_suitability.py`:
      a `MarketSuitabilityGate` wrapping the canonical bar-based regime
      detector (`fix-regime-detection-consistency`), taking a strategy's
      declared `allowed_regimes` and returning allow/deny - actually
      callable as a hard gate, not just a diagnostic value.
- [x] 0.2 Build `app/services/strategy_framework/decision_score.py`: a
      `DecisionScoreEngine` - a weighted accumulator over a strategy-
      declared list of Evidence Items, each with a documented Measurement
      (deterministic computation), Normalization (deterministic mapping to
      a bounded contribution range), Weight, and documented reason it
      contributes to edge. Must reject any factor that isn't expressible
      as a deterministic Measurement + Normalization pair. Must produce a
      human-readable Evidence Report (see `design.md` Pillar 3 for the
      exact format) for every scored trade.
- [x] 0.3 Build `app/services/strategy_framework/adaptive_params.py`:
      shared resolvers (e.g. ATR-percentile-scaled multiplier,
      regime-scaled lookback) strategies call instead of hardcoding. Also
      the landing point for Strategy Edge Management's Category B
      ("parameter mismatch") adaptations.
- [x] 0.4 Build `app/services/strategy_framework/trade_management.py`: a
      `TradeManagementMonitor` hook pattern for continuous post-entry
      thesis re-evaluation.
- [x] 0.5 Build `app/services/strategy_framework/edge_management.py`: a
      `StrategyEdgeManager` - per-bot, per-strategy continuous tracker of
      measurable degradation signals (win rate, expectancy, consecutive
      losses, regime/volatility/liquidity/spread/slippage drift, false-
      breakout frequency, holding time, reward:risk, Decision Score trend,
      execution quality) that classifies detected degradation into exactly
      one of three categories - A (temporary market mismatch: reduce
      activity/raise threshold/wait), B (parameter mismatch: adapt via
      0.3, mathematically justified only), C (edge disappeared: stop,
      require human re-certification to resume) - and can answer "why was
      I profitable / why am I not / can I adapt / should I wait / should I
      stop" from that evidence. Never force-closes an open position -
      dedicated test for this, and a dedicated test per category that a
      synthetic degradation episode classifies correctly.
- [x] 0.6 Extend `Order`/`Trade` storage to persist a summarized
      `DecisionExplanation` (from the existing `ExplanationBuilder`),
      including the Evidence Report and any active edge-management
      category, at execution time, so any historical trade is explainable,
      not just the current live tick (closes the Pillar 8 persistence gap
      found in every strategy's audit).
- [x] 0.7 **Revise `add-unified-position-sizing`'s design.md** before that
      change is implemented: its shared sizing function must take the
      Pillar 3 Decision Score as an input (`add-unified-position-sizing`
      currently takes no such input - its original design predates this
      framework and referred to a "confidence" input that no longer
      exists in this design). Coordinate rather than duplicate.
- [x] 0.8 Author the Strategy Audit Document template (theory + market
      suitability + Evidence Item table + edge-management category
      definitions + performance expectations sections) referenced by
      `audits/README.md`, formalizing the structure already used for the
      initial 6 audits in this change.
- [x] 0.9 Tests: each Phase 0 module unit-tested in isolation (no strategy
      wired in yet); full existing suite still passes.
- [x] 0.10 Define the `StrategyProposal` schema (`app/services/
      strategy_framework/proposal.py`): the field set from `design.md`'s
      "The Strategy Proposal Contract" (`proposal_id` deterministically
      derived from `(bot_id, strategy_id, candle_timestamp)`, `direction`
      enum with BUY/SELL/HOLD/NO_TRADE, `execution_intent` enum with
      NO_ACTION/OPEN_POSITION/ADD_TO_POSITION/REDUCE_POSITION/
      CLOSE_POSITION/HOLD_POSITION, `validity` (`generated_at`,
      `valid_until`), `assumptions` (objective, falsifiable conditions
      only), `decision_score`, `market_suitability`, `edge_status`,
      `suggested_position_size`, `suggested_risk_budget_pct`,
      `expected_holding_horizon`, `expected_edge_estimate` (populated ONLY
      from `add-strategy-validation-tooling`'s validated results, `None`
      otherwise - never self-computed), `reasons_for`/`reasons_against`
      (mechanically derived from `decision_score`, never freely authored),
      `adaptive_parameters_used`, `explanation`). Immutable once
      constructed - implement as a frozen dataclass (or equivalent) so
      immutability is enforced at runtime, not only by contract. This
      REPLACES `TradeSignal` as the strategy-function return type (see
      `design.md` Decisions for why replace, not wrap) - `TradeSignal`'s
      existing fields all map onto proposal fields, so this is a superset,
      not a loss of information. Per the Architecture Freeze, this schema
      is fixed - do not add fields beyond this list without a new,
      evidenced OpenSpec change.
- [x] 0.11 Build the Standalone Adapter: a thin translator (NOT Auto
      Mode - no ranking/comparison logic, exactly one proposal in, one
      order-or-nothing out) that (a) discards the proposal if
      `now >= validity.valid_until`, (b) branches on `execution_intent`
      (not `direction` alone) to determine the action, and (c) extracts
      order intent (`direction`/`suggested_position_size`) from a
      `StrategyProposal` and feeds it into the EXISTING, unchanged
      `_execute_trade` pipeline. NO_ACTION/HOLD_POSITION proposals produce
      no order. Dedicated test proving standalone execution outcome is
      identical before/after the `TradeSignal` -> `StrategyProposal`
      migration for a fixed set of historical scenarios, plus a dedicated
      test proving an expired proposal (past `valid_until`) is discarded
      and never executed.
- [x] 0.12 Documentation-only (no implementation): add a docstring/comment
      block alongside the `StrategyProposal` schema (0.10) recording the
      `CommitteeDecision` and `TrustAdjustment` shapes from `design.md`'s
      "Committee Decision" and "External Trust Layer" sections as the
      fixed future-consumer contract, so a future Auto Mode change starts
      from an already-agreed target instead of re-deriving it. Neither
      class is implemented, instantiated, or wired to anything - Auto Mode
      remains explicitly out of scope for this change.

**Phase 0 acceptance criteria**: all 5 shared modules (0.1-0.5) exist and
are unit tested, including a dedicated test proving `DecisionScoreEngine`
rejects a non-deterministic/unmeasurable factor and a dedicated test per
Strategy Edge Management category; `add-unified-position-sizing`'s design
updated; `Order`/`Trade` persist decision explanations including the
Evidence Report; the `StrategyProposal` schema (0.10, including
`execution_intent` and `validity`, enforced immutable at runtime) and
Standalone Adapter (0.11, including expiry and execution-intent-based
branching) exist and are unit tested, including a determinism test (same
inputs -> same proposal, including `proposal_id`), an expiry test, and a
behavior-preservation test (adapter + existing pipeline produces the same
outcome `TradeSignal` + existing pipeline used to); the `CommitteeDecision`/
`TrustAdjustment` future-contract documentation (0.12) exists; zero
behavior change to any strategy (no strategy calls the new modules yet -
Phase 0 is infrastructure only, wired to zero strategies until Phase 1).

## Phase 1 — volatility_breakout (smallest lift on Pillar 2; already computes what it needs)

**Status: ✓ Complete (9/9), Under review** (see `audits/volatility_breakout.md`'s
Certification Checklist — implementation and testing done; the doc's own
"Under review" status is pending human sign-off, not outstanding agent
work). Migrated in `backend/app/services/trading_engine.py`
(`_strategy_volatility_breakout`, now wired through
`MarketSuitabilityGate`, `AdaptiveParameterResolver`, `DecisionScoreEngine`,
`StrategyEdgeManager`, `StrategyProposal`, and `StandaloneAdapter` — the
full decision flow, no shortcuts, no duplicated framework logic). 21 new
tests (`backend/tests/test_volatility_breakout_framework_migration.py`,
8 classes covering Evidence Generation, Decision Score, Market Suitability,
Edge Management, Proposal Generation, Standalone Adapter Compatibility,
Historical Replay, and Regressions); `test_strategy_activity.py`'s
`TestVolatilityBreakoutDefaults` fixture updated for the new state keys.
Full suite: 1152/1152 passing, zero regressions. Before/after backtest run
across bull (2023-10-01–2023-11-15), bear (2022-11-01–2022-12-15), and
chop (2024-04-01–2024-05-15) windows — see `audits/volatility_breakout.md`'s
Backtesting section for full results, methodology, and an honest
discussion of what a 0%-observed-win-rate small sample can and can't tell
us. That section also documents a pre-existing, out-of-scope backtest-
engine trade-counting bug discovered (not introduced, not fixed) while
running these.

See `audits/volatility_breakout.md`. Gaps: P1, P3, P4 (beyond stop), P6
(no take-profit/time-exit), P7, P9; P2 computed-but-unenforced; P8
sizing/viability gap; P5 needs a Decision Score input.

- [x] 1.1 Author theory (P1) and performance expectations (P9) in
      `audits/volatility_breakout.md`.
- [x] 1.2 Wire the ALREADY-COMPUTED `regime_allows_entry` into the actual
      entry decision (`is_breakout`) - this is the one-line-conceptually
      but behavior-changing fix the earlier `fix-regime-detection-
      consistency` change deferred. Backtest before/after across bull/
      bear/chop, matching that change's own verification pattern.
- [x] 1.3 Replace the boolean `is_breakout` gate with a `DecisionScoreEngine`
      call (Phase 0.2), defining Evidence Items (breakout magnitude,
      compression duration, and any other measurable factors the
      finalized theory doc (1.1) identifies) with documented Measurement/
      Normalization/Weight for each.
- [x] 1.4 Add a take-profit and/or time-based exit to close the Pillar 6
      gap (currently only failed-breakout + trailing stop).
- [x] 1.5 Wire `StrategyEdgeManager` (Phase 0.5) before entry; define this
      strategy's specific Category A/B/C signal thresholds (e.g. what
      volatility-percentile range counts as Category A mismatch for a
      breakout strategy specifically).
- [x] 1.6 Wire Decision-Score-weighted sizing (Phase 0.7's revised shared
      sizer) in place of the flat risk_percent formula.
- [x] 1.7 Close Pillar 8 gaps: add `.check()`/`.metric()` for sizing and
      the fee-viability gate; verify the Evidence Report renders correctly
      for a real trade.
- [x] 1.8 Migrate this strategy's return type from `TradeSignal` to
      `StrategyProposal` (Phase 0.10), including this strategy's specific
      `execution_intent` mapping (breakout entry -> OPEN_POSITION, failed-
      breakout/trailing-stop exit -> CLOSE_POSITION), a `validity.
      valid_until` matching its evaluation interval, and its objective
      market assumptions (e.g. "breakout level not invalidated by close
      back inside the range"); verify standalone execution via the
      Standalone Adapter (Phase 0.11) is behavior-identical to the
      pre-migration `TradeSignal` path for a fixed set of historical
      scenarios.
- [x] 1.9 Certification review against the full checklist; before/after
      backtest comparison recorded in the audit doc.

## Phase 2 — trend_following (highest-urgency zero-suitability strategy)

**Status: ✓ Complete (9/9), Under review** (see `audits/trend_following.md`'s
Certification Checklist — implementation and testing done; the doc's own
"Under review" status is pending human sign-off). Migrated in
`backend/app/services/trading_engine.py` (`_strategy_trend_following`, now
wired through `MarketSuitabilityGate`, `AdaptiveParameterResolver`,
`DecisionScoreEngine`, `StrategyEdgeManager`, `StrategyProposal`, and
`StandaloneAdapter` — the full decision flow, no shortcuts, no duplicated
framework logic). 16 new tests
(`backend/tests/test_trend_following_framework_migration.py`, 7 classes
covering Evidence Generation, Decision Score, Market Suitability, Edge
Management, Proposal Generation, Standalone Adapter Compatibility, and Exit
Paths/Regressions); `test_strategy_validation.py`'s two trend_following
sizing tests and `test_trade_viability.py`'s `_TF_PARAMS` updated to isolate
the sizing/ATR-viability path from the new Pillar 2/3 gates (permissive
`allowed_regimes=["all"]` + `decision_score_threshold=0.0`, mirroring
mean_reversion's `regime_filter_enabled: False` convention). Full suite:
1169/1169 passing, zero regressions. Before/after backtest run across bull/
bear/chop windows — see `audits/trend_following.md`'s Backtesting section.

See `audits/trend_following.md`. Gaps: P1, P2 (zero, build from scratch),
P3, P4 (beyond stop), P7, P9; P8 sizing/stop gap; P5 needs a Decision
Score input; fix the `long_period` docstring/code mismatch (2842 says 200,
code uses 100) while touching this strategy.

- [x] 2.1 Author theory (P1) and performance expectations (P9).
- [x] 2.2 Build and wire a `MarketSuitabilityGate` (Phase 0.1) - this
      strategy has no internal regime awareness at all today.
- [x] 2.3 Replace the dual EMA-cross-plus-confirmation-count entry with a
      `DecisionScoreEngine` call, defining Evidence Items for trend
      maturity/strength, confirmation strength, and any other measurable
      factors from the theory doc, each with documented Measurement/
      Normalization/Weight.
- [x] 2.4 Fix the `long_period` docstring/code mismatch.
- [x] 2.5 Wire `StrategyEdgeManager` before entry; define this strategy's
      specific Category A/B/C signal thresholds.
- [x] 2.6 Wire Decision-Score-weighted sizing.
- [x] 2.7 Close Pillar 8 gaps: sizing calculation and initial stop
      placement need `.check()` calls; verify the Evidence Report.
- [x] 2.8 Migrate this strategy's return type from `TradeSignal` to
      `StrategyProposal`, including this strategy's `execution_intent`
      mapping (trend-confirmed entry -> OPEN_POSITION, EMA-cross-against-
      trend exit -> CLOSE_POSITION), a `validity.valid_until` matching its
      evaluation interval, and its objective market assumptions (e.g.
      "trend direction unchanged, EMA ordering not reversed"); verify
      standalone execution via the Standalone Adapter is behavior-identical
      to the pre-migration path.
- [x] 2.9 Certification review; before/after backtest comparison.

## Phase 3 — dip_recovery (zero suitability, but best overall foundation)

**Status: ✓ Complete (8/8), Under review** (see `audits/dip_recovery.md`'s
Certification Checklist). Migrated across all four methods
(`_strategy_dip_recovery`, `_dip_recovery_manage_setup`,
`_dip_recovery_manage_exit`, `_dip_recovery_exit_signal`), now wired through
`MarketSuitabilityGate` (via the price-only `_detect_market_regime` — see the
audit's Pillar 2 for the documented rationale and the `volatility_expanding`
consequence), `DecisionScoreEngine`, `StrategyEdgeManager`, Decision-Score-
weighted sizing, `StrategyProposal`, and `StandaloneAdapter`. The pre-migration
"never buy on the way down" safety preconditions are PRESERVED as hard gates.
16 new tests (`backend/tests/test_dip_recovery_framework_migration.py`);
`test_dip_recovery_strategy.py`'s BUY-reason assertion updated for the Pillar-10
mechanically-derived reason (25 of 26 unchanged). Full suite: 1185/1185
passing, zero regressions. Before/after backtest across bull/bear/chop — see
`audits/dip_recovery.md`'s Backtesting section.

See `audits/dip_recovery.md`. Gaps: P1, P2 (zero, build from scratch), P7,
P9; P3 partial (score computed but unused - wire it in); P8 emergency-stop
and sizing gaps; P5 needs a Decision Score input.

- [x] 3.1 Author theory (P1) and performance expectations (P9).
- [x] 3.2 Build and wire a `MarketSuitabilityGate`.
- [x] 3.3 Migrate the ALREADY-COMPUTED `opportunity_score` into a real
      `DecisionScoreEngine` call feeding `entry_ready`, formalizing its
      inputs as documented Evidence Items instead of discarding the score
      as diagnostics-only (same pattern as Phase 1.2/1.3, smaller lift
      than Phase 2 since the scoring infrastructure already exists here).
- [x] 3.4 Wire `StrategyEdgeManager` before entry; define this strategy's
      specific Category A/B/C signal thresholds.
- [x] 3.5 Wire Decision-Score-weighted sizing.
- [x] 3.6 Close Pillar 8 gaps: add the missing "Emergency stop hit"
      `.check()` and sizing/viability checks; verify the Evidence Report.
- [x] 3.7 Migrate this strategy's return type from `TradeSignal` to
      `StrategyProposal`, including this strategy's `execution_intent`
      mapping (recovery-confirmed entry -> OPEN_POSITION, take-profit/
      trailing-stop/emergency-stop/time-exit -> CLOSE_POSITION), a
      `validity.valid_until` matching its evaluation interval, and its
      objective market assumptions (e.g. "no new low since entry,
      recovery threshold not reversed"); verify standalone execution via
      the Standalone Adapter is behavior-identical to the pre-migration
      path.
- [x] 3.8 Certification review; before/after backtest comparison.

## Phase 4 — mean_reversion (P2/P6 already solid; needs P3/P5/P7 built)

**Status: ✓ Complete (8/8), Under review** (see `audits/mean_reversion.md`'s
Certification Checklist). Migrated in `_strategy_mean_reversion` (single
method), now wired through `MarketSuitabilityGate` (its existing reference
Pillar 2 routed through the shared gate), `DecisionScoreEngine` (3 Evidence
Items, weighted so no single factor reaches the threshold), `StrategyEdgeManager`,
Decision-Score-weighted sizing, `StrategyProposal`, and `StandaloneAdapter`. The
band-touch entry precondition and all four exits (Pillar 6) are preserved.
`bollinger_std` docstring/code mismatch fixed. 16 new tests
(`backend/tests/test_mean_reversion_framework_migration.py`); four existing
suites' MR fixtures isolated from the new score gate via
`decision_score_threshold: 0.0` (mirroring Phase 2). Full suite: 1201/1201
passing, zero regressions. Before/after backtest across bull/bear/chop — see
`audits/mean_reversion.md`'s Backtesting section.

See `audits/mean_reversion.md`. Gaps: P1, P3, P4 (beyond stop), P7, P9;
P8 fee-gate/sizing gap; P5 needs a Decision Score input; fix the
`bollinger_std` docstring/code mismatch (2130 says 2.0, code uses 1.8)
while touching this strategy.

- [x] 4.1 Author theory (P1) and performance expectations (P9).
- [x] 4.2 Replace the single band-touch entry condition with a
      `DecisionScoreEngine` call (this strategy's Pillar 2/6 are already
      the reference implementation - reuse its existing regime-detection
      pattern as a model for Phases 1-3's suitability gates), defining
      Evidence Items with documented Measurement/Normalization/Weight.
- [x] 4.3 Fix the `bollinger_std` docstring/code mismatch.
- [x] 4.4 Wire `StrategyEdgeManager` before entry; define this strategy's
      specific Category A/B/C signal thresholds.
- [x] 4.5 Wire Decision-Score-weighted sizing in place of the flat
      `order_size_percent`.
- [x] 4.6 Close Pillar 8 gaps: fee-viability gate and sizing/min-order
      floor need `.check()` calls; verify the Evidence Report.
- [x] 4.7 Migrate this strategy's return type from `TradeSignal` to
      `StrategyProposal`, including this strategy's `execution_intent`
      mapping (band-touch entry -> OPEN_POSITION, mean-reversion exit ->
      CLOSE_POSITION), a `validity.valid_until` matching its evaluation
      interval, and its objective market assumptions (e.g. "range/band
      structure not broken, regime still range-bound"); verify standalone
      execution via the Standalone Adapter is behavior-identical to the
      pre-migration path.
- [x] 4.8 Certification review; before/after backtest comparison.

## Phase 5 — adaptive_grid (most complex state machine; sequenced after the framework is proven)

**Status: ✓ Complete (7/7), Under review** (see `audits/adaptive_grid.md`'s
Certification Checklist — implementation and testing done; the doc's own
"Under review" status is pending human sign-off on the before/after backtest
comparison). Migrated in `backend/app/services/trading_engine.py`
(`_strategy_grid`, now wired through `MarketSuitabilityGate`,
`DecisionScoreEngine`, `StrategyEdgeManager`, `StrategyProposal`, and
`StandaloneAdapter`; new `_ema_slope_pct` helper for the range-bound-conviction
Evidence Item). 28 new tests
(`backend/tests/test_adaptive_grid_framework_migration.py`, 8 classes covering
Evidence Generation, Decision Score, Market Suitability, Edge Management,
Proposal Generation, Standalone Adapter Compatibility, Kill-Switch Diagnostics,
and Regressions). Full suite green, zero regressions. Before/after backtest
across the same bull/bear/chop windows as Phases 1–4 (see
`audits/adaptive_grid.md`): fees and drawdown fell in every window and return
moved to uniformly non-negative (+0.00 / +0.20 / +0.10), with the grid fully
abstaining in the bull trend — capital preservation through selectivity, not a
demonstrated positive edge (the grid is structurally near-dormant at 1m). Also
fixed a pre-existing O(n²) in the backtest harness (it bypassed
`_execute_strategy`, so the per-bar explanation builder was never reset;
`backend/app/backtesting/engine.py`) — observability-only, changes speed not
results, benefits every migrated strategy. Behaviour preserved by design:
`decision_score_threshold` defaults to 0.0 and all three Evidence Items
are non-negative in range-bound conditions, so the grid's mechanical
buy-low/sell-high behaviour is unchanged by default; the score's primary role
is Pillar 5 sizing. The one-order-per-bar invariant is why a single
`StrategyProposal` per evaluation remains sufficient — no multiple concurrent
proposals or staged execution intents were required.

See `audits/adaptive_grid.md`. Gaps (pre-migration): P1, P3, P7 (telemetry
exists, not adaptive), P9; P4/P5/P6 partial; P8's biggest single gap (kill
switches invisible to diagnostics) — all closed except P4/P6 (intentionally
partial, unchanged).

- [x] 5.1 Author theory (P1) and performance expectations (P9).
- [x] 5.2 Replace the single grid-level-crossing trigger with a
      `DecisionScoreEngine` call per level (or per batch of levels),
      defining Evidence Items with documented Measurement/Normalization/
      Weight.
- [x] 5.3 Promote the existing kill-switch telemetry
      (`kill_switch_count`, `lifetime_return_pct`,
      `lifetime_max_drawdown_pct`) into real `StrategyEdgeManager` inputs
      instead of unused telemetry - this strategy already tracks the raw
      numbers Phase 0.5 needs, it just never reads them back or
      classifies why a kill switch fired (Category A regime shift vs.
      Category B spacing/multiplier mismatch vs. Category C).
- [x] 5.4 Wire Decision-Score-weighted sizing in place of the flat
      base-%-times-depth-multiplier formula, preserving the depth
      multiplier as `add-unified-position-sizing` already specifies.
- [x] 5.5 **Fix the single biggest Pillar 8 gap found in the audit**: add
      `self._explain()` calls to both kill switches (drawdown,
      ATR-distance) - currently invisible to the diagnostics UI entirely.
- [x] 5.6 Migrate this strategy's return type from `TradeSignal` to
      `StrategyProposal` - including translating grid-level fills (not a
      single position) into the proposal shape; confirm with a dedicated
      test that `HOLD` is the correct direction for "no grid level
      crossed this tick" rather than misusing `NO_TRADE`, paired with
      `execution_intent=HOLD_POSITION`, while a level crossing that opens
      new virtual inventory maps to `ADD_TO_POSITION` (grid entries are
      incremental, not single fresh positions like Phases 1-4) and an
      opposing-level exit fill maps to `REDUCE_POSITION`/`CLOSE_POSITION`
      depending on remaining depth. Define `validity.valid_until` against
      this strategy's per-bar re-evaluation interval, and its objective
      market assumptions (e.g. "regime remains trend_flat/volatility_
      medium, price within the current grid range"). Verify standalone
      execution via the Standalone Adapter is behavior-identical to the
      pre-migration path.
- [x] 5.7 Certification review; before/after backtest comparison.

## Phase 6 — dca_accumulator (requires a design decision before mechanical work)

See `audits/dca_accumulator.md`. Special case: structurally has no
position to manage (never sells). Sequenced last so the framework is
mature/validated on 5 signal-driven strategies before tackling a
fundamentally different strategy shape.

- [x] 6.1 **Design decision (blocking, not mechanical)**: does "DCA never
      sells" remain correct given the finding that its apparent 2020-2026
      profitability is a bull-market-window artifact? **RESOLVED: Option
      (a) — keep DCA a classic accumulator that buys in timely chunks *no
      matter the market conditions*.** DCA stays pure and never-sell; the
      framework migration improves operational discipline, auditability,
      portfolio integration, execution quality, and risk governance, but
      must **not** convert DCA into a market-timing strategy. The
      load-bearing rule: gating is legitimate only when justified by
      **execution quality, portfolio constraints, or long-term-thesis
      invalidation** — never by expected short/medium-term price direction.
      Consequently the existing trend regime filter (pauses buying in
      `trend_down`) is re-scoped: it is *market timing* and is removed as
      the default (see 6.3). Documented with the full market-timing-vs-
      execution-quality distinction and the obligations it imposes on 6.2–
      6.8 in `audits/dca_accumulator.md`'s "Phase 6.1 Design Decision"
      section and `design.md`'s "Pure-accumulator exception."
- [x] 6.2 Author theory (P1) and performance expectations (P9), informed
      by 6.1. Theory: DCA's value is **direction-agnostic entry-timing-
      variance reduction plus disciplined scheduled deployment and good
      execution quality** — it holds no directional view and by design does
      not forecast short/medium-term price. Performance expectations must
      state honestly that DCA **underperforms lump-sum in a monotonic bull**
      (cash drag) and **keeps accumulating through, and holds, the full
      bear drawdown with no pause and no loss-cutting**; its claim is
      discipline + execution + governance, not a return figure. No P6 exit
      thesis is authored (Not Applicable by design — 6.1).
- [x] 6.3 **Re-scope Pillar 2 away from market timing.** Remove the trend
      `allowed_regimes` gate as DCA's default (retain only as an explicit,
      clearly-labelled non-classic operator override, `regime_filter_
      enabled` defaulting to `False`). In its place build the accumulator-
      appropriate suitability gate: it may skip/defer/trim a scheduled buy
      ONLY for (a) degraded **execution quality** on this fill (spread,
      liquidity, expected slippage — a bounded deferral/trim that can never
      become "wait indefinitely for a better price"), (b) **portfolio
      constraints** (exposure cap via `PortfolioRiskService`, concentration,
      budget exhaustion), or (c) **long-term-thesis invalidation** (a
      structural stop, not a price move). No factor in this gate may be a
      short/medium-term price-direction forecast (verified by a test that a
      pure `trend_down` regime, absent any execution/portfolio/thesis
      problem, does NOT block a scheduled buy — the inverse of the other
      five strategies' suitability tests). If any Decision Score is used at
      all, its Evidence Items are execution-cost factors that modulate
      *how* a due chunk is deployed, never *whether* to deploy it on
      direction.
- [x] 6.4 Wire `StrategyEdgeManager` with accumulator-correct categories:
      **ordinary mark-to-market drawdown is explicitly NOT a degradation
      signal** (DCA is designed to hold through it). Category C =
      **long-term-investment-thesis invalidation** (structural; requires
      human re-certification), not a losing streak or falling balance.
      Category A/B = **execution/portfolio** adaptation only (e.g. chunk
      size vs. available liquidity, exposure headroom), never a directional
      "wait for a better regime." Dedicated test that a simulated bear
      drawdown alone does not trip Category C, and that a simulated thesis-
      invalidation signal does.
- [x] 6.5 Do NOT introduce Decision-Score-weighted sizing (edge-weighted
      sizing would smuggle in a directional/quality view DCA rejects).
      **Document why flat, schedule-fixed chunk sizing is correct** for a
      classic accumulator (Pillar 4/5 "document why fixed" escape hatch),
      with the one permitted modulation being a **portfolio-cap trim**
      (never an increase, never direction-driven). Verify sizing still
      respects the `PortfolioRiskService` exposure cap.
- [x] 6.6 Close Pillar 8 gaps: add a positive `.check()` when the new
      execution/portfolio/thesis suitability gate PASSES (today only the
      old regime failure path is explained), surface the individual
      execution-quality / portfolio / thesis checks, and explain the
      buy-amount branching logic (fixed vs. percent, capped, floored,
      portfolio-trimmed) rather than only the final `buy_amount` metric.
- [x] 6.7 Migrate this strategy's return type from `TradeSignal` to
      `StrategyProposal`. DCA never emits `SELL` (6.1 keeps never-sell), so
      `direction` is only ever `BUY` (scheduled buy) or a non-buying tick.
      A non-buying tick (schedule not due, or an execution/portfolio/thesis
      gate deferred it) is **`NO_TRADE`** — no *entry* decision is being
      made this interval — paired with `execution_intent=NO_ACTION` (a pure
      accumulator has no actively-managed open position, so `HOLD`/`HOLD_
      POSITION` would misrepresent it). A due, ungated buy maps to
      `OPEN_POSITION` (first buy) or `ADD_TO_POSITION` (subsequent buys).
      Define `validity.valid_until` against the buy interval (not a candle
      timeframe). Every stated assumption must be objective and
      execution/portfolio/thesis-based (e.g. "spread within tolerance,"
      "exposure below cap," "long-term thesis not invalidated") — **never a
      direction forecast**. Verify standalone execution via the Standalone
      Adapter is behavior-identical to the pre-migration path.
- [ ] 6.8 Certification review; before/after backtest across bull, bear,
      AND chop windows. Because 6.3 removes the trend pause, this is a
      **deliberate behaviour change**: the backtest must show DCA now
      **keeps accumulating on schedule through the bear window** (where it
      previously paused), and that any skipped/deferred/trimmed buys are
      attributable to execution quality, portfolio limits, or thesis
      invalidation — never to price direction. Certification confirms DCA
      behaves as a disciplined, direction-agnostic accumulator with honest
      Pillar 9 expectations, not that it is profitable in all regimes.

## Acceptance Criteria (all phases)

A strategy phase is complete only when ALL of the following hold:
- Strategy Audit Document has no `UNDOCUMENTED` sections remaining.
- Market suitability gate is implemented AND actually blocks entries
  (verified by a test that asserts no trade in a disallowed regime).
- Evidence-Based Decision Score is implemented with a configured,
  documented threshold, every Evidence Item has a documented Measurement/
  Normalization/Weight/Reason, and no subjective or unmeasurable factor is
  present (verified by a test asserting a single-factor setup is
  insufficient to trade, and by review of the Evidence Item table).
- Every parameter is classified adaptive (with formula) or fixed (with
  documented reason) in the audit document - no undocumented constants.
- Position sizing takes the Decision Score as an input and is verified to
  respect the portfolio exposure cap (`PortfolioRiskService`).
- Trade management re-evaluates the open position every cycle (verified
  by a test that changes a thesis-relevant condition mid-trade and
  asserts the strategy reacts).
- Strategy Edge Management is wired and verified to: (a) correctly
  classify a simulated Category A, B, and C episode each into the right
  category, (b) take the corresponding bounded action for each, and (c)
  never force-close an open position regardless of category.
- Every decision point (entry, sizing, stop, exit) has a corresponding
  `.check()`/`.metric()`, a historical trade's decision explanation
  (including its Evidence Report) is retrievable, and the Evidence Report
  format matches `design.md`'s Pillar 3 example.
- The strategy returns `StrategyProposal`, not `TradeSignal`, with every
  required field populated; the Standalone Adapter routes it through the
  unchanged execution pipeline; and a before/after test proves standalone
  execution outcome is unchanged by the migration itself (isolating "the
  interface changed" from "the strategy's behavior changed" — the latter
  is expected from pillars 2-7's remediation, the former must be a no-op).
- The proposal's `execution_intent` is consistent with its `direction` per
  `design.md`'s pairing table (verified by a test covering each intent the
  strategy can produce); `validity.valid_until` is set and a test proves
  an expired proposal is never executed; every stated assumption is
  objective/falsifiable (reviewed against `design.md`'s Market Assumptions
  requirement, no subjective language); the proposal is immutable (a test
  attempting to construct then compare two independently-produced
  identical proposals confirms field-for-field equality, and the schema
  itself is a frozen dataclass or equivalent); and `expected_edge_estimate`
  is `None` unless sourced from `add-strategy-validation-tooling`'s
  validated results (never self-computed).
- Before/after backtest comparison is recorded, including trade count
  (expected to drop - document by how much and confirm it's not zero in
  reasonable conditions).
- All existing tests continue to pass; new tests cover each pillar above.

## Explicitly Out of Scope (this change and all its phases)

- `_strategy_auto` / Auto Mode, including its investment-committee logic
  (comparing, ranking, or selecting among multiple proposals; applying
  external context; capital allocation across proposals). Only the
  `StrategyProposal` contract those capabilities consume is specified
  here (pillar 10) — none of them are built. Depends on individual
  strategies being certified first; fully architected (ten-step Committee
  Process, ranking/tie-breaking, CommitteeDecision/TrustAdjustment) by the
  sibling change `add-auto-mode-investment-committee`, implemented by a
  future, separate change against that specification.
- `CommitteeDecision` and `TrustAdjustment` as running code. Their shapes
  are documented (0.12) here and fully specified by
  `add-auto-mode-investment-committee`, but neither is implemented,
  instantiated, or wired to anything in this change — no committee logic,
  no allocation algorithm, no trust-adjustment computation exists yet.
- Futures/short-selling support. `execution_intent` and `direction` are
  designed to extend to SHORT/COVER without a schema change (see
  `design.md`'s "Execution Intent"), but no futures/short-selling
  behavior is implemented here — that remains `add-short-selling-
  support`'s scope entirely.
- Calibrating final numeric threshold/weight values, Evidence Item
  weights, or edge-degradation category cutoffs in the abstract - each
  phase's certification uses `add-strategy-validation-tooling` (once
  available) or, at minimum, manual before/after backtesting matching the
  pattern already used in `fix-regime-detection-consistency`.
- Inventing or implementing any specific adaptive-parameter algorithm
  ahead of a strategy's own certification phase.
- Any external context source (sentiment, macro regime, Fear & Greed,
  news, funding/futures data, social sentiment, portfolio concentration).
  `StrategyProposal` is designed so these can be added later without
  touching a strategy (see `design.md`'s "External Trust Layer"), but none
  of them exist yet and none are built here.
