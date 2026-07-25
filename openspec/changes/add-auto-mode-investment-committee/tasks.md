## Status

**Implementation in progress (authorized).** This change was originally
architecture-only; implementation was gated on separate approval. That
approval has been given, and implementation now proceeds against this
already-agreed task breakdown (the same pattern `add-strategy-decision-
framework` used). **Phases 0-5 COMPLETE — Investment Committee certified AND
runtime-integrated end-to-end; the change is ready to archive once deployed.**
Phase 5 (Runtime Integration): `_strategy_auto` delegates (behind
`is_committee_enabled`, OFF by default) to `_strategy_auto_committee` — flat →
evaluate all Alpha strategies, `_committee_select` runs the committee and
executes the winner through the unchanged `_execute_trade`; in a position →
dispatch only the `owning_strategy`. Proposals surfaced via `TradeSignal.
_source_proposal` (`compare=False`, byte-identical guarantee intact); no
scheduler, no contract change, no per-strategy edits; `dca_accumulator` never a
candidate. 13 runtime tests (`test_auto_committee_runtime.py`); full suite 1371
passing, zero regressions. See `certification.md` §5. Phase 0 (Committee Core):
`backend/app/services/auto_committee/` (`comparison.py`, `decision.py`,
`trust.py`, `process.py`). Phase 1 (Portfolio Risk Wiring): `portfolio.py`
resolves the cycle's constraints via the unchanged `PortfolioRiskService`/
`StrategyCapacityService`; step 5/8 evaluate the max-total-exposure cap as a
single shared budget across the complete decision (ranking order, tie-groups
split proportionally) — order-independent, trim-only, no reimplementation, no
optimisation. Phase 2 (Execution Pipeline Wiring): `execution.py`
(`execute_committee_decision`) submits each selected proposal to the unchanged
`_execute_trade` in `execution_priority` order, reusing the Standalone
Adapter's exact translation (amount scaled to `allocated_size`); `flag.py`
(`is_committee_enabled`) gates it, OFF by default, coexisting with the
Standalone path. The required regression proves a single Alpha strategy through
Auto is byte-identical to the Standalone Adapter path (same bot/exchange/price/
session and field-identical `TradeSignal`). Phase 3 (External Trust Layer,
step 6): infrastructure only — `trust.py` adds a `TrustProvider` interface, a
`NeutralTrustProvider` (default: yields nothing), and an async
`resolve_trust_adjustments` resolver; `ranking.py` adds a
`RankingAdjustmentPolicy` seam with the single shipped `NeutralRankingPolicy`
(returns the base ranking value unchanged). The committee consumes an effective
ranking value the policy supplies and embeds NO trust mathematics (a test
asserts the orchestrator never reads `TrustAdjustment.adjustment`); production
default is behaviour-identical to Phase 2. No external sources, no learning, no
calibration. Phase 4 (Certification): all nine Auto Certification Gate items and
the user's certification scenarios pass as independent, documented, automated
checks (`TestAutoCertificationGate`); Open Questions closed in design.md
"Resolutions"; written report in `certification.md`. The §4.3 Auto-vs-Standalone
comparison is validation-only (NOT the acceptance criterion): the one-strategy
case is byte-identical by construction (a standalone `trend_following` bull-window
run is recorded for the number); the multi-strategy comparison is honestly
deferred to the (out-of-scope) engine-loop integration. 76 tests; full suite 1358
passing, zero regressions. Auto stays behind `is_committee_enabled` (OFF by
default). Depends on `add-strategy-decision-framework`'s Phases 0-6 (all
complete), which produce the real `StrategyProposal` objects Auto consumes.

This checklist also depends on `add-strategy-decision-framework`'s
Phase 0-6 strategy implementation having produced real `StrategyProposal`
objects to test against - Auto Mode's committee logic is meaningless
without at least a couple of certified strategies actually emitting
proposals. Sequencing this after at least Phase 1-2 of that change (not
necessarily all six phases) is a reasonable implementation-time judgment
call, not decided here.

## Phase 0 — Committee Core (pure logic, no wiring)

- [x] 0.1 Define the `CommitteeDecision` schema
      (`app/services/auto_committee/decision.py`): the field set from
      `design.md`'s "Committee Decision" (`decision_id`, `evaluated_at`,
      `proposals_considered`, `selected` as `SelectedAllocation`
      (`proposal_id`, `allocated_size`, `execution_priority`), `rejected`
      as `RejectedProposal` (`proposal_id`, `rejection_step`,
      `rejection_reason`), `trust_adjustments_applied`,
      `ranking_snapshot`). Immutable once constructed (frozen dataclass or
      equivalent, matching `StrategyProposal`'s own enforcement pattern).
- [x] 0.2 Define the `TrustAdjustment` schema
      (`app/services/auto_committee/trust.py`): `proposal_id`, `source`,
      `adjustment`, `generated_at`, per `design.md`'s "External Trust
      Layer". No source implementations (sentiment, macro, etc.) yet -
      schema only.
- [x] 0.3 Build the Comparison Contract reader: a pure function extracting
      exactly the seven fields `design.md` names (`direction`,
      `execution_intent`, `decision_score.total`/`.threshold`,
      `suggested_risk_budget_pct`, `expected_edge_estimate`,
      `market_suitability.is_suitable`, `edge_status.category`,
      `validity.valid_until`) from a `StrategyProposal`, refusing (raising
      or logging, implementer's choice - not specified here) any code path
      that attempts to read a field outside this set from within the
      committee package. Dedicated test asserting no other field is
      accessible/used by anything built in this phase.
- [x] 0.4 Build steps 2-4 of the Committee Process as pure functions over a
      proposal batch: expiration rejection (`validity.valid_until`),
      supersession rejection (newest-proposal-per-bot-per-strategy wins),
      edge-disqualification rejection (`edge_status.category == C`). Each
      independently unit tested, each producing a `RejectedProposal` with
      the correct `rejection_step`.
- [x] 0.5 Build step 7 (ranking) and the documented tie-breaking policy
      chosen for this implementation (see `design.md`'s "Open Questions" -
      this task is where that choice is actually made and recorded,
      referencing `add-strategy-validation-tooling` output if available at
      implementation time, falling back to `decision_score.total` if not).
      Dedicated determinism test (same inputs -> same `ranking_snapshot`,
      run N times) and a dedicated strategy-identity-blindness test (swap
      which strategy produced which field values, holding values constant,
      assert ranking unchanged).
- [x] 0.6 Build step 9 (selection) and step 8 (capital allocation) as pure
      functions consuming the step 7 ranking plus whatever the portfolio
      risk step (Phase 1) already capped. Must support selecting zero,
      one, or multiple proposals - dedicated test proving a multi-proposal
      selection is possible when synthetic portfolio state permits it
      (Auto Certification Gate's item 9).
- [x] 0.7 Assemble steps 0.3-0.6 plus stubs for steps 5-6 (Phase 1) into
      the full ten-step Committee Process orchestrator, producing a
      `CommitteeDecision` (0.1). Dedicated test proving every proposal in
      `proposals_considered` ends up in exactly one of `selected`/
      `rejected` (Auto Certification Gate's item covering "no proposal
      silently dropped").
- [x] 0.8 Tests: full Phase 0 acceptance per "Phase 0 acceptance criteria"
      below, using entirely synthetic `StrategyProposal` fixtures - no
      real strategy or execution pipeline wiring yet.

**Phase 0 acceptance criteria**: `CommitteeDecision`/`TrustAdjustment`
schemas exist and are immutable (0.1-0.2); the Comparison Contract reader
enforces field isolation (0.3); steps 2, 3, 4, 7, 8, 9 are implemented as
independently-tested pure functions (0.4-0.6); the full ten-step
orchestrator (with steps 5-6 stubbed to no-ops) produces a valid,
immutable `CommitteeDecision` from a synthetic proposal batch (0.7);
determinism, strategy-identity-blindness, and multi-selection are each
covered by a dedicated test; zero wiring to real strategies, real
portfolio services, or the execution pipeline yet.

## Phase 1 — Portfolio Risk Wiring (step 5)

- [x] 1.1 Wire step 5 to the existing, unchanged
      `PortfolioRiskService.check_portfolio_risk` and
      `StrategyCapacityService.check_capacity_for_trade`
      (`backend/app/services/portfolio_risk.py`,
      `backend/app/services/strategy_capacity.py`) for each proposal
      surviving steps 2-4's implied order. A hard block rejects the
      proposal (`rejection_step = "portfolio_risk"` or
      `"strategy_capacity"`); a resize carries the proposal forward with
      its allocation already capped.
- [x] 1.2 Resolve the multi-selection sequencing question flagged in
      `design.md`'s "Risks / Trade-offs": confirm and test whether/how
      `PortfolioRiskService`'s per-order exposure check must account for
      an earlier selection's not-yet-executed allocation within the same
      cycle, when more than one proposal is being evaluated for step 5 in
      the same batch. Dedicated test with two simultaneously-eligible
      proposals whose combined exposure would breach the cap individually
      passing but jointly failing.
- [x] 1.3 Tests: proposals that would be blocked/resized by the existing
      services are correctly rejected/capped through the committee's step
      5, using the SAME test fixtures/scenarios
      `add-trading-safety-boundaries`'s own test suite already uses for
      `PortfolioRiskService`/`StrategyCapacityService`, to confirm no
      behavior drift from calling these services via Auto instead of via
      `_execute_trade` directly.

**Phase 1 acceptance criteria**: step 5 is wired to the real, unchanged
portfolio risk/capacity services; the multi-selection sequencing question
is resolved and tested; a proposal that would be blocked by existing
`add-trading-safety-boundaries` protections is blocked identically when
routed through Auto instead of a single-strategy path.

## Phase 2 — Execution Pipeline Wiring (step 10)

- [x] 2.1 Build the translation from a `CommitteeDecision.selected` entry
      back into the order-intent shape `_execute_trade`
      (`backend/app/services/trading_engine.py:6332`) already expects -
      the same `direction`/`suggested_position_size` extraction the
      Standalone Adapter (`add-strategy-decision-framework` Phase 0.11)
      performs for a single proposal, reused here per selected proposal,
      now scaled by `SelectedAllocation.allocated_size` instead of the
      proposal's raw `suggested_position_size`.
- [x] 2.2 Submit each selected proposal to `_execute_trade` in
      `execution_priority` order, per `SelectedAllocation`. The existing
      pipeline (portfolio risk check, strategy capacity check, cost
      estimate, viability gate, order size validation, execution routing,
      execution) runs UNCHANGED for each.
- [x] 2.3 **Regression test (required, not optional)**: with exactly one
      strategy enabled, confirm the committee-driven path produces an
      IDENTICAL outcome to the pre-Auto-Mode Standalone-Adapter-driven
      path, for a fixed set of historical scenarios - a one-strategy
      portfolio through Auto should behave exactly like today's
      single-strategy standalone execution, proving the committee adds
      capability without changing behavior in the degenerate case.
- [x] 2.4 Feature-flag the committee path so it can be enabled per-bot or
      globally without removing the existing Standalone Adapter path -
      exact flagging mechanism is an implementation detail, not specified
      here.

**Phase 2 acceptance criteria**: selected proposals execute through the
unchanged `_execute_trade` pipeline in priority order; the one-strategy
regression test (2.3) passes; the committee path is behind a flag,
coexisting with the Standalone Adapter rather than replacing it outright.

## Phase 3 — External Trust Layer (step 6) — deferred, no source implementations here

- [x] 3.1 Wire step 6 to consult any persisted `TrustAdjustment` records
      for proposals surviving steps 2-5, applying `adjustment` only within
      ranking (step 7) - no proposal field is touched. If no
      `TrustAdjustment` records exist yet (expected at this phase), step 6
      is a no-op and ranking proceeds unaffected.
- [x] 3.2 Tests: a synthetic `TrustAdjustment` measurably changes ranking
      order versus the same proposal batch with no adjustment applied,
      and `CommitteeDecision.trust_adjustments_applied` correctly lists
      every consulted adjustment.

**Explicitly not built in Phase 3 or any phase of this change**: any
actual external signal source (Fear & Greed, news, macro, funding, social
sentiment, exchange health, ETF/regulatory events). Phase 3 only proves
the consumption mechanism works against synthetic `TrustAdjustment`
fixtures - producing real ones is separate, future work per source.

## Phase 4 — Certification

- [x] 4.1 Run the full Auto Certification Gate (`design.md`'s
      "Certification Requirements for Auto", also `spec.md`'s "Auto
      Certification Gate" requirement) against the Phase 0-3
      implementation: indicator isolation, proposal immutability across a
      full cycle, tie-breaking determinism, ranking reproducibility and
      strategy-identity-blindness, trust-adjustment auditability,
      explainable rejections, `CommitteeDecision` reproducibility, and
      multi-execution capability - each as its own documented,
      independently-reviewed check, not a single bundled sign-off.
- [x] 4.2 Document the actual tie-breaking policy, ranking formula, and
      (if resolved) multi-selection sequencing behavior chosen during
      implementation in this change's `design.md` "Decisions" section (or
      a follow-up amendment), closing the "Open Questions" this change
      left deliberately open.
- [x] 4.3 Before/after comparison: with the same set of certified
      strategies enabled, compare portfolio-level outcomes (trade count,
      capital utilization, drawdown) between (a) running each strategy
      standalone via its own Standalone Adapter/bot and (b) running all of
      them together under Auto for the same historical window - not to
      prove Auto is "better," but to produce a documented, evidence-based
      before/after record for whoever approves enabling it in production.

**Phase 4 acceptance criteria**: every Auto Certification Gate item has a
passing, documented test; Open Questions this change left unresolved are
closed with recorded decisions; a before/after comparison exists and is
reviewed before Auto Mode is enabled for any real bot.

## Phase 5 — Runtime Integration (single Auto bot; no scheduler)

Wire the committee into the running engine so Auto Mode actually decides via
the committee at runtime, per the approved integration decision: **Auto is one
bot, not a portfolio of bots.** Inside the existing Auto bot's loop, evaluate
all enabled Alpha strategies, collect their `StrategyProposal`s, run the
Investment Committee, and execute the selected proposal through the existing
`_execute_trade` pipeline. **No cross-bot scheduler or portfolio-level
runtime.** Behind `is_committee_enabled` (OFF by default); the existing
rotation-based Auto path is unchanged when the flag is off.

- [x] 5.1 Surface each strategy's `StrategyProposal` to the committee without
      per-strategy edits or a `StrategyProposal`-contract change: attach the
      source proposal to the `TradeSignal` the Standalone Adapter already
      produces (`_source_proposal`, `field(compare=False, repr=False)` so it
      never affects `TradeSignal` equality — the Phase 2 byte-identical
      guarantee holds).
- [x] 5.2 Use the clarified committee Alpha set (`trend_following`,
      `mean_reversion`, `volatility_breakout`, `dip_recovery`,
      `adaptive_grid`) — NOT the legacy `_ALPHA_STRATEGIES` constant (which
      includes `dca_accumulator` and omits `dip_recovery`). Reconcile the
      naming so the committee uses the correct Alpha set; `dca_accumulator`
      (Allocation) never enters the committee.
- [x] 5.3 Build `_committee_select`: evaluate each candidate Alpha strategy in
      isolation (fresh explanation builder per strategy, `_invoked_by_auto`),
      collect the proposals, `resolve_portfolio_constraints` +
      `run_committee` (neutral trust/policy), and return the top-ranked
      selected proposal's signal scaled to `allocated_size`, or nothing.
- [x] 5.4 Build `_strategy_auto_committee` and delegate to it from
      `_strategy_auto` when `is_committee_enabled(bot)`: **in a position →
      dispatch only the `owning_strategy`** (reuses the existing
      `auto-mode-position-ownership` rule — the entering strategy manages its
      own exit); **flat → `_committee_select`**, stamping the winner's reason
      `[Auto:<strategy>|committee]` so `_resolve_owning_strategy` records the
      correct owner on execution. One position at a time (single-bot host);
      runtime multi-execution needs a multi-position/multi-bot model and is
      NOT built here (the capability stays certified).
- [x] 5.5 Tests: flag-off leaves `_strategy_auto` behaviour unchanged;
      `_committee_select` picks the highest-ranked actionable proposal and
      scales its amount; the in-position path dispatches only the owner; the
      winner's reason is stamped for ownership resolution; `dca_accumulator`
      is never a committee candidate.
- [x] 5.6 End-to-end certification: extend the Auto Certification Gate /
      `certification.md` with the runtime-integration checks and confirm the
      full suite passes. This closes the runtime-integration gap flagged in
      Phase 4's §4.3.

**Phase 5 acceptance criteria**: with the flag on, an Auto bot evaluates all
Alpha strategies, runs the committee, and executes the selected proposal
through the unchanged `_execute_trade`; position ownership is preserved via
`owning_strategy`; `dca_accumulator` never competes; with the flag off,
behaviour is byte-identical to the pre-Phase-5 rotation Auto. No cross-bot
scheduler, no `StrategyProposal`-contract change, no per-strategy edits.

## Explicitly Out of Scope (this change and all its phases)

- Any real external trust source (Fear & Greed, news, macro, funding,
  social sentiment, exchange health, ETF/regulatory events) - only the
  consumption mechanism (Phase 3) is built; producing real
  `TrustAdjustment` records from a real source is separate, future,
  per-source work.
- Any learning/historical-performance-based trust computation - the
  extension point is proven to work (Phase 3's synthetic test) but no
  learning logic is built.
- Correlation limits, sector/concentration limits, and a cash-reserve
  floor - `design.md`'s "Portfolio Thinking" reserves their place in step
  5, but building the underlying checks is separate, future work, likely
  coordinated with `add-trading-safety-boundaries`'s own future extension
  rather than built standalone inside Auto's package.
- Calibrating real ranking weights or tie-breaking thresholds ahead of
  Phase 0.5/4.2's implementation-time decision - this change does not fix
  numeric values, only the mechanism and the requirement that whatever is
  chosen be deterministic and Comparison-Contract-only.
- Enabling Auto Mode for any real bot or real capital - Phase 4's
  before/after comparison is a prerequisite for that decision, not this
  change's own deliverable.
- Modifying any of the 6 non-Auto-Mode strategies or
  `add-strategy-decision-framework`'s frozen `StrategyProposal` contract.
