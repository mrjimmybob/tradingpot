## Context

Six discretionary strategies exist in `backend/app/services/trading_engine.py`
(`_strategy_dca`, `_strategy_grid`, `_strategy_mean_reversion`,
`_strategy_trend_following`, `_strategy_volatility_breakout`,
`_strategy_dip_recovery`), plus `_strategy_auto` (Auto Mode, explicitly out
of scope for this change). A fresh audit of all 6 against the 9 pillars
below (see `audits/*.md` for full file:line evidence) found a consistent
pattern: every strategy has *some* market-suitability and trade-management
logic, but none has an evidence-based decision score, none manages its own
edge (they neither detect nor classify why performance degrades), and
position sizing is either flat-percentage (3 of 6: dca, grid,
mean_reversion) or risk/ATR-scaled (3 of 6: trend_following,
volatility_breakout, dip_recovery) with neither group weighting size by
decision quality — and, critically, **volatility_breakout's own
market-suitability computation is diagnostic-only and never actually gates
its entry decision** (discovered and left uncorrected as an architecture
question during `fix-regime-detection-consistency`, which fixed the
*computation* but correctly scoped the *re-enabling of enforcement* out as
a separate, validated behavior change — this pillar's remediation is that
follow-up).

**Revision note**: this design was revised after initial review to remove
subjective language. The original draft used "confidence" for pillar 3/5
and "failure detection" for pillar 7. Both were replaced: "confidence"
implies a feeling a trading system cannot have and cannot measure;
"failure detection" only answers "am I losing," not "why, and what should
I do about it." See "Evidence-Based Decision Score" and "Strategy Edge
Management" below for the replacements. This revision is design-only — no
shared module or strategy code exists yet to migrate.

Existing shared infrastructure this change builds on rather than duplicates:
- `ExplanationBuilder`/`DecisionExplanation` (`app/services/strategy_explain.py`)
  — live, structured, exception-safe decision diagnostics already used by
  every strategy. Covers most of pillar 8 already; the gap is persistence
  (see Pillar 8 below).
- `_detect_market_regime_bar_based` (`trading_engine.py`, made canonical by
  `fix-regime-detection-consistency`) — the regime detector pillar 2's
  shared gate should standardize on, and a direct input to several
  Evidence Items (pillar 3) and edge-degradation signals (pillar 7).
- `PortfolioRiskService` / `StrategyCapacityService` (`add-trading-safety-
  boundaries`, complete) — portfolio- and allocation-level risk boundaries.
  Neither tracks a strategy's own recent win/loss performance or *why* it
  changed — that gap is pillar 7, a different, complementary concern (a
  strategy's personal edge validity, not the portfolio's aggregate
  exposure).
- `evaluate_reward_risk` (`trading_engine.py:258-289`) — a binary
  pass/fail per-trade expectancy gate, already centrally enforced. This is
  the closest existing thing to pillar 3 but is not a scored, multi-factor
  system, cannot be configured per-strategy beyond its single reward/risk
  ratio, and produces no Evidence Report (see below).

## Goals / Non-Goals

- Goals:
  - Define the 10-pillar standard as a concrete, verifiable OpenSpec
    capability (requirements + scenarios), not a prose aspiration.
  - Every scored or classified decision the framework makes must be
    traceable to objectively measurable, reproducible, deterministic
    inputs — never a subjective or unmeasurable concept.
  - Design shared infrastructure once, used by all 6 strategies, so a
    pillar can never be silently skipped by one strategy while present in
    another (exactly the inconsistency the audit found: 3 different
    position-sizing conventions, 2 different regime detectors, a
    suitability check that's computed but not enforced).
  - Define a single, stable output contract (`StrategyProposal`, pillar
    10) between every strategy and everything downstream, so a future
    Auto Mode redesign, portfolio allocation, or external-signal
    integration (sentiment, macro, funding) never requires touching a
    strategy's implementation a second time.
  - Produce a concrete, evidence-based audit of all 6 strategies and a
    phased, worst-first remediation plan.
- Non-Goals:
  - Implementing any of the remediation now (planning-only change).
  - Redesigning Auto Mode (explicitly deferred by the user until individual
    strategies are certified).
  - ML-based scoring. Start with a transparent, rule-based weighted-factor
    score — pillar 8 (self-diagnostics) requires every score to be
    explainable to a human trader, which a black-box model would fail
    until separate interpretability tooling exists (not proposed here).
  - Calibrating the actual numeric thresholds/weights (decision-score
    cutoff, evidence-item weights, edge-degradation classification
    thresholds, adaptive-parameter formulas). Those need empirical
    validation via `add-strategy-validation-tooling`'s optimizer/walk-
    forward harness, sequenced as part of each strategy's certification
    phase, not decided in the abstract here.
  - Inventing or implementing any adaptive-parameter algorithm or specific
    degradation-classification threshold in this change. This change
    defines the *structure* evidence, scoring, and classification must
    take; the *values* are certification-phase, per-strategy decisions
    grounded in each strategy's own theory document and backtest evidence.
  - Implementing `StrategyProposal`, the Standalone Adapter, or any Auto
    Mode investment-committee logic. Pillar 10 and "The Strategy Proposal
    Contract" below are a specification only — the dataclass/schema,
    adapter, and (much later) committee are Phase 0 / future-change
    implementation work, not done here.

## The 10 Pillars: what exists, what's shared, what's strategy-specific

For each pillar: the shared infrastructure a strategy composes into (if
any), and how it's enforced (spec requirement + scenario in the delta spec).

**1. Theory** — Documentation only, no runtime code. Every strategy gets a
`audits/<strategy>.md`-style Strategy Audit Document (this change ships the
initial 6 as-is audits; certification requires updating each one alongside
its remediation) with: Market Inefficiency, Why The Edge Should Exist,
Evidence, Conditions For Success, Conditions For Failure, Assumptions.
Enforced at certification review time, not by code. Pillar 1 is also the
justification anchor for pillar 3's Evidence Items (each item's "reason it
contributes to edge" must trace back to this document) and pillar 7's edge
classification (a Category C finding is, definitionally, evidence that
this document's edge thesis no longer holds).

**2. Market Suitability** — Shared `MarketSuitabilityGate` built on
`_detect_market_regime_bar_based`. Each strategy declares its suitable
regimes (reusing/extending the existing `_get_strategy_capabilities()`
`allowed_regimes` convention, today only consulted by Auto Mode dispatch —
see Context). The gate must actually block standalone entries, not just
compute a diagnostic value only wired to Auto Mode's OWN separate,
already-correct check.

**Pure-accumulator exception (`dca_accumulator`).** A schedule-driven
Dollar-Cost-Averaging strategy *deliberately ignores short- and
medium-term market direction*: gating accumulation on a "suitable" trend
regime would convert it into a market-timing strategy, which is precisely
the behaviour a classic DCA rejects. There is a hard line here between
**market timing** (deciding *whether to accumulate* from expected
short/medium-term price movement — not allowed for DCA) and **execution
quality / portfolio governance / thesis validity** (deciding *how well and
whether it is still appropriate* to deploy a chunk whose timing was already
fixed by the schedule — allowed). For `dca_accumulator`, Pillar 2's gate is
therefore *redefined, not removed*: "suitability" means
**execution-quality, portfolio-constraint, and long-term-investment-thesis**
suitability, never expected short-term direction. It declares **no trend
`allowed_regimes` gate**; the gate may only block, defer, or trim a
scheduled buy for (a) degraded execution quality on this fill (spread,
liquidity, slippage), (b) portfolio risk-governance limits (exposure cap,
budget exhaustion, concentration), or (c) invalidation of the long-term
investment thesis (a structural condition, never a price-direction
forecast). This is the one documented per-strategy exception to the hard
trend-regime gate above — it changes only *what DCA gates on*, not the
`StrategyProposal` contract or any other strategy. See
`audits/dca_accumulator.md`'s "Phase 6.1 Design Decision" and the Phase 6
tasks.

**3. Evidence-Based Decision Score** (formerly "Trade Quality Score" /
"Confidence") — A trading system must never "feel confident." Every
candidate trade's score SHALL be entirely derived from measurable
observations. Replaces single/dual-condition entry triggers (`price <=
lower_band`, `EMA cross + price above EMA`, `price crosses grid level`,
`breakout above upper band`) as the actual gate before a `buy`/`sell`
signal is emitted. This is the single largest architectural change of the
9.

Shared `DecisionScoreEngine` (`app/services/strategy_framework/
decision_score.py`): a weighted accumulator over a strategy-declared list
of **Evidence Items**. Every Evidence Item SHALL define:

| Field | Meaning |
|---|---|
| Evidence Item | Name of the observable factor (e.g. "Trend strength") |
| Measurement | The deterministic, reproducible computation that produces a raw value from available data (e.g. "EMA(20) slope over the last 10 bars, in %") |
| Normalization | The deterministic mapping from raw measurement to a bounded contribution range (e.g. "clamp(slope_pct / 2.0, -1, 1)") |
| Weight | This item's maximum point contribution to the total score |
| Reason it contributes to edge | Cross-reference to the strategy's Theory document (pillar 1) explaining *why* this measurable factor is evidence for or against the strategy's specific edge |

A strategy's Decision Score is the sum of each Evidence Item's
`normalized_value * weight`, scaled to 0-100, compared against a
per-strategy-configurable minimum threshold. Candidate factors (not
exhaustive, not all applicable to every strategy): trend strength, higher
timeframe agreement, market regime compatibility, volatility state,
volatility expansion/contraction, volume confirmation, liquidity,
support/resistance distance, risk/reward ratio, pullback quality, breakout
confirmation, momentum confirmation, indicator agreement, correlation,
market structure, recent execution quality, spread, slippage, expected
transaction cost, probability estimates derived from historical frequency.

Every Evidence Item SHALL be objectively measurable, reproducible,
deterministic, computable from data the system already has access to, and
documented. The framework explicitly REJECTS subjective or
unmeasurable factors — "looks bullish," "strong feeling," "good setup," or
any factor whose value depends on human judgement rather than a
data-derived computation. A factor that cannot be written as a Measurement
+ Normalization pair does not belong in the score.

Every executed trade SHALL produce a complete, human-readable Evidence
Report — an explanation, not a UI mockup — of the form:

```
Evidence
✓ Trend strength                 +18
✓ Higher timeframe aligned       +15
✓ Volume confirmation            +12
✓ Pullback quality               +11
✓ Risk/Reward                    +17
✓ Regime compatible              +20
✗ Nearby resistance               -8

Total Decision Score:
85 / 100

Minimum required:
75

Trade approved.
```

This is produced from the same structured data pillar 8 already persists
(`DecisionExplanation`/`ExplanationBuilder`, extended per pillar 8 below) —
the Evidence Report is a specific, required rendering of that structure,
not a separate system.

The exact scoring method (linear weighted sum vs. another deterministic
combination) is left open; it MUST be deterministic and reproducible —
the same inputs always produce the same score and the same Evidence
Report.

**4. Parameter Adaptation** — Shared `AdaptiveParameterResolver` utility
(e.g. ATR-multiplier-from-volatility-percentile, lookback-from-regime).
Every strategy's audit document must classify every parameter as ADAPTIVE
(cite the formula) or FIXED (document why fixed is correct — e.g. a
persistence-bar count for regime-flip debouncing is reasonably a small
fixed integer, not something that needs to scale with volatility). No
silent, undocumented hardcoding is allowed post-certification. This
pillar is also where pillar 7's Category B ("parameter mismatch")
remediation lands — an adaptive parameter that pillar 7 determines is
miscalibrated for current conditions is adapted here, mathematically
justified, not guessed.

**5. Decision-Score-Weighted Position Sizing** (formerly
"Confidence-Weighted Position Sizing") — Extends (does not duplicate) the
shared sizing function already planned in `add-unified-position-sizing` to
also take the pillar-3 Decision Score as an input (e.g. `size =
base_risk_size * size_multiplier(decision_score)`), and to consult
`PortfolioRiskService`/`StrategyCapacityService` for exposure/allocation
awareness (already built, already correct post `add-trading-safety-
boundaries`). A higher Decision Score does not automatically mean maximum
size — the multiplier function itself is subject to the same
determinism/reproducibility requirement as pillar 3. **This is a required
revision to `add-unified-position-sizing`'s design before that change is
implemented** — its current design doesn't take a Decision Score input.

**6. Trade Management** — Shared `TradeManagementMonitor` hook pattern:
every strategy's post-entry branch calls it every tick to re-evaluate (a)
is the market-suitability check (pillar 2) still passing, (b) has
volatility regime changed materially since entry, (c) should the stop
tighten beyond what a pure trailing-stop formula already does, (d) should
partial profit be taken. Three of six strategies (mean_reversion,
trend_following, dip_recovery per the audit) already re-evaluate multiple
exit conditions every tick — this pillar formalizes and completes that
pattern for grid/volatility_breakout/dca (the latter structurally has no
position to manage, by design — see its audit) rather than inventing it
from scratch.

**7. Strategy Edge Management** (formerly "Failure Detection") —
Detecting "I am losing" is insufficient; professional traders investigate
*why*. Shared `StrategyEdgeManager` (`app/services/strategy_framework/
edge_management.py`): a per-bot, per-strategy continuous monitor that
tracks measurable degradation signals and classifies *why* performance is
degrading, not merely *that* it is. Candidate signals (not exhaustive):
market regime changed, volatility outside the strategy's design
assumptions, liquidity degraded, spread increased, slippage increased,
false-breakout frequency increasing, stop-outs increasing, average holding
time collapsing, average reward:risk deteriorating, expectancy decreasing,
win rate statistically deteriorating, Evidence-Based Decision Scores
trending down, execution quality deteriorating.

Every detected degradation SHALL be classified into exactly one of three
categories, each with a specific, bounded action:

| Category | Meaning | Action |
|---|---|---|
| **A — Temporary market mismatch** | Conditions (regime, volatility, liquidity) have moved outside this strategy's suitable range, but nothing suggests the underlying edge is gone | Reduce activity; raise the Decision Score threshold; wait. Never a permanent stop. |
| **B — Parameter mismatch** | The strategy's thesis may still be valid, but its current parameters (ATR multiplier, lookback, confirmation threshold, stop distance, etc.) no longer fit current conditions | Adapt the specific parameter(s) via pillar 4's `AdaptiveParameterResolver`, only where the degradation signal mathematically justifies a specific adaptation — never a blanket re-tune. |
| **C — Edge disappeared** | Degradation persists after Category A/B responses, or the measurable evidence directly contradicts the strategy's Theory document (pillar 1) | Stop trading this strategy. Do not continue indefinitely; do not retry without a human review that updates or re-validates the Theory document. |

Category assignment SHALL itself be evidence-based and deterministic
(e.g. "volatility percentile outside the 5th-95th range the strategy's
Theory document declares as its operating range" → Category A; "expectancy
has been negative for N trades AND parameter X's current value predates
the current volatility regime" → Category B; "expectancy remains negative
after a Category B adaptation was applied and given M trades to prove out"
→ Category C). The exact numeric thresholds per signal are a
certification-phase, per-strategy decision (see Non-Goals) — this design
fixes the three-category *structure* and the requirement that
classification be measurable and documented, not the specific cutoffs.

Every strategy SHALL be able to continuously answer, from measurable
evidence: Why am I profitable? Why am I no longer profitable? Can I adapt?
Should I wait? Should I stop? — never merely "I am losing."

This never force-closes an existing position (mirrors how
`PortfolioRiskService`'s loss caps already only block new trades, not
liquidate open ones — see Risks). Distinct from and complementary to the
portfolio-level circuit breaker (`add-trading-safety-boundaries`,
aggregate exposure across bots) and `StrategyCapacityService` (allocation
limits) — neither tracks a single strategy's own performance or why it
changed; the audit found zero pillar-7 coverage in any of the 6
strategies.

**8. Self-Diagnostics** — Mostly exists via `ExplanationBuilder`/
`_explain()`, used by all 6 strategies for live, structured, current-tick
diagnostics. Two gaps found consistently across the audit: (a) some
decision points have no `.check()`/`.metric()` call at all (e.g. sizing
math, fee-viability gates, kill-switch triggers — see each strategy's audit
for specifics), and (b) the full structured `DecisionExplanation` is held
in an **in-memory-only, current-state-only** `DiagnosticsStore`
(`app/services/diagnostics.py`) — not persisted per trade, so a historical
trade can only be explained by its terse `Order.reason` string, not the
full evidence that produced it. Remediation: (a) close per-strategy
`.check()` gaps as part of each strategy's certification phase; (b)
persist a summarized `DecisionExplanation` — including the pillar-3
Evidence Report and, once triggered, any pillar-7 category classification
— onto the `Order`/`Trade` record at execution time, once, as shared
infrastructure (not per-strategy).

**9. Performance Expectations** — Documentation only (Strategy Audit
Document), like pillar 1. Validated empirically against real walk-forward
results once `add-strategy-validation-tooling` exists — a strategy whose
real behavior contradicts its own documented expectation is itself a
finding (and a likely Category C signal for pillar 7), not just an unmet
target.

**10. Strategy Proposal Interface** (NEW — added in this revision, see
"The Strategy Proposal Contract" below for the full specification) —
Every strategy SHALL return a `StrategyProposal`, not execute a trade
directly. Pillars 2-8's outputs (suitability result, Decision Score +
Evidence Report, resolved adaptive parameters, edge status, diagnostics)
are not separate, ad hoc return values each strategy wires up its own
way — they are the required fields of one stable, shared object. This is
the permanent contract between every strategy and everything downstream
(today's execution pipeline, and — not built in this change — a future
Auto Mode investment committee, portfolio allocation, and any future
external signal source). A strategy becomes an analyst that recommends;
it never executes.

## The Strategy Proposal Contract

### Why this exists

Without a stable output contract, redesigning Auto Mode into an
"investment committee" that compares proposals from multiple strategies
would require touching every strategy's return type a second time — once
for this framework (pillars 1-9), and again when Auto Mode needs richer
per-strategy output than a bare buy/sell action. `StrategyProposal` is
designed once, now, so that never has to happen: everything a future
comparison/selection/allocation layer could need is already a field on
this object, populated by the SAME pillar 2-8 machinery this change
already specifies — no new computation is invented here, only a
stable container for outputs that already exist per-pillar.

This directly formalizes something the codebase already gestured at and
stopped short of: `TradeSignal` (`trading_engine.py:199-255`) already has
`score`/`threshold` fields explicitly commented "Observability only (not
used for trading)," and `expected_move_pct`/`expected_risk_pct` fields
used only by the central `evaluate_reward_risk` gate. `StrategyProposal`
is what those fields were reaching for: a scored, explainable,
richly-described recommendation, not a bare instruction — and unlike
`TradeSignal`'s score/threshold, its fields are load-bearing, not
observability-only.

### Field specification

`StrategyProposal` SHALL contain:

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | str | Deterministic identifier derived from `(bot_id, strategy_id, candle_timestamp)` — NOT a random UUID, so re-running the same inputs reproduces the same id (Non-Negotiable: nothing subjective, nothing non-reproducible, see Decisions). |
| `strategy_id` | str | e.g. `"trend_following"`. |
| `bot_id` | int | Which bot instance produced this. |
| `generated_at` | datetime | The candle/tick timestamp the proposal was computed against — the determinism anchor; re-evaluating the same historical timestamp with the same state must reproduce the same proposal. |
| `direction` | enum | `BUY` \| `SELL` \| `HOLD` \| `NO_TRADE` — WHICH WAY. Extensible to `SHORT`/`COVER` later (see "Execution Intent" below) without changing the rest of the contract. |
| `execution_intent` | enum | `NO_ACTION` \| `OPEN_POSITION` \| `ADD_TO_POSITION` \| `REDUCE_POSITION` \| `CLOSE_POSITION` \| `HOLD_POSITION` — WHAT portfolio action, orthogonal to direction. See "Execution Intent" below. |
| `validity` | ProposalValidity | When this proposal stops being executable. See "Proposal Validity and Expiration" below. |
| `decision_score` | DecisionScore | Pillar 3's output in full: total (0-100), configured threshold, and the complete list of Evidence Items with their individual contributions (the Evidence Report is this field, rendered). |
| `market_suitability` | MarketSuitabilityResult | Pillar 2's output: whether current regime matched the strategy's declared `allowed_regimes`, and the regime tags evaluated against. |
| `edge_status` | EdgeStatus | Pillar 7's output: current category (`NONE` \| `A` \| `B` \| `C`), the measurable signal(s) that produced that classification, and the category's bounded action. |
| `suggested_position_size` | Optional[float] | Quote-currency notional, as computed by pillar 5's shared, Decision-Score-weighted sizer. **Advisory** — see "Recommendation, not an order" below. |
| `suggested_risk_budget_pct` | Optional[float] | % of this bot's capital the suggested size risks. Carried separately from the absolute size so a future capital-allocation layer can rescale the notional without re-deriving the strategy's own risk assessment. |
| `expected_holding_horizon` | Optional[str] | e.g. `"short"` / `"medium"` / `"long"`, or an explicit estimated bar count — from the strategy's own theory/typical behavior, not a guess per trade. |
| `expected_edge_estimate` | Optional[EdgeEstimate] | Expectancy/win-rate/profit-factor. **SHALL be populated ONLY from `add-strategy-validation-tooling`'s statistically-validated walk-forward/backtest results for this exact strategy configuration — NEVER computed live by the strategy from its own recent trades.** See "Expected Edge Requires Statistical Validation" below; this corrects an imprecision in the prior revision, which suggested this could tie to pillar 7's live rolling window. Per-trade reward/risk (what `TradeSignal.expected_move_pct`/`expected_risk_pct` captured) is a different, legitimately per-trade-computable concept and belongs in `decision_score`'s Risk/Reward Evidence Item instead — not here. |
| `assumptions` | List[str] | The objective, falsifiable conditions that justified this proposal — see "Market Assumptions" below. |
| `reasons_for` | List[str] | Mechanically derived from `decision_score`'s positive-contribution Evidence Items — not independently authored text (see Decisions: this avoids reintroducing subjectivity through the back door). |
| `reasons_against` | List[str] | Mechanically derived from `decision_score`'s negative-contribution Evidence Items plus any active `edge_status` warning. |
| `adaptive_parameters_used` | Dict[str, float] | Pillar 4's resolved parameter values for this evaluation cycle (e.g. `{"atr_stop_multiplier": 2.3}`), so a downstream consumer (or a human reviewing a historical proposal) can see exactly what the strategy used, not just that "parameters are adaptive." |
| `explanation` | DecisionExplanation | Pillar 8's full structured, persisted diagnostic object (existing `ExplanationBuilder` output, extended per pillar 8 above). |

### Execution Intent

Direction alone ("BUY"/"SELL") is ambiguous about the actual portfolio
action: a `BUY` could mean opening a brand-new position or adding to one
already open; a `SELL` could mean a full exit or a partial reduction.
`execution_intent` makes the portfolio action explicit and separates it
from directional bias, giving a two-axis contract — WHICH WAY
(`direction`) and WHAT ACTION (`execution_intent`):

| `direction` | `execution_intent` | Meaning |
|---|---|---|
| `NO_TRADE` | `NO_ACTION` | No position exists; nothing recommended. |
| `HOLD` | `HOLD_POSITION` | Position exists; maintain as-is (pillar 6 concluded no change). |
| `BUY` | `OPEN_POSITION` | Recommend a new long entry. |
| `BUY` | `ADD_TO_POSITION` | Recommend scaling into an existing long. |
| `SELL` | `REDUCE_POSITION` | Recommend a partial exit of an existing long. |
| `SELL` | `CLOSE_POSITION` | Recommend a full exit of an existing long. |

Only these pairings are valid; certification SHALL test that a strategy
never produces an inconsistent pairing (e.g. `NO_TRADE` +
`OPEN_POSITION`). This table is exactly why the design remains
forward-compatible with short-selling/futures without another redesign:
`execution_intent`'s six values are side-agnostic — they mean the same
thing whether the position is long or short. Extending `direction` with
`SHORT`/`COVER` later (`add-short-selling-support`) adds new rows to this
table using the SAME six `execution_intent` values
(`SHORT`+`OPEN_POSITION`, `SHORT`+`ADD_TO_POSITION`,
`COVER`+`REDUCE_POSITION`, `COVER`+`CLOSE_POSITION`) — no new
`execution_intent` value, no field changes, no strategy touched. Futures
support (naming aside) is the same extension: a new `direction` value
paired with the existing `execution_intent` vocabulary. This change does
not implement short-selling or futures — it only ensures the contract
does not need to be redesigned when they arrive.

### Proposal Validity and Expiration

Auto (and the Standalone Adapter) SHALL NEVER execute a stale proposal.
Every `StrategyProposal` carries a `validity.valid_until` timestamp,
computed deterministically as `generated_at` plus the strategy's
evaluation interval (its timeframe) — this covers "valid until next
candle" and "valid until the current timeframe closes" as the same
mechanism: a proposal generated on any candle is, at the latest, expired
once the next candle for that strategy's timeframe has begun. Any
consumer SHALL check `now < validity.valid_until` before executing and
SHALL discard (never execute) an expired proposal.

A proposal is ALSO implicitly superseded — and must be discarded even
before `valid_until` — the instant the strategy produces a NEWER proposal
for the same bot. This is how "valid until market assumptions become
invalid" is satisfied without requiring Auto to evaluate assumptions
itself (see "Market Assumptions" below): the strategy re-checks its own
assumptions every evaluation cycle as part of pillar 6's continuous
re-evaluation, and if one breaks, its next proposal reflects that (a
changed `direction`/`execution_intent`, a changed `edge_status`, or both)
— consumers always act on the latest proposal for a bot, never a cached
older one. `valid_until` is the outer, time-based bound; supersession by
a fresher proposal is the inner, condition-based bound. Whichever comes
first governs.

### Market Assumptions

Every proposal's `assumptions` SHALL be objective and falsifiable —
concrete, measurable conditions the strategy itself can and does
re-evaluate on its own next cycle (e.g. "EMA(20) remains above EMA(50)
[trend intact]", "ATR percentile remains above the 60th [volatility above
threshold]", "price remains above $X [support not broken]"), never a
subjective concept ("feels like the trend continues"). Assumptions are
drawn from pillar 1's Theory document and pillar 2's regime match, not
freely authored per trade.

Auto SHALL NOT independently evaluate whether a listed assumption still
holds — that would require Auto to understand the strategy's own
indicators, which "Auto Must Never Understand Strategy Indicators" below
forbids. Assumptions exist for two purposes only: (1) audit/explainability
(pillar 8 — a human or the strategy's own next cycle can see exactly what
was assumed), and (2) as the documented basis for the supersession
behavior described in "Proposal Validity and Expiration" above — the
strategy, not Auto, is responsible for detecting when its own assumption
has broken and issuing an updated proposal.

### Proposal Immutability

A `StrategyProposal`, once constructed, is **immutable**. This is a
formal architectural rule, not a convention: Auto (or any future
consumer) SHALL NEVER modify a proposal's `decision_score`, `evidence`
(the Evidence Report inside `decision_score`), `suggested_position_size`
or `suggested_risk_budget_pct`, `reasons_for`/`reasons_against`,
`adaptive_parameters_used`, `market_suitability`, `edge_status`, or any
other field, for any reason — including to apply external context (see
"External Trust Layer" below) or to resize an allocation. Auto only ever
decides WHAT TO DO with a proposal (execute, reject, resize the actual
order, deprioritize) — it records that decision in a separate
`CommitteeDecision` (below), never by rewriting the proposal. This
preserves an honest, permanent audit trail: what the strategy itself
concluded, using only its own market data, remains inspectable exactly as
produced, forever separate from what was later done with it.

### Committee Decision (future contract — specified now, not implemented)

Auto Mode's future redesign consumes `StrategyProposal` objects and
produces a separate `CommitteeDecision` — proposals are never mutated
into decisions, they are referenced by one. Specifying this shape now
(without building it) means the eventual Auto Mode change has a stable
target and never needs to renegotiate what a strategy hands it:

| Field | Type | Meaning |
|---|---|---|
| `decision_id` | str | Deterministic identifier for this committee evaluation cycle. |
| `evaluated_at` | datetime | When the committee ran. |
| `proposals_considered` | List[str] | `proposal_id` references only — the committee never copies or embeds proposal content. |
| `selected` | List[SelectedAllocation] | `{proposal_id, allocated_size, execution_priority}` for each proposal chosen to execute. |
| `rejected` | List[RejectedProposal] | `{proposal_id, rejection_reason}` for each proposal considered and declined — reasons are as measurable/documented as everything else in this framework, never "didn't like it." |
| `trust_adjustments_applied` | List[str] | References to the `TrustAdjustment` records (see below) the committee consulted for this cycle, for audit. |

Not implemented by this change or any of its phases — this schema exists
so Phase 0-6 never has to guess what shape their `StrategyProposal`
output will eventually be consumed into. **`add-auto-mode-investment-
committee` (sibling change, architecture-only) fully specifies this
shape** (adding `rejection_step` detail on `RejectedProposal` and a
`ranking_snapshot` field) plus the entire ten-step process that produces
it — this section remains here for `StrategyProposal`-side context, but
is no longer the canonical source for `CommitteeDecision`'s definition.

### Auto Must Never Understand Strategy Indicators

Formal architectural rule: Auto SHALL NEVER inspect EMA, ATR, MACD, RSI,
Bollinger Bands, or any other strategy-specific indicator, computation, or
internal state. Indicator interpretation belongs permanently to
strategies. Auto may only ever reason about the Comparison Contract fields
below — this is not a style preference, it is the mechanism that keeps
strategies swappable, comparable, and independently certifiable without
Auto's own logic ever needing to change when a strategy's internals do.

### The Comparison Contract

The following subset of `StrategyProposal` is guaranteed sufficient for
cross-strategy comparison and is the ONLY subset a comparison/ranking
layer is allowed to depend on:

- `direction` and `execution_intent`
- `decision_score.total` and `decision_score.threshold` (always the same
  0-100 scale regardless of strategy, by pillar 3's own requirement)
- `suggested_risk_budget_pct`
- `expected_edge_estimate` (when present — see "Expected Edge Requires
  Statistical Validation")
- `market_suitability.is_suitable`
- `edge_status.category`
- `validity.valid_until`

Every other field (`decision_score`'s individual Evidence Items,
`adaptive_parameters_used`, `explanation`) exists for audit/explainability
(pillar 8) and for a human reviewing a specific strategy, not for
cross-strategy ranking. This split is what lets Auto Mode compare a
mean-reversion band-touch against a trend-following breakout without ever
knowing what a Bollinger Band is.

### External Trust Layer

Fear & Greed, macro regime, news/sentiment analysis, social sentiment,
funding rates, futures/options information, and portfolio concentration
are all explicitly **not** fields on `StrategyProposal` and never will
be. A strategy cannot know about them (pillars 1-9's scope is a
strategy's own market data), and baking them into the proposal would mean
every new external signal source requires touching every strategy again —
exactly the problem this contract exists to prevent.

Instead, any future consumer produces an independent `TrustAdjustment`
that references, but never modifies, the original proposal:

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | str | Which proposal this adjusts. |
| `source` | str | e.g. `"sentiment"`, `"macro_regime"`, `"funding_rate"`. |
| `adjustment` | float | A multiplier or delta applied only to the committee's ranking/selection of this proposal — never to `decision_score` or any other proposal field. |
| `generated_at` | datetime | |

```
StrategyProposal  +  TrustAdjustment(s)  ---->  CommitteeDecision
   (immutable,         (external context,          (references both,
    strategy-only        FUTURE, not built)          FUTURE, not built)
    data)
```

The proposal remains an unmodified, permanent record regardless of how
many external systems later weigh in on what to do with it. Not
implemented by this change — specified so a future external-signal
integration never requires touching `StrategyProposal` or any strategy.
**`add-auto-mode-investment-committee`** confirms this layer is owned
exclusively by Auto (never by strategies) and specifies exactly where in
its Committee Process a `TrustAdjustment` is consumed (step 6, ranking
only) — see that change for the canonical specification.

### Expected Edge Requires Statistical Validation

A strategy SHALL NEVER invent or self-report an expected edge.
`expected_edge_estimate` remains `None` until the strategy has gone
through `add-strategy-validation-tooling`'s walk-forward/backtest
validation process for its exact certified configuration, at which point
that tooling — and only that tooling — may populate it. A strategy's own
live `StrategyEdgeManager` rolling statistics (pillar 7) exist to detect
degradation in an ALREADY-VALIDATED edge; they are not a substitute
source for establishing that an edge exists or its expected magnitude,
and SHALL NOT populate this field. An unvalidated strategy proposing
trades with no `expected_edge_estimate` is the correct, honest state —
not a gap to work around.

### Recommendation, not an order

A `StrategyProposal` is a recommendation, never a binding instruction.
`suggested_position_size` and `suggested_risk_budget_pct` are exactly
that — suggested. The existing execution pipeline (portfolio risk caps,
strategy capacity limits, the viability gate, cost estimation — all
already built) remains the sole authority on whether and how much
actually executes, in both the standalone and future Auto Mode paths
below. A strategy's job ends at producing a well-evidenced proposal; it
never decides that its own proposal executes.

### Standalone execution flow (Auto Mode disabled — today's behavior, preserved)

```
Market tick
    |
    v
Strategy function (pillars 2-8 run internally, as already specified)
    |
    v
StrategyProposal(direction, execution_intent, validity, decision_score,
                  market_suitability, edge_status, suggested_position_size,
                  assumptions, ...)                       [IMMUTABLE]
    |
    v
Standalone Adapter (NEW - thin, no ranking/comparison: exactly one
                     proposal exists, so it either executes or it doesn't)
    |  now >= validity.valid_until? -> expired, discard, no order.
    |  execution_intent == NO_ACTION or HOLD_POSITION? -> no order, done.
    |  execution_intent in {OPEN_POSITION, ADD_TO_POSITION,
    |     REDUCE_POSITION, CLOSE_POSITION}? -> extract order intent
    |     (direction + suggested_position_size)
    v
_execute_trade()  [UNCHANGED: portfolio risk check, strategy capacity
                    check, execution cost estimate, viability gate,
                    per-bot risk checks, execution - Steps 1-8 exactly
                    as they exist today]
    |
    v
Order / Position
```

A single-strategy bot's *outcome* is unchanged from today — same checks,
same execution mechanics, same result. Only the shape of what the
strategy function returns changed, and a small adapter (not Auto Mode)
unwraps it. This is exactly the "only the interface changes" requirement.
The adapter never rewrites the proposal (Proposal Immutability) — it only
reads it to decide whether to call the unchanged execution pipeline.

### Future Auto Mode execution flow (NOT implemented by this change — shown to prove the contract supports it without strategy changes)

**Fully specified by `add-auto-mode-investment-committee`** (a sibling
change, architecture-only, also frozen — see that change's `design.md`
for the complete ten-step Committee Process, ranking/tie-breaking policy
shape, `CommitteeDecision`/`TrustAdjustment` field specifications, and
Auto's own certification requirements). The sketch below predates that
specification and is kept here only to show that this change's
`StrategyProposal` contract was designed to support it without requiring
any strategy to be touched again — for Auto Mode's actual internal
design, defer to `add-auto-mode-investment-committee`.

```
Market tick
    |
    +----------------+----------------+----------------+
    v                v                v                v
Strategy A       Strategy B       Strategy C       Strategy D ...
    |                |                |                |
    v                v                v                v
Proposal A       Proposal B       Proposal C       Proposal D
 [IMMUTABLE]      [IMMUTABLE]      [IMMUTABLE]      [IMMUTABLE]
    |                |                |                |
    +----------------+--------+-------+----------------+
                               v
              Discard any proposal past validity.valid_until
              (Auto never executes a stale proposal)
                               v
                Auto Mode investment committee
                [FUTURE - specified in add-auto-mode-investment-
                 committee, not built here]
                               |
        +----------------------+-----------------------+
        v                      v                        v
  Compare using ONLY the  Consult TrustAdjustment(s)  Apply portfolio
  Comparison Contract     per proposal - external      constraints (the
  fields (direction,      context (sentiment, macro,   ALREADY-BUILT
  execution_intent,       funding/futures, social) -    PortfolioRiskService/
  decision_score.total,   FUTURE, not built here.        StrategyCapacityService)
  edge_status.category,   References proposals, NEVER
  ...) - NEVER inspects   modifies them.
  a strategy's own
  indicators (formal rule)
        +----------------------+-----------------------+
                               v
        Selection: execute one / execute none /
                   split capital across several /
                   reject for portfolio reasons
                               |
                               v
                  CommitteeDecision(selected, rejected,
                    allocation, execution_priority,
                    rejection_reasons)     [FUTURE, not built]
                    -- proposals A-D remain unmodified;
                       this is a SEPARATE record referencing them --
                               |
                               v
              chosen StrategyProposal(s), per CommitteeDecision.selected
                               |
                               v
                  _execute_trade()  [SAME pipeline, UNCHANGED]
                               |
                               v
                        Order / Position
```

The only new component Auto Mode's future redesign has to build is the
"investment committee" box — comparison, external-context weighting, and
selection, producing a `CommitteeDecision`. It consumes `StrategyProposal`
objects exactly as strategies already produce them today (per this
change), never modifies them, and the execution pipeline below the
committee is identical to the standalone path. No strategy is touched
again.

### Future capabilities this unlocks without further strategy changes

Because these all consume only `StrategyProposal` (via the Comparison
Contract, or the full object for audit purposes) rather than any
strategy-specific code:
- Auto Mode redesign as a comparison/selection/allocation committee.
- Portfolio-level capital allocation across multiple simultaneously-valid
  proposals.
- Sentiment / news / social-media analysis as an external trust-adjustment
  layer.
- Macro regime / Fear & Greed Index as an external trust-adjustment layer.
- Funding-rate / futures information as an external trust-adjustment
  layer — a natural extension point for `add-short-selling-support`'s
  future short-side proposals (the `direction` enum is designed to be
  extended with `SHORT`/`COVER` values later without changing the rest of
  the contract).
- Portfolio concentration/correlation-aware rejection.
- Multi-strategy backtesting/reporting that compares strategies on common
  terms (a natural extension for `add-strategy-validation-tooling`).
- Any future change to HOW proposals are ranked or selected (e.g. a
  learned ranking model later) touches only the committee, never a
  strategy.
- The `CommitteeDecision` and `TrustAdjustment` shapes are already
  specified (see above), so Auto Mode's eventual implementation starts
  from a fixed target rather than inventing its output contract from
  scratch when that change is finally approved.

## Certification Process

A strategy may not be implemented or materially modified until:
1. Its Strategy Audit Document exists/is updated (pillars 1, 9, every
   pillar-4 fixed-parameter justification, and every pillar-3 Evidence
   Item's Measurement/Normalization/Weight/Reason).
2. It passes the certification checklist (this change's spec, "Strategy
   Certification" requirement): market suitability gate present and
   actually enforced, Evidence-Based Decision Score present with a
   configured threshold and every Evidence Item documented, adaptive-
   parameter resolver used for volatility-sensitive parameters,
   Decision-Score-weighted sizing wired, trade management hook present,
   Strategy Edge Management present with all three categories reachable
   and tested, diagnostics gaps closed including a working Evidence
   Report, and — pillar 10 — that the strategy returns a valid
   `StrategyProposal` satisfying ALL of:
   - **Deterministic**: identical inputs (market data + prior state)
     produce an identical proposal, including `proposal_id`.
   - **Immutable**: no test or code path modifies a proposal after
     construction.
   - **Assumptions documented**: `assumptions` is non-empty for any
     proposal with `execution_intent != NO_ACTION`, and every entry is
     objective/falsifiable (no subjective language — see "Market
     Assumptions").
   - **Expiration defined**: `validity.valid_until` is set and a test
     proves an expired proposal is never executed by the Standalone
     Adapter.
   - **Execution intent consistent**: only valid `direction` +
     `execution_intent` pairings (see "Execution Intent") are ever
     produced.
   - **Evidence measurable**: every `decision_score` Evidence Item has a
     documented Measurement/Normalization/Weight/Reason (restates pillar
     3's own requirement, checked again here at the proposal boundary).
   - **Explanation reproducible**: `explanation` (and the Evidence Report
     inside `decision_score`) is byte-identical across two evaluations of
     the same historical inputs.
   - **No subjective information**: no field (`reasons_for`/
     `reasons_against` included) contains freely-authored text not
     mechanically derived from a measurable source.
   - **Expected edge sourced correctly**: `expected_edge_estimate` is
     either `None` or was populated by `add-strategy-validation-tooling`
     — never self-computed by the strategy.
3. This becomes a standing requirement for any *future* `add-<strategy>`/
   `fix-<strategy>`/`update-<strategy>` OpenSpec proposal, referenced from
   this capability's spec — not just a one-time audit of the current 6.

## Decisions

- Decision: build ONE shared `app/services/strategy_framework/` package
  (`market_suitability.py`, `decision_score.py`, `adaptive_params.py`,
  `trade_management.py`, `edge_management.py`) that all 6 strategies
  compose into, rather than 6 independent implementations.
  - Why: the audit's core finding is that ad hoc, duplicated,
    inconsistently-applied logic (3 sizing conventions, 2 regime detectors,
    a computed-but-unenforced suitability check) is *why* strategies don't
    behave consistently. Centralizing removes the option to silently skip
    a pillar, and matches the precedent already set by
    `PortfolioRiskService`/`StrategyCapacityService` being shared services
    rather than per-strategy copies.
- Decision: replace "Confidence" with "Evidence-Based Decision Score,"
  built entirely from Evidence Items that are objectively measurable,
  reproducible, deterministic, computable, and documented.
  - Why: a trading system cannot "feel confident" — that language invites
    exactly the kind of unmeasurable, unreproducible judgement the audit
    is trying to eliminate. Every prior pillar-3 mention of "confidence"
    is replaced throughout this change's files.
- Decision: the Decision Score is a transparent, rule-based weighted sum
  over documented Evidence Items, not a learned model, at least initially.
  - Why: pillar 8 requires every score to be explainable to a human
    trader via a concrete Evidence Report; a black-box score would fail
    that bar without separate interpretability work not proposed here.
- Decision: replace "Failure Detection" with "Strategy Edge Management" —
  continuous monitoring that classifies *why* performance is degrading
  into three categories (temporary mismatch / parameter mismatch / edge
  gone), each with a specific bounded action, rather than a single
  win/lose signal.
  - Why: "I am losing, therefore stop" throws away information a
    professional trader would use — most degradation is regime-driven and
    temporary (Category A) or a calibration problem (Category B), not
    evidence the edge itself is gone (Category C). Collapsing all three
    into one binary response either overreacts (stopping a strategy that
    would recover) or underreacts (never stopping a genuinely dead
    strategy). The categories map directly to the three responses a
    professional trader actually has available: wait, adapt, or stop.
- Decision: remediate strategy-by-strategy in phases (see tasks.md),
  ordered by which strategies have zero *enforced* market suitability
  first, then by implementation complexity — not a big-bang rewrite of all
  6 at once, and not strictly by raw pillar-count score (see tasks.md's
  phase-ordering rationale for the full reasoning).
  - Why: matches how this session's earlier Tier 1 fixes were validated
    one at a time (backtest before/after each change, not a bundled
    rewrite) — smaller blast radius per change, independently reviewable
    and backtestable.
- Decision: `StrategyProposal` REPLACES `TradeSignal` as what a strategy
  function returns, rather than a second parallel object strategies
  produce alongside it.
  - Why: the user's framing is explicit ("Strategies must no longer
    return a trade decision") and a stable single contract is the entire
    point — maintaining two return shapes long-term would recreate the
    "ad hoc, inconsistent" problem pillars 1-9 exist to eliminate.
    `TradeSignal`'s existing fields all map onto `StrategyProposal`
    fields (`action`→`direction`+`execution_intent`,
    `amount`→`suggested_position_size`, `reason`→`reasons_for`/
    `reasons_against` + `explanation`, `score`/`threshold`→
    `decision_score`, `expected_move_pct`/`expected_risk_pct`→ a
    Risk/Reward Evidence Item inside `decision_score`, NOT
    `expected_edge_estimate` — that field is reserved exclusively for
    `add-strategy-validation-tooling`'s statistically-validated results,
    see "Expected Edge Requires Statistical Validation"), so nothing
    existing is lost, only formalized and enriched. The
    Standalone Adapter (see "The Strategy Proposal Contract") is what
    translates a proposal into the order intent `_execute_trade` already
    expects, so the execution pipeline itself does not need to change
    shape — only its input's origin does.
- Decision: `reasons_for`/`reasons_against` are mechanically derived from
  `decision_score`'s Evidence Items, never independently authored strings.
  - Why: allowing a strategy to write free-text reasons alongside a
    numeric score would reopen exactly the "looks bullish" subjectivity
    hole pillar 3 closes — if a reason isn't backed by a scored Evidence
    Item, it isn't a real reason.
- Decision: external context (sentiment, macro, funding, portfolio
  concentration, etc.) is never a field on `StrategyProposal` — it is
  applied by a future consumer as a separate layer that references, but
  never mutates, the original proposal.
  - Why: this is the specific mechanism that satisfies "future inputs can
    influence execution WITHOUT requiring strategy redesign" — if
    external signals were proposal fields, every new signal source would
    mean touching every strategy again, recreating the exact problem this
    change exists to prevent. It also preserves an honest audit trail:
    what the strategy itself concluded from its own data stays visible
    and unmodified, separate from what a future committee did with it.
- Decision: `proposal_id` is a deterministic hash of `(bot_id,
  strategy_id, candle_timestamp)`, not a random UUID.
  - Why: "the proposal must be entirely deterministic... nothing
    subjective" (user's explicit requirement) applies to the whole
    object, including its identity — a random id would make two
    otherwise-identical replayed evaluations produce "different"
    proposals for no evidentiary reason.
- Decision: `direction` and `execution_intent` are two separate fields
  (WHICH WAY vs. WHAT ACTION) rather than one combined enum.
  - Why: a combined enum (e.g. `OPEN_LONG`, `ADD_LONG`, `CLOSE_LONG`,
    `OPEN_SHORT`, ...) doubles in size every time a new direction is
    added, and every consumer that only cares about the action (e.g. "is
    this exiting risk?") would still have to enumerate every
    direction-qualified variant. Two orthogonal fields let `direction`
    grow (SHORT/COVER) without `execution_intent`'s six values ever
    changing, satisfying "forward-compatible... without requiring
    redesign" directly.
- Decision: `validity.valid_until` is a deterministic timestamp
  (`generated_at` + evaluation interval), with supersession by a newer
  proposal as an additional, tighter, inner bound — not a set of
  strategy-specific condition checks Auto must evaluate.
  - Why: the alternative (Auto checking "is the trend still intact?"
    itself) would require Auto to understand strategy-specific
    assumptions, directly violating "Auto must never understand strategy
    indicators." A deterministic timestamp plus "a fresher proposal
    always wins" gives Auto a purely mechanical staleness check while the
    strategy itself (which does understand its own assumptions) is the
    one that re-validates them each cycle and produces the superseding
    proposal.
- Decision: Proposal Immutability is enforced as a documented contract
  (a formal rule strategies and Auto are certified/reviewed against), not
  by a runtime-enforced frozen/read-only object at this stage.
  - Why: this change specifies architecture, not implementation
    (explicit instruction: "Do NOT implement code"). Whether immutability
    is additionally enforced at runtime (e.g. a frozen dataclass) is a
    Phase 0 implementation detail left to whoever builds the
    `StrategyProposal` schema — the contract-level requirement is fixed
    now regardless of the eventual enforcement mechanism.
- Decision: `CommitteeDecision` and `TrustAdjustment` are specified now,
  as part of this change, even though Auto Mode itself is out of scope
  and explicitly not being implemented.
  - Why: directly serves "no future strategy implementation should
    require architectural redesign" — without fixing these shapes now,
    Phase 0-6 would have to guess at what a future committee needs from a
    proposal, and Auto Mode's eventual design would risk discovering it
    needs a proposal field that doesn't exist, reopening every strategy's
    certification. Specifying the consumer contract now, without building
    the consumer, costs nothing at implementation time and removes that
    risk entirely.

## Risks / Trade-offs

- An Evidence-Based Decision Score threshold will, by construction, reduce
  trade frequency for every strategy — fewer trades clear a multi-factor
  bar than a single indicator cross. This is the user's explicit stated
  goal ("no trade is often the optimal decision"), but each strategy's
  certification must backtest before/after trade count to confirm it
  doesn't collapse to zero trades in reasonable market conditions, not
  just assume less is better.
- Pillar 5's Decision-Score-weighted sizing depends on `add-trading-
  safety-boundaries` (done) and requires revising `add-unified-position-
  sizing` before that change is implemented — sequencing risk if that
  change proceeds independently first. Flagged explicitly in this change's
  tasks and in that change's own tracking.
- Pillar 7's Strategy Edge Management, if the classification thresholds
  are set carelessly during a strategy's certification, could either (a)
  misclassify a genuine Category C edge-loss as Category A/B and keep
  trading a dead strategy, or (b) misclassify a temporary Category A
  regime mismatch as Category C and stop a strategy that would have
  recovered. Mitigation: classification logic and its thresholds must be
  backtested against historical degradation episodes during certification,
  not assumed correct on first implementation — this is explicitly listed
  in each phase's acceptance criteria.
- Building shared infrastructure before any strategy uses it means the
  first strategy's certification phase (Phase 1) carries the cost of
  building AND integrating the framework, making it look disproportionately
  large compared to later phases. This is intentional and unavoidable —
  see tasks.md Phase 0.
- Replacing `TradeSignal` with `StrategyProposal` touches the return type
  every strategy function and `_execute_trade`'s caller depend on — a
  wide-blast-radius change if done carelessly. Mitigation: the Standalone
  Adapter is designed specifically to isolate this — `_execute_trade` and
  everything inside it (portfolio risk, capacity, cost model, viability
  gate) never changes; only a thin translation step is added ahead of it.
  Each strategy's Phase 1-6 certification migrates one strategy's return
  type at a time, backtested before/after, rather than a single flag-day
  cutover of all 6.
- A `StrategyProposal` that is well-evidenced by pillars 1-9 but wrong
  about `suggested_position_size` or `suggested_risk_budget_pct` is still
  just as risky as a bad `TradeSignal.amount` was — pillar 10 does not by
  itself reduce sizing risk. Mitigation: unchanged — the existing
  execution-pipeline checks (portfolio risk caps, capacity limits, cost
  model, viability gate) remain the actual authority over what executes,
  exactly as "Recommendation, not an order" specifies.
- `validity.valid_until`'s outer bound relies on each strategy choosing a
  sane evaluation interval — a strategy that sets it too long risks
  executing a proposal against assumptions that broke mid-interval; too
  short risks discarding still-valid proposals as falsely stale. Mitigation:
  left as a per-strategy, per-timeframe certification-phase decision (see
  "Open Questions"), backtested like every other threshold, not fixed
  globally in this design.
- Proposal Immutability is a discipline enforced by contract and
  certification review at this stage, not (yet) by a runtime guarantee —
  a careless future implementation could still mutate a `StrategyProposal`
  in place before this is backed by a frozen/read-only object. Mitigation:
  flagged explicitly for Phase 0 to consider runtime enforcement (e.g. a
  frozen dataclass) when the schema is actually built, and the
  Certification Process's checklist requires reviewing immutability for
  every strategy regardless.

## Migration Plan

See `tasks.md`: Phase 0 (shared framework infrastructure, now including
the `StrategyProposal` schema — with `execution_intent` and `validity` —
and Standalone Adapter) → Phases 1-6 (one per strategy, ordered by
suitability-enforcement urgency then complexity, each migrating that
strategy from `TradeSignal` to `StrategyProposal`) → Auto Mode
implementation (specified in full, and separately frozen, by
`add-auto-mode-investment-committee`; not part of this change's scope,
but its architecture is now completely defined and no longer blocked on
this change beyond `StrategyProposal` itself being available to consume).

## Architecture Freeze

This design revision is the final architecture pass before implementation
begins. As of this revision, the `StrategyProposal` contract — its field
specification, Execution Intent, Proposal Validity and Expiration, Market
Assumptions, Proposal Immutability, the Comparison Contract, the External
Trust Layer, and the future `CommitteeDecision`/`TrustAdjustment` shapes —
is considered **frozen**.

`add-auto-mode-investment-committee` (sibling change) separately freezes
Auto Mode's own internal architecture — the ten-step Committee Process,
ranking/tie-breaking mechanism, portfolio-validation integration, and the
finalized `CommitteeDecision`/`TrustAdjustment` field specifications. The
two freezes together mean the full strategy-to-execution architecture is
complete: this change fixes what a strategy hands off, that change fixes
what happens to it next.

- Phase 0 through Phase 6 (`tasks.md`) SHALL implement against this
  contract as specified. They SHALL NOT introduce new top-level
  `StrategyProposal` fields, change the meaning of `direction` or
  `execution_intent`, or weaken proposal immutability.
- A future architecture change to this contract requires **demonstrated
  evidence** discovered during implementation or historical backtesting —
  e.g. a certification phase proving a specific field cannot express a
  real strategy's behavior, or a backtest showing the validity/expiration
  model produces incorrect staleness handling in practice. A hypothetical
  future capability ("we might want X later") is not sufficient
  justification to reopen this design; the contract was deliberately
  built with several such extension points (execution_intent's
  forward-compatibility with SHORT/COVER, the External Trust Layer for
  future signal sources, the CommitteeDecision contract for Auto Mode) so
  that most anticipated future needs do not require redesign at all.
- If a concrete limitation is discovered, it SHALL be raised as a new,
  separate OpenSpec change proposing a specific, evidenced amendment to
  this contract — not as an ad hoc change made silently during a
  certification phase's implementation.
- This freeze applies to the `StrategyProposal`/`CommitteeDecision`/
  `TrustAdjustment` contract shapes specifically. It does not freeze
  strategy-internal logic (Evidence Items, adaptive parameter formulas,
  edge-classification thresholds) — those remain intentionally Open
  (see "Open Questions" below) and are expected to be calibrated per
  strategy during certification.

## Open Questions

- Exact Decision Score scale/weights, per-strategy Evidence Item sets, edge
  degradation classification thresholds (what specifically separates
  Category A from B from C for a given signal), and adaptive-parameter
  formulas are proposed structurally here (0-100 score, configurable
  weights, three fixed categories with bounded actions, ATR/regime-scaled
  resolvers) but their actual numeric values are Open — to be calibrated
  via `add-strategy-validation-tooling` during each strategy's
  certification phase, not decided abstractly in this design doc.
- Should the Decision Score threshold itself be regime-adaptive (e.g.
  demand a higher score in a regime the strategy is less ideally suited
  to, blending pillars 2 and 3) rather than pillar 2 being a hard pass/fail
  gate? Deferred — start with a hard suitability gate (simpler, matches the
  user's literal framing: "If market conditions do not match... DO NOT
  TRADE") and revisit blending only if certification backtests show the
  hard gate is too blunt for a specific strategy.
- Where exactly should `StrategyEdgeManager`'s performance window live
  (rolling N trades vs. rolling N days), and what specific signals feed
  Category A vs. B vs. C for each strategy — likely strategy-specific
  given very different expected trade frequencies (dca trades hourly-ish,
  volatility_breakout trades a few times a month) — left to each
  strategy's certification phase to decide, not standardized here.
- Should a strategy that reaches Category C be able to re-enter
  certification automatically once conditions change, or does every
  Category C classification require a human-reviewed re-certification
  before the strategy trades again? Leaning toward the latter (matches
  "do not continue indefinitely... without a human review" in the pillar 7
  definition above) but not finalized — a certification-phase decision.
- Exact normalization/units for `expected_edge_estimate` across strategies
  with very different trade frequencies and typical move sizes (e.g. dca's
  per-buy expectancy is a different kind of number than
  volatility_breakout's per-trade expectancy) — needs a common unit (likely
  % expectancy per trade, annualized separately if needed) so a future Auto
  Mode can compare it via the Comparison Contract without strategy-specific
  interpretation. Left to `add-strategy-validation-tooling` and each
  strategy's certification phase, not fixed here.
- Minimum sample size before `expected_edge_estimate` is populated (rather
  than `None`) is unspecified — likely ties to the same rolling window
  `StrategyEdgeManager` (pillar 7) uses, but the exact count is a
  certification-phase, per-strategy decision, not standardized here.
- Whether the Standalone Adapter is truly Auto-Mode-agnostic infrastructure
  (Phase 0, shared) or turns out to need per-strategy special cases once
  actually implemented (e.g. how it should treat a `HOLD` proposal for a
  strategy where "holding" itself has side effects, like adaptive_grid's
  virtual inventory) — flagged for whoever implements Phase 0 to confirm
  against each of the 6 strategies' actual `_manage_exit`-style logic, not
  resolved here since this change implements no code.
- The exact evaluation-interval value used to compute `validity.valid_until`
  (e.g. one candle of the strategy's own timeframe vs. a fixed wall-clock
  duration) is left to each strategy's certification phase — the contract
  only fixes the mechanism (`generated_at` + interval, tightened by
  supersession), not the interval's value per strategy/timeframe
  combination.
- The exact allocation algorithm, rejection-reason taxonomy, and
  `TrustAdjustment` magnitude/decay rules inside a future `CommitteeDecision`
  are intentionally unspecified — only the contract shapes are frozen (see
  "Architecture Freeze"). Auto Mode's own future change proposal will
  design that logic; this change only guarantees it will have a fixed,
  already-proven-workable set of inputs to design against.
