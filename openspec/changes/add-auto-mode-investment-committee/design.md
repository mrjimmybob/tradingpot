## Context

`add-strategy-decision-framework` (frozen) redefined every non-Auto-Mode
strategy as an independent market expert: each evaluation cycle, a
strategy produces exactly one immutable `StrategyProposal` — a
recommendation, never an order — and stops. That change specified the
`StrategyProposal` contract completely (direction, execution_intent,
validity, decision_score, market_suitability, edge_status, suggested
sizing, assumptions, reasons, adaptive parameters used, explanation) and
stubbed two forward-looking shapes, `CommitteeDecision` and
`TrustAdjustment`, as a "future contract — specified now, not
implemented," so Phase 0-6 strategy implementation would never have to
guess what its output is eventually consumed into.

That stub was deliberately shallow — enough to fix the shape strategies
write to, not enough to actually build Auto Mode from. This change
finishes the other half: it specifies, in full, what Auto Mode itself
does with the proposals it receives. Auto Mode is no longer a strategy
(it no longer computes indicators, entries, or exits) — it is the
portfolio investment committee that decides which of the market experts'
recommendations, if any, get executed, with how much capital, and in what
order.

This is planning only. No code changes. `_strategy_auto` in
`backend/app/services/trading_engine.py` is not touched. No
`CommitteeDecision`/`TrustAdjustment` class, ranking algorithm, or
allocation logic is implemented. This change produces a specification a
later, separate implementation change builds against.

**Existing infrastructure this design reuses, not reinvents** (confirmed
by reading the code, not assumed):
- `PortfolioRiskService` (`backend/app/services/portfolio_risk.py:36-375`,
  `check_portfolio_risk`) already enforces, aggregated across every bot an
  owner has: daily loss cap, weekly loss cap, max drawdown cap, and max
  total exposure cap (resizing or blocking an order as needed). This is
  real, already-built portfolio risk — Auto calls it, does not reimplement
  it.
- `StrategyCapacityService` (`backend/app/services/strategy_capacity.py:
  77-329`, `check_capacity_for_trade`) already enforces per-strategy
  capital allocation percentage limits (concurrent-bot-count limits exist
  in config but are not yet enforced at trade time — noted honestly, not
  glossed over).
- `_execute_trade` (`backend/app/services/trading_engine.py:6332-6607+`)
  already runs, in order: portfolio risk check, strategy capacity check,
  execution cost estimation, the trade viability gate, order size
  validation, execution routing, and execution. This pipeline is
  UNCHANGED by this design — Auto's `CommitteeDecision` feeds into it
  exactly as a single proposal does today via the Standalone Adapter.
- **Correlation limits, sector/concentration limits, and a minimum
  cash-reserve floor do NOT exist anywhere in the codebase today** —
  confirmed absent, not assumed present. This design gives them a defined
  place in the Committee Process (see "Portfolio Thinking" below) without
  fabricating that they already work; building them is explicitly future
  work, out of scope for this change and for `add-trading-safety-
  boundaries`.

## Goals / Non-Goals

**Goals**
- Completely specify Auto Mode's responsibilities, inputs, process, and
  output, so strategy implementation (already frozen, per
  `add-strategy-decision-framework`) targets a fully-defined end-to-end
  architecture, not just a defined-up-to-the-strategy-boundary one.
- Keep the permanent separation strict: strategies answer "what trade do
  I recommend," Auto answers "which recommendation should I trust." Auto
  never re-derives a trading decision from market data.
- Make every Auto decision deterministic, reproducible, and explainable —
  the same discipline `add-strategy-decision-framework` already requires
  of strategies, now required of the committee that consumes them.
- Define extension points (external trust, learning) that let future
  capabilities plug in without ever requiring a `StrategyProposal` field
  change or a strategy re-certification.

**Non-Goals**
- Implementing any of this. No code, no `_strategy_auto` changes.
- Calibrating actual ranking weights, tie-breaking thresholds, or trust
  adjustment magnitudes — those are implementation-phase, evidence-backed
  decisions (see "Open Questions"), not decided here.
- Building correlation/concentration/cash-reserve checks. This design
  reserves their place in the Committee Process; building them is
  separate, future work.
- Building the learning system. Only extension points are defined.
- Modifying `StrategyProposal`, `add-strategy-decision-framework`'s
  frozen contract, or any strategy implementation.

## Core Philosophy

Strategies trade. Auto allocates trust and capital.

A strategy answers: *"What trade do I recommend, and why?"* — fully,
using only its own market analysis (pillars 1-9 of the Strategy Decision
Framework), producing one immutable `StrategyProposal`.

Auto answers a completely different question: *"Given everything every
enabled strategy currently recommends, which recommendation(s), if any,
deserve real capital right now?"* Auto never asks "is this a good trade"
in the sense of re-evaluating market conditions — that question was
already answered, exhaustively and independently, by the strategy that
produced the proposal. Auto asks "given the portfolio's current state,
which of these already-evaluated recommendations should I trust and fund."

This distinction is permanent, not a phase-in convenience:
- A strategy that stops producing well-evidenced proposals must be fixed
  at the strategy level (better Evidence Items, corrected assumptions,
  Strategy Edge Management) — Auto cannot fix a bad proposal by ranking it
  differently.
- Auto that makes a bad portfolio-level call (over-concentrated,
  over-levered, badly timed relative to other open positions) must be
  fixed at the Auto level — no strategy can fix a bad allocation decision
  by producing a better proposal.

## Auto Inputs

Every evaluation cycle:
- Every enabled strategy is evaluated independently, against the same
  market state (same candle/tick), with no strategy aware that any other
  strategy exists, ran, or produced anything. This is unchanged from
  `add-strategy-decision-framework` and is not renegotiated here.
- Each strategy produces exactly one immutable `StrategyProposal` (or logs
  nothing, if disabled/erroring — Auto only ever sees proposals that were
  actually produced).
- `direction` may be `BUY`, `SELL`, `HOLD`, or `NO_TRADE` today, and
  `execution_intent` any of `NO_ACTION`/`OPEN_POSITION`/`ADD_TO_POSITION`/
  `REDUCE_POSITION`/`CLOSE_POSITION`/`HOLD_POSITION` — both enums remain
  exactly as `add-strategy-decision-framework` defined them; this change
  adds no new values and requires none to function correctly. Future
  execution intents (e.g. from short-selling) are consumed by Auto exactly
  as today's are, without a committee redesign, because the Comparison
  Contract is intent-value-agnostic (see below).
- Auto receives every proposal produced this cycle simultaneously, as a
  batch — it does not evaluate them one at a time or in strategy-defined
  order (see "Proposal Ranking").

## Auto Reaffirms Proposal Immutability

`add-strategy-decision-framework` already established that a
`StrategyProposal`, once constructed, is immutable, and named the
specific fields Auto must never rewrite. This change reaffirms that rule
from Auto's side, exhaustively, because Auto is the component with the
most opportunity and the most temptation to violate it (e.g. "just bump
the risk budget a little for this one" or "fold the sentiment score into
the decision score"):

Auto SHALL NEVER modify, on any `StrategyProposal`: `decision_score` (or
any Evidence Item inside it), `market_suitability`, `edge_status`,
`adaptive_parameters_used`, `explanation`, `suggested_position_size`,
`suggested_risk_budget_pct`, `expected_holding_horizon`,
`expected_edge_estimate`, `assumptions`, `reasons_for`/`reasons_against`,
`direction`, `execution_intent`, `validity`, or any other field, for any
reason — including to apply portfolio context, external trust
adjustments, or capital scaling. Strategies own analysis. Auto owns
selection. Every Auto decision is recorded on a separate, new
`CommitteeDecision` object (see below) that *references* proposals by
`proposal_id`; it never edits them in place.

## Auto Responsibilities

Auto performs portfolio-level decisions ONLY:
- Proposal comparison and ranking
- Proposal rejection (expired, assumption-invalidated, edge-disqualified,
  or portfolio-rejected)
- Portfolio exposure validation (via the existing `PortfolioRiskService`)
- Strategy capacity/health validation (via the existing
  `StrategyCapacityService` and the Strategy Edge Manager's current
  category for that strategy)
- Capital allocation across selected proposals
- Execution ordering
- Proposal expiration checks (`validity.valid_until`, per
  `add-strategy-decision-framework`)
- Portfolio concentration and correlation checks (reserved place in the
  process; not yet built anywhere in this codebase — see "Portfolio
  Thinking")
- Future external trust adjustments (see "External Trust Layer")

Nothing else. In particular:

**Indicator interpretation is permanently forbidden.** This extends
`add-strategy-decision-framework`'s "Auto Must Never Understand Strategy
Indicators" rule from a design intention into Auto's own operating
constraint: Auto SHALL NEVER inspect, interpret, or branch on EMA, ATR,
MACD, RSI, Bollinger Bands, Donchian channels, ADX, or any other
strategy-specific indicator or internal computation, under any
circumstance, including inside ranking, tie-breaking, or trust-adjustment
logic. If a ranking or tie-breaking rule would require indicator
knowledge to implement, that rule is wrong and must be redesigned to use
only Comparison Contract fields — it is never acceptable to reach into a
strategy's internals as a shortcut.

## The Comparison Contract (reused, not redefined)

`add-strategy-decision-framework` already fixed the exact subset of
`StrategyProposal` fields a comparison layer may depend on. This change
does not add to or narrow that list — it is repeated here for
completeness, since it is the literal vocabulary the Committee Process
below is written in:

- `direction` and `execution_intent`
- `decision_score.total` and `decision_score.threshold` (0-100, same scale
  regardless of strategy)
- `suggested_risk_budget_pct`
- `expected_edge_estimate` (when present — `None` otherwise, never
  treated as zero or ignored-but-present)
- `market_suitability.is_suitable`
- `edge_status.category`
- `validity.valid_until`

Auto's ranking, tie-breaking, and rejection logic are defined entirely in
terms of these seven fields (plus portfolio-level state that is not part
of any proposal at all — balances, exposure, correlation). No Auto
capability specified in this change requires a wider view of a
`StrategyProposal` than this.

## The Committee Process

Every evaluation cycle, Auto runs the following ten steps, in this exact
order. The order is itself part of the specification — a proposal
rejected at an earlier step is never evaluated by a later one, so
rejection reasons are unambiguous and the process is reproducible.

1. **Collect every `StrategyProposal`** produced this cycle by every
   enabled strategy. This is the complete candidate set for the cycle;
   nothing is added or removed from it except by the steps below.
2. **Reject expired proposals.** Any proposal with
   `now >= validity.valid_until` is discarded — a pure timestamp
   comparison, per `add-strategy-decision-framework`'s Proposal Validity
   rule. Auto never executes a stale proposal.
3. **Reject proposals whose assumptions are already invalid.** Auto does
   NOT re-evaluate a proposal's `assumptions` itself (doing so would
   require understanding strategy-specific indicators, which is
   forbidden). Instead, this step is satisfied structurally: if the same
   strategy has since produced a NEWER proposal for the same bot (the
   supersession rule `add-strategy-decision-framework` already defines),
   the older one is discarded here as superseded — its assumptions are
   presumed invalidated by the existence of a fresher evaluation. A
   proposal with no newer sibling and `now < validity.valid_until` is
   presumed to still hold its stated assumptions.
4. **Reject proposals whose Strategy Edge Manager currently indicates the
   strategy should not trade.** Reads `edge_status.category` from the
   Comparison Contract only: Category C (edge disappeared) proposals are
   rejected outright; Category A (temporary market mismatch) proposals
   are eligible but their strategy's Decision Score threshold is already
   elevated by the strategy itself (per `add-strategy-decision-framework`
   pillar 7 — Auto does not additionally penalize them); Category B
   (parameter mismatch, being adapted) proposals are eligible normally.
   This step reads a field, it does not diagnose the strategy.
5. **Apply existing portfolio risk constraints.** Calls the
   already-built `PortfolioRiskService.check_portfolio_risk` (daily loss,
   weekly loss, drawdown, total exposure) and `StrategyCapacityService.
   check_capacity_for_trade` (per-strategy allocation) for each surviving
   proposal's implied order. A proposal that would be hard-blocked by
   either service is rejected here; one that would be resized is carried
   forward with its `allocated_size` already capped (see "Capital
   Allocation" under Committee Decision) — this reuses existing,
   unchanged infrastructure, it does not reimplement portfolio risk
   logic.
6. **Apply future external trust adjustments.** Consult any
   `TrustAdjustment` records referencing surviving proposals (Fear &
   Greed, news, macro, funding, social sentiment, exchange health, etc. —
   see "External Trust Layer"). Not built by this change; when built,
   plugs in at exactly this step, adjusting ranking only, never proposal
   content.
7. **Rank remaining proposals**, deterministically, using only Comparison
   Contract fields (see "Proposal Ranking and Tie-Breaking").
8. **Allocate capital** across the proposals selected for execution,
   respecting whatever the portfolio risk/capacity steps above already
   capped.
9. **Select** one or more proposals (or none) for execution, per the
   ranking and allocation from steps 7-8.
10. **Submit selected proposals to the existing execution pipeline** as a
    `CommitteeDecision` (see below) — `_execute_trade` runs exactly as it
    does today, unaware that a committee, rather than a single strategy's
    Standalone Adapter, produced its input.

## Proposal Ranking and Tie-Breaking

Ranking SHALL be deterministic: the same set of surviving proposals,
portfolio state, and trust adjustments SHALL always produce the same
ranking. Ranking SHALL depend only on Comparison Contract fields and
portfolio-level state — never on strategy identity. `trend_following`
receives no special treatment over `mean_reversion`; `adaptive_grid`
receives no special treatment over any other strategy. A ranking rule
that special-cases a named strategy is a defect, not a tuning choice.

**Ties.** If two or more proposals rank nearly identically after step 7,
Auto SHALL resolve the tie using one of the following deterministic
policies, chosen and documented per the eventual implementation's
certification (not decided here — see "Open Questions"):
- Split allocation: reduce capital assigned to each tied proposal
  proportionally rather than picking one.
- Historical-strength tiebreak: prefer the strategy with the
  statistically stronger validated `expected_edge_estimate` (from
  `add-strategy-validation-tooling`), falling back to `decision_score.
  total` if neither has a validated edge estimate yet.
- Any other documented, deterministic, Comparison-Contract-only rule.

Whichever policy is chosen, it SHALL be applied identically every cycle
and SHALL be documented in Auto's own certification record (see
"Certification Requirements for Auto" below) — an undocumented or
non-deterministic tie-break is a certification failure.

**Auto SHALL NOT freeze or default to inaction merely because multiple
strategies disagree** (e.g. one proposes `BUY`, another `SELL`, on the
same asset). Professional discretionary and systematic portfolios
routinely hold or size positions despite disagreement between signals;
"do nothing because it's ambiguous" is itself a decision Auto must make
deliberately, via its ranking and allocation logic, not adopt as a silent
fallback for every disagreement. Executing none is always a valid outcome
of the process (see "Multiple Executions" below) — but it must be the
ranking/allocation logic's actual conclusion, not a bypass of steps 7-9.

## Multiple Executions

The architecture SHALL support, in a single evaluation cycle:
- Executing none of the collected proposals (steps 2-9 leave nothing
  eligible, or allocation determines no proposal clears its bar).
- Executing exactly one.
- Executing multiple simultaneously, across different strategies and/or
  different assets, provided portfolio risk constraints (step 5) permit
  the combined exposure.

This is a capability the architecture must support, not a behavior it
must exhibit every cycle — most cycles, especially with few strategies
enabled or highly correlated signals, may reasonably select zero or one
proposal. `CommitteeDecision.selected` is a list specifically so this
capability requires no future schema change when multi-proposal execution
is actually exercised.

## External Trust Layer (Auto-owned)

External information belongs exclusively to Auto — never to strategies.
Strategies remain purely market-analysis engines producing proposals from
their own market data only (`add-strategy-decision-framework`'s scope,
unchanged). Examples of future external sources: Fear & Greed index,
news, an economic calendar, funding rates, open interest, social
sentiment, whale-wallet activity, exchange outages/health, ETF approval
events, government/regulatory announcements.

These systems, when built (future, separate changes — none are built by
this one), produce `TrustAdjustment` records:

| Field | Type | Meaning |
|---|---|---|
| `proposal_id` | str | Which proposal this adjusts. |
| `source` | str | e.g. `"fear_greed"`, `"news_sentiment"`, `"funding_rate"`, `"exchange_health"`. |
| `adjustment` | float | A multiplier or delta applied only within step 6/7 (ranking), never to any `StrategyProposal` field. |
| `generated_at` | datetime | |

`TrustAdjustment` records SHALL NEVER rewrite a proposal — this is the
same rule `add-strategy-decision-framework` already established, repeated
here because this change is where trust adjustments are actually consumed
(step 6). A proposal with three conflicting trust adjustments applied
still shows, on inspection, exactly what the strategy originally
concluded; the adjustments and their effect on ranking are recorded
separately, on the `CommitteeDecision`.

## Portfolio Thinking

Auto reasons about portfolio-level state that no single strategy is
responsible for or aware of. Some of this is already real, working
infrastructure Auto reuses (step 5); some is not built anywhere yet and
is explicitly reserved space in the Committee Process, not implemented
here:

**Already built, reused unchanged (step 5):**
- Daily loss cap, weekly loss cap, max drawdown cap, max total exposure
  cap — `PortfolioRiskService`.
- Per-strategy capital allocation percentage — `StrategyCapacityService`.

**Not built anywhere in this codebase; reserved for a future, separate
change** (this change specifies WHERE they plug into the Committee
Process — step 5, alongside the existing risk services — so building them
later requires no committee redesign, but builds none of the underlying
logic itself):
- Correlation limits (e.g. "already long two correlated assets").
- Sector/asset concentration limits.
- A minimum cash-reserve floor / available-capital check.
- Per-strategy temporary suspension driven by portfolio-level (not
  strategy-level) conditions, distinct from Strategy Edge Management's
  own Category C stop (which is strategy-level and already covered by
  step 4).

Listing these honestly as unbuilt, rather than assuming or implying they
exist, is deliberate: a future implementer must not discover mid-build
that "concentration limits" was assumed already handled.

## Learning Extension Points (future capability only — not implemented)

Auto SHALL be designed to support, in a later change, trust adjustments
informed by:
- Validated historical expectancy (from `add-strategy-validation-tooling`,
  once it exists).
- Recent statistical performance (e.g. rolling realized win rate per
  strategy, distinct from a strategy's own internal Strategy Edge
  Management, which governs whether the strategy proposes at all, not how
  much Auto trusts it when it does).
- Proposal quality trends (e.g. a strategy whose `decision_score.total`
  has been trending down cycle over cycle, independent of any single
  proposal's score).
- Market-regime-conditioned success (a strategy that historically
  performs better in specific regimes gets more trust in exactly those
  regimes) — sourced from `add-strategy-validation-tooling`'s
  regime-conditioned reporting, once it exists.
- Execution quality (realized slippage/fill quality versus what a
  proposal's sizing assumed).

**This change implements none of this.** The only thing this change does
is confirm the extension point exists and costs nothing: any future
learning system SHALL express itself as a `TrustAdjustment` (see
"External Trust Layer" above) or as an input to Proposal Ranking/
Tie-Breaking (see above) — both already have a defined place in the
Committee Process (steps 6-7). A learning system built later needs no new
step, no `StrategyProposal` change, and no strategy re-certification.

## Committee Decision (finalized)

`add-strategy-decision-framework` stubbed `CommitteeDecision` as a future
contract. This change finalizes it as Auto's sole, immutable output — the
only object the execution pipeline accepts from Auto, exactly as it
accepts a single translated proposal from the Standalone Adapter today:

| Field | Type | Meaning |
|---|---|---|
| `decision_id` | str | Deterministic identifier for this committee evaluation cycle (derived from the cycle's timestamp and the set of `proposal_id`s considered — not a random UUID, for the same reason `proposal_id` is deterministic). |
| `evaluated_at` | datetime | When the committee ran this cycle. |
| `proposals_considered` | List[str] | Every `proposal_id` collected in step 1 — the full candidate set, including ones later rejected. |
| `selected` | List[SelectedAllocation] | `{proposal_id, allocated_size, execution_priority}` for each proposal chosen to execute (zero, one, or many — see "Multiple Executions"). |
| `rejected` | List[RejectedProposal] | `{proposal_id, rejection_step, rejection_reason}` for every proposal considered and not selected. `rejection_step` names which of the 10 Committee Process steps rejected it (e.g. `"expired"`, `"edge_disqualified"`, `"portfolio_risk"`, `"ranked_below_selection"`) — never an unmeasurable judgement. |
| `trust_adjustments_applied` | List[str] | References to every `TrustAdjustment` record consulted this cycle, for audit — empty until the External Trust Layer is actually built. |
| `ranking_snapshot` | List[str] | The full ranked order of every proposal that survived through step 7, by `proposal_id` — so a human (or a future learning system) can reconstruct exactly how selection followed from ranking, not just what was finally selected. |

`CommitteeDecision` is itself immutable once produced, for the same
reason every `StrategyProposal` is: a decision, once made and (if
applicable) executed, is a permanent audit record, not a draft. A new
cycle produces a new `CommitteeDecision`, never an edit to a prior one.

```
Every enabled strategy, same market state, same tick
    |         |         |         |
    v         v         v         v
 Proposal  Proposal  Proposal  Proposal      [each IMMUTABLE,
    A         B         C         D           per add-strategy-
    |         |         |         |           decision-framework]
    +----+----+----+----+----+----+
                   v
        AUTO MODE INVESTMENT COMMITTEE
     (this change's entire specification)
                   v
   Step 1  Collect every proposal
   Step 2  Reject expired (validity.valid_until)
   Step 3  Reject superseded (assumptions presumed stale)
   Step 4  Reject edge-disqualified (edge_status.category == C)
   Step 5  Apply PortfolioRiskService + StrategyCapacityService
           (existing, unchanged)
   Step 6  Apply TrustAdjustment(s)     [future, not built]
   Step 7  Rank (Comparison Contract fields only, deterministic,
           documented tie-break)
   Step 8  Allocate capital
   Step 9  Select (none / one / many)
                   v
   Step 10  CommitteeDecision(selected, rejected, ranking_snapshot,
            trust_adjustments_applied)          [IMMUTABLE]
            -- proposals A-D remain unmodified; this is a
               SEPARATE record referencing them by proposal_id --
                   v
        chosen StrategyProposal(s), per CommitteeDecision.selected
                   v
        _execute_trade()   [SAME pipeline, UNCHANGED — portfolio
                             risk check, strategy capacity check,
                             cost estimate, viability gate, order
                             size validation, execution routing,
                             execution]
                   v
              Order / Position
```

The only genuinely new component in this diagram versus
`add-strategy-decision-framework`'s own future-Auto-Mode sketch is the
fully-specified interior of the "AUTO MODE INVESTMENT COMMITTEE" box —
that sketch left it as an unlabeled black box; this change is its
complete specification.

## Certification Requirements for Auto

Distinct from, and additional to, each strategy's own certification
(`add-strategy-decision-framework`'s nine-item checklist, which governs
`StrategyProposal` quality). Auto Mode's eventual implementation SHALL
NOT be considered certified for production use until a written review
confirms:

1. **Auto never reads indicators.** Code review confirms no ranking,
   tie-breaking, rejection, or trust-adjustment logic references any
   strategy-specific indicator, field, or internal state outside the
   Comparison Contract's seven fields plus portfolio-level state.
2. **Auto never rewrites `StrategyProposal`.** A test constructs proposals,
   runs a full committee cycle including rejection/ranking/allocation/
   trust-adjustment paths, and asserts every field on every proposal
   (selected, rejected, and superseded alike) is byte-identical to its
   pre-cycle value.
3. **Tie-breaking is deterministic.** A test presents a fixed near-tie
   proposal set N times and asserts identical output every time, and
   across process restarts (no reliance on unseeded randomness, wall-clock
   jitter, or dict/set iteration order).
4. **Ranking is reproducible.** A test presents the same proposal set,
   portfolio state, and trust adjustments twice and asserts an identical
   `ranking_snapshot` both times.
5. **Ranking is strategy-identity-blind.** A test swaps which strategy
   produced which Comparison Contract field values (holding the values
   themselves constant) and asserts the ranking is unchanged — proving the
   ranking function has no strategy-name special case.
6. **Trust adjustments are auditable.** Every `TrustAdjustment` consulted
   in a cycle is retrievable via `CommitteeDecision.trust_adjustments_
   applied`, and its effect on that cycle's ranking is reconstructable
   from `ranking_snapshot` plus the adjustment's own recorded value.
7. **Portfolio decisions are explainable.** Every rejection in
   `CommitteeDecision.rejected` names the specific step and a measurable
   reason (never "didn't like it" or an unmeasurable judgement) — same
   discipline `add-strategy-decision-framework` requires of Evidence
   Items, applied to rejection reasons.
8. **`CommitteeDecision` is reproducible.** Re-running a historical cycle
   against the same recorded proposals, portfolio state, and trust
   adjustments produces a `CommitteeDecision` identical in every field
   except `decision_id`'s own timestamp component (if any) — the same
   determinism standard `add-strategy-decision-framework` holds
   `StrategyProposal` to.
9. **Multiple-execution capability is exercised at least once in testing.**
   A test proves the committee can select more than one proposal in a
   single cycle when portfolio risk permits it — not merely that the code
   path exists, but that it was actually run and produced a valid
   multi-selection `CommitteeDecision`.

## Decisions

- Decision: the Committee Process is a fixed, ordered ten-step sequence,
  not a configurable pipeline of interchangeable stages.
  - Why: "the same set of proposals always produces the same decision"
    (this change's own determinism requirement) is much harder to
    guarantee if step order itself is a runtime variable — a fixed order
    makes rejection reasons unambiguous (a proposal rejected at step 3 was
    never evaluated by step 5) and makes the certification tests above
    tractable to write once, not per-configuration.
- Decision: rejection at any step is terminal for that cycle — a proposal
  rejected at step 2 is not reconsidered at step 5 even if, hypothetically,
  it would have passed.
  - Why: matches how `_execute_trade`'s own existing pipeline already
    behaves (each of its 8 steps can hard-block, and does) — Auto's
    process is modeled on the same fail-fast, ordered-gates pattern
    already proven in this codebase, not a novel one.
- Decision: `CommitteeDecision.rejected` records a `rejection_step`, not
  just a `rejection_reason` string.
  - Why: satisfies "portfolio decisions are explainable" concretely — a
    human or a future learning system can distinguish "rejected because
    stale" from "rejected because it lost a ranking comparison" from
    "rejected because portfolio risk blocked it" without parsing free text.
- Decision: correlation, concentration, and cash-reserve checks are
  explicitly named as NOT built, with a reserved place in step 5, rather
  than silently omitted or vaguely gestured at.
  - Why: the user's own portfolio-thinking examples included concentration
    and correlation; inventing that they already exist would be dishonest
    about the codebase's actual state (confirmed absent by direct
    search), and silently omitting them would leave a future implementer
    to discover the gap mid-build. Naming the gap and its slot in the
    process satisfies "no future strategy [or Auto] implementation should
    require architectural redesign" without fabricating current
    capability.
- Decision: Auto's own certification (nine items above) is separate from,
  not merged into, each strategy's certification.
  - Why: they check different things at different layers — a strategy
    can be perfectly certified (deterministic, immutable, well-evidenced
    proposals) while Auto's ranking logic is broken, and vice versa.
    Conflating them would make a certification failure ambiguous about
    which layer actually has the defect.
- Decision: `TrustAdjustment.adjustment` affects ranking only (step 6-7),
  never allocation math directly and never any `StrategyProposal` field.
  - Why: keeps the External Trust Layer's blast radius contained to
    exactly the concern it exists for (should this proposal be trusted
    more or less relative to others) rather than letting it silently
    reach into capital sizing (which remains portfolio-risk-governed, per
    step 5/8) or proposal content (forbidden outright, per Proposal
    Immutability).

## Risks / Trade-offs

- A fixed ten-step order (see Decisions) means adding a genuinely new
  portfolio concern later (e.g. correlation limits) requires deciding
  where in the existing order it belongs, not just appending it at the
  end — mitigated by already reserving its place (step 5, alongside the
  existing risk services) in this design, but the exact insertion order
  relative to `PortfolioRiskService`/`StrategyCapacityService` is left to
  whichever future change actually builds it.
- Deterministic, strategy-identity-blind ranking means a strategy with a
  slightly-better-tuned Decision Score threshold can systematically
  out-rank a strategy with genuinely better real-world edge but a more
  conservative scoring style — mitigated by the Learning Extension Points
  (trust adjustment from validated historical performance), but that
  mitigation is explicitly future, not built now; until it exists, ranking
  is purely proposal-quality-blind to track record.
- The External Trust Layer and Learning extension points are specified
  but empty — a reader could mistake "the architecture supports X" for
  "X exists." Mitigated by this document's repeated, explicit "not built"
  labeling rather than a single disclaimer at the top.
- Multiple-execution support (architecture requirement) increases the
  surface area portfolio risk checks (step 5) must correctly handle
  simultaneously (N proposals' combined exposure, not one proposal's
  exposure in isolation) — `PortfolioRiskService.check_portfolio_risk` as
  it exists today is called per-order; a future implementation must
  confirm it correctly handles being called for multiple orders within
  the same cycle before the cumulative exposure of an earlier selection
  is committed (a sequencing detail flagged here, resolved in
  implementation, not decided in this design).

## Migration Plan

No migration — this is a specification-only change. A future, separate
implementation change is expected to build, in order: (1) the Comparison
Contract reader and the ten-step Committee Process as pure functions over
proposal batches, unit-tested against synthetic proposal sets; (2) the
`CommitteeDecision` schema; (3) the ranking/tie-breaking policy chosen and
documented per "Open Questions"; (4) wiring steps 5 (portfolio risk) to
the existing `PortfolioRiskService`/`StrategyCapacityService`; (5) wiring
step 10's output into the existing, unchanged execution pipeline in place
of the Standalone Adapter, behind a flag, with before/after backtest
comparison against the Standalone-Adapter-driven single-strategy behavior
for the degenerate one-strategy-enabled case (which SHALL produce
identical outcomes — a strong regression check, since a one-strategy
portfolio is exactly what the Standalone Adapter already handles
correctly today). None of this is scoped, sequenced, or scheduled by this
change; it is intentionally left to the future implementation change this
specification exists to unblock.

## Architecture Freeze

This design is the final architectural specification for Auto Mode.
Combined with `add-strategy-decision-framework`'s already-frozen
`StrategyProposal` contract, this completes the end-to-end architecture:
strategies produce proposals against a frozen contract, Auto consumes
them via a frozen, fully-specified committee process and produces a
frozen `CommitteeDecision` shape, which feeds the existing, unchanged
execution pipeline.

- The Committee Process's ten steps, their order, the Comparison Contract,
  the `CommitteeDecision` shape, the `TrustAdjustment` shape, and Auto's
  responsibility boundary (portfolio-level only, indicator interpretation
  forbidden) are frozen as of this revision.
- A future implementation change SHALL implement against this
  specification. It SHALL NOT reorder, remove, or merge Committee Process
  steps, add new top-level fields to `CommitteeDecision`/
  `TrustAdjustment`, or weaken Auto's indicator-isolation or
  proposal-immutability rules.
- Calibrating actual ranking weights, the specific tie-breaking policy,
  trust-adjustment magnitudes, and correlation/concentration/cash-reserve
  thresholds are NOT frozen — these are evidence-backed, implementation-
  phase decisions per "Open Questions," exactly analogous to how
  `add-strategy-decision-framework` freezes the `StrategyProposal`
  contract shape while leaving each strategy's Evidence Item weights open.
- As with the strategy architecture, a concrete limitation discovered
  during implementation or historical backtesting is grounds to propose a
  new, separate, evidenced OpenSpec change amending this specification —
  a hypothetical future capability is not.
- After this change: both the `StrategyProposal` architecture and the
  Auto Mode Investment Committee architecture are frozen. Strategy
  implementation (`add-strategy-decision-framework` Phases 0-6) and a
  future Auto Mode implementation change may both proceed without further
  architectural redesign, barring demonstrated evidence.

## Open Questions

- The exact tie-breaking policy (split allocation vs. historical-strength
  preference vs. another deterministic rule) is structurally specified
  (must be deterministic, Comparison-Contract-only, documented) but not
  chosen — left to the future implementation change, likely informed by
  `add-strategy-validation-tooling` results once real validated edge
  estimates exist to break ties with.
- Exact ranking weights/formula combining `decision_score.total`,
  `suggested_risk_budget_pct`, and `expected_edge_estimate` into a single
  rank are not specified numerically — only that they must be
  deterministic and Comparison-Contract-only. Calibration is an
  implementation-phase, evidence-backed decision.
- Where step 5's future correlation/concentration/cash-reserve checks
  should live implementation-wise (extending `PortfolioRiskService`
  itself, or a new sibling service Auto calls alongside it) is left open
  — either satisfies this design's requirement that they slot into step 5
  without a Committee Process redesign.
- Whether `CommitteeDecision.decision_id`'s determinism should be scoped
  per-bot-portfolio or per-owner-across-all-bots is left to implementation
  — this design only requires that it be deterministic, not which
  granularity it's keyed at.
- Exact multi-selection sequencing against `PortfolioRiskService` (see
  "Risks / Trade-offs" — whether exposure checks for a second selected
  proposal in the same cycle must account for the first proposal's
  not-yet-executed allocation) is flagged, not resolved here — an
  implementation-phase concern with a correctness consequence, worth
  resolving with a dedicated test before the multi-execution capability
  is ever exercised in production.
- Whether Auto's own certification (the nine items above) should be a
  one-time gate (certify Auto once, like a strategy) or a continuously
  re-verified property (re-run the certification test suite on every
  change to ranking/tie-breaking logic, given how much more central Auto
  is than any single strategy) is left to whoever implements it — this
  design specifies WHAT must hold, not the process discipline around
  re-verifying it over time.
