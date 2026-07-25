# auto-mode-investment-committee Specification

## Purpose
TBD - created by archiving change add-auto-mode-investment-committee. Update Purpose after archive.
## Requirements
### Requirement: Auto Never Trades Directly
Auto Mode SHALL NOT compute, interpret, or derive a trading decision from
market data. Auto SHALL only select among, allocate capital to, and
reject `StrategyProposal` objects that strategies have already produced.
Strategies answer what trade is recommended; Auto answers which
recommendation is trusted.

#### Scenario: Auto does not evaluate market conditions itself
- **WHEN** Auto Mode runs an evaluation cycle
- **THEN** it acts only on `StrategyProposal` objects already produced by
  enabled strategies, and does not independently compute or consult any
  price, indicator, or candle data to arrive at its own trading view

#### Scenario: A strategy problem is not fixed by Auto
- **WHEN** a strategy produces poorly-evidenced or incorrect proposals
- **THEN** Auto's ranking or allocation logic cannot compensate for it;
  the defect must be corrected in the strategy itself, not worked around
  in Auto

### Requirement: Auto Reaffirms Proposal Immutability
Auto Mode SHALL NEVER modify any field of a `StrategyProposal`, including
`decision_score`, `market_suitability`, `edge_status`,
`adaptive_parameters_used`, `explanation`, `suggested_position_size`,
`suggested_risk_budget_pct`, `expected_holding_horizon`,
`expected_edge_estimate`, `assumptions`, `reasons_for`/`reasons_against`,
`direction`, `execution_intent`, or `validity`, for any reason including
applying portfolio context or external trust adjustments.

#### Scenario: A full committee cycle leaves every proposal unchanged
- **WHEN** Auto runs a full evaluation cycle, including rejecting some
  proposals, ranking others, applying trust adjustments, and selecting a
  subset for execution
- **THEN** every field on every proposal considered - selected, rejected,
  and superseded alike - remains exactly as the strategy produced it

#### Scenario: Auto records decisions separately, never in place
- **WHEN** Auto decides to reject, deprioritize, or resize the executed
  amount of a proposal
- **THEN** that decision is recorded on a new `CommitteeDecision` object
  referencing the proposal by `proposal_id`, never written onto the
  proposal itself

### Requirement: Alpha and Allocation Strategy Categories
Strategies SHALL be classified into two categories by purpose: Alpha
strategies (which generate positive risk-adjusted returns and participate in
the Investment Committee) and Allocation strategies (which deploy capital
according to an investment policy and do NOT participate in Committee
ranking). The Investment Committee SHALL evaluate only Alpha
`StrategyProposal` objects. Allocation strategies SHALL execute independently
according to portfolio policy, subject to the same portfolio risk and
capacity services as any order. This classification SHALL be Auto-side
metadata keyed by strategy identity and SHALL NOT modify the
`StrategyProposal` contract or any Strategy Decision Framework interface.

#### Scenario: Allocation proposals are excluded from the committee
- **WHEN** an Allocation strategy (e.g. `dca_accumulator`) produces a
  `StrategyProposal` in the same cycle as Alpha strategies
- **THEN** the Allocation proposal is not collected at Committee Process step
  1, is never ranked against an Alpha proposal, and does not appear in any
  `CommitteeDecision.proposals_considered`

#### Scenario: Allocation strategies still execute under portfolio governance
- **WHEN** an Allocation strategy's proposal is ready to execute
- **THEN** it reaches the execution pipeline via its own Standalone-Adapter
  path and is still gated by `PortfolioRiskService` and
  `StrategyCapacityService`, without being subject to comparative ranking

#### Scenario: Categorization changes no contract
- **WHEN** a strategy is classified as Alpha or Allocation, or reclassified
- **THEN** only Auto-side registry metadata keyed by strategy identity
  changes; the `StrategyProposal` contract, the Comparison Contract, and
  every Strategy Decision Framework interface remain unchanged

### Requirement: Auto Input Contract
Every evaluation cycle, Auto SHALL receive exactly one `StrategyProposal`
per enabled Alpha strategy that produced one, evaluated independently against
the same market state, with no strategy aware of any other strategy's
existence or output.

#### Scenario: Strategies remain mutually unaware
- **WHEN** multiple strategies are enabled and evaluated in the same cycle
- **THEN** no strategy's evaluation logic references another strategy's
  state, proposal, or existence - cross-strategy awareness exists only
  inside Auto

#### Scenario: Auto receives the full batch, not a stream
- **WHEN** an evaluation cycle completes strategy evaluation
- **THEN** Auto receives every proposal produced that cycle together, and
  evaluates them as one batch through the Committee Process rather than
  acting on proposals one at a time as they arrive

### Requirement: Auto Responsibility Boundary
Auto Mode SHALL restrict itself to portfolio-level decisions only:
proposal comparison, ranking, rejection, portfolio exposure and capacity
validation, capital allocation, execution ordering, proposal expiration
checks, portfolio concentration and correlation checks, and external
trust adjustments. Auto SHALL NEVER inspect, interpret, or branch on any
strategy-specific indicator, including EMA, ATR, MACD, RSI, Bollinger
Bands, Donchian channels, or ADX.

#### Scenario: Ranking logic never reaches into strategy internals
- **WHEN** Auto's ranking, tie-breaking, or rejection logic is reviewed
- **THEN** every input it reads is either a Comparison Contract field
  (`direction`, `execution_intent`, `decision_score.total`/`.threshold`,
  `suggested_risk_budget_pct`, `expected_edge_estimate`,
  `market_suitability.is_suitable`, `edge_status.category`,
  `validity.valid_until`) or portfolio-level state - never a
  strategy-specific indicator value

#### Scenario: A new strategy requires no Auto changes
- **WHEN** a new strategy using indicators no existing strategy uses is
  certified and enabled
- **THEN** Auto requires no code changes to evaluate its proposals,
  because it only ever reads Comparison Contract fields and portfolio
  state

### Requirement: Committee Process Order
Auto SHALL execute a fixed, ordered ten-step Committee Process every
evaluation cycle: collect proposals, reject expired, reject superseded,
reject edge-disqualified, apply portfolio risk constraints, apply trust
adjustments, rank, allocate capital, select, and submit to execution. A
proposal rejected at an earlier step SHALL NOT be evaluated by a later
step in the same cycle.

#### Scenario: Rejection is terminal for the cycle
- **WHEN** a proposal is rejected at any step (e.g. expired at step 2)
- **THEN** it is not reconsidered by any subsequent step (e.g. portfolio
  risk at step 5) in that same evaluation cycle

#### Scenario: Every proposal's fate is traceable to a specific step
- **WHEN** a `CommitteeDecision` is produced
- **THEN** every non-selected proposal's rejection is attributable to
  exactly one of the ten steps, not a generic or unspecified rejection

#### Scenario: Stale proposals are never executed
- **WHEN** a `StrategyProposal`'s `validity.valid_until` has passed by the
  time Auto evaluates it
- **THEN** it is rejected at step 2 and never reaches ranking, allocation,
  or execution

#### Scenario: Edge-disqualified strategies are excluded before ranking
- **WHEN** a proposal's `edge_status.category` is `C` (edge disappeared)
- **THEN** it is rejected at step 4 and never competes in ranking against
  proposals from strategies whose edge remains valid

### Requirement: Deterministic, Strategy-Identity-Blind Proposal Ranking
Auto SHALL rank surviving proposals deterministically, using only
Comparison Contract fields and portfolio-level state. Ranking SHALL NOT
depend on which strategy produced a proposal.

#### Scenario: Identical inputs produce identical ranking
- **WHEN** the same set of surviving proposals, portfolio state, and
  trust adjustments is evaluated twice
- **THEN** the resulting ranking is identical both times

#### Scenario: No strategy receives special treatment
- **WHEN** Comparison Contract field values are held constant but which
  strategy produced them is varied
- **THEN** the resulting ranking is unchanged - proving no strategy name
  or identity influences the outcome

### Requirement: Deterministic Tie-Breaking Policy
When two or more proposals rank nearly identically, Auto SHALL resolve
the tie using a single documented, deterministic policy expressed only in
terms of Comparison Contract fields. Auto SHALL NOT resolve a tie using
unseeded randomness, wall-clock timing, or strategy identity.

#### Scenario: Repeated evaluation of a tie produces the same outcome
- **WHEN** the same near-tie proposal set is evaluated multiple times,
  including across process restarts
- **THEN** the tie-breaking outcome is identical every time

#### Scenario: Disagreement between strategies does not stall the committee
- **WHEN** two proposals disagree on direction for the same asset (e.g.
  one recommends BUY, another recommends SELL)
- **THEN** Auto resolves the situation through its ranking, tie-breaking,
  and allocation logic rather than defaulting to no action solely because
  the proposals disagree

### Requirement: Multiple Execution Support
The Committee Process SHALL support selecting zero, one, or multiple
`StrategyProposal` objects for execution in a single evaluation cycle,
subject to portfolio risk constraints permitting the combined exposure.

#### Scenario: Zero proposals selected is a valid outcome
- **WHEN** no surviving proposal clears the ranking/allocation bar, or
  portfolio risk constraints block every candidate
- **THEN** the `CommitteeDecision` for that cycle has an empty `selected`
  list, and this is a valid, non-error outcome

#### Scenario: Multiple proposals can be selected in one cycle
- **WHEN** more than one surviving proposal clears the ranking/allocation
  bar and their combined exposure is within portfolio risk limits
- **THEN** the `CommitteeDecision`'s `selected` list contains more than
  one entry, each with its own allocated size and execution priority

### Requirement: External Trust Layer Is Auto-Owned
External information sources SHALL be consumed only by Auto, never by
strategies, and SHALL be represented as independent `TrustAdjustment`
records referencing a proposal by identifier. A `TrustAdjustment` SHALL
influence only proposal ranking, and SHALL NEVER modify any
`StrategyProposal` field.

#### Scenario: A new external signal source requires no strategy changes
- **WHEN** a new external context source (e.g. a Fear & Greed index) is
  added to influence which proposals Auto trusts
- **THEN** no strategy implementation changes, because the new source
  produces `TrustAdjustment` records consumed by Auto at Committee Process
  step 6, not a change to any strategy or to `StrategyProposal`

#### Scenario: Trust adjustments are auditable
- **WHEN** a `CommitteeDecision` reflects the influence of one or more
  `TrustAdjustment` records
- **THEN** every consulted `TrustAdjustment` is listed in
  `trust_adjustments_applied`, and its effect on that cycle's ranking is
  reconstructable from the recorded `ranking_snapshot`

### Requirement: Portfolio-Level Validation
Auto SHALL apply the existing, already-built `PortfolioRiskService` and
`StrategyCapacityService` checks to every surviving proposal's implied
order before ranking. Correlation limits, concentration limits, and a
cash-reserve floor SHALL have a defined place in the Committee Process
even where their underlying checks do not yet exist.

#### Scenario: Existing portfolio risk checks gate every candidate
- **WHEN** a proposal survives steps 2-4 of the Committee Process
- **THEN** its implied order is validated against
  `PortfolioRiskService.check_portfolio_risk` and
  `StrategyCapacityService.check_capacity_for_trade` before it is ranked,
  and is rejected or resized exactly as a single-strategy order would be
  today

#### Scenario: Unbuilt portfolio checks do not block committee operation
- **WHEN** correlation, concentration, or cash-reserve checks are not yet
  implemented
- **THEN** the Committee Process still runs to completion using the
  portfolio checks that do exist, and a future addition of those checks
  requires no reordering of the existing ten steps

### Requirement: Learning Extension Points Are Defined, Not Implemented
Auto's architecture SHALL provide a defined extension point for future
trust adjustments informed by validated historical performance, without
implementing any learning logic. Any future learning capability SHALL
express itself as a `TrustAdjustment` or as a ranking input, requiring no
new Committee Process step and no `StrategyProposal` change.

#### Scenario: A future learning system integrates without redesign
- **WHEN** a future capability computes trust based on validated
  historical expectancy or regime-conditioned performance
- **THEN** it integrates by producing `TrustAdjustment` records consumed
  at the existing step 6, without requiring a new Committee Process step,
  a `StrategyProposal` field change, or any strategy re-certification

#### Scenario: No learning logic exists yet
- **WHEN** this change is implemented
- **THEN** no historical-performance-based trust computation exists; the
  extension point is defined and unused until a future, separate change
  builds it

### Requirement: Committee Decision Is Auto's Sole, Immutable Output
Auto SHALL produce exactly one immutable `CommitteeDecision` per
evaluation cycle, recording `proposals_considered`, `selected` (with
allocated size and execution priority), `rejected` (with the rejecting
step and reason for each), `trust_adjustments_applied`, and a
`ranking_snapshot`. The execution pipeline SHALL accept only a
`CommitteeDecision` from Auto, never a raw proposal list.

#### Scenario: Every proposal's outcome is recorded
- **WHEN** a `CommitteeDecision` is produced
- **THEN** every proposal in `proposals_considered` appears in either
  `selected` or `rejected`, with no proposal silently dropped

#### Scenario: A historical decision is reproducible
- **WHEN** a past evaluation cycle is re-run against the same recorded
  proposals, portfolio state, and trust adjustments
- **THEN** the resulting `CommitteeDecision` is identical in every field
  except any timestamp component of `decision_id` itself

#### Scenario: CommitteeDecision remains unmodified after production
- **WHEN** a `CommitteeDecision` has been produced and submitted to the
  execution pipeline
- **THEN** it is never subsequently edited; any change in outcome is
  represented by a new `CommitteeDecision` from a later evaluation cycle

### Requirement: Single-Bot Runtime Integration
Auto Mode SHALL run the Investment Committee inside a single Auto bot's own
evaluation loop: it evaluates all enabled Alpha strategies, collects their
`StrategyProposal` objects, runs the Committee Process, and executes the
selected proposal through the existing, unchanged `_execute_trade` pipeline.
Auto is one bot, not a portfolio of bots: there SHALL be no cross-bot
scheduler or portfolio-level runtime. `dca_accumulator` (an Allocation
strategy) SHALL NOT be a committee candidate. The committee runtime SHALL be
behind a feature flag, OFF by default, leaving the pre-existing Auto behaviour
unchanged when disabled.

#### Scenario: Committee runs inside the single Auto bot when flat
- **WHEN** an Auto bot with the committee flag enabled evaluates while holding
  no position
- **THEN** it evaluates every enabled Alpha strategy, runs the committee over
  their proposals, and executes the selected proposal through `_execute_trade`,
  without any cross-bot scheduler

#### Scenario: Position ownership is preserved
- **WHEN** the Auto bot holds a position opened by a committee-selected strategy
- **THEN** only that `owning_strategy` is dispatched to manage the exit, and no
  other strategy's proposal is executed against that position until it closes

#### Scenario: Flag off leaves Auto unchanged
- **WHEN** the committee flag is disabled
- **THEN** the Auto bot uses its pre-existing selection behaviour, unchanged

### Requirement: Auto Certification Gate
Auto Mode's implementation SHALL NOT be considered certified for
production use until a written review confirms: it never reads
indicators, it never rewrites a `StrategyProposal`, tie-breaking is
deterministic, ranking is reproducible and strategy-identity-blind, trust
adjustments are auditable, portfolio decisions are explainable with a
measurable reason per rejection, `CommitteeDecision` is reproducible, and
multiple-execution capability has been exercised at least once in
testing.

#### Scenario: Uncertified Auto implementation is blocked from production
- **WHEN** an Auto Mode implementation does not reference a passing
  certification review against this capability
- **THEN** it does not proceed to production use

#### Scenario: Certification is evaluated item by item
- **WHEN** Auto Mode undergoes certification review
- **THEN** each of the nine certification properties is checked and
  recorded individually, not approved as a single unexamined bundle

