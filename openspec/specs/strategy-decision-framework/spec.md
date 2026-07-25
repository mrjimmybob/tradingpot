# strategy-decision-framework Specification

## Purpose
TBD - created by archiving change add-strategy-decision-framework. Update Purpose after archive.
## Requirements
### Requirement: Strategy Theory Documentation
Every non-Auto-Mode strategy SHALL have a written Strategy Audit Document
stating the market inefficiency it exploits, why the edge should exist,
supporting evidence, the conditions under which it should work, the
conditions under which it should fail, and its core assumptions.

#### Scenario: Strategy lacks a theory document
- **WHEN** a strategy has no Strategy Audit Document, or its document does
  not answer why the edge should exist
- **THEN** the strategy SHALL NOT be certified for implementation or
  continued production use

### Requirement: Market Suitability Gate
Every strategy SHALL determine, before evaluating an entry, whether current
market conditions (trend state, volatility level and direction, liquidity)
match its declared assumptions, and SHALL refuse to open new positions when
they do not.

#### Scenario: Regime does not match strategy assumptions
- **WHEN** current market regime is outside a strategy's declared
  `allowed_regimes`
- **THEN** the strategy does not open a new position, regardless of any
  other entry condition being satisfied

#### Scenario: Suitability check must actually gate entries, not just be computed
- **WHEN** a strategy computes a market-suitability value
- **THEN** that value is consulted by the entry decision itself, not merely
  logged or exposed as a diagnostic with no effect on whether a trade is
  placed

### Requirement: Evidence-Based Decision Score
The system SHALL accumulate multi-factor, objectively measurable evidence
for a candidate trade into a Decision Score, and SHALL only execute trades
scoring above a configurable threshold. A trading system SHALL NOT rely on
subjective or unmeasurable concepts ("looks bullish," "strong feeling,"
"good setup," or any factor requiring human judgement) as scoring inputs.

#### Scenario: Single-condition trigger is insufficient
- **WHEN** only one or two indicator conditions align (e.g. a single band
  touch or EMA cross) with no other supporting evidence
- **THEN** the strategy does not treat this alone as sufficient to trade if
  its configured Evidence Items and threshold are not also satisfied

#### Scenario: High-evidence trade scores above threshold
- **WHEN** multiple independent factors relevant to the strategy's thesis
  align (e.g. trend strength, volume confirmation, structure, risk/reward)
- **THEN** the accumulated score reflects that alignment and can clear the
  configured threshold

#### Scenario: Every Evidence Item is measurable, reproducible, and documented
- **WHEN** a strategy declares an Evidence Item that contributes to its
  Decision Score
- **THEN** that item has a documented Measurement (a deterministic
  computation from available data), a Normalization (a deterministic
  mapping to a bounded contribution range), a Weight, and a documented
  reason it contributes to the strategy's edge, and produces the same
  value given the same inputs every time

#### Scenario: Subjective or unmeasurable factors are rejected
- **WHEN** a proposed Evidence Item cannot be expressed as a deterministic
  Measurement and Normalization computable from available data
- **THEN** it SHALL NOT be added to the Decision Score

#### Scenario: Every executed trade produces a complete Evidence Report
- **WHEN** a trade executes
- **THEN** the system can produce a human-readable report listing every
  Evidence Item that participated, its individual contribution, the total
  Decision Score, the minimum required threshold, and the resulting
  approve/reject decision

### Requirement: Adaptive Parameters
The system SHALL compute volatility-, regime-, or noise-sensitive strategy
parameters adaptively rather than hardcoding them. Any parameter that
remains fixed SHALL have a documented reason.

#### Scenario: Volatility-sensitive parameter hardcoded without justification
- **WHEN** a strategy uses a fixed stop distance, lookback window, or
  similar volatility-sensitive parameter
- **THEN** its Strategy Audit Document either shows the parameter is
  computed from current volatility/regime, or documents why a fixed value
  is correct

### Requirement: Decision-Score-Weighted Position Sizing
The system SHALL size positions using the trade's Evidence-Based Decision
Score, portfolio exposure, recent drawdown, and configured risk budget -
not the Decision Score alone, and not a flat percentage regardless of the
evidence behind the trade.

#### Scenario: Marginal-score trade sized down
- **WHEN** a trade clears the Decision Score threshold only marginally
- **THEN** its position size is smaller than a trade with a materially
  higher Decision Score, all else equal

#### Scenario: Sizing respects portfolio-level exposure caps
- **WHEN** a sized order would breach the portfolio's configured exposure
  cap
- **THEN** the order is resized or blocked per the existing portfolio risk
  enforcement, not sized in isolation from account-wide state

### Requirement: Continuous Trade Management
After entry, every strategy SHALL continuously re-evaluate its open
position every evaluation cycle - whether the original thesis still holds,
whether volatility has changed materially, and whether the stop, partial
profit, or full exit should be adjusted - rather than only checking a
fixed stop and target set at entry.

#### Scenario: Thesis invalidated after entry
- **WHEN** conditions that justified entry (e.g. the market-suitability
  regime, momentum, or volatility profile) change materially while a
  position is open
- **THEN** the strategy's exit evaluation reflects the changed conditions,
  not only the original fixed stop/target

### Requirement: Strategy Edge Management
The system SHALL continuously track each strategy's own measurable
performance and market-condition signals (win rate, expectancy,
consecutive losses, regime, volatility versus design assumptions,
liquidity, spread, slippage, false-breakout frequency, holding time,
reward:risk, Decision Score trend, execution quality) and, when
degradation is detected, SHALL classify its cause into exactly one of
three categories - never merely record that the strategy is losing.

#### Scenario: Category A - temporary market mismatch
- **WHEN** degradation is attributable to current market conditions
  moving outside the strategy's declared suitable range, with no evidence
  the underlying edge is gone
- **THEN** the system reduces trading activity and/or raises the Decision
  Score threshold and waits, without a permanent stop

#### Scenario: Category B - parameter mismatch
- **WHEN** degradation is attributable to a specific adaptive parameter no
  longer fitting current conditions, while the strategy's thesis remains
  otherwise supported
- **THEN** the system adapts only that parameter, via the Adaptive
  Parameters mechanism, with the adaptation mathematically justified by
  the measured signal - not a blanket re-tune of unrelated parameters

#### Scenario: Category C - edge disappeared
- **WHEN** degradation persists after Category A or B responses, or
  measured evidence directly contradicts the strategy's documented Theory
- **THEN** the system stops trading that strategy and does not resume
  automatically without a human-reviewed re-certification

#### Scenario: Classification is evidence-based, not a guess
- **WHEN** the system classifies a degradation event into Category A, B,
  or C
- **THEN** the classification is derived from the same measurable,
  documented signals as the rest of the framework, not an unmeasurable
  judgement call

#### Scenario: Strategy Edge Management never force-closes an open position
- **WHEN** a strategy is classified into any category, including Category C
- **THEN** any existing open position continues to be managed by its
  normal Continuous Trade Management exit logic, not force-closed solely
  because of the classification

#### Scenario: The strategy can answer why its edge changed
- **WHEN** an operator asks why a strategy's performance changed
- **THEN** the system can state, from measurable evidence, why it was
  profitable, why it is no longer profitable (if applicable), whether it
  can adapt, whether it should wait, or whether it should stop

### Requirement: Self-Diagnostics for Every Decision Point
The system SHALL make every strategy decision - entry, position size, stop
placement, and exit - explainable in terms understandable to a human
trader, and that explanation SHALL be retrievable for any historical
trade, not only the strategy's current live state.

#### Scenario: Historical trade is explainable
- **WHEN** an operator inspects a trade that occurred in the past
- **THEN** the system can show why entry happened, why then, why that
  size, why that stop, and why the strategy expected positive expected
  value - not only the terse order reason string

#### Scenario: Every decision point has a corresponding check
- **WHEN** a strategy computes a value that participates in a trading
  decision (sizing, a viability gate, a kill switch, a stop calculation)
- **THEN** that computation is recorded as a check or metric in the
  strategy's structured explanation, not only logged

### Requirement: Documented Performance Expectations
Every strategy's Strategy Audit Document SHALL state its expected trade
frequency, holding time, win rate, profit factor, and drawdown, the market
conditions it's typically used in versus its worst-case conditions, and
why it should out- or under-perform buy-and-hold.

#### Scenario: Real performance contradicts documented expectations
- **WHEN** a strategy's actual walk-forward or live performance
  materially contradicts its documented expectations
- **THEN** this is treated as a finding requiring review, not silently
  accepted as normal variance

### Requirement: Strategy Proposal Interface
Every strategy SHALL return a StrategyProposal rather than executing a
trade directly. A StrategyProposal is a recommendation, not an order, and
SHALL be entirely deterministic - the same market data and strategy state
SHALL always produce the same proposal.

#### Scenario: Strategy returns a recommendation, not an executed trade
- **WHEN** a strategy function completes its evaluation for a market tick
- **THEN** it returns a StrategyProposal describing what it recommends and
  why, and does not itself place or execute an order

#### Scenario: A proposal carries direction, evidence, and risk information
- **WHEN** a StrategyProposal is produced
- **THEN** it includes at minimum: a strategy identifier, a direction
  (BUY, SELL, HOLD, or NO_TRADE), the Evidence-Based Decision Score and
  its full Evidence Report, the Market Suitability result, the Strategy
  Edge status, a suggested position size and risk budget, reasons for and
  against the trade derived from the Evidence Report, the adaptive
  parameters used, and the decision explanation

#### Scenario: Identical inputs produce an identical proposal
- **WHEN** a strategy is evaluated twice against the same historical
  market data and the same prior state
- **THEN** the two resulting StrategyProposal objects are identical,
  including their identifier

#### Scenario: Standalone execution is behavior-preserving
- **WHEN** a bot runs a single strategy directly (no multi-strategy
  selection layer involved)
- **THEN** its StrategyProposal, if accepted by the existing portfolio
  risk, capacity, cost, and viability checks, executes with the same
  outcome as the pre-StrategyProposal direct-execution behavior

### Requirement: Execution Intent
Every StrategyProposal SHALL state both a `direction` (which way the
market is expected to move: BUY, SELL, HOLD, or NO_TRADE) and a separate
`execution_intent` (the specific portfolio action recommended: NO_ACTION,
OPEN_POSITION, ADD_TO_POSITION, REDUCE_POSITION, CLOSE_POSITION, or
HOLD_POSITION). Direction alone SHALL NOT be treated as sufficient to
determine what action to take.

#### Scenario: Direction alone does not determine the action
- **WHEN** a StrategyProposal has a BUY direction
- **THEN** the specific action taken (opening a new position vs. adding to
  an existing one) is determined by `execution_intent`, not inferred from
  `direction` alone

#### Scenario: Execution intent is forward-compatible with new directions
- **WHEN** a future direction value (e.g. SHORT or COVER, for
  short-selling) is added to the `direction` enum
- **THEN** it pairs with the same existing `execution_intent` values
  without requiring `execution_intent` itself to change shape

### Requirement: Proposal Validity and Expiration
Every StrategyProposal SHALL define a `validity` describing when it
expires (`valid_until`, a deterministic timestamp derived from
`generated_at` and the strategy's evaluation interval). A proposal SHALL
also be considered invalid the moment a newer proposal from the same
strategy and bot is produced, whichever bound is reached first.

#### Scenario: Expired proposal is not executed
- **WHEN** the current time is at or past a StrategyProposal's
  `validity.valid_until`
- **THEN** the proposal SHALL NOT be executed, whether in standalone mode
  or by a future Auto Mode

#### Scenario: A superseding proposal invalidates the prior one
- **WHEN** a strategy produces a new StrategyProposal for the same bot
  before the previous one's `valid_until` is reached
- **THEN** the previous proposal SHALL be treated as invalid, and only the
  newer proposal SHALL be eligible for execution

#### Scenario: Validity is deterministic, not condition-polled by Auto
- **WHEN** a consumer of StrategyProposal checks whether a proposal is
  still valid
- **THEN** it does so by comparing the current time to `valid_until` and
  checking for a newer superseding proposal, never by independently
  re-evaluating the strategy's market assumptions itself

### Requirement: Market Assumptions
Every StrategyProposal SHALL state the objective, falsifiable market
conditions that justified it (e.g. trend remains intact, volatility
remains above a stated threshold, a specific support level is not broken,
a range remains valid). Assumptions SHALL NOT contain subjective or
unmeasurable language.

#### Scenario: Assumptions are objective and falsifiable
- **WHEN** a strategy states a market assumption on a StrategyProposal
- **THEN** the assumption is expressed as a condition that can be checked
  against market data and determined true or false, not a subjective
  judgement

#### Scenario: A broken assumption is surfaced through a new proposal, not Auto re-evaluation
- **WHEN** a strategy's stated assumption becomes false
- **THEN** the strategy's own next evaluation cycle reflects this (via
  Continuous Trade Management and Market Suitability) and produces a
  superseding proposal; a consumer of proposals never re-evaluates a
  strategy's assumptions itself, since doing so would require
  understanding strategy-specific indicators

### Requirement: Proposal Immutability
StrategyProposal objects SHALL be immutable once created. No consumer of
a StrategyProposal - including a future Auto Mode or committee - SHALL
modify a proposal's decision score, evidence, risk fields, reasoning,
parameters, market suitability, assumptions, validity, or any other field
after it is produced.

#### Scenario: A consumer only decides what to do with a proposal, never edits it
- **WHEN** a future Auto Mode or committee evaluates a StrategyProposal
- **THEN** it may select, reject, prioritize, or adjust trust in the
  proposal, but every field the strategy originally set on that proposal
  remains exactly as produced

#### Scenario: New information produces a new proposal, not a mutated one
- **WHEN** conditions change in a way that would affect a prior
  proposal's validity
- **THEN** the system represents this as a new StrategyProposal (or the
  prior one expiring per its validity), never as an in-place edit of the
  original object

### Requirement: Committee Decision Separation
A future multi-strategy committee's output SHALL be represented as a
distinct CommitteeDecision object, separate from the StrategyProposal
objects it consumed. A CommitteeDecision SHALL record which proposals were
considered, which was selected (with allocation and execution priority),
which were rejected (with rejection reasons), and which trust adjustments
were applied - without altering any consumed proposal.

#### Scenario: Committee output is a separate object from the proposal
- **WHEN** a future committee produces a decision from one or more
  StrategyProposal objects
- **THEN** that decision is a CommitteeDecision object referencing the
  proposals by identifier, and every consumed StrategyProposal remains
  unmodified and independently retrievable

#### Scenario: Rejected proposals retain their original reasoning
- **WHEN** a committee rejects a StrategyProposal in favor of another
- **THEN** the rejected proposal's own fields (decision score, evidence,
  reasoning) are unchanged, and the rejection reason lives on the
  CommitteeDecision, not on the rejected proposal

### Requirement: Auto Must Never Understand Strategy Indicators
Auto Mode and any future committee logic SHALL NOT inspect,
interpret, or branch on any strategy-specific indicator (including but
not limited to EMA, ATR, MACD, RSI, or Bollinger Bands). Indicator
interpretation belongs exclusively to the strategy that computed it. Auto
SHALL only read the fixed, strategy-agnostic Comparison Contract subset of
a StrategyProposal's fields.

#### Scenario: Auto compares proposals without indicator knowledge
- **WHEN** Auto Mode or a future committee compares StrategyProposal
  objects from different strategies
- **THEN** it uses only `direction`, `execution_intent`,
  `decision_score.total`/`.threshold`, `suggested_risk_budget_pct`,
  `expected_edge_estimate`, `market_suitability.is_suitable`,
  `edge_status.category`, and `validity.valid_until` - never a
  strategy-specific indicator value

#### Scenario: A new strategy never requires Auto Mode changes
- **WHEN** a new strategy is added that uses indicators no existing
  strategy uses
- **THEN** Auto Mode requires no code changes to evaluate its proposals,
  because it only ever reads the fixed Comparison Contract fields

### Requirement: External Trust Layer
External information sources (news, social sentiment, a Fear & Greed index, macro conditions, funding rates, options or futures data) SHALL NOT modify a StrategyProposal, and SHALL instead produce independent TrustAdjustment records, referencing a proposal by identifier, consumed only by a future committee - never merged into the proposal itself.

#### Scenario: A new external signal source does not require strategy changes
- **WHEN** a new external context source (e.g. a sentiment feed) is added
  to influence trade selection
- **THEN** no strategy implementation changes, because the new source
  produces TrustAdjustment records consumed by a future committee, not a
  change to the proposal schema or the strategies that produce it

#### Scenario: The original proposal remains an unmodified record
- **WHEN** external context adjusts how a proposal is treated (e.g. its
  effective priority or whether it executes)
- **THEN** the original StrategyProposal the strategy produced remains
  retrievable unmodified, separate from whatever TrustAdjustment was
  applied

### Requirement: Expected Edge Requires Statistical Validation
A strategy SHALL NOT invent or self-compute an `expected_edge_estimate`
from its own live or recent trades. This field SHALL remain unpopulated
(`None`) until a statistically validated estimate exists from the
strategy validation framework's walk-forward or backtest results.

#### Scenario: No statistically validated estimate exists yet
- **WHEN** a strategy has not yet undergone walk-forward or backtest
  validation producing a statistically supported edge estimate
- **THEN** its StrategyProposal's `expected_edge_estimate` field is `None`,
  not a self-computed guess

#### Scenario: A validated estimate becomes available
- **WHEN** the strategy validation framework produces a statistically
  supported edge estimate for a strategy
- **THEN** that estimate, and only that estimate, may populate
  `expected_edge_estimate` on subsequent proposals

### Requirement: Strategy Certification Gate
No strategy SHALL be added or materially modified via an OpenSpec change
until it has a complete Strategy Audit Document and passes a written
certification review confirming: the market edge is credible, entry and
exit logic are justified, adaptive behavior exists where appropriate,
overtrading and edge-degradation protections exist and correctly classify
into the three categories, market rejection exists, the Evidence-Based
Decision Score exists with every Evidence Item documented, Decision-Score-
weighted risk sizing exists, diagnostics and Evidence Reports exist, and
the strategy returns a valid StrategyProposal that is: deterministic;
immutable once produced; documented with objective, falsifiable market
assumptions; defined with an explicit validity and expiration; backed by
measurable evidence; reproducibly explainable; free of subjective
information; and sourced with an `expected_edge_estimate` that is either
`None` or drawn only from statistically validated results, never
self-invented.

#### Scenario: Uncertified strategy change is blocked
- **WHEN** a proposal to add or materially modify a strategy does not
  reference a passing certification review against this capability
- **THEN** the proposal does not proceed to implementation

#### Scenario: Certification checklist is evaluated item by item
- **WHEN** a strategy undergoes certification review
- **THEN** each of the nine proposal-quality properties (deterministic,
  immutable, assumptions documented, expiration defined, evidence
  measurable, explanation reproducible, no subjective information,
  execution intent consistent with direction, expected edge sourced
  correctly) is checked and recorded individually, not approved as a
  single unexamined bundle

