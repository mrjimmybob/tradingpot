# Change: Define Auto Mode as a portfolio investment committee

## Why
`add-strategy-decision-framework` finished redefining every strategy as an
independent market expert that produces exactly one immutable
`StrategyProposal` per evaluation cycle — a recommendation, never an
order. That change deliberately left Auto Mode unspecified beyond the
`StrategyProposal`/`CommitteeDecision`/`TrustAdjustment` contract shapes,
so strategy implementation could begin without waiting on Auto Mode's own
design. Auto Mode's internal behavior — how it compares proposals from
different strategies, how it ranks them, how it breaks ties, how it
applies portfolio-level risk and future external trust adjustments, and
what it hands to the execution pipeline — was never actually designed.

Without that design finished now, strategy implementation would target a
consumer contract (`StrategyProposal`) whose actual consumer (Auto Mode)
remains an open question — any surprise discovered later in Auto Mode's
design could still force a `StrategyProposal` field change, reopening
every strategy's certification. This change closes that gap: it
completely specifies Auto Mode as the portfolio investment committee, so
that when strategy implementation (Phases 0-6 of
`add-strategy-decision-framework`) begins, it targets an architecture that
is *fully* defined end-to-end, not just defined up to the point where a
strategy stops being responsible.

This is a planning-only change. It does not implement Auto Mode, does not
modify any strategy, and does not touch `_strategy_auto` or any other
code. It defines the architecture so that a later, separate
implementation change can build Auto Mode without further architectural
redesign.

## What Changes
- Add an "Auto Mode Investment Committee" capability: the complete
  specification of Auto Mode as a portfolio-level decision layer sitting
  between strategies (which produce `StrategyProposal` objects) and the
  existing, unchanged execution pipeline.
- Formalize the core philosophy: strategies trade, Auto allocates trust
  and capital — Auto never re-derives a trading decision, it only decides
  which already-made recommendation(s) deserve execution.
- Specify Auto's inputs (one proposal per enabled strategy per cycle, no
  cross-strategy awareness at the strategy level) and reaffirm Proposal
  Immutability from Auto's side: Auto never modifies any `StrategyProposal`
  field, including via external trust adjustments.
- Specify Auto's responsibilities as strictly portfolio-level: comparison,
  ranking, rejection, portfolio exposure/health/concentration/correlation
  validation, capital allocation, execution ordering, expiration checks,
  and (future) external trust adjustments — and explicitly forbid
  indicator interpretation, reaffirming and extending
  `add-strategy-decision-framework`'s "Auto Must Never Understand Strategy
  Indicators" rule.
- Specify the 10-step Committee Process every evaluation cycle runs:
  collect proposals, reject expired, reject assumption-invalidated, reject
  edge-disqualified, apply portfolio risk constraints, apply trust
  adjustments, rank, allocate capital, select, submit to execution.
- Specify deterministic Proposal Ranking, independent of strategy
  identity, with a documented, deterministic tie-breaking policy.
- Specify that the architecture supports executing none, one, or multiple
  proposals simultaneously (without requiring multiple executions to
  always occur).
- Formalize the External Trust Layer as belonging exclusively to Auto
  (never to strategies), producing `TrustAdjustment` records that
  influence ranking only, never proposal content.
- Specify Auto's Portfolio Thinking inputs (correlation, concentration,
  cash reserve, drawdown, per-strategy health/suspension) as inputs
  outside any strategy's responsibility.
- Define extension points for future learning (trust adjustment based on
  validated historical performance) without implementing any learning
  logic now.
- Finalize the `CommitteeDecision` shape (already stubbed as a future
  contract in `add-strategy-decision-framework`) as Auto's sole,
  immutable output, and the only object the execution pipeline accepts
  from Auto.
- Define Auto's own certification requirements, separate from and
  additional to each strategy's certification.
- Declare the Auto Mode architecture frozen upon completion of this
  change, alongside the already-frozen `StrategyProposal` architecture.

## Impact
- Affected specs: auto-mode-investment-committee (new capability)
- Affected code: none. This change implements no code — no
  `_strategy_auto` changes, no committee logic, no ranking/allocation
  code, no `CommitteeDecision`/`TrustAdjustment` classes.
- Explicitly NOT touched: any of the 6 non-Auto-Mode strategies, and
  `add-strategy-decision-framework`'s Phase 0-6 task list (this change
  updates that framework's `design.md`/`proposal.md` only to
  cross-reference this new, more detailed Auto Mode specification —
  it does not alter the `StrategyProposal` contract itself, which
  remains frozen as specified there).
- Relationship to the existing roadmap:
  - Depends entirely on `add-strategy-decision-framework`'s
    `StrategyProposal`/`CommitteeDecision`/`TrustAdjustment` contract
    (frozen) — this change specifies the committee logic that consumes
    and produces those shapes, without changing them.
  - Reuses the existing, already-built `PortfolioRiskService` and
    `StrategyCapacityService` (`add-trading-safety-boundaries`) for
    portfolio-level risk/capacity checks — no new risk-checking
    infrastructure is invented here, Auto calls what already exists.
  - Complements `add-strategy-validation-tooling` — Auto's future
    learning extension points (trust adjustment from validated historical
    performance) are designed to consume that tooling's output, once it
    exists; no learning logic is implemented now.
  - Once this change and `add-strategy-decision-framework`'s Phases 0-6
    are both complete, a future, separate implementation change builds
    Auto Mode itself against both frozen specifications — not scoped or
    scheduled here.
- After this change: both the `StrategyProposal` architecture (frozen by
  `add-strategy-decision-framework`) and the Auto Mode Investment
  Committee architecture (frozen by this change) are complete. Strategy
  implementation may proceed against a fully stable, end-to-end-defined
  consumer.
