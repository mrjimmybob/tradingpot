# Change: Add a strategy decision-making framework and certification standard

## Why
Backtesting 2020-2026 shows every discretionary strategy is technically
implemented (passes unit tests, executes correctly) yet loses money,
overtrades, trades in the wrong market, ignores its own losing streaks, or
keeps trading after its edge is gone. The root cause is architectural, not
individual bugs: every strategy is built as an `indicator -> entry -> exit`
pipeline, not as a trading decision system that continuously asks "should I
trade at all right now?" A professional trader spends most of their time
NOT trading; a fresh, evidence-based audit of all 6 non-Auto-Mode strategies
(see `audits/`) confirms none of them can currently refuse to trade beyond a
single indicator condition, none score trade quality against measurable
evidence, and none know *why* they are no longer profitable when they stop
being profitable.

This change defines the engineering standard every strategy must meet (10
pillars: theory, market suitability, an Evidence-Based Decision Score,
adaptive parameters, Decision-Score-weighted sizing, continuous trade
management, Strategy Edge Management, self-diagnostics, performance
expectations, and a Strategy Proposal Interface), a certification process
that gates any strategy from being implemented or modified until it passes
a written review against that standard, and a strategy-by-strategy audit
of where each of the 6 existing strategies currently stands. A trading
system must never rely on subjective judgement ("looks bullish," "feels
risky") — every scored or classified decision in this framework SHALL be
derived from objectively measurable, reproducible, deterministic evidence.

**Architectural extension (added in this revision)**: strategies
currently remain responsible for producing an executable trade decision
directly. A later, separate change will redesign Auto Mode into an
investment committee that evaluates every strategy before deciding what
to execute — if strategies keep returning bare trade instructions, that
redesign would require rewriting every strategy a second time. This
change adds pillar 10, the **Strategy Proposal Interface**: every strategy
now returns a `StrategyProposal` (a recommendation, not an order) instead
of executing a trade. This is the permanent contract between every
strategy and everything downstream — today's execution pipeline
(unchanged), and, later, Auto Mode, portfolio allocation, and any future
external signal source (sentiment, macro regime, funding/futures data) —
without ever requiring a strategy to be touched again. See `design.md`'s
"The Strategy Proposal Contract" for the full specification, standalone
and future-Auto-Mode execution flows, and the Comparison Contract that
lets Auto Mode rank proposals without knowing what any strategy's
indicators are.

**Final architecture refinement (added in this revision)**: this pass
closes the gaps that would otherwise have forced a second redesign once
Auto Mode implementation began, and freezes the architecture. Direction
alone was insufficient to describe what a strategy wanted done, so every
`StrategyProposal` now carries a separate **Execution Intent**
(`execution_intent`: NO_ACTION, OPEN_POSITION, ADD_TO_POSITION,
REDUCE_POSITION, CLOSE_POSITION, HOLD_POSITION) alongside `direction`,
designed as two independent axes so `direction` can later extend to
SHORT/COVER without `execution_intent` changing shape — no futures/
short-selling logic is added here, only the extension point. Every
proposal now has a formal **Proposal Validity and Expiration**
(`validity.valid_until`, a deterministic timestamp, tightened by
supersession from a newer proposal) so a consumer can never execute a
stale recommendation without needing to understand why it went stale.
Every proposal must state its **Market Assumptions** as objective,
falsifiable conditions — re-checked by the strategy itself on its next
cycle, never independently evaluated by Auto, which would require Auto to
understand strategy-specific indicators. **Proposal Immutability** is now
a formal rule with an explicit list of fields no consumer may ever rewrite
(decision score, evidence, risk, reasoning, parameters, market
suitability, assumptions, validity, or any other field); a consumer only
decides what to do with a proposal, never edits it. The eventual Auto Mode
committee's output is now specified as a distinct **Committee Decision**
object (`CommitteeDecision`: proposals considered, selected allocation and
priority, rejected proposals with reasons, trust adjustments applied) —
documented now, not implemented, so Auto Mode's future design starts from
an already-fixed target instead of guessing. **Auto Must Never Understand
Strategy Indicators** is now a formal architectural rule naming the
specific indicators it applies to (EMA, ATR, MACD, RSI, Bollinger Bands,
and any other strategy-specific value) — Auto only ever reads the fixed
Comparison Contract subset. Future external signal sources (news, social
sentiment, Fear & Greed, macro, funding, options, futures) are formalized
as an **External Trust Layer**: independent `TrustAdjustment` records
referencing a proposal by identifier, consumed only by a future committee,
never modifying the proposal itself. A latent design error from the prior
revision is corrected: `expected_edge_estimate` is now explicitly reserved
exclusively for `add-strategy-validation-tooling`'s statistically
validated walk-forward/backtest results and SHALL remain `None` until such
validation exists — a strategy must never invent its own edge estimate
from recent trades; legitimate per-trade figures (e.g. `expected_move_pct`/
`expected_risk_pct`) instead become a Risk/Reward Evidence Item inside
`decision_score`. Certification now requires nine explicit checks per
strategy: deterministic, immutable, assumptions documented, expiration
defined, execution intent consistent with direction, evidence measurable,
explanation reproducible, no subjective information, and expected edge
sourced correctly.

**Architecture Freeze**: as of this revision, the `StrategyProposal`
contract — including Execution Intent, Proposal Validity, Market
Assumptions, Proposal Immutability, the Comparison Contract, the External
Trust Layer, and the `CommitteeDecision`/`TrustAdjustment` shapes — is
frozen. Phases 0-6 (`tasks.md`) implement against this fixed contract; no
future strategy implementation phase should require architectural
redesign. Any further change to the contract itself requires demonstrated
evidence discovered during implementation or historical backtesting, not
a hypothetical future need — see `design.md`'s "Architecture Freeze"
section. This revision touches no code: `openspec`, `design.md`,
`tasks.md`, and this proposal only.

## What Changes
- Add a "strategy decision framework" capability: shared infrastructure
  (market suitability gate, an Evidence-Based Decision Score engine,
  adaptive parameter resolver, continuous trade-management hook,
  per-strategy Strategy Edge Manager, persisted decision explanations,
  and the `StrategyProposal` schema + a Standalone Adapter that lets a
  proposal flow through today's unchanged execution pipeline) that every
  strategy composes into, replacing ad hoc, inconsistent, or missing
  per-strategy versions of the same concerns.
- Add a Strategy Certification process: no strategy may be added or
  materially modified without a written Strategy Audit Document (theory,
  market suitability, expected performance) and a passing certification
  checklist — now including nine explicit `StrategyProposal` quality
  checks (deterministic, immutable, assumptions documented, expiration
  defined, execution intent consistent with direction, evidence
  measurable, explanation reproducible, no subjective information,
  expected edge sourced correctly) — gating entry into OpenSpec Stage 2
  implementation for any future strategy change.
- Audit all 6 existing strategies (`dca_accumulator`, `adaptive_grid`,
  `mean_reversion`, `trend_following`, `volatility_breakout`,
  `dip_recovery`) against the 10 pillars — see `audits/*.md` — and record
  exactly which pillars each currently fails, with file:line evidence.
- Define a phased, strategy-by-strategy remediation plan (this change's
  `tasks.md`), ordered worst-violation-first, where each strategy's phase
  now also migrates it from returning `TradeSignal` to returning
  `StrategyProposal`.

## Impact
- Affected specs: strategy-decision-framework (new capability)
- Affected code (not touched by THIS change — planning only): all 6
  strategy implementations in `backend/app/services/trading_engine.py`,
  plus new shared infrastructure modules to be added under
  `backend/app/services/`.
- Explicitly NOT touched: `_strategy_auto` / Auto Mode, and no Auto Mode
  logic is implemented by this change. It depends on individual strategies
  being certified first. Auto Mode's own architecture — the investment
  committee that will consume `StrategyProposal` objects as defined here
  without requiring any strategy to change again — is now fully specified
  by the sibling change `add-auto-mode-investment-committee` (also
  architecture-only, also frozen); actual implementation is a later,
  separate change once this one's remediation phases complete.
- Relationship to the existing roadmap:
  - Extends/supersedes part of `add-unified-position-sizing` — pillar 5
    requires Decision-Score-weighted sizing, which that change's current
    design does not yet include (it was designed before this framework and
    referred to a "confidence" input that no longer exists in this
    design). That change's design.md needs revision once this one is
    approved, before it is implemented.
  - Complements `add-strategy-validation-tooling` — pillars 3/4/7's
    thresholds, Evidence Item weights, edge-degradation classification
    cutoffs, and adaptive-parameter formulas all need empirical
    calibration, which is exactly what that tooling's optimizer/walk-
    forward harness provides. Certifying a strategy's specific threshold
    VALUES (not just the presence of a threshold) should use that tooling
    once available.
  - Does not touch `add-trading-safety-boundaries` or
    `fix-regime-detection-consistency` (both already complete) but reuses
    their output: the now-correct bar-based regime detector
    (`fix-regime-detection-consistency`) is the basis for pillar 2's shared
    market suitability gate, and the now-fixed `PortfolioRiskService`
    (`add-trading-safety-boundaries`) is a required input to pillar 5's
    exposure-aware sizing.
- This is a planning-only change: no strategy code is modified until a
  follow-on implementation phase is separately approved per the phasing in
  `tasks.md`. Pillar 10 (`StrategyProposal`) is likewise specification
  only in this revision — no dataclass, adapter, or Auto Mode code is
  implemented here.
- This revision extends the framework approved in the prior version of
  this change; it does not invalidate or remove any of pillars 1-9 — see
  `design.md`'s "The Strategy Proposal Contract" for how pillar 10
  composes the existing pillars' outputs rather than replacing their
  logic.
- This revision is the final architecture pass before implementation
  begins. It formalizes Execution Intent, Proposal Validity, Market
  Assumptions, Proposal Immutability, and Auto's indicator-isolation rule;
  specifies (but does not implement) `CommitteeDecision` and
  `TrustAdjustment` as the future Auto Mode committee's fixed contract;
  and corrects `expected_edge_estimate`'s sourcing rule. No strategy code,
  no Auto Mode code, and no `StrategyProposal` implementation exists after
  this revision — only `openspec/changes/add-strategy-decision-framework/`
  markdown changed. Per "Architecture Freeze" above, the contract itself
  is now frozen going into Phase 0 implementation.
