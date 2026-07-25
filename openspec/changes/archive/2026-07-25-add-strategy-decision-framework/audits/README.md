# Strategy Audit Documents

One file per non-Auto-Mode strategy, auditing its CURRENT state against the
10-pillar standard defined in `../design.md`. These are evidence-based
snapshots (file:line citations against `backend/app/services/
trading_engine.py` as of this audit) — no code was changed to produce them.

Each file uses the same structure:
- **Pillars 1 & 9 (Theory / Performance Expectations)**: these are
  documentation pillars. Where the strategy's docstring doesn't already
  answer them, the section says so explicitly (`UNDOCUMENTED`) rather than
  inventing plausible-sounding rationale — authoring real theory and
  performance expectations is deliberately left as a certification task for
  someone with trading domain judgment (and, where possible, real
  walk-forward evidence from `add-strategy-validation-tooling`), not
  something to fabricate during an audit.
- **Pillars 2-8**: current-state findings with citations, rated
  Full / Partial / Not Present.
- **Pillar 10 (Strategy Proposal Interface)**: added in a later revision,
  after the coverage-score table below was established. Every strategy is
  uniformly "Not present" — none of the six return anything but
  `TradeSignal` today, since `StrategyProposal` didn't exist until this
  revision. It's intentionally excluded from the /14 coverage score
  (adding a column every strategy scores identically on wouldn't change
  the relative ranking, only the arithmetic) and does not change the
  Phase 1-6 ordering below — see each audit's own "Pillar 10" section for
  strategy-specific migration notes (e.g. which existing computations,
  like dip_recovery's `opportunity_score`, carry forward into proposal
  fields).
- **Coverage score**: pillars 2-8 rated 0 (not present) / 1 (partial) / 2
  (full), summed out of 14, used only as a rough remediation-priority
  signal — see `../design.md` and `../tasks.md` for how phase order was
  actually decided (pillar-2 enforcement gaps weighted above raw score,
  since "trades in inappropriate markets" was the user's first-named
  symptom).

| Strategy | P2 Suitability | P3 Decision Score | P5 Sizing | P6 Mgmt | P7 Edge Management | P10 Proposal | Score /14 |
|---|---|---|---|---|---|---|---|
| dca_accumulator | Full | Not present | Flat | N/A (never sells) | Not present | Not present | 4 |
| trend_following | **Not enforced** | Not present | Risk-scaled | Full | Not present | Not present | 5 |
| volatility_breakout | **Computed, not enforced** | Not present | Risk-scaled | Partial | Not present | Not present | 5 |
| mean_reversion | Full | Not present | Flat | Full | Not present | Not present | 6 |
| adaptive_grid | Full | Not present | Flat+depth | Partial | Not present (telemetry only) | Not present | 6 |
| dip_recovery | **Not enforced** | Partial (computed, unused) | Risk-scaled | Full | Not present | Not present | 7 |

**Zero strategies have Pillar 3 (Evidence-Based Decision Score), Pillar 7
(Strategy Edge Management), or Pillar 10 (Strategy Proposal Interface)
today.** Three strategies (trend_following, volatility_breakout,
dip_recovery) have **zero enforced** market suitability when run
standalone — volatility_breakout computes the check and silently discards
it, trend_following and dip_recovery never compute it at all. This is the
most acute, common violation and directly matches the user's first-named
symptom ("trade in inappropriate markets... ignore market regime").

**Terminology note**: pillars 3 and 7 were originally named "Trade Quality
Score"/"Confidence" and "Failure Detection" in an earlier draft. Both were
replaced during design review — "confidence" is not something a trading
system can measure, and "failure detection" only answers *that* a strategy
is losing, not *why*. See `../design.md`'s revision note for the full
rationale. This file and the six per-strategy audits below use the current
terminology throughout.

**Architecture note**: pillar 10 (Strategy Proposal Interface) was added
in a later revision. Strategies no longer execute trades directly — they
return a `StrategyProposal` (a recommendation), which flows through a
Standalone Adapter into the existing, unchanged execution pipeline today,
and will be the contract a future Auto Mode investment committee consumes
without requiring any strategy to be rewritten again. See
`../design.md`'s "The Strategy Proposal Contract" for the full
specification, architecture diagrams, and the standalone/future-Auto-Mode
execution flows.

**Template note** (Phase 0.8): these six files are CURRENT-STATE audits —
what exists today, evidenced by file:line citations, with `UNDOCUMENTED`/
"Not present" where a pillar is missing. `TEMPLATE.md` in this directory
is the certification-ready structure each of these six becomes DURING that
strategy's own Phase 1-6 remediation (`../tasks.md`) — every placeholder
filled in, every gap closed, ready for the certification review
`../specs/strategy-decision-framework/spec.md`'s "Strategy Certification
Gate" requirement gates on. Do not treat the current-state audits below as
the final form; they are the starting evidence base for `TEMPLATE.md`, not
a substitute for it.
