# Strategy Audit: dca_accumulator

`_strategy_dca`, `backend/app/services/trading_engine.py:1112-1363`.
Coverage score: **4/14** (lowest of the six — see note on structural
status below before treating this as a simple ranking).

## Special note before the pillar audit

DCA is structurally different from the other five: it never sells
(explicit in its own docstring, line 1130: "DCA NEVER SELLS"). Several
pillars (3, 6) don't map onto it the same way they do onto a
signal-driven strategy. Its low score is *not* purely "fewer safeguards
implemented" — part of it is "this strategy type doesn't have an exit
thesis at all," which is itself the finding: DCA's apparent 2020-2026
profitability is a backtest-window artifact (it can't lose in a bull run by
construction, and can't cut losses in a bear market either, because it
never exits — see prior session's analysis). Certifying DCA requires an
explicit design decision — does "never sell" remain correct, or does DCA
need a real exit thesis — before mechanical pillar remediation can even be
scoped. This is why it's still sequenced early despite the caveat (see
`../tasks.md`), not skipped as "structurally exempt."

## Phase 6.1 Design Decision (RESOLVED): keep DCA a classic accumulator

**Decision: Option (a). DCA remains a pure, never-sell, schedule-driven
accumulator that deliberately ignores short- and medium-term market
direction.** It accumulates in timely chunks *no matter the market
conditions*. The Phase 6 framework migration improves **operational
discipline, auditability, portfolio integration, execution quality, and
risk governance** — it does **not** turn DCA into a market-timing strategy.
The certification work in 6.2–6.8 remediates it *as an accumulator*, not by
adding an exit thesis and not by adding a direction-based accumulate/skip
gate.

### The load-bearing distinction: market timing vs. execution quality

Every Phase 6 gate must fall on the correct side of one line:

- **Market timing (NOT allowed for DCA).** Deciding *whether to
  accumulate this interval* on the basis of expected short- or medium-term
  price movement — "the market is trending down, so don't buy," "wait for a
  better price," "momentum is negative, skip." A classic DCA rejects this
  by construction: its entire premise is that it cannot and will not
  forecast direction, so it buys the dip and the rip alike and lets time
  average the entry. **DCA has no directional view and must never act as if
  it does.**

- **Execution quality / portfolio governance / thesis validity (allowed).**
  The schedule has already decided *that* a chunk is due; these gates only
  govern *how well, and whether it is still appropriate,* to deploy it:
  1. **Execution quality** — is *this fill* good? Spread, top-of-book
     liquidity, and expected slippage for the chunk size. A pathological
     spread or thin book justifies deferring or trimming *this fill within a
     bounded tolerance* — it is a statement about execution cost, not about
     where price is going. This must be bounded so it can never degrade into
     "wait indefinitely for a better price" (that would be timing through
     the back door).
  2. **Portfolio constraints** — would this buy breach risk governance?
     Portfolio exposure cap (`PortfolioRiskService`), per-asset
     concentration, or an exhausted budget. Gating here is position/risk
     governance, not a price forecast.
  3. **Long-term investment-thesis invalidation** — is the asset still a
     valid long-term hold at all? This is a *structural* condition (e.g.
     delisting, a broken fundamental premise, the operator revoking the
     thesis), explicitly **not** a short/medium-term price move. This is the
     only condition under which DCA stops accumulating.

Anything justified by (1)–(3) is legitimate; anything justified by "I
expect price to move" is not.

### Why this is correct *for this strategy type* (evidence-based)

The audit's core finding — "its 2020-2026 profitability is a bull-window
artifact; it can't lose in a bull run by construction and can't cut losses
in a bear market because it never exits" — is **accepted as true and is not
a defect to fix.** It is a *performance-expectations disclosure* issue
(Pillar 9), not a design error:

1. **DCA's value is direction-agnostic entry-timing-variance reduction, not
   round-trip alpha.** A pure accumulator converts a fixed capital budget
   into a target asset while minimizing the variance introduced by *when*
   you happen to buy. Spreading buys across time lowers the dispersion of
   the average entry price versus one lump-sum buy at an arbitrary instant.
   That property depends on buying *on schedule regardless of direction* —
   pausing in downtrends (buying only "good" regimes) actively *destroys* it
   by reintroducing the very timing risk DCA exists to remove.

2. **Exit timing is a capability DCA structurally lacks and should delegate.**
   A "sell above threshold" / "rebalance" rule needs an exit thesis
   (Pillar 6): a falsifiable claim about when the position is complete or
   invalidated. DCA has no such view — it is a deployment schedule, not a
   fair-value model. Bolting on a half-justified exit creates a round-trip
   strategy with an unmeasured exit edge, strictly worse than delegating
   exits to a layer that actually has one (a signal strategy, a portfolio
   rebalancer, or the operator). **Conflating accumulation with exit timing
   is the failure mode, not the fix.**

3. **Risk management for an accumulator is execution + governance + thesis,
   not direction.** The correct controls are: execute each chunk well
   (execution-quality gate), stay inside portfolio risk limits (exposure
   cap / concentration), size sanely (fixed schedule chunks, trimmed only by
   portfolio caps), and stop only when the *long-term thesis* is invalidated
   (Pillar 7 Category C). None of these require a directional forecast, and
   none require an exit.

### What this changes in the CURRENT implementation

This decision **reverses the existing trend regime filter's role.** Today
DCA defaults `regime_filter_enabled=True` with
`allowed_regimes=["trend_up","trend_flat"]`, so it *pauses buying in
`trend_down`* (`trading_engine.py:1223-1261`). **That gate is market timing**
— it skips scheduled accumulation because of expected short-term
weakness — and it is exactly what a classic DCA must not do. Phase 6.3
re-scopes Pillar 2 for DCA: remove the trend `allowed_regimes` gate as the
default (retained only as an explicit, clearly-labelled non-classic
operator override, off by default) and replace it with the
execution-quality / portfolio-constraint / thesis-validity gate above. This
is a **deliberate, documented behaviour change**: the 6.8 before/after
backtest is expected to show DCA now *keeps accumulating through the bear
window* where it previously paused. See Pillar 2 below and `../design.md`'s
"Pure-accumulator exception."

### What this decision obligates the rest of Phase 6 to do

Choosing (a) is **not** a licence to certify DCA as "profitable in all
regimes." It obligates the opposite — an honest Pillar 9 (6.2) stating
plainly:

- DCA will **underperform lump-sum** deployment in a monotonic bull market
  (cash drag while capital is still being deployed);
- DCA will **keep accumulating through, and hold, the full drawdown** in a
  bear market with no loss-cutting and no pause — accepting mark-to-market
  drawdown is the price of being direction-agnostic;
- DCA's defensible claim is **reduced entry-timing variance, disciplined
  scheduled deployment, and good execution quality**, verified by the 6.8
  bull/bear/**chop** backtest showing it accumulates on schedule across all
  regimes and that any skipped/deferred/trimmed buys are attributable to
  execution quality, portfolio limits, or thesis invalidation — *never* to
  price direction. The claim is discipline + execution + governance, not a
  return figure.

And it fixes the Pillar 7 edge-management design (6.4): for a pure
accumulator, **ordinary mark-to-market drawdown is NOT a degradation signal
— the strategy is designed to hold through it.** The meaningful Category C
signal is **long-term investment-thesis invalidation** (a structural stop
requiring human re-certification), not a losing streak or a falling balance.
Category A/B reduce to *execution/portfolio* adaptation (e.g. chunk size vs.
available liquidity), never a directional "wait for a better regime."

### Consequences for Pillar 10 (6.7)

With never-sell settled, DCA never emits `SELL`; `direction` is only ever
`BUY` (scheduled buy) or the non-buying tick. A non-buying tick means "no
*entry* decision is being made this interval" (the schedule isn't due, or an
execution/portfolio/thesis gate deferred it), which reads as `NO_TRADE` more
naturally than `HOLD` (HOLD implies an actively managed open position, which
a pure accumulator does not have). Either way `execution_intent=NO_ACTION`,
so the Standalone Adapter behaves identically — see 6.7. Every stated
proposal assumption must be objective and execution/portfolio/thesis-based
(e.g. "spread within tolerance," "exposure below cap," "long-term thesis not
invalidated"), never a direction forecast.

## Pillar 1 — Theory: UNDOCUMENTED (partial)

Docstring (1119-1149) describes mechanism and a coarse regime-filter
rationale ("protects capital... during strong downtrends," 1132-1133) but
never states *why* DCA has edge, when it should underperform lump-sum
investing, or its core assumption (a long-run uptrend). No inefficiency is
named. **Authoring real theory is a certification task, not done here.**

## Pillar 2 — Market Suitability: Present, but mis-scoped for a classic DCA (see 6.1)

Real, enforced refusal-to-trade gate (1179-1216): computes regime via
`_detect_market_regime`, blocks buying when `trend_state` isn't in
`allowed_regimes` (default `["trend_up","trend_flat"]`, line 1175).
`regime_filter_enabled` defaults `True` but is user-disableable.

**Re-assessment under the Phase 6.1 Design Decision:** this trend-regime
gate is *market timing* — it skips scheduled accumulation because of
expected short-term weakness (`trend_down`), which is precisely what a
classic DCA must not do. It is therefore **not** a strength to preserve but
a mis-scoped gate to correct. Phase 6.3 removes it as the default (retained
only as an explicit non-classic operator override, off by default) and
redefines Pillar 2 for DCA as an **execution-quality / portfolio-constraint
/ long-term-thesis** gate — never a direction gate. See the 6.1 section
above and `../design.md`'s "Pure-accumulator exception."

## Pillar 3 — Evidence-Based Decision Score: Not present

Entry fires purely on elapsed clock time (`_interval_ok`, 1242,
1274-1298) — no multi-factor scoring at all.

## Pillar 4 — Parameter Adaptation: Not present

All hardcoded: `interval_minutes=60`, `amount_percent=10`,
`allowed_regimes` (1170-1175). Nothing scales with volatility or regime
beyond the binary allow/deny gate.

## Pillar 5 — Position Sizing: Flat

`buy_amount = bot.current_balance * amount_percent` (line 1316, fixed 10%
of balance), or a fixed USD override — no Decision Score, exposure, or
drawdown input.

## Pillar 6 — Trade Management: Not applicable (by design, resolved)

No post-entry logic exists because there is no position lifecycle to
manage — DCA only ever buys. **Resolved in the Phase 6.1 Design Decision
above: this remains correct.** DCA stays never-sell; Pillar 6 is
legitimately Not Applicable for a pure accumulator (documented, not merely
absent). Risk management is relocated entirely to the entry side (Pillars
2/5/7).

## Pillar 7 — Strategy Edge Management: Not present

No tracking of DCA's own performance anywhere in the function.

## Pillar 8 — Self-Diagnostics: Partial

Regime-block state explained (1204-1210) and interval timer state every
tick (1240-1271). Gap: no positive `check()` when the regime check
*passes* (only the failure path is explained); the buy-amount branching
(fixed vs. percent, capped, floored) surfaces only a single `buy_amount`
metric (1355), not the branch logic itself.

## Pillar 9 — Performance Expectations: UNDOCUMENTED

No stated win rate, profit factor, drawdown, or over/underperformance
conditions vs. buy-and-hold anywhere in the docstring. **To be authored at
certification**, ideally after Pillar 6's design question is resolved.

## Pillar 10 — Strategy Proposal Interface: Not present

Returns `TradeSignal` directly (`trading_engine.py:1357`,
`return TradeSignal(action="buy", ...)`), the same as every other
strategy — no `StrategyProposal` concept exists yet anywhere in the
codebase. Migration note: DCA's "never sells" shape makes its
`HOLD`-vs-`NO_TRADE` mapping for non-buying ticks a genuine open question
once it has accumulated holdings — see `../tasks.md` Phase 6.7. This
strategy's migration should happen only after Phase 6.1's design decision
(does never-sell remain correct) is resolved, since that decision affects
whether `SELL`/`HOLD` are ever meaningful directions for it at all.
