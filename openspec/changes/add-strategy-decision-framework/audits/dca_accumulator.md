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

## Pillar 1 — Theory: DOCUMENTED (Phase 6)

**DCA is a capital-deployment discipline, not an alpha strategy.** It
converts a fixed budget into a target asset in **fixed-size, fixed-interval
chunks**, and it takes **no view on price direction** at any horizon. Its
purpose is not to earn a trading edge; it is to deploy capital with low
entry-timing variance and full behavioural discipline. Naming the
"inefficiency" it exploits is therefore a category question: DCA does not
exploit a market inefficiency — it neutralises a *behavioural and
timing-variance* problem on the operator's side.

**The one mechanical property, stated precisely.** Buying a fixed *dollar*
amount at each interval mechanically purchases more units when price is low
and fewer when price is high. The resulting average cost per unit is the
**harmonic mean** of the purchase prices, which is always ≤ their arithmetic
mean (equal only if every price is identical). This is a deterministic
algebraic identity that holds for *every* price path — up, down, or
sideways — and requires **no forecast whatsoever**. It is the whole of DCA's
mechanical claim, and it is why the strategy can be fully deterministic and
still defensible.

**Why direction-agnosticism is load-bearing, not a limitation.** The
harmonic-averaging benefit is produced *precisely by the low-price buys* —
the purchases made when the market is weak. Any rule that skips or reduces
buying in a downtrend (the pre-6.1 regime filter) removes exactly the buys
that most improve the average cost, reintroduces the entry-timing risk DCA
exists to remove, and converts the strategy into a (poor, lagging)
market-timing model. This is the theoretical basis for 6.1's decision to
strip the trend gate: **a classic DCA must buy the dip and the rip alike.**

**What DCA deliberately gives up (honest failure modes).**
- **Expected return vs. lump-sum.** For an asset with positive expected
  drift, lump-sum deployment wins *in expectation* because it is invested
  sooner; DCA carries idle cash while it deploys ("cash drag"). DCA trades
  that expected return away in exchange for lower dispersion of outcomes and
  removal of the timing decision. Underperforming lump-sum in a bull run is
  **correct behaviour, not a defect.**
- **No exit, no loss-cutting.** DCA never sells (Pillar 6, by design). In a
  secular decline that never recovers it loses like buy-and-hold. Its *only*
  protection is the long-term-thesis-invalidation stop (Pillar 7
  Category C) — a structural, non-price condition (delisting, broken
  fundamental premise, operator revocation), never a drawdown trigger.
- **Assumption of eventual recovery / long-run participation.** DCA's
  implicit premise is that the operator wants long-run exposure to this
  asset and can hold through drawdowns. If that premise is false, DCA is the
  wrong tool — but that is a *selection* decision made before the strategy
  runs, not something DCA should try to detect and trade around.

**Reference-benchmark role (a first-class design goal).** Because a classic
DCA is fully determined by `(interval, chunk size, start time, price path)`
and takes no directional view, it is the natural **null model** for
accumulation: any adaptive or AI-assisted accumulation strategy must beat
*this* on a risk-adjusted basis to justify its added complexity and
non-determinism. Preserving DCA's simplicity and determinism is therefore an
explicit objective of this migration — the framework wraps DCA in
governance, auditability, and execution-quality discipline **without** making
it "smarter." Where two implementations are equally correct, the simpler,
more deterministic one is chosen precisely to protect this benchmark role.

**Pre-Phase-6 finding**: docstring gave the mechanism and a coarse
regime-filter rationale but never stated the harmonic-averaging property,
the direction-agnosticism requirement, the honest lump-sum/drawdown
trade-offs, or the benchmark role. Authoring that theory was this
certification task.

## Pillar 2 — Market Suitability: Re-scoped away from market timing (Phase 6.3)

**Pre-Phase-6 state:** a real, enforced refusal-to-trade gate that computed
regime via `_detect_market_regime` and blocked buying when `trend_state`
wasn't in `allowed_regimes` (default `["trend_up","trend_flat"]`), with
`regime_filter_enabled` defaulting `True`. Under the 6.1 decision this was
re-assessed as **market timing** — it skipped scheduled accumulation on
expected short-term weakness (`trend_down`), which is precisely what a
classic DCA must not do — so it was a mis-scoped gate to correct, not a
strength to preserve.

**Implemented (Phase 6.3):** Pillar 2 for DCA is now an
**execution-quality / portfolio-constraint / long-term-thesis** gate, never
a direction gate:

- The trend-regime filter is **removed as the default**
  (`regime_filter_enabled` now defaults `False` in both `_strategy_dca` and
  the `config.py` schema) and survives only as an explicit, clearly-labelled
  **non-classic market-timing overlay** an operator can opt into. When
  disabled (the default) the regime detector is never consulted — direction
  is not an input to the decision (test: `test_regime_detector_not_
  consulted_by_default`).
- The one structural halt on suitability grounds is the new
  **`thesis_invalidated`** stop — a non-price, operator/edge-manager-set
  condition (Pillar 7 Category C, wired in 6.4) that stops all accumulation
  with a `THESIS_INVALIDATED` explain state. It is not a direction forecast.
- **Execution feasibility (a)** and **portfolio constraints (b)** are left to
  the existing downstream mechanisms — the `MIN_ORDER_USD` floor and
  fee-adjusted balance cap (below), and the execution pipeline's
  `PortfolioRiskService.check_portfolio_risk` plus the budget-exhaustion
  floor — rather than duplicated here. Leaning on the single downstream
  source of truth keeps this gate simple and deterministic, protecting DCA's
  reference-benchmark role.

The defining test is the **inverse** of the other five strategies' suitability
tests: `test_trend_down_does_not_block_scheduled_buy` asserts a scheduled buy
still fires in a pure downtrend absent any execution/portfolio/thesis problem.
The non-classic overlay's pause behaviour is preserved under explicit opt-in
(`test_overlay_opt_in_still_pauses_in_downtrend`). 8 tests in
`backend/tests/test_dca_framework_migration.py`; full suite 1237 passing,
zero regressions. See the 6.1 section above and `../design.md`'s
"Pure-accumulator exception."

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

## Pillar 7 — Strategy Edge Management: Present, accumulation-health axis (Phase 6.4)

**Pre-Phase-6 state:** no tracking of any kind in the function.

**Implemented (Phase 6.4):** a dedicated `DcaEdgeManager`
(`strategy_framework/edge_management.py`), wired into `_strategy_dca` exactly
as the other strategies wire `StrategyEdgeManager` (lazy-instantiated,
evaluated before entry, category surfaced via `_explain().metric(
"edge_status_category", ...)` + a `"Accumulation thesis intact"` check, and a
Category-C `should_stop` that halts). It reuses the shared `EdgeStatus` /
`EdgeCategory` / `EdgeSignal` output contract, so DCA is wired to Pillar 7
identically to the rest — **only the classification axis differs**, and
deliberately so.

The load-bearing design point: **a classic DCA does not fail because the
market falls; it fails only when its investment thesis fails.** So the manager
monitors the health of the **accumulation process**, never mark-to-market
profitability. `DcaEdgeManager` records no trade outcomes and computes no
performance statistics — its `evaluate()` exposes **no pnl / price / drawdown /
win-rate parameter at all** (pinned by a signature test), so an ordinary
drawdown *cannot* be an input, let alone a degradation signal.

- **Category C** (stop, human re-certification) — the ONLY halt. Fires solely
  on an objective, permanent, whitelisted invalidation condition:
  `operator_invalidated`, `asset_delisted`, `fundamental_failure`,
  `regulatory_impossibility`, `portfolio_withdrawn`
  (`DCA_THESIS_INVALIDATION_CONDITIONS`). Unrecognised (e.g. price/trend/
  performance-flavoured) keys are ignored, so nothing can invalidate the
  thesis through the back door. The Phase 6.3 `thesis_invalidated` param maps
  to `operator_invalidated`.
- **Category A** (wait, never a stop) — a *temporary* operational/portfolio
  pause. Wired where DCA actually hits one: balance below the minimum order
  (capital temporarily unavailable).
- **Category B** (adapt, never directional) — an operational/portfolio
  parameter adaptation. Wired where DCA actually hits one: the configured
  chunk floored up to the `$MIN_ORDER_USD` executable minimum.
- **NONE** — accumulation process healthy; continue on schedule (the normal
  case, including through a full drawdown).

29 tests in `backend/tests/test_dca_framework_migration.py` (classifier unit
tests for each category, the "unrecognised condition cannot invalidate"
guarantee, the no-price/performance-input signature test, and integration
tests proving a down-regime drawdown stays NONE while each thesis condition
halts with Category C). Full suite 1257 passing, zero regressions.

## Pillar 8 — Self-Diagnostics: Partial

Regime-block state explained (1204-1210) and interval timer state every
tick (1240-1271). Gap: no positive `check()` when the regime check
*passes* (only the failure path is explained); the buy-amount branching
(fixed vs. percent, capped, floored) surfaces only a single `buy_amount`
metric (1355), not the branch logic itself.

## Pillar 9 — Performance Expectations: DOCUMENTED (Phase 6)

Stated honestly and up front: **DCA is not evaluated on return.** Several of
the usual metrics are structurally undefined for it, and saying so is part of
the honest specification.

- **Trade frequency**: exactly **one buy per `interval_minutes`** while
  budget remains and no execution/portfolio/thesis gate defers it —
  **deterministic and regime-independent.** This is DCA's defining property
  and the one that most distinguishes it from the other five strategies:
  frequency does **not** rise or fall with market conditions. (Every other
  migrated strategy's frequency collapses in an unsuitable regime; DCA's does
  not, by design.)
- **Regime behaviour**: **accumulates through every regime, including
  `trend_down`** (the deliberate 6.1 change). The only events that
  defer/trim/stop a scheduled buy are degraded execution quality, a portfolio
  cap / budget exhaustion, or long-term-thesis invalidation — never price
  direction.
- **Holding time**: **infinite** (never sells). No time stop.
- **Win rate**: **not a meaningful metric** — DCA has no round trips, so
  there are no wins or losses to rate. Any win-rate figure reported for DCA
  would be an artefact; the honest value is "N/A (no exits)."
- **Profit factor**: **N/A** for the same reason — no realised losses. All
  P&L is mark-to-market on an ever-growing position until the operator (or a
  separate strategy/rebalancer) exits it outside DCA's scope.
- **Drawdown**: **unbounded mark-to-market by design.** DCA accepts the full
  drawdown of the underlying asset because it never cuts losses. There is no
  drawdown target and no drawdown stop; a bear market shows up as a deepening
  unrealised loss *while DCA keeps buying into it.* This is the explicit,
  accepted cost of being direction-agnostic.
- **Vs. lump-sum**: **underperforms in expectation** for an upward-drifting
  asset (cash drag while deploying); **outperforms** on paths that fall after
  the start and later recover (lower average cost via harmonic averaging);
  **lower outcome variance** than lump-sum in all cases. DCA is deliberately
  on the lower-return / lower-variance side of that trade-off.
- **Vs. buy-and-hold**: during deployment it lags a fully-invested
  buy-and-hold in a bull and cushions it in a bear (partial exposure); once
  the budget is fully deployed it converges to buy-and-hold of the
  accumulated position.
- **Success criterion (what certification actually checks)**: faithful,
  low-variance, **direction-agnostic scheduled deployment** with correct
  governance — *not* a return figure. The 6.8 backtest certifies that DCA
  accumulates on schedule across bull/bear/chop and that every
  skipped/deferred/trimmed buy is attributable to execution quality,
  portfolio limits, or thesis invalidation.
- **Benchmark expectation**: as the project's reference accumulation strategy
  (Pillar 1), DCA's numbers are the **baseline other accumulation strategies
  are measured against**, so its expectations are stated as behaviour to be
  reproduced exactly, not performance to be optimised.

**Pre-Phase-6 finding**: undocumented; no metrics or over/underperformance
conditions stated anywhere.

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
