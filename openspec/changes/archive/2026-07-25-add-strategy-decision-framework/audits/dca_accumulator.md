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

## Pillar 3 — Evidence-Based Decision Score: Intentionally absent by design (certified 6.8)

Entry fires on the deterministic clock schedule, not a multi-factor score —
**and that is correct for a classic DCA, not a gap.** A conviction/edge score
that decided *whether* to buy would be market timing (6.1/6.3); one that
scaled *how much* would be adaptive sizing (6.5). Both are explicitly
forbidden for a direction-agnostic accumulator. The frozen `StrategyProposal`
contract still requires a `decision_score` field, so each proposal carries a
single *descriptive* deterministic evidence item restating the schedule/thesis
decision (Pillar 10) — it introduces no scoring intelligence and never gates
or sizes. This is the framework's documented "no real score → describe the
decision" path, exercised by adaptive_grid's non-scored branches too.

## Pillar 4 — Parameter Adaptation: Intentionally absent by design (certified 6.8)

All parameters are **fixed by design**: `interval_minutes` (schedule),
`amount_percent`/`amount_usd` (chunk), `thesis_invalidation` flags
(structural), and the off-by-default `regime_filter_enabled` overlay. Nothing
scales with volatility, regime, or price — the framework's Pillar 4 "document
why fixed" escape hatch applies in full: a classic DCA's value comes from
*not* adapting (deterministic, benchmarkable deployment). `adaptive_
parameters_used` on every proposal is empty, confirming no adaptation runs.

## Pillar 5 — Position Sizing: Flat by design, deterministic (Phase 6.5)

`buy_amount = bot.current_balance * amount_percent` (fixed % of balance), or
a fixed USD override — no Decision Score, no price, no regime, no
expected-move input. **This is the correct implementation, not a gap**, and
Phase 6.5 makes that a documented, tested design decision rather than an
unexamined default.

**Why flat scheduled sizing is correct for a classic DCA under the
framework.** The framework's general rule (design.md Pillar 5,
`add-unified-position-sizing`) is that sizing takes the Decision Score as an
input so conviction scales exposure. That rule exists to *maximise
risk-adjusted return* for signal-driven strategies. A classic DCA has a
different objective: it does **not** attempt to maximise return — it deploys
capital **consistently and professionally**. Scaling chunk size by
conviction, volatility, or expected move would be:

- **market timing through sizing** — buying more when the model "likes" the
  price is a directional bet, which 6.1/6.3 established DCA must not make; and
- **a determinism/benchmark violation** — DCA is the project's reference
  accumulation strategy (Pillar 1). Its value as a benchmark depends on being
  fully determined by `(interval, chunk, start, price path)`. Any adaptive
  sizing contaminates the baseline that future adaptive/AI accumulators are
  measured against.

So DCA deliberately takes **no** Decision Score (it has none — 6.3 added no
score) and no conviction weighting. Per the framework's own "document why
fixed" escape hatch (Pillar 4/5), flat schedule-driven sizing is the
justified, correct choice.

**The only legitimate adjuster is objective portfolio governance — enforced
downstream, not re-implemented here.** Consistent with the single-source-of-
truth pattern from 6.3, DCA emits a flat chunk and the execution pipeline
applies objective governance to it uniformly:
`PortfolioRiskService.check_portfolio_risk` blocks or **resizes** (only ever
a *reduction*, `adjusted_amount`) at STEP 3, and `StrategyCapacityService`
trims at STEP 4 (`trading_engine.py` ~8237–8286). At the strategy layer, the
fee-adjusted `_BUY_BALANCE_FRACTION` cap and the `MIN_ORDER_USD` floor keep a
buy inside available funds. None of these is an increase and none is
directional — matching the framework requirement that sizing respect the
portfolio exposure cap without DCA duplicating that logic.

**Behaviour unchanged → no backtest required for 6.5.** The sizing math is
identical to pre-migration; 6.5 adds documentation, in-code Pillar-5 markers,
and tests only. The phase's one behaviour change (the 6.3 regime-gate
removal) is certified by the 6.8 bull/bear/chop backtest. Six sizing tests in
`backend/tests/test_dca_framework_migration.py` (flat %, fixed-USD override,
price/regime independence, cross-call determinism, fee-adjusted budget cap,
and a source-level guarantee that `_strategy_dca` contains no
Decision-Score-weighted sizing path). Full suite 1263 passing, zero
regressions.

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

## Pillar 8 — Self-Diagnostics: Full (Phase 6.6)

**Pre-Phase-6 state:** regime-block state and interval-timer state were
explained, but only the *failure* paths were — there was no positive
`check()` when the gate passed, and the buy-amount branching (fixed vs.
percent, capped, floored) surfaced only a single `buy_amount` metric, not the
branch logic.

**Implemented (Phases 6.3–6.6):** every decision point now has a `.check()`/
`.metric()`, passing paths included:

- **Suitability / thesis (Pillars 2/7)**: the `"Accumulation thesis intact"`
  check is emitted on *every* tick — passing when the thesis is valid, not
  only on a Category-C halt — alongside the `edge_status_category` metric
  (6.4). The optional non-classic overlay still explains its `WAITING_REGIME`
  pause when opted in.
- **Schedule (Pillar 2)**: the `"Interval elapsed"` check and interval-timer
  metrics every tick, with `INTERVAL_DUE` / `WAITING_INTERVAL` state.
- **Buy-amount branching (Pillar 8's specific gap, closed in 6.6)**: the buy
  path now records `sizing_basis` (fixed_usd vs percent), `chunk_raw` (the
  pre-floor/cap chunk), `sizing_floored`, `sizing_capped`, and the final
  `buy_amount`, plus two execution-feasibility checks — `"Buy clears minimum
  order size"` and `"Buy within fee-adjusted budget"` — with a `BUYING`
  state. The two operational branches are explained too: the chunk-floor
  (Category B) records the adaptation, and the budget-exhaustion (Category A)
  emits a *failed* `"Buy clears minimum order size"` check with an
  `ACCUMULATION_PAUSED` state.

Portfolio-cap adjustments are intentionally *not* re-explained here — they
happen in the execution pipeline (Pillar 5 / 6.5), which has its own
diagnostics, keeping the strategy's single source of truth. 5 diagnostics
tests in `backend/tests/test_dca_framework_migration.py` (passing-suitability
explained on a buy, flat/floored/fixed-USD branching metrics, and the failed
min-order check on budget exhaustion). Full suite 1268 passing, zero
regressions.

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

## Pillar 10 — Strategy Proposal Interface: Present (Phase 6.7)

**Pre-Phase-6 state:** returned `TradeSignal` directly, like every strategy
before migration; no `StrategyProposal` concept.

**Implemented (Phase 6.7):** `_strategy_dca` now builds a `StrategyProposal`
per evaluation and routes it through the `StandaloneAdapter` (the executor
still returns `TradeSignal`, exactly as the other five migrated strategies
do — the proposal is the internal decision object, the adapter translates it
into the unchanged `_execute_trade` pipeline).

The open `HOLD`-vs-`NO_TRADE` question from the pre-migration note is
**resolved by the 6.1 design decision**: a pure accumulator has no
actively-managed open position, so a non-buying tick is `NO_TRADE` +
`NO_ACTION`, never `HOLD`/`HOLD_POSITION`. Intent mapping:

- **Scheduled buy, first ever (`order_count == 0`)** → `BUY` +
  `OPEN_POSITION`.
- **Scheduled buy, subsequent** → `BUY` + `ADD_TO_POSITION` (accumulation is
  incremental).
- **Every non-buying tick** — thesis invalidated (Category C), interval not
  elapsed, budget exhausted (Category A), or the opt-in overlay pausing —
  → `NO_TRADE` + `NO_ACTION`. The adapter returns `None` for these no-order
  intents, so the original explicit hold reason is preserved unchanged.
- DCA **never emits `SELL`** (never-sell), verified by a test sweeping every
  branch.

Other contract details:

- **`validity.valid_until`** is tied to the **buy interval** (`generated_at +
  interval_minutes*60`, floored at 1s for the zero-interval backtest config),
  not a candle timeframe.
- **Assumptions** are objective / falsifiable and execution/portfolio/thesis-
  based — "long-term investment thesis remains valid", "buy clears the $10
  minimum order size", "buy fits within available balance after fees" — with
  a test asserting no directional word appears.
- **`decision_score`**: DCA has no conviction score (6.3/6.5), but the frozen
  contract requires the field, so — exactly as adaptive_grid does for its
  non-scored branches — each proposal carries a single *descriptive*,
  deterministic evidence item restating the schedule/thesis decision (weight
  100, threshold 0). It introduces no scoring intelligence and never scales
  sizing.
- **`adaptive_parameters_used`** is empty (flat by design); **`expected_edge_
  estimate`** is `None` (never self-computed).
- **Behaviour preserved**: the buy is routed with `is_accumulation=True`, so
  the execution outcome (action `buy`, amount = the flat chunk, market order,
  accumulation flag) is identical to the pre-migration `TradeSignal` path — a
  dedicated test asserts this. Proposal immutability and deterministic
  `proposal_id` are tested; an expired-proposal test confirms discard.

14 Pillar-10 tests in `backend/tests/test_dca_framework_migration.py` (53 in
the file). Full suite 1282 passing, zero regressions; three pre-existing
auto-mode DCA tests updated from the old `"DCA buy #N"` reason string to the
mechanically-derived accumulation reason (execution outcome unchanged).

## Backtesting & Certification (Phase 6.8)

**No strategy code was changed during certification** — no defect was found,
no parameter was tuned, no threshold optimised. This phase measures the
completed implementation only.

**Method.** Same migrated engine throughout. "AFTER" = the migrated default
(`regime_filter_enabled=False`). "BEFORE" = the migrated code with the *old
default* regime overlay re-enabled (`regime_filter_enabled=True,
allowed_regimes=["trend_up","trend_flat"]`), which reproduces the
pre-migration gate. binance BTCUSDT 1m, `--starting-balance 10000`, default
`interval_minutes=60`, standard windows used across Phases 1–5.

**Results (AFTER, migrated default):**

| Window | Dates | DCA return | Buy&Hold | DCA − B&H | Max DD | Fees (≈ deployed) |
|---|---|---|---|---|---|---|
| Bull | 2023-10-01→11-15 | +30.72% | +31.84% | **−1.12 pp** | 8.04% | $9.98 |
| Bear | 2022-11-01→12-15 | −13.26% | −13.07% | **−0.19 pp** | 27.72% | $9.98 |
| Chop | 2024-04-01→05-15 | −11.94% | −13.59% | **+1.65 pp** | 22.16% | $9.98 |

("Trades = 0" in every window because DCA never closes a round trip — the
metric counts completed buy→sell cycles, which a never-sell accumulator has
none of. The real activity is the ≈$9,980 deployed, reflected in fees.)

**Before/After comparison — the key finding.** BEFORE and AFTER are
**byte-identical** in every window (bear: both $8,674.47 / −13.26%; bull: both
$13,071.75 / +30.72%). The reason is important and stated honestly: **in the
backtest harness the overlay's regime detector always returns `flat`** — the
harness does not populate the live tick price-history that
`_detect_market_regime` needs for trend detection (the documented "regime flat
until history populated" behaviour). Since `flat` is inside the old default
`allowed_regimes`, the pre-migration gate *never actually paused in backtests
either* — the trend-pause was always a **live-only** behaviour. Consequently
the regime-removal difference **cannot be demonstrated through the CLI
backtest**; it is instead proven by unit tests that control the regime input
directly (`test_trend_down_does_not_block_scheduled_buy`,
`test_overlay_opt_in_still_pauses_in_downtrend`,
`test_regime_detector_not_consulted_by_default`).

**Gate-engaged proxy.** Restricting the overlay to `allowed_regimes=
["trend_up"]` (so the always-`flat` harness regime is disallowed) makes the
gate fire on every tick: **$10,000.00 / +0.00% / $0 fees — zero capital
deployed.** This is a faithful backtest illustration of what a regime-gated
accumulator does when its gate rejects the prevailing regime (the live bear
case), versus the classic AFTER which deploys the full ≈$9,980 on schedule.

**Determinism.** The bear AFTER run reproduced **byte-identically** across two
independent invocations ($8,674.47 / −13.26%), and fees are **exactly $9.98 in
all three regimes** — the same deterministic deployment regardless of market
direction.

### Explicit verifications required by Phase 6.8

- **Accumulates through sustained bear markets (no regime pause):** ✅ Bear
  AFTER deployed ≈$9,980 on schedule across the entire 6-week downtrend, ending
  $8,674 (tracking the −13% asset) — it did **not** pause. Contrast the
  gate-engaged proxy's $0 deployment. Trend-down-specific non-pause proven by
  unit test.
- **Accumulation schedule remains deterministic:** ✅ Byte-identical re-runs;
  identical $9.98 fees across all regimes; 53 deterministic unit tests.
- **No market timing introduced:** ✅ Regime detector not consulted by default
  (test); BEFORE≡AFTER for the default gate; suitability is
  execution/portfolio/thesis only.
- **No adaptive sizing introduced:** ✅ `adaptive_parameters_used` empty on
  every proposal; identical deployment/fees across regimes; flat-sizing tests.
- **No selling behaviour introduced:** ✅ Never emits `SELL` (branch-sweep
  test); Trades = 0 (no round trips) in every window.
- **Portfolio governance remains external:** ✅ Enforced by the execution
  pipeline (`PortfolioRiskService` resize/block, `StrategyCapacityService`),
  not the strategy (Pillar 5).
- **`StrategyProposal` migration behaviour-identical from the pipeline's
  view:** ✅ BEFORE≡AFTER numbers (the proposal layer is present in both and
  changes no outcome); the buy execution-outcome test; NO_TRADE holds preserve
  their original reasons.

## Certification Report (Phase 6.8)

- **Was the implementation correct?** Yes. The full suite (1282 tests) passes
  with zero regressions, the strategy runs cleanly across bull/bear/chop, and
  results match the Pillar 9 expectations quantitatively. **No defect was
  found and no strategy code was changed during certification.**
- **Did the migration preserve the intended philosophy?** Yes. DCA remains
  direction-agnostic, never-sell, flat-sized, and fully deterministic; the
  reference-benchmark role is intact.
- **What behaviour intentionally changed?** Exactly one thing: the trend
  regime overlay is **off by default**, so a classic DCA accumulates through
  downtrends instead of pausing (6.3). This is a **live-behaviour** change; it
  is not visible in the backtest harness (regime stays `flat` there) and is
  certified by unit tests. Secondary, non-behavioural: the return type is now
  `StrategyProposal`, and diagnostics/edge-management are far richer.
- **What behaviour intentionally did NOT change?** Chunk sizing (flat),
  schedule (deterministic), never-sell, the execution outcome of a buy, and
  the *location* of portfolio governance (external pipeline).
- **What measurable benefits were obtained?** Not return. The benefits are:
  (1) correct-axis edge management — thesis invalidation, never drawdown (6.4);
  (2) full self-diagnostics — every decision point explained (6.6);
  (3) `StrategyProposal`/governance integration (6.5/6.7); (4) honest,
  documented performance expectations that the backtests confirm; (5) a
  deterministic, benchmarkable baseline for future accumulation strategies.
- **What limitations remain?** DCA **underperforms Buy & Hold in the bull
  window (−1.12 pp, cash drag)** — stated plainly, and expected. It roughly
  matches B&H in the bear (−0.19 pp) with **unbounded mark-to-market drawdown
  by design** (no loss-cutting). It only *beats* B&H in the chop window
  (+1.65 pp, the entry-averaging benefit). The backtests are single-window,
  1-minute, not a walk-forward, and **DCA makes no positive-edge claim.** The
  regime-removal difference is not observable in the harness (verified by unit
  tests instead).

**Honest bottom line.** DCA does **not** beat Buy & Hold in general — it loses
to it in the bull run and ~ties in the bear, winning only in chop. That is the
correct and expected result for a classic DCA, whose purpose is disciplined,
low-variance, direction-agnostic deployment, not return maximisation. On that
purpose the implementation is **behaving exactly as a professional
implementation of classic DCA should**: identical fixed-chunk deployment across
every regime, never selling, never timing, never adapting, fully deterministic,
fully auditable, with governance external and edge management tied to the
investment thesis rather than to price. **Certified.**

## Certification Checklist

- [x] Audit has no `UNDOCUMENTED` sections (Pillars 3/4 documented as
      intentionally-absent-by-design).
- [x] Market suitability re-scoped and enforced (execution/portfolio/thesis);
      direction never gates (inverse-suitability test).
- [x] Decision Score: intentionally absent by design, documented (no market
      timing, no conviction sizing); frozen-contract field carried as a
      descriptive, non-scoring evidence item.
- [x] Every parameter classified fixed-with-reason (Pillar 4); no undocumented
      constants; `adaptive_parameters_used` empty.
- [x] Sizing flat/deterministic; portfolio exposure cap respected via the
      external pipeline (Pillar 5).
- [x] Trade management N/A by design (never-sell), documented (Pillar 6).
- [x] Edge management wired on the accumulation-health axis; A/B/C classified;
      never force-closes; drawdown cannot trip C (signature test).
- [x] Self-diagnostics: every decision point has a `.check()`/`.metric()`,
      passing paths included; Evidence Report renders.
- [x] Returns `StrategyProposal`; adapter routes through the unchanged
      pipeline; buy execution outcome behaviour-identical (test).
- [x] `execution_intent` consistent with `direction`; `validity.valid_until`
      set (buy interval); expired proposal discarded (test); assumptions
      objective; proposal immutable + deterministic `proposal_id`;
      `expected_edge_estimate` is `None`.
- [x] Before/after backtest recorded across bull/bear/chop; behaviour change
      (regime removal) certified by unit tests where the harness cannot show
      it; honest report produced.
- [x] Full suite passes (1282); new tests cover every pillar.

**Status: certified — implementation complete, philosophy preserved, no defect
found, success not overstated.**
