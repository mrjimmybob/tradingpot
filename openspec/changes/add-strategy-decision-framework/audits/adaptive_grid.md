# Strategy Audit: `adaptive_grid`

`_strategy_grid`, `backend/app/services/trading_engine.py` (helpers
`_get_grid_state`/`_save_grid_state`/`calculate_atr_proxy`/`_ema_slope_pct`).

**Certification status**: ☐ Draft ☑ Under review ☐ Certified (date: ____ —
pending human sign-off on the before/after backtest comparison and the
Self-Audit findings)
**Certified against**: `add-strategy-decision-framework` 10-pillar standard,
Architecture Freeze revision (Phase 5 migration)

This document supersedes the pre-migration current-state audit (coverage
**6/14** — the most mechanically complex of the six: virtual grid inventory,
not a single position; a real suitability gate and two kill switches already,
but zero quality scoring, flat×depth sizing, unused edge telemetry, and two
kill switches invisible to diagnostics). Original findings preserved as
**Pre-Phase-5 finding** under each pillar.

## Pillar 1 — Theory: DOCUMENTED (Phase 5)

The grid **manufactures profit from mean reversion inside a bounded range**.
In a range-bound (non-trending) regime, price oscillates around a center; the
grid pre-positions resting buy levels below and sell levels above and harvests
each oscillation as a completed buy→sell cycle whose gross profit is one grid
spacing. Long-biased for crypto's secular drift (7 buy / 3 sell of 10 levels):
the process accumulates the base asset at progressively better prices on the
way down (convex payoff via the depth multiplier) and realises inventory into
strength on the way up.

**Why the edge should exist statistically**: in a genuine range, short-horizon
price moves are dominated by transient order-flow (liquidity provision,
noise-trader imbalance) that reverts, not by information that trends. A grid is
a systematic liquidity-provision / market-making posture that earns the
bid-offer-like spread between adjacent levels as long as reversion dominates.

**Failure mode (explicit)**: the edge exists ONLY while the market ranges. A
sustained trend turns "buy the dip" into "catch a falling knife" — inventory
accumulates into a one-way move and never gets sold back at a profit. The two
kill switches are the theory's own falsification test: (a) a trend the regime
gate misses shows up as a **range-escape kill** (price leaves the grid faster
than the soft recenter reacts) — a spacing/multiplier *parameter* mismatch
(Pillar 7 Category B); (b) a sustained bleed shows up as a **drawdown kill** —
a genuine loss of edge (Pillar 7 Category C candidate).

**Pre-Phase-5 finding**: docstring gave the design philosophy and risk
controls but never stated *why* a range-bound edge should exist statistically
nor a formal failure-mode discussion beyond "pauses in trends/high
volatility." Authoring that theory was a certification task — done above and in
the strategy docstring.

## Pillar 2 — Market Suitability: Full (now routed through the shared gate)

Real, enforced gate: regime is computed from bar closes via the price-only
`_detect_market_regime` and evaluated by the shared `MarketSuitabilityGate`
against `allowed_regimes` (default `["trend_flat","volatility_medium"]`); an
unsuitable regime returns a `NO_TRADE`/`NO_ACTION` proposal with a
`WAITING_REGIME` explain state. The gate's OR-across-tags semantics reproduce
the pre-migration `any(r in allowed_regimes ...)` verdict bit-for-bit for the
default tags. `regime_filter_enabled=False` maps to `allowed=["all"]` so a
proposal always carries a real `MarketSuitabilityResult` while Pillar 2 stays
enforced by default. TWO additional capital-preservation gates remain: the
drawdown kill switch and the range-escape kill switch, both forcing a cooldown.

Suitability is now computed once per completed bar, before the cooldown check,
so every proposal (cooldown, regime, kill, no-fill, and fills) carries it —
the order of the hard gates is unchanged.

## Pillar 3 — Evidence-Based Decision Score: Present (Phase 5)

The level crossing remains the hard **precondition** (analogous to mean
reversion's band touch); a `DecisionScoreEngine` call over three
side-independent Evidence Items then GRADES the crossing:

| Evidence Item | Measurement | Normalization | Weight | Why it is edge |
|---|---|---|---|---|
| Range-bound conviction | EMA(20) fractional slope over the window (`_ema_slope_pct`) | `clamp((0.005 − \|slope\|)/0.005, −1, 1)` | 40 | The grid's edge exists only while ranging; a flat EMA is direct evidence the range holds, a steepening slope is evidence against (a trend). Independent of which level filled. |
| Post-fee spacing margin | `(spacing/price − fee_floor)/fee_floor` | `clamp(r, −1, 1)` | 35 | A cycle's net edge is one spacing minus round-trip fees; the more spacing exceeds the fee floor, the more each harvested cycle actually earns. |
| Volatility adequacy | `ATR/price` | `clamp((r − 0.0005)/(0.004 − 0.0005), −1, 1)` | 25 | The grid harvests oscillation; too little volatility means levels never traverse and no cycles complete. |

Deliberately **NONE references level depth** — depth is the grid's *intended*
convex payoff, rewarded separately by the depth multiplier, never penalised as
"far from center." The three items are side-independent (identical for a buy or
sell crossing on the same bar), so the score can never introduce buy/sell
inventory asymmetry.

`decision_score_threshold` defaults to **0.0**, which is a deliberate,
documented choice for a mechanical strategy: all three items are non-negative
in the range-bound conditions the grid is designed for, so at 0.0 the gate
never suppresses a normal fill — it declines a crossing only when net evidence
has gone negative (a trend the gate let slip, or a dead market). The score's
primary role here is Pillar 5 sizing. An operator may raise the threshold to
demand higher-quality fills; a blocked crossing leaves the level unfilled and
the virtual wallet untouched (no desync), and the grid retries next bar.

**Pre-Phase-5 finding**: entry/exit fired on a single condition (price crossing
a pre-computed grid level). No multi-factor accumulation.

## Pillar 4 — Parameter Adaptation: Partial (unchanged — already second-best of six)

Genuinely adaptive and untouched by Phase 5: grid spacing/range from live ATR
(`grid_range = atr × atr_range_multiplier`, `grid_spacing = grid_range /
grid_count`), spacing floored to a fee-aware minimum, soft recenter at 50%
half-range drift, and depth-aware order sizing. Hardcoded (by design):
`grid_count=10`, `atr_range_multiplier=8.0`, `base_order_size_percent=5`,
`depth_multiplier=1.5`, `max_drawdown_percent=15`, `kill_atr_multiplier=3.0`,
`cooldown_after_kill_hours=2`.

## Pillar 5 — Position Sizing: Decision-Score-weighted × depth multiplier (Phase 5)

`order = initial_capital × base% × depth_multiplier^(depth−1) ×
score_multiplier`, where `score_multiplier = 0.5 + clamp((total − threshold)/
(100 − threshold), 0, 1) ∈ [0.5, 1.5]`. The depth multiplier (convex payoff) is
**unchanged** from the pre-migration formula, per `add-unified-position-sizing`;
the score multiplier is the new, orthogonal factor. Both are recorded in the
proposal's `adaptive_parameters_used`. Capped at `balance ×
_BUY_BALANCE_FRACTION` (never exceeds real funds) exactly as before.

**Pre-Phase-5 finding**: flat base % of *initial* capital × depth multiplier
only; no conviction, exposure, or correlation input.

## Pillar 6 — Trade Management: Partial (unchanged)

Every completed bar re-evaluates regime, both kill switches, and soft-recenter
— real continuous re-evaluation at the grid level. It operates on virtual grid
inventory, not a single position with stop/target/partial mechanics; individual
fills exit only via opposing grid-level crossings. Phase 5 did not change this;
the grid's "management" IS its kill switches plus opposing-level realisation.

## Pillar 7 — Strategy Edge Management: Present (Phase 5 — telemetry promoted to real inputs)

The previously-unused telemetry (`kill_switch_count`, `lifetime_return_pct`,
`lifetime_max_drawdown_pct`) is promoted into a real `StrategyEdgeManager`:

- **Trade outcomes** are recorded at the grid's actual profit/loss events: each
  realised **SELL fill** is a harvested buy→sell cycle → a *win* of roughly one
  grid spacing (gross); each **kill switch** is a *loss* episode. This models
  the grid's true risk profile (many small wins punctuated by occasional kills)
  and lets the manager detect when kills dominate.
- **Decision Scores** are recorded on every fill, feeding the Decision-Score
  trend signal.
- **Classification of a kill** (the audit's explicit ask): a **range-escape
  kill** cites parameter-mismatch evidence (the grid span didn't contain
  realised volatility — a spacing/`atr_range_multiplier` mismatch) → **Category
  B** (adaptable, non-blocking). A **drawdown kill** cites no parameter, so in a
  suitable regime it surfaces as **Category C** (edge gone → stop). Regime
  unsuitability → **Category A**. This is tracked via `state["last_kill_reason"]`.

The fill gate blocks a new crossing only on **Category C** — and even then never
force-closes: virtual inventory is left intact (the grid has its own kill
switches for liquidation). Dedicated tests cover Category-C-blocks-without-
liquidating, range-escape→B, and sell→win recording.

**Modelling caveat (documented, honest)**: the virtual crypto pool is fungible,
so a SELL fill's win is *approximated* as one grid spacing rather than a paired
buy-price P&L; kills use the drawdown / escape magnitude as the loss. This is
faithful for degradation *detection* (win/loss composition trends), not an
exact P&L ledger.

**Pre-Phase-5 finding**: raw numbers tracked but never read back — telemetry
only, fixed-threshold circuit breakers, no win-rate/expectancy self-awareness.

## Pillar 8 — Self-Diagnostics: Full (Phase 5 — the audit's single biggest gap closed)

Both kill switches now call `self._explain()`: the drawdown kill emits a
`KILL_SWITCH_DRAWDOWN` state with the full drawdown/peak/portfolio math and a
failing `check`; the range-escape kill emits `KILL_SWITCH_RANGE_ESCAPE` with
the ATR/distance/kill-distance math. Both carry the resulting edge-management
category. The fill path adds `.check()`s for Decision-Score-clears-threshold,
edge-not-Category-C, and fee-viability, plus `.metric()`s for the score total,
threshold, size multiplier, and edge category. Bar-wait, cooldown, regime, and
the full pre-decision spacing/nearest-level math were already covered.

**Pre-Phase-5 finding**: **both kill switches were `logger.warning`-only — a
kill firing was completely invisible to the diagnostics UI**, the single
largest Pillar 8 gap found in the whole audit. Closed.

## Pillar 9 — Performance Expectations: DOCUMENTED (Phase 5)

- **Trade frequency**: highly regime- and timeframe-dependent. In a clean range
  it should cycle steadily (multiple fills/day at 1m); in a trend it should go
  near-dormant (kill switches + regime gate). At 1-minute resolution on BTC the
  realised ATR is tiny relative to the fee-floored spacing, so range-escape
  kills dominate and *net order flow is very low* — see Backtesting.
- **Holding time**: `short` per fill (inventory is meant to be realised within
  a few oscillations); no fixed time stop.
- **Win rate**: individual cycles should win often (>50%) in a genuine range;
  the tail risk is the kill loss, not per-cycle losses.
- **Profit factor**: expected only modestly above 1.0 in-range; the strategy is
  a small-edge, high-turnover market-making posture, not a convex trend bet.
- **Drawdown target**: hard-bounded by `max_drawdown_percent` (15%) and the
  range-escape kill; realised DD should stay well inside that.
- **Buy-and-hold**: the grid is NOT designed to beat buy-and-hold in a bull
  trend (it sells into strength and pauses in trends); its purpose is
  positive-carry capital preservation in sideways markets.

**Pre-Phase-5 finding**: undocumented.

## Pillar 10 — Strategy Proposal Interface: Present (Phase 5)

Every return is now a `StrategyProposal` via the Standalone Adapter, except the
two pre-suitability structural waits ("starting new bar", "bar in progress"),
which remain plain `TradeSignal` holds (mirroring mean_reversion's warmup — no
suitability computed yet). Intent mapping (tasks.md 5.6):

- **No level crossed** → `HOLD` + `HOLD_POSITION` (grid live, inventory held).
- **Buy fill** → `BUY` + `ADD_TO_POSITION` (grid entries are *incremental*
  inventory adds, not fresh `OPEN_POSITION`s like Phases 1–4).
- **Sell fill** → `SELL` + `REDUCE_POSITION`, or `CLOSE_POSITION` when the sell
  empties the virtual crypto pool ("remaining depth").
- **Cooldown / regime / kill / insufficient-funds / fee-reject / score-gate /
  below-min** → `NO_TRADE` + `NO_ACTION`.

`validity.valid_until = generated_at + bar_interval_seconds` (the grid's
per-bar re-evaluation interval). Objective assumptions on fills: regime remains
range-bound, price within the grid range, spacing still covers fees. The
per-bar single-order invariant is exactly why ONE proposal per evaluation
suffices — no multiple concurrent proposals or staged intents are needed.

**Pre-Phase-5 finding**: returned `TradeSignal` per grid-level fill; no
proposal concept.

## Backtesting

**Methodology**: BTCUSDT, 1-minute bars, `$10,000` starting balance, default
strategy parameters (none tuned), 0.1%/side fee, no spread/slippage modeled.
Same three ~6-week windows as Phases 1–4. "Before" = pre-migration
adaptive_grid at commit `ffb9a2c`; "after" = this migration. Both on the
corrected backtest engine.

**Before vs. after (full pre/post across all three windows):**

| Window | | Return % | Trades | Max DD % | Fees | Buy-and-hold % |
|---|---|---|---|---|---|---|
| Bull | before | −0.26% | 0 | 0.41% | $0.50 | +31.84% |
| Bull | after  | **+0.00%** | 0 | 0.00% | $0.00 | +31.84% |
| Bear | before | −0.13% | 0 | 3.97% | $23.50 | −13.07% |
| Bear | after  | **+0.20%** | 0 | 1.00% | $8.09 | −13.07% |
| Chop | before | +0.29% | 0 | 1.43% | $6.25 | −13.59% |
| Chop | after  | **+0.10%** | 0 | 0.34% | $1.55 | −13.59% |

(`Trades` is the engine's closed-round-trip count; the grid's incremental
buys/sells rarely register as closed round-trips at this granularity, so the
column reads 0 in every run even though small fee-paying order flow occurred —
see the non-zero fees. Both before and after were re-run on the corrected
harness; the "after" also carries a harness fix — see the note below.)

**Interpretation, honestly**: at 1-minute resolution BTC's realised ATR is tiny
relative to the fee-floored spacing, so the grid is dominated by range-escape
kills (515–532 per window, unchanged) and never accumulates meaningful net
inventory — this is a structurally **near-dormant** strategy at 1m, before and
after. The migration did NOT change that regime, nor was it meant to.

What it DID change is uniformly in the direction of **more selectivity**:

- The score gate (declining a crossing whose net evidence has gone negative,
  even at the default 0.0 threshold) and the score-weighted sizing (0.5–1.5×)
  cut residual order flow. Fees fell in every window (bull $0.50→$0, bear
  $23.50→$8.09, chop $6.25→$1.55) and max drawdown fell in every window (bear
  3.97%→1.00%, chop 1.43%→0.34%, bull 0.41%→0.00%).
- Return moved from mixed (−0.26 / −0.13 / +0.29) to uniformly non-negative
  (+0.00 / +0.20 / +0.10). In the **bull** window the grid now abstains
  completely (0 orders, $0 fees) — correct behaviour for a range strategy in a
  trend, and a strict improvement over the pre-migration small fee bleed.

This is NOT a demonstration of positive grid edge — at 1m the strategy barely
acts, so these are capital-preservation deltas (less fee bleed, less drawdown),
not evidence of profitable cycling. As with mean_reversion, a definitive edge
ruling awaits `add-strategy-validation-tooling`'s walk-forward validation on the
timeframe/volatility where a grid's cycles would actually complete; the Decision
Score threshold/weights are uncalibrated first-pass values (deliberately not
tuned to these windows). The migration's value here is governance, transparency
(both kill switches now explained), edge self-awareness (Pillar 7), and the
removal of unconditional fee bleed — not a re-tune of the grid's aggressiveness.

**Backtest-harness note**: while running these, the per-bar
`StrategyProposal.explanation = self._explain(bot.id).to_dict()` serialisation
exposed a pre-existing harness inefficiency — `BacktestEngine` calls the
strategy executor directly (bypassing `_execute_strategy`), so the per-bot
explanation builder was never reset per bar and accumulated the whole run's
checks, making any framework-migrated strategy's replay O(n²). Fixed by
resetting the builder before each executor call, mirroring the live path
(`backend/app/backtesting/engine.py`). This is observability-only and changes
run speed, never results (verified: identical metrics before/after the fix,
`test_backtest_determinism` still green). It benefits every migrated strategy's
backtests, not just the grid's.

## Certification Checklist

- [x] Deterministic — `test_decision_score_deterministic`.
- [x] Immutable — frozen `StrategyProposal` dataclass.
- [x] Assumptions documented — objective, falsifiable, traced to Pillar 1.
- [x] Expiration defined — `validity.valid_until` set and tested.
- [x] Execution intent consistent with direction — structurally enforced;
      `test_all_grid_proposals_have_valid_intent_pairings`, plus the
      ADD/REDUCE/CLOSE/HOLD grid-specific mapping tests.
- [x] Evidence measurable — Pillar 3 table complete; three-factor,
      depth-independent; single-factor insufficiency verified.
- [x] Explanation reproducible — determinism test; both kill switches now
      explained (Pillar 8).
- [x] No subjective information — every factor a deterministic Measurement +
      Normalization pair.
- [x] Expected edge sourced correctly — always `None`, mechanically enforced.
- [x] Before/after backtest comparison recorded — see Backtesting section
      (full pre/post across bull/bear/chop: fees and drawdown fell in every
      window, return moved to uniformly non-negative, grid fully abstains in the
      bull trend — capital-preservation gains from selectivity, positive edge
      not demonstrated at 1m).
- [x] All existing tests pass (full suite) + 28 new tests
      (`test_adaptive_grid_framework_migration.py`).

## Self-Audit (critical review against trading theory, not code)

1. Behaves like a disciplined range market-maker: pre-positions liquidity,
   accumulates convexly on dips, realises into strength, and *admits failure
   fast* via kill switches when the range breaks.
2. Should refuse to trade / should stop: a trending market (implemented —
   Pillar 2 gate + range-escape kill classified Category B for adaptation); a
   dead market with no cycles (implemented — Volatility adequacy evidence goes
   negative); a fee-uncoverable spacing (implemented — min-spacing floor + fee
   viability check); a sustained capital bleed (implemented — drawdown kill →
   Category C stop). NOT implemented: order-book depth / real spread refusal (no
   such data in backtest/most live paths — documented).
3. Honest limitation: the Pillar 7 win/loss modelling (sell = +spacing, kill =
   −magnitude) is an approximation of a fungible virtual pool, sufficient for
   degradation detection, not an exact P&L ledger. And at 1-minute resolution
   the grid is structurally near-dormant (kills dominate) — the migration
   governs and explains that behaviour honestly rather than masking it with a
   re-tune.
