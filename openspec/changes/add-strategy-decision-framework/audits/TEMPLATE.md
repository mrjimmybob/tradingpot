# Strategy Audit Document — Template

Phase 0.8 deliverable: the formal template `audits/README.md` references.
The six existing `audits/*.md` files (excluding this one) are CURRENT-STATE
snapshots produced before any remediation — they document what exists
today, with `UNDOCUMENTED`/`Not present` where a pillar is missing. THIS
template is what each of those files becomes DURING that strategy's own
certification phase (Phase 1-6 in `tasks.md`): every placeholder below
filled in, every `UNDOCUMENTED` resolved, ready for the certification
review `spec.md`'s "Strategy Certification Gate" requirement gates on.

Copy this file to `audits/<strategy_name>.md` (replacing the existing
current-state audit) when that strategy's certification phase begins. Do
not delete the current-state findings wholesale — carry forward any
still-accurate file:line evidence, and mark clearly what changed.

---

# Strategy Audit: `<strategy_name>`

`<function_name>`, `backend/app/services/trading_engine.py:<start>-<end>`.

**Certification status**: ☐ Draft ☐ Under review ☐ **Certified** (date: ____)
**Certified against**: `add-strategy-decision-framework` 10-pillar standard, revision ____ (Architecture Freeze)

## Pillar 1 — Theory

- **Market Inefficiency**: _What specific, nameable inefficiency does this
  strategy exploit? (e.g. overreaction/panic-selling mean reversion,
  momentum continuation, liquidity provision in range-bound markets)_
- **Why The Edge Should Exist**: _The causal mechanism — WHY does this
  inefficiency persist rather than being arbitraged away? Cite market
  microstructure, behavioral finance, or structural reasons — not just "it
  has worked historically."_
- **Evidence**: _What supports this thesis — prior backtest results (cite
  specific runs), academic/practitioner literature, or a clearly-labeled
  hypothesis awaiting `add-strategy-validation-tooling` validation?_
- **Conditions For Success**: _The market regime(s)/conditions under which
  this edge should be strongest. Must map onto Pillar 2's `allowed_regimes`
  — if they don't agree, one of the two is wrong._
- **Conditions For Failure**: _The market regime(s)/conditions under which
  this strategy should be EXPECTED to lose money or sit idle. A strategy
  with no documented failure conditions has not actually been thought
  through._
- **Assumptions**: _The objective, falsifiable conditions this strategy's
  edge depends on remaining true — this is the source list for Pillar 10's
  `StrategyProposal.assumptions` (see Pillar 10 below). Subjective language
  ("feels like," "generally") is not acceptable here._

## Pillar 2 — Market Suitability

- **Declared `allowed_regimes`**: `[...]` (the exact list, from
  `_get_strategy_capabilities()` and/or this strategy's own standalone
  gate — both SHOULD agree; if they don't, that is a certification finding
  to resolve, not ship as-is).
- **Enforcement citation**: `file:line` where
  `MarketSuitabilityGate.gate(...)` (Phase 0.1) is called and its `False`
  result actually blocks a new entry — not merely computed/logged.
- **Test proving the gate blocks entries**: `tests/<file>.py::<test_name>`.

## Pillar 3 — Evidence-Based Decision Score

**Threshold**: `____` (0-100 scale, this strategy's configured minimum).

| Evidence Item | Measurement | Normalization | Weight | Reason it contributes to edge (cross-ref Pillar 1) |
|---|---|---|---|---|
| _e.g. Trend strength_ | _e.g. EMA(20) slope over last 10 bars, %_ | _e.g. clamp(slope_pct / 2.0, -1, 1)_ | _e.g. 20_ | _cite Pillar 1 section_ |
| ... | ... | ... | ... | ... |

_Weights SHOULD sum to <= 100 (see `decision_score.py`'s
`DecisionScoreEngine` docstring). Every row must be expressible as a
deterministic Python callable pair — if it isn't, it does not belong in
this table (see `spec.md`'s "Subjective or unmeasurable factors are
rejected" scenario)._

- **Determinism test**: `tests/<file>.py::<test_name>` (same inputs -> same
  score, same Evidence Report).
- **Example Evidence Report** (from a real or synthetic trade): _paste
  `EvidenceReport.render()` output here._

## Pillar 4 — Parameter Adaptation

| Parameter | Classification | Formula (if ADAPTIVE) / Reason (if FIXED) |
|---|---|---|
| _e.g. stop_atr_multiplier_ | ADAPTIVE | `AdaptiveParameterResolver.atr_percentile_scaled_multiplier(...)`, see `file:line` |
| _e.g. cooldown_seconds_ | FIXED | _why a fixed value is correct here_ |
| ... | ... | ... |

_No parameter may be silently hardcoded post-certification — every row is
mandatory, not just the interesting ones._

## Pillar 5 — Decision-Score-Weighted Position Sizing

- **Formula**: _cite the shared sizing function once `add-unified-
  position-sizing` lands; until then, describe this strategy's current
  formula and confirm it at least consults `PortfolioRiskService`/
  `StrategyCapacityService`._
- **Decision Score coupling**: _how `decision_score.total`/`.threshold`
  scale the sized amount (see `add-unified-position-sizing/design.md`'s
  Decisions section for the required shape)._
- **Test proving a marginal-score trade sizes smaller**: `tests/<file>.py::<test_name>`.

## Pillar 6 — Continuous Trade Management

Which of the four re-evaluation checks does this strategy implement, and how?

| Check | Implemented? | Citation |
|---|---|---|
| (a) Thesis/suitability still passing | ☐ | `file:line` or `TradeManagementMonitor` usage |
| (b) Volatility changed materially | ☐ | |
| (c) Stop should tighten | ☐ | |
| (d) Partial profit should be taken | ☐ | |

_A strategy that structurally has no open position to manage (see
`dca_accumulator`'s audit) documents WHY here instead of leaving this
blank._

## Pillar 7 — Strategy Edge Management

- **`StrategyEdgeManager` configuration**: `min_sample_size=____`,
  `outcome_window=____`, `expectancy_floor=____`, `win_rate_floor=____`
  (cite the certification-phase backtest evidence that justified these
  values — NOT the shared module's generic defaults, which are placeholders
  only).
- **Category A condition for this strategy**: _what regime-mismatch
  evidence (from `MarketSuitabilityGate`) triggers Category A specifically
  for this strategy's thesis._
- **Category B condition for this strategy**: _which specific adaptive
  parameter(s) (from Pillar 4's table) a Category B classification is
  allowed to adjust, and via which `AdaptiveParameterResolver` call._
- **Category C threshold**: _what persistence/evidence standard must be met
  before concluding the edge is gone, per this strategy's Theory document._
- **Tests**: dedicated Category A/B/C classification tests (synthetic
  episodes) — `tests/<file>.py::<test_names>`.

## Pillar 8 — Self-Diagnostics

- **`.check()`/`.metric()` coverage**: every decision point (entry, sizing,
  stop, exit, kill-switches) has a corresponding call — list any known
  gaps found during the current-state audit and confirm each is closed.
- **`Order.decision_explanation` populated**: confirm a real (or test)
  trade produces a non-null `Order.decision_explanation` (Phase 0.6
  persistence) including the Evidence Report and, once Pillar 7 is wired,
  the active edge-management category (see `explanation_persistence.py`'s
  well-known metric keys: record via
  `self._explain(bot.id).metric("evidence_report", ...)` and
  `.metric("edge_status_category", status.category.value)`).

## Pillar 9 — Performance Expectations

- **Expected trade frequency**: ____
- **Expected holding time**: ____
- **Expected win rate**: ____
- **Expected profit factor**: ____
- **Expected drawdown**: ____
- **Typical vs. worst-case market conditions**: ____
- **Why out/under-perform buy-and-hold, and under what conditions**: ____
- **Real walk-forward result vs. this documented expectation** (once
  `add-strategy-validation-tooling` is available): ____ (a material
  contradiction is itself a finding, not silently accepted — see `spec.md`).

## Pillar 10 — Strategy Proposal Interface

- **`execution_intent` mapping for this strategy**: _entry ->
  `OPEN_POSITION`/`ADD_TO_POSITION`, exit ->
  `REDUCE_POSITION`/`CLOSE_POSITION`, no-op ->
  `NO_ACTION`/`HOLD_POSITION` — be explicit about which of this strategy's
  own states map to which, especially any ambiguous case (see
  `tasks.md`'s Phase 5.6/6.7 notes for adaptive_grid/dca_accumulator's own
  resolved ambiguities as examples)._
- **`validity.valid_until` interval**: ____ (tied to this strategy's own
  evaluation cadence/timeframe — cite the reasoning).
- **`assumptions` list**: _pull directly from Pillar 1's Assumptions,
  phrased as the objective/falsifiable conditions `StrategyProposal`
  requires._
- **`expected_edge_estimate`**: `None` unless a validated
  `add-strategy-validation-tooling` result exists for this exact
  configuration — never self-computed (see `proposal.py`'s
  `EdgeEstimate.source` enforcement).
- **Behavior-preservation test**: `tests/<file>.py::<test_name>` proving
  the Standalone Adapter + existing execution pipeline produces the same
  outcome the pre-migration `TradeSignal` path did.

## Certification Checklist

Per `spec.md`'s "Strategy Certification Gate" (nine `StrategyProposal`
properties) plus the pillar-level checks above:

- [ ] Deterministic — same inputs always produce the same proposal.
- [ ] Immutable — enforced at runtime (frozen dataclass), not only by review.
- [ ] Assumptions documented — objective, falsifiable, traced to Pillar 1.
- [ ] Expiration defined — `validity.valid_until` set and tested.
- [ ] Execution intent consistent with direction — enforced by
      `StrategyProposal.__post_init__`'s pairing table, verify no strategy
      code works around it.
- [ ] Evidence measurable — every Evidence Item has Measurement +
      Normalization + Weight + Reason (Pillar 3 table above complete).
- [ ] Explanation reproducible — Pillar 8's Evidence Report renders
      identically for the same inputs.
- [ ] No subjective information — Pillar 3 table contains no factor that
      couldn't be expressed as a deterministic callable pair.
- [ ] Expected edge sourced correctly — `None` or
      `EdgeEstimate.source == "add-strategy-validation-tooling"`, never
      self-invented.
- [ ] Before/after backtest comparison recorded (trade count, and
      confirmation it's not zero in reasonable conditions).
- [ ] All existing tests pass; new tests cover every pillar above.
