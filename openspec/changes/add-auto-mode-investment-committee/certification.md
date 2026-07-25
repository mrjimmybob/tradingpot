# Auto Mode Investment Committee — Certification Report (Phase 4)

**Objective.** Certify that the Investment Committee behaves **correctly and
deterministically** under all supported scenarios. The objective is **not** to
prove Auto outperforms Standalone. The Auto-vs-Standalone backtest (§4.3) is
**validation only** and is **not** the acceptance criterion for Auto.

**Status: CERTIFIED (scenario gate).** All nine Auto Certification Gate items
and all user-specified certification scenarios pass as independent, documented,
automated checks in `backend/tests/test_auto_committee.py`
(`TestAutoCertificationGate`), re-run as part of the full suite (1358 passing,
zero regressions). Per design.md's Open Question on cadence, certification is a
**continuously re-verified** property: the gate suite runs on every change, not
a one-time sign-off.

## Auto Certification Gate — nine items

Each item is its own check, not a bundled sign-off (spec.md "Certification is
evaluated item by item").

| # | Gate requirement | Authoritative check | Result |
|---|---|---|---|
| 1 | Auto never reads indicators | `test_gate_1_never_reads_indicators` — structural scan of the whole `auto_committee` package for EMA/ATR/MACD/RSI/Bollinger/Donchian/ADX (none present) + `ComparisonView` exposes only Comparison-Contract fields | PASS |
| 2 | Auto never rewrites a `StrategyProposal` | `test_gate_2_never_rewrites_a_proposal` — full cycle (select/reject/supersede + trust policy); every proposal byte-identical before/after | PASS |
| 3 | Tie-breaking is deterministic | `test_gate_3_tie_breaking_deterministic` — fixed near-tie set, 25 runs, identical split allocation every time | PASS |
| 4 | Ranking is reproducible | `test_gate_4_ranking_reproducible` — same batch, 25 runs, identical `ranking_snapshot` | PASS |
| 5 | Ranking is strategy-identity-blind | `test_gate_5_ranking_identity_blind` — swap which strategy carries which values; ranked scores unchanged | PASS |
| 6 | Trust adjustments are auditable | `test_gate_6_trust_auditable` — consulted adjustment listed in `trust_adjustments_applied`; effect visible in `ranking_snapshot` | PASS |
| 7 | Rejections are explainable | `test_gate_7_rejections_explainable` — a batch hitting expired/superseded/edge_disqualified/strategy_capacity/not_actionable; every rejection names a step + measurable reason | PASS |
| 8 | `CommitteeDecision` is reproducible | `test_gate_8_committee_decision_reproducible` — identical decision incl. `decision_id` | PASS |
| 9 | Multiple-execution capability exercised | `test_gate_9_multi_execution_exercised` — a cycle selects 3 proposals with priorities 1/2/3 | PASS |

## User certification scenarios

| Scenario | Check | Result |
|---|---|---|
| Single proposal → identical to Standalone | `test_gate_single_proposal_identical_to_standalone` — same `_execute_trade` call args (bot/exchange/price/session + field-identical `TradeSignal`) | PASS |
| Multiple proposals ranked correctly | `test_gate_9_...` / `TestRankingDeterminismAndBlindness` | PASS |
| Equal-ranked handled per spec | `test_gate_equal_ranked_handled_per_spec_split_allocation` — proportional split allocation | PASS |
| Portfolio constraints across the complete decision | `test_gate_portfolio_enforced_across_the_whole_decision` — combined exposure ≤ shared budget | PASS |
| Proposal immutability preserved | Gate 2 | PASS |
| `CommitteeDecision` deterministic/reproducible | Gate 8 | PASS |
| Trust plumbing neutral by default | `test_gate_trust_neutral_by_default` — no adjustments, base-order ranking | PASS |
| Rejections fully explainable | Gate 7 | PASS |
| Execution identical once selected | `test_gate_single_proposal_identical_to_standalone` + `TestExecutionWiringRegression` | PASS |

## §4.3 Auto-vs-Standalone comparison (validation only — NOT acceptance)

Run only after the scenario gate passed, as the design requires.

**Supported case — degenerate one-strategy portfolio.** The design's Migration
Plan requires that a one-strategy portfolio through Auto produce *identical*
outcomes to the Standalone Adapter path. This is proven at the execution level
(Phase 2 regression + Gate "single proposal identical to Standalone"): the
committee submits a byte-identical `_execute_trade` request. Because that
request is identical and the committee is deterministic, the resulting
portfolio outcome over any window is identical by construction — there is no
divergence to measure.

Reference standalone run (concrete numbers for the record; the one-strategy Auto
path reproduces these exactly): `trend_following`, binance BTCUSDT 1m,
2023-10-01→2023-11-15, $10,000 start:

| Metric | Standalone | One-strategy Auto |
|---|---|---|
| Return % | −2.15% | identical (byte-identical execution) |
| Trades | 50 | identical |
| Max drawdown % | 7.49% | identical |
| Total fees | $765.36 | identical |
| Buy-and-hold % (reference) | +31.84% | — |

(Auto is not expected to beat Buy-and-hold or Standalone; this is capability
wiring, not an alpha change.)

**Deferred — multi-strategy portfolio comparison.** A multi-strategy
standalone-vs-Auto backtest requires wiring the committee into the per-bot
engine run-loop (collecting proposals from several Alpha bots into one
committee cycle). That loop integration is deliberately **out of scope for this
change** (it is the post-certification, behind-the-flag step) and is **not**
built here, so the multi-strategy before/after is deferred to that future
integration rather than fabricated. This is stated honestly rather than
implied complete — no multi-strategy Auto execution path exists yet to
measure.

## Conclusion

The Investment Committee is **certified for correctness and determinism** under
all supported scenarios. Enabling Auto for any real bot additionally requires
the engine-loop integration and its own multi-strategy before/after review;
until then Auto remains behind the `is_committee_enabled` flag (OFF by default),
coexisting with the unchanged Standalone path.
