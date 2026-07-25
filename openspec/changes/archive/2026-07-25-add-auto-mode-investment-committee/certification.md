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

## §5 Runtime integration (Phase 5) — now implemented

The runtime integration that Phase 4 recorded as deferred is now built (Phase
5), per the approved single-bot decision: **Auto is one bot, not a portfolio of
bots.** Inside the existing Auto bot's loop, `_strategy_auto` delegates (behind
`is_committee_enabled`, OFF by default) to `_strategy_auto_committee`, which:

- **Flat** → evaluates all Alpha strategies (`_COMMITTEE_ALPHA_STRATEGIES` =
  trend_following, mean_reversion, volatility_breakout, dip_recovery,
  adaptive_grid — `dca_accumulator` excluded), collects each strategy's
  `StrategyProposal` (surfaced on the returned `TradeSignal` via
  `_source_proposal`, `compare=False` so the Phase 2 byte-identical guarantee
  holds), runs `resolve_portfolio_constraints` + `run_committee`, and executes
  the top-ranked selection through the **unchanged** `_execute_trade`.
- **In a position** → dispatches only the `owning_strategy` (its own exit
  rules), reusing the existing `auto-mode-position-ownership` rule; the winner's
  reason is stamped `[Auto:<strategy>|committee]` so `_resolve_owning_strategy`
  records the correct owner on execution.

No cross-bot scheduler, no portfolio-level runtime, no `StrategyProposal`-
contract change, no per-strategy edits.

**Runtime-integration checks (`tests/test_auto_committee_runtime.py`, 13 tests):**

| Property | Check | Result |
|---|---|---|
| Proposal surfaced without contract change | `to_trade_signal` carries `_source_proposal`; equality unaffected (`compare=False`) | PASS |
| Winner = highest-ranked, amount scaled, owner-stamped | `_committee_select` picks top rank, scales to `allocated_size`, stamps `[Auto:…|committee]` | PASS |
| One failing strategy doesn't sink the cycle | erroring strategy skipped, others still compete | PASS |
| In a position → only owner dispatched | `_committee_select` never called; owner's exit returned | PASS |
| Flat + no selection → hold | committee returns nothing → HOLD | PASS |
| Flag on → delegates; flag off → unchanged | delegation asserted; flag OFF by default (1371-test suite runs on the unchanged path) | PASS |
| `dca_accumulator` never a candidate; `dip_recovery` is | `_COMMITTEE_ALPHA_STRATEGIES` membership | PASS |

**Single-bot scope (honest):** a single-position Auto bot realises **one**
position per cycle — the committee's top selection. Runtime *multi-execution*
(multiple simultaneous positions) needs a multi-position/multi-bot model and is
**not** built; the multi-execution *capability* remains certified (Gate item 9).
The multi-strategy standalone-vs-Auto backtest still requires that multi-bot
model and remains future validation work — but the runtime integration itself
is complete, so nothing about Auto's decision path is left unimplemented.

## Conclusion

The Investment Committee is **certified for correctness and determinism** under
all supported scenarios, **and the runtime integration is complete**: an Auto
bot decides via the committee end-to-end, behind `is_committee_enabled` (OFF by
default), with position ownership preserved and execution byte-identical to the
Standalone path in the degenerate one-strategy case. Enabling Auto for a real
bot remains an operator decision (flip the flag) after whatever live review the
operator requires; the code path is finished, not deferred.
