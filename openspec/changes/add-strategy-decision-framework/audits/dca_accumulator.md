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

## Pillar 1 — Theory: UNDOCUMENTED (partial)

Docstring (1119-1149) describes mechanism and a coarse regime-filter
rationale ("protects capital... during strong downtrends," 1132-1133) but
never states *why* DCA has edge, when it should underperform lump-sum
investing, or its core assumption (a long-run uptrend). No inefficiency is
named. **Authoring real theory is a certification task, not done here.**

## Pillar 2 — Market Suitability: Full

Real, enforced refusal-to-trade gate (1179-1216): computes regime via
`_detect_market_regime`, blocks buying when `trend_state` isn't in
`allowed_regimes` (default `["trend_up","trend_flat"]`, line 1175).
`regime_filter_enabled` defaults `True` but is user-disableable.

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

## Pillar 6 — Trade Management: Not applicable (see note above)

No post-entry logic exists because there is no position lifecycle to
manage — DCA only ever buys. Certification must decide whether this
remains correct.

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
