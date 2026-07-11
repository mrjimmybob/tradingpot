# Strategy Audit: volatility_breakout

`_strategy_volatility_breakout`, `backend/app/services/trading_engine.py:3273-3963`.
Coverage score: **5/14**. Market suitability is **computed but silently
discarded** — the smallest-lift fix of the six for pillar 2 (the value
already exists; it just needs to actually gate the decision, as a real,
validated behavior change — see below).

## Pillar 1 — Theory: UNDOCUMENTED (partial)

Docstring (3280-3330) describes mechanics well ("trades volatility
expansion after compression," ATR stop "can ONLY TIGHTEN") and notes one
assumption ("no volume data," 3794 comment), but never names the market
inefficiency being exploited (e.g. stop-run/liquidity-vacuum, momentum
ignition) or states explicit failure conditions (choppy fake-outs,
low-liquidity gaps). **Authoring real theory is a certification task.**

## Pillar 2 — Market Suitability: COMPUTED, NOT ENFORCED (confirmed, previously discovered)

`_detect_market_regime_bar_based` is called and `regime_allows_entry` is
computed (line 3534), but the actual entry condition is purely
`is_breakout = compression_satisfied and last_bar_close > upper_band`
(3775) — `regime_allows_entry` never appears in that condition or any other
gate. This was discovered (not newly found) during
`fix-regime-detection-consistency`: a prior change deliberately removed
the regime veto as a hard gate (in-code comment, 3684-3692: "the
compression-then-breakout sequence already encodes the volatility
thesis... the separate volatility veto was redundant"). **Re-enabling
enforcement here is explicitly the kind of validated behavior change that
change's design.md deferred to this framework's certification process** —
it must be backtested before/after, not silently flipped on.

## Pillar 3 — Evidence-Based Decision Score: Not present

Entry fires on a single boolean AND of two conditions (3775) — no
accumulated multi-factor score.

## Pillar 4 — Parameter Adaptation: Not present (stop distance only)

Hardcoded: `bb_period=20`, `bb_std=2.0`, `compression_percentile=20`,
`atr_threshold_multiplier=0.8`, `min_compression_bars=5`,
`atr_stop_multiplier=2.0`, `cooldown_hours=24`, `failed_breakout_bars=3`
(3332-3345). Only the stop distance scales with ATR; every period and
threshold that determines *when* the strategy considers itself
"compressed" or "breaking out" is fixed.

## Pillar 5 — Position Sizing: Risk-scaled (partial credit)

Identical convention to trend_following: `risk_amount = balance *
risk_percent`, `position_coins = risk_amount / (atr * atr_stop_mult)`
(3866-3871). No Decision Score/exposure/drawdown input.

## Pillar 6 — Trade Management: Partial

Two exits: failed-breakout (`bars_since_entry <= failed_breakout_bars`
and `last_bar_close < upper_band`, 3608-3636) and a monotonic-tighten-only
trailing stop (3638-3661). No take-profit, no time-based exit — less
complete than mean_reversion/trend_following/dip_recovery's confluence.

## Pillar 7 — Strategy Edge Management: Not present

No performance fields in the strategy's own state dict (3355-3371); the
only win-rate/profit-factor code (`_score_strategy`) is Auto Mode's
cross-strategy selector, not this strategy's self-awareness.

## Pillar 8 — Self-Diagnostics: Partial

Warm-up, exit gates, and entry gates (compression armed, breakout above
band) all explained (3420-3425, 3562-3594, 3795-3843). Gap: the
position-sizing decision and fee-viability check (3860-3923) are never
surfaced via `ExplanationBuilder` — only via `logger.info` (3925), which
isn't visible in the operator-facing diagnostics UI.

## Pillar 9 — Performance Expectations: Partial (frequency only)

Docstring states expected frequency: "Enters rarely (default: few trades
per month)" and "Complements trend_following (often enters earlier)"
(3284-3285). No win rate, profit factor, holding time, drawdown, or
buy-and-hold comparison anywhere. **Remaining expectations to be authored
at certification.**

## Pillar 10 — Strategy Proposal Interface: Not present

Returns `TradeSignal` directly (e.g. `trading_engine.py:3920`,
`return TradeSignal(action="buy", ...)`) — no `StrategyProposal` concept
exists yet. Notably, this strategy already computes and discards a
regime value (Pillar 2) and never populates `TradeSignal.score` (Pillar
3) — the same "compute it, then throw it away instead of using it"
pattern found twice already in this strategy would need to be avoided a
third time when wiring its future `StrategyProposal.market_suitability`
and `.decision_score` fields from real, enforced computations.
