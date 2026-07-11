## 1. Implementation
- [x] 1.1 Add volatility-direction computation to `volatility_breakout`'s own
      bar-aggregation state (reuse the bar-based detector's rate-of-change +
      3-bar persistence logic, `trading_engine.py:5207-5223` / `:5249-5291`).
      Implemented by calling `_detect_market_regime_bar_based` directly
      against `state["bars"]`, with a new persisted `state["regime_state"]`
      key for cross-call hysteresis.
- [x] 1.2 Replace the `volatility_state -> {"low":"contracting",...}` label
      remap with a real direction check in the regime gate. Now maps
      `volatility_direction` ("expanding"/"contracting"/"stable") instead of
      `volatility_state` ("low"/"medium"/"high").
- [x] 1.3 Add unit tests asserting the fix: high-but-flat volatility reports
      direction "stable" not "expanding"; volatility that just started
      rising after compression reports "expanding". Added both a
      detector-level test (`_detect_market_regime_bar_based` called
      directly with synthetic bars) and a strategy-level integration test
      (`_strategy_volatility_breakout` persists the correct
      `state["regime_state"]`). `backend/tests/test_strategy_activity.py`,
      `TestVolatilityDirectionRegimeGate` (4 tests).
- [x] 1.4 Re-ran `python -m app.backtesting.run --strategy
      volatility_breakout` across bull (2023), bear (2022), and chop
      (2024 H1) windows, before and after the fix (isolated via a temporary
      revert-and-restore of just this change, not the earlier session's
      other fixes). **Result: byte-identical output in all 3 windows** —
      see the discovery in 1.3.5 below for why that's the *expected*,
      correct outcome here, not a null result.
- [x] 1.5 Updated docstrings/comments in `_strategy_volatility_breakout`
      (class docstring, `regime_filter_enabled`/`allowed_regimes` param
      docs, the REGIME section's own comments) and `_get_strategy_capabilities`
      (`:6099-6115` area) so the `"volatility_expanding"` label's meaning -
      and the gate's actual (non-)effect on trading - is documented
      correctly in both places.

### Discovery made during implementation (not a separate task, but material to how 1.3/1.4 read)
`regime_allows_entry`/`volatility_regime_name` turned out to be **computed
but never consulted by the entry decision** in `_strategy_volatility_breakout`
- `is_breakout = compression_satisfied and last_bar_close > upper_band`
(`trading_engine.py:3752` at time of writing) is the only behavioral entry
gate. A prior, already-in-place change had deliberately removed the regime
veto as a hard gate (in-code comment at `:3684-3692`: "the compression-then-
breakout sequence already encodes the volatility thesis... the separate
volatility veto was redundant"), but the class docstring still claimed
"Pauses during unfavorable regimes... to prevent entries" - that claim was
stale/false and has been corrected in 1.5. This means:
- This fix corrects what gets **computed and reported** (the diagnostic
  value logged and surfaced via `_explain`) for internal consistency with
  how `_strategy_auto`/`_is_strategy_eligible` already computed it
  correctly - it does not, and was never going to, change any trade
  decision for the standalone strategy path, which is exactly what 1.4's
  identical before/after backtests confirm.
- Re-enabling `regime_allows_entry` as an actual hard entry gate (making the
  old docstring claim true again) is a real behavior change, out of scope
  here, and would need its own validation cycle via
  `add-strategy-validation-tooling` before being trusted - not done as part
  of this change.
