## Context
`models/position.py` already defines `PositionSide.SHORT` and
`calculate_unrealized_pnl` already has a short-side branch — the data model
anticipated this, but nothing produces a short position today. Going short
introduces risks that don't exist in the current spot-only system: margin
calls/liquidation, borrow cost, and (for perpetuals) funding-rate exposure
that can be negative (cost to hold) or positive (paid to hold) depending on
market skew.

## Goals / Non-Goals
- Goals:
  - Backtest engine can simulate at least one short-capable strategy with
    realistic borrow/funding cost, validated out-of-sample before being
    considered for live trading.
  - Live execution path can place and manage a short position on an
    exchange that supports it, gated behind explicit configuration (no bot
    goes short by accident).
- Non-Goals:
  - Full margin/leverage trading generally (e.g. leveraged longs) — scope is
    strictly "the strategy can also go short," not "the bot can use
    leverage."
  - Multi-strategy ensemble long+short blending — one strategy at a time,
    consistent with `_strategy_auto`'s current one-strategy-at-a-time model.
  - Cross-margin / portfolio-margin accounting across multiple short
    positions — start with isolated-margin-style, one position at a time,
    matching the existing single-position-per-bot assumption throughout the
    codebase (`BacktestPortfolio` is single-position; `_get_bot_positions`
    usage throughout `trading_engine.py` assumes this too).

## Decisions
- Decision: backtest short support first, live second, and require a
  walk-forward-validated positive out-of-sample expectancy (via
  `add-strategy-validation-tooling`) before any live wiring work starts.
  - Why: shorting is the single highest-risk addition in this roadmap
    (liquidation risk is new); it must clear a higher evidence bar than any
    other item here before touching real capital.
- Decision: start with `trend_following`'s mirror-image short rather than a
  new strategy, since its long-side logic (EMA cross, ATR trailing stop) is
  already well-understood and tested, minimizing new untested logic.
  - Alternative considered: a dedicated short-only mean-reversion-at-highs
    strategy. Rejected for v1 as higher-risk/less-understood; can follow
    once `trend_following`-short is validated.
- Decision: model borrow/funding cost explicitly in the backtest execution
  model (not zero), matching this codebase's existing preference for
  pessimistic-by-default cost assumptions established in
  `add-trading-safety-boundaries`'s execution-cost-modeling work.

## Risks / Trade-offs
- Liquidation risk is categorically new — must be modeled even in backtest
  (a short position that would have been liquidated intraday must be closed
  at the liquidation price, not silently ride out an adverse move the way
  spot can).
- Funding-rate data availability/quality for backtesting perpetuals needs
  verification — `exchange.py`'s existing funding-rate lookups are live/
  read-only helpers, not a historical data source; historical funding data
  may need its own CSV/data pipeline alongside `data/backtest/`.

## Migration Plan
1. Backtest portfolio + execution model support for short positions with
   borrow/funding cost and liquidation modeling.
2. `trend_following`-short strategy variant, backtested and walk-forward
   validated across bull/bear/chop.
3. Only after step 2 shows validated positive out-of-sample expectancy:
   live execution wiring, gated behind explicit per-bot opt-in config
   (default: shorting disabled).
4. Portfolio risk service exposure calc updated to include short notional.

## Open Questions
- Which exchange(s) actually support the shorting/perpetual mechanism this
  bot would use live (confirm Hyperliquid perpetuals, given existing
  funding-rate code hints)? This determines how much of the live-wiring
  work is exchange-specific.
- Historical funding-rate data source for backtesting — does one already
  exist, or does this need its own data pipeline alongside
  `data/backtest/`?
