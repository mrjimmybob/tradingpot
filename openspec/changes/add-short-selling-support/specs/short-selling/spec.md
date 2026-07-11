## ADDED Requirements
### Requirement: Backtest Short Position Accounting
The backtest engine SHALL model short positions with negative base amount,
borrow/funding cost, and liquidation risk, separate from its existing
long-only accounting.

#### Scenario: Short position profits from a price decline
- **WHEN** a short is opened and the price declines before it is covered
- **THEN** the backtest portfolio's realized P&L reflects the price decline
  minus borrow/funding cost and fees

#### Scenario: Adverse move forces liquidation
- **WHEN** a short position's mark-to-market loss would breach the modeled
  margin requirement
- **THEN** the backtest force-closes the position at the liquidation price
  rather than allowing it to continue

### Requirement: Short Signals Reuse Existing No-Lookahead Guarantees
Strategies emitting short/cover signals SHALL be subject to the same
no-lookahead guarantee as long signals: a decision at candle i only sees
candles 0..i, and any resulting fill occurs at candle i+1.

#### Scenario: Short decision does not see the fill candle
- **WHEN** a strategy emits a short or cover signal while evaluating candle i
- **THEN** the fill occurs at candle i+1's open, and the decision at candle i
  never read candle i+1's data

### Requirement: Short Exposure in Portfolio Risk
The portfolio risk service SHALL include short position notional in its
total exposure calculation.

#### Scenario: Exposure cap counts short and long together
- **WHEN** an owner holds both long and short positions across their bots
- **THEN** the portfolio exposure cap check sums both toward the configured
  max_total_exposure_pct
