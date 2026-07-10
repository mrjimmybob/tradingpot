## ADDED Requirements

### Requirement: Fee Modeling
The backtest execution model SHALL apply a per-side fee, defaulting to 0.1%, to every simulated fill.

#### Scenario: Fees reduce returns
- **WHEN** a backtest is run with the default 0.1% per-side fee versus a hypothetical zero-fee run over the
  same trades
- **THEN** the fee-inclusive run's ending balance and total return are lower by the sum of the modeled fees

### Requirement: Configurable Spread and Slippage
The execution model SHALL support configurable spread and slippage parameters, applied to simulated fill
prices.

#### Scenario: Non-zero spread/slippage worsens fill price
- **WHEN** spread and/or slippage parameters are set above zero
- **THEN** simulated buy fills occur at a price at or above the reference candle open, and sell fills at a
  price at or below it, by the configured amount

### Requirement: No-Lookahead Fill Timing
A market order generated from a decision on candle N SHALL execute at candle `N+1`'s open price. The
execution model SHALL NOT have access to candle `N+1` or later at decision time.

#### Scenario: Fill price is next candle's open
- **WHEN** a decision on candle N produces a market order
- **THEN** the simulated execution price is candle `N+1`'s open, adjusted by configured fee/spread/slippage
