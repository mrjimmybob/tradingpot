## ADDED Requirements
### Requirement: Unified Volatility-Scaled Position Sizing
All strategies SHALL size orders using one shared, volatility (ATR) scaled,
risk-percent-of-balance formula rather than strategy-specific ad hoc
conventions.

#### Scenario: Order size scales down in a volatile market
- **WHEN** two otherwise-identical entry signals occur, one in a low-ATR
  market and one in a high-ATR market
- **THEN** the high-ATR signal produces a smaller position size for the same
  risk_percent and balance

#### Scenario: Existing safety floors and caps still apply
- **WHEN** the shared sizing function computes an order size
- **THEN** the result is still floored to the exchange minimum order size and
  capped to the configured maximum fraction of available balance

### Requirement: Strategy-Specific Sizing Ceilings Preserved
The system SHALL treat a strategy's previous fixed percentage/dollar sizing
value as a ceiling on the new volatility-scaled size, not a value the new
formula ignores.

#### Scenario: Grid's depth multiplier still applies
- **WHEN** adaptive_grid sizes an order at a given grid depth
- **THEN** the depth-based convex multiplier is applied on top of the shared
  base size, preserving "bigger orders at deeper discount levels"
