## ADDED Requirements
### Requirement: Strategy Parameter Search
The system SHALL search a strategy's parameter space against historical
candle data using the existing backtest engine and rank results by a
configurable objective.

#### Scenario: Grid search finds a better parameter set
- **WHEN** an operator runs the optimizer over a defined parameter grid for a
  strategy/symbol/date range
- **THEN** the system reports each combination's backtest metrics and ranks
  them by the configured objective

#### Scenario: Optimizer never changes backtest engine behavior
- **WHEN** the optimizer runs any parameter combination
- **THEN** it calls `BacktestEngine.run_candles` unmodified, with no
  lookahead or calculation differences from a manually-run backtest of the
  same parameters

### Requirement: Drawdown-Constrained Objective
The system SHALL support an optimization objective that ranks parameter sets
by expectancy while excluding or penalizing sets whose max drawdown exceeds a
configured ceiling.

#### Scenario: High-return high-ruin-risk parameters are deprioritized
- **WHEN** two parameter sets are compared and one has higher expectancy but
  exceeds the configured max-drawdown ceiling
- **THEN** the optimizer ranks the set within the drawdown ceiling higher
