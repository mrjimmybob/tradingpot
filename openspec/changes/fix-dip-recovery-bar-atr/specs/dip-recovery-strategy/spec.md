## ADDED Requirements

### Requirement: Volatility Is Measured Over Time, Not Over Evaluation Ticks
`dip_recovery` SHALL derive its ATR volatility estimate from price bars of a
fixed wall-clock duration (`bar_interval_seconds`), using each bar's high-low
range, and SHALL NOT derive it from consecutive evaluation-tick price
differences. Every threshold the strategy scales by ATR — drop, recovery
confirmation, take-profit, trailing stop, emergency stop, and position sizing —
SHALL therefore denote the same span of market time regardless of how often the
strategy is evaluated.

#### Scenario: Evaluation cadence does not change the volatility estimate
- **WHEN** the strategy is evaluated many times within one bar interval
- **THEN** the ATR it uses reflects the bar's high-low range over that interval,
  not the price change between consecutive evaluations

#### Scenario: A faster caller does not shrink ATR-derived distances
- **WHEN** the strategy is evaluated at a cadence far finer than
  `bar_interval_seconds`
- **THEN** its take-profit and stop distances are unchanged by that cadence,
  rather than shrinking in proportion to it

### Requirement: ATR-Derived Distances Clear The Round-Trip Fee Hurdle
The ATR `dip_recovery` uses SHALL be floored so that ATR-derived exit distances
can never sit inside the round-trip fee hurdle plus the configured safety margin,
including before enough bars have been collected to compute a bar-based ATR.

#### Scenario: Warm-up does not produce microscopic stops
- **WHEN** fewer than `atr_period` bars have been collected
- **THEN** the ATR used is at least the round-trip fee hurdle plus the safety
  margin, rather than a near-zero value

#### Scenario: A real volatility estimate takes over once available
- **WHEN** enough bars have been collected and the bar-based ATR exceeds the
  fee-coverage floor
- **THEN** the bar-based ATR is used
