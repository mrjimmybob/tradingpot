## ADDED Requirements

### Requirement: Benchmark Equity Curves
The system SHALL construct benchmark equity curves from the same candle series a
measurement ran on, as deterministic functions of those candles, applying the
same execution costs the measured strategy paid. It SHALL provide at least a
buy-and-hold benchmark (full capital deployed at the first candle and held) and
a periodic-DCA benchmark (capital deployed in equal instalments at a fixed
cadence and never sold). The system SHALL NOT run a strategy or the backtest
replay loop to produce a benchmark.

#### Scenario: Buy-and-hold tracks the asset from the first candle
- **WHEN** a buy-and-hold benchmark is built over a candle series
- **THEN** its equity follows the asset's price from the first candle onward,
  net of the entry cost implied by the measurement's fee model

#### Scenario: Periodic DCA deploys gradually and never sells
- **WHEN** a periodic-DCA benchmark is built with a given cadence
- **THEN** capital is deployed in equal instalments at that cadence, is never
  sold, and each instalment pays the measurement's fee model

#### Scenario: Benchmarks are deterministic
- **WHEN** the same benchmark is built twice over the same candles and costs
- **THEN** the two equity curves are identical

### Requirement: Benchmark-Relative Return And Drawdown
The system SHALL report, for a measured strategy over a given span, its terminal
return, its maximum drawdown, its return per unit of maximum drawdown, and — for
each benchmark — the excess return and the drawdown difference relative to that
benchmark. These measures SHALL be derived from the strategy's recorded equity
curve, so they are defined even when the strategy closed no round trips.

#### Scenario: A strategy that closes no trades is still measured
- **WHEN** a strategy completes a measurement having closed zero round trips
- **THEN** its return, drawdown, and benchmark-relative measures are reported
  from its equity curve, instead of being reported as absent or as zero

#### Scenario: Both return and drawdown are compared
- **WHEN** benchmark-relative measures are reported for a strategy
- **THEN** the report shows the strategy's return and drawdown against each
  benchmark's return and drawdown, not return alone

### Requirement: Exposure Is Reported Alongside Benchmark Comparison
The system SHALL report a measure of the strategy's realised exposure to the
asset alongside any benchmark comparison, and SHALL state that a difference in
exposure — not skill — can account for a difference in return. Where the
exposure measure cannot be computed meaningfully, the system SHALL report it as
unavailable rather than emit a misleading value.

#### Scenario: An under-deployed strategy is not misread as a poor one
- **WHEN** a gradually deploying strategy is compared against buy-and-hold
- **THEN** its realised exposure is shown on the same report, so a shortfall
  explained by lower exposure is distinguishable from one explained by selection

#### Scenario: A degenerate exposure estimate is withheld
- **WHEN** the exposure measure has a degenerate basis, such as a span with too
  few points or an asset with no price variation
- **THEN** the system reports exposure as unavailable rather than a number

### Requirement: Benchmark Comparison Is Measurement, Not Selection
Benchmark-relative reporting SHALL compare what was measured and SHALL NOT rank,
score, or recommend a strategy, and SHALL NOT alter, search, or select any
strategy parameter or benchmark parameter. Benchmark parameters, including the
periodic-DCA cadence, SHALL be reported with the result and SHALL NOT be varied
to improve any strategy's apparent standing.

#### Scenario: Results are never ordered by performance
- **WHEN** benchmark-relative results for several strategies are reported
  together
- **THEN** they appear in a fixed order that does not depend on their results,
  and the report states that it is a comparison rather than a ranking

#### Scenario: Benchmark parameters are disclosed, not tuned
- **WHEN** a benchmark with a configurable parameter is reported
- **THEN** the parameter's value is shown with the result, and no code path
  exists that varies it to change a strategy's apparent standing
