## ADDED Requirements

### Requirement: Setup Detection Advances In Market Time
`dip_recovery` SHALL advance its setup lifecycle — reference-high tracking,
decline detection, low tracking, reversal confirmation and its confirmation
counters — once per completed bar of `bar_interval_seconds`, and SHALL derive
every setup input from that bar series rather than from consecutive evaluations.
Parameters expressed as counts SHALL therefore denote bars of market time.

#### Scenario: Setup progress does not depend on evaluation frequency
- **WHEN** the strategy is evaluated many times within one bar interval
- **THEN** its setup lifecycle advances once, and its confirmation counters
  increase by at most one

#### Scenario: The reference high spans the configured number of bars
- **WHEN** a decline is measured against the recent high
- **THEN** the high is taken over `reference_high_lookback_ticks` bars of market
  time, not over that many evaluations

### Requirement: Exit Management Is Not Deferred To Bar Close
Position exits SHALL continue to be evaluated on every evaluation, independently
of bar completion, so a stop reacts when price moves rather than at the end of a
bar.

#### Scenario: A stop triggers mid-bar
- **WHEN** an open position's stop condition is met partway through a bar
- **THEN** the exit is produced on that evaluation rather than deferred
