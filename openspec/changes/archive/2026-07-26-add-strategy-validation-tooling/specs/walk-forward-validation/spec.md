## ADDED Requirements

### Requirement: Rolling Out-of-Sample Windows
The system SHALL split a historical date range into a sequence of rolling
windows so a strategy's fixed configuration can be measured on independent,
successive out-of-sample periods rather than one full-period backtest. The
window split SHALL exist only to make samples independent; it SHALL NOT be used
to search, tune, or select parameters.

#### Scenario: Successive windows cover the range
- **WHEN** an operator runs walk-forward measurement over a date range with a
  configured window size and step
- **THEN** the system produces a sequence of successive windows together
  covering the requested range, each measured independently

### Requirement: Fixed-Configuration Out-of-Sample Measurement
The system SHALL measure the SAME operator-supplied fixed parameters on every
window and report each window's metrics (expectancy, win rate, profit factor,
drawdown, trade count) side by side, so consistency or inconsistency of the
strategy's edge across time is visible. The system SHALL NOT alter parameters
between windows.

#### Scenario: An inconsistent edge is shown, not fixed
- **WHEN** a strategy's fixed configuration shows an edge in some windows and
  not in others
- **THEN** the report shows the per-window results side by side so the
  inconsistency is visible, and the tooling makes no attempt to change the
  parameters to improve later windows

#### Scenario: Same parameters on every window
- **WHEN** walk-forward measurement runs across all windows
- **THEN** every window is measured with the identical operator-supplied
  parameters, with no per-window parameter change of any kind
