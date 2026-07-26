# strategy-validation-measurement Specification

## Purpose
Measure a strategy's given, fixed configuration objectively and explain the
result — and keep that strictly separate from choosing a better one. This
capability exists because every strategy default in this codebase was set by
hand and never checked against data, so "this strategy is profitable" could not
be distinguished from "the backtest window happened to be a bull market". It
owns the measurement/optimisation boundary itself, and the validated
measurement record (the `EdgeEstimate` shape) that the Strategy Decision
Framework defined but nothing could legitimately produce.

Parameter optimisation, search, and tuning are **out of scope for this
capability by construction**, not merely unimplemented: the tooling has no code
path that alters or writes a strategy parameter, and that is enforced by
structural guard tests rather than by convention.
## Requirements
### Requirement: Measurement Is Separate From Optimisation
The strategy validation tooling SHALL measure a strategy's given, fixed
configuration and explain the result, and SHALL NOT optimise. It SHALL NOT
search a parameter space, tune parameters, or select a "better" parameter set,
and SHALL NOT write `strategy_params` (or any strategy parameter) to any bot,
database, or configuration. Any parameter optimisation or search is out of
scope for this capability and belongs to a separate, future change.

#### Scenario: Measurement never searches parameters
- **WHEN** the validation tooling is run for a strategy, symbol, and date range
- **THEN** it measures exactly the fixed parameters it was given, and provides
  no code path that tries alternative parameters or ranks parameter sets

#### Scenario: Measurement never writes parameters back
- **WHEN** any validation measurement or report completes
- **THEN** no strategy parameter is written to any bot, database, or config —
  the output is a read-only report a human reviews

#### Scenario: Reuses the backtest engine unmodified
- **WHEN** the tooling measures a configuration
- **THEN** it calls `BacktestEngine.run_candles` unmodified, with the same
  no-lookahead behaviour as a manually run backtest of the same parameters

### Requirement: Validated Measurement Record
The tooling SHALL produce, from out-of-sample measurements only, a validated
measurement record with expectancy, win rate, profit factor, and sample size —
the `EdgeEstimate` shape defined by the Strategy Decision Framework, constructed
with the framework's validated-source marker. The tooling SHALL only report
this record; it SHALL NOT populate any live `StrategyProposal.expected_edge_
estimate` (that remains a separate, future change).

#### Scenario: A validated record is produced from out-of-sample measurement
- **WHEN** a strategy's fixed configuration is measured across the out-of-sample
  windows
- **THEN** the tooling can emit a validated measurement record (expectancy,
  win rate, profit factor, sample size) marked as validated-source, reflecting
  those out-of-sample windows rather than a single in-sample fit

#### Scenario: Runtime proposals are unaffected
- **WHEN** this change is implemented
- **THEN** every live `StrategyProposal.expected_edge_estimate` remains `None`;
  producing the validated record here does not wire it into runtime

