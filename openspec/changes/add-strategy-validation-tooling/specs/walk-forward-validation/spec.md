## ADDED Requirements
### Requirement: Rolling Train/Validate Splitting
The system SHALL split a historical date range into rolling train and
validate windows for out-of-sample parameter validation.

#### Scenario: Non-overlapping validate windows across the full range
- **WHEN** an operator runs walk-forward validation over a date range with a
  configured train/validate window size
- **THEN** the system produces a sequence of train windows each followed by
  a held-out validate window, together covering the full requested range

### Requirement: Out-of-Sample Scoring
The system SHALL score a strategy's optimized parameters on each validate
window using only data the optimizer did not see during that window's train
step.

#### Scenario: Overfit parameters are flagged
- **WHEN** a parameter set's validate-window expectancy is materially worse
  than its train-window expectancy
- **THEN** the walk-forward report flags that window as a likely overfit
