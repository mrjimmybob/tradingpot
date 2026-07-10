## ADDED Requirements

### Requirement: Position Ownership Persistence
The system SHALL persist the identity of the strategy that opened a position, along with its entry reason
and any minimal strategy state required for exit decisions, on the position record itself at the moment the
position is opened.

#### Scenario: Opening a position records ownership
- **WHEN** any strategy executor opens a new position
- **THEN** the resulting `Position` row has `owning_strategy` set to that strategy's name and `entry_reason`
  set to a human-readable reason string

#### Scenario: Ownership survives a service restart
- **WHEN** a bot with an open, owned position is restarted (state saved and reloaded from
  `Bot.strategy_state`)
- **THEN** Auto Mode resolves the owning strategy from the persisted `Position` row, independent of whether
  the in-memory strategy-selection pointer was correctly restored

### Requirement: Ownership Enforcement During Open Position
While a position is open, the system SHALL dispatch market updates only to the strategy recorded as its
owner. Auto Mode MUST NOT switch strategy or allow a different strategy to act on an owned open position,
regardless of regime changes or relative strategy scoring.

#### Scenario: Strategy A opens, Strategy B cannot close
- **GIVEN** Auto Mode selects `mean_reversion`, which opens a position (`owning_strategy="mean_reversion"`)
- **WHEN** the market regime changes such that `trend_following` would score higher, while `mean_reversion`
  itself remains entry-eligible
- **THEN** Auto Mode continues dispatching to `mean_reversion` for this position; `trend_following` never
  receives a market update for it, and only `mean_reversion`'s own exit rules can close it

#### Scenario: Ineligible owning strategy still manages its own exit
- **GIVEN** a strategy owns an open position and then loses entry eligibility (e.g., regime no longer
  matches)
- **WHEN** Auto Mode evaluates the next market update
- **THEN** the owning strategy is still dispatched to manage the position's own exit rules; no other
  strategy is substituted

### Requirement: Reselection Only After Position Close
Auto Mode SHALL only re-score and reselect among eligible strategies when no position is currently open.

#### Scenario: Auto Mode reallocates after close
- **GIVEN** the owning strategy's position has fully closed
- **WHEN** Auto Mode evaluates the next market update
- **THEN** Auto Mode is free to score all eligible strategies and select a new one, including a different
  strategy than the one that just closed
