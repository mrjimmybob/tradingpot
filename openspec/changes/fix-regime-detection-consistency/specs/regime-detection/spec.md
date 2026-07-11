## ADDED Requirements
### Requirement: Volatility Direction Detection
The system SHALL classify volatility as expanding, contracting, or stable
based on its rate of change, distinct from its absolute level (low/medium/
high).

#### Scenario: Sustained high volatility is not "expanding"
- **WHEN** volatility has been at a high level and roughly flat for several
  bars
- **THEN** the detector reports volatility_direction as "stable", not
  "expanding"

#### Scenario: Volatility accelerating after compression is "expanding"
- **WHEN** volatility was low/compressed and has been rising for at least the
  detector's persistence window
- **THEN** the detector reports volatility_direction as "expanding"

### Requirement: Volatility Breakout Entry Gate Uses Direction
The volatility_breakout strategy's regime gate SHALL require
volatility_direction == "expanding" to allow entry, not merely
volatility_state == "high".

#### Scenario: High-but-flat volatility blocks entry
- **WHEN** volatility_state is "high" but volatility_direction is "stable" or
  "contracting"
- **THEN** volatility_breakout's entry gate blocks new entries

#### Scenario: Genuine expansion after compression allows entry
- **WHEN** volatility_direction is "expanding" and the strategy's other entry
  conditions (armed breakout, upper-band close) are met
- **THEN** volatility_breakout is permitted to enter
