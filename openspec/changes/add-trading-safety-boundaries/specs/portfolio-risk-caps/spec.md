## ADDED Requirements
### Requirement: Portfolio Risk Cap Configuration
The system SHALL allow configuration of portfolio-level risk caps that apply across all bots.

#### Scenario: Configure portfolio caps
- **WHEN** the operator sets portfolio loss and exposure caps
- **THEN** the system stores and validates those caps for enforcement

### Requirement: Portfolio Risk Cap Enforcement
The system SHALL block or pause trading when portfolio-level caps are reached.

#### Scenario: Block trade due to portfolio loss cap
- **WHEN** a new trade is about to execute and the portfolio daily loss cap is exceeded
- **THEN** the trade is rejected and the bot is paused with a clear reason
