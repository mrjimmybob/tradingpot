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

#### Scenario: Caps aggregate across every bot the owner has, not just one
- **WHEN** an owner has multiple bots and any combination of them together
  breaches a configured loss, drawdown, or exposure cap
- **THEN** the cap check reflects the combined state of all of that owner's
  bots, even if the specific bot currently placing an order is individually
  within its own limits

#### Scenario: Loss caps reflect realized trading P&L, not just costs
- **WHEN** the daily or weekly loss cap is evaluated
- **THEN** the loss figure includes realized price-driven trading losses for
  the period, not only exchange fees and modeled execution costs
