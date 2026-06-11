## ADDED Requirements
### Requirement: Execution Cost Estimation
The system SHALL estimate execution costs for each trade using configured fee, spread, and slippage/impact parameters.

#### Scenario: Estimate cost for a market buy
- **WHEN** a market buy is executed with size, price, and current spread data
- **THEN** the system estimates total execution cost and returns it to the execution layer

### Requirement: Cost Persistence
The system SHALL store modeled execution cost components alongside each order.

#### Scenario: Persist modeled costs on order creation
- **WHEN** an order is recorded
- **THEN** modeled fee, spread cost, and slippage/impact cost are stored with the order record

### Requirement: Cost-Aware PnL and Risk
The system SHALL apply modeled execution costs to wallet and PnL calculations used by risk checks.

#### Scenario: Loss limit uses cost-aware PnL
- **WHEN** daily loss is computed for risk checks
- **THEN** modeled execution costs are included in the loss calculation
