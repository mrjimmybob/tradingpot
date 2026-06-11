# live-order-execution

## ADDED Requirements

### Requirement: Order pre-flight validation against exchange rules

Before submitting a live order, the system SHALL round the order amount (and limit price, when present) to the exchange's precision rules and SHALL verify the order satisfies the market's minimum amount and minimum cost limits. Orders failing the limits SHALL be rejected locally with a log message stating the limit violated, without contacting the exchange order endpoint.

#### Scenario: Amount rounded to exchange precision

- **WHEN** a live market order is placed with an amount of higher precision than the market allows
- **THEN** the submitted amount is rounded to the market's precision before submission

#### Scenario: Below minimum cost rejected locally

- **WHEN** a live order's notional value is below the market's minimum cost
- **THEN** no order is sent to the exchange and the rejection reason (including the minimum) is logged

#### Scenario: Dry-run unaffected

- **WHEN** a dry-run bot places a simulated order
- **THEN** no pre-flight exchange-rule validation is applied (fills remain simulated and immediate)

### Requirement: Raw exchange response audit logging

The system SHALL log the complete raw exchange response for every live order placement.

#### Scenario: Live order placed

- **WHEN** a live order is submitted and the exchange responds
- **THEN** the full raw response payload is written to the log associated with the order

### Requirement: Trades recorded from actual execution results

The system SHALL record trades, ledger entries, and position updates from the exchange-reported execution results: the actual filled amount, the exchange-reported total cost when available, and the exchange-reported fee currency (falling back to the quote asset when the exchange does not report one). This applies to market, limit, TWAP, and VWAP execution paths.

#### Scenario: Partial fill

- **WHEN** a live order fills partially (filled < requested)
- **THEN** the trade record, ledger entries, and the position update all use the filled amount, not the requested amount

#### Scenario: Fee charged in base asset

- **WHEN** the exchange reports the order fee in the base asset
- **THEN** the trade record stores that fee currency rather than assuming the quote asset
