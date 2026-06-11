# balance-reconciliation

## ADDED Requirements

### Requirement: Periodic live balance reconciliation

The system SHALL periodically (default every 300 seconds, configurable) compare the exchange account's actual balances against the aggregate expectations of all running live bots: total virtual cash in the quote currency and total open position amounts per base asset. Dry-run bots SHALL be excluded.

#### Scenario: Balances sufficient

- **WHEN** reconciliation runs and the exchange balances cover the live bots' expected cash and positions within tolerance
- **THEN** the result is logged at info level and no alert is created

#### Scenario: No live bots running

- **WHEN** no live bots are running
- **THEN** reconciliation performs no exchange calls

### Requirement: Reconciliation drift alerting

When the exchange balance for any checked asset is below the live bots' aggregate expectation by more than the tolerance (default 1%), the system SHALL log a warning and create an `Alert` record (alert_type `balance_reconciliation`) describing the asset, the expected amount, and the actual amount. Reconciliation SHALL NOT stop or pause bots.

#### Scenario: Quote currency shortfall

- **WHEN** the exchange's quote-currency balance is more than the tolerance below the sum of running live bots' virtual cash balances
- **THEN** a warning is logged and an Alert is created naming the quote currency with expected and actual amounts

#### Scenario: Base asset shortfall

- **WHEN** the exchange's balance of a base asset is more than the tolerance below the sum of running live bots' open position amounts in that asset
- **THEN** a warning is logged and an Alert is created naming the base asset with expected and actual amounts

#### Scenario: Trading continues

- **WHEN** a reconciliation alert is created
- **THEN** running bots keep running (alert-only behavior)
