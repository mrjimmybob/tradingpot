# exchange-credentials

## ADDED Requirements

### Requirement: Environment-variable credential override

The system SHALL resolve exchange API credentials in this order: environment variables (`<EXCHANGE_ID>_API_KEY` / `<EXCHANGE_ID>_API_SECRET`, e.g. `MEXC_API_KEY`), then the exchange YAML config, then unset. Credentials matching placeholder patterns (values starting with `YOUR_`) or empty strings SHALL be treated as unset.

#### Scenario: Env var wins over YAML

- **WHEN** `MEXC_API_KEY` is set in the environment and a different value exists in `config/exchanges.yaml`
- **THEN** the environment value is used to connect

#### Scenario: Placeholder treated as unset

- **WHEN** the YAML config contains `api_key: "YOUR_MEXC_API_KEY"` and no environment variable is set
- **THEN** the system treats credentials as unset and never sends the placeholder to the exchange

### Requirement: Live bots require usable credentials

The system SHALL refuse to start a non-dry-run (live) bot when exchange credentials are unset, reporting a clear error.

#### Scenario: Live start without credentials

- **WHEN** a user starts a live bot and no usable exchange credentials are configured
- **THEN** the bot does not start, no trading loop is created, and the error states that exchange credentials are required for live trading

#### Scenario: Dry-run unaffected

- **WHEN** a user starts a dry-run bot with no credentials configured
- **THEN** the bot starts normally

### Requirement: Exchange connection failures abort bot start

The system SHALL verify that the exchange connection succeeds during bot start and SHALL fail the start (no trading loop, clear error) when the connection fails.

#### Scenario: Connect failure on live start

- **WHEN** `connect()` to the exchange fails while starting a bot
- **THEN** the bot is not started, its status reflects the failure, and the API response contains the reason
