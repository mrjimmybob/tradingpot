# dry-run-market-data

## ADDED Requirements

### Requirement: Dry-run bots use real market data

The system SHALL source all market data (tickers) for dry-run bots from the exchange's public market-data API, so that dry-run strategy behavior reflects real market conditions. Account state (balances, order fills, order history) SHALL remain simulated.

#### Scenario: Ticker reflects live market price

- **WHEN** a dry-run bot requests a ticker for a trading pair
- **THEN** the returned bid/ask/last/volume come from the exchange's public API for that pair, not from hardcoded values

#### Scenario: Simulated fills use real prices

- **WHEN** a dry-run bot places a market order
- **THEN** the simulated fill price is derived from the real current ticker (ask for buys, bid for sells) and only the simulated balances are mutated

#### Scenario: No credentials required for dry-run

- **WHEN** a dry-run bot starts and no exchange API credentials are configured
- **THEN** the bot starts successfully and receives real market data via public endpoints

### Requirement: No fabricated market data on failure

When real market data cannot be obtained, the system SHALL surface the failure (no ticker) rather than fall back to fabricated prices, and the bot loop SHALL skip trading until data is available again.

#### Scenario: Public API unreachable

- **WHEN** the exchange public API is unreachable and a dry-run bot requests a ticker
- **THEN** no ticker is returned, no trade is executed in that iteration, and the bot retries on a subsequent iteration

### Requirement: Ticker request rate limiting

The system SHALL cache tickers for a short interval (default 2 seconds, configurable) so that per-second bot loops do not exceed exchange public rate limits.

#### Scenario: Repeated requests within TTL

- **WHEN** two ticker requests for the same symbol arrive within the cache TTL
- **THEN** at most one request is sent to the exchange public API
