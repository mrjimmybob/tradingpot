# api-security

## ADDED Requirements

### Requirement: Bearer-token authentication for the API

The system SHALL require a bearer token (configured via the `TRADINGBOT_API_TOKEN` environment variable, with `server.api_token` in `config.yaml` as fallback) on all `/api` HTTP routes when a token is configured. The root endpoint and `/api/health` SHALL remain unauthenticated for liveness checks.

#### Scenario: Request without token is rejected

- **WHEN** a token is configured and a request to an `/api` route (other than `/api/health`) arrives without a valid `Authorization: Bearer` header
- **THEN** the request is rejected with HTTP 401 and no handler logic executes

#### Scenario: Request with valid token is served

- **WHEN** a token is configured and a request carries `Authorization: Bearer <token>` matching the configured token
- **THEN** the request is processed normally

#### Scenario: Health check stays open

- **WHEN** a token is configured and a request arrives at `/api/health` without a token
- **THEN** the request succeeds

### Requirement: WebSocket authentication

The system SHALL require the configured token on WebSocket connections (via `token` query parameter) before accepting the connection, when a token is configured.

#### Scenario: WebSocket without token is refused

- **WHEN** a token is configured and a WebSocket client connects without a valid `token` query parameter
- **THEN** the connection is closed without being registered with the WebSocket manager

### Requirement: Fail-safe network binding

The server SHALL bind to loopback (`127.0.0.1`) by default. The server SHALL refuse to start when configured to bind a non-loopback address while no API token is configured.

#### Scenario: Default local startup

- **WHEN** the server starts with default configuration and no token
- **THEN** it binds `127.0.0.1` and serves requests without authentication

#### Scenario: Public binding without token is blocked

- **WHEN** the server is configured with a non-loopback host (e.g. `0.0.0.0`) and no API token
- **THEN** startup aborts with a clear error explaining that a token must be set

### Requirement: Frontend sends the token

The frontend SHALL send the configured API token (stored locally, settable in the Settings page) on every API request and WebSocket connection, and SHALL surface authentication failures to the user.

#### Scenario: Authenticated frontend request

- **WHEN** a token is stored in the frontend and any page issues an API request
- **THEN** the request carries the `Authorization: Bearer` header with that token

#### Scenario: Auth failure is visible

- **WHEN** an API request is rejected with HTTP 401
- **THEN** the user sees a clear message indicating the API token is missing or invalid
