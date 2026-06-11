# Change: Add live-readiness hardening (real dry-run data, API auth, credential safety)

## Why

The system cannot currently be trusted for the intended rollout path (dry-run validation → small live-money test):

1. **Dry-run mode uses hardcoded constant prices** (`SimulatedExchangeService.get_ticker` returns BTC=45000 always), so paper trading validates nothing about strategy behavior against real markets.
2. **The API has zero authentication** and the run script binds uvicorn to `0.0.0.0`, so anyone with network access can start/kill bots or trigger live trades.
3. **Exchange credentials live only in plaintext YAML** with no environment-variable path, and placeholder values ("YOUR_MEXC_API_KEY") are passed to the exchange as if real.
4. **`start_bot` ignores the result of `exchange.connect()`**, so a live bot whose exchange connection failed still starts its trading loop.

## What Changes

- **Dry-run market data**: `SimulatedExchangeService` fetches real tickers from the exchange's public REST API (no credentials required) while keeping balances and order fills simulated. No fake-price fallback: if real data is unavailable, the ticker is unavailable and the bot loop skips the iteration (existing safe path). A short-TTL ticker cache limits REST load from 1-second bot loops.
- **API security**: bearer-token authentication for all `/api` routes and the WebSocket, configured via the `TRADINGBOT_API_TOKEN` environment variable. Server default binding changes from `0.0.0.0` to `127.0.0.1`. Fail-safe startup check: refuse to start on a non-loopback host without a token configured. Frontend gets a central API client that attaches the token.
- **Exchange credentials**: environment variables (`MEXC_API_KEY`, `MEXC_API_SECRET`) override YAML values; placeholder values are treated as unset; live (non-dry-run) bots refuse to start without usable credentials.
- **Connect failure handling**: `start_bot` checks the `connect()` result and fails the bot start with a clear error instead of starting the loop.

Out of scope (deliberately): slippage/fill realism for simulated limit orders (covered by the pending `add-trading-safety-boundaries` change), multi-user auth, TLS, dependency pinning.

## Impact

- Affected specs: `dry-run-market-data` (new), `api-security` (new), `exchange-credentials` (new)
- Affected code:
  - `backend/app/services/exchange.py` (simulator market data, env-var credentials, placeholder detection)
  - `backend/app/services/trading_engine.py` (connect-failure handling, live credential guard)
  - `backend/app/main.py` (auth middleware, startup binding fail-safe)
  - `backend/config.yaml`, `backend/run.ps1` (default host 127.0.0.1)
  - `frontend/src/lib/api.ts` (new central client) and the ~15 files using `fetch` directly
  - `frontend/src/contexts/WebSocketContext.tsx` / `frontend/src/hooks/useWebSocket.ts` (token on WS connect)
- Affected tests: existing tests keep passing (they mock `get_ticker`); new tests for auth, credential resolution, simulator delegation, and connect-failure handling.
