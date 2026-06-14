# Configuration

TradingBot uses layered configuration so a continuously deployed server can be
reproducible: required runtime behaviour lives in tracked files or the host
environment and is **not** lost when CI/CD resets the working tree.

## Precedence (lowest → highest)

Configuration is merged from four sources. Higher layers override lower ones;
unlisted keys fall through to the layer below.

| # | Layer | Source | Tracked in git? |
|---|-------|--------|-----------------|
| 1 | Base | `backend/config.yaml` | yes |
| 2 | Profile | `backend/config.<name>.yaml` (selected at startup) | yes (no secrets) |
| 3 | Env files | `TRADINGBOT_ENV_FILE`, `deploy/tradingbot.env` | **no** (secrets) |
| 4 | Process env | `TRADINGBOT__SECTION__KEY` variables | n/a (host) |

Merging is deep: e.g. a profile that sets only `server.port` keeps every other
`server.*` value from `config.yaml`.

The merged result is validated against the schema in
`backend/app/services/config.py`. **Unknown keys are rejected** and startup
fails with a clear message — this applies to env overrides too, so a typo'd
`TRADINGBOT__SERVER__BOGUS` is caught immediately.

## Profile selection

Select a profile with the `TRADINGBOT_PROFILE` environment variable:

```bash
export TRADINGBOT_PROFILE=production    # loads backend/config.production.yaml
```

- The profile file is resolved **next to `config.yaml`** (i.e.
  `backend/config.<name>.yaml`), independent of the current working directory.
- If the selected profile file is missing, startup logs a **warning and
  continues** with the remaining layers (it does not crash).
- Profiles are committed and contain only reproducible, **non-secret**
  overrides (ports, intervals, CORS origins, log level). Keep secrets out of
  them.

Example `backend/config.production.yaml`:

```yaml
logging:
  level: "WARNING"
trading:
  reconciliation_interval_seconds: 30
```

## Environment files

Environment files are optional `KEY=VALUE` files loaded into the process
environment at startup.

- Loaded, in priority order: the path in `TRADINGBOT_ENV_FILE` (if set), then
  `deploy/tradingbot.env`.
- Loading is **gap-fill**: a variable already present in the process
  environment is never overwritten, so **process environment variables take
  precedence over file values**.
- Lines beginning with `#` are ignored; an optional leading `export ` is
  stripped; surrounding quotes on the value are removed.
- Missing files are skipped silently (optional).

This is where secrets belong (`TRADINGBOT_API_TOKEN`, `MEXC_API_KEY`,
`MEXC_API_SECRET`, …). Under systemd these are also provided via
`EnvironmentFile=` — either way they end up in the process environment.

## Process-environment overrides

Any schema key can be overridden with an environment variable named
`TRADINGBOT__<SECTION>__<KEY>` (double underscore separates nesting levels, so
single-underscore key segments such as `frontend_dist` stay intact):

```bash
export TRADINGBOT__SERVER__PORT=8001                 # -> server.port (int)
export TRADINGBOT__SERVER__DEBUG=true                # -> server.debug (bool)
export TRADINGBOT__SERVER__CORS_ORIGINS="https://a,https://b"   # -> list
export TRADINGBOT__SERVER__FRONTEND_DIST=/srv/ui     # -> server.frontend_dist
```

Values are strings on the wire and are coerced to the schema type
(`int`/`float`/`bool`/`list`/`str`). `bool` accepts `1/true/yes/on`; `list`
splits on commas. A value that cannot be coerced fails validation at startup.

### Legacy single-underscore variables

These predate the generic mechanism and are handled specially (they are **not**
generic config overrides and never collide with `TRADINGBOT__…`):

- `TRADINGBOT_API_TOKEN` — API bearer token (overrides `server.api_token`).
- `TRADINGBOT_DATABASE_URL` — database URL (resolved directly by the
  persistence layer; takes precedence over `database.url`).
- `MEXC_API_KEY` / `MEXC_API_SECRET` — exchange credentials.

## Frontend serving (auto-discovery)

The web GUI is part of the product and is served **by default**:

- At startup the backend auto-discovers the build at `frontend/dist` (resolved
  relative to the backend directory, CWD-independent) and serves it from the
  API origin.
- If no build is present, startup logs a warning and continues in **API-only
  mode** (the JSON root and `/docs` still work).
- The location is configurable via `server.frontend_dist`
  (or `TRADINGBOT__SERVER__FRONTEND_DIST`) for non-default build paths.

Because serving is automatic, the UI works with **no manual config edit** and
survives CI/CD `git reset` of tracked files.

## Market-data feed (UI live price)

The dashboard's live price and indicators are fed from one of two sources,
selected by `market_data.source`. **This does not affect trading** — the engine
always reads its own REST tickers regardless.

| `market_data.source` | Behaviour |
|----------------------|-----------|
| `rest` (default) | Polls `fetch_ticker` every `market_data.rest_poll_interval_seconds` (default 2.0s) for subscribed symbols and broadcasts price updates. Works wherever REST works. |
| `websocket` | Uses the native MEXC stream connector (richer orderbook/volume indicators). |

MEXC **blocks its public websocket streams from many IPs/regions** (the stream
subscribe returns `Reason: Blocked!`) while REST stays available, which is why
`rest` is the default. Use `websocket` only on a host whose IP MEXC permits for
streaming. In `rest` mode the orderbook/volume/spread indicators are limited
(they need stream data); price and price-based indicators (e.g. realized
volatility) still populate.

## CI/CD reproducibility

Automated deploys reset the working tree to the repository state
(`git reset --hard`). To stay correct across that:

- **Tracked, reproducible config** → `config.yaml` and profile files. They are
  restored to a known state on every deploy.
- **Secrets / host-specific values** → env files and process environment.
  Untracked files survive `git reset` (it does not delete untracked files), and
  the systemd `EnvironmentFile=` is outside the repo entirely.
- **No required behaviour depends on a manual edit to a tracked file** — the
  previous "uncomment `frontend_dist`" step is gone (auto-discovery), so a
  deploy can never silently disable the UI.
