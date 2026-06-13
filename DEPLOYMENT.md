# Deployment — Unattended Dry Run (Debian)

Runbook for deploying the bot to a Debian home-lab server for the 30-day dry
run, with the backend bound to **loopback** (reached via SSH tunnel), supervised
by **systemd**, and the **web UI served by the backend** on the same origin.

> Dry run uses the exchange's **public market data API** with simulated balances
> and fills. **No exchange credentials are required.** The server does need
> **outbound internet** (HTTPS to MEXC).

---

## 1. Get the code

```bash
sudo mkdir -p /opt/tradingbot && sudo chown "$USER" /opt/tradingbot
git clone <your-repo-url> /opt/tradingbot
cd /opt/tradingbot
```

Python 3.11+ (Debian 12 ships 3.11, Debian 13/Trixie ships 3.13 — both fine).

## 2. One-command deploy (recommended)

```bash
./deploy/install.sh
```

This is the reproducible path. It installs apt prerequisites, builds the
backend venv and web UI, creates the service user, **validates the install,
and only then enables the service**. It renders the systemd unit to point at
wherever you actually cloned the repo, so no manual path-editing is needed.

The deployment is checked **before** the service is enabled, so a broken
backend environment fails the installer with a clear message instead of
surfacing later as `systemctl status` → `status=203/EXEC`. You can re-run the
validation at any time:

```bash
./deploy/validate_deployment.sh --deep
```

Then skip to **§3 Serve the web UI** (config tweak) and **§6 onwards**. The
manual steps below (§2a–§5) are the same thing broken out, for the curious or
for non-`/opt/tradingbot` setups.

After deploy, in `/opt/tradingbot/backend/config.yaml` under `server:`
uncomment `frontend_dist: "../frontend/dist"` so the API serves the built UI
on one origin, then `sudo systemctl restart tradingbot`. Leave
`host: "127.0.0.1"` (loopback); no API token is needed on loopback.

<details>
<summary><b>Manual steps (what install.sh automates)</b></summary>

### 2a. Prerequisites

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
# Node 18+ is needed to BUILD the web UI. Debian's nodejs may be too old; if so:
#   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs npm
```

### 2b. Build the backend env

```bash
cd /opt/tradingbot
./init.sh          # creates venv, installs backend + frontend deps, writes config templates
```

`init.sh` is self-verifying: it rebuilds a broken/partial venv and **aborts if
the venv or `venv/bin/uvicorn` is missing** after install. It ends by
suggesting `uvicorn --reload` — **ignore that for the dry run**; use the
systemd service (`--reload` is dev-only and won't survive a crash).

### 3. Build the web UI

```bash
cd /opt/tradingbot/frontend
npm run build       # produces frontend/dist
```

### 4. Service user and permissions

```bash
sudo useradd --system --home /opt/tradingbot --shell /usr/sbin/nologin tradingbot
sudo chown -R tradingbot:tradingbot /opt/tradingbot
```

The service writes to `backend/tradingbot.db`, `backend/logs/`, and
`backend/backups/`, so the `tradingbot` user must own the tree (above does this).

### 5. Validate, install, and start the service

```bash
cp /opt/tradingbot/deploy/tradingbot.env.example /opt/tradingbot/deploy/tradingbot.env
# (loopback dry run: nothing to edit)

# Gate: confirm venv + uvicorn + packages + ExecStart target all exist.
/opt/tradingbot/deploy/validate_deployment.sh --deep \
  --unit /opt/tradingbot/deploy/tradingbot.service

sudo cp /opt/tradingbot/deploy/tradingbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tradingbot
systemctl status tradingbot
```

If you cloned somewhere other than `/opt/tradingbot`, edit the `WorkingDirectory`,
`EnvironmentFile`, `ExecStart`, and `ExecStartPre` paths in the unit first
(or just use `./deploy/install.sh`, which does this for you). The unit also
runs `validate_deployment.sh` as an `ExecStartPre`, so even a hand-installed
service refuses to start with a broken environment rather than dying with
`status=203/EXEC`.

</details>

A fresh database is created automatically on first start (schema includes the
`strategy_state` column). **Reusing an older DB?** Run
`venv/bin/python backend/migrations/run_migrations.py` once before starting.

## 6. Reach the UI/API from your laptop

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<server>
```

Then open `http://localhost:8000` (web UI) and `http://localhost:8000/docs`
(API). The WebSocket also rides port 8000, so this single tunnel covers
everything.

## 7. Start the dry run (create bots)

Create one or more **dry-run** bots via the UI, or the API:

```bash
curl -X POST http://localhost:8000/api/bots \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "BTC funding-carry (dry)",
        "trading_pair": "BTC/USDT",
        "strategy": "funding_carry",
        "strategy_params": {},
        "budget": 1000,
        "is_dry_run": true,
        "stop_loss_percent": 10
      }'
# then start it:
curl -X POST http://localhost:8000/api/bots/1/start
```

`is_dry_run: true` is what keeps it on simulated fills. Running bots **auto-resume**
after a restart, so the run continues across reboots/crashes.

## 8. Monitor

- Service logs: `journalctl -u tradingbot -f`
- Per-bot activity/trades: `backend/logs/<bot_id>/`
- API/UI: `/docs`, `/api/stats`, `/api/bots`
- DB: `backend/tradingbot.db`

## 9. Data, backups, recovery

- State lives under `backend/`: `tradingbot.db`, `logs/`, `backups/` (paths are
  absolute/stable regardless of working directory).
- Backups are automatic: one on startup, then every `backup_interval_seconds`
  (default 1h), kept under `backend/backups/` (last 7).
- Restore: stop the service, copy a chosen `backend/backups/tradingbot-*.db` over
  `backend/tradingbot.db`, start again.

## 10. Update / stop

```bash
# update
cd /opt/tradingbot && git pull
venv/bin/pip install -r backend/requirements.txt
( cd frontend && npm run build )
./deploy/validate_deployment.sh --deep   # re-check before restarting
sudo systemctl restart tradingbot

# stop (graceful: saves bot state; bots stay RUNNING and resume on next start)
sudo systemctl stop tradingbot
```

## Going live later (not part of the dry run)

Set `MEXC_API_KEY`/`MEXC_API_SECRET` in `deploy/tradingbot.env`, create bots with
`is_dry_run: false`, and start with a small budget. To expose the UI on the LAN
instead of a tunnel, set `server.host: "0.0.0.0"`, a `TRADINGBOT_API_TOKEN`, and
`server.cors_origins` — and put it behind a TLS reverse proxy.
