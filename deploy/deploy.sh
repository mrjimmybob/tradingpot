#!/usr/bin/env bash
#
# deploy/deploy.sh — in-place deploy of the tradingbot backend + web UI on the
# homelab server. Invoked by the self-hosted GitHub Actions runner (see
# deploy/RUNNER_SETUP.md), and also safe to run by hand on the server.
#
# Flow:
#   fetch -> reset to target ref -> backend deps -> build UI -> validate env
#   -> stop service -> back up DB -> migrate -> start service -> health check.
#
# On ANY failure during deploy (build error, migration error, the service not
# coming back healthy, ...) it rolls the CODE back to the commit that was live
# before this run, rebuilds, and restarts — so the server is never left on a
# half-deployed or unhealthy build. Database migrations are additive/idempotent
# and are intentionally NOT reverted (rolling them back could discard real
# ledger rows; older code tolerates the newer schema).
#
# Configuration (all env vars optional):
#   TRADINGBOT_ROOT        repo / deploy root      (default /opt/tradingbot)
#   TRADINGBOT_SERVICE     systemd unit name       (default tradingbot)
#   TRADINGBOT_HEALTH_URL  health endpoint         (default http://127.0.0.1:8000/api/health)
#   DEPLOY_REF             git ref/sha to deploy   (default origin/main)
#
set -uo pipefail

ROOT="${TRADINGBOT_ROOT:-/opt/tradingbot}"
SERVICE="${TRADINGBOT_SERVICE:-tradingbot}"
HEALTH_URL="${TRADINGBOT_HEALTH_URL:-http://127.0.0.1:8000/api/health}"
DEPLOY_REF="${DEPLOY_REF:-origin/main}"

VENV_PY="$ROOT/venv/bin/python"
VENV_PIP="$ROOT/venv/bin/pip"

log() { echo -e "\n=== $* ==="; }
err() { echo "ERROR: $*" >&2; }

cd "$ROOT" || { err "deploy root not found: $ROOT"; exit 2; }

# --- helpers ---------------------------------------------------------------

backup_db() {
    local db="$ROOT/backend/tradingbot.db"
    if [ ! -f "$db" ]; then
        log "No database yet at $db; skipping pre-migration backup"
        return 0
    fi
    local dir="$ROOT/backend/backups"
    mkdir -p "$dir"
    local dest="$dir/predeploy-$(date -u +%Y%m%dT%H%M%SZ).db"
    log "Backing up database -> $dest"
    cp "$db" "$dest"
    # WAL/SHM sidecars (service is stopped here, so these copy consistently)
    [ -f "$db-wal" ] && cp "$db-wal" "$dest-wal"
    [ -f "$db-shm" ] && cp "$db-shm" "$dest-shm"
    return 0
}

# build_and_restart <git-ref> [migrate|skip-migrate]
# Reset the working tree to a ref, rebuild backend+frontend, then stop the
# service, (optionally) migrate, and start it again. Returns non-zero on the
# first failing step. ALWAYS call this inside an `if` so `set -e`-style aborts
# don't skip the caller's failure handling.
#
# The migrate mode (default) runs DB migrations; rollback passes "skip-migrate".
# Migrations are forward-only and additive, and older code tolerates the newer
# schema, so on rollback re-running them is unnecessary — and if a migration was
# itself the failure, re-running it would only fail the rollback too (the
# original defect that left the service stopped and the box needing a manual
# start). The migration runner is independently self-sufficient (it creates the
# ORM baseline before applying SQL), so a forward migrate no longer depends on
# the app having started first.
build_and_restart() {
    local ref="$1"
    local migrate="${2:-migrate}"
    git reset --hard "$ref" || { err "git reset to $ref failed"; return 1; }
    log "Checked out $(git rev-parse --short HEAD)  ($(git log -1 --pretty=%s))"

    log "Installing backend dependencies"
    "$VENV_PIP" install --quiet --upgrade -r backend/requirements.txt || { err "pip install failed"; return 1; }

    log "Building web UI"
    ( cd frontend && npm ci && npm run build ) || { err "frontend build failed"; return 1; }

    log "Validating environment"
    deploy/validate_deployment.sh --deep --root "$ROOT" --quiet || { err "deployment validation failed"; return 1; }

    log "Stopping $SERVICE (graceful: saves bot state, flushes DB backup)"
    sudo systemctl stop "$SERVICE" || { err "could not stop $SERVICE"; return 1; }

    backup_db

    if [ "$migrate" = "skip-migrate" ]; then
        log "Skipping migrations (rollback: schema is forward-only/additive)"
    else
        log "Applying database migrations (idempotent)"
        ( cd backend && "$VENV_PY" migrations/run_migrations.py ) || { err "migrations failed"; return 1; }
    fi

    log "Starting $SERVICE"
    sudo systemctl start "$SERVICE" || { err "could not start $SERVICE"; return 1; }

    return 0
}

# Last-resort safety net: ensure SOMETHING is serving on whatever code is
# currently checked out, so a failed deploy+rollback never leaves the box with a
# stopped service. ExecStartPre validation still guards against a broken env.
ensure_service_running() {
    log "Safety net: ensuring $SERVICE is started"
    sudo systemctl start "$SERVICE" || err "safety-net start failed"
}

# Poll the health endpoint until it returns success or the window elapses.
health_ok() {
    local tries=30
    log "Health check: $HEALTH_URL (up to $((tries * 2))s)"
    for _ in $(seq 1 "$tries"); do
        if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# --- record rollback target ------------------------------------------------
git fetch --prune origin || { err "git fetch failed; aborting before any change"; exit 2; }
PREV_SHA="$(git rev-parse HEAD)"
log "Live commit before deploy (rollback target): $(git rev-parse --short "$PREV_SHA")"

# --- deploy ----------------------------------------------------------------
log "Deploying ref: $DEPLOY_REF"
deploy_ok=0
if build_and_restart "$DEPLOY_REF" migrate && health_ok; then
    deploy_ok=1
fi

if [ "$deploy_ok" = 1 ]; then
    log "Deploy healthy at $(git rev-parse --short HEAD). Done."
    exit 0
fi

# --- rollback --------------------------------------------------------------
# Code-only rollback: revert to the previously-live commit and restart WITHOUT
# re-running migrations (additive schema is forward-compatible with older code).
err "Deploy failed or unhealthy. Rolling code back to $(git rev-parse --short "$PREV_SHA")."
if build_and_restart "$PREV_SHA" skip-migrate && health_ok; then
    err "Rolled back to $(git rev-parse --short "$PREV_SHA") and service is healthy. DEPLOY FAILED (rolled back)."
    exit 1
fi

# Rollback build/restart did not complete cleanly (e.g. a rebuild step failed).
# Make a last-resort attempt to bring the service up on whatever is checked out,
# so we never exit leaving a stopped service.
err "Rollback did not become healthy on the first attempt; trying safety-net restart."
ensure_service_running
if health_ok; then
    err "Safety-net restart healthy on $(git rev-parse --short HEAD). DEPLOY FAILED (rolled back; manual review advised)."
    exit 1
fi

err "ROLLBACK did not become healthy. MANUAL INTERVENTION REQUIRED."
err "Inspect logs: journalctl -u $SERVICE -n 100 --no-pager"
exit 2
