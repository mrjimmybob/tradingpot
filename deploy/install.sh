#!/bin/bash
#
# Reproducible, end-to-end deployment for a clean Debian host.
#
# Idempotent and safe to re-run. The key property: the deployment is
# VALIDATED before the systemd service is ever enabled, so a broken backend
# environment fails here with a clear message instead of as status=203/EXEC.
#
# It also pins the installed unit to THIS install's path, so cloning somewhere
# other than /opt/tradingbot needs no manual editing of the unit file.
#
# Usage (run from the repo, as a user with sudo):
#   ./deploy/install.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="tradingbot"
SERVICE_USER="tradingbot"
UNIT_TEMPLATE="$SCRIPT_DIR/${SERVICE_NAME}.service"
UNIT_DST="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_TEMPLATE="$SCRIPT_DIR/${SERVICE_NAME}.env.example"
ENV_DST="$SCRIPT_DIR/${SERVICE_NAME}.env"

log() { echo -e "\n=== $* ==="; }

# 0. System prerequisites ---------------------------------------------------
log "Installing system prerequisites (apt)"
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip nodejs npm

# 1. Backend venv + frontend deps + config templates ------------------------
# init.sh is now self-verifying: it aborts if the venv or uvicorn is missing.
log "Building backend/frontend environment (init.sh)"
"$ROOT/init.sh"

# 2. Build the web UI -------------------------------------------------------
log "Building web UI (npm run build)"
( cd "$ROOT/frontend" && npm run build )

# 3. Service env file -------------------------------------------------------
if [ ! -f "$ENV_DST" ]; then
    cp "$ENV_TEMPLATE" "$ENV_DST"
    log "Created $ENV_DST from template"
fi

# 4. Generate the unit pinned to THIS install's path ------------------------
# The committed template uses /opt/tradingbot; rewrite it to the real root so
# WorkingDirectory / EnvironmentFile / ExecStart / ExecStartPre all line up.
log "Rendering systemd unit for root: $ROOT"
RENDERED_UNIT="$(mktemp)"
trap 'rm -f "$RENDERED_UNIT"' EXIT
sed "s|/opt/tradingbot|$ROOT|g" "$UNIT_TEMPLATE" > "$RENDERED_UNIT"

# 5. VALIDATE before enabling anything --------------------------------------
# --unit points at the rendered unit so we confirm its ExecStart target
# (the venv's uvicorn) actually exists on disk.
log "Validating deployment (fails here if anything is missing)"
"$SCRIPT_DIR/validate_deployment.sh" --deep --root "$ROOT" --unit "$RENDERED_UNIT"

# 6. Service user + ownership ----------------------------------------------
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    log "Creating service user: $SERVICE_USER"
    sudo useradd --system --home "$ROOT" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
log "Setting ownership to $SERVICE_USER"
sudo chown -R "$SERVICE_USER":"$SERVICE_USER" "$ROOT"

# 7. Install + enable the (validated) service -------------------------------
log "Installing systemd unit to $UNIT_DST"
sudo cp "$RENDERED_UNIT" "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

# 8. Report -----------------------------------------------------------------
log "Service status"
systemctl --no-pager --full status "$SERVICE_NAME" || true

echo
echo "Deployment complete. Tail logs with:  journalctl -u $SERVICE_NAME -f"
echo "Reach the UI/API via an SSH tunnel:   ssh -L 8000:127.0.0.1:8000 <user>@<server>"
