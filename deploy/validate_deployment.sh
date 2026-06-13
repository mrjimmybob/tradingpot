#!/bin/bash
#
# Deployment validation for the tradingbot backend.
#
# Fails (non-zero) early and loudly if the installed environment cannot
# actually run the systemd service. Designed to be used three ways:
#   * by hand                : deploy/validate_deployment.sh
#   * from install.sh        : as a gate BEFORE enabling the service
#   * as systemd ExecStartPre: so a broken env fails with a clear journal
#                              message instead of an opaque status=203/EXEC
#
# Checks performed:
#   1. the virtual environment exists and has a working interpreter + pip
#   2. uvicorn (the ExecStart target) exists and is executable
#   3. all required Python packages are importable
#   4. (only with --unit) the ExecStart path inside a systemd unit exists
#   5. (only with --deep) the FastAPI app object imports
#
# Usage:
#   validate_deployment.sh [--root DIR] [--unit FILE] [--deep] [--quiet]
#
set -u

QUIET=0
DEEP=0
ROOT=""
UNIT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --quiet) QUIET=1 ;;
        --deep)  DEEP=1 ;;
        --root)  ROOT="${2:-}"; shift ;;
        --unit)  UNIT="${2:-}"; shift ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# Resolve the repo root from this script's own location (deploy/ lives under
# the repo root), so the same script works no matter where the repo is cloned.
SELF="${BASH_SOURCE[0]}"
command -v realpath >/dev/null 2>&1 && SELF="$(realpath "$SELF")"
DEPLOY_DIR="$(cd "$(dirname "$SELF")" && pwd)"
[ -n "$ROOT" ] || ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"

VENV_DIR="$ROOT/venv"
if [ -d "$VENV_DIR/bin" ]; then VENV_BIN="$VENV_DIR/bin"
elif [ -d "$VENV_DIR/Scripts" ]; then VENV_BIN="$VENV_DIR/Scripts"
else VENV_BIN="$VENV_DIR/bin"; fi
PYTHON="$VENV_BIN/python"
[ -x "$PYTHON" ] || { [ -x "$PYTHON.exe" ] && PYTHON="$PYTHON.exe"; }

FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }
ok()   { [ "$QUIET" = 1 ] || echo "ok:   $*"; }

# 1. Virtual environment ----------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    fail "virtualenv missing at $VENV_DIR (run ./init.sh)"
    PYTHON=""   # nothing else that needs the interpreter can run
elif [ ! -x "$PYTHON" ]; then
    fail "virtualenv interpreter missing at $VENV_BIN/python"
    PYTHON=""
elif ! "$PYTHON" -c 'import pip' >/dev/null 2>&1; then
    fail "virtualenv at $VENV_DIR has no working pip"
else
    ok "virtualenv present at $VENV_DIR"
fi

# 2. uvicorn binary (the ExecStart target) ----------------------------------
if [ -x "$VENV_BIN/uvicorn" ] || [ -f "$VENV_BIN/uvicorn.exe" ]; then
    ok "uvicorn present at $VENV_BIN/uvicorn"
else
    fail "uvicorn missing at $VENV_BIN/uvicorn (pip install -r backend/requirements.txt)"
fi

# 3. Required Python packages importable ------------------------------------
REQUIRED_IMPORTS="fastapi uvicorn sqlalchemy aiosqlite ccxt yaml pydantic pydantic_settings aiohttp websockets aiosmtplib httpx dateutil"
if [ -n "$PYTHON" ]; then
    missing="$("$PYTHON" - "$REQUIRED_IMPORTS" <<'PY' 2>/dev/null
import importlib.util, sys
mods = sys.argv[1].split()
print(" ".join(m for m in mods if importlib.util.find_spec(m) is None))
PY
)"
    if [ -n "${missing// /}" ]; then
        fail "required Python packages not importable: $missing"
    else
        ok "all required Python packages importable"
    fi
fi

# 4. systemd ExecStart target exists (opt-in via --unit) --------------------
if [ -n "$UNIT" ]; then
    if [ ! -f "$UNIT" ]; then
        fail "systemd unit not found at $UNIT"
    else
        exec_line="$(grep -E '^[[:space:]]*ExecStart=' "$UNIT" | head -1 | sed 's/^[[:space:]]*ExecStart=//')"
        exec_bin="${exec_line%% *}"   # first whitespace-delimited token
        exec_bin="${exec_bin#@}"       # strip systemd '@' argv0 prefix if present
        if [ -z "$exec_bin" ]; then
            fail "could not parse ExecStart from $UNIT"
        elif [ -x "$exec_bin" ] || [ -f "$exec_bin" ]; then
            ok "ExecStart target exists: $exec_bin"
        else
            fail "ExecStart target does not exist: $exec_bin (unit=$UNIT)"
        fi
    fi
fi

# 5. Deep check: the app actually imports (opt-in via --deep) ---------------
if [ "$DEEP" = 1 ] && [ -n "$PYTHON" ]; then
    if ( cd "$ROOT/backend" && "$PYTHON" -c 'import app.main' >/dev/null 2>&1 ); then
        ok "backend app.main imports"
    else
        fail "backend app.main failed to import (cd backend && python -c 'import app.main')"
    fi
fi

if [ "$FAILED" = 1 ]; then
    echo "Deployment validation FAILED." >&2
    exit 1
fi
[ "$QUIET" = 1 ] || echo "Deployment validation passed."
exit 0
