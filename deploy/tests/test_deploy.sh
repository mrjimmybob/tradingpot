#!/bin/bash
#
# Tests for the deployment hardening. Runs anywhere bash is available (no
# systemd / no real Debian required) by building throwaway fixtures and
# asserting that validate_deployment.sh accepts a good install and REJECTS
# each way the install can be broken — including the exact "uvicorn missing"
# failure that produced status=203/EXEC in production.
#
# Usage:  deploy/tests/test_deploy.sh
#
set -u

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$TESTS_DIR/.." && pwd)"
REPO_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
VALIDATE="$DEPLOY_DIR/validate_deployment.sh"

PASS=0
FAIL=0
note() { echo "  $*"; }
check() {  # check <description> <expected:pass|fail> <actual_rc> [output]
    local desc="$1" expect="$2" rc="$3" out="${4:-}"
    if { [ "$expect" = pass ] && [ "$rc" -eq 0 ]; } || { [ "$expect" = fail ] && [ "$rc" -ne 0 ]; }; then
        echo "PASS: $desc"; PASS=$((PASS+1))
    else
        echo "FAIL: $desc (expected $expect, rc=$rc)"; [ -n "$out" ] && echo "$out" | sed 's/^/      /'; FAIL=$((FAIL+1))
    fi
}
contains() {  # contains <description> <needle> <haystack>
    if printf '%s' "$3" | grep -qF "$2"; then
        echo "PASS: $1"; PASS=$((PASS+1))
    else
        echo "FAIL: $1 (missing \"$2\")"; FAIL=$((FAIL+1))
    fi
}

# --- 1. Syntax-check every shipped shell script ----------------------------
echo "== syntax checks =="
for f in "$REPO_DIR/init.sh" "$DEPLOY_DIR/validate_deployment.sh" "$DEPLOY_DIR/install.sh" "$TESTS_DIR/test_deploy.sh"; do
    bash -n "$f"; check "bash -n $(basename "$f")" pass "$?"
done

# --- 2. Static consistency between init.sh and the unit's ExecStart ---------
echo "== static consistency =="
grep -qE '^ExecStart=.*/venv/bin/uvicorn ' "$DEPLOY_DIR/tradingbot.service"
check "unit ExecStart uses venv/bin/uvicorn (the layout init.sh builds)" pass "$?"
grep -qE '^ExecStartPre=.*validate_deployment\.sh' "$DEPLOY_DIR/tradingbot.service"
check "unit has ExecStartPre validation gate" pass "$?"

# --- 3. Behavioural tests with throwaway fixtures --------------------------
echo "== behavioural validation =="
FIX="$(mktemp -d)"
trap 'rm -rf "$FIX"' EXIT

make_good_root() {  # make_good_root <dir>
    local root="$1" bin="$1/venv/bin"
    mkdir -p "$bin" "$root/backend/app"
    # stub interpreter: satisfies `import pip`, find_spec (prints nothing =>
    # no missing modules), and `import app.main`.
    cat > "$bin/python" <<'STUB'
#!/bin/bash
exit 0
STUB
    chmod +x "$bin/python"
    printf '#!/bin/bash\nexit 0\n' > "$bin/uvicorn"; chmod +x "$bin/uvicorn"
}

render_unit() {  # render_unit <root> <exec_target> -> prints unit path
    local root="$1" target="$2" unit="$1/unit.service"
    cat > "$unit" <<EOF
[Service]
ExecStart=$target app.main:app --host 127.0.0.1 --port 8000
EOF
    echo "$unit"
}

# 3a. fully good install -> PASS
GOOD="$FIX/good"; make_good_root "$GOOD"
UNIT="$(render_unit "$GOOD" "$GOOD/venv/bin/uvicorn")"
out="$(bash "$VALIDATE" --deep --root "$GOOD" --unit "$UNIT" 2>&1)"; rc=$?
check "good install validates" pass "$rc" "$out"

# 3b. uvicorn missing -> FAIL mentioning uvicorn (the production bug)
BADUVI="$FIX/baduvi"; make_good_root "$BADUVI"; rm -f "$BADUVI/venv/bin/uvicorn"
UNIT="$(render_unit "$BADUVI" "$BADUVI/venv/bin/uvicorn")"
out="$(bash "$VALIDATE" --root "$BADUVI" --unit "$UNIT" 2>&1)"; rc=$?
check "missing uvicorn is rejected" fail "$rc" "$out"
contains "  -> error names uvicorn" "uvicorn" "$out"

# 3c. ExecStart target points at a non-existent path -> FAIL
GHOST="$FIX/ghost"; make_good_root "$GHOST"
UNIT="$(render_unit "$GHOST" "$GHOST/venv/bin/does-not-exist")"
out="$(bash "$VALIDATE" --root "$GHOST" --unit "$UNIT" 2>&1)"; rc=$?
check "non-existent ExecStart target is rejected" fail "$rc" "$out"
contains "  -> error names ExecStart" "ExecStart target does not exist" "$out"

# 3d. no venv at all (clean machine, init.sh never ran) -> FAIL
EMPTY="$FIX/empty"; mkdir -p "$EMPTY"
out="$(bash "$VALIDATE" --root "$EMPTY" 2>&1)"; rc=$?
check "absent virtualenv is rejected" fail "$rc" "$out"
contains "  -> error names virtualenv" "virtualenv missing" "$out"

echo
echo "== summary: $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
