#!/usr/bin/env bash
#
# deploy/tests/test_deploy_rollback.sh — orchestration tests for deploy.sh.
#
# deploy.sh shells out to git/systemctl/npm/curl/the venv python. We can't run
# the real ones in CI, so we put stubs on PATH (and a fake venv) that record
# what deploy.sh did and let us inject failures. This validates the deploy
# ORCHESTRATION: successful deploy, failed migration, automatic rollback,
# post-rollback health, and the safety-net restart — without a real server.
#
# Run: bash deploy/tests/test_deploy_rollback.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_SH="$REPO_ROOT/deploy/deploy.sh"

PASS=0; FAIL=0
ok()  { echo "    ok: $1"; PASS=$((PASS + 1)); }
bad() { echo "    FAIL: $1"; FAIL=$((FAIL + 1)); }
expect_eq() { [ "$1" = "$2" ] && ok "$3 ($1)" || bad "$3 (expected '$2', got '$1')"; }

# Build an isolated fake deploy root + stub PATH. Echoes the work dir.
make_sandbox() {
    local work; work="$(mktemp -d)"
    local state="$work/state" bin="$work/bin" root="$work/root"
    mkdir -p "$state" "$bin" "$root/venv/bin" "$root/frontend" \
             "$root/backend/migrations" "$root/deploy"
    cp "$DEPLOY_SH" "$root/deploy/deploy.sh"

    echo started > "$state/service"     # service is running before deploy
    echo OLDSHA0 > "$state/sha"         # currently-live commit
    echo 0       > "$state/migrate_calls"
    echo 0       > "$state/start_calls"

    cat > "$bin/sudo" <<'EOF'
#!/usr/bin/env bash
exec "$@"
EOF
    cat > "$bin/git" <<'EOF'
#!/usr/bin/env bash
S="$STATE_DIR"
case "$1" in
  fetch) exit 0 ;;
  log)   echo "stub commit subject"; exit 0 ;;
  rev-parse)
    if [ "${2:-}" = "--short" ]; then
      ref="${3:-HEAD}"; [ "$ref" = "HEAD" ] && ref="$(cat "$S/sha")"; echo "${ref:0:7}"
    else
      cat "$S/sha"
    fi ;;
  reset)  # reset --hard <ref>
    [ -n "${FAIL_RESET:-}" ] && { echo "git reset failed" >&2; exit 1; }
    echo "$3" > "$S/sha"; exit 0 ;;
  *) exit 0 ;;
esac
EOF
    cat > "$bin/systemctl" <<'EOF'
#!/usr/bin/env bash
S="$STATE_DIR"
case "$1" in
  stop)  echo stopped > "$S/service"; exit 0 ;;
  start)
    echo $(( $(cat "$S/start_calls") + 1 )) > "$S/start_calls"
    [ -n "${FAIL_START:-}" ] && exit 1
    echo started > "$S/service"; exit 0 ;;
  *) exit 0 ;;
esac
EOF
    cat > "$bin/npm" <<'EOF'
#!/usr/bin/env bash
# Optionally fail the rebuild for a specific checked-out sha (models a rollback
# whose build step fails), else succeed.
[ -n "${FAIL_NPM_ON_SHA:-}" ] && [ "$(cat "$STATE_DIR/sha")" = "$FAIL_NPM_ON_SHA" ] && exit 1
exit 0
EOF
    cat > "$bin/curl" <<'EOF'
#!/usr/bin/env bash
# Health endpoint: healthy iff the service is "started".
[ "$(cat "$STATE_DIR/service")" = started ] && exit 0
exit 22
EOF
    cat > "$bin/sleep" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    cat > "$root/venv/bin/python" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in
    *run_migrations.py)
      echo $(( $(cat "$STATE_DIR/migrate_calls") + 1 )) > "$STATE_DIR/migrate_calls"
      exit "${FAIL_MIGRATE:-0}" ;;
  esac
done
exit 0
EOF
    cat > "$root/venv/bin/pip" <<'EOF'
#!/usr/bin/env bash
[ -n "${FAIL_PIP:-}" ] && exit 1
exit 0
EOF
    cat > "$root/deploy/validate_deployment.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
    chmod +x "$bin"/* "$root/venv/bin"/* "$root/deploy/validate_deployment.sh"
    echo "$work"
}

# run_deploy <workdir> [extra env KEY=VAL ...] -> echoes exit code
# Inherit the real environment (so bash/coreutils resolve) but PREPEND the stub
# dir so git/systemctl/npm/curl/sleep resolve to our stubs.
run_deploy() {
    local work="$1"; shift
    (
        export PATH="$work/bin:$PATH"
        export STATE_DIR="$work/state"
        export TRADINGBOT_ROOT="$work/root"
        export TRADINGBOT_SERVICE="tradingbot"
        export TRADINGBOT_HEALTH_URL="http://127.0.0.1:8000/api/health"
        export DEPLOY_REF="NEWSHA1"
        local kv
        for kv in "$@"; do export "${kv?}"; done
        bash "$work/root/deploy/deploy.sh" >"$work/out.log" 2>&1
    )
    echo $?
}

echo "== Scenario A: successful deployment =="
W="$(make_sandbox)"
rc="$(run_deploy "$W")"
expect_eq "$rc" "0" "exit code"
expect_eq "$(cat "$W/state/service")" "started" "service running"
expect_eq "$(cat "$W/state/sha")" "NEWSHA1" "checked out new commit"
expect_eq "$(cat "$W/state/migrate_calls")" "1" "migrations ran once"
rm -rf "$W"

echo "== Scenario B: failed migration -> automatic rollback -> healthy =="
W="$(make_sandbox)"
rc="$(run_deploy "$W" FAIL_MIGRATE=1)"
expect_eq "$rc" "1" "exit code (rolled back)"
expect_eq "$(cat "$W/state/service")" "started" "service brought back up, not left stopped"
expect_eq "$(cat "$W/state/sha")" "OLDSHA0" "rolled back to previous commit"
expect_eq "$(cat "$W/state/migrate_calls")" "1" "migration NOT re-run on rollback (skip-migrate)"
rm -rf "$W"

echo "== Scenario C: migration fails (svc stopped) AND rollback build fails -> safety net =="
W="$(make_sandbox)"
rc="$(run_deploy "$W" FAIL_MIGRATE=1 FAIL_NPM_ON_SHA=OLDSHA0)"
expect_eq "$rc" "1" "exit code (recovered via safety net)"
expect_eq "$(cat "$W/state/service")" "started" "safety net restarted the stopped service"
rm -rf "$W"

echo "== Scenario D: total failure (migration + every start fails) -> exit 2 =="
W="$(make_sandbox)"
rc="$(run_deploy "$W" FAIL_MIGRATE=1 FAIL_START=1)"
expect_eq "$rc" "2" "exit code (manual intervention)"
attempts="$(cat "$W/state/start_calls")"
[ "$attempts" -ge 2 ] && ok "safety-net start attempted (start_calls=$attempts)" \
                       || bad "expected >=2 start attempts, got $attempts"
rm -rf "$W"

echo
echo "Deploy orchestration tests: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
