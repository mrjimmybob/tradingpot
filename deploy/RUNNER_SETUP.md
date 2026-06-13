# Self-hosted GitHub Actions runner (homelab CD)

This sets up **continuous deployment** for the tradingbot: push to `main` →
tests run on GitHub-hosted runners → on success, a runner **on the Debian
server** pulls the new code, rebuilds, migrates, restarts the `tradingbot`
systemd service, health-checks it, and rolls back if it’s unhealthy.

```
 dev laptop ──push main──> GitHub ──CI (github-hosted)──> tests pass
                                          │
                                          └─► self-hosted runner (the server)
                                                runs deploy/deploy.sh:
                                                fetch → build → migrate → restart → health → rollback?
```

## ⚠️ Security — this repository is PUBLIC

A self-hosted runner executes whatever a workflow tells it to, on your home
network. For a public repo that is dangerous **unless** untrusted code can
never reach the runner. This setup is safe because:

- `deploy.yml` triggers **only** on `push` to `main` and manual dispatch —
  **never** `pull_request` / `pull_request_target`. Fork PRs cannot run on the
  runner. **Do not add those triggers.**
- The runner’s `deploy` job is gated behind a GitHub-hosted `test` job and an
  `if: github.ref == 'refs/heads/main'` guard.
- All tests for PRs run via `ci.yml` on **GitHub-hosted** runners (isolated,
  no secrets, no LAN access).

Recommended hardening in repo settings → Actions:

- **Settings → Actions → General → Fork pull request workflows**: keep
  “Require approval for all external contributors” (the default) so no PR
  workflow runs without a maintainer’s click. (CI is GitHub-hosted, so even an
  approved fork PR never touches the runner — but this is good hygiene.)
- Consider a branch protection rule on `main` requiring the CI checks to pass.

## Prerequisites on the server

The bot must already be installed and running via the normal runbook
(`deploy/install.sh` → systemd `tradingbot.service`, repo at
`/opt/tradingbot`, venv at `/opt/tradingbot/venv`, owned by the `tradingbot`
user). See `DEPLOYMENT.md`. The CD runner only takes over *updates*; the first
install is still manual.

`node`/`npm`, `python3-venv`, and `git` are already present from that install.

## 1. Let the `tradingbot` user restart the service (narrow sudoers)

The runner will run as the **`tradingbot`** user (it already owns
`/opt/tradingbot`, so it can `git`/`pip`/`npm` in place). It only needs
elevated rights for the three systemctl verbs `deploy.sh` uses — nothing else:

```bash
sudo tee /etc/sudoers.d/tradingbot-deploy >/dev/null <<'EOF'
tradingbot ALL=(root) NOPASSWD: /bin/systemctl start tradingbot, /bin/systemctl stop tradingbot, /bin/systemctl restart tradingbot
EOF
sudo visudo -c            # validate syntax
```

> If `systemctl` lives at `/usr/bin/systemctl` on your box (`command -v
> systemctl`), use that path instead.

## 2. Install the runner as the `tradingbot` user

Get the registration token from GitHub: **repo → Settings → Actions → Runners
→ New self-hosted runner** (Linux x64). Then, as a sudo-capable user:

```bash
# Dedicated install dir owned by the service user
sudo mkdir -p /opt/actions-runner
sudo chown tradingbot:tradingbot /opt/actions-runner
cd /opt/actions-runner

# Download (use the version/URL GitHub shows on the "New runner" page)
RUNNER_VER=2.319.1
sudo -u tradingbot curl -o actions-runner.tar.gz -L \
  https://github.com/actions/runner/releases/download/v${RUNNER_VER}/actions-runner-linux-x64-${RUNNER_VER}.tar.gz
sudo -u tradingbot tar xzf actions-runner.tar.gz && rm actions-runner.tar.gz

# Register against THIS repo, with the label the workflow targets: `tradingbot`
sudo -u tradingbot ./config.sh \
  --url https://github.com/mrjimmybob/tradingpot \
  --token <REGISTRATION_TOKEN> \
  --name homelab-debian \
  --labels tradingbot \
  --unattended --replace
```

`sudo -u tradingbot ...` works even though that user has a `nologin` shell —
`sudo` runs the command directly, not a login shell.

## 3. Run the runner as a service

```bash
cd /opt/actions-runner
sudo ./svc.sh install tradingbot     # run the service AS the tradingbot user
sudo ./svc.sh start
sudo ./svc.sh status
```

The runner now appears **Idle** under repo → Settings → Actions → Runners,
with the `tradingbot` label.

## 4. First deploy

`deploy.yml` matches `runs-on: [self-hosted, tradingbot]`. Trigger it by:

- pushing to `main`, or
- **Actions → Deploy → Run workflow** (manual dispatch).

Watch it in the Actions tab; on the server, `journalctl -u tradingbot -f`.

`deploy/deploy.sh` runs from the on-disk copy at `/opt/tradingbot`. The very
first CD run uses the `deploy.sh` already present from your manual install;
each run updates it for the next. To dry-run the logic by hand:

```bash
sudo -u tradingbot DEPLOY_REF=origin/main /opt/tradingbot/deploy/deploy.sh
```

## What a deploy does (and how rollback works)

`deploy/deploy.sh`, in order:

1. `git fetch` and record the **current** commit as the rollback target.
2. `git reset --hard` to the pushed commit.
3. `pip install -r backend/requirements.txt`, `npm ci && npm run build`.
4. `validate_deployment.sh --deep` (env can actually run uvicorn + import app).
5. `systemctl stop` → back up the SQLite DB (+ `-wal`/`-shm`) to
   `backend/backups/predeploy-<ts>.db` → run idempotent migrations → `start`.
6. Poll `http://127.0.0.1:8000/api/health` for up to ~60s.
7. **If unhealthy or any step failed:** reset code to the previous commit,
   rebuild, restart, re-check. The server ends on the last known-good build.
   Migrations are additive/idempotent and are **not** reverted (reverting could
   drop real ledger rows; old code tolerates the newer schema). If even the
   rollback is unhealthy, the job exits non-zero and asks for manual help.

Override defaults via env (`TRADINGBOT_ROOT`, `TRADINGBOT_SERVICE`,
`TRADINGBOT_HEALTH_URL`, `DEPLOY_REF`) — the workflow sets `TRADINGBOT_ROOT`
and `DEPLOY_REF` for you.

## Going live later

During the dry run, fully automatic deploy on every push to `main` is fine. For
live trading with real capital, add a manual approval gate so a human confirms
each production deploy: create a GitHub **Environment** (e.g. `production`) with
required reviewers, then add `environment: production` to the `deploy` job in
`deploy.yml`. The deploy will then pause for approval before restarting the bot.

## Maintenance

```bash
# Update the runner binary: GitHub auto-updates it; or re-run config with --replace.
# Remove the runner:
cd /opt/actions-runner
sudo ./svc.sh stop && sudo ./svc.sh uninstall
sudo -u tradingbot ./config.sh remove --token <REMOVAL_TOKEN>
```
