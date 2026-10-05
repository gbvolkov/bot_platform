# cheetan deployment

The host has other applications on ports 8000 and 8080. Use both Compose files:

```sh
docker compose --env-file /home/volkov/bot-platform-deploy/configuration/deployment.env \
  -f docker-compose.yml -f deploy/cheetan.compose.yml config --quiet
docker compose --env-file /home/volkov/bot-platform-deploy/configuration/deployment.env \
  -f docker-compose.yml -f deploy/cheetan.compose.yml up -d --no-build
```

Private configuration, persistent state, and release source snapshots are under
`/home/volkov/bot-platform-deploy`. Credentials stay on the server in files with
mode 0600. The `backup-path` file records the verified pre-deployment backup.
Do not print the resolved Compose configuration: it contains worker credentials.

## nginx and access

The existing `/etc/nginx/sites-available/agents.gbvolkoff.name` routes `/` and
`/webchat` to localhost:5173 and `/api/` to localhost:8009. Those ports belong to
the separate GWP Chat frontend/backend. Preserve its login system, SQLite history,
attachments, and search index. The backend talks to `openai_proxy:8084` internally.
Build the existing GUI commit separately using the files under `deploy/gui`; its
local backend `uv.lock` must accompany the source snapshot. Export that lock
without resolving upgrades to `backend/requirements.deploy.lock` before building.
Copy `vite.deploy.config.ts`, `nginx.deploy.conf`, and `dockerignore` into the GUI
build context (the last as `.dockerignore`). The frontend build uses `/api` on
the current origin, and produces static assets rather than running Vite in service.

The host nginx already disables proxy buffering. OpenAI streaming retains the
legacy periodic heartbeat comments so quiet tools do not hit proxy idle limits.
The existing host nginx configuration can remain unchanged for this cutover.

| Loopback port | Service |
|---|---|
| 5173 | Existing GWP Chat, served as static production assets |
| 8009 | Existing GWP Chat API and authentication |
| 8014 | Native platform API and operational readiness |
| 8015 | Bundled lightweight web chat |
| 8084 | OpenAI compatibility API |

The coordinator, Redis notifications, retrieval API, and worker services have no
published host ports. The host's unrelated nginx sites and Redis instance are
outside the cleanup scope.

## Cleanup and state

The prior platform processes were already stopped at inspection. Its Redis queue
and active-job set were empty. SQLite backup and integrity checks covered the
application database, graph checkpoints, retrieval database, and GUI database.
A migration rehearsal on an independent copy preserved all 281 conversations
and 3,718 messages and assigned their original legacy runtime.

The owner selected archived history with new conversations on workers. The 281
legacy conversations were explicitly closed in the deployment copy; all 3,718
messages were verified unchanged. Closed legacy history is read from SQLite
without contacting the retired runtime. Original checkpoints remain unchanged.
The previous platform and GUI checkouts, environments, and data were moved to
`/home/volkov/bot-platform-backups/20261005-080455/previous-installation` before
starting the replacement. Their obsolete webchat PM2 startup entries were backed
up and removed. Shared services and Docker resources were left intact.

To roll back this archive cutover, first drain and stop only this Compose project.
Preserve the new state separately; restore the archived installation directories
to their original paths and the saved PM2 entries if needed. The untouched
pre-cutover databases are the rollback source. Do not copy an old database over
newly accepted conversations or reset checkpoints as part of rollback.

## Verification

Check the native `/healthz` and `/readyz`, the OpenAI model listing, GUI health,
the login page through host nginx, and the static assets. Inspect plugin-specific
initialization errors separately from application readiness. Run the deterministic
suite inside the worker image and installed-package checks inside access/retrieval
images. Do not replay historical side-effecting tool calls as a deployment check.
