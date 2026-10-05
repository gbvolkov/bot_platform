# Separated runtime: deployment and operation

This release introduces a new public entry point, `platform_access.main:app`.
The existing `bot_service.main:app` remains the legacy host. Do not replace or
restart a live legacy process while it owns conversations or privacy mappings.
The two iSmart algorithms remain distinct; no prompts or generation algorithms
are merged. Local `docs/ismart` assets remain ignored and are mounted separately.

## Components and review order

| Work package | Implementation | Verification |
|---|---|---|
| 1. Baseline | Root `uv.lock`, vendored fork, `config_defaults`, component dependency locks | Frozen lock check; clean access installation |
| 2. Completion delivery | Legacy Redis terminal reconciliation and subscription cleanup | `test_task_queue_delivery.py` |
| 3. Contracts/catalog | `platform_contracts`, `platform_sdk`; metadata never loads implementations | Catalog, dependency and installed-package checks |
| 4. Runtime/pilots | `agent_runtime`; existing factory/graph/compiler adapters; explicit pilot model factories | Pilot graph/CLI tests and runtime conformance |
| 5. Capabilities | `platform_capabilities`; existing tool/guardrail ownership; canonical GAZ package; Mycroft resolver injection | Retrieval, GAZ, guardrail and Mycroft tests |
| 6. Durable application | `platform_application`, SQLite migrations and repositories | Atomic acceptance, ordering, deduplication, projection, recovery tests |
| 7. Workers | Private `platform_access.coordinator`; direct interactive/batch execution | Lifecycle, checkpoint, cancellation, filesystem isolation tests |
| 8. Agents/clients | All 17 descriptors; REST/OpenAI/web clients; local CLI composition; separate iSmart adapters | Adapter conformance and existing business-graph/CLI tests |
| 9. Packaging/rollout | Component wheels, immutable build stages, Compose, operations and build checks | Installed-package checks, Compose validation, Linux image build gate |

Compatibility imports are deliberately retained for old and inactive consumers.
Application code imports contracts and repository/notification protocols. Access
processes install without model or agent packages. Capability code and configured
platform tools cannot import business agents. Plugins may depend on capabilities;
Mycroft resolves child graphs through the runtime without acquiring another root
run slot. Its tracked scenario skills stay under `agents/mycroft_agent/scenarios`.

The old manifest silently discarded several constructor settings. Their exact
values are recorded in `config_defaults/removed_ignored_settings.json`; the new
defaults omit only those unused settings. New unsupported constructor arguments
raise validation errors, including factories with a generic `**kwargs` parameter.

## Reproducible development

Use Python 3.13. From a clean checkout:

```text
uv sync --frozen
uv pip install --no-deps ./packages/gaz_index_builder
uv run --no-sync python scripts/verify_platform.py
uv lock --check
```

`uv.lock` preserves the resolved baseline. `vendor/llm-guard` is the clean local
fork at `31d7a16964ba68bae6e00650c96d32538b1fe764`, with its source and MIT license.
No sibling checkout is required. `scripts/package_platform.py` derives component
metadata and dependency subsets from that lock without resolving new versions.
Do not mix dependency upgrades with this extraction.

The deterministic runner excludes the removed inactive iSmart MVP, the real
privacy-model smoke test, and the older generator's tests requiring missing local
course/prompt assets. The latter exclusion was requested by the user. When those
assets are available, add `--with-local-ismart-assets`. Do not synthesize replacement
prompts or commit the ignored documents. Ordinary tests use fake agents/models;
historical side-effecting tools are never replayed.

For an access-only environment, install `deploy/requirements.access.lock`, then
the contracts, SDK, application, infrastructure, client, access, proxy and web-chat
wheels with `--no-deps`. Run `scripts/check_component_packages.py access` with
`python -I` from outside the checkout. It asserts that agents and model libraries
are absent. The analogous `capabilities` check loads the retrieval services from
installed wheels without business-agent packages. Service-specific data, model
configuration and credentials are still supplied externally.

## Persistent state and configuration

Copy `deploy/platform.env.example` to an ignored environment file and set the real
paths. SQLite is authoritative. The application database must be the *same file*
used by the legacy host, including its containing directory so WAL/SHM files are
shared. Do not substitute an empty database. The checkpoint path and thread IDs
remain unchanged. The retrieval database remains separate.

| Storage | Mount/configuration | Owner |
|---|---|---|
| Application history, assignments, runs, attempts, events, artifact metadata | `PLATFORM_STATE_DIR` + `PLATFORM_DATABASE_FILENAME` | Public application/private coordinator |
| LangGraph checkpoints | `PLATFORM_CHECKPOINT_DIR` + existing filename | Runtime workers and unchanged legacy host |
| Original and generated attachment payloads | Compose `artifacts` volume | Access/storage adapter |
| Worker-local parsed/raw attachments | Compose `runtime-artifacts` volume | Runtime |
| Retrieval metadata | `PLATFORM_RETRIEVAL_STATE_DIR` | Retrieval API/worker |
| Product data and indexes | `PLATFORM_DATA_DIR` | Capability services/workers |
| Local iSmart reference documents | `PLATFORM_ISMART_DOCS_DIR`, read only | Both iSmart implementations |
| New hosted iSmart generated output | Compose `generated-output` volume | Batch worker |
| Logs | Compose `logs` volume | Workers/capabilities |

Existing legacy directories and environment remain untouched. If the old host has
different settings from the sanitized manifest, review and supply an explicit
`PLATFORM_CATALOG_PATH` to **both** access and workers before assigning a cohort.
External model credentials stay in the worker env file. Public access processes
do not need model credentials. The coordinator requires `PLATFORM_SERVICE_TOKEN`,
has no published host port, and is reachable only on the private worker network.

```text
docker compose config --quiet
docker compose build
docker compose up -d
```

Images are built from checkout content and frozen dependency files. Startup never
pulls Git changes or installs dependencies. Redis stores advisory wake-ups only;
its data may be discarded. Workers poll the database through the coordinator even
when Redis is unavailable. Retrieval services have a separate image/dependency
set and install without `agents`.
The worker image includes Python 3.12 for iSmart's existing generated-code checks,
while platform services and dependencies continue to use Python 3.13.

## Migration, cohorts and rollback

1. Back up the application and checkpoint databases consistently and rehearse on
   copied fixtures. Additive application migrations never modify checkpoints.
2. Keep the original legacy checkout, dependency environment, process, paths and
   privacy sessions running. Route public traffic through the new access service;
   do not allow clients to bypass coordination and call legacy `/messages` directly.
3. Start the new services with `PLATFORM_WORKER_AGENT_IDS` empty. Migration assigns
   all existing conversations to `legacy`. The bridge serializes their calls while
   the old host remains the only writer of their message results.
   The bridge is initially gated by the `legacy-dispatch` Compose profile. Switch
   new submissions to the new access service, let previously accepted legacy
   requests and the old Redis queue drain, and keep the original conversation host
   alive. Only then run `docker compose --profile legacy-dispatch up -d legacy_bridge`.
   Work accepted by the new API waits durably during this barrier. This prevents an
   old in-flight turn from overlapping the first coordinated turn. Retain the old
   Redis/worker processes until their accepted jobs finish.
4. Check `/readyz` and enable `simple_agent,simple_agent_en`, then
   `artifact_creator_agent`, then the remaining agent cohorts after their fixtures
   and deployment initialization pass. `*` enables all active IDs for **new**
   conversations. Existing assignments do not change when this setting changes.
5. Keep workers for older revisions available while their conversations remain
   open. Descriptors and claims carry an implementation revision; catalog updates
   do not make a compatible pinned session appear unready. Bump revisions whenever
   an implementation or its effective settings change.
6. Close idle conversations explicitly with the close endpoint. Idle time never
   moves a conversation, resets checkpoints, or releases privacy affinity.

Rollback stops new assignments to an affected cohort and preserves existing
assignments. Keep the compatible image/process serving those assignments; never
route them to a different revision as an automatic fallback. Additive tables remain
in place. The legacy host can be retired only after its final conversation closes.

## Run API and semantics

Existing message routes still wait or stream; native synchronous responses retain
HTTP 201 and their three fields, and SSE retains conversation IDs and terminal
payloads. OpenAI listings include ready models only and retain the existing prompt,
history and usage interpretation. The proxy's request-wait timeout is unchanged.

| Endpoint | Operation |
|---|---|
| `POST /api/conversations/{id}/runs` | Persist accepted input and run atomically; return HTTP 202 and run ID |
| `GET /api/runs/{id}` | Durable status, outcome, error and artifact references |
| `GET /api/runs/{id}/events?after=N` | Replay/follow ordered durable events with SSE IDs |
| `POST /api/runs/{id}/resume` | `{ "interrupt_id": "...", "response": "..." }` |
| `POST /api/runs/{id}/cancel` | Cancel queued work or request cooperative stop |
| `POST /api/conversations/{id}/close` | Close an idle assignment |
| `GET /api/artifacts/{id}` | Retrieve the owner's original attachment payload |

Existing user identity/role headers apply. `Idempotency-Key` identifies a retry;
equal text is never treated as a duplicate. Same key/different input returns 409.
Each conversation has durable root-turn ordering. An interrupt blocks previously
queued turns; only a newly accepted post-interrupt message becomes the legacy
resume response. Explicit resume targets the interrupt, and duplicate responses
are rejected or return the same idempotent run.

Disconnecting only detaches the client. Queued cancellation is immediate. Existing
graphs can run synchronous tools on threads, so their safe cooperative cancellation
boundary is the end of the current root turn: the run remains running/cancelling
until execution stops. A plugin may explicitly support earlier asynchronous
cancellation. The unchanged legacy host cannot cooperatively stop a live call; its
actual completion is recorded. No execution deadline is introduced. Planned
shutdown drains work; a forced container kill requires recovery.

Input files cross process boundaries by artifact ID and are parsed on workers.
Generated attachments are uploaded by ID, with original wire payloads restored
by the access adapter. Hosted iSmart also exports generated packages as ZIP
artifacts without changing its local output/resume files or text response.

## Recovery and operations

An expired claim before execution starts returns to the durable queue. After start,
worker loss records `recovery_required` and never replays the run automatically.
Attempt IDs, generation IDs and lease checks fence stale events/completions. Failed
or cancelled execution blocks subsequent turns until checkpoint reconciliation.
Workers compare the current graph checkpoint to the last committed application
reference before invoking, preventing silent continuation after checkpoint/result
disagreement.

An operator must verify that the old execution has stopped, inspect its tools and
checkpoint, then call the authenticated private endpoint
`POST /runs/{run_id}/reconcile` with `checkpoint_reference` and `generation`.
An explicit JSON null means a verified empty checkpoint. This action releases the
queue; it does **not** replay the failed run. Submit any deliberate retry as a new
turn. If portable privacy mappings are unavailable after losing the owning process,
close the conversation: reconciliation rejects transfer to another generation.
Roots delegating to privacy-enabled children must also declare privacy affinity;
the runtime rejects a child resolution that would escape the root's affinity.

`/healthz` checks the access process. `/readyz` distinguishes application readiness
from individual plugin initialization failures and worker readiness. It exposes
queue age, active claims, recovery-required counts, worker generations/readiness,
initialization failures, artifact failures and the last event-delivery lag. Replay
deltas/custom events expire six hours after termination; outcomes, terminal events,
artifacts and final results remain with history.

## Release evidence and outstanding gates

Local verification on 2026-10-05: **622 passed, 7 skipped** in the deterministic
release suite. The seven skips are optional guardrail checks; the missing older
iSmart assets and live privacy-model smoke test are excluded as described above.
The frozen lock validates all 410 resolved packages without dependency upgrades.
All twelve component wheels build, and Compose configuration validates with the
gated legacy-dispatch profile.
Installed-package checks also pass from outside the checkout: API, OpenAI proxy,
and web chat import with no agent/model packages installed; capability/retrieval
packages import with business-agent and legacy API imports blocked.

Source-level tests cover deterministic runtime conformance for all 17 active IDs,
real graph interrupt/resume/checkpoint reconciliation, API/worker filesystem
separation, ordering, cancellation, missed notifications, late subscription and
legacy projection ownership. These adapter fixtures do not claim that each real
plugin has initialized against production credentials and datasets.

Before declaring the release deployed, build/run the Linux images on a host with
Docker available, rehearse migration/rollback against copied production state,
and confirm all real plugins' readiness and supported-channel integration in that
environment. This workstation has no running Docker engine, so image execution and
live session rollout have not been verified here. No production database or live
legacy process was changed during implementation.
