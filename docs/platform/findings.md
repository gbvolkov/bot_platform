# Repository Findings

Back to the [documentation index](index.md).

This page captures a repository survey of `C:\Projects\bot_platform` as of March 18, 2026. Runtime code and configuration remain the source of truth; this note summarizes the current execution path, the main subsystem boundaries, and the areas where documentation or repo shape can mislead future edits.

## Executive summary

- The current request path is `openai_proxy` -> Redis-backed `services/task_queue` -> `bot_service` -> agent implementation, with the worker also republishing streaming progress and terminal events.
- `bot_service` is the internal control plane: it initializes persistence, preloads active agents, exposes `/api/agents` and `/api/conversations`, and owns attachment normalization plus agent invocation.
- Agent registration is config-driven from `data/config/bot_service/load.json` through `bot_service/agent_registry.py`; startup behavior and API visibility depend on both config and async readiness.
- `services/kb_manager` is a real subsystem with its own FastAPI app and indexing service, not just a helper module.
- The repository is not source-only. It already contains local state such as `.env`, `.venv`, `.attachments_store`, logs, SQLite files, and data indexes, so bulk cleanup and broad file assumptions are risky.

## Runtime topology

1. `openai_proxy/main.py` accepts OpenAI-compatible requests, builds prompt payloads, hydrates attachments, creates or reuses conversations in `bot_service`, and enqueues jobs in Redis.
2. `services/task_queue/worker.py` dequeues jobs, calls `bot_service` over HTTP via `services/bot_client.py`, emits status, chunk, completion, interrupt, and failure events, and maintains worker heartbeats plus stale-job watchdog handling.
3. `bot_service/main.py` initializes database models, starts async agent preloading, and mounts the `/api` router from `bot_service/api/`.
4. `bot_service/service.py`, `bot_service/attachments.py`, `bot_service/db.py`, and `bot_service/models.py` cover conversation/message persistence, attachment storage and extraction, and agent execution.
5. `services/kb_manager/app.py` and `services/kb_manager/service.py` provide a separate knowledge-base service and document/indexing utilities that are also reused by `bot_service`.

## Main entry points

- API service: `uvicorn bot_service.main:app --reload --host 0.0.0.0 --port 8000`
- Background worker: `python -m services.task_queue.worker`
- OpenAI-compatible facade: `uvicorn openai_proxy.main:app --reload --host 0.0.0.0 --port 8080`
- Manual registry/debug client: `python chat_client.py list-agents` and `python chat_client.py chat <agent_id>`
- Local wrappers: `runbs.py` for `bot_service` and `runoai.py` for `openai_proxy`
- Legacy-looking root entry: `main.py` appears to be an older/demo runner rather than the current platform bootstrap

## Agent registry and configuration

- `bot_service/agent_registry.py` resolves `settings.agent_config_path` and loads agent definitions from `data/config/bot_service/load.json` rather than hard-coding the registry in Python.
- `agent_registry.preload_all()` is called during `bot_service` startup and asynchronously initializes every active agent.
- API visibility is readiness-based. `GET /api/agents/` can omit an agent that exists in `data/config/bot_service/load.json` if initialization has not finished yet.
- Attachment behavior is controlled by both `supported_content_types` and `allow_raw_attachments`; changes here affect whether files are converted to text or persisted under `.attachments_store` and passed through raw.
- Configuration is split between legacy root `config.py`, which reads `gv.env`, and newer service-local settings modules (`bot_service/config.py`, `openai_proxy/config.py`, `services/task_queue/config.py`), which read prefixed values from `.env`.

## Local development signals

- `README.md` expects Python 3.13, Redis 7+, MySQL or MariaDB, and environment setup in `.env` and `gv.env`.
- The repo uses `uv` via `pyproject.toml` and also pulls `palimpsest` and `rag-lib` directly from Git.
- The clearest manual checks surfaced in the inspected docs are:
  - `python chat_client.py list-agents`
  - `python chat_client.py chat <agent_id>`
  - `curl http://localhost:8080/v1/models`
  - `python -m py_compile openai_proxy/main.py services/task_queue/worker.py agents/ingos_product_agent/agent.py`
- The repository contains many simulation and helper scripts, but the inspected entry points did not surface a single canonical automated test suite.

## Documentation drift and edit risks

- Repository instructions still describe agent registration as edits in `bot_service/agent_registry.py`, but the live implementation is config-driven from `data/config/bot_service/load.json`.
- Older documentation can point at outdated file shapes. For example, current API routes live in the `bot_service/api/` package rather than a flat `bot_service/api.py`.
- The repo root contains committed or generated local state, so destructive cleanup, blanket formatting, or assumptions about a pristine worktree are unsafe.
- A "missing" agent from `/api/agents/` may only be initializing; check startup state before treating it as a broken registration.
- Changes around attachments, streaming, or queue events cross service boundaries in `openai_proxy`, `services/task_queue`, and `bot_service`, so regressions are likely unless exercised end to end.

## Survey basis

This summary was assembled from the current runtime code and supporting docs, including `README.md`, `services.md`, `pyproject.toml`, `config.py`, `bot_service/*`, `openai_proxy/*`, `services/task_queue/*`, `services/kb_manager/*`, `data/config/bot_service/load.json`, `chat_client.py`, `runbs.py`, `runoai.py`, and the root `main.py`.
