# Bot Platform

An orchestration layer for knowledge-base driven agents. The project contains:

- **`bot_service`** – FastAPI backend that manages conversations, agents, and persistence.
- **`openai_proxy`** – An OpenAI-compatible HTTP façade that forwards chat completion requests to `bot_service` via a Redis-backed task queue and streams results to clients.
- **`services/task_queue`** – Asynchronous worker/dispatcher that executes long running agent jobs.
- **`agents/*`** – Business logic for individual agents (e.g., `ingos_product_agent`) with retrieval utilities, prompts, and state management.

This document describes how to set up the development environment from scratch, configure required services, and exercise the key workflows.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.13 | The repo targets 3.13. Use [`uv`](https://github.com/astral-sh/uv) or `pyenv` to install. |
| Redis 7+    | Required for the task queue / PubSub. Default URI `redis://localhost:6380/0`. |
| MySQL 8+ or MariaDB 10.5+ | `bot_service` persists data via SQLAlchemy + `aiomysql`. |
| Git, Make (optional) | For repository management and helper scripts. |

Install `uv` (recommended) if you have not already:

```bash
pip install uv
```

---

## 2. Clone & Create Virtual Environment

```bash
git clone https://github.com/<your-org>/bot_platform.git
cd bot_platform

# Create and activate virtual environment managed by uv
uv venv
. .venv/Scripts/activate      # PowerShell on Windows
# or
source .venv/bin/activate     # Linux / macOS
```

Install all project dependencies (uv reads `pyproject.toml`):

```bash
uv pip install -r pyproject.toml
```

If you prefer classic tooling, you can install with `pip install -r requirements.txt` (generate with `uv pip compile` if needed).

---

## 3. Environment Configuration

The project reads settings via Pydantic from `.env` files. At minimum, copy `gv.env` or `.env.example` if available; otherwise create one manually.

### 3.1 Core Variables

Create `.env` in the repo root with:

```dotenv
# Bot service database (MySQL / MariaDB)
BOT_SERVICE_DATABASE_URL=mysql+aiomysql://username:password@localhost:3306/bot_platform

# Redis connection for task queue / PubSub
TASK_QUEUE_REDIS_URL=redis://localhost:6380/0

# Optional overrides
TASK_QUEUE_BOT_REQUEST_TIMEOUT_SECONDS=300     # allow long agent runs
TASK_QUEUE_WORKER_HEARTBEAT_SECONDS=5
TASK_QUEUE_HEARTBEAT_STALE_AFTER_SECONDS=120

# OpenAI proxy defaults
OPENAI_PROXY_BOT_SERVICE_BASE_URL=http://localhost:8000/api
OPENAI_PROXY_DEFAULT_ATTACHMENT_PROMPT=Act as per your configured purpose using the provided attachments.

# Vector store paths (override defaults if necessary)
INGOS_VECTOR_DOCS_PATH=./data/docs
INGOS_VECTOR_STORE_PATH=./data/vector_store

# Model / API keys (if required by your agents)
OPENAI_API_KEY=sk-...
YA_API_KEY=...
GEMINI_API_KEY=...
```

The legacy `gv.env` file (referenced in `config.py`) should also reside either in the project root or `%USERPROFILE%/.env/gv.env` with secrets such as database passwords or provider API keys.

### 3.2 Database Bootstrapping

```bash
mysql -u root -p
CREATE DATABASE bot_platform CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'bot_user'@'localhost' IDENTIFIED BY 'strong_password';
GRANT ALL PRIVILEGES ON bot_platform.* TO 'bot_user'@'localhost';
FLUSH PRIVILEGES;
```

Update `BOT_SERVICE_DATABASE_URL` accordingly. Run migrations or allow SQLAlchemy to create tables on first start (if the code path creates metadata automatically).

### 3.3 Redis

Launch Redis locally on port `6379` or `6380`:

```bash
redis-server --port 6380
```

Adjust `TASK_QUEUE_REDIS_URL` if you use a different host/port.

---

## 4. Running Services

Open separate terminals (or background processes) for each component.

### 4.1 Bot Service API

```bash
uvicorn bot_service.main:app --reload --host 0.0.0.0 --port 8000
```

This exposes REST endpoints under `/api`, e.g.:
- `GET /api/agents/`
- `POST /api/conversations/`
- `POST /api/conversations/{conversation_id}/messages`

### 4.2 Task Queue Worker

The worker dequeues jobs, invokes agents, and streams status updates back through Redis.

```bash
python -m services.task_queue.worker
```

Ensure the environment variables (Redis URL, bot service base URL) are visible to this process.

### 4.3 OpenAI-Compatible Proxy

```bash
uvicorn openai_proxy.main:app --reload --host 0.0.0.0 --port 8080
```

The proxy connects to `bot_service` and the task queue. It supports both streamed and non-streamed chat completions. Default endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`

Example request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "find_job",
        "messages": [
          {"role": "user", "content": "Какие вакансии доступны?"}
        ],
        "stream": true
      }'
```

You will receive Server-Sent Events with status pulses, partial deltas, and a final `[DONE]` marker.

### 4.4 Optional Web Chat Frontend

```bash
uvicorn web_chat.main:app --reload --host 0.0.0.0 --port 8081
```

Navigate to `http://localhost:8081/` to interact with the service via a simple UI.

---

## 5. Agent Configuration (Ingos Product Agent)

The `ingos_product_agent` uses a vector store plus tools for follow-up retrieval. Key settings:

- `INGOS_VECTOR_DOCS_PATH` / `INGOS_VECTOR_STORE_PATH` – location of documents and persisted Chroma store.
- `INGOS_RETRIEVER` – set to `faiss` or `vector` in `config.py` / environment.
- `initialize_agent(..., prefetch_top_k=3)` automatically prefetches three documents from the vector store before the LLM runs. You can adjust this by passing a different `prefetch_top_k` when registering the agent in `agent_registry`.

Ensure the vector store is built (see `agents/ingos_product_agent/retrievers/vector_store.py`) before starting conversations.

---

## 6. Testing & Troubleshooting

### 6.1 Basic Diagnostics

- **Database connectivity** – run `mysqladmin ping -u bot_user -p`.
- **Redis** – `redis-cli -p 6380 ping`.
- **Task queue** – watch worker logs for heartbeat messages (`Heartbeat sent job_id=...`) and ensure no stale jobs accumulate (`Watchdog marked stale jobs as failed`).
- **OpenAI proxy** – check logs for `Prefetched N documents` entries; errors include full request payloads for debugging.

### 6.2 Common Issues

| Symptom | Resolution |
|---------|------------|
| `httpx.ReadTimeout` in worker | Increase `TASK_QUEUE_BOT_REQUEST_TIMEOUT_SECONDS` or confirm backend responds. |
| SSE stream closes early | Verify worker is running and heartbeats are visible; confirm Redis connectivity. |
| `Invalid HTTP request received` | Inspect proxy logs (raw bytes captured) for malformed client traffic. |
| 422 Validation errors | The proxy logs the full body—ensure the payload matches OpenAI schema. |

### 6.3 Automated Checks

The repository currently relies on manual testing. Code is linted/validated by `py_compile` in CI scripts; you can run:

```bash
python -m py_compile \
    openai_proxy/main.py \
    services/task_queue/worker.py \
    agents/ingos_product_agent/agent.py
```

Add your own pytest or integration suites as needed.

---

## 7. Useful Commands

| Command | Purpose |
|---------|---------|
| `python chat_client.py list-agents` | List available agents exposed by `bot_service`. |
| `python chat_client.py chat <agent_id>` | Send one-off messages to an agent for debugging. |
| `curl http://localhost:8080/v1/models` | Inspect proxy-visible models/agents. |
| `redis-cli monitor` | Observe queue/publish events in real time. |

---

## 8. Contributing

1. Fork + branch (`git checkout -b feature/my-change`).
2. Ensure code paths compile and services start.
3. Update documentation/README when adding new configuration steps.
4. Submit a pull request describing changes, risks, and testing strategy.

---

## 9. License

Include your license text here (MIT, Apache 2.0, proprietary, etc.). Update this section if a `LICENSE` file is added.

---

For further questions, contact the maintainers or raise an issue in the repository. Continuous logs (worker, proxy, bot service) provide detailed traceability to help diagnose runtime behaviour.
