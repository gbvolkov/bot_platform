# Platform Building Blocks

Back to the [documentation index](../index.md).

## Configuration and model selection

- `bot_service/config.py`
  - Defines API base path, database URL, attachment store path, and agent config path.
- `openai_proxy/config.py`
  - Defines the bot-service URL, default proxy user identity, request timeouts, and default attachment prompt.
- `services/task_queue/config.py`
  - Defines Redis keys, TTL, chunk size, heartbeat timings, worker timeouts, and watchdog thresholds.
- `models.toml`
  - Maps logical provider/mode pairs to concrete model names.
- `agents/llm_utils.py`
  - Turns provider + mode into LangChain chat model instances and fallback middleware.

## Agent registration and lifecycle

- `bot_service/agent_registry.py`
  - Loads agent definitions from `data/load.json`.
  - Tracks initialization state: `pending`, `initializing`, `ready`, `error`.
  - Stores per-agent attachment, streaming, and checkpoint-saver metadata.
- `.env`
  - Overrides `BOT_SERVICE_AGENT_CONFIG_PATH` to `./data/load.json`, making that file the live registry source in this repo.

## Shared state and config conventions

- `agents/state/state.py`
  - Defines `ConfigSchema` keys such as `user_id`, `user_role`, `thread_id`, and `attachments`.
- `bot_service/service.py`
  - Converts incoming API payloads into `HumanMessage` objects and `RunnableConfig.configurable`.
- `bot_service/schemas.py`
  - Defines content categories, message payloads, and bot-service response models.

## Shared tooling and middleware

- `agents/utils.py`
  - Provides shared enums, callback cleanup, nested invoke config handling, text extraction, and miscellaneous agent utilities.
- `agents/structured_prompt_utils.py`
  - Helps TypedDict-based structured-output prompting and provider-to-tool fallback.
- `agents/tools/think.py`
  - Internal reasoning/scratchpad tool used by several ideation and artifact agents.
- `agents/tools/store.py`
  - Stores a user-requested artifact and returns a Markdown link.
- `agents/tools/yandex_search.py`
  - Wrapped web search and summarization against Yandex Search APIs.

## Persistence and artifact output

- `bot_service/models.py`
  - Conversation/message persistence in SQL.
- `agents/store_artifacts.py`
  - Renders artifact sets to HTML/PDF, falls back to Markdown, then uploads via MinIO-compatible storage.
- `platform_utils/storage_svc.py`
  - Uploads generated artifacts and returns presigned download links.
- `platform_utils/llm_logger.py`
  - Writes JSONL traces for LLM and tool activity into `logs/`.

## Attachment and content processing

- `openai_proxy/schemas.py`
  - Normalizes OpenAI-style content parts into internal attachment payloads.
- `openai_proxy/main.py`
  - Hydrates URL and data-URL attachments into inline payloads.
- `bot_service/attachments.py`
  - Classifies attachments, stages files, converts unsupported ones to text, and optionally persists raw files.

## Platform integrations used by agents

- Knowledge-base retrieval
  - Product and service-desk style agents register KB reload listeners and use search tools or vector stores.
- External search
  - Several agents use Yandex web search; some service-desk logic also falls back to web results when validation fails.
- Job search
  - `find_job_agent` integrates hh.ru search and reranking utilities.
- Analytics and reporting
  - `bi_agent` delegates report generation to SQL/query helpers and returns generated files.

## Design implication

The reusable platform contract is intentionally broader than any single agent. New agents are expected to plug into:

- config-driven registration
- `initialize_agent(...)`
- LangGraph state handling
- `RunnableConfig.configurable`
- the attachment/message normalization rules enforced by `bot_service`
