# Agent Support Modules

Back to the [documentation index](../index.md).

This page documents shared agent platform code that is not itself a standalone agent.

## Core shared modules

| Module or folder | Role in the platform |
|---|---|
| `agents/utils.py` | Shared `ModelType`, invoke config helpers, callback cleanup, text extraction, and miscellaneous agent utilities. |
| `agents/llm_utils.py` | Provider/model registry, LangChain chat-model construction, and model fallback helpers. |
| `agents/state/state.py` | Shared `CommonAgentState` and `ConfigSchema` conventions used by several graphs. |
| `agents/structured_prompt_utils.py` | Structured-output prompt generation and provider-to-tool fallback middleware. |
| `agents/user_info.py` | Pulls `user_id` and `user_role` from `RunnableConfig.configurable`. |
| `agents/store_artifacts.py` | Renders artifact sets, stores them, and returns upload links. |

## Shared tooling

| Module | Purpose |
|---|---|
| `agents/tools/think.py` | Internal scratchpad tool for explicit reasoning. |
| `agents/tools/store.py` | User-facing artifact save tool. |
| `agents/tools/yandex_search.py` | Web search wrapper and optional summarizer. |
| `agents/yandex_tools/*` | Lower-level Yandex model/tool integration used by `agents/llm_utils.py`. |

## Retrieval and KB support

| Module or folder | Purpose |
|---|---|
| `agents/retrievers/*` | Shared retrieval support used by older internal QA/service-desk style agents. |
| `agents/ingos_product_agent/retrievers/*` | Product-agent-specific vector-store and retriever helpers. |
| `services.kb_manager.notifications` | KB reload notifications consumed by several agents. |

## Output and formatting helpers

| Module | Purpose |
|---|---|
| `agents/prettifier.py` | Markdown/format cleanup for user-facing output in some legacy agents. |
| `agents/sql_query_gen.py` | Analytics/BI report generation backend used by `bi_agent`. |

## Platform utilities outside `agents/`

| Module | Purpose |
|---|---|
| `platform_utils/llm_logger.py` | JSONL tracing callback for model and tool activity. |
| `platform_utils/storage_svc.py` | MinIO-compatible upload helper used by artifact storage. |
| `platform_utils/periodic_task.py` | Background periodic-task helper used by platform processes. |

## Design note

These modules are intentionally documented separately from agent implementation pages because they form the reusable agent platform:

- model bootstrapping
- shared config/state keys
- search/store/think tools
- artifact persistence
- tracing and uploads

New agents should typically reuse these modules instead of reimplementing equivalent platform logic.
