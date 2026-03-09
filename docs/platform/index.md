# Bot Platform Documentation

This documentation set describes the current platform implementation in `C:\Projects\bot_platform` as of March 9, 2026. Runtime code and configuration are the source of truth; older prose in `README.md`, `services.md`, and `docs/` is supplemental.

## Reading order

1. [Platform Principles And Topology](architecture/principles-and-topology.md)
2. [Runtime Flows](architecture/runtime-flows.md)
3. [Platform Building Blocks](architecture/platform-building-blocks.md)
4. [Registry And Variants](architecture/registry-and-variants.md)
5. [Common Agent Architecture](agents/common-agent-architecture.md)
6. [Agent Support Modules](agents/support-modules.md)
7. Agent implementation pages in [`agents/`](agents/index.md)
8. Service and protocol references in [`interfaces/`](interfaces/index.md)

## Documentation map

- [`architecture/`](architecture/index.md)
  - Platform principles, service boundaries, execution flows, shared components, and registry/config behavior.
- [`agents/`](agents/index.md)
  - Common agent contract, shared agent support modules, and one page per implementation module.
- [`interfaces/`](interfaces/index.md)
  - `bot_service` API, task queue and worker protocol, and the OpenAI-compatible proxy API.

## Major subsystems

- `bot_service`
  - Internal FastAPI API for conversations, agents, attachments, persistence, and synchronous agent execution.
- `openai_proxy`
  - OpenAI-compatible facade that converts chat-completions requests into platform jobs.
- `services/task_queue`
  - Redis-backed queue, event stream, watchdog, and worker loop for asynchronous execution.
- `agents/*`
  - LangGraph-based agent implementations plus shared agent support code.
- `platform_utils/*`
  - Shared tracing, storage upload, and periodic-task helpers used by multiple components.

## Quick links

- [Common Agent Architecture](agents/common-agent-architecture.md)
- [Support Modules](agents/support-modules.md)
- [Bot Service API](interfaces/bot_service_api.md)
- [Task Queue And Worker](interfaces/task_queue_and_worker.md)
- [OpenAI Proxy API](interfaces/openai_proxy_api.md)

