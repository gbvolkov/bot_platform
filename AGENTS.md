# Repository conventions

This file describes the conventions used across the bot_platform repository.

## Layout
- agents/ contains agent implementations. Each agent lives in its own folder.
- agent_runtime/ owns agent initialization, invocation, streaming, and worker execution; bot_service/agent_registry.py is a compatibility import of agent_runtime.registry.
- platform_contracts/ and platform_sdk/ define shared contracts, manifests, and invocation context.
- platform_capabilities/ contains reusable model, ingestion, rendering, storage, retrieval, and tracing helpers.
- platform_application/, platform_infrastructure/, and platform_access/ contain the newer application, storage, and access layers. Use the layer relevant to the current execution path.
- bot_service/ is the FastAPI backend for conversations, agents, and storage.
- openai_proxy/ is an OpenAI-compatible facade that forwards to bot_service.
- services/task_queue/ runs background worker jobs for agent execution.
- data/ stores product docs, vector stores, and indexes used by agents.
- logs/ holds local logs written by agents or service processes.
- .attachments_store/ stores raw attachments when pass-through is enabled.

## Agent development
- Put each agent in agents/<agent_name>/.
- Provide initialize_agent(...) in agents/<agent_name>/agent.py.
- Keep prompts in prompts.py and state schemas in state.py when applicable.
- Use shared helpers in agents/utils.py and agents/llm_utils.py for model setup.
- Those helpers are compatibility imports. For new code, prefer their canonical implementations in platform_capabilities.agent_helpers and platform_capabilities.models, or dependencies injected by the runtime; retain the existing imports in legacy agents unless a change is necessary.
- Reuse existing prototypes and declared libraries before adding an implementation, abstraction, service, or dependency. Add only what an actual requirement cannot obtain by reuse or a small compatible extension.

## Agent registry
- The active registry implementation is agent_runtime/registry.py; bot_service/agent_registry.py is a compatibility import.
- Register exposed agents through the configured manifest, normally data/config/bot_service/load.json, rather than introducing a second hardcoded registry.
- Manifest entries describe id, name, description, module, activation, params, supported_content_types, raw attachment policy, and streaming. AgentDefinition is the runtime representation, with fields including factory and default_provider.
- allow_raw_attachments controls whether raw files are passed to the agent.
- Product agents are auto-registered from data/docs/<product> as product_<product>.

## Content types and attachments
- ContentType is defined in platform_contracts; platform_sdk.schemas and bot_service.schemas expose compatible types.
- supported_content_types should reflect what the agent can safely parse.
- Use platform_capabilities.ingestion and the existing artifact/storage adapters. Attachment storage configuration depends on the active legacy or newer application path; bot_service/config.py is part of the legacy configuration surface.

## Configuration
- Root config.py loads gv.env from the repo root or ~/.env/gv.env.
- bot_service/config.py uses BOT_SERVICE_ prefixed env vars and reads .env.
- Keep secrets in env files; do not commit credentials.

## Services and entry points
- bot_service/main.py starts the API service.
- openai_proxy/main.py starts the OpenAI-compatible proxy.
- services/task_queue/worker.py runs the task queue worker.

## Testing and debugging
- Use python chat_client.py list-agents to verify registry entries.
- Use python chat_client.py chat <agent_id> for manual agent checks.
- README.md lists a minimal py_compile smoke check for key modules.
- Verify agent behavior through the CLI first, then through the actual runtime/streaming path. Keep agent defects distinct from deployment/queue/storage defects.
- Keep new Python code typed, use snake_case modules/functions/fields and PascalCase classes, and preserve existing public names and serialized schemas. Use meaningful behavior and contract tests for changes; do not add tests that merely mirror implementation details.

## Docs
- README.md covers setup, environment, and run commands.
- services.md describes service architecture and API details.
