# Repository conventions

This file describes the conventions used across the bot_platform repository.

## Layout
- agents/ contains agent implementations. Each agent lives in its own folder.
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

## Agent registry
- Register agents in bot_service/agent_registry.py via AgentDefinition.
- Required fields: id, name, description, factory, default_provider, supported_content_types.
- allow_raw_attachments controls whether raw files are passed to the agent.
- Product agents are auto-registered from data/docs/<product> as product_<product>.

## Content types and attachments
- ContentType is defined in bot_service/schemas.py.
- supported_content_types should reflect what the agent can safely parse.
- Attachment storage path is configured in bot_service/config.py.

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

## Docs
- README.md covers setup, environment, and run commands.
- services.md describes service architecture and API details.
