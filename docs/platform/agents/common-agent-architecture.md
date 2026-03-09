# Common Agent Architecture

Back to the [documentation index](../index.md).

## Standard agent contract

Most platform agents follow this contract:

- expose `initialize_agent(...)`
- return a compiled LangGraph runnable
- accept some combination of:
  - `provider`
  - `use_platform_store`
  - `locale`
  - `role`
  - `checkpoint_saver`
  - `streaming`
  - implementation-specific knobs such as `product`, `artifact_id`, or `prefetch_top_k`

The registry imports the module, discovers `initialize_agent`, and passes config-derived params.

## Shared execution model

- Input shape
  - `bot_service` converts the current turn into `HumanMessage` content parts.
- Runtime context
  - `RunnableConfig.configurable` commonly carries `user_id`, `user_role`, `thread_id`, and optional raw attachments.
- Output shape
  - Agents usually return final `AIMessage` content, sometimes with structured attachments or interrupt payloads.
- Persistence
  - `bot_service` persists normalized user and assistant messages regardless of the internal graph structure.

## Common LangGraph patterns

- Linear flows
  - `simple_agent`, `ismart_task_variator_agent`, and `bi_agent` use short state-machine pipelines.
- Router flows
  - `sd_ass_agent`, `ideator_*`, and `ismart_tutor_agent` branch based on phase, user state, or structured decisions.
- Orchestrator + nested subgraph flows
  - `theodor_agent` and `new_theodor_agent` delegate artifact-specific work to choice subgraphs.
- Reset handling
  - Several legacy/internal agents treat a `reset` content part as a signal to clear state or message history.

## Streaming conventions

- The registry can declare per-agent stream modes.
- Many newer agents add callback handlers that forward tool and chain lifecycle events through LangGraph `custom` stream mode.
- The bot-service streaming path merges:
  - `messages` deltas
  - `values` snapshots
  - `custom` events

## Interrupt conventions

- An agent can return `__interrupt__` to request more user input.
- `bot_service` converts that into:
  - assistant message metadata with `agent_status=interrupted`
  - conversation metadata with `pending_interrupt`
- On the next turn, `bot_service` resumes the same graph with `Command(resume=raw_user_text)`.

## Attachment conventions

- Agent capability declaration lives in the registry entry:
  - `supported_content_types`
  - `allow_raw_attachments`
- When raw attachments are not allowed or not supported, `bot_service` converts them into text segments and appends them to the human message.
- When raw attachments are allowed, the raw file paths are passed in `configurable.attachments`.

## Common observability and storage patterns

- JSON trace logging
  - Many agents attach `platform_utils.llm_logger.JSONFileTracer`.
- Langfuse
  - Several agents conditionally add Langfuse callbacks when configured.
- Checkpointing
  - Agents generally use an in-memory saver unless a registry/config setting injects a shared SQLite saver.

## How to read the implementation pages

Each agent page in this folder answers the same questions:

- What problem does the agent solve?
- What module and initializer define it?
- What graph phases or subgraphs does it use?
- What inputs, tools, outputs, and persistence behavior matter?
- What registry variants or caveats should an operator know?
