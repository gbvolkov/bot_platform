# `ismart_task_variator_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Generates multiple variations of a task from a source task description. It is designed for short interactive sessions rather than long multi-phase orchestration.

## Entry module and initialization

- Implementation: `agents/ismart_task_variator_agent/agent.py`
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver, streaming=True)`
- Registry variant: `ismart_task_variator_agent`

## State graph / phases / routing

- `greetings`
- `cleanup`
- `run`

The graph can also skip the normal greeting path when runtime context sets `mode=auto`.

## Inputs, context, and attachments

- Reads the latest human message as the source task.
- A user can prefix the message with `/N` to request `N` generated options.
- No attachment support is declared in the registry.

## Tools and integrations

- Dynamic prompt generation only; no platform tool set is attached by default.
- Uses standard stream-writer callbacks, JSON tracing, and optional Langfuse.

## Outputs and persistence

- Returns task variants as normal assistant text.
- Stores option-count and conversation state in the current checkpoint thread.

## Special behavior

- This agent is intentionally minimal and is closer to `simple_agent` than to the larger orchestrator-style agents.
