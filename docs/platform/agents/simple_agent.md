# `simple_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Minimal general-purpose assistant that can accept a user-defined system prompt and then answer subsequent turns under that prompt.

## Entry module and initialization

- Implementation: `agents/simple_agent/agent.py`
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver, streaming=True)`
- Registry variants: `simple_agent`, `simple_agent_en`

## State graph / phases / routing

- `greetings`
- `set_prompt`
- `cleanup`
- `run`

The first interaction usually captures or confirms the desired system prompt, after which the graph collapses prior setup messages and routes normal turns into the main agent node.

## Inputs, context, and attachments

- Uses the latest human text for prompt setup and subsequent chat turns.
- No attachment support is declared in the registry.
- Locale differs by variant (`ru` vs `en`).

## Tools and integrations

- No application tools are attached by default.
- Uses dynamic prompt middleware, stream-writer callbacks, JSON tracing, and optional Langfuse.

## Outputs and persistence

- Returns normal assistant text.
- Persists system-prompt state and conversation history through the selected checkpoint saver.

## Special behavior

- This is the repository's baseline conversational agent and the easiest starting point for new agent development.
