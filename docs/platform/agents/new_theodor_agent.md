# `new_theodor_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

New product-mentor orchestrator that leads a user through a fixed artifact sequence and produces a final stored report.

## Entry module and initialization

- Implementation: `agents/new_theodor_agent/agent.py`
- Key subgraph: `agents/new_theodor_agent/choice_agent.py`
- Contract: exposes `initialize_agent(provider, role, use_platform_store, notify_on_reload, locale, checkpoint_saver, streaming=True)`
- Registry variant: `new_theodor_agent`

## State graph / phases / routing

High-level flow:

- `init`
- optional `greetings`
- `progress_<artifact>`
- `choice_agent_<artifact>`
- `advance`
- `final_output`

The outer graph iterates over `ARTIFACTS`; each artifact delegates option/choice work to a nested choice-agent graph.

## Inputs, context, and attachments

- Main input is the original product idea/prompt.
- No attachment support is declared in the registry.
- Locale-specific prompts and artifact labels are resolved during initialization.

## Tools and integrations

- artifact definitions in `artifacts_defs.py`
- nested choice agent
- artifact storage through `agents/store_artifacts.py`
- stream-writer callbacks for tool/chain lifecycle visibility

## Outputs and persistence

- Emits progress banners between artifacts.
- Produces final artifact content per stage and one final report link at the end.
- Uses LangGraph checkpointing to preserve the multi-artifact conversation thread.

## Special behavior

- Registry streaming config enables `messages`, `custom`, and `values` with subgraph streaming.
- This implementation is simpler at the outer level than `theodor_agent`; more of the detail is isolated in the nested choice agent.
