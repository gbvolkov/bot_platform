# `ideator_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Legacy but active ideation agent that turns a scout report into sense lines and then into idea candidates grounded in report articles.

## Entry module and initialization

- Implementation: `agents/ideator_agent/agent.py` (registry imports the package root `agents.ideator_agent`)
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver)`
- Registry variant: `ideator`

## State graph / phases / routing

- `init`
- `sense_lines`
- `ideas`
- `await`

The graph stores phase in state and returns after each major generation step so the conversation can continue interactively.

## Inputs, context, and attachments

- Accepts report input from:
  - runtime context `report_path`
  - JSON/text in the last user message
  - JSON attachment paths when available in state
- Registry configuration allows raw attachments and declares `jsons` as supported content.

## Tools and integrations

- structured-output subagents for sense lines and ideas
- `think_tool`
- Yandex web search in the ideas stage
- locale-aware prompt and report-processing helpers

## Outputs and persistence

- Produces conversational sense-line and idea lists grounded in article references.
- Keeps report, filtered articles, selected line, and idea state in the checkpoint.

## Special behavior

- This is the older active ideation flow; it is more explicitly phased than `new_ideator_agent`.
- It does not publish a final external artifact by itself.
