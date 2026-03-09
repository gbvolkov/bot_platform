# `ideator_old_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

`ideator_old_agent` is an earlier ideation implementation kept in the codebase as a legacy version of the report-to-ideas workflow.

## Entry module and initialization

- Implementation: `agents/ideator_old_agent/agent.py`
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver)`
- Registry variant: `ideator_old` (inactive)

## State graph / phases / routing

Like the other legacy ideator flow, it is organized around:

- report initialization
- sense-line generation
- idea generation
- waiting for the next user decision

## Inputs, context, and attachments

- Designed for scout-report style inputs.
- Registry configuration allows raw attachments and declares `jsons` support.
- Locale is configurable through initialization params.

## Tools and integrations

- structured-output prompting
- report loading/parsing helpers
- ideation prompts and article-grounding logic
- standard JSON tracing and optional Langfuse

## Outputs and persistence

- Produces sense lines and idea proposals as conversational output.
- Stores intermediate ideation state in the graph checkpoint rather than publishing an external artifact set.

## Special behavior

- This implementation is kept for backward comparison and is superseded operationally by `ideator` and `new_ideator`.
