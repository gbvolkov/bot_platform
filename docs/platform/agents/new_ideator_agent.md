# `new_ideator_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

New ideation flow that accepts a scout report and then uses a single agentic run loop to produce thematic threads, ideas, and a final docset, with optional artifact storage.

## Entry module and initialization

- Implementation: `agents/new_ideator_agent/agent.py`
- Contract: exposes `initialize_agent(provider, use_platform_store, locale, checkpoint_saver, streaming=True)`
- Registry variant: `new_ideator`

## State graph / phases / routing

- `greetings`
- `set_report`
- `run`

Unlike the older ideator implementations, most of the work happens inside one `create_agent(...)` node backed by middleware and tools.

## Inputs, context, and attachments

- Accepts scout report content from runtime context or the latest user message.
- Registry declares `allow_raw_attachments=true` and `supported_content_types=[jsons]`.
- Locale and provider are passed via registry params.

## Tools and integrations

- `commit_thematic_threads`
- `commit_ideas`
- `commit_final_docset`
- `store_artifact_tool`
- Yandex web search tool
- `SummarizationMiddleware` to compress long histories

## Outputs and persistence

- Produces conversational ideation output and can store the resulting docset as an external artifact.
- Keeps ideation state inside the checkpoint and emits custom stream events via callback handlers.

## Special behavior

- This is the most tool-oriented ideator implementation in the repository.
- It is the only ideator variant that directly includes the generic artifact storage tool in its main agent toolset.
