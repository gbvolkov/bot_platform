# `theodor_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Legacy but active product-mentor orchestrator that walks a user through a multi-artifact product methodology and finishes by publishing a final report.

## Entry module and initialization

- Implementation: `agents/theodor_agent/agent.py`
- Key subgraph: `agents/theodor_agent/choice_agent.py`
- Contract: exposes `initialize_agent(provider, role, use_platform_store, notify_on_reload, locale, checkpoint_saver)`
- Registry variant: `theodor_agent`

## State graph / phases / routing

Outer graph:

- `init`
- repeated `progress_banner_<artifact>` -> `choice_agent_<artifact>` -> `cleanup_<artifact>` -> `confirmed_banner_<artifact>`
- `final_output`

Inner choice-agent graph:

- `init`
- `generate_options`
- `select_option`
- `generate_aftifact`
- `confirm`

## Inputs, context, and attachments

- Primary input is the user's product idea or startup concept.
- No attachment support is declared in the registry.
- Locale and methodology details are resolved from artifact definitions and locale helpers.

## Tools and integrations

- `think_tool`
- Yandex search tool
- structured-output schemas for options and final artifacts
- model fallback middleware and summarization middleware
- context-reduction helpers that summarize prior artifact discussions
- artifact storage through `agents/store_artifacts.py`

## Outputs and persistence

- Produces option lists, confirmed artifact texts, progress banners, and a final report link.
- Stores artifact summaries back into state so earlier phases can be reduced without losing key conclusions.

## Special behavior

- This implementation is the clearest example of interrupt/resume in the agent layer.
- The nested choice agent uses LangGraph `interrupt(...)` to request user selection and confirmation between artifact steps.
