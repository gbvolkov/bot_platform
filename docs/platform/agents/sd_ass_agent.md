# `sd_ass_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Legacy internal support assistant that routes a request to one of several knowledge-backed response branches: service desk, sales manager, or a default internal-assistant branch.

## Entry module and initialization

- Implementation: `agents/sd_ass_agent/agent.py`
- Contract: exposes `initialize_agent(provider, role, use_platform_store, checkpoint_saver)`
- Registry variant: `service_desk` (inactive)

## State graph / phases / routing

- `fetch_user_info`
- `reset_memory` or `augment_query`
- `route_request`
- one of `sd_agent`, `sm_agent`, `default_agent`

## Inputs, context, and attachments

- Reads human text turns and optional reset commands.
- No attachment support is declared in the registry.
- User role influences classification and prompt framing.

## Tools and integrations

- KB search tool
- glossary and abbreviation lookup tools
- ticket search tool on the service-desk branch
- Yandex web search fallback when the answer validator rejects a generated answer
- optional anonymization middleware via Palimpsest
- KB reload listener for retriever refresh

## Outputs and persistence

- Returns text answers only.
- If validation fails, the final assistant message can be replaced with a web-search-based answer.

## Special behavior

- This is a router agent plus validator loop, not a single monolithic prompt.
- It is one of the clearest examples of branch-per-domain behavior in the repository.
