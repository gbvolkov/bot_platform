# `ingos_product_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

Product-knowledge assistant for Ingosstrakh products. Each registry variant selects a product and uses product retrieval plus optional vector prefetch to answer user questions.

## Entry module and initialization

- Implementation: `agents/ingos_product_agent/agent.py`
- Contract: exposes `initialize_agent(provider, product, use_platform_store, checkpoint_saver, prefetch_top_k=3)`
- Registry variants: all `product_*` IDs in `data/load.json`

## State graph / phases / routing

- `fetch_user_info`
- `reset_memory` or `prefetch_context`
- `default_agent`

The main user-visible branch is a retrieval-backed agent call; `prefetch_context` injects vector-retrieved system context before the agent runs.

## Inputs, context, and attachments

- Reads the latest human message as the product question.
- Supports reset-style cleanup.
- Current registry variants do not declare input attachments.
- Product choice comes from registry params such as `product=Car`.

## Tools and integrations

- product KB search tool
- Chroma/vector-store prefetch
- optional anonymization middleware
- KB reload listener for retriever refresh

## Outputs and persistence

- Returns text answers grounded in KB search and prefetched vector context.
- Persists conversation history in SQL and agent state in LangGraph checkpoints.

## Special behavior

- Multiple public IDs map to the same implementation with different `product` params.
- Only `product_Car` is active in the current registry file.
