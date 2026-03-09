# `find_job_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

`find_job_agent` turns a resume-like user input into ranked vacancies. It extracts structured candidate features, searches hh.ru, reranks matches, and formats the top results as a Markdown vacancy list.

## Entry module and initialization

- Implementation: `agents/find_job_agent/find_job_agent.py`
- Contract: exposes `initialize_agent(provider, role, use_platform_store, checkpoint_saver)`
- Registry variant: `find_job` (currently inactive)

## State graph / phases / routing

- `fetch_user_info`
- `reset_memory` or `capture_resume`
- `extract_features`
- `job_lookup`
- `rank_jobs`
- `respond`

## Inputs, context, and attachments

- Uses the latest human text as resume content.
- Supports `reset` for memory cleanup.
- The active registry configuration does not declare raw or structured input attachments.

## Tools and integrations

- LLM-based feature extraction from resume text
- hh.ru area resolution and vacancy search helpers
- semantic scorer and optional embedding reranker
- optional DeepL translation of the final answer when `JOB_FIND_TRANSLATE=true`

## Outputs and persistence

- Returns Markdown text with ranked vacancies, company/location/salary/skills, and direct links.
- Keeps extracted features and ranked jobs in graph state for the current thread.

## Special behavior

- Search is query-expanded from inferred positions and locations.
- Reranking mixes heuristic matching with semantic scoring.
