# `bi_agent`

See also: [Common Agent Architecture](common-agent-architecture.md), [Support Modules](support-modules.md), [Registry And Variants](../architecture/registry-and-variants.md).

## Purpose and use cases

`bi_agent` answers analytics questions by generating a report from tabular data and optionally returning a chart image plus the generated data file.

## Entry module and initialization

- Implementation: `agents/bi_agent/bi_agent.py`
- Contract: exposes `initialize_agent(provider, role, use_platform_store, notify_on_reload, checkpoint_saver)`
- Registry variant: `ai_bi` (currently inactive)

## State graph / phases / routing

- `fetch_user_info`
- `reset_memory` or `generate_report`
- `respond`

The agent is intentionally linear: it extracts the latest human question, calls the report backend, then returns a text answer with optional attachments.

## Inputs, context, and attachments

- Reads the latest human text as the analytics question.
- Supports a `reset` message for state cleanup.
- No input attachment types are declared in the registry.

## Tools and integrations

- `agents/sql_query_gen.py` via `get_response(...)`
- optional Langfuse callbacks
- JSON trace logging
- KB reload hook is registered but currently only logs that no KB-backed retriever is configured

## Outputs and persistence

- Assistant message can contain:
  - text summary
  - a tabular file attachment (`csv` or spreadsheet)
  - an image attachment for a generated chart
- `bot_service` serializes those parts into normal assistant-message content and metadata.

## Special behavior

- Default data source is `data/data.csv`.
- Temporary output files are base64-encoded into message content and then removed from local disk.
