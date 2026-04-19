# Mycroft GAZ Direct Tools And Actions

This catalog defines direct tools and operational actions available to Mycroft in the GAZ-sales configuration.

Use this document from skills that need to draft/send emails or export artifacts.

## Delegation tool: task

`task` is used to call configured stateless subagents.

Use it for:
- `marketing_analyst`;
- `gaz_pricing_bi_int`;
- `web_search_agent`.

Rules:
- The `subagent_type` must be an exact configured subagent name.
- The prompt must be a concrete business request, not a skill name.
- The prompt must include enough context for the subagent to answer without guessing.

## Tool: gmail_create_draft

Creates an email draft.

Use when the user asks to prepare, draft, or stage an email.

Input should include:
- recipient if known;
- subject;
- body;
- any caveats that must remain in the text.

Returns:
- draft creation result;
- draft identifier or link, depending on MCP response.

Do not use it when the user only asks for text in chat.

## Tool: gmail_send_message

Sends an email.

Use only when the user explicitly asks to send now.

Input should include:
- recipient;
- subject;
- body;
- confirmation that the user asked to send.

Runtime interrupt/approval applies to this tool. If required details are missing, ask before sending.

## Tool: store_artifact_tool

Stores an artifact and returns a link.

Use only when the user explicitly asks to save, export, persist, or create a downloadable file.

Input should include:
- artifact title;
- final user-facing content;
- intended format if available.

Do not store internal traces, raw subagent prompts, or unprocessed tool outputs unless the user explicitly asked for an execution report.

## Out of GAZ-sales skill scope

Maps and NHTSA workflows are not part of the current GAZ-sales Mycroft skill catalog.
Do not create or trigger dealer lookup, route computation, VIN decoding, or recall workflows from these GAZ-sales skills unless a separate deployment explicitly adds them back.
