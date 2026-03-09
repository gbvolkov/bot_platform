# `bot_service` API

Back to the [documentation index](../index.md).

## Purpose

`bot_service` is the internal FastAPI service that owns:

- conversation creation and lookup
- message persistence
- attachment processing
- synchronous and streaming agent execution
- interrupt/resume semantics

Base path: `/api`

## Authentication and caller identity

There is no auth layer in the current code. Caller identity is carried by headers:

- `X-User-Id`
- `X-User-Role` (optional, defaults to `BOT_SERVICE_DEFAULT_USER_ROLE`)

## Core models

- `ConversationCreate`
  - `agent_id`, optional `title`, optional `user_role`, optional `metadata`
- `MessageCreate`
  - wraps `MessagePayload`
- `MessagePayload`
  - `type`: `text` or `reset`
  - `text`
  - `metadata`
  - `attachments`
- `AttachmentPayload`
  - `filename`, optional `content_type`, optional base64 `data`, optional extracted `text`

## Endpoints

### `GET /agents/`

Returns only ready agents from `agent_registry.list_ready_agents()`.

Response fields:

- `id`
- `name`
- `description`
- `provider`
- `supported_content_types`

### `POST /conversations/`

Creates a conversation for a selected agent.

Behavior:

- calls `ensure_agent_ready(agent_id)`
- returns `201` with `status=active` if the agent is ready
- returns `202` with `status=pending` if initialization is still in progress

### `GET /conversations/`

Lists the caller's conversations ordered by `last_message_at desc`.

Behavior:

- pending conversations are upgraded to `active` if the agent becomes ready

### `GET /conversations/{conversation_id}`

Returns the conversation plus persisted messages.

Behavior:

- verifies ownership by `user_id`
- may promote `pending -> active` if the agent becomes ready

### `POST /conversations/{conversation_id}/messages`

Primary execution endpoint.

Query parameter:

- `stream=true|false`

Behavior:

1. Loads and optionally locks the conversation.
2. Ensures the agent is ready.
3. Processes attachments based on registry capabilities.
4. Detects `pending_interrupt` and switches to resume mode when needed.
5. Invokes the agent synchronously or through the streaming wrapper.
6. Persists both user and assistant messages.
7. Updates conversation status and `pending_interrupt`.

Success response:

- `201 Created`
- body: `SendMessageResponse`

## Conversation and message lifecycle

- User input is normalized into content segments.
- Unsupported attachments may be converted to text and appended into `attachment_text_segments`.
- Assistant output is normalized into:
  - `raw_text`
  - structured content payload
  - extracted attachments, if any

## Interrupt semantics

When an agent returns `__interrupt__`:

- assistant message metadata gets:
  - `agent_status=interrupted`
  - `interrupt_payload`
  - optionally `question` and `content`
- conversation status becomes `waiting_user`
- conversation metadata stores `pending_interrupt`

The next user turn resumes the graph via `Command(resume=raw_user_text)`.

## Streaming behavior

When `stream=true`, `bot_service` emits SSE-like event payloads internally:

- `chunk`
- `custom`
- terminal `completed` or `interrupt`

This path is mainly consumed by the worker, not by the external OpenAI-compatible client.

## Error behavior

- `404`
  - unknown agent, missing conversation, or wrong user ownership
- `409`
  - conversation is still initializing
- `400`
  - invalid reset + attachments combination
- `500`
  - agent initialization failure or attachment processing failure
- `502`
  - agent invocation failure

## Persistence model

- `conversations`
  - `agent_id`, `user_id`, `user_role`, `status`, `metadata`, timestamps
- `messages`
  - `conversation_id`, `role`, structured `content`, `raw_text`, `metadata`, timestamp
