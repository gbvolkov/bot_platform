# `openai_proxy` API

Back to the [documentation index](../index.md).

## Purpose

`openai_proxy` exposes an OpenAI-compatible surface over the internal conversation platform.

It is responsible for:

- exposing model cards derived from ready agents
- accepting chat-completions requests
- creating or reusing bot-service conversations
- normalizing OpenAI-style attachments/content parts
- delegating asynchronous execution to Redis + worker
- converting queue events into SSE or final JSON

## Endpoints

### `GET /healthz`

Returns `{ "status": "ok" }`.

### `GET /v1/models`

Returns a model list derived from `bot_service /agents/`.

### `GET /v1/models/{model_id}`

Returns one ready model card or `404` if the agent is unknown or not visible.

### `POST /v1/chat/completions`

Accepts:

- `model`
- `messages`
- optional `user`
- optional `conversation_id`
- optional `stream`

## Request-shaping behavior

- The proxy preserves only the latest user message text as the prompt sent to the worker.
- The latest user message attachments are forwarded separately.
- If the latest user message has attachments but no text, the proxy injects `OPENAI_PROXY_DEFAULT_ATTACHMENT_PROMPT`.

## Attachment normalization

`openai_proxy/schemas.py` accepts OpenAI-style content parts and converts them into internal attachments.

Supported forms include:

- inline `attachments`
- `data:` URLs
- `http(s)` URLs
- image content parts such as `image_url` and `input_image`

Hydration behavior:

- remote `http(s)` attachments are downloaded eagerly
- data URLs are decoded into base64 payload fields
- filenames are synthesized if missing
- attachments without usable `data` or `text` are rejected with `400`

## Conversation behavior

- No `conversation_id`
  - the proxy creates a new conversation in `bot_service`
- Existing `conversation_id`
  - the proxy fetches and validates that conversation
- Pending conversations
  - the proxy polls for up to 30 seconds and then returns `503` with `Retry-After: 1` if the agent is still initializing

## Streaming response behavior

For `stream=true`, the proxy returns Server-Sent Events.

Event mapping:

- queue `status` -> empty `chat.completion.chunk` with `agent_status`
- queue `chunk` -> `delta.content`
- queue `completed` -> terminal chunk with `finish_reason=stop`
- queue `interrupt` -> terminal chunk with `agent_status=interrupted`
- queue `failed` -> SSE JSON error payload
- queue `heartbeat` -> SSE comment line

The stream ends with:

- `data: [DONE]`

## Non-streaming response behavior

For `stream=false`, the proxy waits for a terminal queue event and returns a single `chat.completion` response.

Terminal cases:

- `completed`
  - final assistant content and optional attachment metadata
- `interrupt`
  - assistant content is the interrupt question/content
- `failed`
  - returned as `502 Bad Gateway`

## OpenAI-compatibility limits

- The proxy exposes a chat-completions-like interface, not the full OpenAI API surface.
- Prompt construction is intentionally simplified to the latest user turn.
- Tool-calling is not surfaced as OpenAI tools at the proxy boundary; agent-side tools remain internal platform behavior.

## Important operational notes

- Ready models only
  - `/v1/models` shows only agents that are currently initialized in `bot_service`.
- Default user identity
  - if the caller omits `user`, the proxy uses `OPENAI_PROXY_DEFAULT_USER_ID`.
- Status-rich SSE
  - clients should tolerate empty-delta status chunks and heartbeat comments, not only token chunks.
