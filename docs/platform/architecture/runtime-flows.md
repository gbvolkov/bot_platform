# Runtime Flows

Back to the [documentation index](../index.md).

## Synchronous `bot_service` execution flow

This is the direct internal path through `POST /api/conversations/{id}/messages`.

```mermaid
sequenceDiagram
    participant Client
    participant API as bot_service API
    participant Attach as attachment processor
    participant Registry as agent_registry
    participant Agent as LangGraph agent
    participant DB as SQL database

    Client->>API: POST /api/conversations/{id}/messages
    API->>DB: lock and load conversation
    API->>Registry: ensure_agent_ready(agent_id)
    API->>Attach: process attachments and metadata
    API->>Agent: invoke(initial state or Command.resume)
    Agent-->>API: AI result or __interrupt__
    API->>DB: persist user + assistant messages
    API->>DB: update conversation status and pending_interrupt
    API-->>Client: SendMessageResponse
```

### Conversation statuses

- `pending`
  - Conversation exists, but the selected agent is still initializing.
- `active`
  - Conversation is ready for a normal user turn.
- `waiting_user`
  - The previous agent turn returned an interrupt and the next user message will resume the graph.

## Asynchronous proxy -> Redis -> worker -> bot flow

This is the path behind `POST /v1/chat/completions`.

```mermaid
sequenceDiagram
    participant Client
    participant Proxy as openai_proxy
    participant Redis
    participant Worker
    participant Bot as bot_service

    Client->>Proxy: POST /v1/chat/completions
    Proxy->>Bot: create/fetch conversation
    Proxy->>Redis: enqueue EnqueuePayload
    Worker->>Redis: BLPOP job
    Worker->>Redis: mark running + heartbeat
    Worker->>Bot: POST /api/conversations/{id}/messages
    Bot-->>Worker: SendMessageResponse or stream events
    Worker->>Redis: publish status/chunk/completed|interrupt|failed
    Proxy->>Redis: subscribe or wait_for_completion
    Proxy-->>Client: SSE chunks or final JSON completion
```

### Queue job stages

- `queued`
- `running`
- `streaming`
- `completed`
- `failed`
- `interrupted`

### Queue event types

- `status`
- `chunk`
- `completed`
- `failed`
- `heartbeat`
- `interrupt`

## Interrupt / resume lifecycle

The platform supports LangGraph interrupts in both direct and proxy paths.

```mermaid
sequenceDiagram
    participant User
    participant Bot as bot_service
    participant Agent
    participant Proxy as openai_proxy/worker

    User->>Bot: normal user turn
    Bot->>Agent: invoke(initial state)
    Agent-->>Bot: __interrupt__ payload
    Bot-->>User: assistant question + waiting_user state
    User->>Proxy: follow-up user reply with same conversation_id
    Proxy->>Bot: raw_user_text preserved in payload
    Bot->>Agent: Command(resume=raw_user_text)
    Agent-->>Bot: completed response
    Bot-->>User: normal assistant response and active state
```

### Interrupt persistence rules

- The assistant message metadata stores `agent_status=interrupted`.
- Conversation metadata stores `pending_interrupt`.
- The next user turn is converted into `Command(resume=raw_user_text)` instead of a new ordinary message state.

## Startup and readiness flow

- `bot_service` startup
  - Initializes SQL models and calls `agent_registry.preload_all()`.
- `openai_proxy` startup
  - Starts the bot client and Redis task queue client.
- `task worker` startup
  - Connects to Redis and `bot_service`, then starts the main loop and watchdog.

## Operational behavior worth documenting

- Heartbeats
  - Workers refresh `last_heartbeat` and publish `heartbeat` events during long jobs.
- Watchdog
  - The watchdog marks stale jobs as failed when heartbeat age exceeds `TASK_QUEUE_HEARTBEAT_STALE_AFTER_SECONDS`.
- Attachment handling
  - The proxy hydrates `http(s)` and `data:` attachment URLs into inline payloads.
  - `bot_service` either forwards raw attachments or converts unsupported ones into text segments before invoking an agent.
