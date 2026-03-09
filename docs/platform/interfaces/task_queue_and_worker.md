# Task Queue And Worker

Back to the [documentation index](../index.md).

## Purpose

`services/task_queue` implements the asynchronous execution layer used by `openai_proxy`.

It provides:

- Redis-backed job enqueue/dequeue
- job status persistence
- Pub/Sub event streaming
- worker heartbeats
- stale-job watchdog handling

## Key modules

- `services/task_queue/models.py`
  - queue payload and event schemas
- `services/task_queue/redis_queue.py`
  - Redis wrapper for queue, status, events, and watchdog support
- `services/task_queue/worker.py`
  - dequeue loop, heartbeat loop, bot-service calls, chunk publishing, result/failure handling
- `services/task_queue/config.py`
  - Redis URL, key names, TTL, heartbeat intervals, timeout values

## Redis contract

### Queue key

- default: `agent:jobs`

### Status hashes

- prefix: `agent:status:`
- fields commonly include:
  - `status`
  - `created_at`
  - `updated_at`
  - `last_heartbeat`
  - `conversation_id`
  - `model`
  - `user_id`
  - `result`
  - `error`

### Event channels

- prefix: `agent:events:`
- one Pub/Sub channel per `job_id`

### Active jobs set

- key: `agent:status:active_jobs`
- stores heartbeat timestamps for watchdog scans

## Enqueue payload

`EnqueuePayload` fields:

- `job_id`
- `model`
- `conversation_id`
- `user_id`
- optional `user_role`
- `text`
- optional `raw_user_text`
- optional `attachments`
- optional `metadata`
- `stream`

## Queue events

`QueueEvent` can carry:

- `job_id`
- `type`
  - `status`
  - `chunk`
  - `completed`
  - `failed`
  - `heartbeat`
  - `interrupt`
- optional `status`
- optional `content`
- optional `metadata`
- optional `usage`
- optional `error`

## Worker execution flow

1. `BLPOP` a job from Redis.
2. Mark it `running`.
3. Register it in the active-job set.
4. Start the heartbeat loop.
5. Call `bot_service` with either `send_message(...)` or `send_message_stream(...)`.
6. Publish:
   - `status=streaming`
   - `chunk` events
   - terminal `completed`, `interrupt`, or `failed`
7. Store final result or failure in the status hash.
8. Remove the job from the active set.

## Heartbeat and watchdog behavior

- Worker heartbeat interval
  - `TASK_QUEUE_WORKER_HEARTBEAT_SECONDS`
- Stale threshold
  - `TASK_QUEUE_HEARTBEAT_STALE_AFTER_SECONDS`
- Watchdog scan interval
  - `TASK_QUEUE_WATCHDOG_INTERVAL_SECONDS`

If a running job stops updating its heartbeat before it reaches a terminal state, the watchdog marks it as failed with `Heartbeat timeout exceeded`.

## Interrupt handling

If `bot_service` returns an assistant message with `agent_status=interrupted`:

- the worker marks the job `interrupted`
- the worker publishes an `interrupt` event with question/content metadata
- the job becomes terminal from the queue perspective

## Streaming behavior

- Native bot-service stream available
  - If the payload requests streaming, the worker relays bot-service chunks directly into Redis events.
- Fallback chunking
  - If the final response arrives as one large string, the worker splits it into chunks using `TASK_QUEUE_CHUNK_CHAR_LIMIT`.

## Failure behavior

Any worker-side exception becomes:

- job status `failed`
- a stored `error`
- a `failed` queue event
