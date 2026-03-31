from __future__ import annotations

import asyncio

from services.task_queue.models import EnqueuePayload
from services.task_queue import worker as task_worker


class _FakeClient:
    async def send_message_stream(self, **kwargs):
        yield {
            "type": "custom",
            "data": {
                "type": "progress",
                "tool": "purchase_search_tool",
                "stage": "crawler_search",
                "message": "Looking zakupki.gov.ru with search string: страхован",
            },
        }
        yield {
            "type": "completed",
            "content": "Done",
            "metadata": {},
        }


class _FakeClientWithChunk:
    async def send_message_stream(self, **kwargs):
        yield {
            "type": "chunk",
            "content": "Hello ",
        }
        yield {
            "type": "chunk",
            "content": "world",
        }
        yield {
            "type": "completed",
            "content": "Hello world",
            "metadata": {},
        }


class _FakeClientWithCustomThenChunk:
    async def send_message_stream(self, **kwargs):
        yield {
            "type": "custom",
            "data": {"type": "progress", "message": "Thinking"},
        }
        yield {
            "type": "chunk",
            "content": "Hello world",
        }
        yield {
            "type": "completed",
            "content": "Hello world",
            "metadata": {},
        }


class _FakeQueue:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.events = []
        self.result = None
        self.failure = None

    async def mark_status(self, job_id: str, status: str, extra=None) -> None:
        self.statuses.append(status)

    async def publish_event(self, event) -> None:
        self.events.append(event)

    async def register_active_job(self, job_id: str) -> None:
        return None

    async def update_heartbeat(self, job_id: str, status: str | None = None) -> None:
        return None

    async def clear_active_job(self, job_id: str) -> None:
        return None

    async def store_result(self, job_id: str, result) -> None:
        self.result = result

    async def store_failure(self, job_id: str, error: str) -> None:
        self.failure = error


def test_process_job_forwards_custom_stream_events(monkeypatch):
    async def no_heartbeat(queue, job_id, status_fn) -> None:
        return None

    monkeypatch.setattr(task_worker, "_heartbeat_loop", no_heartbeat)
    payload = EnqueuePayload(
        job_id="job-1",
        model="sales_lead_agent",
        conversation_id="conv-1",
        user_id="user-1",
        text="Find procurements",
        stream=True,
    )
    queue = _FakeQueue()

    asyncio.run(
        task_worker._process_job(
            payload=payload,
            client=_FakeClient(),
            queue=queue,
        )
    )

    event_types = [event.type for event in queue.events]
    assert event_types[:3] == ["status", "status", "custom"]
    assert queue.events[2].data["stage"] == "crawler_search"
    assert queue.result is not None
    assert queue.failure is None


def test_process_job_does_not_repeat_content_in_completed_event_after_chunks(monkeypatch):
    async def no_heartbeat(queue, job_id, status_fn) -> None:
        return None

    monkeypatch.setattr(task_worker, "_heartbeat_loop", no_heartbeat)
    payload = EnqueuePayload(
        job_id="job-2",
        model="sales_lead_agent",
        conversation_id="conv-2",
        user_id="user-2",
        text="Find transport",
        stream=True,
    )
    queue = _FakeQueue()

    asyncio.run(
        task_worker._process_job(
            payload=payload,
            client=_FakeClientWithChunk(),
            queue=queue,
        )
    )

    completed_event = queue.events[-1]
    assert completed_event.type == "completed"
    assert completed_event.metadata == {"conversation_id": "conv-2"}
    assert queue.result == {
        "conversation_id": "conv-2",
        "content": "Hello world",
        "response": {
            "agent_message": {
                "raw_text": "Hello world",
                "metadata": {},
            }
        },
    }


def test_process_job_does_not_replay_final_text_after_custom_then_chunk(monkeypatch):
    async def no_heartbeat(queue, job_id, status_fn) -> None:
        return None

    monkeypatch.setattr(task_worker, "_heartbeat_loop", no_heartbeat)
    payload = EnqueuePayload(
        job_id="job-3",
        model="gaz_agent",
        conversation_id="conv-3",
        user_id="user-3",
        text="Help me choose a fleet",
        stream=True,
    )
    queue = _FakeQueue()

    asyncio.run(
        task_worker._process_job(
            payload=payload,
            client=_FakeClientWithCustomThenChunk(),
            queue=queue,
        )
    )

    event_types = [event.type for event in queue.events]
    assert event_types == ["status", "status", "custom", "chunk", "completed"]
    chunk_events = [event for event in queue.events if event.type == "chunk"]
    assert len(chunk_events) == 1
    assert chunk_events[0].content == "Hello world"
