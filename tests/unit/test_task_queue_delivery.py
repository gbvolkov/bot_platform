import asyncio
from contextlib import asynccontextmanager

import pytest

from services.task_queue.redis_queue import RedisTaskQueue


class Queue(RedisTaskQueue):
    def __init__(self, snapshot, messages=()):
        self.snapshot = snapshot
        self.messages = list(messages)
        self.closed = False
        self.reads = 0

    @asynccontextmanager
    async def subscribe(self, job_id):
        try:
            yield self
        finally:
            self.closed = True

    async def get_status(self, job_id):
        self.reads += 1
        return self.snapshot(self.reads) if callable(self.snapshot) else self.snapshot

    async def get_message(self, **kwargs):
        return self.messages.pop(0) if self.messages else None


@pytest.mark.parametrize("stage,event", [("completed", "completed"), ("failed", "failed"), ("interrupted", "interrupt")])
def test_late_subscriber_receives_terminal_and_closes(stage, event):
    queue = Queue({"status": stage, "result": {"content": "Done", "interrupt_id": "i"}, "error": "Broken"})
    async def run():
        events = [item async for item in queue.iter_events("job", include_status_snapshot=True)]
        assert len(events) == 1
        assert events[0].type == event
        assert queue.closed
        assert (await queue.wait_for_completion("job", timeout=0.5)).type == event
    asyncio.run(run())


def test_missed_publish_after_partial_stream_reconciles_without_repeating_text():
    queue = Queue(lambda n: {"status": "running"} if n == 1 else
                  {"status": "completed", "result": {"content": "Hello world"}},
                  [{"type": "message", "data": '{"job_id":"job","type":"chunk","content":"Hello "}'}])
    async def run():
        events = [item async for item in queue.iter_events("job")]
        assert [e.type for e in events] == ["chunk", "completed"]
        assert events[-1].metadata["content"] == "world"
        assert queue.closed
    asyncio.run(run())
