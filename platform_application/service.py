import asyncio
from typing import Any, Protocol

from platform_contracts import TERMINAL


class Conflict(ValueError):
    pass


class NotFound(LookupError):
    pass


class Repository(Protocol):
    def submit(self, conversation_id: str, user_id: str, payload: dict,
               idempotency_key: str | None = None, resume: dict | None = None) -> dict: ...
    def get_run(self, run_id: str, user_id: str | None = None) -> dict: ...
    def events(self, run_id: str, after: int = 0) -> list[dict]: ...
    def cancel(self, run_id: str, user_id: str) -> dict: ...
    def close_conversation(self, conversation_id: str, user_id: str) -> dict: ...
    def create_conversation(self, descriptor, user_id, user_role, title=None, metadata=None, runtime="worker", conversation_id=None) -> dict: ...
    def conversation(self, conversation_id, user_id, detail=False) -> dict: ...
    def conversations(self, user_id) -> list[dict]: ...
    def assignment(self, conversation_id) -> dict: ...
    def attach_legacy(self, conversation_id) -> None: ...
    def register_worker(self, generation, execution_class, ready, errors) -> None: ...
    def claim(self, generation) -> dict | None: ...
    def start(self, run_id, attempt_id, generation) -> dict: ...
    def heartbeat(self, run_id, attempt_id, generation) -> dict: ...
    def append_event(self, run_id, attempt_id, generation, source_sequence, event_type, payload) -> int: ...
    def finish(self, run_id, attempt_id, generation, status, result=None, error=None) -> dict: ...
    def reconcile(self, run_id, *, checkpoint_reference, generation) -> dict: ...
    def migrate(self) -> None: ...
    def maintenance(self) -> None: ...
    def readiness(self) -> dict: ...
    def observe(self, name, value=1, *, increment=False) -> None: ...


class Notifications(Protocol):
    async def publish(self) -> None: ...
    async def wait(self, timeout: float) -> None: ...
    async def close(self) -> None: ...


class Application:
    def __init__(self, repository: Repository, poll_seconds: float = 0.25, notifications: Notifications | None = None):
        self.repository = repository
        self.poll_seconds = poll_seconds
        self.notifications = notifications

    async def call(self, operation: str, *args, **kwargs) -> Any:
        result = await asyncio.to_thread(getattr(self.repository, operation), *args, **kwargs)
        if self.notifications is not None and operation in {"submit", "finish", "append_event", "cancel", "reconcile"}:
            await self.notifications.publish()
        return result

    async def close(self):
        if self.notifications is not None:
            await self.notifications.close()

    async def submit(self, conversation_id, user_id, payload, idempotency_key=None, resume=None):
        return await self.call("submit", conversation_id, user_id, payload, idempotency_key, resume)

    async def follow(self, run_id, user_id, after=0):
        await self.call("get_run", run_id, user_id)
        while True:
            events = await self.call("events", run_id, after)
            for event in events:
                after = event["sequence"]
                yield event
            if events:
                import time
                await self.call("observe", "event_delivery_lag_seconds", max(0, time.time() - events[-1]["created_at"]))
            run = await self.call("get_run", run_id, user_id)
            if run["status"] in TERMINAL:
                # Completion and its terminal event commit in one transaction.
                # Reread when completion raced the preceding event query.
                if after < run["event_sequence"]:
                    continue
                return
            if self.notifications is None:
                await asyncio.sleep(self.poll_seconds)
            else:
                await self.notifications.wait(self.poll_seconds)

    async def wait(self, run_id, user_id):
        async for _ in self.follow(run_id, user_id):
            pass
        return await self.call("get_run", run_id, user_id)
