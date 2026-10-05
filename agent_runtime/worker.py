"""Single-slot worker. SQLite/coordinator owns dispatch and all run state."""
import asyncio
import contextlib
import logging
import signal
import uuid
import copy

import httpx

from platform_client.coordinator import CoordinatorClient
from platform_infrastructure.settings import PlatformSettings
from platform_contracts import ArtifactError

LOG = logging.getLogger(__name__)


class Worker:
    def __init__(self, client, runtime, execution_class="interactive", generation=None, notifications=None):
        self.client, self.runtime = client, runtime
        self.execution_class = execution_class
        self.generation = generation or str(uuid.uuid4())
        self.ready, self.errors = {}, {}
        self.lease_seconds = 30
        self.notifications = notifications

    async def register(self):
        response = await self.client.post("/workers/register", {"generation": self.generation,
            "execution_class": self.execution_class, "ready": self.ready, "errors": self.errors})
        self.lease_seconds = response["lease_seconds"]

    async def export_result(self, path, identity, result):
        if result is None:
            return None
        result = copy.deepcopy(result)
        refs = {}
        async def upload(payload):
            reference = await self.client.post(path + "/artifacts", {**identity, "payload": payload})
            refs[reference["id"]] = reference
            return {"artifact_id": reference["id"]}
        for artifact in result.pop("artifact_payloads", []):
            await upload(artifact)
        message = result.get("agent_message") or {}
        attachments = message.get("metadata", {}).get("attachments", [])
        if attachments:
            message["metadata"]["attachments"] = [await upload(item) for item in attachments]
        content = message.get("content") or {}
        if isinstance(content, dict):
            for index, part in enumerate(content.get("parts", [])):
                if isinstance(part, dict) and part.get("type") in {"file", "image", "audio", "video", "attachment"}:
                    content["parts"][index] = await upload(part)
        if refs:
            result["artifacts"] = list(refs.values())
        return result

    async def process(self, run):
        identity = {"attempt_id": run["attempt_id"], "generation": self.generation}
        path = f"/runs/{run['id']}"
        # Materialize payloads before the execution-start fence. A failed fetch
        # leaves a reclaimable unstarted claim; no graph has been invoked.
        attachments = [await self.client.artifact(run["id"], aid) for aid in run["input"].get("artifact_refs", [])]
        run = {**run, "input": {**run["input"], "attachments": attachments}}
        await self.client.post(path + "/start", identity)
        source_sequence = 0

        async def emit(event):
            nonlocal source_sequence
            source_sequence += 1
            await self.client.post(path + "/events", {**identity, "source_sequence": source_sequence,
                "event_type": event["type"], "payload": {k: v for k, v in event.items() if k != "type"}})

        execution = asyncio.create_task(self.runtime.execute(run, emit))
        requested_cancel = False
        ownership_lost = False

        async def heartbeat():
            nonlocal requested_cancel
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                state = await self.client.post(path + "/heartbeat", identity)
                if state["cancel_requested"]:
                    requested_cancel = True
                    if getattr(self.runtime, "cooperative_cancellation", False):
                        execution.cancel()
                        return

        pulse = asyncio.create_task(heartbeat())
        status, result, error = "recovery_required", None, None
        try:
            done, _ = await asyncio.wait({execution, pulse}, return_when=asyncio.FIRST_COMPLETED)
            if pulse in done and not requested_cancel:
                ownership_lost = True
                pulse.result()
            try:
                status, result = await asyncio.shield(execution)
                result = await self.export_result(path, identity, result)
                error = (result or {}).get("error")
                if requested_cancel and getattr(self.runtime, "cancel_at_completion", True):
                    status = "cancelled"
            except asyncio.CancelledError:
                if not requested_cancel:
                    raise
                status = "cancelled"
        except asyncio.CancelledError:
            error = "Worker stopped during execution; checkpoint reconciliation required"
        except Exception as exc:
            status = "recovery_required" if ownership_lost else "failed"
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ArtifactError):
                result = {"error_category": "artifact"}
        finally:
            # Shutdown or a lost lease must not release a slot while a blocking
            # graph tool is still executing. SIGTERM drains the current turn.
            await asyncio.gather(execution, return_exceptions=True)
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
        await self.client.post(path + "/finish", {**identity, "status": status, "result": result, "error": error})

    async def run(self, stop):
        async def progress(ready, errors):
            self.ready, self.errors = ready, errors
        initialization = asyncio.create_task(self.runtime.initialize(self.execution_class, progress=progress))
        while not stop.is_set():
            try:
                if initialization.done():
                    self.ready, self.errors = initialization.result()
                await self.register()
                run = await self.client.post("/runs/claim", {"generation": self.generation})
                if run:
                    await self.process(run)
                else:
                    try:
                        if self.notifications is None:
                            await asyncio.wait_for(stop.wait(), timeout=1)
                        else:
                            await self.notifications.wait(1)
                    except TimeoutError:
                        pass
            except httpx.HTTPError:
                LOG.exception("Coordinator communication failed; claims remain durable")
                await asyncio.sleep(1)
        await asyncio.gather(initialization, return_exceptions=True)


async def main():
    settings = PlatformSettings()
    client = CoordinatorClient(settings.coordinator_url, settings.service_token)
    from platform_infrastructure.notifications import RedisNotifications
    notifications = RedisNotifications(settings.redis_url) if settings.redis_url else None
    if settings.execution_class == "legacy":
        from .legacy import LegacyRuntime
        runtime = LegacyRuntime(settings.legacy_url)
    else:
        from .runtime import AgentRuntime
        from platform_capabilities.models import get_llm
        runtime = AgentRuntime(settings.catalog_path, capabilities={"models": get_llm})
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    try:
        await Worker(client, runtime, settings.execution_class, notifications=notifications).run(stop)
    finally:
        await runtime.close()
        await client.close()
        if notifications is not None:
            await notifications.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
