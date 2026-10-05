"""Preserve the proxy's queue-event format over durable application runs."""
import asyncio
import json
import time
from collections import OrderedDict

import httpx

from platform_sdk.queue import QueueEvent
from platform_contracts import TERMINAL


class DurableQueueClient:
    def __init__(self, base_url, *, transport=None):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=None, transport=transport)
        self.jobs = OrderedDict()

    async def startup(self):
        return None

    async def shutdown(self):
        self.jobs.clear()
        await self.client.aclose()

    async def managed_stream(self, events, job_id):
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()
            self.release(job_id)

    async def enqueue(self, payload):
        metadata = dict(payload.metadata or {})
        if payload.raw_user_text:
            metadata["raw_user_text"] = payload.raw_user_text
        headers = {"X-User-Id": payload.user_id, "Idempotency-Key": payload.job_id}
        if payload.user_role:
            headers["X-User-Role"] = payload.user_role
        response = await self.client.post(f"/conversations/{payload.conversation_id}/runs", headers=headers,
            json={"payload": {"type": "text", "text": payload.text, "metadata": metadata, "attachments": payload.attachments or []}})
        response.raise_for_status()
        self.jobs[payload.job_id] = (response.json()["id"], payload.user_id, time.monotonic())
        # Entries only locate a request's durable run. No execution/result state
        # depends on this cache; detached callers use the native run endpoint.
        expired = [key for key, value in self.jobs.items() if time.monotonic()-value[2] > 21600]
        for key in expired:
            self.jobs.pop(key, None)

    async def get_status(self, job_id):
        run_id, user_id, _ = self.jobs[job_id]
        response = await self.client.get(f"/runs/{run_id}", headers={"X-User-Id": user_id})
        response.raise_for_status()
        run = response.json()
        result = run.get("result") or {}
        message = result.get("agent_message") or {}
        return {"status": run["status"], "error": run.get("error"), "result": {
            "conversation_id": run["conversation_id"], "content": message.get("raw_text", ""),
            "response": result, "attachments": message.get("metadata", {}).get("attachments", [])}}

    async def iter_events(self, job_id, *, include_status_snapshot=False):
        run_id, user_id, _ = self.jobs[job_id]
        saw_chunk = False
        async with self.client.stream("GET", f"/runs/{run_id}/events", headers={"X-User-Id": user_id}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                kind, payload = event["type"], event["payload"]
                if kind in {"chunk", "custom"}:
                    saw_chunk = saw_chunk or kind == "chunk"
                    yield QueueEvent(job_id=job_id, type=kind, **payload)
                elif kind in {"completed", "interrupt"}:
                    result = payload.get("result") or {}
                    message = result.get("agent_message") or {}
                    metadata = dict(message.get("metadata") or {})
                    metadata["conversation_id"] = result.get("conversation", {}).get("id")
                    if not saw_chunk:
                        metadata["content"] = message.get("raw_text", "")
                    elif kind == "completed":
                        metadata.pop("content", None)
                    yield QueueEvent(job_id=job_id, type=kind, status="interrupted" if kind == "interrupt" else "completed", metadata=metadata)
                elif kind in {"failed", "recovery_required", "cancelled"}:
                    yield QueueEvent(job_id=job_id, type="failed", status="failed", error=payload.get("error") or kind)
                elif kind == "status" and include_status_snapshot:
                    stage = payload.get("status")
                    yield QueueEvent(job_id=job_id, type="status", status="running" if stage in {"claimed", "cancelling"} else stage)

    async def wait_for_completion(self, job_id, timeout=None):
        async def wait():
            async for event in self.iter_events(job_id):
                if event.type in {"completed", "interrupt", "failed"}:
                    return event
            raise RuntimeError("Run event stream ended without a terminal event")
        return await asyncio.wait_for(wait(), timeout)

    def release(self, job_id):
        self.jobs.pop(job_id, None)
