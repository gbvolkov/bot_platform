from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from platform_application.service import Conflict, NotFound
from platform_sdk.schemas import ConversationCreate, MessageCreate
from .composition import compose


class ResumeRequest(BaseModel):
    interrupt_id: str
    response: str


def wire_event(event, *, conversation_id=None, saw_chunk=False):
    kind, payload = event["type"], event["payload"]
    if kind in {"chunk", "custom"}:
        return {"type": kind, **payload, "conversation_id": conversation_id}
    if kind in {"completed", "interrupt"}:
        result = payload.get("result") or {}
        message = result.get("agent_message") or {}
        metadata = dict(message.get("metadata") or {})
        value = {"type": kind, "metadata": metadata, "conversation_id": conversation_id}
        if not saw_chunk:
            value["content"] = message.get("raw_text", "")
        return value
    if kind in {"failed", "recovery_required", "cancelled"}:
        return {"type": "failed", "error": payload.get("error") or kind}
    return {"type": "status", "status": payload.get("status", kind)}


def create_app(settings=None, repository=None):
    settings, service, catalog, artifacts = compose(settings, repository)

    async def maintenance():
        while True:
            await service.call("maintenance")
            await asyncio.sleep(min(5, settings.lease_seconds / 3))

    @asynccontextmanager
    async def lifespan(app):
        await service.call("migrate")
        task = asyncio.create_task(maintenance())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await service.close()

    app = FastAPI(title="Bot Platform API", lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(NotFound)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        return await service.call("readiness")

    async def ready_ids(revision=None):
        state = await service.call("readiness")
        ids = set()
        for worker in state["workers"]:
            if worker["heartbeat"] >= time.time() - settings.lease_seconds:
                ids.update(k for k, available_revision in worker["ready"].items()
                           if (revision is not None and available_revision == revision) or
                           (revision is None and k in catalog.agents and catalog.agents[k].revision == available_revision))
        return ids

    async def legacy(method, path, user_id, user_role="default", body=None):
        async with httpx.AsyncClient(base_url=settings.legacy_url, timeout=None) as client:
            response = await client.request(method, path, json=body, headers={"X-User-Id": user_id, "X-User-Role": user_role})
            if response.is_error:
                raise HTTPException(response.status_code, response.text)
            return response

    @app.get("/api/agents/")
    async def agents():
        ready = await ready_ids()
        cohorts = settings.cohorts()
        entries = [{"id": a.id, "name": a.name, "description": a.description,
                    "provider": a.settings.get("provider", "openai"), "supported_content_types": list(a.supported_content_types)}
                   for a in catalog.agents.values() if a.active and a.id in ready and ("*" in cohorts or a.id in cohorts)]
        if "*" not in cohorts:
            try:
                response = await legacy("GET", "/agents/", "anonymous")
                entries += [a for a in response.json() if a["id"] not in cohorts]
            except (httpx.HTTPError, HTTPException):
                # Report worker readiness independently of legacy availability.
                pass
        return entries

    @app.post("/api/conversations/")
    async def create_conversation(body: ConversationCreate, x_user_id: Annotated[str, Header()] = "anonymous",
                                  x_user_role: Annotated[str, Header()] = "default"):
        try:
            descriptor = catalog.get(body.agent_id)
        except KeyError:
            raise NotFound("Agent not found")
        cohorts = settings.cohorts()
        if "*" not in cohorts and descriptor.id not in cohorts:
            response = await legacy("POST", "/conversations/", x_user_id, x_user_role, body.model_dump())
            value = response.json()
            # The legacy host writes its original database. This deployment must
            # point the coordinator at that same SQLite database before migration.
            await service.call("attach_legacy", value["id"])
            return JSONResponse(status_code=response.status_code, content=value)
        value = await service.call("create_conversation", descriptor, x_user_id, body.user_role or x_user_role,
                                   body.title, body.metadata)
        if descriptor.id not in await ready_ids():
            value["status"] = "pending"
            return JSONResponse(status_code=202, content=value)
        # Existing native adapter returned 200 on immediate readiness.
        return value

    @app.get("/api/conversations/")
    async def conversations(x_user_id: Annotated[str, Header()] = "anonymous"):
        values = await service.call("conversations", x_user_id)
        for value in values:
            assignment = await service.call("assignment", value["id"])
            if assignment["runtime"] == "worker" and value["status"] == "active" and value["agent_id"] not in await ready_ids(assignment["revision"]):
                value["status"] = "pending"
        return values

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(conversation_id: str, x_user_id: Annotated[str, Header()] = "anonymous"):
        value = await service.call("conversation", conversation_id, x_user_id, True)
        assignment = await service.call("assignment", conversation_id)
        if assignment["runtime"] == "legacy" and not assignment["closed"]:
            return (await legacy("GET", f"/conversations/{conversation_id}", x_user_id)).json()
        if value["status"] == "active" and value["agent_id"] not in await ready_ids(assignment["revision"]):
            value["status"] = "pending"
        return value

    async def accept(cid, user_id, body, key=None, resume=None):
        if body.payload.type == "reset" and body.payload.attachments:
            raise HTTPException(400, "Reset does not accept attachments")
        try:
            payload = await asyncio.to_thread(artifacts.stage, cid, user_id, body.payload.model_dump())
        except (ValueError, TypeError) as exc:
            await service.call("observe", "artifact_failures", increment=True)
            raise HTTPException(400, str(exc)) from exc
        return await service.submit(cid, user_id, payload, key, resume)

    @app.post("/api/conversations/{conversation_id}/runs", status_code=202)
    async def submit(conversation_id: str, body: MessageCreate, x_user_id: Annotated[str, Header()] = "anonymous",
                     idempotency_key: Annotated[str | None, Header()] = None):
        return await accept(conversation_id, x_user_id, body, idempotency_key)

    @app.post("/api/conversations/{conversation_id}/messages", status_code=201)
    async def message(conversation_id: str, body: MessageCreate, stream: bool = False,
                      x_user_id: Annotated[str, Header()] = "anonymous",
                      idempotency_key: Annotated[str | None, Header()] = None):
        conversation = await service.call("conversation", conversation_id, x_user_id)
        assignment = await service.call("assignment", conversation_id)
        if assignment["runtime"] == "worker" and conversation["agent_id"] not in await ready_ids(assignment["revision"]):
            raise HTTPException(409, "Agent is not ready")
        run = await accept(conversation_id, x_user_id, body, idempotency_key)
        if stream:
            async def events():
                saw_chunk = False
                async for event in service.follow(run["id"], x_user_id):
                    value = wire_event(event, conversation_id=conversation_id, saw_chunk=saw_chunk)
                    saw_chunk = saw_chunk or value["type"] == "chunk"
                    # Native legacy stream exposed chunks/custom/terminal only.
                    if value["type"] != "status":
                        yield "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"
                    if value["type"] == "failed":
                        return
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        completed = await service.wait(run["id"], x_user_id)
        if completed["status"] not in {"completed", "interrupted"}:
            raise HTTPException(502, completed["error"] or completed["status"])
        return {key: completed["result"][key] for key in ("conversation", "user_message", "agent_message")}

    @app.get("/api/runs/{run_id}")
    async def run(run_id: str, x_user_id: Annotated[str, Header()] = "anonymous"):
        return await service.call("get_run", run_id, x_user_id)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, after: int = 0, x_user_id: Annotated[str, Header()] = "anonymous"):
        if after < 0:
            raise HTTPException(400, "Event sequence must be nonnegative")
        await service.call("get_run", run_id, x_user_id)
        async def stream():
            async for event in service.follow(run_id, x_user_id, after):
                yield f"id: {event['sequence']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    async def resume(run_id: str, body: ResumeRequest, x_user_id: Annotated[str, Header()] = "anonymous",
                     idempotency_key: Annotated[str | None, Header()] = None):
        run = await service.call("get_run", run_id, x_user_id)
        if run["status"] != "interrupted":
            raise Conflict("Run is not interrupted")
        interrupt = (run.get("result") or {}).get("agent_message", {}).get("metadata", {}).get("interrupt_payload", {})
        if interrupt.get("interrupt_id") != body.interrupt_id:
            raise Conflict("Interrupt does not belong to this run")
        return await service.submit(run["conversation_id"], x_user_id, {"text": body.response},
                                    idempotency_key, body.model_dump())

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str, x_user_id: Annotated[str, Header()] = "anonymous"):
        return await service.call("cancel", run_id, x_user_id)

    @app.post("/api/conversations/{conversation_id}/close")
    async def close(conversation_id: str, x_user_id: Annotated[str, Header()] = "anonymous"):
        return await service.call("close_conversation", conversation_id, x_user_id)

    @app.get("/api/artifacts/{artifact_id}")
    async def artifact(artifact_id: str, x_user_id: Annotated[str, Header()] = "anonymous"):
        return await asyncio.to_thread(artifacts.get, artifact_id, x_user_id)

    return app


app = create_app()
