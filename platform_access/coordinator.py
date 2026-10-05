"""Private listener; deploy only on the worker network, never the public port."""
import asyncio
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from platform_application.service import Conflict, NotFound
from .composition import compose


class Worker(BaseModel):
    generation: str
    execution_class: str
    ready: dict[str, str] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


class Claim(BaseModel):
    generation: str


class Attempt(Claim):
    attempt_id: str


class Event(Attempt):
    source_sequence: int
    event_type: str
    payload: dict


class Completion(Attempt):
    status: str
    result: dict | None = None
    error: str | None = None


class ArtifactUpload(Attempt):
    payload: dict


class Reconciliation(BaseModel):
    checkpoint_reference: str | None
    generation: str


def create_app(settings=None, repository=None):
    settings, service, _, artifacts = compose(settings, repository)

    async def authenticate(authorization: Annotated[str | None, Header()] = None):
        if not settings.service_token or not secrets.compare_digest(authorization or "", "Bearer " + settings.service_token):
            raise HTTPException(401, "Invalid service credential")

    @asynccontextmanager
    async def lifespan(app):
        if not settings.service_token:
            raise RuntimeError("PLATFORM_SERVICE_TOKEN is required for worker coordination")
        await service.call("migrate")
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(authenticate)])

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(NotFound)
    async def missing(request, error):
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.post("/workers/register")
    async def register(body: Worker):
        await service.call("register_worker", **body.model_dump())
        return {"lease_seconds": settings.lease_seconds}

    @app.post("/runs/claim")
    async def claim(body: Claim):
        return await service.call("claim", body.generation)

    @app.post("/runs/{run_id}/start")
    async def start(run_id: str, body: Attempt):
        return await service.call("start", run_id, **body.model_dump())

    @app.post("/runs/{run_id}/heartbeat")
    async def heartbeat(run_id: str, body: Attempt):
        return await service.call("heartbeat", run_id, **body.model_dump())

    @app.post("/runs/{run_id}/events")
    async def event(run_id: str, body: Event):
        return {"sequence": await service.call("append_event", run_id, **body.model_dump())}

    @app.post("/runs/{run_id}/finish")
    async def finish(run_id: str, body: Completion):
        run = await service.call("get_run", run_id)
        value = body.model_dump()
        value["result"] = await asyncio.to_thread(artifacts.materialize_result, run["conversation_id"], body.result)
        return await service.call("finish", run_id, **value)

    @app.post("/runs/{run_id}/artifacts")
    async def upload(run_id: str, body: ArtifactUpload):
        try:
            return await asyncio.to_thread(artifacts.upload, run_id, **body.model_dump())
        except (ValueError, OSError):
            await service.call("observe", "artifact_failures", increment=True)
            raise

    @app.post("/runs/{run_id}/reconcile")
    async def reconcile(run_id: str, body: Reconciliation):
        return await service.call("reconcile", run_id, **body.model_dump())

    @app.get("/runs/{run_id}/artifacts/{artifact_id}")
    async def artifact(run_id: str, artifact_id: str):
        run = await service.call("get_run", run_id)
        if artifact_id not in run["input"].get("artifact_refs", []):
            raise NotFound("Artifact is not part of this run")
        return await asyncio.to_thread(artifacts.get, artifact_id, conversation_id=run["conversation_id"])

    return app


app = create_app()
