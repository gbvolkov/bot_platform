from __future__ import annotations

import asyncio
import base64
import os
import secrets
from typing import Annotated
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from platform_capabilities.analysis_runner import DockerAnalysisRunner, AnalysisUnavailable

app = FastAPI(docs_url=None, redoc_url=None)
slots = asyncio.Semaphore(3)
jobs: dict[str, asyncio.Task] = {}


class Job(BaseModel):
    id: str = Field(pattern="^[a-f0-9]{32}$")
    code: str = Field(max_length=65536)
    files: dict[str, str]


def authenticate(authorization: str | None):
    token = os.environ.get("PLATFORM_ANALYSIS_TOKEN", "")
    if not token or not secrets.compare_digest(authorization or "", "Bearer " + token):
        raise HTTPException(401, "Invalid analysis credential")


def launcher() -> DockerAnalysisRunner:
    return DockerAnalysisRunner(os.environ.get("PLATFORM_ANALYSIS_IMAGE", "bot-platform-analysis:mentor-v1"),
        staging_root=os.environ.get("PLATFORM_ANALYSIS_STAGING_ROOT"))


@app.get("/identity")
async def identity(authorization: Annotated[str | None, Header()] = None):
    authenticate(authorization)
    try:
        return {"environment": await launcher().identity()}
    except AnalysisUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.delete("/analyze/{job_id}")
async def cancel(job_id: str, authorization: Annotated[str | None, Header()] = None):
    authenticate(authorization)
    task = jobs.get(job_id)
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return {"cancelled": True}


@app.post("/analyze")
async def analyze(body: Job, authorization: Annotated[str | None, Header()] = None):
    authenticate(authorization)
    if sum(map(len, body.files.values())) > 45 * 1024 * 1024:
        raise HTTPException(413, "Job input exceeds limit")
    try:
        files = {name: base64.b64decode(value, validate=True) for name, value in body.files.items()}
        if body.id in jobs:
            raise HTTPException(409, "Job already exists")
        async def execute():
            await asyncio.wait_for(slots.acquire(), 5)
            try:
                return await launcher().run(body.code, files)
            finally:
                slots.release()
        task = asyncio.create_task(execute())
        jobs[body.id] = task
        try:
            try:
                return await task
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                raise HTTPException(409, "Analysis job cancelled") from None
        finally:
            jobs.pop(body.id, None)
    except AnalysisUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ValueError, TimeoutError) as exc:
        raise HTTPException(422, str(exc)) from exc
