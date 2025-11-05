from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import router as api_router
from .agent_registry import agent_registry
from .config import settings
from .db import init_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_models()
    agent_registry.preload_all()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix=settings.api_base_path)

    @app.get("/healthz", tags=["system"])
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()
