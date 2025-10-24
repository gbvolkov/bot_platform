from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import httpx
from services.bot_client import BotServiceClient
from .config import settings


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

bot_client = BotServiceClient(
    base_url=str(settings.bot_service_base_url),
    request_timeout=settings.request_timeout_seconds,
    connect_timeout=settings.connect_timeout_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bot_client.startup()
    try:
        yield
    finally:
        await bot_client.shutdown()


def get_client() -> BotServiceClient:
    return bot_client


app = FastAPI(title="Lightweight Web Chat", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/agents")
async def list_agents(client: BotServiceClient = Depends(get_client)) -> JSONResponse:
    agents = await client.list_agents()
    return JSONResponse(agents)


@app.post("/api/conversations")
async def create_conversation(payload: dict, client: BotServiceClient = Depends(get_client)) -> JSONResponse:
    agent_id = payload.get("agent_id")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    await client.ensure_agent(agent_id)
    conv, ready = await client.create_conversation(
        agent_id=agent_id,
        user_id=settings.default_user_id,
        user_role=payload.get("user_role") or settings.default_user_role,
        title=payload.get("title"),
    )
    return JSONResponse(conv, status_code=201 if ready else 202)


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    payload: dict,
    client: BotServiceClient = Depends(get_client),
) -> JSONResponse:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        response = await client.send_message(
            conversation_id=conversation_id,
            user_id=settings.default_user_id,
            user_role=settings.default_user_role,
            text=text,
            metadata=payload.get("metadata"),
        )
    except httpx.HTTPStatusError as exc:  # pragma: no cover - passthrough
        if exc.response.status_code == httpx.codes.CONFLICT:
            raise HTTPException(status_code=409, detail="Conversation is still initializing.") from exc
        detail = exc.response.json().get("detail") if exc.response.headers.get("content-type", "").startswith("application/json") else exc.response.text
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    return JSONResponse(response)


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_detail(
    conversation_id: str,
    client: BotServiceClient = Depends(get_client),
) -> JSONResponse:
    try:
        response = await client.get_conversation(conversation_id, settings.default_user_id)
    except httpx.HTTPStatusError as exc:  # pragma: no cover
        detail = exc.response.json().get("detail") if exc.response.headers.get("content-type", "").startswith("application/json") else exc.response.text
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    return JSONResponse(response)
