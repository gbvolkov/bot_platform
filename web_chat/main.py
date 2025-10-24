from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    conv = await client.create_conversation(
        agent_id=agent_id,
        user_id=settings.default_user_id,
        user_role=payload.get("user_role") or settings.default_user_role,
        title=payload.get("title"),
    )
    return JSONResponse(conv)


@app.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    payload: dict,
    client: BotServiceClient = Depends(get_client),
) -> JSONResponse:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    response = await client.send_message(
        conversation_id=conversation_id,
        user_id=settings.default_user_id,
        user_role=settings.default_user_role,
        text=text,
        metadata=payload.get("metadata"),
    )
    return JSONResponse(response)
