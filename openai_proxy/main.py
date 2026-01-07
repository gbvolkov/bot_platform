from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from services.task_queue import RedisTaskQueue
from services.task_queue.config import settings as task_queue_settings
from services.task_queue.models import EnqueuePayload

from .client import BotServiceClient
from .config import settings
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)
from .utils import build_prompt
from pydantic import ValidationError


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_LOGGED_BODY_BYTES = 4096
MAX_REMOTE_ATTACHMENT_BYTES = 25 * 1024 * 1024


def _clean_content_type(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.split(";", 1)[0].strip()
    return cleaned or None


def _parse_data_url(url: str) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    header, sep, payload = url.partition(",")
    if not sep:
        return None, None
    meta = header[5:]
    mime = None
    is_base64 = False
    if meta:
        parts = [part.strip() for part in meta.split(";") if part.strip()]
        if parts:
            mime = parts[0]
        is_base64 = "base64" in parts[1:]
    if not is_base64:
        return None, None
    return (mime or None), (payload.strip() or None)


def _guess_content_type(filename: Optional[str], url: Optional[str]) -> Optional[str]:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    if url:
        path = urlparse(url).path
        if path:
            guessed, _ = mimetypes.guess_type(path)
            if guessed:
                return guessed
    return None


def _ensure_filename(attachment: Dict[str, Any]) -> None:
    if attachment.get("filename"):
        return
    content_type = _clean_content_type(attachment.get("content_type"))
    ext = None
    if content_type:
        ext = mimetypes.guess_extension(content_type)
    if not ext:
        url_value = attachment.get("url")
        if isinstance(url_value, str) and url_value:
            ext = Path(urlparse(url_value).path).suffix or None
    if not ext:
        ext = ".bin"
    if not ext.startswith("."):
        ext = f".{ext}"
    attachment["filename"] = f"attachment_{uuid.uuid4().hex}{ext}"


async def _fetch_url_data(url: str) -> tuple[Optional[str], Optional[str]]:
    timeout = httpx.Timeout(15.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = _clean_content_type(response.headers.get("content-type"))
            buffer = bytearray()
            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)
                if len(buffer) > MAX_REMOTE_ATTACHMENT_BYTES:
                    raise ValueError(f"Remote attachment exceeds {MAX_REMOTE_ATTACHMENT_BYTES} bytes")
            data = base64.b64encode(bytes(buffer)).decode("ascii")
            return content_type, data


async def _hydrate_attachments(attachments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        item = dict(attachment)
        _ensure_filename(item)
        if item.get("data") or item.get("text"):
            item.pop("url", None)
            if not item.get("content_type"):
                item["content_type"] = _guess_content_type(item.get("filename"), None)
            hydrated.append(item)
            continue
        url_value = item.get("url")
        if isinstance(url_value, str) and url_value:
            mime, data = _parse_data_url(url_value)
            if data:
                item["data"] = data
                if mime and not item.get("content_type"):
                    item["content_type"] = mime
                item.pop("url", None)
                hydrated.append(item)
                continue
            if url_value.startswith(("http://", "https://")):
                mime, data = await _fetch_url_data(url_value)
                if data:
                    item["data"] = data
                if mime and not item.get("content_type"):
                    item["content_type"] = mime
                item.pop("url", None)
                if not item.get("content_type"):
                    item["content_type"] = _guess_content_type(item.get("filename"), url_value)
                hydrated.append(item)
                continue
        hydrated.append(item)
    return hydrated


bot_client = BotServiceClient(
    base_url=str(settings.bot_service_base_url),
    request_timeout=settings.request_timeout_seconds,
    connect_timeout=settings.connect_timeout_seconds,
)
task_queue = RedisTaskQueue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.gather(bot_client.startup(), task_queue.startup())
    try:
        yield
    finally:
        await asyncio.gather(bot_client.shutdown(), task_queue.shutdown())


def get_client() -> BotServiceClient:
    return bot_client


app = FastAPI(
    title="OpenAI Compatible Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body_bytes = await request.body()
    request.state.raw_body = body_bytes
    truncated_body = body_bytes[:MAX_LOGGED_BODY_BYTES]

    redacted_headers = {
        key: ("<redacted>" if key.lower() in {"authorization", "x-api-key"} else value)
        for key, value in request.headers.items()
    }
    logger.debug(
        "Incoming request method=%s path=%s headers=%s body=%s",
        request.method,
        request.url.path,
        redacted_headers,
        truncated_body.decode("utf-8", errors="replace"),
    )

    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Outgoing response method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body_bytes = await request.body()
    logger.error(
        "Request validation error path=%s client=%s errors=%s body=%s",
        request.url.path,
        request.client,
        exc.errors(),
        body_bytes[:MAX_LOGGED_BODY_BYTES].decode("utf-8", errors="replace"),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.get("/healthz", tags=["system"])
async def health_check():
    return {"status": "ok"}


def agent_to_model_card(agent: Dict[str, Any]) -> ModelCard:
    owned_by = agent.get("provider") or "bot-service"
    metadata = {key: value for key, value in agent.items() if key not in {"id"}}
    return ModelCard(
        id=agent["id"],
        name=agent.get("name"),
        description=agent.get("description"),
        provider=agent.get("provider"),
        owned_by=owned_by,
        metadata=metadata,
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models(
    client: Annotated[BotServiceClient, Depends(get_client)],
) -> ModelList:
    agents = await client.list_agents()
    models = [agent_to_model_card(agent) for agent in agents]
    return ModelList(data=models)


@app.get("/v1/models/{model_id}", response_model=ModelCard)
async def retrieve_model(
    model_id: str,
    client: Annotated[BotServiceClient, Depends(get_client)],
) -> ModelCard:
    try:
        agent = await client.get_agent(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return agent_to_model_card(agent)


def _build_sse_payload(
    *,
    model: str,
    job_id: str,
    conversation_id: str,
    delta: Dict[str, Any],
    finish_reason: str | None = None,
    agent_status: str | None = None,
    usage: Dict[str, Any] | None = None,
    message_metadata: Dict[str, Any] | None = None,
) -> str:
    payload: Dict[str, Any] = {
        "id": job_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
        "conversation_id": conversation_id,
    }
    if agent_status:
        payload["agent_status"] = agent_status
    if usage:
        payload["usage"] = usage
    if message_metadata:
        payload["message_metadata"] = message_metadata
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_events(
    *,
    job_id: str,
    model: str,
    conversation_id: str,
) -> AsyncIterator[str]:
    role_announced = False
    terminal_seen = False
    async for event in task_queue.iter_events(job_id, include_status_snapshot=True):
        if event.type == "status":
            logger.debug("SSE status pulse job_id=%s status=%s", job_id, event.status)
            yield _build_sse_payload(
                model=model,
                job_id=job_id,
                conversation_id=conversation_id,
                delta={},
                agent_status=event.status or "running",
            )
        elif event.type == "chunk":
            if not role_announced:
                logger.debug("SSE role announcement job_id=%s", job_id)
                yield _build_sse_payload(
                    model=model,
                    job_id=job_id,
                    conversation_id=conversation_id,
                    delta={"role": "assistant"},
                )
                role_announced = True
            if event.content:
                logger.debug(
                    "SSE chunk job_id=%s size=%d",
                    job_id,
                    len(event.content),
                )
                yield _build_sse_payload(
                    model=model,
                    job_id=job_id,
                    conversation_id=conversation_id,
                    delta={"content": event.content},
                )
        elif event.type == "completed":
            if not role_announced:
                logger.debug("SSE late role announcement job_id=%s", job_id)
                yield _build_sse_payload(
                    model=model,
                    job_id=job_id,
                    conversation_id=conversation_id,
                    delta={"role": "assistant"},
                )
                role_announced = True
            metadata_payload: Dict[str, Any] | None = None
            if event.metadata:
                attachments = event.metadata.get("attachments")
                if attachments:
                    metadata_payload = {"attachments": attachments}
            yield _build_sse_payload(
                model=model,
                job_id=job_id,
                conversation_id=conversation_id,
                delta={},
                finish_reason="stop",
                agent_status="completed",
                usage=event.usage or {},
                message_metadata=metadata_payload,
            )
            logger.debug("SSE completed job_id=%s usage=%s", job_id, event.usage)
            terminal_seen = True
            break
        elif event.type == "heartbeat":
            logger.debug("SSE heartbeat job_id=%s status=%s", job_id, event.status)
            yield f": heartbeat {event.status or ''}\n\n"
        elif event.type == "interrupt":
            metadata = event.metadata or {}
            content = metadata.get("question") or metadata.get("content") or ""
            message_metadata = metadata or None
            if not role_announced:
                yield _build_sse_payload(
                    model=model,
                    job_id=job_id,
                    conversation_id=conversation_id,
                    delta={"role": "assistant"},
                )
                role_announced = True
            yield _build_sse_payload(
                model=model,
                job_id=job_id,
                conversation_id=conversation_id,
                delta={"content": content},
                finish_reason="stop",
                agent_status="interrupted",
                message_metadata=message_metadata,
            )
            terminal_seen = True
            break
        elif event.type == "failed":
            error_payload = {
                "error": {
                    "message": event.error or "Agent execution failed.",
                    "type": "agent_error",
                },
                "conversation_id": conversation_id,
                "job_id": job_id,
            }
            logger.error("SSE failure job_id=%s error=%s", job_id, event.error)
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            terminal_seen = True
            break
    if not terminal_seen:
        status_snapshot = await task_queue.get_status(job_id)
        status = status_snapshot.get("status", "unknown")
        logger.debug("SSE fallback status job_id=%s status=%s", job_id, status)
        yield _build_sse_payload(
            model=model,
            job_id=job_id,
            conversation_id=conversation_id,
            delta={},
            agent_status=status,
        )
    logger.debug("SSE stream done job_id=%s", job_id)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    client: Annotated[BotServiceClient, Depends(get_client)],
) -> ChatCompletionResponse:
    try:
        await client.ensure_agent(request.model)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    user_id = request.user or settings.default_user_id
    user_role = settings.default_user_role

    logger.debug(
        "chat.completions request model=%s stream=%s messages=%d conversation_id=%s user_id=%s",
        request.model,
        bool(request.stream),
        len(request.messages),
        request.conversation_id,
        user_id,
    )

    conversation_id = request.conversation_id
    if conversation_id is None:
        conv, ready = await client.create_conversation(
            agent_id=request.model,
            user_id=user_id,
            user_role=user_role,
        )
        conversation_id = conv["id"]
        if not ready:
            for _ in range(30):
                await asyncio.sleep(1.0)
                detail = await client.get_conversation(conversation_id, user_id)
                if detail.get("status") == "active":
                    ready = True
                    break
            if not ready:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Agent is initializing. Retry the request shortly.",
                )

    try:
        prompt, default_prompt_used = build_prompt(
            request.messages,
            default_user_prompt=settings.default_attachment_prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    logger.debug("Built prompt job model=%s chars=%d", request.model, len(prompt))

    raw_user_text = ""
    for message in reversed(request.messages):
        if message.role == "user":
            raw_user_text = message.content or ""
            break
    if default_prompt_used and not raw_user_text:
        raw_user_text = settings.default_attachment_prompt or ""

    latest_attachments: List[Dict[str, Any]] = []
    for message in reversed(request.messages):
        if message.role == "user":
            if message.attachments:
                latest_attachments = [
                    attachment.model_dump(exclude_none=True)
                    for attachment in message.attachments
                ]
            break
    if latest_attachments:
        try:
            latest_attachments = await _hydrate_attachments(latest_attachments)
        except Exception as exc:
            logger.error("Attachment hydration failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to prepare attachments: {exc}",
            ) from exc
        invalid = [
            attachment.get("filename")
            for attachment in latest_attachments
            if not attachment.get("data") and not attachment.get("text")
        ]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Attachments missing data or text: {', '.join(invalid)}",
            )

    if default_prompt_used:
        logger.debug(
            "Applied default attachment prompt for conversation_id=%s model=%s",
            conversation_id,
            request.model,
        )

    job_id = f"job-{uuid.uuid4().hex}"
    payload = EnqueuePayload(
        job_id=job_id,
        model=request.model,
        conversation_id=conversation_id,
        user_id=user_id,
        user_role=user_role,
        text=prompt,
        raw_user_text=raw_user_text or None,
        attachments=latest_attachments or None,
    )
    await task_queue.enqueue(payload)
    logger.debug(
        "Enqueued job job_id=%s conversation_id=%s attachments=%d stream=%s",
        job_id,
        conversation_id,
        len(latest_attachments),
        bool(request.stream),
    )

    if request.stream:
        logger.debug("Starting streaming response job_id=%s", job_id)
        return StreamingResponse(
            _stream_events(job_id=job_id, model=request.model, conversation_id=conversation_id),
            media_type="text/event-stream",
        )

    logger.debug("Awaiting completion job_id=%s", job_id)
    completion_event = await task_queue.wait_for_completion(
        job_id=job_id,
        timeout=task_queue_settings.completion_wait_timeout_seconds,
    )
    if completion_event.type == "failed":
        logger.error(
            "Job failed job_id=%s error=%s",
            job_id,
            completion_event.error,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=completion_event.error or "Agent execution failed.",
        )

    if completion_event.type == "interrupt":
        metadata = completion_event.metadata or {}
        content = metadata.get("question") or metadata.get("content") or ""
        message_metadata: Dict[str, Any] | None = None
        if metadata:
            message_metadata = metadata
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessageResponse(content=content, metadata=message_metadata),
            finish_reason="stop",
        )
        return ChatCompletionResponse(
            id=job_id,
            model=request.model,
            choices=[choice],
            usage=UsageInfo(),
            conversation_id=conversation_id,
        )

    metadata = completion_event.metadata or {}
    content = metadata.get("content") or ""
    usage_payload = metadata.get("usage") or {}
    if not content:
        status_snapshot = await task_queue.get_status(job_id)
        result_snapshot = status_snapshot.get("result")
        if isinstance(result_snapshot, dict):
            content = result_snapshot.get("content") or content
            usage_payload = usage_payload or result_snapshot.get("usage") or {}

    try:
        usage = UsageInfo(**usage_payload)
    except (TypeError, ValidationError):
        usage = UsageInfo()

    message_metadata: Dict[str, Any] | None = None
    attachments = metadata.get("attachments")
    if attachments:
        message_metadata = {"attachments": attachments}

    logger.debug(
        "Job complete job_id=%s conversation_id=%s content_chars=%d usage=%s",
        job_id,
        conversation_id,
        len(content),
        usage.model_dump(),
    )

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessageResponse(content=content, metadata=message_metadata),
        finish_reason="stop",
    )
    return ChatCompletionResponse(
        id=job_id,
        model=request.model,
        choices=[choice],
        usage=usage,
        conversation_id=conversation_id,
    )
