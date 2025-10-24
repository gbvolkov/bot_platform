from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status

from .client import BotServiceClient
from .config import settings
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
)
from .utils import build_prompt


bot_client = BotServiceClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bot_client.startup()
    try:
        yield
    finally:
        await bot_client.shutdown()


def get_client() -> BotServiceClient:
    return bot_client


app = FastAPI(
    title="OpenAI Compatible Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz", tags=["system"])
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    client: Annotated[BotServiceClient, Depends(get_client)],
) -> ChatCompletionResponse:
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming responses are not supported yet.",
        )

    try:
        await client.ensure_agent(request.model)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    user_id = request.user or settings.default_user_id
    user_role = settings.default_user_role

    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = await client.create_conversation(
            agent_id=request.model,
            user_id=user_id,
            user_role=user_role,
        )

    try:
        prompt = build_prompt(request.messages)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    response_data = await client.send_message(
        conversation_id=conversation_id,
        user_id=user_id,
        user_role=user_role,
        text=prompt,
    )
    agent_message = response_data["agent_message"]["raw_text"]

    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessageResponse(content=agent_message),
        finish_reason="stop",
    )
    return ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        conversation_id=conversation_id,
    )

