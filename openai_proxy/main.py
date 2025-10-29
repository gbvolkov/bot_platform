from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status

from .client import BotServiceClient
from .config import settings
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    ModelCard,
    ModelList,
)
from .utils import build_prompt


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


app = FastAPI(
    title="OpenAI Compatible Proxy",
    version="0.1.0",
    lifespan=lifespan,
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
