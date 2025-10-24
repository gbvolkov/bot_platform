from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent_registry import agent_registry
from ..models import Conversation, Message
from ..schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationView,
    MessageCreate,
    MessageView,
    SendMessageResponse,
)
from ..service import invoke_agent, serialise_message
from .deps import DbSession, UserContextDep

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _conversation_to_view(entity: Conversation) -> ConversationView:
    return ConversationView(
        id=entity.id,
        agent_id=entity.agent_id,
        user_id=entity.user_id,
        user_role=entity.user_role,
        title=entity.title,
        status=entity.status,
        metadata=entity.metadata_json or {},
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        last_message_at=entity.last_message_at,
    )


def _message_to_view(entity: Message) -> MessageView:
    return MessageView(
        id=entity.id,
        role=entity.role,
        content=entity.content,
        raw_text=entity.raw_text,
        metadata=entity.metadata_json or {},
        created_at=entity.created_at,
    )


@router.post("/", response_model=ConversationView)
async def create_conversation(
    payload: ConversationCreate,
    session: DbSession,
    user: UserContextDep,
):
    try:
        ready = await agent_registry.ensure_agent_ready(payload.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    conversation = Conversation(
        agent_id=payload.agent_id,
        user_id=user.user_id,
        user_role=payload.user_role or user.user_role,
        title=payload.title,
        metadata_json=payload.metadata or {},
        status="active" if ready else "pending",
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    view = _conversation_to_view(conversation)
    if ready:
        return view
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=jsonable_encoder(view),
    )


@router.get("/", response_model=list[ConversationView])
async def list_conversations(session: DbSession, user: UserContextDep) -> list[ConversationView]:
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user.user_id)
        .order_by(Conversation.last_message_at.desc())
    )
    result = await session.scalars(stmt)
    conversations = result.all()
    updated = False
    for conv in conversations:
        if conv.status != "active" and agent_registry.is_ready(conv.agent_id):
            conv.status = "active"
            updated = True
    if updated:
        await session.commit()
        for conv in conversations:
            await session.refresh(conv)
    return [_conversation_to_view(conv) for conv in conversations]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str,
    session: DbSession,
    user: UserContextDep,
) -> ConversationDetail:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    ready = await agent_registry.ensure_agent_ready(conversation.agent_id)
    if ready and conversation.status != "active":
        conversation.status = "active"
        await session.commit()
        await session.refresh(conversation)
    await session.refresh(conversation, attribute_names=["messages"])
    messages = [_message_to_view(msg) for msg in conversation.messages]
    return ConversationDetail(**_conversation_to_view(conversation).model_dump(), messages=messages)


@router.post(
    "/{conversation_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def post_message(
    conversation_id: str,
    payload: MessageCreate,
    session: DbSession,
    user: UserContextDep,
) -> SendMessageResponse:
    conversation = await session.get(Conversation, conversation_id, with_for_update=True)
    if conversation is None or conversation.user_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    ready = await agent_registry.ensure_agent_ready(conversation.agent_id)
    if not ready:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Conversation is still initializing.")

    if conversation.status != "active":
        conversation.status = "active"

    try:
        agent = agent_registry.get_agent(conversation.agent_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        agent_result = await invoke_agent(
            agent=agent,
            payload=payload.payload,
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            user_role=conversation.user_role,
        )
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Agent invocation failed: {exc}",
        ) from exc

    human_payload = serialise_message(agent_result["human"])
    human_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=human_payload["content"],
        raw_text=human_payload["raw_text"],
        metadata_json=payload.payload.metadata,
    )
    session.add(human_message)

    ai_payload = serialise_message(agent_result["ai"])
    agent_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=ai_payload["content"],
        raw_text=ai_payload["raw_text"],
        metadata_json={"agent_id": conversation.agent_id},
    )
    session.add(agent_message)

    conversation.last_message_at = func.now()

    await session.commit()
    await session.refresh(conversation)
    await session.refresh(human_message)
    await session.refresh(agent_message)

    return SendMessageResponse(
        conversation=_conversation_to_view(conversation),
        user_message=_message_to_view(human_message),
        agent_message=_message_to_view(agent_message),
    )
