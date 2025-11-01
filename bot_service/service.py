from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.state.state import ConfigSchema

from .config import settings
from .schemas import MessagePayload


def _normalise_content(message: BaseMessage) -> Dict[str, Any]:
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item)
            else:
                parts.append({"type": "text", "text": str(item)})
        return {"type": "segments", "parts": parts}
    return {"type": "text", "text": str(content)}


def _extract_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                texts.append(item.get("text") or "")
            else:
                texts.append(str(item))
        return "\n".join(filter(None, texts))
    return str(content)


def build_human_message(payload: MessagePayload) -> HumanMessage:
    if payload.type == "reset":
        reset_text = payload.text or "RESET"
        content = [{"type": "reset", "text": reset_text}]
    else:
        content: List[Dict[str, str]] = []
        if payload.text:
            content.append({"type": "text", "text": payload.text})
        attachment_segments = []
        if isinstance(payload.metadata, dict):
            attachment_segments = payload.metadata.get("attachment_text_segments") or []
        for segment in attachment_segments:
            if isinstance(segment, str) and segment.strip():
                content.append({"type": "text", "text": segment})
        if not content:
            content.append({"type": "text", "text": ""})
    return HumanMessage(content=content)


def build_agent_config(conversation_id: str, user_id: str, user_role: Optional[str]) -> RunnableConfig:
    role = user_role or settings.default_user_role
    configurable: ConfigSchema = {
        "user_id": user_id,
        "user_role": role,
        "thread_id": conversation_id,
    }
    return {"configurable": configurable}


async def invoke_agent(
    agent: Any,
    payload: MessagePayload,
    conversation_id: str,
    user_id: str,
    user_role: Optional[str],
) -> Dict[str, Any]:
    human = build_human_message(payload)
    config = build_agent_config(conversation_id, user_id, user_role)

    def _invoke() -> Dict[str, Any]:
        response = agent.invoke({"messages": [human]}, config=config)
        if isinstance(response, dict):
            return response
        return {"messages": response}

    result = await run_in_threadpool(_invoke)
    messages = result.get("messages") or []

    ai_message: Optional[AIMessage] = None
    if isinstance(messages, Iterable):
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage):
                ai_message = msg
                break

    if ai_message is None:
        ai_message = AIMessage(content="")  # fallback empty response

    return {
        "human": human,
        "ai": ai_message,
        "raw_result": result,
    }


def serialise_message(message: BaseMessage) -> Dict[str, Any]:
    return {
        "raw_text": _extract_text(message),
        "content": _normalise_content(message),
    }
