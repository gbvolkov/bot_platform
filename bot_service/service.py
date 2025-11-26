from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional
import logging

from fastapi.concurrency import run_in_threadpool
import uuid
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from agents.state.state import ConfigSchema

from .config import settings
from .schemas import MessagePayload


ATTACHMENT_TYPES = {"file", "image", "audio", "video", "attachment"}


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


def _normalise_attachment_piece(piece: Dict[str, Any]) -> Dict[str, Any]:
    attachment: Dict[str, Any] = {"type": piece.get("type")}

    def _copy(field: str, target_field: Optional[str] = None) -> None:
        value = piece.get(field)
        if value is None:
            return
        attachment[target_field or field] = value

    _copy("filename")
    _copy("name")
    _copy("title")
    _copy("caption")
    _copy("format")
    _copy("graphic_type")

    content_type = piece.get("mime_type") or piece.get("content_type")
    if content_type:
        attachment["content_type"] = content_type

    data = piece.get("data") or piece.get("base64_data")
    if isinstance(data, str) and data:
        attachment["data"] = data

    url = piece.get("url")
    if not url:
        image_url = piece.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
    if url:
        attachment["url"] = url

    text_value = piece.get("text")
    if isinstance(text_value, str) and text_value:
        attachment["text"] = text_value

    metadata_value = piece.get("metadata")
    if isinstance(metadata_value, dict) and metadata_value:
        attachment["metadata"] = metadata_value

    return attachment


def _extract_attachments(message: BaseMessage) -> List[Dict[str, Any]]:
    content = getattr(message, "content", None)
    attachments: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        return attachments
    for piece in content:
        if not isinstance(piece, dict):
            continue
        piece_type = piece.get("type")
        if not isinstance(piece_type, str):
            continue
        if piece_type.lower() not in ATTACHMENT_TYPES:
            continue
        attachments.append(_normalise_attachment_piece(piece))
    return attachments


def build_human_message(payload: MessagePayload, raw_text_override: Optional[str] = None) -> HumanMessage:
    if payload.type == "reset":
        reset_text = payload.text or "RESET"
        content = [{"type": "reset", "text": reset_text}]
    else:
        content: List[Dict[str, str]] = []
        user_text = raw_text_override if raw_text_override is not None else payload.text
        if user_text:
            content.append({"type": "text", "text": user_text})
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
    pending_interrupt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_user_text = None
    if isinstance(payload.metadata, dict):
        raw_user_text = payload.metadata.get("raw_user_text")
    human = build_human_message(payload, raw_text_override=raw_user_text if pending_interrupt else None)
    config = build_agent_config(conversation_id, user_id, user_role)

    def _invoke() -> Dict[str, Any]:
        if pending_interrupt:
            logging.info(
                "invoke_agent resume conversation_id=%s interrupt_id=%s raw_user_text_chars=%d",
                conversation_id,
                pending_interrupt.get("interrupt_id") if isinstance(pending_interrupt, dict) else None,
                len(raw_user_text or payload.text or ""),
            )
            response = agent.invoke(Command(resume=raw_user_text or payload.text or ""), config=config)
        else:
            response = agent.invoke({"messages": [human]}, config=config)
        if isinstance(response, dict):
            return response
        return {"messages": response}

    result = await run_in_threadpool(_invoke)
    messages = result.get("messages") or []

    ai_message: Optional[AIMessage] = None
    if "__interrupt__" in result:
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            latest = interrupts[-1]
            interrupt_payload = getattr(latest, "value", latest)
        else:
            interrupt_payload = {}
        if isinstance(interrupt_payload, dict) and "interrupt_id" not in interrupt_payload:
            interrupt_payload = {**interrupt_payload, "interrupt_id": f"int-{uuid.uuid4().hex}"}
        question = ""
        if isinstance(interrupt_payload, dict):
            question = interrupt_payload.get("question") or interrupt_payload.get("content") or ""
        logging.info("invoke_agent interrupt detected conversation_id=%s interrupt_id=%s", conversation_id, interrupt_payload.get("interrupt_id") if isinstance(interrupt_payload, dict) else None)
        ai_message = AIMessage(content=question)
        return {
            "human": human,
            "ai": ai_message,
            "raw_result": result,
            "agent_status": "interrupted",
            "interrupt_payload": interrupt_payload,
        }

    if isinstance(messages, Iterable):
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage):
                ai_message = msg
                break

    if ai_message is None:
        ai_message = AIMessage(content="")  # fallback empty response

    logging.info(
        "invoke_agent completed conversation_id=%s agent_status=%s",
        conversation_id,
        "completed",
    )
    return {
        "human": human,
        "ai": ai_message,
        "raw_result": result,
        "agent_status": "completed",
        "interrupt_payload": None,
    }


def serialise_message(message: BaseMessage) -> Dict[str, Any]:
    return {
        "raw_text": _extract_text(message),
        "content": _normalise_content(message),
        "attachments": _extract_attachments(message),
    }
