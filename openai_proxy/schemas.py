from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Any

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                return text
        return str(value)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    user: Optional[str] = None
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation identifier to continue a previous session.",
    )
    stream: Optional[bool] = Field(default=False)


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length"] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    conversation_id: str


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = Field(default="bot-service")
    name: Optional[str] = None
    description: Optional[str] = None
    provider: Optional[str] = None
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]
