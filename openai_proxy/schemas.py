from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class AttachmentInput(BaseModel):
    filename: str
    content_type: Optional[str] = None
    data: Optional[str] = Field(
        default=None,
        description="Base64-encoded file content for binary attachments.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Pre-extracted text content for the attachment.",
    )


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Any
    attachments: List[AttachmentInput] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalise_message(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        content = data.get("content")
        has_explicit_attachments = "attachments" in data and data["attachments"] is not None

        if isinstance(content, list):
            text_parts: List[str] = []
            extracted_attachments: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text_value = item.get("text")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                    elif item_type and item_type.startswith("input_"):
                        attachment: Dict[str, Any] = {}
                        filename = item.get("filename")
                        if isinstance(filename, str):
                            attachment["filename"] = filename
                        media_type = item.get("media_type") or item.get("content_type")
                        if isinstance(media_type, str):
                            attachment["content_type"] = media_type
                        data_field = item.get("data") or item.get("base64_data")
                        if isinstance(data_field, str):
                            attachment["data"] = data_field
                        text_value = item.get("text")
                        if isinstance(text_value, str):
                            attachment["text"] = text_value
                        if attachment:
                            extracted_attachments.append(attachment)
                    else:
                        text_value = item.get("text")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                data["content"] = "\n".join(text_parts)
            elif not text_parts and not extracted_attachments:
                # ensure content is at least string
                data["content"] = ""
            if extracted_attachments and not has_explicit_attachments:
                data["attachments"] = extracted_attachments
        return data

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
    metadata: Optional[Dict[str, Any]] = None


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
