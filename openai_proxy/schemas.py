from __future__ import annotations

import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


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


def _guess_extension(content_type: Optional[str], url: Optional[str]) -> Optional[str]:
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip().lower())
        if ext:
            return ext
    if url:
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix
        if suffix:
            return suffix
    return None


def _ensure_filename(
    attachment: Dict[str, Any],
    *,
    default_prefix: str,
    content_type: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    if attachment.get("filename"):
        return
    ext = _guess_extension(content_type, url) or ".bin"
    if not ext.startswith("."):
        ext = f".{ext}"
    attachment["filename"] = f"{default_prefix}_{uuid.uuid4().hex}{ext}"


class AttachmentInput(BaseModel):
    filename: str
    content_type: Optional[str] = None
    url: Optional[str] = Field(
        default=None,
        description="Optional URL of the attachment when raw data is not provided.",
    )
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

            def _add_attachment(
                attachment: Dict[str, Any],
                *,
                default_prefix: str,
            ) -> None:
                if not attachment:
                    return
                if attachment.get("data") and not attachment.get("content_type") and default_prefix == "image":
                    attachment["content_type"] = "image/png"
                _ensure_filename(
                    attachment,
                    default_prefix=default_prefix,
                    content_type=attachment.get("content_type"),
                    url=attachment.get("url"),
                )
                extracted_attachments.append(attachment)

            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text_value = item.get("text")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                    elif item_type == "image_url":
                        url_value: Optional[str] = None
                        image_url = item.get("image_url")
                        if isinstance(image_url, dict):
                            url_value = image_url.get("url")
                        elif isinstance(image_url, str):
                            url_value = image_url
                        if not url_value:
                            url_value = item.get("url")
                        if isinstance(url_value, str) and url_value:
                            attachment: Dict[str, Any] = {"url": url_value}
                            mime, data_field = _parse_data_url(url_value)
                            if isinstance(mime, str):
                                attachment["content_type"] = mime
                            if isinstance(data_field, str):
                                attachment["data"] = data_field
                            _add_attachment(attachment, default_prefix="image")
                    elif item_type == "input_image":
                        attachment = {}
                        filename = item.get("filename")
                        if isinstance(filename, str):
                            attachment["filename"] = filename
                        media_type = item.get("media_type") or item.get("content_type") or item.get("mime_type")
                        if isinstance(media_type, str):
                            attachment["content_type"] = media_type
                        image_payload = item.get("image")
                        if isinstance(image_payload, dict):
                            data_field = (
                                image_payload.get("data")
                                or image_payload.get("base64_data")
                                or image_payload.get("image_base64")
                            )
                            if isinstance(data_field, str):
                                attachment["data"] = data_field
                            image_mime = image_payload.get("mime_type") or image_payload.get("content_type")
                            if isinstance(image_mime, str) and "content_type" not in attachment:
                                attachment["content_type"] = image_mime
                        data_field = item.get("data") or item.get("base64_data") or item.get("image_base64")
                        if isinstance(data_field, str):
                            attachment["data"] = data_field
                        image_url = item.get("image_url") or item.get("url")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url")
                        if isinstance(image_url, str) and image_url:
                            attachment["url"] = image_url
                            mime, url_data = _parse_data_url(image_url)
                            if isinstance(mime, str) and "content_type" not in attachment:
                                attachment["content_type"] = mime
                            if isinstance(url_data, str) and "data" not in attachment:
                                attachment["data"] = url_data
                        _add_attachment(attachment, default_prefix="image")
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
                        url_value = item.get("url")
                        if isinstance(url_value, str):
                            attachment["url"] = url_value
                        _add_attachment(attachment, default_prefix="attachment")
                    else:
                        text_value = item.get("text")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                data["content"] = "\n".join(text_parts)
            else:
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
