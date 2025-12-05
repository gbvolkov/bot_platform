from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    IMAGES = "images"
    PDFS = "pdfs"
    TEXT_FILES = "text_files"
    MARKDOWN = "mds"
    DOCX_DOCUMENTS = "docx_documents"
    CSVS = "csvs"
    EXCELS = "excels"
    SOUNDS = "sounds"
    VIDEOS = "videos"
    JSONS = "jsons"


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    provider: Optional[str] = None
    supported_content_types: List[ContentType] = Field(default_factory=list)


class ConversationCreate(BaseModel):
    agent_id: str = Field(description="Identifier of the agent to route messages to.")
    title: Optional[str] = None
    user_role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationView(BaseModel):
    id: str
    agent_id: str
    user_id: str
    user_role: str
    title: Optional[str] = None
    status: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime


class AttachmentPayload(BaseModel):
    filename: str
    content_type: Optional[str] = None
    data: Optional[str] = Field(
        default=None,
        description="Base64-encoded file content for binary attachments.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Pre-extracted plain text for the attachment (optional).",
    )


class MessagePayload(BaseModel):
    type: Literal["text", "reset"] = "text"
    text: Optional[str] = Field(
        default=None, description="Text content when the type is 'text'."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[AttachmentPayload] = Field(default_factory=list)


class MessageCreate(BaseModel):
    payload: MessagePayload


class MessageView(BaseModel):
    id: str
    role: str
    content: Dict[str, Any]
    raw_text: str
    metadata: Dict[str, Any]
    created_at: datetime


class ConversationDetail(ConversationView):
    messages: List[MessageView] = Field(default_factory=list)


class SendMessageResponse(BaseModel):
    conversation: ConversationView
    user_message: MessageView
    agent_message: MessageView
