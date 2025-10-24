from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str
    provider: Optional[str] = None


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


class MessagePayload(BaseModel):
    type: Literal["text", "reset"] = "text"
    text: Optional[str] = Field(
        default=None, description="Text content when the type is 'text'."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


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

