from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


JobStage = Literal[
    "queued",
    "running",
    "streaming",
    "completed",
    "failed",
    "interrupted",
]
EventKind = Literal[
    "status",
    "chunk",
    "completed",
    "failed",
    "heartbeat",
    "interrupt",
]


class EnqueuePayload(BaseModel):
    """Payload stored in Redis for worker consumption."""

    job_id: str = Field(description="Unique identifier of the queued job.")
    model: str = Field(description="Agent/model identifier to execute.")
    conversation_id: str = Field(description="Conversation identifier within bot_service.")
    user_id: str = Field(description="User identifier forwarded to bot_service.")
    user_role: Optional[str] = Field(
        default=None,
        description="Optional user role forwarded to bot_service.",
    )
    text: str = Field(default="", description="Rendered prompt text to deliver to the agent.")
    raw_user_text: Optional[str] = Field(
        default=None,
        description="Last user utterance before rendering prompt (used to resume interrupts).",
    )
    attachments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Attachment metadata forwarded to bot_service.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata forwarded to bot_service.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to request streaming responses from bot_service.",
    )


class QueueEvent(BaseModel):
    """Event published to Redis Pub/Sub for progress and streaming."""

    job_id: str = Field(description="Associated job identifier.")
    type: EventKind = Field(description="Event flavor.")
    status: Optional[JobStage] = Field(
        default=None,
        description="High-level job stage for status events.",
    )
    content: Optional[str] = Field(
        default=None,
        description="Text chunk emitted for streaming responses.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata included with the event.",
    )
    usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Token usage stats, when available.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error description for failed jobs.",
    )
