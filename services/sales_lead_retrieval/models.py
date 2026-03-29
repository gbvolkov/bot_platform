from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class SalesLeadRetrievalJob(Base):
    __tablename__ = "sales_lead_retrieval_jobs"
    __table_args__ = (
        UniqueConstraint("conversation_id", "request_hash", name="uq_sales_lead_retrieval_jobs_conv_hash"),
        UniqueConstraint("active_conversation_id", name="uq_sales_lead_retrieval_jobs_active_conversation"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    active_conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    request_hash: Mapped[str] = mapped_column(String(64), index=True)
    request_payload_json: Mapped[dict[str, Any]] = mapped_column("request_payload", JSON, default=dict)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    index_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_json: Mapped[dict[str, Any]] = mapped_column("progress", JSON, default=dict)
    items_snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    documents_snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    ready_items_count: Mapped[int] = mapped_column(Integer, default=0)
    ready_documents_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_segments_count: Mapped[int] = mapped_column(Integer, default=0)
    completion_announced: Mapped[bool] = mapped_column(Boolean, default=False)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    snapshot_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SalesLeadRetrievalEvent(Base):
    __tablename__ = "sales_lead_retrieval_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    retrieval_id: Mapped[str] = mapped_column(String(36), index=True)
    stage: Mapped[str] = mapped_column(String(64), index=True)
    level: Mapped[str] = mapped_column(String(16), default="info")
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict[str, Any]] = mapped_column("payload", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
