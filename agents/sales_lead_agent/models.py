from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from bot_service.db import Base


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_by_user_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    dedup_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    company_name: Mapped[str] = mapped_column(String(255), index=True)
    inn: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    ogrn: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    region: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    website: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    event_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    event_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    event_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    event_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    object_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    lead_priority: Mapped[str] = mapped_column(String(32), default="insufficient_data", index=True)
    lead_score: Mapped[float] = mapped_column(Float, default=0.0)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    missing_data: Mapped[list[str]] = mapped_column(JSON, default=list)
    manual_review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    workflow_status: Mapped[str] = mapped_column(String(32), default="new", index=True)
    feedback_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    digest_included: Mapped[bool] = mapped_column(Boolean, default=False)
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    sources: Mapped[list["LeadSource"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadSource.retrieved_at",
    )
    documents: Mapped[list["LeadDocument"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadDocument.created_at",
    )
    facts: Mapped[list["LeadFact"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadFact.created_at",
    )
    contacts: Mapped[list["LeadContact"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadContact.created_at",
    )
    enrichments: Mapped[list["LeadEnrichment"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadEnrichment.created_at",
    )
    feedback_entries: Mapped[list["LeadFeedback"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadFeedback.created_at",
    )
    index_chunks: Mapped[list["LeadIndexChunk"]] = relationship(
        back_populates="lead",
        cascade="all, delete-orphan",
        order_by="LeadIndexChunk.chunk_index",
    )


class LeadSource(Base):
    __tablename__ = "lead_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    lead: Mapped[Lead] = relationship(back_populates="sources")


class LeadDocument(Base):
    __tablename__ = "lead_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    document_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    stored_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parse_status: Mapped[str] = mapped_column(String(32), default="pending")
    index_status: Mapped[str] = mapped_column(String(32), default="pending")
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extracted_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="documents")
    index_chunks: Mapped[list["LeadIndexChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="LeadIndexChunk.chunk_index",
    )


class LeadFact(Base):
    __tablename__ = "lead_facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str] = mapped_column(String(128), index=True)
    value_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="facts")


class LeadContact(Base):
    __tablename__ = "lead_contacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    contact_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    contact_role: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    contact_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    contact_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    contact_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="contacts")


class LeadEnrichment(Base):
    __tablename__ = "lead_enrichments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    payload_json: Mapped[Dict[str, Any]] = mapped_column("payload", JSON, default=dict)
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrieved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="enrichments")


class LeadFeedback(Base):
    __tablename__ = "lead_feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lead: Mapped[Lead] = relationship(back_populates="feedback_entries")


class LeadIndexChunk(Base):
    __tablename__ = "lead_index_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    lead_id: Mapped[str] = mapped_column(ForeignKey("leads.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[Optional[str]] = mapped_column(ForeignKey("lead_documents.id", ondelete="CASCADE"), nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, index=True)
    text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    position_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    position_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    lead: Mapped[Lead] = relationship(back_populates="index_chunks")
    document: Mapped[Optional[LeadDocument]] = relationship(back_populates="index_chunks")


class LeadExport(Base):
    __tablename__ = "lead_exports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    requested_by: Mapped[str] = mapped_column(String(128), index=True)
    format: Mapped[str] = mapped_column(String(16))
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[Dict[str, Any]] = mapped_column("filters", JSON, default=dict)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
