from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LeadSourceView(BaseModel):
    id: str
    source_type: str
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    source_reference: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    confidence: Optional[float] = None
    is_primary: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LeadDocumentView(BaseModel):
    id: str
    document_url: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    stored_path: Optional[str] = None
    parse_status: str
    index_status: str
    source_reference: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    confidence: Optional[float] = None
    extracted_excerpt: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LeadFactView(BaseModel):
    id: str
    field_name: str
    value: Dict[str, Any] = Field(default_factory=dict)
    source_reference: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    confidence: Optional[float] = None


class LeadContactView(BaseModel):
    id: str
    contact_name: Optional[str] = None
    contact_role: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_source: Optional[str] = None
    contact_confidence: Optional[float] = None
    source_reference: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LeadEnrichmentView(BaseModel):
    id: str
    provider: str
    status: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    source_reference: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    confidence: Optional[float] = None


class LeadIndexHitView(BaseModel):
    lead_id: str
    company_name: str
    lead_priority: Optional[str] = None
    score: float
    snippet: str
    source_reference: Optional[str] = None
    source_url: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    chunk_index: int
    page_number: Optional[int] = None
    position_start: Optional[int] = None
    position_end: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LeadView(BaseModel):
    id: str
    created_by_user_id: Optional[str] = None
    dedup_key: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    company_name: str
    inn: Optional[str] = None
    ogrn: Optional[str] = None
    region: Optional[str] = None
    website: Optional[str] = None
    event_type: Optional[str] = None
    event_title: Optional[str] = None
    event_date: Optional[datetime] = None
    event_summary: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    object_type: Optional[str] = None
    lead_priority: str
    lead_score: float
    rationale: Optional[str] = None
    missing_data: List[str] = Field(default_factory=list)
    manual_review_required: bool = False
    workflow_status: str
    feedback_status: str
    tags: List[str] = Field(default_factory=list)
    digest_included: bool = False
    source_reference: Optional[str] = None
    retrieved_at: Optional[datetime] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    sources: List[LeadSourceView] = Field(default_factory=list)
    documents: List[LeadDocumentView] = Field(default_factory=list)
    facts: List[LeadFactView] = Field(default_factory=list)
    contacts: List[LeadContactView] = Field(default_factory=list)
    enrichments: List[LeadEnrichmentView] = Field(default_factory=list)


class LeadFeedbackCreate(BaseModel):
    status: str
    comment: Optional[str] = None


class LeadFeedbackView(BaseModel):
    id: str
    lead_id: str
    user_id: str
    status: str
    comment: Optional[str] = None
    created_at: datetime


class LeadExportRequest(BaseModel):
    format: str = "xlsx"
    period_from: Optional[datetime] = None
    period_to: Optional[datetime] = None
    region: Optional[str] = None
    priority: Optional[str] = None
    source_type: Optional[str] = None
    only_with_inn: bool = False
    only_with_contacts: bool = False
    limit: int = Field(default=100, ge=1, le=1000)


class LeadExportView(BaseModel):
    id: str
    requested_by: str
    format: str
    filename: str
    path: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    row_count: int
    created_at: datetime
