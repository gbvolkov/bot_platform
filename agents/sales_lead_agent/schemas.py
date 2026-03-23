from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AnswerType = Literal["lead_list", "lead_card", "company_check", "comparison"]
TaskKind = Literal[
    "procurement_search",
    "procurement_analysis",
    "company_check",
    "fact_lookup",
    "comparison",
]
LawType = Literal["44-FZ", "223-FZ"]
PriorityType = Literal["high", "medium", "low", "unknown"]
FactStatusType = Literal["document", "external_api", "open_source", "not_found"]
ValidationStatus = Literal["clean", "partial", "failed_verification"]
ValidationSeverity = Literal["warning", "error"]
PreparedOrigin = Literal["purchase", "open_source"]
PreparedFileType = Literal["pdf", "docx", "xlsx", "html", "other"]
ToolStatus = Literal["success", "partial", "failed"]
IndexStatus = Literal["ready", "failed"]
SourceKind = Literal["purchase", "open_source"]
EvidenceSource = Literal["purchase", "document", "open_source", "scoring", "fssp"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchFilters(StrictBaseModel):
    query_text: str | None = None
    law: LawType | None = None
    region: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    published_from: str | None = None
    published_to: str | None = None
    submission_deadline_from: str | None = None
    submission_deadline_to: str | None = None
    customer_name: str | None = None
    customer_inn: str | None = None
    supplier_hint: str | None = None


class PreparedDocumentEntities(StrictBaseModel):
    inn: list[str] = Field(default_factory=list)
    company_names: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)


class PreparedDocument(StrictBaseModel):
    document_id: str
    origin: PreparedOrigin
    bundle_id: str
    registry_number: str | None = None
    source_url: str | None = None
    original_source_url: str | None = None
    original_file_name: str | None = None
    original_content_type: str | None = None
    derived_artifact_path: str | None = None
    file_path: str
    file_name: str
    file_type: PreparedFileType
    parse_status: ToolStatus
    index_status: IndexStatus
    text_excerpt: str
    entities: PreparedDocumentEntities = Field(default_factory=PreparedDocumentEntities)
    chunks_count: int = 0
    error: str | None = None


class PurchaseSearchItem(StrictBaseModel):
    bundle_id: str
    registry_number: str
    law: LawType | None = None
    purchase_title: str
    customer_name: str
    price_text: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    submission_deadline: str | None = None
    detail_url: str
    common_info_url: str | None = None
    documents_url: str | None = None
    document_urls: list[str] = Field(default_factory=list)
    downloaded_files: list[str] = Field(default_factory=list)
    prepared_document_ids: list[str] = Field(default_factory=list)
    documents_json: str | None = None
    common_info_json: str | None = None
    lots_json: str | None = None
    crawl_status: ToolStatus
    crawl_error: str | None = None
    crawl_ts_utc: str


class PurchaseSearchRequest(StrictBaseModel):
    run_id: str | None = None
    search_url: str | None = None
    query_text: str | None = None
    law: LawType | None = None
    region: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    published_from: str | None = None
    published_to: str | None = None
    submission_deadline_from: str | None = None
    submission_deadline_to: str | None = None
    customer_name: str | None = None
    customer_inn: str | None = None
    supplier_hint: str | None = None
    max_pages: int | None = None
    headless: bool | None = None


class PurchaseSearchResponse(StrictBaseModel):
    source: Literal["purchase_adapter"] = "purchase_adapter"
    run_id: str
    index_id: str
    status: ToolStatus
    errors: list[str] = Field(default_factory=list)
    items: list[PurchaseSearchItem] = Field(default_factory=list)
    prepared_documents: list[PreparedDocument] = Field(default_factory=list)


class OpenSourceFetchRequest(StrictBaseModel):
    run_id: str | None = None
    url: str
    depth: int | None = None
    follow_download_links: bool | None = None
    max_concurrency: int | None = None


class OpenSourcePage(StrictBaseModel):
    bundle_id: str
    url: str
    title: str | None = None
    text: str
    attachments: list[str] = Field(default_factory=list)
    prepared_document_ids: list[str] = Field(default_factory=list)


class OpenSourceFetchResponse(StrictBaseModel):
    source: Literal["rag_lib"] = "rag_lib"
    run_id: str
    index_id: str
    status: ToolStatus
    errors: list[str] = Field(default_factory=list)
    pages: list[OpenSourcePage] = Field(default_factory=list)
    prepared_documents: list[PreparedDocument] = Field(default_factory=list)


class DocSearchRequest(StrictBaseModel):
    index_id: str | None = None
    query: str
    top_k: int | None = None
    source_kind: SourceKind | None = None
    bundle_id: str | None = None


class DocSearchMatch(StrictBaseModel):
    document_id: str
    bundle_id: str
    file_path: str
    page: int | None = None
    locator: str | None = None
    snippet: str
    score: float
    source_kind: SourceKind
    source_url: str | None = None


class DocSearchResponse(StrictBaseModel):
    index_id: str
    matches: list[DocSearchMatch] = Field(default_factory=list)


class TopFactor(StrictBaseModel):
    name: str
    value: float | None = None
    nwoe: float | None = None


class ScorePayload(StrictBaseModel):
    risk_value: float | None = None
    risk_zone: str | None = None
    score_value: float | None = None
    score_zone: str | None = None
    reliability_value: float | None = None
    reliability_zone: str | None = None
    top_factors: list[TopFactor] = Field(default_factory=list)


class Fincoef(StrictBaseModel):
    name: str
    value: float | None = None
    norm: float | None = None
    comparison: str | None = None


class CounterpartyScoringRequest(StrictBaseModel):
    inn: str
    model: str | None = None
    include_fincoefs: bool | None = None


class CounterpartyScoringResponse(StrictBaseModel):
    source: Literal["damia_scoring"] = "damia_scoring"
    status: Literal["success", "failed"]
    error: str | None = None
    inn: str
    score: ScorePayload = Field(default_factory=ScorePayload)
    fincoefs: list[Fincoef] = Field(default_factory=list)


class FSSPGroupedRecord(StrictBaseModel):
    year: int
    status: str
    subject: str
    amount: float | None = None
    count: int
    proceeding_ids: list[str] = Field(default_factory=list)


class CounterpartyFSSPRequest(StrictBaseModel):
    inn: str
    from_date: str | None = None
    to_date: str | None = None
    format: Literal[1, 2] | None = None


class CounterpartyFSSPResponse(StrictBaseModel):
    source: Literal["damia_fssp"] = "damia_fssp"
    status: Literal["success", "failed"]
    error: str | None = None
    inn: str
    grouped: list[FSSPGroupedRecord] = Field(default_factory=list)
    raw_format: Literal[1, 2] = 1


class FactStatus(StrictBaseModel):
    fact_key: str
    status: FactStatusType


class TaskUnderstandingResult(StrictBaseModel):
    answer_type: AnswerType = Field(
        description=(
            "Final answer shape only. Allowed values: lead_list, lead_card, company_check, comparison. "
            "Never use task_kind values such as procurement_search, procurement_analysis, or fact_lookup here."
        )
    )
    task_kind: TaskKind = Field(
        description=(
            "Execution intent only. Allowed values: procurement_search, procurement_analysis, "
            "company_check, fact_lookup, comparison."
        )
    )
    search_url: str | None = None
    search_filters: SearchFilters = Field(default_factory=SearchFilters)
    requested_company_inns: list[str] = Field(default_factory=list)
    comparison_targets: list[str] = Field(default_factory=list)
    document_questions: list[str] = Field(default_factory=list)
    needs_purchase_search: bool
    needs_open_source: bool
    needs_doc_search: bool
    needs_enrichment: bool
    missing_data: list[str] = Field(default_factory=list)


class EnrichmentAssessmentResult(StrictBaseModel):
    priority: PriorityType
    reasons: list[str] = Field(default_factory=list)
    risk_summary: str
    manual_review_required: bool
    significant_signals: list[str] = Field(default_factory=list)
    fact_statuses: list[FactStatus] = Field(default_factory=list)
    recommended_next_step: str | None = None


class RiskVerificationResult(EnrichmentAssessmentResult):
    pass


class ProcurementRelevanceDecision(StrictBaseModel):
    bundle_id: str
    registry_number: str
    is_relevant: bool
    reason: str


class ProcurementRelevanceBatch(StrictBaseModel):
    decisions: list[ProcurementRelevanceDecision] = Field(default_factory=list)


class TurnValidationIssue(StrictBaseModel):
    stage: str
    code: str
    message: str
    severity: ValidationSeverity = "error"
    recoverable: bool = True
    affects_final_answer: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)


class TurnValidationResult(StrictBaseModel):
    status: ValidationStatus = "clean"
    issues: list[TurnValidationIssue] = Field(default_factory=list)
    manual_review_required: bool = False


class EvidenceItem(StrictBaseModel):
    source: EvidenceSource
    source_url: str | None = None
    file_path: str | None = None
    page: int | None = None
    locator: str | None = None
    snippet: str


class LeadAnswerItem(StrictBaseModel):
    company_name: str | None = None
    inn: str | None = None
    event_title: str | None = None
    source_url: str | None = None
    region: str | None = None
    amount_text: str | None = None
    contacts: list[str] = Field(default_factory=list)
    scoring: dict | None = None
    fssp: dict | None = None
    priority: PriorityType
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    fact_statuses: list[FactStatus] = Field(default_factory=list)


class LeadAnswerContract(StrictBaseModel):
    answer_type: AnswerType
    summary: str
    items: list[LeadAnswerItem] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    recommended_next_step: str | None = None
