from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RetrievalStatus = Literal["queued", "in_progress", "completed", "failed"]


class RetrievalProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_queries: int = 0
    completed_queries: int = 0
    total_purchases: int = 0
    processed_purchases: int = 0
    total_files: int = 0
    processed_files: int = 0
    prepared_documents: int = 0
    indexed_segments: int = 0


class RetrievalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_id: str
    conversation_id: str
    request_hash: str
    run_id: str
    index_id: str
    status: RetrievalStatus
    stage: str
    message: str
    completion_announced: bool
    snapshot_updated_at: str | None
    request_payload: dict[str, Any] = Field(default_factory=dict)
    progress: RetrievalProgress = Field(default_factory=RetrievalProgress)
    items: list[dict[str, Any]] = Field(default_factory=list)
    prepared_documents: list[dict[str, Any]] = Field(default_factory=list)
    error_text: str | None = None

    def active_retrieval_context(self) -> dict[str, Any]:
        return {
            "retrieval_id": self.retrieval_id,
            "request_hash": self.request_hash,
            "run_id": self.run_id,
            "index_id": self.index_id,
            "retrieval_status": self.status,
            "retrieval_stage": self.stage,
            "message": self.message,
            "completion_announced": self.completion_announced,
            "snapshot_updated_at": self.snapshot_updated_at,
            "progress": self.progress.model_dump(),
        }


class RetrievalConflictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal["RETRIEVAL_ALREADY_IN_PROGRESS"] = "RETRIEVAL_ALREADY_IN_PROGRESS"
    message: str
    active_snapshot: RetrievalSnapshot


class RetrievalUserInputErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    suggestion: str | None = None
    input_field: str | None = None


class SubmitPurchaseRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    requested_run_id: str | None = None
    search_url: str | None = None
    query_texts: list[str] | None = None
    max_pages: int | None = None
    agent_id: str = "sales_lead_agent"
