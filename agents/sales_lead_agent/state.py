from __future__ import annotations

from typing import Any, Literal

from langchain.agents import AgentState
from typing_extensions import NotRequired, TypedDict


RetrievalStatus = Literal["queued", "in_progress", "completed", "failed"]


class RetrievalProgressState(TypedDict):
    total_queries: int
    completed_queries: int
    total_purchases: int
    processed_purchases: int
    total_files: int
    processed_files: int
    prepared_documents: int
    indexed_segments: int


class SalesLeadAgentState(AgentState):
    conversation_id: NotRequired[str | None]
    default_index_id: NotRequired[str | None]
    active_retrieval_id: NotRequired[str | None]
    active_retrieval_request_hash: NotRequired[str | None]
    active_retrieval_run_id: NotRequired[str | None]
    active_retrieval_index_id: NotRequired[str | None]
    active_retrieval_status: NotRequired[RetrievalStatus | None]
    active_retrieval_stage: NotRequired[str | None]
    active_retrieval_message: NotRequired[str | None]
    active_retrieval_progress: NotRequired[RetrievalProgressState | None]
    active_run_id: NotRequired[str | None]
    index_id: NotRequired[str | None]
    turn_tool_usage: NotRequired[list[dict[str, Any]]]
    purchase_search_result: NotRequired[dict[str, Any] | None]
    last_purchase_lookup_result: NotRequired[dict[str, Any] | None]
    last_doc_search_result: NotRequired[dict[str, Any] | None]
    prepared_documents: NotRequired[list[dict[str, Any]]]
    pending_crawl_request: NotRequired[dict[str, Any] | None]
    pending_crawl_reason: NotRequired[str | None]
    pending_crawl_request_hash: NotRequired[str | None]
    pending_crawl_query_preview: NotRequired[list[str] | None]
    normalized_final_answer: NotRequired[dict[str, Any]]
    turn_validation: NotRequired[dict[str, Any]]
