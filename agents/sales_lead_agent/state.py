from __future__ import annotations

from typing import Annotated, Any, Dict, List
from typing_extensions import NotRequired, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class SalesLeadState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    attachments: NotRequired[List[Dict[str, Any]]]
    request_understanding: NotRequired[Dict[str, Any]]
    task_plan: NotRequired[Dict[str, Any]]
    existing_leads: NotRequired[List[Dict[str, Any]]]
    source_hits: NotRequired[List[Dict[str, Any]]]
    documents: NotRequired[List[Dict[str, Any]]]
    index_hits: NotRequired[List[Dict[str, Any]]]
    extracted_leads: NotRequired[List[Dict[str, Any]]]
    persisted_leads: NotRequired[List[Dict[str, Any]]]
    export_record: NotRequired[Dict[str, Any]]
    summary_export_record: NotRequired[Dict[str, Any]]
    feedback_result: NotRequired[Dict[str, Any]]
    response_text: NotRequired[str]
    errors: NotRequired[List[str]]


class SalesLeadContext(TypedDict, total=False):
    locale: str
