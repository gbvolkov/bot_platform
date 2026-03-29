from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolRetryMiddleware, dynamic_prompt
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

import config
from agents.llm_utils import get_llm
from agents.utils import ModelType, extract_text
from services.sales_lead_retrieval.client import get_retrieval_service_client

from .prompts import build_system_prompt
from .state import SalesLeadAgentState
from .tools import (
    ToolUserCorrectableError,
    _purchase_response_from_snapshot,
    _retrieval_state_from_snapshot,
    build_sales_lead_tools,
)

VALID_MODEL_SIZES = {"base", "mini", "nano"}
logger = logging.getLogger(__name__)

_VISIBLE_EVIDENCE_SOURCES = {"purchase", "document", "open_source", "scoring", "fssp", "dadata"}
_VISIBLE_FACT_STATUSES = {"document", "external_api", "open_source", "not_found"}


def _retrieval_state_note(state: Any) -> str | None:
    if not isinstance(state, dict):
        return None
    retrieval_status = str(state.get("active_retrieval_status") or "").strip()
    if not retrieval_status:
        return None
    progress = (
        state.get("active_retrieval_progress")
        if isinstance(state.get("active_retrieval_progress"), dict)
        else {}
    )
    purchase_result = (
        state.get("purchase_search_result")
        if isinstance(state.get("purchase_search_result"), dict)
        else {}
    )
    ready_items = purchase_result.get("items") if isinstance(purchase_result.get("items"), list) else []
    lines = [
        "Background procurement retrieval context:",
        f"- retrieval_status: {retrieval_status}",
        f"- retrieval_stage: {str(state.get('active_retrieval_stage') or '')}",
        f"- index_id: {str(state.get('active_retrieval_index_id') or '')}",
        f"- message: {str(state.get('active_retrieval_message') or '')}",
        (
            "- progress: "
            f"queries {int(progress.get('completed_queries', 0))}/{int(progress.get('total_queries', 0))}, "
            f"purchases {int(progress.get('processed_purchases', 0))}/{int(progress.get('total_purchases', 0))}, "
            f"files {int(progress.get('processed_files', 0))}/{int(progress.get('total_files', 0))}, "
            f"prepared_documents {int(progress.get('prepared_documents', 0))}, "
            f"indexed_segments {int(progress.get('indexed_segments', 0))}"
        ),
        f"- ready snapshot: procurement_items {len(ready_items)}",
    ]
    if ready_items:
        preview: list[str] = []
        for raw_item in ready_items[:3]:
            if not isinstance(raw_item, dict):
                continue
            registry_number = str(raw_item.get("registry_number") or "").strip()
            title = str(raw_item.get("purchase_title") or raw_item.get("title") or "").strip()
            if registry_number and title:
                preview.append(f"{registry_number}: {title}")
            elif title:
                preview.append(title)
            elif registry_number:
                preview.append(registry_number)
        if preview:
            lines.append("- current procurement examples:")
            lines.extend(f"  - {item}" for item in preview)
    if retrieval_status in {"queued", "in_progress"}:
        lines.extend(
            [
                "- The procurement retrieval is still running in the background.",
                "- Use the current index_id with doc_search_tool for questions about the retrieved materials.",
                "- If you already know an exact document_id, use read_cached_document_tool to read that cached file content without fetching anything new.",
                "- read_cached_document_tool can reuse the current index context automatically; do not block on explicitly re-supplying index_id when the active procurement context is already present.",
                "- If purchase_search_tool already exposed downloaded procurement files, read_cached_document_tool can read one by bundle_id + file_name before search matches appear. file_name may be the exact name, a downloaded file path, or a short unique file hint.",
                "- You may answer using only the materials already indexed so far.",
                "- If you answer from partial materials, explicitly say retrieval is still in progress.",
                "- Do not say retrieval is complete.",
            ]
        )
    elif retrieval_status == "completed":
        lines.extend(
            [
                "- Tell the user that procurement retrieval is complete and the materials are ready.",
                "- Use the current index_id with doc_search_tool for questions about the retrieved materials.",
                "- If you already know an exact document_id, use read_cached_document_tool to read that cached file content directly.",
                "- read_cached_document_tool can reuse the current index context automatically; do not block on explicitly re-supplying index_id when the active procurement context is already present.",
                "- If purchase_search_tool already exposed downloaded procurement files, read_cached_document_tool can read one by bundle_id + file_name. file_name may be the exact name, a downloaded file path, or a short unique file hint.",
            ]
        )
    elif retrieval_status == "failed":
        lines.extend(
            [
                "- Tell the user that procurement retrieval ended with a failure.",
                "- Use the current index_id with doc_search_tool only if partial materials were indexed before the failure.",
                "- If a document_id is already known from partial materials, read_cached_document_tool can still read its cached content.",
                "- read_cached_document_tool can reuse the current index context automatically; do not block on explicitly re-supplying index_id when the active procurement context is already present.",
                "- If purchase_search_tool already exposed downloaded procurement files, read_cached_document_tool can read one by bundle_id + file_name from the cached artifacts. file_name may be the exact name, a downloaded file path, or a short unique file hint.",
                "- You may still answer from indexed partial materials, but be explicit about the failure.",
            ]
        )
    return "\n".join(lines)


def _is_retryable_http_status(exc: httpx.HTTPStatusError) -> bool:
    status_code = exc.response.status_code if exc.response is not None else None
    return status_code == 429 or (status_code is not None and 500 <= status_code < 600)


def _should_retry_tool_exception(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, PlaywrightTimeoutError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_retryable_http_status(exc)
    return False


def _parse_json_object(raw_value: Any) -> dict[str, Any] | None:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _message_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _current_turn_messages(messages: list[Any]) -> list[Any]:
    last_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = index
    if last_human_index < 0:
        return list(messages)
    return list(messages[last_human_index + 1 :])


def _latest_ai_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""


def _tool_usage_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    tool_name_by_call_id: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                tool_call_id = str(tool_call.get("id") or "").strip()
                tool_name = str(tool_call.get("name") or "").strip()
                if tool_call_id and tool_name:
                    tool_name_by_call_id[tool_call_id] = tool_name
            continue
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = str(message.tool_call_id or "").strip()
        tool_name = str(message.name or tool_name_by_call_id.get(tool_call_id) or "").strip()
        if not tool_name:
            continue
        payload = _parse_json_object(message.content)
        status = str(getattr(message, "status", "") or "").strip() or "success"
        if isinstance(payload, dict) and payload.get("ok") is False:
            status = "error"
        entries.append(
            {
                "tool": tool_name,
                "status": status,
                "tool_call_id": tool_call_id or None,
            }
        )
    return entries


def _tool_payloads(
    messages: list[Any],
    *,
    tool_name: str,
    successful_only: bool = True,
) -> list[dict[str, Any]]:
    tool_name_by_call_id: dict[str, str] = {}
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                tool_call_id = str(tool_call.get("id") or "").strip()
                candidate_name = str(tool_call.get("name") or "").strip()
                if tool_call_id and candidate_name:
                    tool_name_by_call_id[tool_call_id] = candidate_name
            continue
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = str(message.tool_call_id or "").strip()
        candidate_name = str(message.name or tool_name_by_call_id.get(tool_call_id) or "").strip()
        if candidate_name != tool_name:
            continue
        payload = _parse_json_object(message.content)
        if not isinstance(payload, dict):
            continue
        if successful_only and payload.get("ok") is False:
            continue
        payloads.append(payload)
    return payloads


def _latest_tool_payload(
    messages: list[Any],
    *,
    tool_name: str,
    successful_only: bool = True,
) -> dict[str, Any] | None:
    payloads = _tool_payloads(messages, tool_name=tool_name, successful_only=successful_only)
    return payloads[-1] if payloads else None


def _evidence(
    *,
    source: str,
    snippet: str,
    source_url: str | None = None,
    file_path: str | None = None,
    page: int | None = None,
    locator: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"source": source, "snippet": snippet}
    if source_url:
        payload["source_url"] = source_url
    if file_path:
        payload["file_path"] = file_path
    if page is not None:
        payload["page"] = page
    if locator:
        payload["locator"] = locator
    return payload


def _fact_status(*, field: str, status: str, detail: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"field": field, "status": status}
    if detail:
        payload["detail"] = detail
    return payload


def _purchase_items_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw_item in payload.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        registry_number = str(raw_item.get("registry_number") or "").strip()
        title = str(raw_item.get("purchase_title") or "").strip()
        customer_name = str(raw_item.get("customer_name") or "").strip()
        price_text = str(raw_item.get("price_text") or "").strip()
        submission_deadline = str(raw_item.get("submission_deadline") or "").strip()
        snippet_parts = [part for part in (title, customer_name, price_text, submission_deadline) if part]
        if not snippet_parts:
            snippet_parts = [registry_number] if registry_number else []
        items.append(
            {
                "id": registry_number or str(raw_item.get("bundle_id") or ""),
                "title": title or registry_number or "Procurement item",
                "summary": " | ".join(snippet_parts),
                "evidence": [
                    _evidence(
                        source="purchase",
                        snippet=" | ".join(snippet_parts) or "Procurement listing",
                        source_url=str(
                            raw_item.get("detail_url")
                            or raw_item.get("common_info_url")
                            or raw_item.get("documents_url")
                            or ""
                        )
                        or None,
                        locator=f"registry_number={registry_number}" if registry_number else None,
                    )
                ],
                "fact_statuses": [
                    _fact_status(
                        field="procurement_listing",
                        status="external_api",
                        detail=f"registry_number={registry_number}" if registry_number else None,
                    )
                ],
            }
        )
    return items


def _doc_match_items_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, raw_match in enumerate(payload.get("matches") or [], start=1):
        if not isinstance(raw_match, dict):
            continue
        file_path = str(raw_match.get("file_path") or "").strip()
        snippet = str(raw_match.get("snippet") or "").strip()
        if not snippet:
            continue
        page = raw_match.get("page")
        locator = str(raw_match.get("locator") or "").strip() or None
        source_url = str(raw_match.get("source_url") or "").strip() or None
        title = file_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] or f"Document match {index}"
        items.append(
            {
                "id": str(raw_match.get("document_id") or f"document-match-{index}"),
                "title": title,
                "summary": snippet,
                "evidence": [
                    _evidence(
                        source="document",
                        snippet=snippet,
                        file_path=file_path or None,
                        source_url=source_url,
                        page=page if isinstance(page, int) else None,
                        locator=locator,
                    )
                ],
                "fact_statuses": [
                    _fact_status(
                        field="document_match",
                        status="document",
                        detail=locator or (f"page={page}" if isinstance(page, int) else None),
                    )
                ],
            }
        )
    return items


def _counterparty_items_from_payloads(
    *,
    dadata_by_inn: dict[str, dict[str, Any]],
    scoring_by_inn: dict[str, dict[str, Any]],
    fssp_by_inn: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for inn in sorted(set(dadata_by_inn) | set(scoring_by_inn) | set(fssp_by_inn)):
        dadata_payload = dadata_by_inn.get(inn) or {}
        scoring_payload = scoring_by_inn.get(inn) or {}
        fssp_payload = fssp_by_inn.get(inn) or {}
        evidence: list[dict[str, Any]] = []
        fact_statuses: list[dict[str, Any]] = []
        resolved_name = str(dadata_payload.get("name") or "").strip()

        if dadata_payload and bool(dadata_payload.get("found")):
            snippet_parts = [part for part in (
                resolved_name or None,
                str(dadata_payload.get("ogrn") or "").strip() or None,
                str(dadata_payload.get("state_status") or "").strip() or None,
            ) if part]
            evidence.append(_evidence(source="dadata", snippet=" | ".join(snippet_parts) or f"INN {inn}"))
            fact_statuses.append(_fact_status(field="counterparty_lookup", status="external_api"))

        score_payload = scoring_payload.get("score") if isinstance(scoring_payload.get("score"), dict) else {}
        if score_payload:
            snippet = (
                f"INN {inn}: risk_zone={score_payload.get('risk_zone') or 'n/a'}, "
                f"score_zone={score_payload.get('score_zone') or 'n/a'}, "
                f"reliability_zone={score_payload.get('reliability_zone') or 'n/a'}"
            )
            evidence.append(_evidence(source="scoring", snippet=snippet))
            fact_statuses.append(_fact_status(field="scoring", status="external_api"))

        grouped = fssp_payload.get("grouped") if isinstance(fssp_payload.get("grouped"), list) else []
        if grouped:
            total_records = sum(int(item.get("count") or 0) for item in grouped if isinstance(item, dict))
            snippet = (
                f"INN {inn}: FSSP grouped records={len(grouped)}, "
                f"total proceedings={total_records}"
            )
            evidence.append(_evidence(source="fssp", snippet=snippet))
            fact_statuses.append(_fact_status(field="fssp", status="external_api"))

        if not evidence:
            continue
        items.append(
            {
                "id": inn,
                "title": resolved_name or f"INN {inn}",
                "summary": " | ".join(entry["snippet"] for entry in evidence),
                "evidence": evidence,
                "fact_statuses": fact_statuses,
            }
        )
    return items


def _recommended_next_step(
    *,
    retrieval_status: str | None,
    has_issues: bool,
) -> str | None:
    if retrieval_status in {"queued", "in_progress"}:
        return (
            "Wait for the procurement retrieval to progress, ask for current status, "
            "or ask about materials already indexed so far."
        )
    if retrieval_status == "failed":
        return (
            "Review the reported retrieval failure, then retry the search or ask for answers "
            "based only on the partial materials already indexed."
        )
    if has_issues:
        return "Fix the reported issue and ask the agent to continue."
    return None


def _normalized_answer_type(
    *,
    purchase_payload: dict[str, Any] | None,
    doc_search_payload: dict[str, Any] | None,
    counterparty_items: list[dict[str, Any]],
    previous_answer_type: str | None,
) -> str:
    if doc_search_payload and (doc_search_payload.get("matches") or []):
        return "lead_card"
    if len(counterparty_items) > 1:
        return "comparison"
    if len(counterparty_items) == 1:
        return "company_check"
    if purchase_payload and (purchase_payload.get("items") or []):
        return "lead_list"
    if purchase_payload and str(purchase_payload.get("retrieval_status") or "").strip():
        return "status_update"
    if previous_answer_type:
        return previous_answer_type
    return "text"


def _build_normalized_final_answer(state: SalesLeadAgentState) -> dict[str, Any]:
    messages = _message_list(state.get("messages"))
    current_turn_messages = _current_turn_messages(messages)
    previous_normalized = (
        state.get("normalized_final_answer")
        if isinstance(state.get("normalized_final_answer"), dict)
        else {}
    )

    purchase_payload = _latest_tool_payload(current_turn_messages, tool_name="purchase_search_tool")
    if not purchase_payload and isinstance(state.get("purchase_search_result"), dict):
        purchase_payload = dict(state.get("purchase_search_result") or {})
    retrieve_page_payload = _latest_tool_payload(current_turn_messages, tool_name="retrieve_page_tool")
    doc_search_payload = _latest_tool_payload(current_turn_messages, tool_name="doc_search_tool")
    if not doc_search_payload and isinstance(state.get("last_doc_search_result"), dict):
        doc_search_payload = dict(state.get("last_doc_search_result") or {})
    dadata_by_inn = {
        str(payload.get("inn") or "").strip(): payload
        for payload in _tool_payloads(current_turn_messages, tool_name="counterparty_lookup_tool")
        if isinstance(payload, dict) and str(payload.get("inn") or "").strip()
    }
    scoring_by_inn = {
        str(payload.get("inn") or "").strip(): payload
        for payload in _tool_payloads(current_turn_messages, tool_name="counterparty_scoring_tool")
        if isinstance(payload, dict) and str(payload.get("inn") or "").strip()
    }
    fssp_by_inn = {
        str(payload.get("inn") or "").strip(): payload
        for payload in _tool_payloads(current_turn_messages, tool_name="counterparty_fssp_tool")
        if isinstance(payload, dict) and str(payload.get("inn") or "").strip()
    }

    counterparty_items = _counterparty_items_from_payloads(
        dadata_by_inn=dadata_by_inn,
        scoring_by_inn=scoring_by_inn,
        fssp_by_inn=fssp_by_inn,
    )
    items: list[dict[str, Any]]
    if doc_search_payload and (doc_search_payload.get("matches") or []):
        items = _doc_match_items_from_payload(doc_search_payload)
    elif counterparty_items:
        items = counterparty_items
    elif purchase_payload and (purchase_payload.get("items") or []):
        items = _purchase_items_from_payload(purchase_payload)
    elif not current_turn_messages and isinstance(previous_normalized.get("items"), list):
        items = previous_normalized.get("items") or []
    elif not _tool_usage_from_messages(current_turn_messages) and isinstance(previous_normalized.get("items"), list):
        items = previous_normalized.get("items") or []
    else:
        items = []

    retrieval_status = str(
        (purchase_payload or {}).get("retrieval_status")
        or state.get("active_retrieval_status")
        or ""
    ).strip() or None
    answer_type = _normalized_answer_type(
        purchase_payload=purchase_payload,
        doc_search_payload=doc_search_payload,
        counterparty_items=counterparty_items,
        previous_answer_type=str(previous_normalized.get("answer_type") or "").strip() or None,
    )

    missing_data: list[str] = []
    if retrieval_status in {"queued", "in_progress"}:
        missing_data.append("background procurement retrieval is still in progress")
    elif retrieval_status == "failed":
        missing_data.append("background procurement retrieval failed before full completion")

    issues: list[str] = []
    for entry in _tool_usage_from_messages(current_turn_messages):
        if str(entry.get("status") or "").strip() == "error":
            tool_name = str(entry.get("tool") or "tool").strip()
            issues.append(f"{tool_name} returned an error")

    recommended_next_step = _recommended_next_step(
        retrieval_status=retrieval_status,
        has_issues=bool(issues),
    )
    if not recommended_next_step and missing_data:
        recommended_next_step = "Ask the agent to continue once more materials become available."

    reply_messages = current_turn_messages or messages
    summary = _latest_ai_text(reply_messages)
    if not summary:
        summary = _latest_ai_text(messages)

    normalized: dict[str, Any] = {
        "answer_type": answer_type,
        "summary": summary,
        "items": items,
        "missing_data": missing_data,
    }
    if retrieval_status:
        normalized["retrieval_status"] = retrieval_status
    if purchase_payload and str(purchase_payload.get("message") or "").strip():
        normalized["message"] = str(purchase_payload.get("message") or "").strip()
    if recommended_next_step:
        normalized["recommended_next_step"] = recommended_next_step
    return normalized


def _build_turn_validation(
    *,
    current_turn_messages: list[Any],
    normalized_final_answer: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    for entry in _tool_usage_from_messages(current_turn_messages):
        if str(entry.get("status") or "").strip() == "error":
            tool_name = str(entry.get("tool") or "tool").strip()
            issues.append(f"{tool_name} returned an error")
    if normalized_final_answer.get("missing_data"):
        issues.extend(str(item) for item in normalized_final_answer.get("missing_data") or [])
    return {"issues": issues}


def _finalize_agent_state(state: SalesLeadAgentState) -> dict[str, Any]:
    result = dict(state)
    messages = _message_list(result.get("messages"))
    current_turn_messages = _current_turn_messages(messages)

    turn_tool_usage = _tool_usage_from_messages(current_turn_messages)
    purchase_payload = _latest_tool_payload(current_turn_messages, tool_name="purchase_search_tool")
    retrieve_page_payload = _latest_tool_payload(current_turn_messages, tool_name="retrieve_page_tool")
    doc_search_payload = _latest_tool_payload(current_turn_messages, tool_name="doc_search_tool")

    if turn_tool_usage:
        result["turn_tool_usage"] = turn_tool_usage
    else:
        result["turn_tool_usage"] = []

    if purchase_payload:
        result["purchase_search_result"] = purchase_payload
    else:
        purchase_payload = (
            result.get("purchase_search_result")
            if isinstance(result.get("purchase_search_result"), dict)
            else None
        )
    if doc_search_payload:
        result["last_doc_search_result"] = doc_search_payload
    else:
        doc_search_payload = (
            result.get("last_doc_search_result")
            if isinstance(result.get("last_doc_search_result"), dict)
            else None
        )

    prepared_documents: list[dict[str, Any]] = []
    if isinstance((retrieve_page_payload or {}).get("prepared_documents"), list):
        prepared_documents = list((retrieve_page_payload or {}).get("prepared_documents") or [])
    result["prepared_documents"] = prepared_documents

    active_run_id = (
        str((purchase_payload or {}).get("run_id") or "").strip()
        or str((retrieve_page_payload or {}).get("run_id") or "").strip()
        or str(result.get("active_retrieval_run_id") or "").strip()
        or None
    )
    index_id = (
        str((purchase_payload or {}).get("index_id") or "").strip()
        or str((doc_search_payload or {}).get("index_id") or "").strip()
        or str((retrieve_page_payload or {}).get("index_id") or "").strip()
        or str(result.get("active_retrieval_index_id") or "").strip()
        or None
    )
    result["active_run_id"] = active_run_id
    result["index_id"] = index_id

    normalized_final_answer = _build_normalized_final_answer(result)
    result["normalized_final_answer"] = normalized_final_answer
    result["turn_validation"] = _build_turn_validation(
        current_turn_messages=current_turn_messages,
        normalized_final_answer=normalized_final_answer,
    )
    return result


class ToolErrorJsonMiddleware(AgentMiddleware):
    """Return structured JSON only for LLM-fixable or exhausted transient tool failures."""

    @staticmethod
    def _build_tool_message(
        request: ToolCallRequest,
        *,
        payload: dict[str, Any],
        exc: Exception,
    ) -> ToolMessage:
        tool_name = request.tool_call["name"]
        logger.error(
            "sales_lead_agent tool failed: tool=%s args=%r",
            tool_name,
            request.tool_call.get("args"),
            exc_info=True,
        )
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            artifact={
                "tool": tool_name,
                "args": request.tool_call.get("args"),
                "error_type": exc.__class__.__name__,
            },
            status="error",
        )

    @classmethod
    def _user_correctable_message(
        cls,
        request: ToolCallRequest,
        exc: ToolUserCorrectableError,
    ) -> ToolMessage:
        return cls._build_tool_message(
            request,
            payload={
                "ok": False,
                "error_code": exc.code,
                "message": str(exc),
                "retryable": True,
                "suggestion": exc.suggestion,
                "input_field": exc.input_field,
            },
            exc=exc,
        )

    @classmethod
    def _transient_failure_message(
        cls,
        request: ToolCallRequest,
        exc: Exception,
    ) -> ToolMessage:
        return cls._build_tool_message(
            request,
            payload={
                "ok": False,
                "error_code": "TRANSIENT_TOOL_FAILURE",
                "message": str(exc),
                "retryable": True,
                "suggestion": "Retry the same tool call again.",
                "input_field": None,
            },
            exc=exc,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        try:
            return handler(request)
        except ToolUserCorrectableError as exc:
            return self._user_correctable_message(request, exc)
        except httpx.HTTPStatusError as exc:
            if _is_retryable_http_status(exc):
                return self._transient_failure_message(request, exc)
            raise
        except (
            asyncio.TimeoutError,
            TimeoutError,
            PlaywrightTimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
        ) as exc:
            return self._transient_failure_message(request, exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except ToolUserCorrectableError as exc:
            return self._user_correctable_message(request, exc)
        except httpx.HTTPStatusError as exc:
            if _is_retryable_http_status(exc):
                return self._transient_failure_message(request, exc)
            raise
        except (
            asyncio.TimeoutError,
            TimeoutError,
            PlaywrightTimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.TransportError,
        ) as exc:
            return self._transient_failure_message(request, exc)


def _build_inner_agent(
    *,
    llm: Any,
    tools: Sequence[Any],
    base_system_prompt: str,
) -> Any:
    @dynamic_prompt
    def build_prompt(request: ModelRequest) -> str:
        note = _retrieval_state_note(request.state)
        if not note:
            return base_system_prompt
        return f"{base_system_prompt}\n\n{note}"

    inner_agent = create_agent(
        model=llm,
        tools=list(tools),
        system_prompt=base_system_prompt,
        middleware=[
            build_prompt,
            ToolErrorJsonMiddleware(),
            ToolRetryMiddleware(max_retries=2, retry_on=_should_retry_tool_exception, on_failure="error"),
        ],
        state_schema=SalesLeadAgentState,
        name="sales_lead_agent_inner",
    )
    return inner_agent | RunnableLambda(_finalize_agent_state)


async def _refresh_retrieval_state(
    state: SalesLeadAgentState,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    retrieval_id = state.get("active_retrieval_id")
    if not isinstance(retrieval_id, str) or not retrieval_id.strip():
        return {}

    snapshot = await get_retrieval_service_client().get_retrieval(
        retrieval_id=retrieval_id,
        include_payloads=True,
    )
    if snapshot is None:
        return {
            "active_retrieval_id": None,
            "active_retrieval_request_hash": None,
            "active_retrieval_run_id": None,
            "active_retrieval_index_id": None,
            "active_retrieval_status": None,
            "active_retrieval_stage": None,
            "active_retrieval_message": None,
            "active_retrieval_progress": None,
            "active_run_id": None,
            "index_id": None,
            "purchase_search_result": None,
            "prepared_documents": [],
        }

    if snapshot.status in {"queued", "in_progress"}:
        runtime.stream_writer(
            {
                "type": "progress",
                "tool": "purchase_search_tool",
                "stage": snapshot.stage,
                "message": snapshot.message,
                "retrieval_status": snapshot.status,
                "run_id": snapshot.run_id,
                "index_id": snapshot.index_id,
                "progress": snapshot.progress.model_dump(),
            }
        )
    purchase_response = _purchase_response_from_snapshot(snapshot).model_dump()
    return {
        **_retrieval_state_from_snapshot(snapshot),
        "active_run_id": str(snapshot.run_id),
        "index_id": str(snapshot.index_id),
        "purchase_search_result": purchase_response,
        "prepared_documents": [],
    }


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    *,
    model_size: str = "base",
    temperature: float = 0.1,
    system_prompt: str | None = None,
    tools: Sequence[Any] | None = None,
    checkpoint_saver: Any | None = None,
    streaming: bool = False,
    reasoning: str | None = None,
    max_tool_calls: int | None = 12,
):
    """Initialize the sales lead agent with an explicit init node."""
    if model_size not in VALID_MODEL_SIZES:
        choices = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ValueError(f"Unknown model size '{model_size}'. Available values: {choices}")

    callback_handlers = []
    if config.LANGFUSE_URL:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers.append(CallbackHandler())

    llm = get_llm(
        model=model_size,
        provider=provider.value,
        temperature=temperature,
        streaming=streaming,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
    )
    base_system_prompt = build_system_prompt() if system_prompt is None else system_prompt
    resolved_tools = list(build_sales_lead_tools() if tools is None else tools)
    run_agent = _build_inner_agent(
        llm=llm,
        tools=resolved_tools,
        base_system_prompt=base_system_prompt,
    )

    builder = StateGraph(SalesLeadAgentState)
    builder.add_node("refresh_state", _refresh_retrieval_state)
    builder.add_node("run_agent", run_agent)
    builder.add_edge(START, "refresh_state")
    builder.add_edge("refresh_state", "run_agent")
    builder.add_edge("run_agent", END)

    graph = builder.compile(
        checkpointer=MemorySaver() if checkpoint_saver is None else checkpoint_saver,
        debug=False,
    )
    return graph.with_config({"callbacks": callback_handlers})
