from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.sales_lead_agent import agent as sales_agent
from agents.sales_lead_agent.agent import (
    ToolErrorJsonMiddleware,
    _refresh_retrieval_state,
    _retrieval_state_note,
    initialize_agent,
)
from agents.sales_lead_agent.prompts import build_system_prompt
from agents.sales_lead_agent.state import SalesLeadAgentState
from agents.sales_lead_agent.tools import ToolUserCorrectableError
from agents.utils import ModelType


@tool
def ping_tool() -> str:
    """Return a static value."""
    return "pong"


def test_system_prompt_contains_minimal_tool_only_contract():
    prompt = build_system_prompt()

    assert "There is no hidden orchestration" in prompt
    assert "ok=false" in prompt
    assert "reusing `index_id`" in prompt
    assert "query_texts" in prompt
    assert "counterparty_lookup_tool" in prompt
    assert "ALWAYS start with `web_search`" in prompt
    assert "Call `retrieve_page_tool` only after `web_search`" in prompt
    assert "read_cached_document_tool" in prompt
    assert "record_from" in prompt
    assert "downloaded file path" in prompt
    assert "reuse the current `index_id` automatically" in prompt


def test_initialize_agent_returns_compiled_agent():
    agent = initialize_agent(
        provider=ModelType.GPT,
        model_size="mini",
        tools=[ping_tool],
        streaming=False,
    )

    assert agent is not None
    assert hasattr(agent, "invoke")


def test_initialize_agent_does_not_swallow_unexpected_kwargs():
    with pytest.raises(TypeError):
        initialize_agent(provider=ModelType.GPT, unknown_argument=True)


def test_initialize_agent_builds_inner_agent_with_expected_contract(monkeypatch):
    captured = {}

    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "llm")

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return lambda state: state

    monkeypatch.setattr(sales_agent, "create_agent", fake_create_agent)
    monkeypatch.setattr(
        sales_agent,
        "build_sales_lead_tools",
        lambda: (_ for _ in ()).throw(AssertionError("default tools must not be built")),
    )

    agent = initialize_agent(
        provider=ModelType.GPT,
        model_size="mini",
        tools=[],
        system_prompt="",
        checkpoint_saver=False,
        streaming=False,
    )

    assert agent is not None
    assert captured["tools"] == []
    assert captured["system_prompt"] == ""
    assert len(captured["middleware"]) == 3
    assert isinstance(captured["middleware"][1], ToolErrorJsonMiddleware)
    assert isinstance(captured["middleware"][2], ToolRetryMiddleware)
    assert captured["state_schema"] is SalesLeadAgentState


def test_tool_error_middleware_returns_json_error_tool_message():
    middleware = ToolErrorJsonMiddleware()
    request = SimpleNamespace(
        tool_call={
            "id": "call-1",
            "name": "purchase_search_tool",
            "args": {"query_texts": ["страхован"]},
        }
    )

    async def handler(_request):
        raise ToolUserCorrectableError(
            code="INVALID_QUERY_TEXT",
            message="query_texts must not be empty",
            suggestion="Provide a non-empty query_texts list and call the tool again.",
            input_field="query_texts",
        )

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload == {
        "ok": False,
        "error_code": "INVALID_QUERY_TEXT",
        "message": "query_texts must not be empty",
        "retryable": True,
        "suggestion": "Provide a non-empty query_texts list and call the tool again.",
        "input_field": "query_texts",
    }


def test_tool_error_middleware_does_not_mask_programmer_errors():
    middleware = ToolErrorJsonMiddleware()
    request = SimpleNamespace(
        tool_call={
            "id": "call-1",
            "name": "purchase_search_tool",
            "args": {"query_texts": ["страхован"]},
        }
    )

    async def handler(_request):
        raise AssertionError("bug")

    with pytest.raises(AssertionError, match="bug"):
        asyncio.run(middleware.awrap_tool_call(request, handler))


def test_retrieval_state_note_uses_agent_state_fields():
    note = _retrieval_state_note(
        {
            "messages": [],
            "active_retrieval_id": "ret-1",
            "active_retrieval_run_id": "run-1",
            "active_retrieval_index_id": "idx-1",
            "active_retrieval_status": "in_progress",
            "active_retrieval_stage": "purchase_processing",
            "active_retrieval_message": "Processing procurement 123 [1/2].",
            "active_retrieval_progress": {
                "total_queries": 1,
                "completed_queries": 1,
                "total_purchases": 2,
                "processed_purchases": 1,
                "total_files": 4,
                "processed_files": 2,
                "prepared_documents": 1,
                "indexed_segments": 3,
            },
            "purchase_search_result": {
                "items": [
                    {
                        "registry_number": "123",
                        "purchase_title": "Transport insurance procurement",
                    }
                ]
            },
        }
    )

    assert note is not None
    assert "Shared procurement index context" in note
    assert "default_index_id: sales_lead_permanent" in note
    assert "retrieval_status: in_progress" in note
    assert "index_id: idx-1" in note
    assert "purchase_lookup_tool first" in note
    assert "Use the current index_id with doc_search_tool" in note
    assert "read_cached_document_tool" in note
    assert "short unique file hint" in note
    assert "reuse the current index context automatically" in note
    assert "explicitly say retrieval is still in progress" in note


def test_refresh_state_updates_agent_state_and_streams_progress(monkeypatch):
    snapshot = SimpleNamespace(
        retrieval_id="ret-1",
        request_hash="hash-1",
        run_id="run-1",
        index_id="idx-1",
        status="in_progress",
        stage="purchase_processing",
        message="Processing procurement 123 [1/2].",
        request_payload={"search_urls": ["https://zakupki.gov.ru/example"]},
        items=[
            {
                "bundle_id": "123",
                "registry_number": "123",
                "law": "44-FZ",
                "purchase_title": "Transport insurance procurement",
                "customer_name": "Customer",
                "price_text": None,
                "published_at": None,
                "updated_at": None,
                "submission_deadline": None,
                "detail_url": "https://zakupki.gov.ru/item/123",
                "common_info_url": None,
                "documents_url": None,
                "document_urls": [],
                "downloaded_files": [],
                "prepared_document_ids": [],
                "documents_json": None,
                "common_info_json": None,
                "lots_json": None,
                "crawl_status": "success",
                "crawl_error": None,
                "crawl_ts_utc": None,
            }
        ],
        prepared_documents=[],
        progress=SimpleNamespace(
            model_dump=lambda: {
                "total_queries": 1,
                "completed_queries": 1,
                "total_purchases": 2,
                "processed_purchases": 1,
                "total_files": 4,
                "processed_files": 2,
                "prepared_documents": 1,
                "indexed_segments": 3,
            }
        ),
    )
    events = []

    class FakeClient:
        async def get_retrieval(self, *, retrieval_id: str, include_payloads: bool = False):
            assert retrieval_id == "ret-1"
            assert include_payloads is False
            return snapshot

    monkeypatch.setattr(sales_agent, "get_retrieval_service_client", lambda: FakeClient())

    state = SalesLeadAgentState(messages=[], active_retrieval_id="ret-1")
    runtime = SimpleNamespace(stream_writer=lambda payload: events.append(payload))

    update = asyncio.run(_refresh_retrieval_state(state, runtime))

    assert update["active_retrieval_id"] == "ret-1"
    assert update["active_retrieval_status"] == "in_progress"
    assert update["active_retrieval_index_id"] == "idx-1"
    assert update["active_retrieval_progress"]["indexed_segments"] == 3
    assert update["active_run_id"] == "run-1"
    assert update["default_index_id"] == "sales_lead_permanent"
    assert update["index_id"] == "idx-1"
    assert update["prepared_documents"] == []
    assert events == [
        {
            "type": "progress",
            "tool": "purchase_search_tool",
            "stage": "purchase_processing",
            "message": "Processing procurement 123 [1/2].",
            "retrieval_status": "in_progress",
            "run_id": "run-1",
            "index_id": "idx-1",
            "progress": {
                "total_queries": 1,
                "completed_queries": 1,
                "total_purchases": 2,
                "processed_purchases": 1,
                "total_files": 4,
                "processed_files": 2,
                "prepared_documents": 1,
                "indexed_segments": 3,
            },
        }
    ]


def test_finalize_agent_state_uses_refreshed_purchase_snapshot_without_new_tool_call():
    state = SalesLeadAgentState(
        messages=[
            HumanMessage(content="Find procurements."),
            ToolMessage(
                content=json.dumps(
                    {
                        "run_id": "run-old",
                        "index_id": "idx-old",
                        "retrieval_status": "queued",
                        "retrieval_stage": "queued",
                        "message": "Queued.",
                        "progress": {},
                        "search_urls": [],
                        "items": [],
                    },
                    ensure_ascii=False,
                ),
                tool_call_id="call-1",
                name="purchase_search_tool",
            ),
            AIMessage(content="Queued."),
            HumanMessage(content="Show what is already found."),
            AIMessage(content="There are already procurement items in progress."),
        ],
        active_retrieval_id="ret-1",
        active_retrieval_run_id="run-1",
        active_retrieval_index_id="idx-1",
        active_retrieval_status="in_progress",
        active_retrieval_stage="crawler_search",
        active_retrieval_message="Looking zakupki.gov.ru [2/5] with search string: каско",
        active_retrieval_progress={
            "total_queries": 5,
            "completed_queries": 1,
            "total_purchases": 10,
            "processed_purchases": 0,
            "total_files": 0,
            "processed_files": 0,
            "prepared_documents": 0,
            "indexed_segments": 0,
        },
        purchase_search_result={
            "run_id": "run-1",
            "index_id": "idx-1",
            "retrieval_status": "in_progress",
            "retrieval_stage": "crawler_search",
            "message": "Looking zakupki.gov.ru [2/5] with search string: каско",
            "progress": {
                "total_queries": 5,
                "completed_queries": 1,
                "total_purchases": 10,
                "processed_purchases": 0,
                "total_files": 0,
                "processed_files": 0,
                "prepared_documents": 0,
                "indexed_segments": 0,
            },
            "search_urls": ["https://zakupki.gov.ru/example"],
            "items": [
                {
                    "bundle_id": "123",
                    "registry_number": "123",
                    "law": "44-FZ",
                    "purchase_title": "Transport insurance procurement",
                    "customer_name": "Customer",
                    "price_text": None,
                    "published_at": None,
                    "updated_at": None,
                    "submission_deadline": None,
                    "detail_url": "https://zakupki.gov.ru/item/123",
                    "common_info_url": None,
                    "documents_url": None,
                    "document_urls": [],
                    "downloaded_files": [],
                    "prepared_document_ids": [],
                    "documents_json": None,
                    "common_info_json": None,
                    "lots_json": None,
                    "crawl_status": "success",
                    "crawl_error": None,
                    "crawl_ts_utc": None,
                }
            ],
        },
        last_purchase_lookup_result={
            "index_id": "idx-1",
            "record_from": 0,
            "returned_records": 1,
            "total_ready_records": 1,
            "next_record_from": None,
            "items": [
                {
                    "bundle_id": "123",
                    "registry_number": "123",
                    "law": "44-FZ",
                    "purchase_title": "Transport insurance procurement",
                    "customer_name": "Customer",
                    "detail_url": "https://zakupki.gov.ru/item/123",
                    "crawl_status": "success",
                }
            ],
        },
    )

    result = sales_agent._finalize_agent_state(state)

    assert result["active_run_id"] == "run-1"
    assert result["index_id"] == "idx-1"
    assert result["purchase_search_result"]["items"][0]["registry_number"] == "123"
    assert result["normalized_final_answer"]["answer_type"] == "lead_list"
    assert result["normalized_final_answer"]["items"][0]["id"] == "123"


def test_finalize_agent_state_uses_dadata_lookup_for_company_name():
    state = SalesLeadAgentState(
        messages=[
            HumanMessage(content="Проверь контрагента по ИНН 6663003127"),
            ToolMessage(
                content=json.dumps(
                    {
                        "source": "dadata_party",
                        "status": "success",
                        "inn": "6663003127",
                        "found": True,
                        "name": "АО \"ПФ \"СКБ Контур\"",
                        "full_name": "АКЦИОНЕРНОЕ ОБЩЕСТВО \"ПРОИЗВОДСТВЕННАЯ ФИРМА \\\"СКБ КОНТУР\\\"\"",
                        "state_status": "ACTIVE",
                        "ogrn": "1026605606620",
                    },
                    ensure_ascii=False,
                ),
                tool_call_id="call-lookup",
                name="counterparty_lookup_tool",
            ),
            AIMessage(content="Нашёл официальное наименование."),
        ]
    )

    result = sales_agent._finalize_agent_state(state)

    assert result["normalized_final_answer"]["answer_type"] == "company_check"
    assert result["normalized_final_answer"]["items"][0]["id"] == "6663003127"
    assert result["normalized_final_answer"]["items"][0]["title"] == "АО \"ПФ \"СКБ Контур\""
