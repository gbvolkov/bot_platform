import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage

from agents.sales_lead_agent import agent as sales_agent
from agents.sales_lead_agent.schemas import (
    LeadAnswerContract,
    PreparedDocument,
    PreparedDocumentEntities,
    PurchaseSearchItem,
    PurchaseSearchResponse,
    SearchFilters,
    TurnValidationResult,
)
from agents.sales_lead_agent.tools import SalesLeadAgentDependencies


def _deps() -> SalesLeadAgentDependencies:
    fake = SimpleNamespace()
    return SalesLeadAgentDependencies(
        workspace_manager=fake,
        document_service=fake,
        classifier=fake,
        purchase_adapter=fake,
        counterparty_clients=fake,
        open_source_max_concurrency=4,
    )


def test_initialize_agent_uses_deepagent_bootstrap(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    result = sales_agent.initialize_agent(dependencies=_deps(), streaming=False)

    assert result["graph"] == "ok"
    assert captured["name"] == "sales_lead_agent"
    assert captured["model"] == "fake-llm"
    assert len(captured["tools"]) == 5
    assert isinstance(captured["response_format"], ProviderStrategy)
    assert captured["response_format"].schema is LeadAnswerContract


def test_prepare_runtime_state_records_intent_classifier_failure(monkeypatch):
    captured = {}

    class FailingClassifier:
        def classify_intent(self, **kwargs):
            raise sales_agent.ClassifierExecutionError("intent boom")

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=FailingClassifier(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    update = prepare_runtime_state(
        {"messages": [HumanMessage(content="check company")], "missing_data": []},
        SimpleNamespace(),
    )

    assert update is not None
    assert update["missing_data"] == ["intent_classification"]
    assert update["turn_validation"]["status"] == "failed_verification"
    assert update["turn_validation"]["issues"][0]["code"] == "intent_classifier_failed"
    assert "Retry the request" in update["recommended_next_step"]
    assert update["task_understanding"] is None
    assert update["assessment"] is None
    assert update["risk_verification"] is None


def test_prepare_runtime_state_autoruns_purchase_search_for_procurement_turn(monkeypatch, tmp_path):
    captured = {}
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="index-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    class Classifier:
        def classify_intent(self, **kwargs):
            filters = SearchFilters(query_text="transport insurance")
            return SimpleNamespace(
                answer_type="lead_list",
                task_kind="procurement_search",
                requested_company_inns=[],
                comparison_targets=[],
                search_url=None,
                search_filters=filters,
                needs_purchase_search=True,
                needs_open_source=False,
                needs_doc_search=False,
                needs_enrichment=False,
                missing_data=[],
                model_dump=lambda: {
                    "answer_type": "lead_list",
                    "task_kind": "procurement_search",
                    "search_url": None,
                    "search_filters": filters.model_dump(),
                    "requested_company_inns": [],
                    "comparison_targets": [],
                    "document_questions": [],
                    "needs_purchase_search": True,
                    "needs_open_source": False,
                    "needs_doc_search": False,
                    "needs_enrichment": False,
                    "missing_data": [],
                },
            )

        def classify_procurement_hits(self, **kwargs):
            return SimpleNamespace(decisions=[], model_dump=lambda: {"decisions": []})

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(
            prepare_files=lambda **kwargs: [
                PreparedDocument(
                    document_id="doc-1",
                    origin="purchase",
                    bundle_id="123",
                    registry_number="123",
                    source_url="https://example.test/purchase/123",
                    file_path="C:/tmp/doc.txt",
                    file_name="doc.txt",
                    file_type="other",
                    parse_status="success",
                    index_status="ready",
                    text_excerpt="insurance",
                    entities=PreparedDocumentEntities(),
                    chunks_count=2,
                    error=None,
                )
            ]
        ),
        classifier=Classifier(),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (
                "https://example.test/search",
                PurchaseSearchResponse(
                    run_id="",
                    index_id="",
                    status="success",
                    errors=[],
                    items=[
                        PurchaseSearchItem(
                            bundle_id="123",
                            registry_number="123",
                            law="44-FZ",
                            purchase_title="Transport insurance procurement",
                            customer_name="Acme",
                            price_text="100",
                            published_at=None,
                            updated_at=None,
                            submission_deadline=None,
                            detail_url="https://example.test/purchase/123",
                            common_info_url=None,
                            documents_url=None,
                            document_urls=[],
                            downloaded_files=["C:/tmp/doc.txt"],
                            prepared_document_ids=[],
                            documents_json=None,
                            common_info_json=None,
                            lots_json=None,
                            crawl_status="success",
                            crawl_error=None,
                            crawl_ts_utc="2026-03-23T00:00:00Z",
                        )
                    ],
                    prepared_documents=[],
                ),
            ),
            summarize_hits=lambda response: [
                {
                    "bundle_id": "123",
                    "registry_number": "123",
                    "purchase_title": "Transport insurance procurement",
                }
            ],
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    update = prepare_runtime_state(
        {"messages": [HumanMessage(content="find insurance procurements")], "missing_data": []},
        SimpleNamespace(),
    )

    assert update["turn_tool_usage"][0]["tool"] == "purchase_search_tool"
    assert update["active_run_id"] == "run-1"
    assert update["index_id"] == "index-1"
    assert update["procurement_hits"][0]["registry_number"] == "123"


def test_prepare_runtime_state_autoruns_purchase_search_for_procurement_analysis_when_classifier_requires_acquisition(
    monkeypatch,
    tmp_path,
):
    captured = {}
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="index-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    class Classifier:
        def classify_intent(self, **kwargs):
            filters = SearchFilters(query_text="страхование грузов")
            return SimpleNamespace(
                answer_type="lead_card",
                task_kind="procurement_analysis",
                requested_company_inns=[],
                comparison_targets=[],
                search_url=None,
                search_filters=filters,
                missing_data=[],
                needs_purchase_search=True,
                model_dump=lambda: {
                    "answer_type": "lead_card",
                    "task_kind": "procurement_analysis",
                    "search_url": None,
                    "search_filters": filters.model_dump(),
                    "requested_company_inns": [],
                    "comparison_targets": [],
                    "document_questions": ["Какие риски покрываются?"],
                    "needs_purchase_search": True,
                    "needs_open_source": False,
                    "needs_doc_search": True,
                    "needs_enrichment": False,
                    "missing_data": [],
                },
            )

        def classify_procurement_hits(self, **kwargs):
            return SimpleNamespace(decisions=[], model_dump=lambda: {"decisions": []})

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(
            prepare_files=lambda **kwargs: [
                PreparedDocument(
                    document_id="doc-1",
                    origin="purchase",
                    bundle_id="123",
                    registry_number="123",
                    source_url="https://example.test/purchase/123",
                    file_path="C:/tmp/doc.txt",
                    file_name="doc.txt",
                    file_type="other",
                    parse_status="success",
                    index_status="ready",
                    text_excerpt="insurance",
                    entities=PreparedDocumentEntities(),
                    chunks_count=2,
                    error=None,
                )
            ]
        ),
        classifier=Classifier(),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (
                "https://example.test/search",
                PurchaseSearchResponse(
                    run_id="",
                    index_id="",
                    status="success",
                    errors=[],
                    items=[
                        PurchaseSearchItem(
                            bundle_id="123",
                            registry_number="123",
                            law="44-FZ",
                            purchase_title="Transport insurance procurement",
                            customer_name="Acme",
                            price_text="100",
                            published_at=None,
                            updated_at=None,
                            submission_deadline=None,
                            detail_url="https://example.test/purchase/123",
                            common_info_url=None,
                            documents_url=None,
                            document_urls=[],
                            downloaded_files=["C:/tmp/doc.txt"],
                            prepared_document_ids=[],
                            documents_json=None,
                            common_info_json=None,
                            lots_json=None,
                            crawl_status="success",
                            crawl_error=None,
                            crawl_ts_utc="2026-03-23T00:00:00Z",
                        )
                    ],
                    prepared_documents=[],
                ),
            ),
            summarize_hits=lambda response: [
                {
                    "bundle_id": "123",
                    "registry_number": "123",
                    "purchase_title": "Transport insurance procurement",
                }
            ],
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    update = prepare_runtime_state(
        {"messages": [HumanMessage(content="analyze procurement risks")], "missing_data": []},
        SimpleNamespace(),
    )

    assert update["turn_tool_requirements"]["purchase_search_required"] is True
    assert update["turn_tool_usage"][0]["tool"] == "purchase_search_tool"
    assert update["active_run_id"] == "run-1"
    assert update["index_id"] == "index-1"


def test_finalize_answer_degrades_when_structured_response_missing(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="check company")],
            "task_understanding": {"answer_type": "company_check"},
            "normalized_inns": ["7707083893"],
            "company_names": ["Test LLC"],
            "missing_data": ["company_name"],
        },
        SimpleNamespace(),
    )

    assert update is not None
    assert update["normalized_final_answer"]["answer_type"] == "company_check"
    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert update["turn_validation"]["issues"][0]["code"] == "structured_response_missing"


def test_recover_model_failures_marks_structured_output_validation_as_recoverable(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    recover_model_failures = captured["middleware"][3].wrap_model_call
    finalize_answer = captured["middleware"][2].after_agent

    state = {
        "messages": [HumanMessage(content="find leads")],
        "task_understanding": {"answer_type": "lead_list"},
        "missing_data": [],
    }
    request = SimpleNamespace(state=state)

    def failing_handler(_request):
        raise StructuredOutputValidationError(
            "LeadAnswerContract",
            ValueError("extra field"),
            AIMessage(content=""),
        )

    response = recover_model_failures(request, failing_handler)
    assert isinstance(response, AIMessage)
    assert state["turn_validation"]["issues"][0]["code"] == "structured_output_validation_failed"
    assert "final_structured_output" in state["missing_data"]

    update = finalize_answer(state, SimpleNamespace())
    assert "structured_output_validation_failed" in update["normalized_final_answer"]["missing_data"]


def test_recover_model_failures_marks_provider_error_as_recoverable(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    class FakeOpenAIError(Exception):
        __module__ = "openai"

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    recover_model_failures = captured["middleware"][3].wrap_model_call

    state = {
        "messages": [HumanMessage(content="find leads")],
        "task_understanding": {"answer_type": "lead_list"},
        "missing_data": [],
    }
    request = SimpleNamespace(state=state)

    def failing_handler(_request):
        raise FakeOpenAIError("rate limited")

    response = recover_model_failures(request, failing_handler)
    assert isinstance(response, AIMessage)
    assert state["turn_validation"]["issues"][0]["code"] == "final_model_provider_failed"
    assert "final_model_provider" in state["missing_data"]


def test_recover_model_failures_async_marks_provider_error_as_recoverable(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    class FakeOpenAIError(Exception):
        __module__ = "openai"

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    recover_model_failures = captured["middleware"][3].awrap_model_call

    state = {
        "messages": [HumanMessage(content="find leads")],
        "task_understanding": {"answer_type": "lead_list"},
        "missing_data": [],
    }
    request = SimpleNamespace(state=state)

    async def failing_handler(_request):
        raise FakeOpenAIError("rate limited")

    response = asyncio.run(recover_model_failures(request, failing_handler))
    assert isinstance(response, AIMessage)
    assert state["turn_validation"]["issues"][0]["code"] == "final_model_provider_failed"
    assert "final_model_provider" in state["missing_data"]


def test_recover_model_failures_clears_stale_structured_response(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    recover_model_failures = captured["middleware"][3].wrap_model_call
    finalize_answer = captured["middleware"][2].after_agent

    stale_contract = LeadAnswerContract(
        answer_type="lead_list",
        summary="stale summary",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )
    state = {
        "messages": [HumanMessage(content="find leads")],
        "task_understanding": {"answer_type": "lead_list"},
        "structured_response": stale_contract.model_dump(),
        "missing_data": [],
    }
    request = SimpleNamespace(state=state)

    def failing_handler(_request):
        raise StructuredOutputValidationError(
            "LeadAnswerContract",
            ValueError("extra field"),
            AIMessage(content=""),
        )

    recover_model_failures(request, failing_handler)

    assert state["structured_response"] is None
    update = finalize_answer(state, SimpleNamespace())
    assert update["normalized_final_answer"]["summary"] != "stale summary"
    assert "structured_output_validation_failed" in update["normalized_final_answer"]["missing_data"]


def test_finalize_answer_degrades_valid_structured_response_when_turn_validation_exists(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_list",
        summary="Found 1 relevant lead.",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )
    validation = TurnValidationResult(
        status="failed_verification",
        issues=[
            {
                "stage": "procurement_relevance",
                "code": "procurement_relevance_classifier_failed",
                "message": "relevance failed",
                "metadata": {},
                "severity": "error",
            }
        ],
        manual_review_required=True,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="find leads")],
            "structured_response": contract.model_dump(),
            "turn_validation": validation.model_dump(),
            "missing_data": ["procurement_relevance_verification"],
            "unclassified_procurement_hits": [{"bundle_id": "b1"}],
            "recommended_next_step": "Review the candidates manually.",
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "procurement_relevance_classifier_failed" in update["normalized_final_answer"]["missing_data"]
    assert update["normalized_final_answer"]["recommended_next_step"] == "Review the candidates manually."


def test_prepare_runtime_state_raises_programmer_errors(monkeypatch):
    captured = {}

    class BrokenClassifier:
        def classify_intent(self, **kwargs):
            raise AttributeError("broken wiring")

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=BrokenClassifier(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    with pytest.raises(AttributeError, match="broken wiring"):
        prepare_runtime_state(
            {"messages": [HumanMessage(content="check company")], "missing_data": []},
            SimpleNamespace(),
        )


def test_prepare_runtime_state_reraises_intent_classifier_contract_errors(monkeypatch):
    captured = {}

    class BrokenClassifier:
        def classify_intent(self, **kwargs):
            raise sales_agent.ClassifierContractError("schema-invalid answer_type")

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=BrokenClassifier(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    with pytest.raises(sales_agent.ClassifierContractError, match="schema-invalid answer_type"):
        prepare_runtime_state(
            {"messages": [HumanMessage(content="analyze procurement")], "missing_data": []},
            SimpleNamespace(),
        )


def test_prepare_runtime_state_resets_prior_turn_validation_on_success(monkeypatch):
    captured = {}

    class GoodClassifier:
        def classify_intent(self, **kwargs):
            return SimpleNamespace(
                answer_type="company_check",
                task_kind="company_check",
                requested_company_inns=[],
                comparison_targets=[],
                search_url=None,
                search_filters=None,
                needs_purchase_search=False,
                needs_open_source=False,
                needs_doc_search=False,
                needs_enrichment=False,
                missing_data=[],
                model_dump=lambda: {
                    "answer_type": "company_check",
                    "task_kind": "company_check",
                    "search_url": None,
                    "search_filters": None,
                    "requested_company_inns": [],
                    "comparison_targets": [],
                    "document_questions": [],
                    "needs_purchase_search": False,
                    "needs_open_source": False,
                    "needs_doc_search": False,
                    "needs_enrichment": False,
                    "missing_data": [],
                },
            )

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=GoodClassifier(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    update = prepare_runtime_state(
        {
            "messages": [HumanMessage(content="check company")],
            "missing_data": ["old"],
            "turn_validation": {
                "status": "failed_verification",
                "issues": [
                    {
                        "stage": "old",
                        "code": "old_error",
                        "message": "old error",
                        "metadata": {},
                        "severity": "error",
                    }
                ],
                "manual_review_required": True,
            },
            "task_understanding": {"answer_type": "lead_list"},
            "assessment": {"priority": "high"},
        },
        SimpleNamespace(),
    )

    assert update["turn_validation"]["status"] == "clean"
    assert update["missing_data"] == []
    assert update["task_understanding"]["task_kind"] == "company_check"


def test_prepare_runtime_state_treats_repeated_identical_human_text_as_new_turn(monkeypatch):
    captured = {}

    class GoodClassifier:
        def classify_intent(self, **kwargs):
            return SimpleNamespace(
                answer_type="company_check",
                task_kind="company_check",
                requested_company_inns=[],
                comparison_targets=[],
                search_url=None,
                search_filters=None,
                needs_purchase_search=False,
                needs_open_source=False,
                needs_doc_search=False,
                needs_enrichment=False,
                missing_data=[],
                model_dump=lambda: {
                    "answer_type": "company_check",
                    "task_kind": "company_check",
                    "search_url": None,
                    "search_filters": None,
                    "requested_company_inns": [],
                    "comparison_targets": [],
                    "document_questions": [],
                    "needs_purchase_search": False,
                    "needs_open_source": False,
                    "needs_doc_search": False,
                    "needs_enrichment": False,
                    "missing_data": [],
                },
            )

        def assess_signals(self, **kwargs):
            raise AssertionError("should not be called")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=GoodClassifier(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    messages = [HumanMessage(content="check company"), HumanMessage(content="check company")]
    update = prepare_runtime_state(
        {
            "messages": messages,
            "last_processed_human_message_id": "1:check company",
            "missing_data": ["stale"],
            "turn_tool_usage": [{"tool": "counterparty_scoring_tool", "status": "success"}],
            "turn_validation": {
                "status": "failed_verification",
                "issues": [
                    {
                        "stage": "old",
                        "code": "stale_error",
                        "message": "stale",
                        "metadata": {},
                        "severity": "error",
                    }
                ],
                "manual_review_required": True,
            },
        },
        SimpleNamespace(),
    )

    assert update is not None
    assert update["last_processed_human_message_id"] == "2:check company"
    assert update["turn_validation"]["status"] == "clean"
    assert update["turn_tool_usage"] == []


def test_prepare_runtime_state_records_assessment_failure_and_clears_stale_assessment(monkeypatch):
    captured = {}

    class Classifier:
        def classify_intent(self, **kwargs):
            raise AssertionError("should not be called")

        def assess_signals(self, **kwargs):
            raise sales_agent.ClassifierExecutionError("assessment boom")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=Classifier(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=deps, streaming=False)
    prepare_runtime_state = captured["middleware"][1].before_model

    update = prepare_runtime_state(
        {
            "messages": [],
            "semantic_dirty": True,
            "assessment": {"priority": "high"},
            "risk_verification": {"priority": "high"},
            "missing_data": [],
        },
        SimpleNamespace(),
    )

    assert update["assessment"] is None
    assert update["risk_verification"] is None
    assert update["turn_validation"]["issues"][0]["code"] == "assessment_classifier_failed"
    assert "semantic_assessment" in update["missing_data"]


@pytest.mark.parametrize("task_kind", ["fact_lookup", "procurement_analysis"])
def test_finalize_answer_requires_doc_search_for_document_backed_turns(monkeypatch, task_kind):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_card",
        summary="Document-backed answer.",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="show evidence")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": task_kind,
                "answer_type": "lead_card",
                "doc_search_required": True,
                "scoring_required": False,
                "fssp_required": False,
                "requested_company_inns": [],
                "comparison_targets": [],
            },
            "turn_tool_usage": [],
            "missing_data": [],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "required_doc_search_missing" in update["normalized_final_answer"]["missing_data"]
    assert update["turn_validation"]["issues"][-1]["code"] == "required_doc_search_missing"


def test_finalize_answer_requires_purchase_search_for_procurement_search(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_list",
        summary="Found procurement leads.",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="find procurements")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "procurement_search",
                "answer_type": "lead_list",
                "purchase_search_required": True,
                "doc_search_required": False,
                "scoring_required": False,
                "fssp_required": False,
                "requested_company_inns": [],
                "comparison_targets": [],
            },
            "turn_tool_usage": [],
            "acquisition_attempts": [],
            "missing_data": [],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "required_purchase_search_missing" in update["normalized_final_answer"]["missing_data"]
    assert update["turn_validation"]["issues"][-1]["code"] == "required_purchase_search_missing"


def test_finalize_answer_rebuilds_procurement_degraded_summary_from_state(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_list",
        summary="purchase_search_tool failed because the crawler returned a detailed vendor-specific error.",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="find procurements")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "procurement_search",
                "answer_type": "lead_list",
                "purchase_search_required": True,
                "doc_search_required": False,
                "scoring_required": False,
                "fssp_required": False,
                "requested_company_inns": [],
                "comparison_targets": [],
            },
            "turn_tool_usage": [],
            "acquisition_attempts": [],
            "last_acquisition_error": {
                "tool": "purchase_search_tool",
                "status": "failed",
                "error": "crawler timeout",
            },
            "missing_data": ["procurement_search_source_availability"],
        },
        SimpleNamespace(),
    )

    summary = update["normalized_final_answer"]["summary"]
    assert "crawler timeout" in summary
    assert "vendor-specific error" not in summary


def test_finalize_answer_rebuilds_procurement_partial_with_trusted_hits(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_list",
        summary="Unsupported summary that should be replaced.",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="find procurements")],
            "structured_response": contract.model_dump(),
            "turn_validation": {
                "status": "failed_verification",
                "issues": [
                    {
                        "stage": "purchase_search_tool",
                        "code": "purchase_search_failed",
                        "message": "crawler timeout",
                        "metadata": {},
                        "severity": "error",
                    }
                ],
                "manual_review_required": True,
            },
            "turn_tool_requirements": {
                "task_kind": "procurement_search",
                "answer_type": "lead_list",
                "purchase_search_required": True,
                "doc_search_required": False,
                "scoring_required": False,
                "fssp_required": False,
                "requested_company_inns": [],
                "comparison_targets": [],
            },
            "procurement_hits": [
                {
                    "registry_number": "123",
                    "purchase_title": "Transport insurance services",
                    "customer_name": "Acme",
                    "detail_url": "https://example.test/purchase/123",
                    "price_text": "1000000",
                }
            ],
            "last_acquisition_error": {
                "tool": "purchase_search_tool",
                "status": "failed",
                "error": "crawler timeout",
            },
            "missing_data": ["procurement_search_source_availability"],
        },
        SimpleNamespace(),
    )

    normalized = update["normalized_final_answer"]
    assert len(normalized["items"]) == 1
    assert normalized["items"][0]["event_title"] == "Transport insurance services"
    assert normalized["items"][0]["source_url"] == "https://example.test/purchase/123"
    assert normalized["items"][0]["fact_statuses"][0]["status"] == "document"


def test_finalize_answer_requires_enrichment_attempts_for_company_check(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="company_check",
        summary="Company appears acceptable.",
        items=[
            {
                "company_name": "Test LLC",
                "inn": "7707083893",
                "priority": "medium",
                "reasons": [],
                "evidence": [],
                "fact_statuses": [],
            }
        ],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="check company")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "company_check",
                "answer_type": "company_check",
                "doc_search_required": False,
                "scoring_required": True,
                "fssp_required": True,
                "requested_company_inns": ["7707083893"],
                "comparison_targets": [],
            },
            "turn_tool_usage": [],
            "missing_data": [],
            "normalized_inns": ["7707083893"],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "company_check_scoring_missing" in update["normalized_final_answer"]["missing_data"]
    assert "company_check_fssp_missing" in update["normalized_final_answer"]["missing_data"]


def test_finalize_answer_requires_doc_search_evidence_not_just_zero_match_call(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_card",
        summary="Document-backed answer.",
        items=[],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="show evidence")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "fact_lookup",
                "answer_type": "lead_card",
                "doc_search_required": True,
                "scoring_required": False,
                "fssp_required": False,
                "requested_company_inns": [],
                "comparison_targets": [],
            },
            "turn_tool_usage": [
                {
                    "tool": "doc_search_tool",
                    "status": "success",
                    "index_id": "run-1",
                    "query": "some fact",
                    "match_count": 0,
                }
            ],
            "missing_data": [],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "required_doc_search_evidence_missing" in update["normalized_final_answer"]["missing_data"]


def test_finalize_answer_requires_doc_search_evidence_to_be_rendered(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="lead_card",
        summary="Document-backed answer.",
        items=[
            {
                "company_name": "Test LLC",
                "inn": "7707083893",
                "priority": "medium",
                "reasons": [],
                "evidence": [],
                "fact_statuses": [],
            }
        ],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="show evidence")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "procurement_analysis",
                "answer_type": "lead_card",
                "doc_search_required": True,
                "scoring_required": False,
                "fssp_required": False,
                "requested_company_inns": [],
                "comparison_targets": [],
            },
            "turn_tool_usage": [
                {
                    "tool": "doc_search_tool",
                    "status": "success",
                    "index_id": "run-1",
                    "query": "some fact",
                    "match_count": 2,
                }
            ],
            "missing_data": [],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "required_doc_search_evidence_not_rendered" in update["normalized_final_answer"]["missing_data"]


def test_render_contract_includes_fact_status_labels():
    contract = LeadAnswerContract(
        answer_type="company_check",
        summary="Company review.",
        items=[
            {
                "company_name": "Test LLC",
                "inn": "7707083893",
                "priority": "medium",
                "reasons": [],
                "evidence": [],
                "fact_statuses": [
                    {"fact_key": "financial_risk", "status": "external_api"},
                    {"fact_key": "bidder_experience", "status": "not_found"},
                ],
            }
        ],
        missing_data=[],
        recommended_next_step=None,
    )

    rendered = sales_agent._render_contract(contract)

    assert "Fact status:" in rendered
    assert "financial_risk=external_api" in rendered
    assert "bidder_experience=not_found" in rendered


def test_finalize_answer_accepts_company_check_when_enrichment_attempted(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="company_check",
        summary="Company appears acceptable.",
        items=[
            {
                "company_name": "Test LLC",
                "inn": "7707083893",
                "priority": "medium",
                "reasons": [],
                "evidence": [],
                "fact_statuses": [],
            }
        ],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="check company")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "company_check",
                "answer_type": "company_check",
                "doc_search_required": False,
                "scoring_required": True,
                "fssp_required": True,
                "requested_company_inns": ["7707083893"],
                "comparison_targets": [],
            },
            "turn_tool_usage": [
                {"tool": "counterparty_scoring_tool", "status": "success", "inn": "7707083893"},
                {"tool": "counterparty_fssp_tool", "status": "success", "inn": "7707083893"},
            ],
            "missing_data": [],
            "normalized_inns": ["7707083893"],
        },
        SimpleNamespace(),
    )

    assert update["normalized_final_answer"]["summary"] == "Company appears acceptable."
    assert update["turn_validation"] is None or not update["turn_validation"].get("issues")


def test_finalize_answer_degrades_when_company_check_tool_attempt_failed(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="company_check",
        summary="Company appears acceptable.",
        items=[
            {
                "company_name": "Test LLC",
                "inn": "7707083893",
                "priority": "medium",
                "reasons": [],
                "evidence": [],
                "fact_statuses": [],
            }
        ],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="check company")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "company_check",
                "answer_type": "company_check",
                "doc_search_required": False,
                "scoring_required": True,
                "fssp_required": True,
                "requested_company_inns": ["7707083893"],
                "comparison_targets": [],
            },
            "turn_tool_usage": [
                {"tool": "counterparty_scoring_tool", "status": "success", "inn": "7707083893"},
                {"tool": "counterparty_fssp_tool", "status": "failed", "inn": "7707083893"},
            ],
            "missing_data": [],
            "normalized_inns": ["7707083893"],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "company_check_fssp_failed" in update["normalized_final_answer"]["missing_data"]


def test_finalize_answer_requires_comparison_enrichment_per_target(monkeypatch):
    captured = {}

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return {"graph": "ok", "kwargs": kwargs}

    monkeypatch.setattr(sales_agent, "_import_create_deep_agent", lambda: fake_create_deep_agent)
    monkeypatch.setattr(sales_agent, "get_llm", lambda **kwargs: "fake-llm")

    sales_agent.initialize_agent(dependencies=_deps(), streaming=False)
    finalize_answer = captured["middleware"][2].after_agent

    contract = LeadAnswerContract(
        answer_type="comparison",
        summary="Company A looks stronger than Company B.",
        items=[
            {
                "company_name": "Company A",
                "inn": "7707083893",
                "priority": "high",
                "reasons": [],
                "evidence": [],
                "fact_statuses": [],
            }
        ],
        missing_data=[],
        recommended_next_step=None,
    )

    update = finalize_answer(
        {
            "messages": [HumanMessage(content="compare companies")],
            "structured_response": contract.model_dump(),
            "turn_tool_requirements": {
                "task_kind": "comparison",
                "answer_type": "comparison",
                "doc_search_required": False,
                "scoring_required": True,
                "fssp_required": True,
                "requested_company_inns": [],
                "comparison_targets": ["Company A", "Company B"],
            },
            "turn_tool_usage": [
                {"tool": "counterparty_scoring_tool", "status": "success", "inn": "7707083893"},
                {"tool": "counterparty_fssp_tool", "status": "success", "inn": "7707083893"},
            ],
            "missing_data": [],
            "normalized_inns": ["7707083893"],
        },
        SimpleNamespace(),
    )

    assert "partial" in update["normalized_final_answer"]["summary"].lower()
    assert "comparison_target_inn_missing" in update["normalized_final_answer"]["missing_data"]
