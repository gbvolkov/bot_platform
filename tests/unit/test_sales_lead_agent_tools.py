import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.sales_lead_agent import tools as sales_tools
from agents.sales_lead_agent.services import ClassifierExecutionError
from agents.sales_lead_agent.schemas import (
    DocSearchResponse,
    PreparedDocument,
    PreparedDocumentEntities,
    PurchaseSearchItem,
    PurchaseSearchResponse,
)
from agents.sales_lead_agent.tools import _resolve_purchase_run_id, _search_filters_from_args, build_sales_lead_tools


def test_search_filters_are_reconstructed_from_flat_tool_args():
    filters = _search_filters_from_args(
        query_text="insurance services",
        law="44-FZ",
        customer_name="Acme",
        supplier_hint="CASCO",
    )

    assert filters is not None
    assert filters.query_text == "insurance services"
    assert filters.law == "44-FZ"
    assert filters.customer_name == "Acme"
    assert filters.supplier_hint == "CASCO"


def test_purchase_run_reuses_active_run_only_for_same_query_signature():
    state = {
        "active_run_id": "run-1",
        "current_query_signature": "sig-a",
        "active_run_query_signature": "sig-a",
    }

    assert _resolve_purchase_run_id(state, None) == "run-1"
    assert _resolve_purchase_run_id({**state, "current_query_signature": "sig-b"}, None) is None
    assert _resolve_purchase_run_id(state, "run-2") == "run-2"


def test_purchase_run_reuses_active_run_for_followup_document_turns():
    state = {
        "active_run_id": "run-1",
        "active_run_ready": True,
        "current_query_signature": "sig-b",
        "active_run_query_signature": "sig-a",
        "task_understanding": {"task_kind": "fact_lookup"},
        "prepared_documents": [{"document_id": "doc-1", "index_status": "ready", "chunks_count": 2}],
    }

    assert _resolve_purchase_run_id(state, None) == "run-1"


def test_purchase_run_does_not_reuse_active_run_for_new_followup_acquisition_context():
    state = {
        "active_run_id": "run-1",
        "active_run_ready": True,
        "current_query_signature": "sig-b",
        "active_run_query_signature": "sig-a",
        "task_understanding": {"task_kind": "procurement_analysis"},
        "prepared_documents": [{"document_id": "doc-1", "index_status": "ready", "chunks_count": 2}],
    }

    assert (
        _resolve_purchase_run_id(
            state,
            None,
            current_search_url="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=new",
        )
        is None
    )


def test_purchase_run_does_not_reuse_failed_or_unready_active_context():
    state = {
        "active_run_id": "run-1",
        "active_run_ready": False,
        "current_query_signature": "sig-b",
        "active_run_query_signature": "sig-a",
        "task_understanding": {"task_kind": "fact_lookup"},
        "prepared_documents": [{"document_id": "doc-1", "index_status": "failed", "chunks_count": 0}],
    }

    assert _resolve_purchase_run_id(state, None) is None


def test_purchase_run_does_not_reuse_zero_chunk_non_searchable_context_even_if_marked_ready():
    state = {
        "active_run_id": "run-1",
        "active_run_ready": True,
        "current_query_signature": "sig-b",
        "active_run_query_signature": "sig-a",
        "task_understanding": {"task_kind": "fact_lookup"},
        "prepared_documents": [
            {
                "document_id": "doc-1",
                "parse_status": "partial",
                "index_status": "failed",
                "chunks_count": 0,
                "error": "No indexable content extracted.",
            }
        ],
    }

    assert _resolve_purchase_run_id(state, None) is None


def test_purchase_search_tool_marks_hits_unclassified_when_relevance_classifier_fails(tmp_path: Path):
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="run-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_response = PurchaseSearchResponse(
        run_id="",
        index_id="",
        status="success",
        errors=[],
        items=[
            PurchaseSearchItem(
                bundle_id="b1",
                registry_number="123",
                law="44-FZ",
                purchase_title="Insurance services",
                customer_name="Acme",
                price_text="100",
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                prepared_document_ids=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="success",
                crawl_error=None,
                crawl_ts_utc="2026-03-22T00:00:00Z",
            )
        ],
        prepared_documents=[],
    )

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(prepare_files=lambda **kwargs: []),
        classifier=SimpleNamespace(
            classify_procurement_hits=lambda **kwargs: (_ for _ in ()).throw(
                ClassifierExecutionError("relevance boom")
            )
        ),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: ("https://example.test/search", base_response),
            summarize_hits=lambda response: [{"bundle_id": "b1", "registry_number": "123", "purchase_title": "Insurance services"}],
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[0]
    command = tool.func(
        runtime=SimpleNamespace(
            state={
                "messages": [],
                "missing_data": [],
                "current_user_request": "find insurance procurements",
                "task_understanding": {"task_kind": "procurement_search"},
            }
        ),
        query_text="insurance",
    )

    update = command.update
    assert update["procurement_hits"] == []
    assert len(update["unclassified_procurement_hits"]) == 1
    assert update["turn_validation"]["issues"][0]["code"] == "procurement_relevance_classifier_failed"
    assert "procurement_relevance_verification" in update["missing_data"]


def test_purchase_search_tool_returns_validation_error_when_criteria_missing():
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: None, get=lambda run_id: None),
        document_service=SimpleNamespace(),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[0]
    command = tool.func(runtime=SimpleNamespace(state={"messages": [], "missing_data": []}))

    assert command.update["turn_validation"]["issues"][0]["code"] == "purchase_search_missing_criteria"
    assert "procurement_search_criteria" in command.update["missing_data"]
    assert command.update["turn_tool_usage"][0]["tool"] == "purchase_search_tool"


def test_purchase_search_tool_returns_validation_error_when_filters_invalid():
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: None, get=lambda run_id: None),
        document_service=SimpleNamespace(),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[0]
    command = tool.func(
        runtime=SimpleNamespace(state={"messages": [], "missing_data": []}),
        law="invalid-law",
    )

    assert command.update["turn_validation"]["issues"][0]["code"] == "purchase_search_invalid_filters"
    assert "procurement_search_filters" in command.update["missing_data"]
    assert command.update["turn_tool_usage"][0]["validation_error"] == "purchase_search_invalid_filters"


def test_purchase_search_tool_raises_programmer_errors_from_classifier(tmp_path: Path):
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="run-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_response = PurchaseSearchResponse(
        run_id="",
        index_id="",
        status="success",
        errors=[],
        items=[
            PurchaseSearchItem(
                bundle_id="b1",
                registry_number="123",
                law="44-FZ",
                purchase_title="Insurance services",
                customer_name="Acme",
                price_text="100",
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                prepared_document_ids=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="success",
                crawl_error=None,
                crawl_ts_utc="2026-03-22T00:00:00Z",
            )
        ],
        prepared_documents=[],
    )

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(prepare_files=lambda **kwargs: []),
        classifier=SimpleNamespace(
            classify_procurement_hits=lambda **kwargs: (_ for _ in ()).throw(AttributeError("bad classifier wiring"))
        ),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: ("https://example.test/search", base_response),
            summarize_hits=lambda response: [{"bundle_id": "b1", "registry_number": "123", "purchase_title": "Insurance services"}],
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[0]
    with pytest.raises(AttributeError, match="bad classifier wiring"):
        tool.func(
            runtime=SimpleNamespace(
                state={
                    "messages": [],
                    "missing_data": [],
                    "current_user_request": "find insurance procurements",
                    "task_understanding": {"task_kind": "procurement_search"},
                }
            ),
            query_text="insurance",
        )


def test_purchase_search_failed_attempt_records_acquisition_failure_and_does_not_promote_active_run(
    tmp_path: Path,
):
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="index-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_response = PurchaseSearchResponse(
        run_id="",
        index_id="",
        status="failed",
        errors=["crawler down"],
        items=[],
        prepared_documents=[],
    )

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(prepare_files=lambda **kwargs: []),
        classifier=SimpleNamespace(classify_procurement_hits=lambda **kwargs: None),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: ("https://example.test/search", base_response),
            summarize_hits=lambda response: [],
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[0]
    command = tool.func(
        runtime=SimpleNamespace(state={"messages": [], "missing_data": [], "turn_tool_usage": []}),
        query_text="insurance",
    )

    update = command.update
    assert "active_run_id" not in update
    assert update["acquisition_status"] == "failed"
    assert update["acquisition_attempts"][0]["tool"] == "purchase_search_tool"
    assert update["acquisition_attempts"][0]["active_run_ready"] is False
    assert update["last_acquisition_error"]["error"] == "crawler down"
    assert update["turn_validation"]["issues"][0]["code"] == "purchase_search_failed"


def test_failed_purchase_search_with_searchable_artifacts_still_does_not_promote_active_run(
    tmp_path: Path,
):
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="index-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)

    base_response = PurchaseSearchResponse(
        run_id="",
        index_id="",
        status="failed",
        errors=["crawler down"],
        items=[
            PurchaseSearchItem(
                bundle_id="b1",
                registry_number="123",
                law="44-FZ",
                purchase_title="Insurance services",
                customer_name="Acme",
                price_text="100",
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                prepared_document_ids=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="failed",
                crawl_error="crawler down",
                crawl_ts_utc="2026-03-22T00:00:00Z",
            )
        ],
        prepared_documents=[],
    )

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(
            prepare_files=lambda **kwargs: [
                PreparedDocument(
                    document_id="doc-1",
                    origin="purchase",
                    bundle_id="b1",
                    registry_number="123",
                    source_url="https://example.test/purchase/123",
                    file_path="C:/tmp/doc.txt",
                    file_name="doc.txt",
                    file_type="other",
                    parse_status="success",
                    index_status="ready",
                    text_excerpt="insurance",
                    entities=PreparedDocumentEntities(),
                    chunks_count=4,
                    error=None,
                )
            ]
        ),
        classifier=SimpleNamespace(
            classify_procurement_hits=lambda **kwargs: SimpleNamespace(
                decisions=[],
                model_dump=lambda: {"decisions": []},
            )
        ),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: ("https://example.test/search", base_response),
            summarize_hits=lambda response: [],
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[0]
    command = tool.func(
        runtime=SimpleNamespace(state={"messages": [], "missing_data": [], "turn_tool_usage": []}),
        query_text="insurance",
    )

    update = command.update
    assert "active_run_id" not in update
    assert update["acquisition_attempts"][0]["active_run_ready"] is False
    assert update["turn_tool_usage"][0]["active_run_ready"] is False


def test_open_source_fetch_failed_attempt_records_acquisition_failure_and_does_not_promote_active_run(
    monkeypatch,
    tmp_path: Path,
):
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="index-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
        web_dir=tmp_path / "web",
        index_dir=tmp_path / "index",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)
    workspace.web_dir.mkdir(parents=True, exist_ok=True)
    workspace.index_dir.mkdir(parents=True, exist_ok=True)

    class FailingLoader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            raise RuntimeError("website down")

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", FailingLoader)

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(save_text_artifact=lambda **kwargs: "", prepare_files=lambda **kwargs: []),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[1]
    command = asyncio.run(
        tool.coroutine(
            runtime=SimpleNamespace(state={"messages": [], "missing_data": [], "turn_tool_usage": []}),
            url="https://example.test",
        )
    )

    update = command.update
    assert "active_run_id" not in update
    assert update["acquisition_status"] == "failed"
    assert update["acquisition_attempts"][0]["tool"] == "open_source_fetch_tool"
    assert update["acquisition_attempts"][0]["active_run_ready"] is False
    assert update["last_acquisition_error"]["error"] == "website down"
    assert update["turn_validation"]["issues"][0]["code"] == "open_source_fetch_failed"


def test_open_source_fetch_success_prepares_pages_and_attachments_and_promotes_active_run(
    monkeypatch,
    tmp_path: Path,
):
    workspace = SimpleNamespace(
        run_id="run-1",
        index_id="index-1",
        downloads_dir=tmp_path / "downloads",
        artifacts_dir=tmp_path / "artifacts",
        web_dir=tmp_path / "web",
        index_dir=tmp_path / "index",
    )
    workspace.downloads_dir.mkdir(parents=True, exist_ok=True)
    workspace.artifacts_dir.mkdir(parents=True, exist_ok=True)
    workspace.web_dir.mkdir(parents=True, exist_ok=True)
    workspace.index_dir.mkdir(parents=True, exist_ok=True)

    class SuccessfulLoader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="Main page text about insurance procurement.",
                    metadata={
                        "source": "https://example.test/page",
                        "title": "Insurance page",
                        "content_type": "text/html",
                    },
                ),
                SimpleNamespace(
                    page_content="Attachment text with INN 7707083893 and Acme LLC.",
                    metadata={
                        "source_type": "web_download",
                        "source": "https://example.test/files/spec.pdf",
                        "parent_url": "https://example.test/page",
                        "download_filename": "spec.pdf",
                        "content_type": "application/pdf",
                    },
                ),
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", SuccessfulLoader)

    saved_artifacts: list[tuple[str, str, str]] = []

    def save_text_artifact(*, workspace, relative_dir, file_name, content):
        target_dir = workspace.artifacts_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        target_path.write_text(content, encoding="utf-8")
        saved_artifacts.append((relative_dir, file_name, content))
        return str(target_path)

    def prepare_files(*, workspace, origin, bundle_id, registry_number, source_url, file_paths, provenance_by_path):
        prepared: list[PreparedDocument] = []
        for file_path in file_paths:
            provenance = provenance_by_path[file_path]
            prepared.append(
                PreparedDocument(
                    document_id=f"doc-{Path(file_path).stem}",
                    origin="open_source",
                    bundle_id=bundle_id,
                    registry_number=None,
                    source_url=source_url,
                    original_source_url=provenance.get("original_source_url"),
                    original_file_name=provenance.get("original_file_name"),
                    original_content_type=provenance.get("original_content_type"),
                    derived_artifact_path=provenance.get("derived_artifact_path"),
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    file_type="other",
                    parse_status="success",
                    index_status="ready",
                    text_excerpt="prepared text",
                    entities=PreparedDocumentEntities(
                        inn=["7707083893"],
                        company_names=["Acme LLC"],
                    ),
                    chunks_count=2,
                    error=None,
                )
            )
        return prepared

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(
            save_text_artifact=save_text_artifact,
            prepare_files=prepare_files,
        ),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    tool = build_sales_lead_tools(deps)[1]
    command = asyncio.run(
        tool.coroutine(
            runtime=SimpleNamespace(state={"messages": [], "missing_data": [], "turn_tool_usage": []}),
            url="https://example.test/page",
            follow_download_links=True,
        )
    )

    update = command.update
    assert update["active_run_id"] == "run-1"
    assert update["index_id"] == "index-1"
    assert update["active_run_ready"] is True
    assert update["acquisition_status"] == "success"
    assert update["acquisition_attempts"][0]["tool"] == "open_source_fetch_tool"
    assert update["acquisition_attempts"][0]["searchable_document_count"] == 2
    assert update["last_open_source_fetch_result"]["pages"][0]["attachments"] == [
        "https://example.test/files/spec.pdf"
    ]
    assert len(update["prepared_documents"]) == 2
    assert update["prepared_documents"][0]["original_source_url"] == "https://example.test/page"
    assert update["prepared_documents"][1]["original_file_name"] == "spec.pdf"
    assert update["prepared_documents"][1]["original_content_type"] == "application/pdf"
    assert update["normalized_inns"] == ["7707083893"]
    assert update["company_names"] == ["Acme LLC"]
    assert len(saved_artifacts) == 2


def test_counterparty_scoring_guard_failure_records_failed_attempt():
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(),
        document_service=SimpleNamespace(),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    scoring_tool = build_sales_lead_tools(deps)[3]
    command = scoring_tool.func(
        runtime=SimpleNamespace(state={"messages": [], "missing_data": [], "normalized_inns": []}),
        inn="7707083893",
    )

    assert command.update["turn_validation"]["issues"][0]["code"] == "counterparty_scoring_tool_guard_failed"
    assert command.update["turn_tool_usage"][0]["tool"] == "counterparty_scoring_tool"
    assert command.update["turn_tool_usage"][0]["status"] == "failed"


def test_doc_search_tool_resolves_explicit_index_id_via_index_resolver():
    workspace = SimpleNamespace(run_id="run-1", index_id="index-1")
    calls = []

    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(
            get=lambda run_id: (_ for _ in ()).throw(AssertionError("run resolver should not be used")),
            get_by_index=lambda index_id: calls.append(index_id) or workspace,
        ),
        document_service=SimpleNamespace(
            search=lambda **kwargs: DocSearchResponse(index_id="index-1", matches=[])
        ),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = build_sales_lead_tools(deps)[2]
    command = doc_search_tool.func(
        runtime=SimpleNamespace(state={"messages": [], "missing_data": []}),
        index_id="index-1",
        query="find snippet",
    )

    assert calls == ["index-1"]
    assert command.update["last_doc_search_result"]["index_id"] == "index-1"


def test_doc_search_tool_degrades_when_explicit_index_id_is_unknown():
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(
            get=lambda run_id: (_ for _ in ()).throw(AssertionError("run resolver should not be used")),
            get_by_index=lambda index_id: (_ for _ in ()).throw(ValueError("Index id index-missing is not registered.")),
        ),
        document_service=SimpleNamespace(),
        classifier=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = build_sales_lead_tools(deps)[2]
    command = doc_search_tool.func(
        runtime=SimpleNamespace(state={"messages": [], "missing_data": []}),
        index_id="index-missing",
        query="find snippet",
    )

    assert command.update["turn_validation"]["issues"][0]["code"] == "doc_search_index_resolution_failed"
    assert command.update["turn_tool_usage"][0]["guard"] == "index_resolution_failed"
    assert "index_id:index-missing" in command.update["missing_data"]
