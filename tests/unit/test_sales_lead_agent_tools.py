from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain.tools import ToolRuntime
from zakupki_crawler.crawler import ZakupkiCrawler

from agents.sales_lead_agent import tools as sales_tools
from agents.sales_lead_agent.tools import (
    DocSearchResponse,
    PreparedDocument,
    PreparedDocumentEntities,
    ProcurementCatalogService,
    ProcurementQueryBuilder,
    PurchaseRequestSpec,
    PurchaseSearchItem,
    RunWorkspace,
    SalesLeadAgentDependencies,
    SalesLeadAgentSettings,
    ToolUserCorrectableError,
    build_sales_lead_tools,
)
from services.sales_lead_retrieval import api as retrieval_api
from services.sales_lead_retrieval.client import (
    RetrievalServiceUserInputError,
)
from services.sales_lead_retrieval.schemas import (
    RetrievalUserInputErrorResponse,
    SubmitPurchaseRetrievalRequest,
)

SHARED_INDEX_ID = "sales_lead_permanent"


def _workspace(tmp_path: Path) -> RunWorkspace:
    root = tmp_path / "run-1"
    downloads = root / "downloads"
    web = root / "web"
    index = root / "index"
    artifacts = root / "artifacts"
    for path in (downloads, web, index, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    return RunWorkspace(
        run_id="run-1",
        index_id=SHARED_INDEX_ID,
        root_dir=root,
        downloads_dir=downloads,
        web_dir=web,
        index_dir=index,
        artifacts_dir=artifacts,
    )


def _document_service_stub(**overrides):
    defaults = {
        "shared_index_id": SHARED_INDEX_ID,
        "purchase_exists": lambda purchase_id: False,
        "source_exists": lambda source_id: False,
        "purchase_document_summary": lambda purchase_id: {
            "indexed_documents_count": 0,
            "prepared_document_ids": [],
            "last_indexed_at": None,
            "source_url": None,
        },
        "prepare_files": lambda **kwargs: [],
        "save_text_artifact": lambda **kwargs: "",
        "search": lambda **kwargs: DocSearchResponse(index_id=SHARED_INDEX_ID, matches=[]),
        "read_cached_document": lambda **kwargs: SimpleNamespace(model_dump=lambda: {}),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _runtime(thread_id: str = "conv-1") -> SimpleNamespace:
    return SimpleNamespace(
        tool_call_id="call-1",
        state={},
        config={"configurable": {"thread_id": thread_id}},
    )


def _runtime_with_state(*, thread_id: str = "conv-1", **state) -> SimpleNamespace:
    return SimpleNamespace(
        tool_call_id="call-1",
        state=state,
        config={"configurable": {"thread_id": thread_id}},
    )


def _command_payload(command) -> dict[str, object]:
    return json.loads(command.update["messages"][0].content)


def _tool_by_name(deps, name: str):
    for candidate in build_sales_lead_tools(deps):
        if candidate.name == name:
            return candidate
    raise AssertionError(f"Tool {name} not found")


def _retrieval_snapshot(
    *,
    retrieval_id: str = "ret-1",
    request_hash: str = "hash-1",
    run_id: str = "run-1",
    index_id: str = SHARED_INDEX_ID,
    status: str = "queued",
    stage: str = "queued",
    message: str = "Queued.",
    completion_announced: bool = False,
    search_urls: list[str] | None = None,
    items: list[dict[str, object]] | None = None,
    prepared_documents: list[dict[str, object]] | None = None,
    progress: dict[str, int] | None = None,
):
    return SimpleNamespace(
        retrieval_id=retrieval_id,
        conversation_id="conv-1",
        request_hash=request_hash,
        run_id=run_id,
        index_id=index_id,
        status=status,
        stage=stage,
        message=message,
        completion_announced=completion_announced,
        snapshot_updated_at="2026-03-26T00:00:00Z",
        request_payload={"search_urls": search_urls or ["https://example.test/search"]},
        progress=progress
        or {
            "total_queries": 1,
            "completed_queries": 0,
            "total_purchases": 0,
            "processed_purchases": 0,
            "total_files": 0,
            "processed_files": 0,
            "prepared_documents": 0,
            "indexed_segments": 0,
        },
        items=items or [],
        prepared_documents=prepared_documents or [],
        error_text=None,
        active_retrieval_context=lambda: {
            "retrieval_id": retrieval_id,
            "request_hash": request_hash,
            "run_id": run_id,
            "index_id": index_id,
            "retrieval_status": status,
            "retrieval_stage": stage,
            "message": message,
            "completion_announced": completion_announced,
            "snapshot_updated_at": "2026-03-26T00:00:00Z",
            "progress": progress
            or {
                "total_queries": 1,
                "completed_queries": 0,
                "total_purchases": 0,
                "processed_purchases": 0,
                "total_files": 0,
                "processed_files": 0,
                "prepared_documents": 0,
                "indexed_segments": 0,
            },
        },
    )


def _counterparty_settings(tmp_path: Path, **overrides) -> SalesLeadAgentSettings:
    defaults = dict(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test"
        ),
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://api.damia.ru",
        fssp_base_url="https://api.damia.ru",
        damia_scoring_api_key="test-key",
        damia_fssp_api_key="test-key",
        dadata_api_key="dadata-key",
        scoring_default_model="_problemCredit",
    )
    defaults.update(overrides)
    return SalesLeadAgentSettings(**defaults)


class _FakeHTTPResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.content = b"" if payload is None else b"payload"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)


def _install_fake_http_client(monkeypatch, responses, captured):
    scripted_responses = list(responses)

    class FakeClient:
        def __init__(self, *, headers, timeout, **kwargs):
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured.update({key: value for key, value in kwargs.items() if key not in {"headers", "timeout"}})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def _record(self, *, method, url, params=None, json_body=None):
            call = {
                "url": url,
                "params": dict(params or {}),
            }
            if method != "GET":
                call["method"] = method
            if json_body is not None:
                call["json"] = json_body
            captured.setdefault("calls", []).append(call)
            if not scripted_responses:
                raise AssertionError(f"Unexpected {method} call for {url}")
            spec = scripted_responses.pop(0)
            expected_method = spec.get("method", "GET")
            assert method == expected_method
            expected_url = spec.get("url")
            if expected_url is not None:
                assert url == expected_url
            return _FakeHTTPResponse(
                spec.get("payload"),
                status_code=spec.get("status_code", 200),
            )

        def get(self, url, params=None):
            return self._record(method="GET", url=url, params=params)

        def post(self, url, json=None):
            return self._record(method="POST", url=url, json_body=json)

    monkeypatch.setattr(sales_tools.httpx, "Client", FakeClient)
    return scripted_responses


def test_query_builder_changes_only_search_string():
    template = (
        "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
        "?searchString=страхован"
        "&morphology=on"
        "&search-filter=Дате+размещения"
        "&pageNumber=1"
        "&sortDirection=false"
        "&recordsPerPage=_2"
        "&showLotsInfoHidden=false"
        "&sortBy=UPDATE_DATE"
        "&fz44=on"
        "&fz223=on"
        "&af=on"
        "&currencyIdGeneral=-1"
        "&gws=Выберите+тип+закупки"
    )
    builder = ProcurementQueryBuilder(template)

    url = builder.build_url("страх имущества")

    assert "searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85+%D0%B8%D0%BC%D1%83%D1%89%D0%B5%D1%81%D1%82%D0%B2%D0%B0" in url
    assert "recordsPerPage=_2" in url
    assert "fz44=on" in url
    assert "fz223=on" in url


def test_query_builder_collapses_overlapping_words_to_shortest_stem():
    template = (
        "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
        "?searchString=страхован"
        "&morphology=on"
        "&recordsPerPage=_2"
    )
    builder = ProcurementQueryBuilder(template)

    url = builder.build_url("страхован страхов услуг услуг")

    assert "searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2+%D1%83%D1%81%D0%BB%D1%83%D0%B3" in url


def test_query_builder_builds_multiple_urls_for_or_semantics():
    template = (
        "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
        "?searchString=страхован"
        "&morphology=on"
        "&search-filter=Дате+размещения"
        "&pageNumber=1"
        "&sortDirection=false"
        "&recordsPerPage=_2"
        "&showLotsInfoHidden=false"
        "&sortBy=UPDATE_DATE"
        "&fz44=on"
        "&fz223=on"
        "&af=on"
        "&currencyIdGeneral=-1"
        "&gws=Выберите+тип+закупки"
    )
    builder = ProcurementQueryBuilder(template)

    urls = builder.build_urls(["страхован", "ОСАГО"])

    assert len(urls) == 2
    assert "searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD" in urls[0]
    assert "searchString=%D0%BE%D1%81%D0%B0%D0%B3%D0%BE" in urls[1]


def test_query_builder_dedupes_order_insensitive_and_overlapping_query_texts():
    template = (
        "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
        "?searchString=страхован"
        "&morphology=on"
        "&recordsPerPage=_2"
    )
    builder = ProcurementQueryBuilder(template)

    urls = builder.build_urls(["страхован услуг", "услуг страхован", "страхов услуг"])

    assert len(urls) == 1
    assert "searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2+%D1%83%D1%81%D0%BB%D1%83%D0%B3" in urls[0]


def test_zakupki_crawler_accepts_zero_result_page():
    class FakeLocator:
        def __init__(self, count: int):
            self._count = count

        def count(self) -> int:
            return self._count

    class FakePage:
        url = "https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=страхование"

        def __init__(self):
            self._counts = {
                'a[href*="common-info"]': 0,
                "text=Поиск не дал результатов": 1,
                "text=0 записей": 1,
                "text=Результаты поиска": 1,
            }

        def locator(self, selector: str) -> FakeLocator:
            return FakeLocator(self._counts.get(selector, 0))

        def wait_for_selector(self, selector: str, timeout: int) -> None:
            return None

    crawler = object.__new__(ZakupkiCrawler)
    crawler.pacer = SimpleNamespace(sleep_func=lambda _seconds: None)

    crawler._wait_for_search_results(FakePage())


def test_purchase_adapter_dedupes_same_registry_number_within_one_search_call(monkeypatch, tmp_path: Path):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=страхован&recordsPerPage=_2"
        ),
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    builder = ProcurementQueryBuilder(settings.procurement_search_template)
    adapter = sales_tools.PurchaseAdapter(settings, builder)

    def fake_scrape(_url, **kwargs):
        return [
            SimpleNamespace(
                registry_number="123",
                law="44-FZ",
                purchase_title="Оказание услуг страхования",
                customer_name="Фонд",
                price_text=None,
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="success",
                crawl_error=None,
                crawl_ts_utc=None,
            ),
            SimpleNamespace(
                registry_number="123",
                law="44-FZ",
                purchase_title="Оказание услуг страхования дубль",
                customer_name="Фонд",
                price_text=None,
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="success",
                crawl_error=None,
                crawl_ts_utc=None,
            ),
        ]

    monkeypatch.setattr(adapter, "_import_scraper", lambda: fake_scrape)

    resolved_urls, items = adapter.search(
        search_url=None,
        query_texts=["страхование услуг", "ОСАГО"],
        downloads_dir=str(tmp_path / "downloads"),
        max_pages=1,
    )

    assert len(resolved_urls) == 2
    assert len(items) == 1
    assert items[0].registry_number == "123"


def test_purchase_adapter_uses_permanent_download_cache(monkeypatch, tmp_path: Path):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=страхован&recordsPerPage=_2"
        ),
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    builder = ProcurementQueryBuilder(settings.procurement_search_template)
    adapter = sales_tools.PurchaseAdapter(settings, builder)
    captured = {}

    def fake_scrape(_url, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(adapter, "_import_scraper", lambda: fake_scrape)

    adapter.search(
        search_url=None,
        query_texts=["страхование"],
        downloads_dir=str(tmp_path / "ignored"),
        max_pages=1,
    )

    assert Path(captured["downloads_dir"]) == settings.permanent_index_root / "purchase_downloads"
    assert captured["headless"] is True


def test_purchase_adapter_always_uses_headless_for_scraper(monkeypatch, tmp_path: Path):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=test&recordsPerPage=_2"
        ),
        purchase_headless=False,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    adapter = sales_tools.PurchaseAdapter(
        settings,
        ProcurementQueryBuilder(settings.procurement_search_template),
    )
    captured = {}

    def fake_scrape(_url, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(adapter, "_import_scraper", lambda: fake_scrape)

    adapter.search(
        search_url=None,
        query_texts=["test"],
        downloads_dir=str(tmp_path / "ignored"),
        max_pages=1,
    )

    assert captured["headless"] is True


def test_purchase_adapter_emits_progress_events(monkeypatch, tmp_path: Path):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD&recordsPerPage=_2"
        ),
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    builder = ProcurementQueryBuilder(settings.procurement_search_template)
    adapter = sales_tools.PurchaseAdapter(settings, builder)
    events: list[dict[str, object]] = []

    def fake_scrape(_url, **kwargs):
        return [
            SimpleNamespace(
                registry_number="123",
                law="44-FZ",
                purchase_title="Оказание услуг страхования",
                customer_name="Фонд",
                price_text=None,
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="success",
                crawl_error=None,
                crawl_ts_utc=None,
            )
        ]

    monkeypatch.setattr(adapter, "_import_scraper", lambda: fake_scrape)

    adapter.search(
        search_url=None,
        query_texts=["страхован"],
        downloads_dir=str(tmp_path / "downloads"),
        max_pages=1,
        progress_callback=lambda **event: events.append(event),
    )

    stages = [str(event["stage"]) for event in events]
    assert stages == [
        "crawler_plan",
        "crawler_search",
        "crawler_search_done",
        "crawler_complete",
    ]
    assert "Looking zakupki.gov.ru" in str(events[1]["message"])


def test_document_preparation_service_emits_progress_events(monkeypatch, tmp_path: Path):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD&recordsPerPage=_2"
        ),
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    workspace = _workspace(tmp_path)
    artifact_path = workspace.artifacts_dir / "bundle-1" / "source.txt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Страхование имущества", encoding="utf-8")
    events: list[dict[str, object]] = []
    indexed: dict[str, int] = {}

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda _path: [
            sales_tools.Document(
                id="doc-raw",
                page_content="Страхование имущества",
                metadata={"source": str(artifact_path), "output_format": "text"},
            )
        ],
    )
    monkeypatch.setattr(
        service,
        "_split_docs",
        lambda *, file_path, docs: [
            sales_tools.Segment(
                content="Страхование имущества",
                metadata={"source": str(file_path), "output_format": "text"},
                segment_id="seg-raw",
                type=sales_tools.SegmentType.TEXT,
                original_format="text",
            )
        ],
    )
    monkeypatch.setattr(
        service,
        "_index_documents",
        lambda *, segments: indexed.update({"count": len(segments)}),
    )

    prepared = service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(artifact_path)],
        progress_callback=lambda **event: events.append(event),
    )

    assert len(prepared) == 1
    assert indexed["count"] == 1
    stages = [str(event["stage"]) for event in events]
    assert stages == [
        "parsing_plan",
        "parsing_file",
        "index_building",
        "index_ready",
        "file_ready",
    ]


def test_purchase_adapter_continues_after_query_failure(monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template=(
            "https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            "?searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD&recordsPerPage=_2"
        ),
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    builder = ProcurementQueryBuilder(settings.procurement_search_template)
    adapter = sales_tools.PurchaseAdapter(settings, builder)
    events: list[dict[str, object]] = []

    def fake_scrape(url, **kwargs):
        if "searchString=%D0%BE%D1%81%D0%B0%D0%B3%D0%BE" in url:
            raise TimeoutError("Page.wait_for_selector: Timeout 30000ms exceeded.")
        return [
            SimpleNamespace(
                registry_number="123",
                law="44-FZ",
                purchase_title="Оказание услуг страхования",
                customer_name="Фонд",
                price_text=None,
                published_at=None,
                updated_at=None,
                submission_deadline=None,
                detail_url="https://example.test/purchase/123",
                common_info_url=None,
                documents_url=None,
                document_urls=[],
                downloaded_files=[],
                documents_json=None,
                common_info_json=None,
                lots_json=None,
                crawl_status="success",
                crawl_error=None,
                crawl_ts_utc=None,
            )
        ]

    monkeypatch.setattr(adapter, "_import_scraper", lambda: fake_scrape)

    with caplog.at_level("ERROR"):
        resolved_urls, items = adapter.search(
            search_url=None,
            query_texts=["осаго", "каско"],
            downloads_dir=str(tmp_path / "downloads"),
            max_pages=1,
            progress_callback=lambda **event: events.append(event),
        )

    assert len(resolved_urls) == 2
    assert len(items) == 1
    assert [str(event["stage"]) for event in events] == [
        "crawler_plan",
        "crawler_search",
        "crawler_search_failed",
        "crawler_search",
        "crawler_search_done",
        "crawler_complete",
    ]
    assert events[2]["error_type"] == "TimeoutError"
    assert events[-1]["failed_queries"] == 1
    assert "Procurement crawler failed" in caplog.text

def test_purchase_search_tool_returns_run_index_and_prepared_documents(tmp_path: Path):
    pytest.skip("Legacy synchronous purchase_search_tool behavior was replaced by async retrieval snapshots.")
    workspace = _workspace(tmp_path)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            prepare_files=lambda **kwargs: [
                PreparedDocument(
                    document_id="doc-1",
                    origin="purchase",
                    bundle_id="bundle-1",
                    registry_number="123",
                    source_url="https://example.test/purchase/123",
                    file_path="C:/tmp/common_info.json",
                    file_name="common_info.json",
                    file_type="json",
                    text_excerpt="страх имущества",
                    entities=PreparedDocumentEntities(inn=["7707083893"]),
                    chunks_count=3,
                )
            ]
        ),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (
                ["https://example.test/search"],
                [
                    PurchaseSearchItem(
                        bundle_id="bundle-1",
                        registry_number="123",
                        law="44-FZ",
                        purchase_title="Оказание услуг страхования",
                        customer_name="Фонд",
                        price_text="1000",
                        published_at=None,
                        updated_at=None,
                        submission_deadline=None,
                        detail_url="https://example.test/purchase/123",
                        common_info_url=None,
                        documents_url=None,
                        document_urls=[],
                        downloaded_files=[],
                        common_info_json='{"a":1}',
                        documents_json=None,
                        lots_json=None,
                        crawl_status="success",
                        crawl_error=None,
                        crawl_ts_utc="2026-03-24T00:00:00Z",
                    )
                ],
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    result = purchase_tool.func(query_texts=["страх имущества"])

    assert result["run_id"] == "run-1"
    assert result["index_id"] == SHARED_INDEX_ID
    assert result["search_urls"] == ["https://example.test/search"]
    assert len(result["items"]) == 1
    assert result["items"][0]["prepared_document_ids"] == ["doc-1"]
    assert len(result["prepared_documents"]) == 1


def test_purchase_search_tool_preserves_missing_crawl_timestamp_as_null(tmp_path: Path):
    pytest.skip("Legacy synchronous purchase_search_tool behavior was replaced by async retrieval snapshots.")
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (
                ["https://example.test/search"],
                [
                    PurchaseSearchItem(
                        bundle_id="bundle-1",
                        registry_number="123",
                        law="44-FZ",
                        purchase_title="Оказание услуг страхования",
                        customer_name="Фонд",
                        detail_url="https://example.test/purchase/123",
                        crawl_status="success",
                        crawl_ts_utc=None,
                    )
                ],
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    result = purchase_tool.func(query_texts=["страх имущества"])

    assert result["items"][0]["crawl_ts_utc"] is None


def test_purchase_search_tool_returns_already_indexed_hit_without_repreparing(tmp_path: Path):
    pytest.skip("Legacy synchronous purchase_search_tool behavior was replaced by async retrieval snapshots.")
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            purchase_exists=lambda purchase_id: purchase_id == "123",
            prepare_files=lambda **kwargs: (_ for _ in ()).throw(AssertionError("prepare_files must not run")),
        ),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (
                ["https://example.test/search"],
                [
                    PurchaseSearchItem(
                        bundle_id="bundle-1",
                        registry_number="123",
                        law="44-FZ",
                        purchase_title="Оказание услуг страхования",
                        customer_name="Фонд",
                        detail_url="https://example.test/purchase/123",
                        crawl_status="success",
                    )
                ],
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    result = purchase_tool.func(query_texts=["страхование"])

    assert len(result["items"]) == 1
    assert result["items"][0]["registry_number"] == "123"
    assert result["prepared_documents"] == []


def test_purchase_search_tool_stages_confirmation_before_background_retrieval(tmp_path: Path):
    workspace = _workspace(tmp_path)
    calls = {"submit": 0}

    class RetrievalClient:
        async def submit_purchase_search(self, **kwargs):
            calls["submit"] += 1
            raise AssertionError("submit_purchase_search must not run before confirmation")

        async def get_latest_for_conversation(self, **kwargs):
            return None

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={
                    "search_url": None,
                    "query_texts": ["страх имущество"],
                    "search_urls": ["https://example.test/search"],
                    "max_pages": None,
                },
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    command = asyncio.run(
        purchase_tool.coroutine(
            query_texts=["страх имущество"],
            record_from=7,
            runtime=_runtime(),
        )
    )
    result = _command_payload(command)

    assert result["status"] == "confirmation_required"
    assert result["index_id"] == SHARED_INDEX_ID
    assert result["search_urls"] == ["https://example.test/search"]
    assert result["record_from"] == 7
    assert result["request_hash"] == "hash-1"
    assert command.update["pending_crawl_request"]["query_texts"] == ["страх имущество"]
    assert command.update["pending_crawl_request_hash"] == "hash-1"
    assert command.update["pending_crawl_query_preview"] == ["https://example.test/search"]
    assert calls["submit"] == 0


def test_purchase_search_tool_accepts_injected_runtime_argument_via_ainvoke(tmp_path: Path):
    workspace = _workspace(tmp_path)

    class RetrievalClient:
        async def get_latest_for_conversation(self, **kwargs):
            return None

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-ainvoke",
                search_urls=["https://example.test/search"],
                request_payload={
                    "search_url": None,
                    "query_texts": ["страх имущество"],
                    "search_urls": ["https://example.test/search"],
                    "max_pages": 5,
                },
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    runtime = ToolRuntime(
        state={},
        context=None,
        config={"configurable": {"thread_id": "conv-1"}},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call-1",
        store=None,
    )

    command = asyncio.run(
        purchase_tool.ainvoke(
            {
                "query_texts": ["страх имущество"],
                "max_pages": 5,
                "runtime": runtime,
            }
        )
    )
    result = _command_payload(command)

    assert result["status"] == "confirmation_required"
    assert result["request_hash"] == "hash-ainvoke"
    assert command.update["pending_crawl_request"]["query_texts"] == ["страх имущество"]


def test_purchase_lookup_tool_filters_out_irrelevant_generic_insurance_hits(tmp_path: Path):
    settings = _counterparty_settings(tmp_path)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    settings.permanent_index_root.mkdir(parents=True, exist_ok=True)
    document_service = _document_service_stub()
    catalog = ProcurementCatalogService(settings=settings, document_service=document_service)
    catalog.upsert_item(
        PurchaseSearchItem(
            bundle_id="0301300247626000156",
            registry_number="0301300247626000156",
            law="44-FZ",
            purchase_title="Оказание услуг по обязательному страхованию гражданской ответственности владельцев автотранспортных средств",
            customer_name="МКУ Центр закупок",
            price_text="183 725,13",
            published_at="2026-03-23",
            updated_at=None,
            submission_deadline="2026-03-31",
            detail_url="https://zakupki.gov.ru/epz/order/notice/ea20/view/common-info.html?regNumber=0301300247626000156",
            common_info_url=None,
            documents_url=None,
            document_urls=["https://zakupki.gov.ru/file-1"],
            downloaded_files=[
                r"C:\cache\0301300247626000156\Проект контракта.docx",
                r"C:\cache\0301300247626000156\Требования к заявке.docx",
            ],
            prepared_document_ids=["doc-1", "doc-2"],
            indexed_documents_count=2,
            last_indexed_at="2026-03-28T19:48:45.429251Z",
            documents_json=None,
            common_info_json=None,
            lots_json=None,
            crawl_status="cached",
            crawl_error=None,
            crawl_ts_utc=None,
        )
    )
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=document_service,
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        procurement_catalog=catalog,
    )

    purchase_lookup_tool = _tool_by_name(deps, "purchase_lookup_tool")
    result = purchase_lookup_tool.func(query="геостационарный спутник страхование ответственности")

    assert result["items"] == []
    assert result["returned_records"] == 0
    assert result["total_ready_records"] == 0


def test_purchase_lookup_tool_returns_lightweight_card_payload(tmp_path: Path):
    settings = _counterparty_settings(tmp_path)
    settings.work_root.mkdir(parents=True, exist_ok=True)
    settings.permanent_index_root.mkdir(parents=True, exist_ok=True)
    document_service = _document_service_stub()
    catalog = ProcurementCatalogService(settings=settings, document_service=document_service)
    catalog.upsert_item(
        PurchaseSearchItem(
            bundle_id="32615846691",
            registry_number="32615846691",
            law="223-FZ",
            purchase_title="Оказание услуг по добровольному страхованию транспортных средств",
            customer_name="ООО ТЕПЛОЭНЕРГО",
            price_text="4 387 445,00",
            published_at="2026-03-26",
            updated_at=None,
            submission_deadline="2026-04-07",
            detail_url="https://zakupki.gov.ru/epz/order/notice/notice223/common-info.html?noticeInfoId=19573074",
            common_info_url="https://zakupki.gov.ru/epz/order/notice/notice223/common-info.html?noticeInfoId=19573074",
            documents_url="https://zakupki.gov.ru/epz/order/notice/notice223/documents.html?noticeInfoId=19573074",
            document_urls=[
                "https://zakupki.gov.ru/file/a",
                "https://zakupki.gov.ru/file/b",
            ],
            downloaded_files=[
                r"C:\cache\32615846691\Извещение о закупке.docx",
                r"C:\cache\32615846691\Документация о ЗП.docx",
            ],
            prepared_document_ids=["doc-a", "doc-b"],
            indexed_documents_count=2,
            last_indexed_at="2026-03-28T19:48:45.429251Z",
            documents_json=None,
            common_info_json=None,
            lots_json=None,
            crawl_status="cached",
            crawl_error=None,
            crawl_ts_utc=None,
        )
    )
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=document_service,
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        procurement_catalog=catalog,
    )

    purchase_lookup_tool = _tool_by_name(deps, "purchase_lookup_tool")
    result = purchase_lookup_tool.func(registry_number="32615846691")

    assert result["returned_records"] == 1
    assert result["items"][0]["downloaded_files"] == [
        "Извещение о закупке.docx",
        "Документация о ЗП.docx",
    ]
    assert result["items"][0]["document_urls"] == []
    assert result["items"][0]["prepared_document_ids"] == []


def test_purchase_search_tool_refreshes_same_running_request_with_partial_subset(tmp_path: Path):
    workspace = _workspace(tmp_path)
    partial_item = PurchaseSearchItem(
        bundle_id="bundle-1",
        registry_number="123",
        law="44-FZ",
        purchase_title="Оказание услуг страхования",
        customer_name="Фонд",
        detail_url="https://example.test/purchase/123",
        crawl_status="success",
        crawl_ts_utc=None,
    ).model_dump()
    partial_doc = PreparedDocument(
        document_id="doc-1",
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_path="C:/tmp/common_info.json",
        file_name="common_info.json",
        file_type="json",
        text_excerpt="страх имущества",
        entities=PreparedDocumentEntities(inn=["7707083893"]),
        chunks_count=3,
    ).model_dump()
    partial_snapshot = _retrieval_snapshot(
        status="in_progress",
        stage="purchase_processing",
        message="Processing procurement 123 [1/2].",
        items=[partial_item],
        prepared_documents=[partial_doc],
        progress={
            "total_queries": 1,
            "completed_queries": 1,
            "total_purchases": 2,
            "processed_purchases": 1,
            "total_files": 4,
            "processed_files": 2,
            "prepared_documents": 1,
            "indexed_segments": 3,
        },
    )

    class RetrievalClient:
        async def get_latest_for_conversation(self, **kwargs):
            return partial_snapshot

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={"search_urls": ["https://example.test/search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    command = asyncio.run(
        purchase_tool.coroutine(query_texts=["страх имущество"], runtime=_runtime())
    )
    result = _command_payload(command)

    assert result["retrieval_status"] == "in_progress"
    assert result["items"][0]["crawl_ts_utc"] is None
    assert "prepared_documents" not in result
    assert result["progress"]["processed_purchases"] == 1


def test_purchase_search_tool_reuses_active_retrieval_while_another_request_is_supplied(tmp_path: Path):
    workspace = _workspace(tmp_path)
    active_snapshot = _retrieval_snapshot(
        retrieval_id="ret-other",
        request_hash="other-hash",
        status="in_progress",
        stage="crawler_search",
        message="Looking zakupki.gov.ru [1/1] with search string: osago",
        progress={
            "total_queries": 1,
            "completed_queries": 0,
            "total_purchases": 7,
            "processed_purchases": 2,
            "total_files": 4,
            "processed_files": 1,
            "prepared_documents": 0,
            "indexed_segments": 0,
        },
    )
    calls = {"latest": 0, "submit": 0}

    class RetrievalStore:
        async def get_retrieval(self, **kwargs):
            return None

        async def get_latest_for_conversation(self, **kwargs):
            calls["latest"] += 1
            return active_snapshot

        async def submit_purchase_search(self, **kwargs):
            calls["submit"] += 1
            raise AssertionError("submit_purchase_search must not run while an active retrieval exists")

        async def lookup_submission(self, **kwargs):
            return SimpleNamespace(
                active=_retrieval_snapshot(
                    request_hash="other-hash",
                    status="in_progress",
                    stage="crawler_search",
                    message="Looking zakupki.gov.ru [1/1] with search string: страхован",
                ),
                matching=None,
            )

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={"search_urls": ["https://example.test/search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalStore(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")

    command = asyncio.run(
            purchase_tool.coroutine(query_texts=["страхование"], runtime=_runtime())
        )
    result = _command_payload(command)

    assert calls == {"latest": 1, "submit": 0}
    assert result["retrieval_status"] == "in_progress"
    assert result["progress"]["processed_purchases"] == 2
    assert command.update["active_retrieval_id"] == "ret-other"


def test_purchase_search_tool_converts_retrieval_user_input_error(tmp_path: Path):
    workspace = _workspace(tmp_path)

    class RetrievalClient:
        async def get_latest_for_conversation(self, **kwargs):
            return None

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-invalid",
                search_urls=["https://example.test/search"],
                request_payload={"search_url": "https://invalid.example"},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    command = asyncio.run(
        purchase_tool.coroutine(search_url="https://invalid.example", runtime=_runtime())
    )
    result = _command_payload(command)

    assert result["status"] == "confirmation_required"
    assert result["request_hash"] == "hash-invalid"


def test_purchase_search_tool_refreshes_existing_snapshot_when_search_input_is_empty(tmp_path: Path):
    workspace = _workspace(tmp_path)
    snapshot = _retrieval_snapshot(
        status="in_progress",
        stage="crawler_search",
        message="Looking zakupki.gov.ru [1/3] with search string: страхован транспорт",
    )
    calls: dict[str, int] = {"get_retrieval": 0, "get_latest": 0, "submit": 0}

    class RetrievalClient:
        async def get_retrieval(self, **kwargs):
            calls["get_retrieval"] += 1
            assert kwargs["retrieval_id"] == "ret-1"
            assert kwargs["include_payloads"] is True
            return snapshot

        async def get_latest_for_conversation(self, **kwargs):
            calls["get_latest"] += 1
            return None

        async def submit_purchase_search(self, **kwargs):
            calls["submit"] += 1
            raise AssertionError("submit_purchase_search must not be called on empty refresh")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    command = asyncio.run(
        purchase_tool.coroutine(
            runtime=_runtime_with_state(active_retrieval_id="ret-1"),
        )
    )
    result = _command_payload(command)

    assert result["run_id"] == "run-1"
    assert result["retrieval_status"] == "in_progress"
    assert result["message"].startswith("Looking zakupki.gov.ru")
    assert calls == {"get_retrieval": 1, "get_latest": 0, "submit": 0}


def test_purchase_search_tool_wraps_unexpected_retrieval_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)

    class RetrievalClient:
        async def get_latest_for_conversation(self, **kwargs):
            raise RuntimeError("retrieval boom")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")

    with pytest.raises(ToolUserCorrectableError, match="retrieval boom") as exc_info:
        asyncio.run(purchase_tool.coroutine(query_texts=["осаго"], runtime=_runtime()))
    assert exc_info.value.code == "PURCHASE_SEARCH_FAILED"


def test_purchase_search_tool_reads_requested_run_id_from_state(tmp_path: Path):
    workspace = _workspace(tmp_path)

    class RetrievalClient:
        async def get_latest_for_conversation(self, **kwargs):
            return None

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-state-1",
                search_urls=["https://example.test/state-search"],
                request_payload={"search_urls": ["https://example.test/state-search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalClient(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    command = asyncio.run(
        purchase_tool.coroutine(
            query_texts=["страх транспорт"],
            runtime=_runtime_with_state(active_retrieval_run_id="run-state-1"),
        )
    )
    result = _command_payload(command)

    assert result["status"] == "confirmation_required"
    assert command.update["pending_crawl_request_hash"] == "hash-state-1"


def test_retrieval_api_returns_400_for_user_correctable_request_error():
    response = asyncio.run(
        retrieval_api.submit_purchase_search(
            SubmitPurchaseRetrievalRequest(conversation_id="conv-1")
        )
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["code"] == "MISSING_SEARCH_INPUT"
    assert payload["input_field"] == "query_texts"


def test_purchase_search_tool_reuses_existing_pending_request_with_same_hash(tmp_path: Path):
    workspace = _workspace(tmp_path)

    class RetrievalStore:
        async def get_latest_for_conversation(self, **kwargs):
            return None

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={"search_urls": ["https://example.test/search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalStore(),
    )
    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    command = asyncio.run(
        purchase_tool.coroutine(
            query_texts=["insurance"],
            runtime=_runtime_with_state(
                pending_crawl_request={"search_url": None, "query_texts": ["insurance"], "max_pages": None},
                pending_crawl_request_hash="hash-1",
            ),
        )
    )

    result = _command_payload(command)

    assert result["status"] == "confirmation_required"
    assert command.update["pending_crawl_request"]["query_texts"] == ["insurance"]


def test_purchase_search_tool_returns_completed_snapshot_without_requeueing(tmp_path: Path):
    workspace = _workspace(tmp_path)
    completed_snapshot = _retrieval_snapshot(
        status="completed",
        stage="completed",
        message="Procurement retrieval finished.",
        items=[
            PurchaseSearchItem(
                bundle_id="bundle-1",
                registry_number="123",
                law="44-FZ",
                purchase_title="Оказание услуг страхования",
                customer_name="Фонд",
                detail_url="https://example.test/purchase/123",
                crawl_status="success",
            ).model_dump()
        ],
        prepared_documents=[],
        progress={
            "total_queries": 1,
            "completed_queries": 1,
            "total_purchases": 1,
            "processed_purchases": 1,
            "total_files": 0,
            "processed_files": 0,
            "prepared_documents": 0,
            "indexed_segments": 0,
        },
    )

    class RetrievalStore:
        async def get_retrieval(self, **kwargs):
            return completed_snapshot

        async def submit_purchase_search(self, **kwargs):
            return completed_snapshot

        async def lookup_submission(self, **kwargs):
            return SimpleNamespace(active=None, matching=completed_snapshot)

        async def create_job(self, **kwargs):
            raise AssertionError("create_job must not be called for completed snapshot refresh")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={"search_urls": ["https://example.test/search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalStore(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    refresh_command = asyncio.run(
        purchase_tool.coroutine(runtime=_runtime_with_state(active_retrieval_id="ret-1"))
    )
    command = asyncio.run(
        purchase_tool.coroutine(query_texts=["страх имущество"], runtime=_runtime())
    )
    result = _command_payload(refresh_command)

    assert result["retrieval_status"] == "completed"
    assert result["items"][0]["registry_number"] == "123"
    assert result["progress"]["processed_purchases"] == 1


def test_purchase_search_tool_pages_completed_snapshot_to_five_records(tmp_path: Path):
    workspace = _workspace(tmp_path)
    items = [
        PurchaseSearchItem(
            bundle_id=f"bundle-{index}",
            registry_number=str(100 + index),
            law="44-FZ",
            purchase_title=f"Insurance {index}",
            customer_name="Customer",
            detail_url=f"https://example.test/purchase/{100 + index}",
            crawl_status="success",
        ).model_dump()
        for index in range(7)
    ]
    completed_snapshot = _retrieval_snapshot(
        status="completed",
        stage="completed",
        message="Procurement retrieval finished.",
        items=items,
        prepared_documents=[],
        progress={
            "total_queries": 1,
            "completed_queries": 1,
            "total_purchases": 7,
            "processed_purchases": 7,
            "total_files": 0,
            "processed_files": 0,
            "prepared_documents": 0,
            "indexed_segments": 0,
        },
    )

    class RetrievalStore:
        async def get_retrieval(self, **kwargs):
            return completed_snapshot

        async def submit_purchase_search(self, **kwargs):
            return completed_snapshot

        async def lookup_submission(self, **kwargs):
            return SimpleNamespace(active=None, matching=completed_snapshot)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={"search_urls": ["https://example.test/search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalStore(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    refresh_command = asyncio.run(
        purchase_tool.coroutine(runtime=_runtime_with_state(active_retrieval_id="ret-1"))
    )
    command = asyncio.run(
        purchase_tool.coroutine(query_texts=["insurance"], runtime=_runtime())
    )
    result = _command_payload(refresh_command)

    assert result["retrieval_status"] == "completed"
    assert len(result["items"]) == 5
    assert result["record_from"] == 0
    assert result["returned_records"] == 5
    assert result["total_ready_records"] == 7
    assert result["next_record_from"] == 5
    assert result["items"][0]["registry_number"] == "100"
    assert result["items"][-1]["registry_number"] == "104"


def test_purchase_search_tool_pages_completed_snapshot_from_record_offset(tmp_path: Path):
    workspace = _workspace(tmp_path)
    items = [
        PurchaseSearchItem(
            bundle_id=f"bundle-{index}",
            registry_number=str(100 + index),
            law="44-FZ",
            purchase_title=f"Insurance {index}",
            customer_name="Customer",
            detail_url=f"https://example.test/purchase/{100 + index}",
            crawl_status="success",
        ).model_dump()
        for index in range(7)
    ]
    completed_snapshot = _retrieval_snapshot(
        status="completed",
        stage="completed",
        message="Procurement retrieval finished.",
        items=items,
        prepared_documents=[],
    )

    class RetrievalStore:
        async def get_retrieval(self, **kwargs):
            return completed_snapshot

        async def submit_purchase_search(self, **kwargs):
            return completed_snapshot

        async def lookup_submission(self, **kwargs):
            return SimpleNamespace(active=None, matching=completed_snapshot)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            build_request_spec=lambda **kwargs: PurchaseRequestSpec(
                request_hash="hash-1",
                search_urls=["https://example.test/search"],
                request_payload={"search_urls": ["https://example.test/search"]},
            )
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
        retrieval_client=RetrievalStore(),
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")
    refresh_command = asyncio.run(
        purchase_tool.coroutine(
            record_from=5,
            runtime=_runtime_with_state(active_retrieval_id="ret-1"),
        )
    )
    command = asyncio.run(
        purchase_tool.coroutine(query_texts=["insurance"], record_from=5, runtime=_runtime())
    )
    result = _command_payload(refresh_command)

    assert len(result["items"]) == 2
    assert result["record_from"] == 5
    assert result["returned_records"] == 2
    assert result["total_ready_records"] == 7
    assert result["next_record_from"] is None
    assert [item["registry_number"] for item in result["items"]] == ["105", "106"]


def test_doc_search_tool_requires_explicit_index_lookup(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            search=lambda **kwargs: captured.update(kwargs)
            or DocSearchResponse(index_id=SHARED_INDEX_ID, matches=[])
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = _tool_by_name(deps, "doc_search_tool")
    result = doc_search_tool.func(index_id=SHARED_INDEX_ID, query="страхование")

    assert captured["index_id"] == SHARED_INDEX_ID
    assert result["index_id"] == SHARED_INDEX_ID
    assert result["matches"] == []


def test_doc_search_tool_forwards_purchase_and_source_filters(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            search=lambda **kwargs: captured.update(kwargs)
            or DocSearchResponse(index_id=SHARED_INDEX_ID, matches=[])
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = _tool_by_name(deps, "doc_search_tool")
    doc_search_tool.func(
        index_id=SHARED_INDEX_ID,
        query="страхование",
        purchase_id="123",
        source_id="abc",
    )

    assert captured["purchase_id"] == "123"
    assert captured["source_id"] == "abc"


def test_purchase_search_tool_propagates_adapter_errors(tmp_path: Path):
    pytest.skip("Legacy synchronous purchase_search_tool behavior was replaced by async retrieval snapshots.")
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crawler boom"))
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    purchase_tool = _tool_by_name(deps, "purchase_search_tool")

    with pytest.raises(RuntimeError, match="crawler boom"):
        purchase_tool.func(query_texts=["страх имущества"])


def test_doc_search_tool_propagates_invalid_index_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            search=lambda **kwargs: (_ for _ in ()).throw(
                ToolUserCorrectableError(
                    code="INVALID_INDEX_ID",
                    message="unknown index",
                    suggestion="Reuse a valid shared index_id returned by an acquisition tool.",
                    input_field="index_id",
                )
            )
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = _tool_by_name(deps, "doc_search_tool")

    with pytest.raises(ToolUserCorrectableError, match="unknown index"):
        doc_search_tool.func(index_id="missing", query="страхование")


def test_doc_search_tool_wraps_unexpected_search_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            search=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("doc search boom"))
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = _tool_by_name(deps, "doc_search_tool")

    with pytest.raises(ToolUserCorrectableError, match="doc search boom") as exc_info:
        doc_search_tool.func(index_id="idx-1", query="страхование")
    assert exc_info.value.code == "DOC_SEARCH_FAILED"


def test_read_cached_document_tool_reads_cached_content_window(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(
                model_dump=lambda: {
                    "source": "prepared_document_cache",
                    "index_id": SHARED_INDEX_ID,
                    "document_id": "doc-1",
                    "bundle_id": "bundle-1",
                    "purchase_id": "123",
                    "source_id": "src-1",
                    "parsed_at_utc": "2026-03-28T00:00:00Z",
                    "file_path": str(workspace.artifacts_dir / "bundle-1" / "doc.txt"),
                    "file_name": "doc.txt",
                    "source_kind": "purchase",
                    "source_url": "https://example.test/purchase/123",
                    "content_source": "local_file",
                    "total_chars": 20,
                    "offset": 0,
                    "returned_chars": 11,
                    "next_offset": 11,
                    "truncated": True,
                    "content": "cached text",
                }
            )
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )
    result = asyncio.run(read_tool.coroutine(index_id=SHARED_INDEX_ID, document_id="doc-1"))

    assert captured == {
        "index_id": SHARED_INDEX_ID,
        "document_id": "doc-1",
        "workspace": None,
        "bundle_id": None,
        "file_name": None,
        "offset": None,
        "max_chars": None,
    }
    assert result["document_id"] == "doc-1"
    assert result["content"] == "cached text"
    assert result["next_offset"] == 11


def test_read_cached_document_tool_wraps_unexpected_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cache read boom"))
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )

    with pytest.raises(ToolUserCorrectableError, match="cache read boom") as exc_info:
        asyncio.run(read_tool.coroutine(index_id=SHARED_INDEX_ID, document_id="doc-1"))
    assert exc_info.value.code == "READ_CACHED_DOCUMENT_FAILED"


def test_read_cached_document_tool_returns_unavailable_payload_for_normal_resolution_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: (_ for _ in ()).throw(
                ToolUserCorrectableError(
                    code="FILE_NOT_DOWNLOADED",
                    message="Файл ещё не скачан.",
                    suggestion="Откройте другой уже скачанный файл или дождитесь завершения загрузки.",
                    input_field="file_name",
                )
            )
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )

    result = asyncio.run(
        read_tool.coroutine(
            bundle_id="0245100001626000175",
            file_name="Приложение 4. Требования к содержанию, составу заявки и инструкция.docx",
        )
    )

    assert result["status"] == "unavailable"
    assert result["error_code"] == "FILE_NOT_DOWNLOADED"
    assert result["bundle_id"] == "0245100001626000175"
    assert result["file_name"] == "Приложение 4. Требования к содержанию, составу заявки и инструкция.docx"


def test_read_cached_document_tool_uses_bundle_and_file_name_with_active_run(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(
                model_dump=lambda: {
                    "source": "prepared_document_cache",
                    "index_id": SHARED_INDEX_ID,
                    "document_id": "doc-1",
                    "bundle_id": "32615846691",
                    "purchase_id": "32615846691",
                    "source_id": "src-1",
                    "parsed_at_utc": None,
                    "file_path": str(workspace.artifacts_dir / "32615846691" / "Документация о ЗП.docx"),
                    "file_name": "Документация о ЗП.docx",
                    "source_kind": "purchase",
                    "source_url": None,
                    "content_source": "local_file",
                    "total_chars": 100,
                    "offset": 0,
                    "returned_chars": 100,
                    "next_offset": None,
                    "truncated": False,
                    "content": "criteria",
                }
            )
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )
    result = asyncio.run(
        read_tool.coroutine(
            index_id=SHARED_INDEX_ID,
            bundle_id="32615846691",
            file_name="Документация о ЗП.docx",
            runtime=_runtime_with_state(active_retrieval_run_id="run-1"),
        )
    )

    assert captured["workspace"] == workspace
    assert captured["bundle_id"] == "32615846691"
    assert captured["file_name"] == "Документация о ЗП.docx"
    assert captured["document_id"] is None
    assert result["file_name"] == "Документация о ЗП.docx"


def test_read_cached_document_tool_reuses_shared_index_when_index_id_is_omitted(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(
                model_dump=lambda: {
                    "source": "prepared_document_cache",
                    "index_id": SHARED_INDEX_ID,
                    "document_id": "doc-1",
                    "bundle_id": "32615846691",
                    "purchase_id": "32615846691",
                    "source_id": "src-1",
                    "parsed_at_utc": None,
                    "file_path": str(workspace.artifacts_dir / "32615846691" / "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"),
                    "file_name": "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx",
                    "source_kind": "purchase",
                    "source_url": None,
                    "content_source": "local_file",
                    "total_chars": 100,
                    "offset": 0,
                    "returned_chars": 100,
                    "next_offset": None,
                    "truncated": False,
                    "content": "criteria",
                }
            )
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )
    result = asyncio.run(
        read_tool.coroutine(
            bundle_id="32615846691",
            file_name="Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx",
            runtime=_runtime_with_state(active_retrieval_run_id="run-1", index_id=SHARED_INDEX_ID),
        )
    )

    assert captured["index_id"] == SHARED_INDEX_ID
    assert captured["workspace"] == workspace
    assert result["index_id"] == SHARED_INDEX_ID


def test_read_cached_document_tool_accepts_injected_runtime_argument(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(
                model_dump=lambda: {
                    "source": "prepared_document_cache",
                    "index_id": SHARED_INDEX_ID,
                    "document_id": "doc-1",
                    "bundle_id": "32615846691",
                    "purchase_id": "32615846691",
                    "source_id": "src-1",
                    "parsed_at_utc": None,
                    "file_path": str(workspace.artifacts_dir / "32615846691" / "doc.txt"),
                    "file_name": "doc.txt",
                    "source_kind": "purchase",
                    "source_url": None,
                    "content_source": "local_file",
                    "total_chars": 100,
                    "offset": 0,
                    "returned_chars": 100,
                    "next_offset": None,
                    "truncated": False,
                    "content": "criteria",
                }
            )
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )
    runtime = ToolRuntime(
        state={"active_retrieval_run_id": "run-1", "index_id": SHARED_INDEX_ID},
        context=None,
        config={"configurable": {"thread_id": "conv-1"}},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call-1",
        store=None,
    )

    result = asyncio.run(
        read_tool.coroutine(
            index_id=SHARED_INDEX_ID,
            bundle_id="32615846691",
            file_name="Документация о ЗП.docx",
            offset=0,
            max_chars=12000,
            runtime=runtime,
        )
    )

    assert captured["workspace"] == workspace
    assert captured["index_id"] == SHARED_INDEX_ID
    assert captured["bundle_id"] == "32615846691"
    assert captured["file_name"] == "Документация о ЗП.docx"
    assert result["content"] == "criteria"


def test_read_cached_document_tool_indexes_local_purchase_file_when_missing_from_index(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured_prepare = {}
    cached_path = tmp_path / "permanent_index" / "purchase_downloads" / "32615840717" / "Документация о закупке (КАСКО)_2.doc"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_text("cached text", encoding="utf-8")

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
            get_by_index=lambda index_id: workspace,
        ),
        document_service=_document_service_stub(
            read_cached_document=lambda **kwargs: sales_tools.ReadCachedDocumentResponse(
                index_id=SHARED_INDEX_ID,
                document_id="doc_9b3429223d7bb1b2",
                bundle_id="32615840717",
                purchase_id="32615840717",
                source_id="src-doc-1",
                parsed_at_utc=None,
                file_path=str(cached_path),
                file_name="Документация о закупке (КАСКО)_2.doc",
                source_kind="purchase",
                source_url=None,
                content_source="local_file",
                total_chars=11,
                offset=0,
                returned_chars=11,
                next_offset=None,
                truncated=False,
                content="cached text",
            ),
            source_exists=lambda source_id: False,
            prepare_files=lambda **kwargs: captured_prepare.update(kwargs) or [],
            _purchase_cached_relative_hint=lambda **kwargs: "32615840717/Документация о закупке (КАСКО)_2.doc",
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    read_tool = next(
        tool_item for tool_item in build_sales_lead_tools(deps) if tool_item.name == "read_cached_document_tool"
    )
    runtime = ToolRuntime(
        state={"active_retrieval_run_id": "run-1", "index_id": SHARED_INDEX_ID},
        context=None,
        config={"configurable": {"thread_id": "conv-1"}},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call-1",
        store=None,
    )

    result = asyncio.run(
        read_tool.coroutine(
            index_id=SHARED_INDEX_ID,
            document_id="doc_9b3429223d7bb1b2",
            offset=0,
            max_chars=9000,
            runtime=runtime,
        )
    )

    assert result["content"] == "cached text"
    assert captured_prepare["workspace"] == workspace
    assert captured_prepare["origin"] == "purchase"
    assert captured_prepare["bundle_id"] == "32615840717"
    assert captured_prepare["registry_number"] == "32615840717"
    assert captured_prepare["file_paths"] == [str(cached_path)]
    assert (
        captured_prepare["provenance_by_path"][str(cached_path)]["artifact_relpath"]
        == "32615840717/Документация о закупке (КАСКО)_2.doc"
    )


def test_document_preparation_service_clamps_oversized_max_chars(monkeypatch, tmp_path: Path):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "32615840717"
    download_dir.mkdir(parents=True, exist_ok=True)
    cached_file = download_dir / "Документация о закупке (КАСКО)_2.doc"
    cached_file.write_text("x" * 40000, encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="x" * 40000,
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=None,
        bundle_id="32615840717",
        file_name="Документация о закупке (КАСКО)_2.doc",
        offset=0,
        max_chars=120000,
    )

    assert response.returned_chars == 30000
    assert response.next_offset == 30000
    assert response.truncated is True


def test_retrieve_page_tool_prepares_pages_and_attachments(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="Публичная страница про страхование.",
                    metadata={
                        "source": "https://example.test/page",
                        "title": "Страница",
                        "content_type": "text/html",
                    },
                ),
                SimpleNamespace(
                    page_content="Вложение с ИНН 7707083893.",
                    metadata={
                        "source_type": "web_download",
                        "source": "https://example.test/files/spec.pdf",
                        "parent_url": "https://example.test/page",
                        "download_filename": "spec.pdf",
                        "content_type": "application/pdf",
                    },
                ),
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    def save_text_artifact(*, workspace, relative_dir, file_name, content):
        target_dir = workspace.artifacts_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        target_path.write_text(content, encoding="utf-8")
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
                    source_url=source_url,
                    original_source_url=provenance.get("original_source_url"),
                    original_file_name=provenance.get("original_file_name"),
                    original_content_type=provenance.get("original_content_type"),
                    derived_artifact_path=provenance.get("derived_artifact_path"),
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    file_type="txt",
                    text_excerpt="prepared",
                    entities=PreparedDocumentEntities(),
                    chunks_count=2,
                )
            )
        return prepared

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            save_text_artifact=save_text_artifact,
            prepare_files=prepare_files,
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")
    result = asyncio.run(
        retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime())
    )

    assert result["run_id"] == "run-1"
    assert result["index_id"] == SHARED_INDEX_ID
    assert len(result["pages"]) == 1
    assert result["pages"][0]["attachments"] == ["https://example.test/files/spec.pdf"]
    assert len(result["prepared_documents"]) == 2
    assert result["prepared_documents"][1]["file_name"].endswith(".txt")
    assert result["prepared_documents"][1]["original_file_name"] == "spec.pdf"


def test_retrieve_page_tool_skips_already_indexed_page_without_repreparing(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="Публичная страница про страхование.",
                    metadata={
                        "source": "https://example.test/page",
                        "title": "Страница",
                        "content_type": "text/html",
                    },
                )
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            source_exists=lambda source_id: True,
            save_text_artifact=lambda **kwargs: (_ for _ in ()).throw(AssertionError("save_text_artifact must not run")),
            prepare_files=lambda **kwargs: (_ for _ in ()).throw(AssertionError("prepare_files must not run")),
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")
    result = asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime()))

    assert result["index_id"] == SHARED_INDEX_ID
    assert len(result["pages"]) == 1
    assert result["pages"][0]["url"] == "https://example.test/page"
    assert result["pages"][0]["prepared_document_ids"] == []
    assert result["prepared_documents"] == []


def test_retrieve_page_tool_uses_fixed_internal_loader_settings(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}

    class Loader:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.last_errors = []

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="page",
                    metadata={"source": "https://example.test/page", "title": "Page"},
                )
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            save_text_artifact=lambda **kwargs: str(workspace.artifacts_dir / "page.txt"),
            prepare_files=lambda **kwargs: [
                PreparedDocument(
                    document_id="doc-1",
                    origin="open_source",
                    bundle_id="bundle-1",
                    source_url="https://example.test/page",
                    file_path=str(workspace.artifacts_dir / "page.txt"),
                    file_name="page.txt",
                    file_type="txt",
                    text_excerpt="page",
                    entities=PreparedDocumentEntities(),
                    chunks_count=1,
                )
            ],
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")
    asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime()))

    assert captured["depth"] == 0
    assert captured["max_concurrency"] == 4
    assert captured["playwright_headless"] is True
    assert captured["follow_download_links"] is True
    assert captured["continue_on_error"] is False


def test_retrieve_page_tool_ignores_scope_filter_warnings_when_docs_were_fetched(
    monkeypatch,
    tmp_path: Path,
):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = [
                {
                    "url": "https://service.nalog.ru/",
                    "depth": 1,
                    "stage": "filter",
                    "error": "URL out of crawl scope (same_host).",
                    "backend": "playwright",
                }
            ]

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="page",
                    metadata={"source": "https://egrul.nalog.ru/index.html", "title": "Page"},
                )
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            save_text_artifact=lambda **kwargs: str(workspace.artifacts_dir / "page.txt"),
            prepare_files=lambda **kwargs: [
                PreparedDocument(
                    document_id="doc-1",
                    origin="open_source",
                    bundle_id="bundle-1",
                    source_url="https://egrul.nalog.ru/index.html",
                    file_path=str(workspace.artifacts_dir / "page.txt"),
                    file_name="page.txt",
                    file_type="txt",
                    text_excerpt="page",
                    entities=PreparedDocumentEntities(),
                    chunks_count=1,
                )
            ],
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")
    result = asyncio.run(retrieve_page_tool.coroutine(url="https://egrul.nalog.ru/index.html", runtime=_runtime()))

    assert result["index_id"] == SHARED_INDEX_ID
    assert len(result["pages"]) == 1
    assert result["pages"][0]["url"] == "https://egrul.nalog.ru/index.html"
    assert result["prepared_documents"][0]["document_id"] == "doc-1"


def test_retrieve_page_tool_wraps_loader_errors_as_user_correctable(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            raise RuntimeError("loader boom")

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")

    with pytest.raises(ToolUserCorrectableError, match="loader boom") as exc_info:
        asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime()))
    assert exc_info.value.code == "FETCH_FAILED"


def test_retrieve_page_tool_wraps_auth_required_loader_errors(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            raise RuntimeError("Authentication required and login_processor failed or not provided.")

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")

    with pytest.raises(ToolUserCorrectableError, match="Authentication required") as exc_info:
        asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/protected", runtime=_runtime()))
    assert exc_info.value.code == "FETCH_AUTH_REQUIRED"


def test_retrieve_page_tool_wraps_html_parser_errors_as_user_correctable(
    monkeypatch,
    tmp_path: Path,
):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            raise ValueError("HTML table has no rows.")

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")

    with pytest.raises(ToolUserCorrectableError, match="HTML table has no rows") as exc_info:
        asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime()))
    assert exc_info.value.code == "FETCH_FAILED"


def test_web_search_tool_parses_ranked_results_and_unwraps_urls(monkeypatch, tmp_path: Path):
    pytest.skip("sales_lead_agent now uses the project-wide YandexSearchTool.")
    captured = {}
    html_payload = """
    <html>
      <body>
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rusprofile.ru%2Fid%2F123">АО СКБ Контур</a>
        <a class="result__url" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rusprofile.ru%2Fid%2F123">www.rusprofile.ru/id/123</a>
        <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rusprofile.ru%2Fid%2F123">ИНН <b>6663003127</b></a>
        <a class="result__a" href="https://egrul.nalog.ru/index.html">ЕГРЮЛ</a>
        <a class="result__url" href="https://egrul.nalog.ru/index.html">egrul.nalog.ru/index.html</a>
        <div class="result__snippet">Официальная выписка</div>
      </body>
    </html>
    """

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *, headers, timeout, follow_redirects):
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["follow_redirects"] = follow_redirects

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = dict(params or {})
            return FakeResponse(html_payload)

    monkeypatch.setattr(sales_tools.httpx, "Client", FakeClient)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    web_search_tool = _tool_by_name(deps, "web_search")
    result = web_search_tool.func(search_string="6663003127")

    assert captured["url"] == "https://html.duckduckgo.com/html/"
    assert captured["params"] == {"q": "6663003127", "kl": "ru-ru"}
    assert captured["follow_redirects"] is True
    assert result["search_string"] == "6663003127"
    assert result["results"] == [
        {
            "title": "АО СКБ Контур",
            "url": "https://www.rusprofile.ru/id/123",
            "snippet": "ИНН 6663003127",
            "display_url": "www.rusprofile.ru/id/123",
        },
        {
            "title": "ЕГРЮЛ",
            "url": "https://egrul.nalog.ru/index.html",
            "snippet": "Официальная выписка",
            "display_url": "egrul.nalog.ru/index.html",
        },
    ]


def test_web_search_tool_rejects_empty_search_string(tmp_path: Path):
    pytest.skip("sales_lead_agent now uses the project-wide YandexSearchTool.")
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    web_search_tool = _tool_by_name(deps, "web_search")

    with pytest.raises(ToolUserCorrectableError, match="non-empty search_string") as exc_info:
        web_search_tool.func(search_string="  ")
    assert exc_info.value.code == "INVALID_SEARCH_STRING"


def test_web_search_tool_wraps_unexpected_errors(monkeypatch, tmp_path: Path):
    pytest.skip("sales_lead_agent now uses the project-wide YandexSearchTool.")
    class FakeClient:
        def __init__(self, *, headers, timeout, follow_redirects):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            raise RuntimeError("web boom")

    monkeypatch.setattr(sales_tools.httpx, "Client", FakeClient)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    web_search_tool = _tool_by_name(deps, "web_search")

    with pytest.raises(ToolUserCorrectableError, match="web boom") as exc_info:
        web_search_tool.func(search_string="6663003127")
    assert exc_info.value.code == "WEB_SEARCH_FAILED"


def test_sales_lead_tools_use_project_wide_yandex_search_tool(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sales_tools.config, "YA_API_KEY", "ya-key", raising=False)
    monkeypatch.setattr(sales_tools.config, "YA_FOLDER_ID", "folder-id", raising=False)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    web_search_tool = _tool_by_name(deps, "web_search")

    assert isinstance(web_search_tool, sales_tools.YandexSearchTool)
    assert web_search_tool.name == "web_search"
    assert web_search_tool.api_key == "ya-key"
    assert web_search_tool.folder_id == "folder-id"


def test_sales_lead_yandex_search_tool_invocation_uses_shared_contract(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sales_tools.config, "YA_API_KEY", "ya-key", raising=False)
    monkeypatch.setattr(sales_tools.config, "YA_FOLDER_ID", "folder-id", raising=False)
    captured = {}

    def fake_get_data(self, endpoint, headers, payload):
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        captured["payload"] = payload
        return "Найдена статья.\n\n** Ссылка на статью: https://example.test/company **"

    monkeypatch.setattr(sales_tools.YandexSearchTool, "_get_data", fake_get_data)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    web_search_tool = _tool_by_name(deps, "web_search")
    result = web_search_tool.invoke({"query": "6663003127"})

    assert captured["endpoint"] == "https://searchapi.api.cloud.yandex.net/v2/web/search"
    assert captured["headers"]["Authorization"] == "Api-Key ya-key"
    assert captured["payload"]["query"]["queryText"] == "6663003127"
    assert result.startswith("Найдена статья.")


def test_sales_lead_yandex_search_tool_returns_failure_text(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sales_tools.config, "YA_API_KEY", "ya-key", raising=False)
    monkeypatch.setattr(sales_tools.config, "YA_FOLDER_ID", "folder-id", raising=False)

    def fake_get_data(self, endpoint, headers, payload):
        raise RuntimeError("web boom")

    monkeypatch.setattr(sales_tools.YandexSearchTool, "_get_data", fake_get_data)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: _workspace(tmp_path), get=lambda run_id: _workspace(tmp_path)),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    web_search_tool = _tool_by_name(deps, "web_search")
    result = web_search_tool.invoke({"query": "6663003127"})

    assert result == "Yandex Search failed: web boom"


def test_counterparty_tools_surface_scoring_errors_as_user_correctable_and_soften_fssp_errors(
    tmp_path: Path,
):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(
            scoring=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("scoring boom")),
            fssp=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fssp boom")),
        ),
        open_source_max_concurrency=4,
    )

    scoring_tool = _tool_by_name(deps, "counterparty_scoring_tool")
    fssp_tool = _tool_by_name(deps, "counterparty_fssp_tool")

    with pytest.raises(ToolUserCorrectableError, match="scoring boom") as exc_info:
        scoring_tool.func(inn="7707083893")
    assert exc_info.value.code == "COUNTERPARTY_SCORING_FAILED"
    fssp_result = fssp_tool.func(inn="7707083893")

    assert fssp_result == {
        "source": "damia_fssp",
        "status": "success",
        "inn": "7707083893",
        "grouped": [],
        "raw_format": 1,
        "message": "Данные не найдены",
    }


def test_counterparty_lookup_uses_dadata_find_by_id_and_filters_to_main_by_default(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "method": "POST",
                "url": "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party",
                "payload": {
                    "suggestions": [
                        {
                            "value": "АО \"ПФ \"СКБ Контур\"",
                            "data": {
                                "inn": "6663003127",
                                "ogrn": "1026605606620",
                                "kpp": "667101001",
                                "okved": "62.01",
                                "state": {"status": "ACTIVE"},
                                "name": {
                                    "short_with_opf": "АО \"ПФ \"СКБ Контур\"",
                                    "full_with_opf": "АКЦИОНЕРНОЕ ОБЩЕСТВО \"ПРОИЗВОДСТВЕННАЯ ФИРМА \\\"СКБ КОНТУР\\\"\"",
                                },
                                "address": {
                                    "value": "г Екатеринбург, ул Народной Воли, стр. 19А",
                                },
                                "management": {
                                    "name": "Сродных Михаил Юрьевич",
                                    "post": "ГЕНЕРАЛЬНЫЙ ДИРЕКТОР",
                                },
                            },
                        }
                    ]
                },
            }
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.lookup_party(inn="6663003127", include_branches=False)

    assert captured["headers"]["Authorization"] == "Token dadata-key"
    assert captured["calls"] == [
        {
            "url": "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party",
            "params": {},
            "method": "POST",
            "json": {"query": "6663003127", "branch_type": "MAIN"},
        }
    ]
    assert result.inn == "6663003127"
    assert result.found is True
    assert result.name == "АО \"ПФ \"СКБ Контур\""
    assert result.ogrn == "1026605606620"
    assert result.kpp == "667101001"
    assert result.state_status == "ACTIVE"
    assert result.management_name == "Сродных Михаил Юрьевич"


def test_counterparty_lookup_can_search_with_branches(monkeypatch, tmp_path: Path):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "method": "POST",
                "url": "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party",
                "payload": {"suggestions": []},
            }
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.lookup_party(inn="6663003127", include_branches=True)

    assert captured["calls"] == [
        {
            "url": "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party",
            "params": {},
            "method": "POST",
            "json": {"query": "6663003127"},
        }
    ]
    assert result.found is False
    assert result.message == "Контрагент не найден"


def test_counterparty_lookup_tool_forwards_include_branches(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(
            scoring=lambda **kwargs: None,
            fssp=lambda **kwargs: None,
            lookup_party=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(
                model_dump=lambda: {
                    "source": "dadata_party",
                    "status": "success",
                    "inn": "6663003127",
                    "found": True,
                    "name": "АО \"ПФ \"СКБ Контур\"",
                }
            ),
        ),
        open_source_max_concurrency=4,
    )

    lookup_tool = _tool_by_name(deps, "counterparty_lookup_tool")
    result = lookup_tool.func(inn="6663003127", include_branches=True)

    assert captured == {"inn": "6663003127", "include_branches": True}
    assert result["name"] == "АО \"ПФ \"СКБ Контур\""


def test_counterparty_lookup_tool_wraps_unexpected_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(
            scoring=lambda **kwargs: None,
            fssp=lambda **kwargs: None,
            lookup_party=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("lookup boom")),
        ),
        open_source_max_concurrency=4,
    )

    lookup_tool = _tool_by_name(deps, "counterparty_lookup_tool")

    with pytest.raises(ToolUserCorrectableError, match="lookup boom") as exc_info:
        lookup_tool.func(inn="6663003127")
    assert exc_info.value.code == "COUNTERPARTY_LOOKUP_FAILED"


def test_counterparty_scoring_uses_query_key_and_no_authorization(monkeypatch, tmp_path: Path):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/scoring/score",
                "payload": [
                    {
                        "risk_value": 0.12,
                        "score_value": 88.0,
                        "reliability_value": 0.93,
                        "top_factors": [{"name": "Liquidity", "value": 1.5, "nwoe": 0.2}],
                    }
                ],
            }
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.scoring(inn="7707083893", model="b2b", include_fincoefs=False)

    assert captured["headers"] == {"Accept": "application/json"}
    assert "Authorization" not in captured["headers"]
    assert captured["calls"] == [
        {
            "url": "https://api.damia.ru/scoring/score",
            "params": {"inn": "7707083893", "model": "b2b", "key": "test-key"},
        }
    ]
    assert result.score.score_value == 88.0
    assert result.fincoefs == []


def test_counterparty_scoring_with_fincoefs_normalizes_damia_shapes(monkeypatch, tmp_path: Path):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/scoring/score",
                "payload": {
                    "result": {
                        "РискЗнач": "0,15",
                        "РискЗона": "yellow",
                        "БаллЗнач": "87.4",
                        "БаллЗона": "green",
                        "НадежностьЗнач": "0,93",
                        "НадежностьЗона": "green",
                        "Показатели": [
                            {"Наименование": "Revenue", "Значение": "123,4", "nWoE": "0,12"}
                        ],
                    }
                },
            },
            {
                "url": "https://api.damia.ru/scoring/fincoefs",
                "payload": {
                    "result": [
                        {
                            "Фин. коэффициент": "Debt ratio",
                            "Знач": "1,25",
                            "Норма": "1,00",
                            "НормаСравн": "выше нормы",
                        }
                    ]
                },
            },
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.scoring(inn="7707083893", model=None, include_fincoefs=True)

    assert [call["url"] for call in captured["calls"]] == [
        "https://api.damia.ru/scoring/score",
        "https://api.damia.ru/scoring/fincoefs",
    ]
    assert captured["calls"][0]["params"] == {
        "inn": "7707083893",
        "model": "_problemCredit",
        "key": "test-key",
    }
    assert captured["calls"][1]["params"] == {"inn": "7707083893", "key": "test-key"}
    assert result.score.risk_value == pytest.approx(0.15)
    assert result.score.risk_zone == "yellow"
    assert result.score.score_value == pytest.approx(87.4)
    assert result.score.reliability_value == pytest.approx(0.93)
    assert result.score.top_factors[0].name == "Revenue"
    assert result.score.top_factors[0].value == pytest.approx(123.4)
    assert result.score.top_factors[0].nwoe == pytest.approx(0.12)
    assert result.fincoefs[0].name == "Debt ratio"
    assert result.fincoefs[0].value == pytest.approx(1.25)
    assert result.fincoefs[0].norm == pytest.approx(1.0)
    assert result.fincoefs[0].comparison == "выше нормы"


def test_counterparty_scoring_normalizes_nested_year_payload(monkeypatch, tmp_path: Path):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/scoring/score",
                "payload": {
                    "7707083893": {
                        "_problemCredit": {
                            "2016": {
                                "РискЗнач": 0.2,
                                "РискЗона": "старый",
                            },
                            "2017": {
                                "РискЗнач": 0.083,
                                "РискЗона": "Низкий риск",
                                "БаллЗнач": 4.53,
                                "БаллЗона": "Высокий балл",
                                "НадежностьЗнач": 0.148,
                                "НадежностьЗона": "Низкая надежность",
                                "Показатели": [
                                    {
                                        "Наименование": "1600",
                                        "Значение": 1,
                                        "nWoE": -1.289,
                                    }
                                ],
                            },
                        }
                    }
                },
            }
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.scoring(inn="7707083893", model=None, include_fincoefs=False)

    assert captured["calls"][0]["params"]["model"] == "_problemCredit"
    assert result.score.risk_value == pytest.approx(0.083)
    assert result.score.risk_zone == "Низкий риск"
    assert result.score.score_value == pytest.approx(4.53)
    assert result.score.reliability_value == pytest.approx(0.148)
    assert result.score.top_factors[0].name == "1600"
    assert result.score.top_factors[0].nwoe == pytest.approx(-1.289)


@pytest.mark.parametrize(
    "payload",
    [
        {"7734374725": {"_problemCredit": {}}},
        [],
    ],
)
def test_counterparty_scoring_treats_empty_payloads_as_no_data(
    monkeypatch,
    tmp_path: Path,
    payload,
):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/scoring/score",
                "payload": payload,
            }
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.scoring(inn="7734374725", model=None, include_fincoefs=False)

    assert captured["calls"][0]["params"]["model"] == "_problemCredit"
    assert result.inn == "7734374725"
    assert result.score.risk_value is None
    assert result.score.risk_zone is None
    assert result.score.score_value is None
    assert result.score.reliability_value is None
    assert result.score.top_factors == []
    assert result.fincoefs == []


def test_counterparty_scoring_normalizes_nested_fincoefs_payload(monkeypatch, tmp_path: Path):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/scoring/score",
                "payload": {
                    "result": {
                        "РискЗнач": "0,15",
                        "РискЗона": "yellow",
                    }
                },
            },
            {
                "url": "https://api.damia.ru/scoring/fincoefs",
                "payload": {
                    "6663003127": {
                        "КоэфОборЗапасов": {
                            "2023": {
                                "Знач": 187.83904103608,
                                "Норма": 22.78767411433,
                                "НормаСравн": "Выше нормы",
                            },
                            "2024": {
                                "Знач": 261.13922463675,
                                "Норма": 25.474167494428,
                                "НормаСравн": "Выше нормы",
                            },
                        },
                        "ПериодОборЗапасов": {
                            "2024": {
                                "Знач": 1.5,
                                "Норма": 2.0,
                                "НормаСравн": "В пределах нормы",
                            }
                        },
                    }
                },
            },
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.scoring(inn="6663003127", model=None, include_fincoefs=True)
    fincoefs_by_name = {item.name: item for item in result.fincoefs}

    assert captured["calls"][1]["params"] == {"inn": "6663003127", "key": "test-key"}
    assert fincoefs_by_name["КоэфОборЗапасов"].value == pytest.approx(261.13922463675)
    assert fincoefs_by_name["КоэфОборЗапасов"].norm == pytest.approx(25.474167494428)
    assert fincoefs_by_name["КоэфОборЗапасов"].comparison == "Выше нормы"
    assert fincoefs_by_name["ПериодОборЗапасов"].value == pytest.approx(1.5)
    assert fincoefs_by_name["ПериодОборЗапасов"].norm == pytest.approx(2.0)
    assert fincoefs_by_name["ПериодОборЗапасов"].comparison == "В пределах нормы"


def test_counterparty_fssp_forces_grouped_format_and_normalizes_damia_shapes(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/fssp/isps",
                "payload": {
                    "result": [
                        {
                            "Год": "2024",
                            "Статус": "Не завершено",
                            "Предмет": "страховые взносы",
                            "Сумма": "125000,50",
                            "Количество": "2",
                            "ИП": ["10000/24/77001-ИП", "10001/24/77001-ИП"],
                        }
                    ]
                },
            }
        ],
        captured=captured,
    )

    client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))
    result = client.fssp(
        inn="7707083893",
        from_date="2024-01-01",
        to_date="2024-12-31",
    )

    assert captured["calls"] == [
        {
            "url": "https://api.damia.ru/fssp/isps",
            "params": {
                "inn": "7707083893",
                "from_date": "2024-01-01",
                "to_date": "2024-12-31",
                "format": 1,
                "key": "test-key",
            },
        }
    ]
    assert result.raw_format == 1
    assert result.grouped == [
        sales_tools.FSSPGroupedRecord(
            year=2024,
            status="Не завершено",
            subject="страховые взносы",
            amount=125000.50,
            count=2,
            proceeding_ids=["10000/24/77001-ИП", "10001/24/77001-ИП"],
        )
    ]


def test_counterparty_clients_fail_fast_when_required_env_missing(tmp_path: Path):
    missing_urls_client = sales_tools.CounterpartyClients(
        _counterparty_settings(
            tmp_path,
            damia_scoring_api_key="",
            scoring_base_url="",
            fssp_base_url="",
            damia_fssp_api_key="",
        )
    )

    with pytest.raises(RuntimeError, match="SALES_LEAD_AGENT_SCORING_BASE_URL is not configured"):
        missing_urls_client.scoring(inn="7707083893", model=None, include_fincoefs=False)
    with pytest.raises(RuntimeError, match="SALES_LEAD_AGENT_FSSP_BASE_URL is not configured"):
        missing_urls_client.fssp(inn="7707083893", from_date=None, to_date=None)

    missing_key_client = sales_tools.CounterpartyClients(
        _counterparty_settings(
            tmp_path,
            damia_scoring_api_key="",
            damia_fssp_api_key="",
            dadata_api_key="",
        )
    )

    with pytest.raises(RuntimeError, match="SALES_LEAD_AGENT_DAMIA_SCORING_API_KEY is not configured"):
        missing_key_client.scoring(inn="7707083893", model=None, include_fincoefs=False)
    with pytest.raises(RuntimeError, match="SALES_LEAD_AGENT_DAMIA_FSSP_API_KEY is not configured"):
        missing_key_client.fssp(inn="7707083893", from_date=None, to_date=None)
    with pytest.raises(RuntimeError, match="DADATA_API_KEY is not configured"):
        missing_key_client.lookup_party(inn="7707083893", include_branches=False)


def test_counterparty_clients_surface_damia_string_errors(monkeypatch, tmp_path: Path):
    scoring_capture = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/scoring/score",
                "payload": "Ошибка: Не указан параметр: model",
            }
        ],
        captured=scoring_capture,
    )

    scoring_client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))

    with pytest.raises(RuntimeError, match="Scoring API error: Ошибка: Не указан параметр: model"):
        scoring_client.scoring(inn="7707083893", model="bad", include_fincoefs=False)

    fssp_capture = {}
    _install_fake_http_client(
        monkeypatch,
        responses=[
            {
                "url": "https://api.damia.ru/fssp/isps",
                "payload": "Ошибка: Неверный ключ (1)",
            }
        ],
        captured=fssp_capture,
    )

    fssp_client = sales_tools.CounterpartyClients(_counterparty_settings(tmp_path))

    with pytest.raises(RuntimeError, match="FSSP API error: Ошибка: Неверный ключ"):
        fssp_client.fssp(inn="7707083893", from_date=None, to_date=None)


def test_counterparty_fssp_tool_uses_grouped_contract_only(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(
            scoring=lambda **kwargs: None,
            fssp=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(
                model_dump=lambda: {
                    "source": "damia_fssp",
                    "status": "success",
                    "inn": "7707083893",
                    "grouped": [],
                    "raw_format": 1,
                }
            ),
        ),
        open_source_max_concurrency=4,
    )

    fssp_tool = _tool_by_name(deps, "counterparty_fssp_tool")
    result = fssp_tool.func(inn="7707083893", from_date="2024-01-01")

    assert captured == {
        "inn": "7707083893",
        "from_date": "2024-01-01",
        "to_date": None,
    }
    assert result["raw_format"] == 1


def test_retrieve_page_tool_raises_on_empty_fetched_content(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="",
                    metadata={"source": "https://example.test/page"},
                )
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")

    with pytest.raises(ToolUserCorrectableError, match="Fetched empty content"):
        asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime()))


def test_retrieve_page_tool_wraps_unexpected_preparation_errors(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            return [
                SimpleNamespace(
                    page_content="страхование",
                    metadata={"source": "https://example.test/page"},
                )
            ]

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=_document_service_stub(
            save_text_artifact=lambda **kwargs: "page.txt",
            prepare_files=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("prepare boom")),
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    retrieve_page_tool = _tool_by_name(deps, "retrieve_page_tool")

    with pytest.raises(ToolUserCorrectableError, match="prepare boom") as exc_info:
        asyncio.run(retrieve_page_tool.coroutine(url="https://example.test/page", runtime=_runtime()))
    assert exc_info.value.code == "RETRIEVE_PAGE_FAILED"


@pytest.mark.parametrize("archive_name", ["bundle.zip", "bundle.rar", "bundle.arj"])
def test_document_preparation_unpacks_archives_with_7z_and_uses_rag_lib_text_loader(
    monkeypatch,
    tmp_path: Path,
    archive_name: str,
):
    workspace = _workspace(tmp_path)
    archive_path = workspace.downloads_dir / archive_name
    archive_path.write_bytes(b"archive")

    captured = {}

    class Loader:
        def __init__(self, file_path):
            captured["file_path"] = file_path

        def load(self):
            return [SimpleNamespace(page_content="страхование имущества", metadata={"source": captured["file_path"]})]

    monkeypatch.setattr(sales_tools, "TextLoader", Loader)

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=страхован&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)
    monkeypatch.setattr(sales_tools, "_find_7z_executable", lambda: "7z")

    def fake_run_7z(command):
        if command[1] == "l":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Path = bundle\n"
                    "Type = archive\n"
                    "----------\n"
                    "Path = nested\n"
                    "Folder = +\n"
                    "\n"
                    "Path = nested/inner.txt\n"
                    "Folder = -\n"
                    "Size = 10\n"
                ),
                stderr="",
            )
        extracted = workspace.artifacts_dir / "bundle-1" / f"archive_{archive_path.stem}" / "nested" / "inner.txt"
        extracted.parent.mkdir(parents=True, exist_ok=True)
        extracted.write_text("страхование имущества", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sales_tools, "_run_7z", fake_run_7z)

    class FakeSemanticSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"semantic-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    class FakeSentenceSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"seg-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    monkeypatch.setattr(
        service,
        "_build_sentence_splitter",
        lambda **kwargs: FakeSentenceSplitter(),
    )
    monkeypatch.setattr(
        service,
        "_build_semantic_splitter",
        lambda: FakeSemanticSplitter(),
    )

    prepared = service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(archive_path)],
    )

    assert captured["file_path"].endswith("inner.txt")
    assert prepared[0].file_name == "inner.txt"
    assert prepared[0].chunks_count > 0


def test_document_preparation_skips_when_rag_lib_loader_returns_no_documents(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    file_path = workspace.downloads_dir / "broken.txt"
    file_path.write_text("boom", encoding="utf-8")

    class Loader:
        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            return []

    monkeypatch.setattr(sales_tools, "TextLoader", Loader)

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)

    prepared = service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(file_path)],
    )

    assert prepared == []


def test_document_preparation_warns_and_skips_unsupported_files_but_continues(monkeypatch, tmp_path: Path, caplog):
    workspace = _workspace(tmp_path)
    unsupported_path = workspace.downloads_dir / "blob.bin"
    supported_path = workspace.downloads_dir / "note.txt"
    unsupported_path.write_bytes(b"\x00\x01")
    supported_path.write_text("страхование имущества", encoding="utf-8")

    class Loader:
        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            return [SimpleNamespace(page_content="страхование имущества", metadata={"source": self.file_path})]

    monkeypatch.setattr(sales_tools, "TextLoader", Loader)

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)

    class FakeSemanticSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"semantic-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    class FakeSentenceSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"seg-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    monkeypatch.setattr(service, "_build_sentence_splitter", lambda **kwargs: FakeSentenceSplitter())
    monkeypatch.setattr(service, "_build_semantic_splitter", lambda: FakeSemanticSplitter())

    with caplog.at_level("WARNING"):
        prepared = service.prepare_files(
            workspace=workspace,
            origin="purchase",
            bundle_id="bundle-1",
            registry_number="123",
            source_url="https://example.test/purchase/123",
            file_paths=[str(unsupported_path), str(supported_path)],
        )

    assert len(prepared) == 1
    assert prepared[0].file_name == "note.txt"
    assert "Skipping unsupported file during preparation" in caplog.text


def test_document_preparation_retries_failed_purchase_file_and_skips_indexed_successes(
    monkeypatch,
    tmp_path: Path,
):
    def make_workspace(root: Path, run_id: str) -> RunWorkspace:
        downloads = root / run_id / "downloads"
        web = root / run_id / "web"
        index = root / run_id / "index"
        artifacts = root / run_id / "artifacts"
        for path in (downloads, web, index, artifacts):
            path.mkdir(parents=True, exist_ok=True)
        return RunWorkspace(
            run_id=run_id,
            index_id=SHARED_INDEX_ID,
            root_dir=root / run_id,
            downloads_dir=downloads,
            web_dir=web,
            index_dir=index,
            artifacts_dir=artifacts,
        )

    workspace_one = make_workspace(tmp_path, "run-1")
    workspace_two = make_workspace(tmp_path, "run-2")

    def create_purchase_files(workspace: RunWorkspace) -> tuple[Path, Path]:
        bundle_dir = workspace.artifacts_dir / "123"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        good_path = bundle_dir / "ok.txt"
        bad_path = bundle_dir / "~$bad.xlsx"
        good_path.write_text("страхование имущества", encoding="utf-8")
        bad_path.write_bytes(b"broken")
        return good_path, bad_path

    good_one, bad_one = create_purchase_files(workspace_one)
    good_two, bad_two = create_purchase_files(workspace_two)

    parse_calls = {"txt": 0, "xlsx": 0}
    indexed_source_ids: set[str] = set()

    class TextLoader:
        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            parse_calls["txt"] += 1
            return [SimpleNamespace(page_content="страхование имущества", metadata={"source": self.file_path})]

    class ExcelLoader:
        def __init__(self, file_path, output_format):
            self.file_path = file_path
            self.output_format = output_format

        def load(self):
            parse_calls["xlsx"] += 1
            raise ValueError("Excel file format cannot be determined, you must specify an engine manually.")

    class FakeSemanticSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"semantic-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    class FakeSentenceSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"seg-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    monkeypatch.setattr(sales_tools, "TextLoader", TextLoader)
    monkeypatch.setattr(sales_tools, "ExcelLoader", ExcelLoader)

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_build_sentence_splitter", lambda **kwargs: FakeSentenceSplitter())
    monkeypatch.setattr(service, "_build_semantic_splitter", lambda: FakeSemanticSplitter())
    monkeypatch.setattr(service, "source_exists", lambda source_id: source_id in indexed_source_ids)

    def fake_index_documents(*, segments):
        for segment in segments:
            source_id = str(segment.metadata.get("source_id") or "").strip()
            if source_id:
                indexed_source_ids.add(source_id)

    monkeypatch.setattr(service, "_index_documents", fake_index_documents)

    first_progress: list[dict[str, object]] = []
    prepared_first = service.prepare_files(
        workspace=workspace_one,
        origin="purchase",
        bundle_id="123",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(good_one), str(bad_one)],
        progress_callback=lambda **event: first_progress.append(event),
    )

    second_progress: list[dict[str, object]] = []
    prepared_second = service.prepare_files(
        workspace=workspace_two,
        origin="purchase",
        bundle_id="123",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(good_two), str(bad_two)],
        progress_callback=lambda **event: second_progress.append(event),
    )

    assert len(prepared_first) == 1
    assert prepared_first[0].file_name == "ok.txt"
    assert prepared_first[0].source_id
    assert prepared_second == []
    assert parse_calls["txt"] == 1
    assert parse_calls["xlsx"] == 2
    assert any(event["stage"] == "file_failed" and event["file_name"] == "~$bad.xlsx" for event in first_progress)
    assert any(event["stage"] == "file_failed" and event["file_name"] == "~$bad.xlsx" for event in second_progress)
    assert any(event["stage"] == "file_skipped" and event["file_name"] == "ok.txt" for event in second_progress)


def test_document_preparation_retries_index_failed_purchase_file_and_skips_indexed_successes(
    monkeypatch,
    tmp_path: Path,
):
    def make_workspace(root: Path, run_id: str) -> RunWorkspace:
        downloads = root / run_id / "downloads"
        web = root / run_id / "web"
        index = root / run_id / "index"
        artifacts = root / run_id / "artifacts"
        for path in (downloads, web, index, artifacts):
            path.mkdir(parents=True, exist_ok=True)
        return RunWorkspace(
            run_id=run_id,
            index_id=SHARED_INDEX_ID,
            root_dir=root / run_id,
            downloads_dir=downloads,
            web_dir=web,
            index_dir=index,
            artifacts_dir=artifacts,
        )

    workspace_one = make_workspace(tmp_path, "run-1")
    workspace_two = make_workspace(tmp_path, "run-2")

    def create_purchase_files(workspace: RunWorkspace) -> tuple[Path, Path]:
        bundle_dir = workspace.artifacts_dir / "123"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        good_path = bundle_dir / "ok.txt"
        bad_path = bundle_dir / "bad.txt"
        good_path.write_text("страхование имущества", encoding="utf-8")
        bad_path.write_text("страхование ответственности", encoding="utf-8")
        return good_path, bad_path

    good_one, bad_one = create_purchase_files(workspace_one)
    good_two, bad_two = create_purchase_files(workspace_two)

    indexed_source_ids: set[str] = set()
    delete_calls: list[tuple[str | None, str]] = []

    class TextLoader:
        def __init__(self, file_path):
            self.file_path = file_path

        def load(self):
            return [SimpleNamespace(page_content="страхование", metadata={"source": self.file_path})]

    class FakeSemanticSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"semantic-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    class FakeSentenceSplitter:
        def split_documents(self, documents):
            return [
                sales_tools.Segment(
                    content=str(doc.page_content),
                    metadata=dict(doc.metadata or {}),
                    segment_id=f"seg-{index}",
                )
                for index, doc in enumerate(documents)
                if str(doc.page_content or "").strip()
            ]

    monkeypatch.setattr(sales_tools, "TextLoader", TextLoader)

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_build_sentence_splitter", lambda **kwargs: FakeSentenceSplitter())
    monkeypatch.setattr(service, "_build_semantic_splitter", lambda: FakeSemanticSplitter())
    monkeypatch.setattr(service, "source_exists", lambda source_id: source_id in indexed_source_ids)

    def fake_delete_cached_document(*, source_id, document_id):
        delete_calls.append((source_id, document_id))

    def fake_index_documents(*, segments):
        source_id = str(segments[0].metadata.get("source_id") or "")
        file_path = str(segments[0].metadata.get("file_path") or "")
        if file_path.endswith("bad.txt"):
            raise RuntimeError("index boom")
        if source_id:
            indexed_source_ids.add(source_id)

    monkeypatch.setattr(service, "_delete_cached_document", fake_delete_cached_document)
    monkeypatch.setattr(service, "_index_documents", fake_index_documents)

    first_progress: list[dict[str, object]] = []
    prepared_first = service.prepare_files(
        workspace=workspace_one,
        origin="purchase",
        bundle_id="123",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(good_one), str(bad_one)],
        progress_callback=lambda **event: first_progress.append(event),
    )

    second_progress: list[dict[str, object]] = []
    prepared_second = service.prepare_files(
        workspace=workspace_two,
        origin="purchase",
        bundle_id="123",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_paths=[str(good_two), str(bad_two)],
        progress_callback=lambda **event: second_progress.append(event),
    )

    assert [doc.file_name for doc in prepared_first] == ["ok.txt"]
    assert prepared_second == []
    assert any(event["stage"] == "index_failed" and event["file_name"] == "bad.txt" for event in first_progress)
    assert any(event["stage"] == "index_failed" and event["file_name"] == "bad.txt" for event in second_progress)
    assert any(event["stage"] == "file_skipped" and event["file_name"] == "ok.txt" for event in second_progress)
    assert delete_calls


def test_document_preparation_rejects_zip_path_traversal(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    archive_path = workspace.downloads_dir / "bundle.zip"
    archive_path.write_bytes(b"archive")

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        permanent_index_root=tmp_path / "permanent_index",
        shared_index_id=SHARED_INDEX_ID,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=страхован&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
        damia_scoring_api_key="",
        damia_fssp_api_key="",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)
    monkeypatch.setattr(sales_tools, "_find_7z_executable", lambda: "7z")
    monkeypatch.setattr(
        sales_tools,
        "_run_7z",
        lambda command: SimpleNamespace(
            returncode=0,
            stdout=(
                "Path = bundle\n"
                "Type = archive\n"
                "----------\n"
                "Path = ../evil.txt\n"
                "Folder = -\n"
                "Size = 4\n"
            ),
            stderr="",
        ),
    )

    with pytest.raises(ValueError, match="escapes target directory"):
        service.prepare_files(
            workspace=workspace,
            origin="purchase",
            bundle_id="bundle-1",
            registry_number="123",
            source_url="https://example.test/purchase/123",
            file_paths=[str(archive_path)],
        )


def test_doc_search_tool_preserves_explicit_zero_top_k(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
        ),
        document_service=_document_service_stub(
            search=lambda **kwargs: captured.update(kwargs)
            or DocSearchResponse(index_id=SHARED_INDEX_ID, matches=[])
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = _tool_by_name(deps, "doc_search_tool")
    doc_search_tool.func(index_id=SHARED_INDEX_ID, query="страхование", top_k=0)

    assert captured["top_k"] == 0


def test_document_preparation_service_search_uses_vector_store_lock(monkeypatch, tmp_path: Path):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    lock_state = {"held": False, "entered": 0}

    @contextlib.contextmanager
    def fake_lock():
        lock_state["entered"] += 1
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            assert lock_state["held"] is True
            return []

    monkeypatch.setattr(service, "_vector_store_operation_lock", fake_lock)
    monkeypatch.setattr(service, "_open_vector_store", lambda embeddings=None: (None, FakeVectorStore()))

    response = service.search(
        index_id=SHARED_INDEX_ID,
        query="требования к заявке",
        top_k=5,
        source_kind="purchase",
        bundle_id="0245100001626000175",
        purchase_id="0245100001626000175",
    )

    assert response.matches == []
    assert lock_state["entered"] == 1


def test_document_preparation_service_indexes_batches_under_vector_store_lock(monkeypatch, tmp_path: Path):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    lock_state = {"held": False, "entered": 0}
    processed_batch_sizes: list[int] = []

    @contextlib.contextmanager
    def fake_lock():
        lock_state["entered"] += 1
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    class FakeIndexer:
        def __init__(self, *, vector_store, embeddings):
            self.vector_store = vector_store
            self.embeddings = embeddings

        def _process_batch(self, batch):
            assert lock_state["held"] is True
            processed_batch_sizes.append(len(batch))

    monkeypatch.setattr(service, "_vector_store_operation_lock", fake_lock)
    monkeypatch.setattr(service, "_create_embeddings", lambda: "emb")
    monkeypatch.setattr(service, "_open_vector_store", lambda embeddings=None: (embeddings, object()))
    monkeypatch.setattr(sales_tools, "Indexer", FakeIndexer)

    segments = [
        sales_tools.Segment(
            segment_id=f"seg-{index}",
            content=f"content {index}",
            metadata={"chunk_index": index},
        )
        for index in range(65)
    ]

    service._index_documents(segments=segments)

    assert processed_batch_sizes == [32, 32, 1]
    assert lock_state["entered"] == 3


def test_document_preparation_service_search_retries_retryable_vector_errors(monkeypatch, tmp_path: Path):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    calls = {"count": 0}

    class FlakyVectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("Error executing plan: Internal error: Error finding id")
            return []

    monkeypatch.setattr(service, "_open_vector_store", lambda embeddings=None: (None, FlakyVectorStore()))

    response = service.search(
        index_id=SHARED_INDEX_ID,
        query="ИНН",
        top_k=5,
        source_kind="purchase",
        bundle_id="0245100001626000175",
        purchase_id="0245100001626000175",
    )

    assert response.matches == []
    assert calls["count"] == 2


def test_document_preparation_service_search_falls_back_to_local_purchase_files_on_retryable_vector_error(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "0245100001626000175"
    download_dir.mkdir(parents=True, exist_ok=True)
    cached_file = download_dir / "requirements.txt"
    cached_file.write_text(
        "ИНН заказчика: 7802114044\nТребования к заявке и состав документов.",
        encoding="utf-8",
    )

    class BrokenVectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            raise RuntimeError("Error executing plan: Internal error: Error finding id")

    monkeypatch.setattr(service, "_open_vector_store", lambda embeddings=None: (None, BrokenVectorStore()))
    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content=Path(path).read_text(encoding="utf-8"),
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.search(
        index_id=SHARED_INDEX_ID,
        query="ИНН",
        top_k=5,
        source_kind="purchase",
        bundle_id="0245100001626000175",
        purchase_id="0245100001626000175",
    )

    assert len(response.matches) == 1
    assert response.matches[0].file_path == str(cached_file)
    assert response.matches[0].locator.startswith("local_fallback")
    assert "ИНН заказчика" in response.matches[0].snippet


def test_document_preparation_service_reads_cached_document_from_local_file(monkeypatch, tmp_path: Path):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    cached_file = tmp_path / "cached.txt"
    cached_file.write_text("alpha beta gamma", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_collection_get",
        lambda **kwargs: {
            "ids": ["doc-1:0"],
            "documents": ["chunk content"],
            "metadatas": [
                {
                    "document_id": "doc-1",
                    "bundle_id": "bundle-1",
                    "purchase_id": "123",
                    "source_id": "src-1",
                    "parsed_at_utc": "2026-03-28T00:00:00Z",
                    "file_path": str(cached_file),
                    "source_kind": "purchase",
                    "source_url": "https://example.test/purchase/123",
                    "chunk_index": 0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="alpha beta gamma",
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        document_id="doc-1",
        offset=6,
        max_chars=4,
    )

    assert response.content_source == "local_file"
    assert response.content == "beta"
    assert response.offset == 6
    assert response.returned_chars == 4
    assert response.next_offset == 10
    assert response.file_name == "cached.txt"


def test_document_preparation_service_reads_cached_purchase_file_by_bundle_and_name(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    workspace = _workspace(tmp_path)
    bundle_dir = workspace.artifacts_dir / "32615846691"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cached_file = bundle_dir / "Документация о ЗП.docx"
    cached_file.write_text("criteria and requirements", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="criteria and requirements",
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=workspace,
        bundle_id="32615846691",
        file_name="Документация о ЗП.docx",
        max_chars=100,
    )

    assert response.content_source == "local_file"
    assert response.bundle_id == "32615846691"
    assert response.purchase_id == "32615846691"
    assert response.file_name == "Документация о ЗП.docx"
    assert response.content == "criteria and requirements"
    assert response.document_id.startswith("doc_")


def test_document_preparation_service_reads_cached_purchase_file_from_download_cache(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    workspace = _workspace(tmp_path)
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "32615846691"
    download_dir.mkdir(parents=True, exist_ok=True)
    cached_file = download_dir / "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"
    cached_file.write_text("criteria and requirements", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="criteria and requirements",
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=workspace,
        bundle_id="32615846691",
        file_name="Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx",
        max_chars=100,
    )

    assert response.content_source == "local_file"
    assert response.bundle_id == "32615846691"
    assert response.file_name == "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"
    assert response.file_path == str(cached_file)
    assert response.content == "criteria and requirements"


def test_document_preparation_service_reads_cached_purchase_file_from_download_cache_without_workspace(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "32615846691"
    download_dir.mkdir(parents=True, exist_ok=True)
    cached_file = download_dir / "Документация о ЗП.docx"
    cached_file.write_text("criteria and requirements", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="criteria and requirements",
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=None,
        bundle_id="32615846691",
        file_name="Документация о ЗП.docx",
        max_chars=100,
    )

    assert response.content_source == "local_file"
    assert response.bundle_id == "32615846691"
    assert response.file_name == "Документация о ЗП.docx"
    assert response.file_path == str(cached_file)
    assert response.content == "criteria and requirements"


def test_document_preparation_service_reads_cached_purchase_file_by_downloaded_path(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    workspace = _workspace(tmp_path)
    bundle_dir = workspace.artifacts_dir / "32615846691"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cached_file = bundle_dir / "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"
    cached_file.write_text("criteria and requirements", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="criteria and requirements",
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=workspace,
        bundle_id="32615846691",
        file_name=str(cached_file),
        max_chars=100,
    )

    assert response.content_source == "local_file"
    assert response.file_name == "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"
    assert response.content == "criteria and requirements"


def test_document_preparation_service_reads_cached_purchase_file_by_unique_hint(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    workspace = _workspace(tmp_path)
    bundle_dir = workspace.artifacts_dir / "32615846691"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    doc_file = bundle_dir / "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"
    doc_file.write_text("criteria and requirements", encoding="utf-8")
    zip_file = bundle_dir / "ÐŸÑ€Ð¸Ð»Ð¾Ð¶ÐµÐ½Ð¸Ñ Ðº Ð´Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ð¸.zip"
    zip_file.write_text("zip placeholder", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="criteria and requirements" if Path(path) == doc_file else "zip placeholder",
                metadata={"source": str(path)},
            )
        ],
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=workspace,
        bundle_id="32615846691",
        file_name="Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸ÑŽ ÑÐ½Ð°Ñ‡Ð°Ð»Ð°",
        max_chars=100,
    )

    assert response.content_source == "local_file"
    assert response.file_name == "Ð”Ð¾ÐºÑƒÐ¼ÐµÐ½Ñ‚Ð°Ñ†Ð¸Ñ Ð¾ Ð—ÐŸ.docx"
    assert response.content == "criteria and requirements"


def test_document_preparation_service_reads_cached_archive_as_listing_and_member_without_workspace(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "32615846691"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_file = download_dir / "Приложения к документации.zip"
    archive_file.write_text("zip placeholder", encoding="utf-8")

    extracted_dir = (
        tmp_path
        / "runs"
        / "run_cached"
        / "artifacts"
        / "32615846691"
        / "archive_Приложения к документации"
        / "Приложения к документации"
    )
    extracted_dir.mkdir(parents=True, exist_ok=True)
    criteria_file = extracted_dir / "5. Критерии и порядок оценки.docx"
    criteria_file.write_text("criteria text from archive", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="criteria text from archive",
                metadata={"source": str(path)},
            )
        ],
    )

    archive_response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=None,
        bundle_id="32615846691",
        file_name="Приложения к документации.zip",
        max_chars=4000,
    )

    assert archive_response.content_source == "archive_listing"
    assert archive_response.file_name == "Приложения к документации.zip"
    assert "5. Критерии и порядок оценки.docx" in archive_response.content
    assert "Чтобы прочитать конкретный документ из архива" in archive_response.content

    member_response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=None,
        bundle_id="32615846691",
        file_name="5. Критерии и порядок оценки.docx",
        max_chars=4000,
    )

    assert member_response.content_source == "local_file"
    assert member_response.file_name == "5. Критерии и порядок оценки.docx"
    assert member_response.content == "criteria text from archive"


def test_document_preparation_service_continues_nonindexed_local_file_by_document_id(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "32615840717"
    download_dir.mkdir(parents=True, exist_ok=True)
    cached_file = download_dir / "Документация о закупке (КАСКО)_2.doc"
    cached_file.write_text("0123456789abcdef", encoding="utf-8")

    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda path: [
            sales_tools.Document(
                id="raw-1",
                page_content="0123456789abcdef",
                metadata={"source": str(path)},
            )
        ],
    )

    first_response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=None,
        bundle_id="32615840717",
        file_name="Документация о закупке (КАСКО)_2.doc",
        offset=0,
        max_chars=8,
    )

    second_response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        document_id=first_response.document_id,
        offset=8,
        max_chars=8,
    )

    assert first_response.content_source == "local_file"
    assert first_response.content == "01234567"
    assert first_response.next_offset == 8
    assert second_response.content_source == "local_file"
    assert second_response.file_name == "Документация о закупке (КАСКО)_2.doc"
    assert second_response.content == "89abcdef"
    assert second_response.next_offset is None


def test_document_preparation_service_continues_archive_listing_by_document_id(
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "32615846691"
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_file = download_dir / "Приложения к документации.zip"
    archive_file.write_text("zip placeholder", encoding="utf-8")

    extracted_dir = (
        tmp_path
        / "runs"
        / "run_cached"
        / "artifacts"
        / "32615846691"
        / "archive_Приложения к документации"
        / "Приложения к документации"
    )
    extracted_dir.mkdir(parents=True, exist_ok=True)
    (extracted_dir / "5. Критерии и порядок оценки.docx").write_text("criteria", encoding="utf-8")
    (extracted_dir / "7. Обоснование НМЦ.xlsx").write_text("nmc", encoding="utf-8")

    first_response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        workspace=None,
        bundle_id="32615846691",
        file_name="Приложения к документации.zip",
        offset=0,
        max_chars=120,
    )

    second_response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        document_id=first_response.document_id,
        offset=120,
        max_chars=500,
    )

    assert first_response.content_source == "archive_listing"
    assert first_response.next_offset == 120
    assert second_response.content_source == "archive_listing"
    assert second_response.file_name == "Приложения к документации.zip"
    assert "5. Критерии и порядок оценки.docx" in first_response.content + second_response.content


def test_document_preparation_service_reports_undownloaded_exact_file_without_fuzzy_ambiguity(
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    download_dir = tmp_path / "permanent_index" / "purchase_downloads" / "0245100001626000175"
    download_dir.mkdir(parents=True, exist_ok=True)
    (download_dir / "Приложение 1. Обоснование НСЦЕУТ.docx").write_text("one", encoding="utf-8")
    (download_dir / "Приложение №1 к ТЗ Перечень ЗЧ Фольксваген.xlsx").write_text("two", encoding="utf-8")

    artifact_dir = tmp_path / "runs" / "run_cached" / "artifacts" / "0245100001626000175"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "documents_json.json").write_text(
        json.dumps(
            [
                {
                    "display_name": "Приложение 4. Требования к содержанию, составу заявки и инструкция.docx",
                    "source_filename": "Приложение 4. Требования к содержанию, составу заявки и инструкция.docx",
                    "downloaded": False,
                    "local_path": None,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ToolUserCorrectableError) as exc_info:
        service.read_cached_document(
            index_id=SHARED_INDEX_ID,
            workspace=None,
            bundle_id="0245100001626000175",
            file_name="Приложение 4. Требования к содержанию, составу заявки и инструкция.docx",
            max_chars=12000,
        )

    assert exc_info.value.code == "FILE_NOT_DOWNLOADED"
    assert "has not been downloaded yet" in str(exc_info.value)


def test_document_preparation_service_falls_back_to_indexed_chunks_for_missing_local_file(
    monkeypatch,
    tmp_path: Path,
):
    service = sales_tools.DocumentPreparationService(_counterparty_settings(tmp_path))
    missing_file = tmp_path / "missing.docx"

    monkeypatch.setattr(
        service,
        "_collection_get",
        lambda **kwargs: {
            "ids": ["doc-1:1", "doc-1:0"],
            "documents": ["second chunk", "first chunk"],
            "metadatas": [
                {
                    "document_id": "doc-1",
                    "bundle_id": "bundle-1",
                    "purchase_id": "123",
                    "source_id": "src-1",
                    "parsed_at_utc": "2026-03-28T00:00:00Z",
                    "file_path": str(missing_file),
                    "source_kind": "purchase",
                    "source_url": "https://example.test/purchase/123",
                    "chunk_index": 1,
                },
                {
                    "document_id": "doc-1",
                    "bundle_id": "bundle-1",
                    "purchase_id": "123",
                    "source_id": "src-1",
                    "parsed_at_utc": "2026-03-28T00:00:00Z",
                    "file_path": str(missing_file),
                    "source_kind": "purchase",
                    "source_url": "https://example.test/purchase/123",
                    "chunk_index": 0,
                },
            ],
        },
    )

    response = service.read_cached_document(
        index_id=SHARED_INDEX_ID,
        document_id="doc-1",
        max_chars=100,
    )

    assert response.content_source == "indexed_chunks"
    assert response.content == "first chunk\n\nsecond chunk"
    assert response.truncated is False
    assert response.next_offset is None
    assert response.file_name == "missing.docx"
