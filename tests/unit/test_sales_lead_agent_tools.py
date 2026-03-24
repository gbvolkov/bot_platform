from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from zakupki_crawler.crawler import ZakupkiCrawler

from agents.sales_lead_agent import tools as sales_tools
from agents.sales_lead_agent.tools import (
    DocSearchResponse,
    PreparedDocument,
    PreparedDocumentEntities,
    ProcurementQueryBuilder,
    PurchaseSearchItem,
    RunWorkspace,
    SalesLeadAgentDependencies,
    SalesLeadAgentSettings,
    ToolUserCorrectableError,
    build_sales_lead_tools,
)


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
        index_id="index-1",
        root_dir=root,
        downloads_dir=downloads,
        web_dir=web,
        index_dir=index,
        artifacts_dir=artifacts,
    )


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

    urls = builder.build_urls(["страхован", "страхов"])

    assert len(urls) == 2
    assert "searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2%D0%B0%D0%BD" in urls[0]
    assert "searchString=%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%BE%D0%B2" in urls[1]


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

    crawler = object.__new__(ZakupkiCrawler)
    crawler.pacer = SimpleNamespace(sleep_func=lambda _seconds: None)

    crawler._wait_for_search_results(FakePage())


def test_purchase_search_tool_returns_run_index_and_prepared_documents(tmp_path: Path):
    workspace = _workspace(tmp_path)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(
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

    purchase_tool = build_sales_lead_tools(deps)[0]
    result = purchase_tool.func(query_texts=["страх имущества"])

    assert result["run_id"] == "run-1"
    assert result["index_id"] == "index-1"
    assert result["search_urls"] == ["https://example.test/search"]
    assert len(result["items"]) == 1
    assert result["items"][0]["prepared_document_ids"] == ["doc-1"]
    assert len(result["prepared_documents"]) == 1


def test_purchase_search_tool_preserves_missing_crawl_timestamp_as_null(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(prepare_files=lambda **kwargs: []),
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

    purchase_tool = build_sales_lead_tools(deps)[0]
    result = purchase_tool.func(query_texts=["страх имущества"])

    assert result["items"][0]["crawl_ts_utc"] is None


def test_doc_search_tool_requires_explicit_index_lookup(tmp_path: Path):
    workspace = _workspace(tmp_path)
    calls: list[str] = []
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
            get_by_index=lambda index_id: calls.append(index_id) or workspace,
        ),
        document_service=SimpleNamespace(
            search=lambda **kwargs: DocSearchResponse(index_id="index-1", matches=[])
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = build_sales_lead_tools(deps)[2]
    result = doc_search_tool.func(index_id="index-1", query="страхование")

    assert calls == ["index-1"]
    assert result["index_id"] == "index-1"
    assert result["matches"] == []


def test_purchase_search_tool_propagates_adapter_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(prepare_files=lambda **kwargs: []),
        purchase_adapter=SimpleNamespace(
            search=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crawler boom"))
        ),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    purchase_tool = build_sales_lead_tools(deps)[0]

    with pytest.raises(RuntimeError, match="crawler boom"):
        purchase_tool.func(query_texts=["страх имущества"])


def test_doc_search_tool_propagates_invalid_index_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(
            create_run=lambda: workspace,
            get=lambda run_id: workspace,
            get_by_index=lambda index_id: (_ for _ in ()).throw(ValueError("unknown index")),
        ),
        document_service=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = build_sales_lead_tools(deps)[2]

    with pytest.raises(ToolUserCorrectableError, match="unknown index"):
        doc_search_tool.func(index_id="missing", query="страхование")


def test_open_source_fetch_tool_prepares_pages_and_attachments(monkeypatch, tmp_path: Path):
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
        document_service=SimpleNamespace(save_text_artifact=save_text_artifact, prepare_files=prepare_files),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    open_source_tool = build_sales_lead_tools(deps)[1]
    result = asyncio.run(
        open_source_tool.coroutine(url="https://example.test/page", follow_download_links=True)
    )

    assert result["run_id"] == "run-1"
    assert result["index_id"] == "index-1"
    assert len(result["pages"]) == 1
    assert result["pages"][0]["attachments"] == ["https://example.test/files/spec.pdf"]
    assert len(result["prepared_documents"]) == 2
    assert result["prepared_documents"][1]["file_name"].endswith(".txt")
    assert result["prepared_documents"][1]["original_file_name"] == "spec.pdf"


def test_open_source_fetch_tool_preserves_explicit_zero_transport_args(monkeypatch, tmp_path: Path):
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
        document_service=SimpleNamespace(
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

    open_source_tool = build_sales_lead_tools(deps)[1]
    asyncio.run(
        open_source_tool.coroutine(
            url="https://example.test/page",
            depth=0,
            max_concurrency=0,
        )
    )

    assert captured["depth"] == 0
    assert captured["max_concurrency"] == 0


def test_open_source_fetch_tool_propagates_loader_errors(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)

    class Loader:
        def __init__(self, **kwargs):
            self.last_errors = []

        async def load(self):
            raise RuntimeError("loader boom")

    monkeypatch.setattr(sales_tools, "AsyncWebLoader", Loader)

    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(save_text_artifact=lambda **kwargs: "", prepare_files=lambda **kwargs: []),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    open_source_tool = build_sales_lead_tools(deps)[1]

    with pytest.raises(RuntimeError, match="loader boom"):
        asyncio.run(open_source_tool.coroutine(url="https://example.test/page"))


def test_counterparty_tools_propagate_client_errors(tmp_path: Path):
    workspace = _workspace(tmp_path)
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(
            scoring=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("scoring boom")),
            fssp=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fssp boom")),
        ),
        open_source_max_concurrency=4,
    )

    scoring_tool = build_sales_lead_tools(deps)[3]
    fssp_tool = build_sales_lead_tools(deps)[4]

    with pytest.raises(RuntimeError, match="scoring boom"):
        scoring_tool.func(inn="7707083893")
    with pytest.raises(RuntimeError, match="fssp boom"):
        fssp_tool.func(inn="7707083893")


def test_counterparty_fssp_tool_preserves_explicit_zero_format(tmp_path: Path):
    workspace = _workspace(tmp_path)
    captured = {}
    deps = SalesLeadAgentDependencies(
        workspace_manager=SimpleNamespace(create_run=lambda: workspace, get=lambda run_id: workspace),
        document_service=SimpleNamespace(),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(
            scoring=lambda **kwargs: None,
            fssp=lambda **kwargs: captured.update(kwargs)
            or SimpleNamespace(model_dump=lambda: {"source": "damia_fssp", "status": "success", "inn": "7707083893", "grouped": [], "raw_format": kwargs["response_format"]}),
        ),
        open_source_max_concurrency=4,
    )

    fssp_tool = build_sales_lead_tools(deps)[4]
    result = fssp_tool.func(inn="7707083893", format=0)

    assert captured["response_format"] == 0
    assert result["raw_format"] == 0


def test_open_source_fetch_tool_raises_on_empty_fetched_content(monkeypatch, tmp_path: Path):
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
        document_service=SimpleNamespace(save_text_artifact=lambda **kwargs: "", prepare_files=lambda **kwargs: []),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    open_source_tool = build_sales_lead_tools(deps)[1]

    with pytest.raises(ToolUserCorrectableError, match="Fetched empty content"):
        asyncio.run(open_source_tool.coroutine(url="https://example.test/page"))


def test_document_preparation_unpacks_zip_and_uses_rag_lib_text_loader(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    archive_path = workspace.downloads_dir / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/inner.txt", "страхование имущества")

    captured = {}

    class Loader:
        def __init__(self, file_path):
            captured["file_path"] = file_path

        def load(self):
            return [SimpleNamespace(page_content="страхование имущества", metadata={"source": captured["file_path"]})]

    monkeypatch.setattr(sales_tools, "TextLoader", Loader)

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=страхован&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        damia_api_key="",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)

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


def test_document_preparation_rejects_zip_path_traversal(monkeypatch, tmp_path: Path):
    workspace = _workspace(tmp_path)
    archive_path = workspace.downloads_dir / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.txt", "boom")

    settings = SalesLeadAgentSettings(
        work_root=tmp_path,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=страхован&recordsPerPage=_2",
        purchase_headless=True,
        open_source_max_concurrency=4,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        damia_api_key="",
        scoring_base_url="https://example.test",
        fssp_base_url="https://example.test",
    )
    service = sales_tools.DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)

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
            get_by_index=lambda index_id: workspace,
        ),
        document_service=SimpleNamespace(
            search=lambda **kwargs: captured.update(kwargs) or DocSearchResponse(index_id="index-1", matches=[])
        ),
        purchase_adapter=SimpleNamespace(),
        counterparty_clients=SimpleNamespace(),
        open_source_max_concurrency=4,
    )

    doc_search_tool = build_sales_lead_tools(deps)[2]
    doc_search_tool.func(index_id="index-1", query="страхование", top_k=0)

    assert captured["top_k"] == 0
