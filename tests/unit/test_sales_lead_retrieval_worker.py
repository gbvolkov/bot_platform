from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from agents.sales_lead_agent.tools import (
    PreparedDocument,
    PreparedDocumentEntities,
    PurchaseSearchItem,
    RunWorkspace,
)
from services.sales_lead_retrieval import worker as retrieval_worker


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
        index_id="sales_lead_permanent",
        root_dir=root,
        downloads_dir=downloads,
        web_dir=web,
        index_dir=index,
        artifacts_dir=artifacts,
    )


class _FakeStore:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []

    async def append_event(self, retrieval_id: str, **kwargs) -> None:
        self.events.append({"retrieval_id": retrieval_id, **kwargs})

    async def update_job(self, retrieval_id: str, **kwargs):
        self.updates.append({"retrieval_id": retrieval_id, **kwargs})
        return SimpleNamespace()


def test_process_job_completes_and_updates_progress(tmp_path: Path):
    workspace = _workspace(tmp_path)
    item = PurchaseSearchItem(
        bundle_id="bundle-1",
        registry_number="123",
        law="44-FZ",
        purchase_title="Оказание услуг страхования",
        customer_name="Фонд",
        detail_url="https://example.test/purchase/123",
        downloaded_files=[],
        common_info_json='{"ok": true}',
        crawl_status="success",
    )
    doc = PreparedDocument(
        document_id="doc-1",
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_url="https://example.test/purchase/123",
        file_path=str(workspace.artifacts_dir / "bundle-1" / "common_info_json.json"),
        file_name="common_info_json.json",
        file_type="json",
        text_excerpt="страх имущества",
        entities=PreparedDocumentEntities(),
        chunks_count=4,
    )

    def search(**kwargs):
        kwargs["progress_callback"](stage="crawler_plan", message="Plan", total_queries=1)
        kwargs["item_callback"](item)
        kwargs["progress_callback"](
            stage="crawler_complete",
            message="Complete",
            total_queries=1,
            unique_procurements=1,
        )
        return ["https://example.test/search"], [item]

    def prepare_files(**kwargs):
        kwargs["progress_callback"](stage="parsing_file", message="Parsing file [1/1]")
        return [doc]

    store = _FakeStore()
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(get=lambda run_id: workspace),
        purchase_adapter=SimpleNamespace(search=search),
        document_service=SimpleNamespace(
            purchase_exists=lambda purchase_id: False,
            prepare_files=prepare_files,
        ),
    )
    job = SimpleNamespace(
        retrieval_id="ret-1",
        run_id="run-1",
        request_payload={
            "search_url": None,
            "query_texts": ["страх имущество"],
            "max_pages": None,
        },
    )

    asyncio.run(retrieval_worker._process_job(job=job, store=store, deps=deps))

    assert store.updates[-1]["status"] == "completed"
    assert store.updates[-1]["clear_active"] is True
    assert store.updates[-1]["progress"]["prepared_documents"] == 1
    assert store.updates[-1]["progress"]["indexed_segments"] == 4
    assert any(event["stage"] == "completed" for event in store.events)


def test_process_job_records_purchase_failure_and_completes(tmp_path: Path):
    workspace = _workspace(tmp_path)
    item = PurchaseSearchItem(
        bundle_id="bundle-1",
        registry_number="123",
        law="44-FZ",
        purchase_title="Оказание услуг страхования",
        customer_name="Фонд",
        detail_url="https://example.test/purchase/123",
        common_info_json='{"ok": true}',
        crawl_status="success",
    )

    def search(**kwargs):
        kwargs["item_callback"](item)
        return ["https://example.test/search"], [item]

    def prepare_files(**kwargs):
        raise RuntimeError("parse boom")

    store = _FakeStore()
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(get=lambda run_id: workspace),
        purchase_adapter=SimpleNamespace(search=search),
        document_service=SimpleNamespace(
            purchase_exists=lambda purchase_id: False,
            prepare_files=prepare_files,
        ),
    )
    job = SimpleNamespace(
        retrieval_id="ret-1",
        run_id="run-1",
        request_payload={
            "search_url": None,
            "query_texts": ["страх имущество"],
            "max_pages": None,
        },
    )

    asyncio.run(retrieval_worker._process_job(job=job, store=store, deps=deps))

    assert store.updates[-1]["status"] == "completed"
    assert store.updates[-1]["clear_active"] is True
    assert store.updates[-1]["items"][0]["registry_number"] == "123"
    assert store.updates[-1]["prepared_documents"] == []
    assert any(event["stage"] == "purchase_failed" for event in store.events)
    assert any(event["stage"] == "purchase_failed" and event["level"] == "error" for event in store.events)


def test_process_job_continues_when_prepare_files_reports_file_failure(tmp_path: Path):
    workspace = _workspace(tmp_path)
    item = PurchaseSearchItem(
        bundle_id="bundle-1",
        registry_number="123",
        law="44-FZ",
        purchase_title="Оказание услуг страхования",
        customer_name="Фонд",
        detail_url="https://example.test/purchase/123",
        common_info_json='{"ok": true}',
        crawl_status="success",
    )
    doc = PreparedDocument(
        document_id="doc-1",
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_id="src-1",
        source_url="https://example.test/purchase/123",
        file_path=str(workspace.artifacts_dir / "bundle-1" / "good.txt"),
        file_name="good.txt",
        file_type="txt",
        text_excerpt="страхование",
        entities=PreparedDocumentEntities(),
        chunks_count=2,
    )

    def search(**kwargs):
        kwargs["item_callback"](item)
        return ["https://example.test/search"], [item]

    def prepare_files(**kwargs):
        kwargs["progress_callback"](
            stage="file_failed",
            message="Failed to parse file bad.xlsx: boom",
            file_name="bad.xlsx",
            file_path="C:/tmp/bad.xlsx",
            error="boom",
            error_type="ValueError",
        )
        return [doc]

    store = _FakeStore()
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(get=lambda run_id: workspace),
        purchase_adapter=SimpleNamespace(search=search),
        document_service=SimpleNamespace(
            purchase_exists=lambda purchase_id: False,
            prepare_files=prepare_files,
        ),
    )
    job = SimpleNamespace(
        retrieval_id="ret-1",
        run_id="run-1",
        request_payload={
            "search_url": None,
            "query_texts": ["страх имущество"],
            "max_pages": None,
        },
    )

    asyncio.run(retrieval_worker._process_job(job=job, store=store, deps=deps))

    assert store.updates[-1]["status"] == "completed"
    assert any(event["stage"] == "file_failed" for event in store.events)
    assert any(event["stage"] == "file_failed" and event["level"] == "error" for event in store.events)
    assert store.updates[-1]["prepared_documents"][0]["document_id"] == "doc-1"


def test_process_job_continues_when_search_query_fails(tmp_path: Path):
    workspace = _workspace(tmp_path)

    def search(**kwargs):
        kwargs["progress_callback"](stage="crawler_plan", message="Plan", total_queries=2)
        kwargs["progress_callback"](
            stage="crawler_search_failed",
            message="Failed search [1/2]: timeout",
            current=1,
            total=2,
            error="timeout",
            error_type="TimeoutError",
        )
        kwargs["progress_callback"](
            stage="crawler_complete",
            message="Complete with failures",
            total_queries=2,
            unique_procurements=0,
            failed_queries=1,
        )
        return ["https://example.test/search-1", "https://example.test/search-2"], []

    store = _FakeStore()
    deps = SimpleNamespace(
        workspace_manager=SimpleNamespace(get=lambda run_id: workspace),
        purchase_adapter=SimpleNamespace(search=search),
        document_service=SimpleNamespace(
            purchase_exists=lambda purchase_id: False,
            prepare_files=lambda **kwargs: [],
        ),
    )
    job = SimpleNamespace(
        retrieval_id="ret-1",
        run_id="run-1",
        request_payload={
            "search_url": None,
            "query_texts": ["осаго", "каско"],
            "max_pages": None,
        },
    )

    asyncio.run(retrieval_worker._process_job(job=job, store=store, deps=deps))

    assert store.updates[-1]["status"] == "completed"
    assert any(event["stage"] == "crawler_search_failed" for event in store.events)
    assert any(
        event["stage"] == "crawler_search_failed" and event["level"] == "error"
        for event in store.events
    )
