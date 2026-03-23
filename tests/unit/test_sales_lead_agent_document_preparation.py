from pathlib import Path

import pytest

from agents.sales_lead_agent import settings as settings_module
from agents.sales_lead_agent.services.document_preparation import (
    DocumentPreparationService,
    RunWorkspaceManager,
)
from agents.sales_lead_agent.settings import SalesLeadAgentSettings


def _settings(tmp_path: Path) -> SalesLeadAgentSettings:
    return SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        retention_hours=72,
        damia_api_key="",
        scoring_base_url="",
        fssp_base_url="",
        purchase_headless=True,
        open_source_max_concurrency=4,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test",
    )


def test_prepare_files_extracts_entities_and_returns_schema(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    workspace_manager = RunWorkspaceManager(settings)
    workspace = workspace_manager.create_run()
    service = DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)

    source = tmp_path / "sample.txt"
    source.write_text(
        "ООО Ромашка ИНН 7707083893 email test@example.com "
        "телефон +7 495 123-45-67 сумма 120 000 руб.",
        encoding="utf-8",
    )

    prepared = service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id="bundle-1",
        registry_number="123",
        source_url="https://example.test/item",
        file_paths=[str(source)],
    )

    assert len(prepared) == 1
    item = prepared[0]
    assert item.bundle_id == "bundle-1"
    assert "7707083893" in item.entities.inn
    assert "test@example.com" in item.entities.emails
    assert item.chunks_count >= 1
    assert item.original_source_url == "https://example.test/item"
    assert item.original_file_name == "sample.txt"
    assert item.derived_artifact_path == str(source)
    assert workspace_manager.get_by_index(workspace.index_id).run_id == workspace.run_id
    assert workspace.index_id != workspace.run_id


def test_prepare_files_reports_loader_failure_explicitly_for_binary_inputs(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    workspace = RunWorkspaceManager(settings).create_run()
    service = DocumentPreparationService(settings)
    monkeypatch.setattr(
        "agents.sales_lead_agent.services.document_preparation.load_single_document",
        lambda path: (_ for _ in ()).throw(RuntimeError("loader failed")),
    )

    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF-1.4 broken")

    prepared = service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id="bundle-2",
        registry_number="124",
        source_url="https://example.test/broken",
        file_paths=[str(source)],
    )

    assert len(prepared) == 1
    assert prepared[0].parse_status == "failed"
    assert prepared[0].index_status == "failed"
    assert prepared[0].error == "loader failed"


def test_prepare_files_preserves_original_vs_derived_provenance(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    workspace = RunWorkspaceManager(settings).create_run()
    service = DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)

    derived = tmp_path / "download_001.txt"
    derived.write_text("Some extracted attachment text", encoding="utf-8")

    prepared = service.prepare_files(
        workspace=workspace,
        origin="open_source",
        bundle_id="bundle-3",
        registry_number=None,
        source_url="https://example.test/download.pdf",
        file_paths=[str(derived)],
        provenance_by_path={
            str(derived): {
                "original_source_url": "https://example.test/download.pdf",
                "original_file_name": "download.pdf",
                "original_content_type": "application/pdf",
                "derived_artifact_path": str(derived),
            }
        },
    )

    assert len(prepared) == 1
    item = prepared[0]
    assert item.file_name == "download_001.txt"
    assert item.original_file_name == "download.pdf"
    assert item.original_content_type == "application/pdf"
    assert item.derived_artifact_path == str(derived)


def test_prepare_files_marks_parse_partial_when_no_indexable_content(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    workspace = RunWorkspaceManager(settings).create_run()
    service = DocumentPreparationService(settings)
    monkeypatch.setattr(service, "_index_documents", lambda **kwargs: None)
    monkeypatch.setattr(
        service,
        "_load_docs",
        lambda _file_path: [type("Doc", (), {"page_content": "   ", "metadata": {"source": "blank.txt"}})()],
    )

    source = tmp_path / "blank.txt"
    source.write_text("   ", encoding="utf-8")

    prepared = service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id="bundle-4",
        registry_number="125",
        source_url="https://example.test/blank",
        file_paths=[str(source)],
    )

    assert len(prepared) == 1
    assert prepared[0].parse_status == "partial"
    assert prepared[0].index_status == "failed"
    assert prepared[0].chunks_count == 0
    assert prepared[0].error == "No indexable content extracted."


def test_get_by_index_rejects_registry_mismatch(tmp_path):
    settings = _settings(tmp_path)
    manager = RunWorkspaceManager(settings)
    workspace = manager.create_run()

    registry_path = settings.work_root / "_index_registry" / f"{workspace.index_id}.txt"
    registry_path.write_text(workspace.run_id, encoding="utf-8")
    metadata_path = settings.work_root / workspace.run_id / "workspace.json"
    metadata_path.write_text(
        '{"run_id": "' + workspace.run_id + '", "index_id": "index_other"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Index registry mismatch"):
        manager.get_by_index(workspace.index_id)


def test_get_rejects_workspace_run_id_mismatch(tmp_path):
    settings = _settings(tmp_path)
    manager = RunWorkspaceManager(settings)
    workspace = manager.create_run()

    metadata_path = settings.work_root / workspace.run_id / "workspace.json"
    metadata_path.write_text(
        '{"run_id": "run_other", "index_id": "' + workspace.index_id + '"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Workspace metadata mismatch"):
        manager.get(workspace.run_id)


def test_create_embeddings_uses_settings_provider_and_model(tmp_path, monkeypatch):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        retention_hours=72,
        damia_api_key="",
        scoring_base_url="",
        fssp_base_url="",
        purchase_headless=True,
        open_source_max_concurrency=4,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    service = DocumentPreparationService(settings)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "agents.sales_lead_agent.services.document_preparation.create_embeddings_model",
        lambda *, provider, model_name: calls.append((provider, model_name)) or object(),
    )
    monkeypatch.setattr("agents.sales_lead_agent.services.document_preparation.config.OPENAI_API_KEY", "test-key")

    service._create_embeddings()

    assert calls == [("openai", "text-embedding-3-small")]


def test_create_embeddings_requires_openai_api_key(tmp_path, monkeypatch):
    settings = SalesLeadAgentSettings(
        work_root=tmp_path / "runs",
        retention_hours=72,
        damia_api_key="",
        scoring_base_url="",
        fssp_base_url="",
        purchase_headless=True,
        open_source_max_concurrency=4,
        procurement_search_template="https://zakupki.gov.ru/epz/order/extendedsearch/results.html?searchString=test",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
    )
    service = DocumentPreparationService(settings)

    monkeypatch.setattr("agents.sales_lead_agent.services.document_preparation.config.OPENAI_API_KEY", "   ")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        service._create_embeddings()


def test_get_settings_rejects_blank_embedding_provider(monkeypatch, tmp_path):
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("SALES_LEAD_AGENT_WORK_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("SALES_LEAD_AGENT_EMBEDDING_PROVIDER", "   ")
    monkeypatch.setattr("agents.sales_lead_agent.settings.config.OPENAI_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="SALES_LEAD_AGENT_EMBEDDING_PROVIDER must be configured"):
        settings_module.get_settings()

    settings_module.get_settings.cache_clear()


def test_get_settings_rejects_missing_openai_api_key(monkeypatch, tmp_path):
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("SALES_LEAD_AGENT_WORK_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("SALES_LEAD_AGENT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("SALES_LEAD_AGENT_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr("agents.sales_lead_agent.settings.config.OPENAI_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY must be configured"):
        settings_module.get_settings()

    settings_module.get_settings.cache_clear()


def test_index_documents_uses_configured_embeddings_for_vector_store_and_indexer(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    workspace = RunWorkspaceManager(settings).create_run()
    service = DocumentPreparationService(settings)
    embeddings_marker = object()
    vector_store_calls: list[object] = []
    indexer_calls: list[object] = []

    monkeypatch.setattr(service, "_create_embeddings", lambda: embeddings_marker)
    monkeypatch.setattr(
        "agents.sales_lead_agent.services.document_preparation.create_vector_store",
        lambda **kwargs: vector_store_calls.append(kwargs["embeddings"]) or object(),
    )

    class FakeIndexer:
        def __init__(self, *, vector_store, embeddings):
            indexer_calls.append(embeddings)

        def index(self, segments, batch_size):
            return None

    monkeypatch.setattr("agents.sales_lead_agent.services.document_preparation.Indexer", FakeIndexer)

    service._index_documents(
        workspace=workspace,
        segments=[],
    )
    assert vector_store_calls == []
    assert indexer_calls == []

    from rag_lib.core.domain import Segment

    service._index_documents(
        workspace=workspace,
        segments=[
            Segment(
                content="test",
                metadata={"source": "file.txt"},
                segment_id="seg-1",
                original_format="text",
            )
        ],
    )

    assert vector_store_calls == [embeddings_marker]
    assert indexer_calls == [embeddings_marker]


def test_search_uses_configured_embeddings_for_vector_store(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    workspace = RunWorkspaceManager(settings).create_run()
    service = DocumentPreparationService(settings)
    embeddings_marker = object()
    vector_store_calls: list[object] = []

    monkeypatch.setattr(service, "_create_embeddings", lambda: embeddings_marker)

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            return []

    monkeypatch.setattr(
        "agents.sales_lead_agent.services.document_preparation.create_vector_store",
        lambda **kwargs: vector_store_calls.append(kwargs["embeddings"]) or FakeVectorStore(),
    )

    response = service.search(
        workspace=workspace,
        query="insurance",
    )

    assert response.matches == []
    assert vector_store_calls == [embeddings_marker]
