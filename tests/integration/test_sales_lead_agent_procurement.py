from dataclasses import replace
import os

import pytest

from agents.sales_lead_agent.schemas import SearchFilters
from agents.sales_lead_agent.services.document_preparation import (
    DocumentPreparationService,
    RunWorkspaceManager,
)
from agents.sales_lead_agent.services.purchase_adapter import PurchaseAdapter
from agents.sales_lead_agent.services.query_builder import ProcurementQueryBuilder
from agents.sales_lead_agent.settings import get_settings


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_PROCUREMENT_SMOKE") != "1",
    reason="Real procurement smoke is disabled. Set RUN_REAL_PROCUREMENT_SMOKE=1 to enable.",
)


def test_real_procurement_smoke():
    settings = get_settings()
    adapter = PurchaseAdapter(
        settings=settings,
        query_builder=ProcurementQueryBuilder(settings),
    )
    _, response = adapter.search(
        search_url=None,
        search_filters=SearchFilters(query_text="страхование имущества"),
        downloads_dir=str(settings.work_root / "integration-smoke"),
        max_pages=1,
        headless=True,
    )

    assert response.status in {"success", "partial"}


def test_real_procurement_preparation_and_doc_search_smoke(tmp_path):
    base_settings = get_settings()
    settings = replace(base_settings, work_root=tmp_path / "integration-smoke-runs")
    settings.work_root.mkdir(parents=True, exist_ok=True)

    adapter = PurchaseAdapter(
        settings=settings,
        query_builder=ProcurementQueryBuilder(settings),
    )
    workspace_manager = RunWorkspaceManager(settings)
    workspace = workspace_manager.create_run()
    document_service = DocumentPreparationService(settings)

    _, response = adapter.search(
        search_url=None,
        search_filters=SearchFilters(query_text="страхование имущества"),
        downloads_dir=str(workspace.downloads_dir),
        max_pages=1,
        headless=True,
    )

    assert response.status in {"success", "partial"}
    assert response.items

    item_with_files = next((item for item in response.items if item.downloaded_files), None)
    if item_with_files is None:
        pytest.skip("Live procurement smoke returned no downloaded files to prepare.")

    prepared = document_service.prepare_files(
        workspace=workspace,
        origin="purchase",
        bundle_id=item_with_files.bundle_id,
        registry_number=item_with_files.registry_number,
        source_url=item_with_files.detail_url,
        file_paths=item_with_files.downloaded_files[:2],
    )

    if not prepared:
        pytest.skip("Live procurement files did not produce prepared documents.")

    searchable = [doc for doc in prepared if doc.index_status == "ready" and doc.chunks_count > 0]
    if not searchable:
        pytest.skip("Live procurement files did not produce a searchable prepared corpus.")

    result = document_service.search(
        workspace=workspace,
        query="страхование",
        top_k=1,
        source_kind="purchase",
        bundle_id=item_with_files.bundle_id,
    )

    assert workspace_manager.get_by_index(workspace.index_id).run_id == workspace.run_id
    assert result.index_id == workspace.index_id
    assert result.matches
