import asyncio
from pathlib import Path

from agents.sales_lead_agent.schemas import SearchFilters
from agents.sales_lead_agent.services.purchase_adapter import PurchaseAdapter
from agents.sales_lead_agent.services.query_builder import ProcurementQueryBuilder
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


def test_resolve_search_url_prefers_direct_url(tmp_path):
    adapter = PurchaseAdapter(
        settings=_settings(tmp_path),
        query_builder=ProcurementQueryBuilder(_settings(tmp_path)),
    )

    url = adapter.resolve_search_url(
        search_url="https://example.test/direct",
        search_filters=SearchFilters(query_text="страхование"),
    )

    assert url == "https://example.test/direct"


def test_search_uses_scraper_and_normalizes_records(tmp_path, monkeypatch):
    adapter = PurchaseAdapter(
        settings=_settings(tmp_path),
        query_builder=ProcurementQueryBuilder(_settings(tmp_path)),
    )

    class RawRecord:
        registry_number = "123"
        law = "44-FZ"
        purchase_title = "Страхование имущества"
        customer_name = "ООО Ромашка"
        price_text = "120 000 руб."
        published_at = "2026-03-20"
        updated_at = "2026-03-21"
        submission_deadline = "2026-03-30"
        detail_url = "https://example.test/purchase/123"
        common_info_url = None
        documents_url = None
        document_urls = "https://example.test/doc1.pdf"
        downloaded_files = "C:/tmp/doc1.pdf"
        documents_json = "{}"
        common_info_json = "{}"
        lots_json = "{}"
        crawl_status = "success"
        crawl_error = ""
        crawl_ts_utc = "2026-03-22T00:00:00Z"

    monkeypatch.setattr(adapter, "_import_scraper", lambda: (lambda *args, **kwargs: [RawRecord()]))

    resolved_url, response = adapter.search(
        search_url=None,
        search_filters=SearchFilters(query_text="страхование имущества"),
        downloads_dir=str(tmp_path),
        max_pages=1,
        headless=True,
    )

    assert "searchString=" in resolved_url
    assert response.status == "success"
    assert len(response.items) == 1
    assert response.items[0].registry_number == "123"
    assert response.items[0].document_urls == ["https://example.test/doc1.pdf"]


def test_search_runs_scraper_outside_active_event_loop(tmp_path, monkeypatch):
    adapter = PurchaseAdapter(
        settings=_settings(tmp_path),
        query_builder=ProcurementQueryBuilder(_settings(tmp_path)),
    )

    class RawRecord:
        registry_number = "123"
        law = "44-FZ"
        purchase_title = "Insurance"
        customer_name = "Acme"
        price_text = None
        published_at = None
        updated_at = None
        submission_deadline = None
        detail_url = "https://example.test/purchase/123"
        common_info_url = None
        documents_url = None
        document_urls = ""
        downloaded_files = ""
        documents_json = None
        common_info_json = None
        lots_json = None
        crawl_status = "success"
        crawl_error = ""
        crawl_ts_utc = "2026-03-22T00:00:00Z"

    def fake_scraper(*args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return [RawRecord()]
        raise AssertionError("scraper should not run inside the active event loop")

    monkeypatch.setattr(adapter, "_import_scraper", lambda: fake_scraper)

    async def run_search():
        return adapter.search(
            search_url=None,
            search_filters=SearchFilters(query_text="insurance"),
            downloads_dir=str(tmp_path),
            max_pages=1,
            headless=True,
        )

    _, response = asyncio.run(run_search())

    assert response.status == "success"
    assert response.items[0].registry_number == "123"
