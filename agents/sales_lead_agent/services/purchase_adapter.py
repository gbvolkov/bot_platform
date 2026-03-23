from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from ..schemas import PurchaseSearchItem, PurchaseSearchResponse, SearchFilters
from ..settings import SalesLeadAgentSettings
from .query_builder import ProcurementQueryBuilder


class PurchaseAdapter:
    def __init__(
        self,
        *,
        settings: SalesLeadAgentSettings,
        query_builder: ProcurementQueryBuilder,
    ) -> None:
        self._settings = settings
        self._query_builder = query_builder

    def _import_scraper(self):
        try:
            from zakupki_crawler.api import scrape_purchases
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "The `zakupki_crawler` package is not installed in `.venv`."
            ) from exc
        return scrape_purchases

    def resolve_search_url(
        self,
        *,
        search_url: str | None,
        search_filters: SearchFilters | None,
    ) -> str:
        if search_url:
            return search_url
        if not search_filters:
            raise ValueError("Either `search_url` or `search_filters` must be provided.")
        return self._query_builder.build_url(search_filters)

    def search(
        self,
        *,
        search_url: str | None,
        search_filters: SearchFilters | None,
        downloads_dir: str,
        max_pages: int | None,
        headless: bool | None,
    ) -> tuple[str, PurchaseSearchResponse]:
        resolved_search_url = self.resolve_search_url(
            search_url=search_url,
            search_filters=search_filters,
        )
        scrape_purchases = self._import_scraper()
        try:
            raw_items = self._run_scraper(
                scrape_purchases,
                resolved_search_url=resolved_search_url,
                downloads_dir=downloads_dir,
                max_pages=max_pages,
                headless=self._settings.purchase_headless if headless is None else headless,
            )
        except Exception as exc:
            return resolved_search_url, PurchaseSearchResponse(
                run_id="",
                index_id="",
                status="failed",
                errors=[str(exc)],
                items=[],
                prepared_documents=[],
            )

        items: list[PurchaseSearchItem] = []
        for raw in raw_items:
            items.append(
                PurchaseSearchItem(
                    bundle_id=raw.registry_number,
                    registry_number=raw.registry_number,
                    law=raw.law if raw.law in {"44-FZ", "223-FZ"} else None,
                    purchase_title=raw.purchase_title or "",
                    customer_name=raw.customer_name or "",
                    price_text=raw.price_text,
                    published_at=raw.published_at,
                    updated_at=raw.updated_at,
                    submission_deadline=raw.submission_deadline,
                    detail_url=raw.detail_url,
                    common_info_url=raw.common_info_url,
                    documents_url=raw.documents_url,
                    document_urls=[value for value in (raw.document_urls or "").splitlines() if value],
                    downloaded_files=[value for value in (raw.downloaded_files or "").splitlines() if value],
                    prepared_document_ids=[],
                    documents_json=raw.documents_json or None,
                    common_info_json=raw.common_info_json or None,
                    lots_json=raw.lots_json or None,
                    crawl_status=raw.crawl_status if raw.crawl_status in {"success", "partial", "failed"} else "failed",
                    crawl_error=raw.crawl_error or None,
                    crawl_ts_utc=raw.crawl_ts_utc
                    or datetime.now(timezone.utc).isoformat(),
                )
            )

        status = "success" if items else "partial"
        return resolved_search_url, PurchaseSearchResponse(
            run_id="",
            index_id="",
            status=status,
            errors=[],
            items=items,
            prepared_documents=[],
        )

    def _run_scraper(
        self,
        scrape_purchases,
        *,
        resolved_search_url: str,
        downloads_dir: str,
        max_pages: int | None,
        headless: bool,
    ):
        kwargs = {
            "downloads_dir": downloads_dir,
            "max_pages": max_pages,
            "headless": headless,
        }
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return scrape_purchases(
                resolved_search_url,
                **kwargs,
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                scrape_purchases,
                resolved_search_url,
                **kwargs,
            )
            return future.result()

    def summarize_hits(self, response: PurchaseSearchResponse) -> list[dict[str, Any]]:
        return [
            {
                "bundle_id": item.bundle_id,
                "registry_number": item.registry_number,
                "purchase_title": item.purchase_title,
                "customer_name": item.customer_name,
                "law": item.law,
                "price_text": item.price_text,
            }
            for item in response.items
        ]
