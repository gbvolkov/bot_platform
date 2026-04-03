from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from agents.sales_lead_agent.tools import (
    PreparedDocument,
    PurchaseSearchItem,
    SalesLeadAgentDependencies,
    _ensure_purchase_artifacts,
    get_settings as get_sales_lead_settings,
)

from .config import settings as worker_settings
from .db import init_models
from .schemas import RetrievalSnapshot
from .store import SalesLeadRetrievalStore


logger = logging.getLogger("sales_lead_retrieval.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _initial_progress() -> dict[str, int]:
    return {
        "total_queries": 0,
        "completed_queries": 0,
        "total_purchases": 0,
        "processed_purchases": 0,
        "total_files": 0,
        "processed_files": 0,
        "prepared_documents": 0,
        "indexed_segments": 0,
    }


async def _record(
    store: SalesLeadRetrievalStore,
    *,
    retrieval_id: str,
    stage: str,
    message: str,
    progress: dict[str, Any],
    items: list[dict[str, Any]] | None = None,
    prepared_documents: list[dict[str, Any]] | None = None,
    status: str | None = None,
    error_text: str | None = None,
    clear_active: bool = False,
    finished: bool = False,
    level: str = "info",
    payload: dict[str, Any] | None = None,
) -> None:
    if level == "error":
        logger.error("retrieval=%s stage=%s %s", retrieval_id, stage, message)
    else:
        logger.info("retrieval=%s stage=%s %s", retrieval_id, stage, message)
    await store.append_event(
        retrieval_id,
        stage=stage,
        message=message,
        level=level,
        payload=payload,
    )
    await store.update_job(
        retrieval_id,
        status=status,  # type: ignore[arg-type]
        stage=stage,
        message=message,
        progress=progress,
        items=items,
        prepared_documents=prepared_documents,
        error_text=error_text,
        clear_active=clear_active,
        finished=finished,
    )


async def _process_job(
    *,
    job: RetrievalSnapshot,
    store: SalesLeadRetrievalStore,
    deps: SalesLeadAgentDependencies,
) -> None:
    loop = asyncio.get_running_loop()
    workspace = deps.workspace_manager.get(job.run_id)
    progress = _initial_progress()
    items_payload: list[dict[str, Any]] = []
    items_by_registry: dict[str, dict[str, Any]] = {}
    prepared_payload: list[dict[str, Any]] = []
    request_payload = dict(job.request_payload or {})
    processed_registries: set[str] = set()

    def _sync_record(
        *,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
        progress_update: dict[str, Any] | None = None,
        level: str = "info",
    ) -> None:
        if progress_update:
            progress.update(progress_update)
        future = asyncio.run_coroutine_threadsafe(
            _record(
                store,
                retrieval_id=job.retrieval_id,
                stage=stage,
                message=message,
                progress=dict(progress),
                items=list(items_payload),
                prepared_documents=list(prepared_payload),
                level=level,
                payload=payload,
            ),
            loop,
        )
        future.result()

    def _crawler_progress(**event: Any) -> None:
        stage = str(event.get("stage") or "crawler")
        message = str(event.get("message") or "")
        progress_update: dict[str, Any] = {}
        if stage == "crawler_plan":
            progress_update["total_queries"] = int(event.get("total_queries", progress["total_queries"]))
        elif stage in {"crawler_search_done", "crawler_search_failed"}:
            progress_update["completed_queries"] = int(
                event.get("current", progress["completed_queries"])
            )
        elif stage == "crawler_complete":
            progress_update["total_purchases"] = max(
                int(event.get("unique_procurements", progress["total_purchases"])),
                len(items_payload),
            )
            progress_update["completed_queries"] = progress["total_queries"]
        _sync_record(
            stage=stage,
            message=message,
            payload=event,
            progress_update=progress_update,
            level="error" if stage.endswith("_failed") else "info",
        )

    def _store_item_payload(item: PurchaseSearchItem) -> None:
        items_by_registry[item.registry_number] = item.model_dump()
        items_payload.clear()
        items_payload.extend(items_by_registry.values())

    def _item_discovered(item: PurchaseSearchItem) -> None:
        _store_item_payload(item)
        _sync_record(
            stage="crawler_item",
            message=f"Discovered procurement {item.registry_number}.",
            payload={"purchase_id": item.registry_number},
            progress_update={"total_purchases": len(items_payload)},
        )

    async def _refresh_item_summary(item: PurchaseSearchItem) -> PurchaseSearchItem:
        purchase_document_summary = getattr(deps.document_service, "purchase_document_summary", None)
        if not callable(purchase_document_summary):
            return item
        try:
            summary = await asyncio.to_thread(purchase_document_summary, item.registry_number)
        except Exception as exc:
            await _record(
                store,
                retrieval_id=job.retrieval_id,
                stage="purchase_summary_failed",
                message=(
                    f"Failed to refresh indexed summary for procurement "
                    f"{item.registry_number}: {exc}"
                ),
                progress=progress,
                items=items_payload,
                prepared_documents=prepared_payload,
                status="in_progress",
                level="error",
                payload={
                    "purchase_id": item.registry_number,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            return item
        prepared_document_ids = summary.get("prepared_document_ids")
        if isinstance(prepared_document_ids, list):
            item.prepared_document_ids = [str(value) for value in prepared_document_ids if str(value)]
        indexed_documents_count = summary.get("indexed_documents_count")
        if indexed_documents_count is not None:
            item.indexed_documents_count = int(indexed_documents_count)
        last_indexed_at = summary.get("last_indexed_at")
        if last_indexed_at:
            item.last_indexed_at = str(last_indexed_at)
        source_url = summary.get("source_url")
        if source_url and not item.detail_url:
            item.detail_url = str(source_url)
        return item

    async def _upsert_catalog_item(item: PurchaseSearchItem) -> PurchaseSearchItem:
        procurement_catalog = getattr(deps, "procurement_catalog", None)
        if procurement_catalog is None:
            _store_item_payload(item)
            return item
        try:
            updated_item = await asyncio.to_thread(
                procurement_catalog.upsert_item,
                item.model_copy(deep=True),
            )
        except Exception as exc:
            _store_item_payload(item)
            await _record(
                store,
                retrieval_id=job.retrieval_id,
                stage="catalog_upsert_failed",
                message=(
                    f"Failed to update procurement catalog for "
                    f"{item.registry_number}: {exc}"
                ),
                progress=progress,
                items=items_payload,
                prepared_documents=prepared_payload,
                status="in_progress",
                level="error",
                payload={
                    "purchase_id": item.registry_number,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            return item
        _store_item_payload(updated_item)
        return updated_item

    async def _process_purchase_item(
        *,
        item: PurchaseSearchItem,
        current_purchase: int,
        total_purchases: int,
    ) -> None:
        await _record(
            store,
            retrieval_id=job.retrieval_id,
            stage="purchase_processing",
            message=(
                f"Processing procurement {item.registry_number} "
                f"[{current_purchase}/{total_purchases}]."
            ),
            progress=progress,
            items=items_payload,
            prepared_documents=prepared_payload,
            status="in_progress",
            payload={
                "purchase_id": item.registry_number,
                "current_purchase": current_purchase,
                "total_purchases": total_purchases,
            },
        )
        item = await _refresh_item_summary(item)
        item = await _upsert_catalog_item(item)
        try:
            artifact_paths = _ensure_purchase_artifacts(workspace=workspace, item=item.model_dump())
            progress["total_files"] += len(artifact_paths)
            if artifact_paths:
                await _record(
                    store,
                    retrieval_id=job.retrieval_id,
                    stage="artifacts_ready",
                    message=(
                        f"Collected {len(artifact_paths)} artifact file(s) for "
                        f"procurement {item.registry_number}."
                    ),
                    progress=progress,
                    items=items_payload,
                    prepared_documents=prepared_payload,
                    status="in_progress",
                    payload={
                        "purchase_id": item.registry_number,
                        "total_files": len(artifact_paths),
                    },
                )
                provenance_by_path = {
                    path: {
                        "original_source_url": item.detail_url,
                        "original_file_name": Path(path).name,
                        "derived_artifact_path": path,
                    }
                    for path in artifact_paths
                }

                def _prepare_progress(**event: Any) -> None:
                    stage = str(event.get("stage") or "preparing")
                    _sync_record(
                        stage=stage,
                        message=str(event.get("message") or ""),
                        payload={
                            "purchase_id": item.registry_number,
                            **event,
                        },
                        level="error" if stage.endswith("_failed") else "info",
                    )

                prepared = await asyncio.to_thread(
                    deps.document_service.prepare_files,
                    workspace=workspace,
                    origin="purchase",
                    bundle_id=item.bundle_id,
                    registry_number=item.registry_number,
                    source_url=item.detail_url,
                    file_paths=artifact_paths,
                    provenance_by_path=provenance_by_path,
                    progress_callback=_prepare_progress,
                )
                if prepared:
                    item.prepared_document_ids = [doc.document_id for doc in prepared]
                    prepared_payload.extend(doc.model_dump() for doc in prepared)
                    progress["prepared_documents"] = len(prepared_payload)
                    progress["indexed_segments"] += sum(int(doc.chunks_count) for doc in prepared)
                progress["processed_files"] += len(artifact_paths)
            else:
                await _record(
                    store,
                    retrieval_id=job.retrieval_id,
                    stage="artifacts_missing",
                    message=(
                        f"No artifact files were available for procurement "
                        f"{item.registry_number}."
                    ),
                    progress=progress,
                    items=items_payload,
                    prepared_documents=prepared_payload,
                    status="in_progress",
                    payload={"purchase_id": item.registry_number},
                )
        except Exception as exc:
            await _record(
                store,
                retrieval_id=job.retrieval_id,
                stage="purchase_failed",
                message=(
                    f"Failed to process procurement {item.registry_number}: {exc}"
                ),
                progress=progress,
                items=items_payload,
                prepared_documents=prepared_payload,
                status="in_progress",
                level="error",
                payload={
                    "purchase_id": item.registry_number,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
        item = await _refresh_item_summary(item)
        item = await _upsert_catalog_item(item)
        processed_registries.add(item.registry_number)
        progress["processed_purchases"] = len(processed_registries)
        await _record(
            store,
            retrieval_id=job.retrieval_id,
            stage="purchase_processed",
            message=(
                f"Processed procurement {item.registry_number} "
                f"[{progress['processed_purchases']}/{max(total_purchases, len(items_payload))}]."
            ),
            progress=progress,
            items=items_payload,
            prepared_documents=prepared_payload,
            status="in_progress",
            payload={"purchase_id": item.registry_number},
        )

    async def _process_discovered_batch(
        batch_items: list[PurchaseSearchItem],
        batch_meta: dict[str, Any],
    ) -> None:
        if not batch_items:
            return
        await _record(
            store,
            retrieval_id=job.retrieval_id,
            stage="crawler_query_batch",
            message=(
                f"Prepared {len(batch_items)} new procurement record(s) "
                f"from search [{batch_meta.get('current')}/{batch_meta.get('total')}]."
            ),
            progress=progress,
            items=items_payload,
            prepared_documents=prepared_payload,
            status="in_progress",
            payload=batch_meta,
        )
        for batch_offset, batch_item in enumerate(batch_items, start=1):
            try:
                current_purchase = len(processed_registries) + 1
                total_purchases = max(len(items_payload), current_purchase)
                await _process_purchase_item(
                    item=batch_item,
                    current_purchase=current_purchase,
                    total_purchases=total_purchases,
                )
            except Exception as exc:
                await _record(
                    store,
                    retrieval_id=job.retrieval_id,
                    stage="purchase_failed",
                    message=(
                        f"Unexpected batch processing failure for procurement "
                        f"{batch_item.registry_number}: {exc}"
                    ),
                    progress=progress,
                    items=items_payload,
                    prepared_documents=prepared_payload,
                    status="in_progress",
                    level="error",
                    payload={
                        "purchase_id": batch_item.registry_number,
                        "batch_offset": batch_offset,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        **batch_meta,
                    },
                )
                processed_registries.add(batch_item.registry_number)
                progress["processed_purchases"] = len(processed_registries)

    def _batch_discovered(batch_items: list[PurchaseSearchItem], batch_meta: dict[str, Any]) -> None:
        future = asyncio.run_coroutine_threadsafe(
            _process_discovered_batch(batch_items, dict(batch_meta)),
            loop,
        )
        try:
            future.result()
        except Exception as exc:
            _sync_record(
                stage="crawler_batch_failed",
                message=f"Failed to process crawler batch: {exc}",
                payload={
                    **batch_meta,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
                level="error",
            )

    await _record(
        store,
        retrieval_id=job.retrieval_id,
        stage="starting",
        message="Background procurement retrieval started.",
        progress=progress,
        items=items_payload,
        prepared_documents=prepared_payload,
        status="in_progress",
    )

    try:
        resolved_urls, items = await asyncio.to_thread(
            deps.purchase_adapter.search,
            search_url=request_payload.get("search_url"),
            query_texts=request_payload.get("query_texts"),
            downloads_dir=str(workspace.downloads_dir),
            max_pages=request_payload.get("max_pages"),
            progress_callback=_crawler_progress,
            item_callback=_item_discovered,
            batch_callback=_batch_discovered,
        )
        for item in items:
            if item.registry_number not in items_by_registry:
                _store_item_payload(item)
        progress["total_queries"] = max(progress["total_queries"], len(resolved_urls))
        progress["completed_queries"] = max(progress["completed_queries"], len(resolved_urls))
        progress["total_purchases"] = max(progress["total_purchases"], len(items_payload))
        for item in items:
            if item.registry_number in processed_registries:
                continue
            await _process_purchase_item(
                item=item,
                current_purchase=len(processed_registries) + 1,
                total_purchases=max(len(items_payload), len(items)),
            )

        await _record(
            store,
            retrieval_id=job.retrieval_id,
            stage="completed",
            message=(
                f"Procurement retrieval finished. Collected {len(items_payload)} "
                f"procurement record(s) and prepared {len(prepared_payload)} document(s)."
            ),
            progress=progress,
            items=items_payload,
            prepared_documents=prepared_payload,
            status="completed",
            clear_active=True,
            finished=True,
        )
    except Exception as exc:
        await _record(
            store,
            retrieval_id=job.retrieval_id,
            stage="failed",
            message=f"Procurement retrieval failed: {exc}",
            progress=progress,
            items=items_payload,
            prepared_documents=prepared_payload,
            status="failed",
            error_text=str(exc),
            clear_active=True,
            finished=True,
            level="error",
            payload={"error_type": exc.__class__.__name__},
        )


async def run_worker() -> None:
    logger.info("sales_lead_retrieval worker starting")
    await init_models()
    store = SalesLeadRetrievalStore()
    deps = SalesLeadAgentDependencies.from_settings(get_sales_lead_settings())
    logger.info(
        "sales_lead_retrieval worker started poll_interval_seconds=%s",
        worker_settings.poll_interval_seconds,
    )
    while True:
        job = await store.claim_next_queued_job()
        if job is None:
            await asyncio.sleep(worker_settings.poll_interval_seconds)
            continue
        logger.info(
            "sales_lead_retrieval worker picked retrieval_id=%s run_id=%s",
            job.retrieval_id,
            job.run_id,
        )
        await _process_job(job=job, store=store, deps=deps)


def main() -> None:
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_worker())
    finally:
        logger.info("sales_lead_retrieval worker finished")


if __name__ == "__main__":
    main()
