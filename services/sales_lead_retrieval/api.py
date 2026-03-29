from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from agents.sales_lead_agent.tools import (
    ProcurementQueryBuilder,
    PurchaseAdapter,
    RunWorkspace,
    RunWorkspaceManager,
    ToolUserCorrectableError,
    get_settings as get_sales_lead_settings,
)

from .schemas import (
    RetrievalConflictResponse,
    RetrievalSnapshot,
    RetrievalUserInputErrorResponse,
    SubmitPurchaseRetrievalRequest,
)
from .store import ActiveRetrievalConflictError, SalesLeadRetrievalStore


router = APIRouter(prefix="/retrievals", tags=["retrievals"])
_store = SalesLeadRetrievalStore()
_sales_lead_settings = get_sales_lead_settings()
_workspace_manager = RunWorkspaceManager(_sales_lead_settings)
_purchase_adapter = PurchaseAdapter(
    settings=_sales_lead_settings,
    query_builder=ProcurementQueryBuilder(_sales_lead_settings.procurement_search_template),
)


def _retrieval_snapshot_paths(*, workspace: RunWorkspace, retrieval_id: str) -> tuple[str, str, str]:
    root = workspace.root_dir / "_retrieval" / retrieval_id
    root.mkdir(parents=True, exist_ok=True)
    return (
        str(root / "items.json"),
        str(root / "prepared_documents.json"),
        str(root / "summary.json"),
    )


@router.post("/purchase-search", response_model=RetrievalSnapshot)
async def submit_purchase_search(payload: SubmitPurchaseRetrievalRequest):
    try:
        request_spec = _purchase_adapter.build_request_spec(
            search_url=payload.search_url,
            query_texts=payload.query_texts,
            max_pages=payload.max_pages,
        )
    except ToolUserCorrectableError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=RetrievalUserInputErrorResponse(
                code=exc.code,
                message=str(exc),
                suggestion=exc.suggestion,
                input_field=exc.input_field,
            ).model_dump(),
        )
    lookup = await _store.lookup_submission(
        conversation_id=payload.conversation_id,
        request_hash=request_spec.request_hash,
        include_payloads=True,
    )
    active_snapshot = lookup.active
    if active_snapshot is not None and active_snapshot.request_hash != request_spec.request_hash:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=RetrievalConflictResponse(
                message=(
                    "Another procurement retrieval is already running for this conversation. "
                    "Reuse the active retrieval instead of starting a new one."
                ),
                active_snapshot=active_snapshot,
            ).model_dump(),
        )

    snapshot = lookup.matching
    if snapshot is None:
        workspace = (
            _workspace_manager.get(payload.requested_run_id)
            if payload.requested_run_id
            else _workspace_manager.create_run()
        )
        retrieval_id = str(uuid.uuid4())
        items_path, documents_path, summary_path = _retrieval_snapshot_paths(
            workspace=workspace,
            retrieval_id=retrieval_id,
        )
        try:
            snapshot = await _store.create_job(
                retrieval_id=retrieval_id,
                conversation_id=payload.conversation_id,
                agent_id=payload.agent_id,
                request_hash=request_spec.request_hash,
                request_payload=request_spec.request_payload,
                run_id=workspace.run_id,
                index_id=_sales_lead_settings.shared_index_id,
                items_snapshot_path=items_path,
                documents_snapshot_path=documents_path,
                summary_snapshot_path=summary_path,
                message=(
                    f"Procurement retrieval queued for {len(request_spec.search_urls)} "
                    "search target(s)."
                ),
            )
        except ActiveRetrievalConflictError as exc:
            if exc.snapshot.request_hash == request_spec.request_hash:
                snapshot = exc.snapshot
            else:
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT,
                    content=RetrievalConflictResponse(
                        message=str(exc),
                        active_snapshot=exc.snapshot,
                    ).model_dump(),
                )

    return snapshot


@router.get("/conversations/{conversation_id}/latest", response_model=RetrievalSnapshot | None)
async def get_latest_for_conversation(conversation_id: str, include_payloads: bool = False):
    return await _store.get_latest_for_conversation(
        conversation_id=conversation_id,
        include_payloads=include_payloads,
    )


@router.get("/{retrieval_id}", response_model=RetrievalSnapshot | None)
async def get_retrieval(retrieval_id: str, include_payloads: bool = False):
    return await _store.get_retrieval(
        retrieval_id=retrieval_id,
        include_payloads=include_payloads,
    )


@router.post("/{retrieval_id}/announced")
async def mark_announced(retrieval_id: str):
    try:
        await _store.mark_announced(retrieval_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"status": "ok"}
