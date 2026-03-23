from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import ValidationError
from rag_lib.loaders.web_async import AsyncWebLoader

from .schemas import (
    CounterpartyFSSPRequest,
    CounterpartyScoringRequest,
    DocSearchRequest,
    EvidenceItem,
    OpenSourceFetchRequest,
    OpenSourceFetchResponse,
    OpenSourcePage,
    PurchaseSearchRequest,
    PurchaseSearchResponse,
    SearchFilters,
    TurnValidationIssue,
    TurnValidationResult,
)
from .services import (
    ClassifierExecutionError,
    CounterpartyClients,
    DocumentPreparationService,
    InternalClassifier,
    PurchaseAdapter,
    RunWorkspaceManager,
)
from .state import SalesLeadAgentState


_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class RecoverableToolInputError(ValueError):
    """Recoverable tool-input validation error caused by model arguments."""


@dataclass(frozen=True)
class SalesLeadAgentDependencies:
    workspace_manager: RunWorkspaceManager
    document_service: DocumentPreparationService
    classifier: InternalClassifier
    purchase_adapter: PurchaseAdapter
    counterparty_clients: CounterpartyClients
    open_source_max_concurrency: int


def _tool_message(runtime: ToolRuntime | None, payload: Any, *, status: str = "success") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        tool_call_id=getattr(runtime, "tool_call_id", None) or "",
        status=status,
    )


def _merge_dict_list(existing: list[dict[str, Any]] | None, additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing or [])
    seen = {json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in merged}
    for item in additions:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _merge_strings(existing: list[str] | None, additions: list[str]) -> list[str]:
    merged = list(existing or [])
    for item in additions:
        if item and item not in merged:
            merged.append(item)
    return merged


def _append_tool_usage(
    existing: list[dict[str, Any]] | None,
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    return [*(existing or []), entry]


def _append_acquisition_attempt(
    existing: list[dict[str, Any]] | None,
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    return [*(existing or []), entry]


def _searchable_document_count(documents: list[Any]) -> int:
    count = 0
    for document in documents:
        if isinstance(document, dict):
            index_status = document.get("index_status")
            chunks_count = int(document.get("chunks_count") or 0)
        else:
            index_status = getattr(document, "index_status", None)
            chunks_count = int(getattr(document, "chunks_count", 0) or 0)
        if index_status == "ready" and chunks_count > 0:
            count += 1
    return count


def _active_context_ready(state: SalesLeadAgentState) -> bool:
    return bool(state.get("active_run_id") and state.get("active_run_ready"))


def _has_ready_active_context(state: SalesLeadAgentState) -> bool:
    if not _active_context_ready(state):
        return False
    return _searchable_document_count(state.get("prepared_documents") or []) > 0


def _acquisition_error_payload(
    *,
    tool: str,
    status: str,
    error: str,
    run_id: str | None = None,
    index_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool": tool,
        "status": status,
        "error": error,
        "run_id": run_id,
        "index_id": index_id,
    }
    if context:
        payload.update(context)
    return payload


def _active_context_update(
    state: SalesLeadAgentState,
    *,
    workspace: Any,
    current_query_signature: str | None,
    active_run_ready: bool,
    acquisition_status: str | None,
) -> dict[str, Any]:
    if active_run_ready and acquisition_status in {"success", "partial"}:
        return {
            "active_run_id": workspace.run_id,
            "active_run_query_signature": current_query_signature,
            "current_run_id": workspace.run_id,
            "index_id": workspace.index_id,
            "active_run_ready": True,
        }
    return {
        "current_run_id": workspace.run_id,
        "active_run_ready": _active_context_ready(state),
    }


def _with_issue(
    current: dict[str, Any] | None,
    issue: TurnValidationIssue,
) -> dict[str, Any]:
    validation = TurnValidationResult.model_validate(current or {})
    issues = list(validation.issues)
    issues.append(issue)
    status = "failed_verification" if any(item.severity == "error" for item in issues) else "partial"
    return TurnValidationResult(
        status=status,
        issues=issues,
        manual_review_required=True,
    ).model_dump()


def _safe_artifact_name(name: str, *, default_stem: str) -> str:
    base = _SAFE_FILE_RE.sub("_", name.strip()) or default_stem
    if not base.lower().endswith(".txt"):
        base = f"{base}.txt"
    return base


def _search_filters_from_args(**kwargs: Any) -> SearchFilters | None:
    payload = {
        "query_text": kwargs.get("query_text"),
        "law": kwargs.get("law"),
        "region": kwargs.get("region"),
        "min_price": kwargs.get("min_price"),
        "max_price": kwargs.get("max_price"),
        "published_from": kwargs.get("published_from"),
        "published_to": kwargs.get("published_to"),
        "submission_deadline_from": kwargs.get("submission_deadline_from"),
        "submission_deadline_to": kwargs.get("submission_deadline_to"),
        "customer_name": kwargs.get("customer_name"),
        "customer_inn": kwargs.get("customer_inn"),
        "supplier_hint": kwargs.get("supplier_hint"),
    }
    compact = {key: value for key, value in payload.items() if value not in (None, "", [])}
    if not compact:
        return None
    try:
        return SearchFilters.model_validate(compact)
    except ValidationError as exc:
        raise RecoverableToolInputError("Invalid procurement search filters.") from exc


def _search_filters_payload(filters: SearchFilters | None) -> dict[str, Any] | None:
    if filters is None:
        return None
    return filters.model_dump(exclude_none=True)


def _evidence_from_doc_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for match in matches:
        evidence.append(
            {
                "source": "document",
                "source_url": match.get("source_url"),
                "file_path": match.get("file_path"),
                "page": match.get("page"),
                "locator": match.get("locator"),
                "snippet": match.get("snippet") or "",
                "document_id": match.get("document_id"),
                "bundle_id": match.get("bundle_id"),
            }
        )
    return evidence


def _ensure_purchase_artifacts(*, workspace, item: dict[str, Any]) -> list[str]:
    artifact_paths: list[str] = list(item.get("downloaded_files") or [])
    bundle_dir = workspace.artifacts_dir / item["bundle_id"]
    bundle_dir.mkdir(parents=True, exist_ok=True)
    json_fields = {
        "common_info_json": item.get("common_info_json"),
        "documents_json": item.get("documents_json"),
        "lots_json": item.get("lots_json"),
    }
    for name, payload in json_fields.items():
        if not payload:
            continue
        path = bundle_dir / f"{name}.json"
        path.write_text(str(payload), encoding="utf-8")
        artifact_paths.append(str(path))
    return artifact_paths


def _resolve_purchase_run_id(
    state: SalesLeadAgentState,
    explicit_run_id: str | None,
    *,
    current_search_url: str | None = None,
    current_search_filters: SearchFilters | dict[str, Any] | None = None,
) -> str | None:
    if explicit_run_id:
        return explicit_run_id
    active_run_id = state.get("active_run_id")
    if not active_run_id:
        return None
    current_signature = state.get("current_query_signature")
    active_signature = state.get("active_run_query_signature")
    if current_signature and current_signature == active_signature:
        return active_run_id

    understanding = state.get("task_understanding") or {}
    task_kind = understanding.get("task_kind")
    has_new_acquisition_context = bool(
        current_search_url
        or current_search_filters
        or state.get("search_url")
        or state.get("search_filters")
    )
    if (
        task_kind in {"fact_lookup", "procurement_analysis"}
        and _has_ready_active_context(state)
        and not has_new_acquisition_context
    ):
        return active_run_id
    return None


def _inn_guard_failure(
    *,
    runtime: ToolRuntime[None, SalesLeadAgentState],
    inn: str,
    tool_name: str,
    source: str,
) -> Command:
    payload = {
        "source": source,
        "status": "failed",
        "error": f"INN {inn} is not available in normalized state. Acquire or classify it first.",
        "inn": inn,
    }
    return Command(
        update={
            "turn_validation": _with_issue(
                runtime.state.get("turn_validation"),
                TurnValidationIssue(
                    stage=tool_name,
                    code=f"{tool_name}_guard_failed",
                    message=f"INN {inn} is not available in normalized state. Acquire or classify it first.",
                    metadata={"inn": inn},
                ),
            ),
            "missing_data": _merge_strings(runtime.state.get("missing_data"), [f"normalized_inn:{inn}"]),
            "turn_tool_usage": _append_tool_usage(
                runtime.state.get("turn_tool_usage"),
                {
                    "tool": tool_name,
                    "status": "failed",
                    "inn": inn,
                    "guard": "normalized_inn_missing",
                },
            ),
            "messages": [_tool_message(runtime, payload, status="error")],
        }
    )


def _tool_validation_failure(
    *,
    runtime: ToolRuntime[None, SalesLeadAgentState],
    source: str,
    code: str,
    message: str,
    missing_data: list[str],
    recommended_next_step: str,
    acquisition_context: dict[str, Any] | None = None,
) -> Command:
    payload = {
        "source": source,
        "status": "failed",
        "error": message,
    }
    update: dict[str, Any] = {
        "turn_validation": _with_issue(
            runtime.state.get("turn_validation"),
            TurnValidationIssue(
                stage=source,
                code=code,
                message=message,
                metadata={},
            ),
        ),
        "missing_data": _merge_strings(runtime.state.get("missing_data"), missing_data),
        "recommended_next_step": recommended_next_step,
        "turn_tool_usage": _append_tool_usage(
            runtime.state.get("turn_tool_usage"),
            {
                "tool": source,
                "status": "failed",
                "validation_error": code,
            },
        ),
        "messages": [_tool_message(runtime, payload, status="error")],
    }
    if acquisition_context is not None:
        attempt = {
            "tool": source,
            "status": "failed",
            "active_run_ready": False,
            **acquisition_context,
        }
        update.update(
            {
                "acquisition_status": "failed",
                "acquisition_attempts": _append_acquisition_attempt(
                    runtime.state.get("acquisition_attempts"),
                    attempt,
                ),
                "last_acquisition_error": _acquisition_error_payload(
                    tool=source,
                    status="failed",
                    error=message,
                    run_id=acquisition_context.get("run_id"),
                    index_id=acquisition_context.get("index_id"),
                    context={key: value for key, value in acquisition_context.items() if key not in {"run_id", "index_id"}},
                ),
            }
        )
    return Command(update=update)


def build_sales_lead_tools(deps: SalesLeadAgentDependencies) -> list[Any]:
    @tool(
        "purchase_search_tool",
        args_schema=PurchaseSearchRequest,
        description="Search procurement opportunities, classify relevance, download documents, and prepare them for indexed search.",
    )
    def purchase_search_tool(
        *,
        runtime: ToolRuntime[None, SalesLeadAgentState],
        run_id: str | None = None,
        search_url: str | None = None,
        query_text: str | None = None,
        law: str | None = None,
        region: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        published_from: str | None = None,
        published_to: str | None = None,
        submission_deadline_from: str | None = None,
        submission_deadline_to: str | None = None,
        customer_name: str | None = None,
        customer_inn: str | None = None,
        supplier_hint: str | None = None,
        max_pages: int | None = None,
        headless: bool | None = None,
    ) -> Command:
        """Search procurement opportunities, filter them for semantic relevance, and prepare supporting documents.

        Args:
            run_id: Optional explicit run identifier. When omitted, the tool may reuse the current active run
                for the same acquisition context or create a new run.
            search_url: Optional direct EIS extended-search URL. If provided, it is used as-is.
            query_text: Contextualized procurement search phrase assembled for the EIS search string.
            law: Optional procurement law filter, for example `44-FZ` or `223-FZ`.
            region: Optional region hint used for search narrowing.
            min_price: Optional lower price boundary.
            max_price: Optional upper price boundary.
            published_from: Optional publication start date.
            published_to: Optional publication end date.
            submission_deadline_from: Optional submission deadline start date.
            submission_deadline_to: Optional submission deadline end date.
            customer_name: Optional customer name hint.
            customer_inn: Optional customer INN hint.
            supplier_hint: Optional supplier/service specialization hint.
            max_pages: Optional maximum number of crawler result pages to traverse.
            headless: Optional crawler headless-mode override.

        Returns:
            A `Command` that updates state with raw procurement hits, relevance-classified hits, dropped hits,
            prepared documents, normalized entities, active run metadata, and a tool transcript message.
        """
        try:
            filters_model = _search_filters_from_args(
                query_text=query_text,
                law=law,
                region=region,
                min_price=min_price,
                max_price=max_price,
                published_from=published_from,
                published_to=published_to,
                submission_deadline_from=submission_deadline_from,
                submission_deadline_to=submission_deadline_to,
                customer_name=customer_name,
                customer_inn=customer_inn,
                supplier_hint=supplier_hint,
            )
        except RecoverableToolInputError as exc:
            return _tool_validation_failure(
                runtime=runtime,
                source="purchase_search_tool",
                code="purchase_search_invalid_filters",
                message=str(exc),
                missing_data=["procurement_search_filters"],
                recommended_next_step=(
                    "Provide procurement search criteria in the expected format or retry with a direct search URL."
                ),
                acquisition_context={
                    "search_url": search_url,
                    "query_text": query_text,
                },
            )
        if not search_url and filters_model is None:
            return _tool_validation_failure(
                runtime=runtime,
                source="purchase_search_tool",
                code="purchase_search_missing_criteria",
                message="purchase_search_tool requires either search_url or at least one search filter.",
                missing_data=["procurement_search_criteria"],
                recommended_next_step=(
                    "Provide a direct procurement search URL or at least one procurement search filter."
                ),
                acquisition_context={
                    "search_url": search_url,
                    "query_text": query_text,
                },
            )

        resolved_run_id = _resolve_purchase_run_id(
            runtime.state,
            run_id,
            current_search_url=search_url,
            current_search_filters=filters_model,
        )
        workspace = deps.workspace_manager.get(resolved_run_id) if resolved_run_id else deps.workspace_manager.create_run()
        filters_payload = _search_filters_payload(filters_model)
        resolved_url, base_response = deps.purchase_adapter.search(
            search_url=search_url,
            search_filters=filters_model,
            downloads_dir=str(workspace.downloads_dir),
            max_pages=max_pages,
            headless=headless,
        )
        base_response.run_id = workspace.run_id
        base_response.index_id = workspace.index_id

        hits_summary = deps.purchase_adapter.summarize_hits(base_response)
        relevant_items = list(base_response.items)
        dropped_items: list[dict[str, Any]] = []
        unclassified_items: list[dict[str, Any]] = []
        relevance_artifact = {"decisions": []}
        turn_validation = runtime.state.get("turn_validation")
        missing_data = list(runtime.state.get("missing_data") or [])
        recommended_next_step = runtime.state.get("recommended_next_step")
        if base_response.status == "failed":
            failure_message = "; ".join(error for error in base_response.errors if error) or (
                "purchase_search_tool failed without a structured error message."
            )
            turn_validation = _with_issue(
                turn_validation,
                TurnValidationIssue(
                    stage="purchase_search_tool",
                    code="purchase_search_failed",
                    message=failure_message,
                    metadata={"search_url": resolved_url},
                ),
            )
            missing_data = _merge_strings(missing_data, ["procurement_search_source_availability"])
            recommended_next_step = (
                "Retry the procurement search or provide a direct EIS search URL after the source/crawler issue is resolved."
            )
        if hits_summary:
            classifier_summary = json.dumps(
                {
                    "search_url": resolved_url,
                    "search_filters": filters_payload,
                    "current_request": runtime.state.get("current_user_request"),
                    "task_understanding": runtime.state.get("task_understanding"),
                },
                ensure_ascii=False,
            )
            try:
                relevance = deps.classifier.classify_procurement_hits(
                    summary=classifier_summary,
                    hits=hits_summary,
                )
            except ClassifierExecutionError as exc:
                relevant_items = []
                unclassified_items = [item.model_dump() for item in base_response.items]
                relevance_artifact = {
                    "status": "unclassified",
                    "error": str(exc),
                    "raw_hit_count": len(unclassified_items),
                }
                base_response.status = "partial"
                base_response.errors = list(base_response.errors) + [f"relevance_classifier_failed:{exc}"]
                missing_data = _merge_strings(missing_data, ["procurement_relevance_verification"])
                recommended_next_step = (
                    "Review the procurement candidates manually or retry the search after the relevance classifier recovers."
                )
                turn_validation = _with_issue(
                    turn_validation,
                    TurnValidationIssue(
                        stage="procurement_relevance",
                        code="procurement_relevance_classifier_failed",
                        message=str(exc),
                        metadata={"search_url": resolved_url},
                    ),
                )
            else:
                decisions = {decision.bundle_id: decision for decision in relevance.decisions}
                selected = []
                for item in base_response.items:
                    decision = decisions.get(item.bundle_id)
                    if decision is None or decision.is_relevant:
                        selected.append(item)
                        continue
                    dropped_items.append(
                        {
                            "bundle_id": item.bundle_id,
                            "registry_number": item.registry_number,
                            "purchase_title": item.purchase_title,
                            "reason": decision.reason,
                        }
                    )
                relevant_items = selected
                relevance_artifact = relevance.model_dump()

        prepared_documents = []
        updated_items = []
        normalized_inns = list(runtime.state.get("normalized_inns") or [])
        company_names = list(runtime.state.get("company_names") or [])
        for item in relevant_items:
            artifact_paths = _ensure_purchase_artifacts(
                workspace=workspace,
                item=item.model_dump(),
            )
            provenance_by_path = {
                path: {
                    "original_source_url": item.detail_url,
                    "original_file_name": Path(path).name,
                    "derived_artifact_path": path,
                }
                for path in artifact_paths
            }
            prepared = deps.document_service.prepare_files(
                workspace=workspace,
                origin="purchase",
                bundle_id=item.bundle_id,
                registry_number=item.registry_number,
                source_url=item.detail_url,
                file_paths=artifact_paths,
                provenance_by_path=provenance_by_path,
            )
            item.prepared_document_ids = [doc.document_id for doc in prepared]
            for doc in prepared:
                normalized_inns = _merge_strings(normalized_inns, doc.entities.inn)
                company_names = _merge_strings(company_names, doc.entities.company_names)
            prepared_documents.extend(prepared)
            updated_items.append(item)

        searchable_document_count = _searchable_document_count(prepared_documents)
        response_status = base_response.status
        response_errors = list(base_response.errors)
        if updated_items and searchable_document_count == 0 and response_status != "failed":
            response_status = "partial"
            no_context_error = "Acquisition finished without producing a searchable prepared procurement corpus."
            if no_context_error not in response_errors:
                response_errors.append(no_context_error)
            turn_validation = _with_issue(
                turn_validation,
                TurnValidationIssue(
                    stage="purchase_search_tool",
                    code="purchase_search_context_unready",
                    message=no_context_error,
                    metadata={"run_id": workspace.run_id},
                ),
            )
            missing_data = _merge_strings(missing_data, ["prepared_procurement_corpus"])
            recommended_next_step = recommended_next_step or (
                "Retry the procurement acquisition after document preparation/indexing is available."
            )
        active_run_ready = response_status in {"success", "partial"} and searchable_document_count > 0
        response = PurchaseSearchResponse(
            run_id=workspace.run_id,
            index_id=workspace.index_id,
            status=response_status,  # type: ignore[arg-type]
            errors=response_errors,
            items=updated_items,
            prepared_documents=prepared_documents,
        )
        acquisition_error = None
        if response.status != "success":
            acquisition_error = _acquisition_error_payload(
                tool="purchase_search_tool",
                status=response.status,
                error="; ".join(error for error in response.errors if error)
                or "purchase_search_tool failed without a structured error message.",
                run_id=workspace.run_id,
                index_id=workspace.index_id,
                context={
                    "search_url": resolved_url,
                    "relevant_hits": len(updated_items),
                    "unclassified_hits": len(unclassified_items),
                },
            )
        update = {
            **_active_context_update(
                runtime.state,
                workspace=workspace,
                current_query_signature=runtime.state.get("current_query_signature"),
                active_run_ready=active_run_ready,
                acquisition_status=response.status,
            ),
            "acquisition_status": response.status,
            "acquisition_attempts": _append_acquisition_attempt(
                runtime.state.get("acquisition_attempts"),
                {
                    "tool": "purchase_search_tool",
                    "status": response.status,
                    "run_id": workspace.run_id,
                    "index_id": workspace.index_id,
                    "active_run_ready": active_run_ready,
                    "prepared_document_count": len(prepared_documents),
                    "searchable_document_count": searchable_document_count,
                    "relevant_hits": len(updated_items),
                    "unclassified_hits": len(unclassified_items),
                    "search_url": resolved_url,
                },
            ),
            "last_acquisition_error": acquisition_error,
            "last_purchase_search_result": response.model_dump(),
            "search_url": resolved_url,
            "search_filters": filters_payload,
            "raw_procurement_hits": [item.model_dump() for item in base_response.items],
            "procurement_hits": [item.model_dump() for item in updated_items],
            "dropped_procurement_hits": dropped_items,
            "unclassified_procurement_hits": unclassified_items,
            "prepared_documents": _merge_dict_list(
                runtime.state.get("prepared_documents"),
                [doc.model_dump() for doc in prepared_documents],
            ),
            "normalized_inns": normalized_inns,
            "company_names": company_names,
            "procurement_relevance": relevance_artifact,
            "turn_validation": turn_validation,
            "turn_tool_usage": _append_tool_usage(
                runtime.state.get("turn_tool_usage"),
                {
                    "tool": "purchase_search_tool",
                    "status": response.status,
                    "run_id": workspace.run_id,
                    "index_id": workspace.index_id,
                    "search_url": resolved_url,
                    "relevant_hits": len(updated_items),
                    "unclassified_hits": len(unclassified_items),
                    "searchable_document_count": searchable_document_count,
                    "active_run_ready": active_run_ready,
                },
            ),
            "missing_data": missing_data,
            "recommended_next_step": recommended_next_step,
            "semantic_dirty": True,
            "messages": [
                _tool_message(
                    runtime,
                    response.model_dump(),
                    status="error" if response.status == "failed" else "success",
                )
            ],
        }
        return Command(update=update)

    @tool(
        "open_source_fetch_tool",
        args_schema=OpenSourceFetchRequest,
        description="Fetch public web pages and attachments, prepare them, and add them to the active run index.",
    )
    async def open_source_fetch_tool(
        *,
        runtime: ToolRuntime[None, SalesLeadAgentState],
        run_id: str | None = None,
        url: str,
        depth: int | None = None,
        follow_download_links: bool | None = None,
        max_concurrency: int | None = None,
    ) -> Command:
        """Fetch open-source pages and downloadable attachments, then prepare them for indexed search.

        Args:
            run_id: Optional explicit run identifier. When omitted, the current active run is reused if present.
            url: Start URL for web retrieval.
            depth: Optional recursive crawl depth.
            follow_download_links: Whether downloadable attachments should also be fetched and parsed.
            max_concurrency: Optional override for concurrent crawling workers.

        Returns:
            A `Command` that updates state with fetched open-source bundles, prepared documents, normalized
            entities, active run metadata, and a tool transcript message.
        """
        current_run = run_id or runtime.state.get("active_run_id")
        workspace = deps.workspace_manager.get(current_run) if current_run else deps.workspace_manager.create_run()
        turn_validation = runtime.state.get("turn_validation")
        missing_data = list(runtime.state.get("missing_data") or [])
        recommended_next_step = runtime.state.get("recommended_next_step")
        loader = AsyncWebLoader(
            url=url,
            depth=depth or 0,
            fetch_mode="requests_fallback_playwright",
            follow_download_links=bool(follow_download_links),
            max_concurrency=max_concurrency or deps.open_source_max_concurrency,
            continue_on_error=True,
        )

        errors: list[str] = []
        status = "success"
        try:
            docs = await loader.load()
            errors = [str(item.get("error") or item) for item in loader.last_errors if item.get("error")]
            if errors:
                status = "partial"
        except Exception as exc:
            docs = []
            errors = [str(exc)]
            status = "failed"
        if status == "failed":
            failure_message = "; ".join(error for error in errors if error) or (
                "open_source_fetch_tool failed without a structured error message."
            )
            turn_validation = _with_issue(
                turn_validation,
                TurnValidationIssue(
                    stage="open_source_fetch_tool",
                    code="open_source_fetch_failed",
                    message=failure_message,
                    metadata={"url": url},
                ),
            )
            missing_data = _merge_strings(missing_data, ["open_source_fetch_source_availability"])
            recommended_next_step = (
                "Retry the public-source fetch or provide a more specific URL after the source issue is resolved."
            )

        page_docs: list[tuple[int, Any]] = []
        attachment_groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
        for index, doc in enumerate(docs, start=1):
            metadata = dict(doc.metadata or {})
            if str(metadata.get("source_type") or "") == "web_download":
                attachment_url = str(metadata.get("source") or url)
                parent_url = str(metadata.get("parent_url") or url)
                filename = str(metadata.get("download_filename") or f"download_{index:03d}")
                attachment_groups[(parent_url, attachment_url, filename)].append(doc)
                continue
            page_docs.append((index, doc))

        pages: list[OpenSourcePage] = []
        prepared_documents = []
        attachments_by_parent: dict[str, list[str]] = defaultdict(list)
        normalized_inns = list(runtime.state.get("normalized_inns") or [])
        company_names = list(runtime.state.get("company_names") or [])

        for index, doc in page_docs:
            metadata = dict(doc.metadata or {})
            source_url = str(metadata.get("source") or metadata.get("url") or url)
            bundle_id = f"web_{sha1(source_url.encode('utf-8')).hexdigest()[:12]}"
            file_path = deps.document_service.save_text_artifact(
                workspace=workspace,
                relative_dir=bundle_id,
                file_name=f"page_{index:03d}.txt",
                content=doc.page_content or "",
            )
            prepared = deps.document_service.prepare_files(
                workspace=workspace,
                origin="open_source",
                bundle_id=bundle_id,
                registry_number=None,
                source_url=source_url,
                file_paths=[file_path],
                provenance_by_path={
                    file_path: {
                        "original_source_url": source_url,
                        "original_file_name": f"page_{index:03d}.html",
                        "original_content_type": str(metadata.get("content_type") or "text/html"),
                        "derived_artifact_path": file_path,
                    }
                },
            )
            prepared_documents.extend(prepared)
            for item in prepared:
                normalized_inns = _merge_strings(normalized_inns, item.entities.inn)
                company_names = _merge_strings(company_names, item.entities.company_names)
            pages.append(
                OpenSourcePage(
                    bundle_id=bundle_id,
                    url=source_url,
                    title=str(metadata.get("title") or "") or None,
                    text=(doc.page_content or "")[:4000],
                    attachments=[],
                    prepared_document_ids=[item.document_id for item in prepared],
                )
            )

        for (parent_url, attachment_url, filename), grouped_docs in attachment_groups.items():
            content = "\n\n".join(doc.page_content for doc in grouped_docs if doc.page_content)
            if not content.strip():
                errors.append(f"Attachment {attachment_url} returned no parsed text.")
                status = "partial"
                continue
            bundle_id = f"download_{sha1(attachment_url.encode('utf-8')).hexdigest()[:12]}"
            file_path = deps.document_service.save_text_artifact(
                workspace=workspace,
                relative_dir=bundle_id,
                file_name=_safe_artifact_name(filename, default_stem="download"),
                content=content,
            )
            declared_content_type = next(
                (
                    str((doc.metadata or {}).get("content_type") or (doc.metadata or {}).get("mime_type"))
                    for doc in grouped_docs
                    if (doc.metadata or {}).get("content_type") or (doc.metadata or {}).get("mime_type")
                ),
                None,
            )
            prepared = deps.document_service.prepare_files(
                workspace=workspace,
                origin="open_source",
                bundle_id=bundle_id,
                registry_number=None,
                source_url=attachment_url,
                file_paths=[file_path],
                provenance_by_path={
                    file_path: {
                        "original_source_url": attachment_url,
                        "original_file_name": filename,
                        "original_content_type": declared_content_type,
                        "derived_artifact_path": file_path,
                    }
                },
            )
            prepared_documents.extend(prepared)
            attachments_by_parent[parent_url].append(attachment_url)
            for item in prepared:
                normalized_inns = _merge_strings(normalized_inns, item.entities.inn)
                company_names = _merge_strings(company_names, item.entities.company_names)

        for page in pages:
            page.attachments = attachments_by_parent.get(page.url, [])

        searchable_document_count = _searchable_document_count(prepared_documents)
        if prepared_documents and searchable_document_count == 0 and status != "failed":
            status = "partial"
            no_context_error = "Open-source acquisition finished without producing a searchable prepared corpus."
            if no_context_error not in errors:
                errors.append(no_context_error)
            turn_validation = _with_issue(
                turn_validation,
                TurnValidationIssue(
                    stage="open_source_fetch_tool",
                    code="open_source_fetch_context_unready",
                    message=no_context_error,
                    metadata={"run_id": workspace.run_id, "url": url},
                ),
            )
            missing_data = _merge_strings(missing_data, ["prepared_open_source_corpus"])
            recommended_next_step = recommended_next_step or (
                "Retry the public-source fetch after document preparation/indexing is available."
            )
        active_run_ready = status in {"success", "partial"} and searchable_document_count > 0
        response = OpenSourceFetchResponse(
            run_id=workspace.run_id,
            index_id=workspace.index_id,
            status=status,  # type: ignore[arg-type]
            errors=errors,
            pages=pages,
            prepared_documents=prepared_documents,
        )
        acquisition_error = None
        if response.status != "success":
            acquisition_error = _acquisition_error_payload(
                tool="open_source_fetch_tool",
                status=response.status,
                error="; ".join(error for error in response.errors if error)
                or "open_source_fetch_tool failed without a structured error message.",
                run_id=workspace.run_id,
                index_id=workspace.index_id,
                context={"url": url},
            )
        update = {
            **_active_context_update(
                runtime.state,
                workspace=workspace,
                current_query_signature=runtime.state.get("current_query_signature"),
                active_run_ready=active_run_ready,
                acquisition_status=response.status,
            ),
            "acquisition_status": response.status,
            "acquisition_attempts": _append_acquisition_attempt(
                runtime.state.get("acquisition_attempts"),
                {
                    "tool": "open_source_fetch_tool",
                    "status": response.status,
                    "run_id": workspace.run_id,
                    "index_id": workspace.index_id,
                    "active_run_ready": active_run_ready,
                    "prepared_document_count": len(prepared_documents),
                    "searchable_document_count": searchable_document_count,
                    "url": url,
                },
            ),
            "last_acquisition_error": acquisition_error,
            "last_open_source_fetch_result": response.model_dump(),
            "open_source_hits": _merge_dict_list(
                runtime.state.get("open_source_hits"),
                [page.model_dump() for page in pages],
            ),
            "prepared_documents": _merge_dict_list(
                runtime.state.get("prepared_documents"),
                [doc.model_dump() for doc in prepared_documents],
            ),
            "normalized_inns": normalized_inns,
            "company_names": company_names,
            "turn_validation": turn_validation,
            "missing_data": missing_data,
            "recommended_next_step": recommended_next_step,
            "turn_tool_usage": _append_tool_usage(
                runtime.state.get("turn_tool_usage"),
                {
                    "tool": "open_source_fetch_tool",
                    "status": status,
                    "run_id": workspace.run_id,
                    "index_id": workspace.index_id,
                    "url": url,
                    "searchable_document_count": searchable_document_count,
                    "active_run_ready": active_run_ready,
                },
            ),
            "semantic_dirty": True,
            "messages": [
                _tool_message(
                    runtime,
                    response.model_dump(),
                    status="error" if status == "failed" else "success",
                )
            ],
        }
        return Command(update=update)

    @tool(
        "doc_search_tool",
        args_schema=DocSearchRequest,
        description="Search the active prepared document index and return exact snippets with file and page/locator.",
    )
    def doc_search_tool(
        *,
        runtime: ToolRuntime[None, SalesLeadAgentState],
        index_id: str | None = None,
        query: str,
        top_k: int | None = None,
        source_kind: str | None = None,
        bundle_id: str | None = None,
    ) -> Command:
        """Search the prepared document index and return exact evidence snippets with provenance.

        Args:
            index_id: Optional explicit index identifier. When omitted, the tool uses the active run index.
            query: Semantic query to search in the prepared corpus.
            top_k: Optional maximum number of matches to return.
            source_kind: Optional source narrowing, for example `purchase` or `open_source`.
            bundle_id: Optional bundle narrowing for a specific procurement/web bundle.

        Returns:
            A `Command` that updates state with the latest search result, extracted evidence records,
            and a tool transcript message.
        """
        resolved_index_id = index_id or runtime.state.get("index_id")
        if not resolved_index_id:
            payload = {"index_id": "", "matches": [], "error": "No active run index is available."}
            return Command(
                update={
                    "turn_validation": _with_issue(
                        runtime.state.get("turn_validation"),
                        TurnValidationIssue(
                            stage="doc_search_tool",
                            code="doc_search_active_run_missing",
                            message="No active run index is available for doc_search_tool.",
                            metadata={},
                        ),
                    ),
                    "missing_data": _merge_strings(runtime.state.get("missing_data"), ["active_run_index"]),
                    "turn_tool_usage": _append_tool_usage(
                        runtime.state.get("turn_tool_usage"),
                        {
                            "tool": "doc_search_tool",
                            "status": "failed",
                            "index_id": index_id,
                            "query": query,
                            "source_kind": source_kind,
                            "bundle_id": bundle_id,
                            "guard": "active_run_index_missing",
                        },
                    ),
                    "messages": [_tool_message(runtime, payload, status="error")],
                }
            )
        active_run_id = runtime.state.get("active_run_id")
        try:
            workspace = (
                deps.workspace_manager.get_by_index(resolved_index_id)
                if index_id or not active_run_id
                else deps.workspace_manager.get(active_run_id)
            )
        except ValueError as exc:
            payload = {
                "index_id": resolved_index_id,
                "matches": [],
                "error": str(exc),
            }
            return Command(
                update={
                    "turn_validation": _with_issue(
                        runtime.state.get("turn_validation"),
                        TurnValidationIssue(
                            stage="doc_search_tool",
                            code="doc_search_index_resolution_failed",
                            message=str(exc),
                            metadata={"index_id": resolved_index_id},
                        ),
                    ),
                    "missing_data": _merge_strings(
                        runtime.state.get("missing_data"),
                        [f"index_id:{resolved_index_id}"],
                    ),
                    "turn_tool_usage": _append_tool_usage(
                        runtime.state.get("turn_tool_usage"),
                        {
                            "tool": "doc_search_tool",
                            "status": "failed",
                            "index_id": resolved_index_id,
                            "query": query,
                            "source_kind": source_kind,
                            "bundle_id": bundle_id,
                            "guard": "index_resolution_failed",
                        },
                    ),
                    "messages": [_tool_message(runtime, payload, status="error")],
                }
            )
        response = deps.document_service.search(
            workspace=workspace,
            query=query,
            top_k=top_k or 5,
            source_kind=source_kind,  # type: ignore[arg-type]
            bundle_id=bundle_id,
        )
        evidence = _evidence_from_doc_matches([match.model_dump() for match in response.matches])
        return Command(
            update={
                "last_doc_search_result": response.model_dump(),
                "evidence": _merge_dict_list(runtime.state.get("evidence"), evidence),
                "turn_tool_usage": _append_tool_usage(
                    runtime.state.get("turn_tool_usage"),
                    {
                        "tool": "doc_search_tool",
                        "status": "success",
                        "index_id": response.index_id,
                        "query": query,
                        "source_kind": source_kind,
                        "bundle_id": bundle_id,
                        "match_count": len(response.matches),
                    },
                ),
                "semantic_dirty": True,
                "messages": [_tool_message(runtime, response.model_dump())],
            }
        )

    @tool(
        "counterparty_scoring_tool",
        args_schema=CounterpartyScoringRequest,
        description="Fetch counterparty scoring and optional financial coefficients by normalized INN.",
    )
    def counterparty_scoring_tool(
        *,
        runtime: ToolRuntime[None, SalesLeadAgentState],
        inn: str,
        model: str | None = None,
        include_fincoefs: bool | None = None,
    ) -> Command:
        """Fetch counterparty scoring data for a normalized INN.

        Args:
            inn: Normalized company INN that must already exist in agent state.
            model: Optional external scoring model identifier.
            include_fincoefs: Whether financial coefficients should also be requested.

        Returns:
            A `Command` that updates state with normalized scoring payloads, evidence artifacts,
            and a tool transcript message. If the INN is not already normalized in state, the command
            records an explicit failure instead of calling the external API.
        """
        known_inns = set(runtime.state.get("normalized_inns") or [])
        if inn not in known_inns:
            return _inn_guard_failure(
                runtime=runtime,
                inn=inn,
                tool_name="counterparty_scoring_tool",
                source="damia_scoring",
            )

        response = deps.counterparty_clients.scoring(
            inn=inn,
            model=model,
            include_fincoefs=bool(include_fincoefs),
        )
        scoring_results = dict(runtime.state.get("scoring_results") or {})
        scoring_results[inn] = response.model_dump()
        evidence = []
        if response.status == "success":
            evidence = [
                {
                    "source": "scoring",
                    "source_url": None,
                    "file_path": None,
                    "page": None,
                    "locator": None,
                    "snippet": (
                        f"risk_zone={response.score.risk_zone}; "
                        f"score_zone={response.score.score_zone}; "
                        f"reliability_zone={response.score.reliability_zone}"
                    ),
                    "inn": inn,
                }
            ]
        return Command(
            update={
                "scoring_results": scoring_results,
                "evidence": _merge_dict_list(runtime.state.get("evidence"), evidence),
                "turn_validation": (
                    _with_issue(
                        runtime.state.get("turn_validation"),
                        TurnValidationIssue(
                            stage="counterparty_scoring_tool",
                            code="counterparty_scoring_failed",
                            message=response.error or f"counterparty_scoring_tool failed for INN {inn}.",
                            metadata={"inn": inn},
                        ),
                    )
                    if response.status == "failed"
                    else runtime.state.get("turn_validation")
                ),
                "turn_tool_usage": _append_tool_usage(
                    runtime.state.get("turn_tool_usage"),
                    {
                        "tool": "counterparty_scoring_tool",
                        "status": response.status,
                        "inn": inn,
                    },
                ),
                "semantic_dirty": True,
                "messages": [_tool_message(runtime, response.model_dump(), status="error" if response.status == "failed" else "success")],
            }
        )

    @tool(
        "counterparty_fssp_tool",
        args_schema=CounterpartyFSSPRequest,
        description="Fetch grouped enforcement proceedings by normalized INN.",
    )
    def counterparty_fssp_tool(
        *,
        runtime: ToolRuntime[None, SalesLeadAgentState],
        inn: str,
        from_date: str | None = None,
        to_date: str | None = None,
        format: int | None = None,
    ) -> Command:
        """Fetch grouped FSSP enforcement data for a normalized INN.

        Args:
            inn: Normalized company INN that must already exist in agent state.
            from_date: Optional lower bound for proceedings retrieval.
            to_date: Optional upper bound for proceedings retrieval.
            format: Optional response format expected by the external API.

        Returns:
            A `Command` that updates state with normalized FSSP payloads, evidence artifacts,
            and a tool transcript message. If the INN is not already normalized in state, the command
            records an explicit failure instead of calling the external API.
        """
        known_inns = set(runtime.state.get("normalized_inns") or [])
        if inn not in known_inns:
            return _inn_guard_failure(
                runtime=runtime,
                inn=inn,
                tool_name="counterparty_fssp_tool",
                source="damia_fssp",
            )

        response = deps.counterparty_clients.fssp(
            inn=inn,
            from_date=from_date,
            to_date=to_date,
            response_format=format or 1,
        )
        fssp_results = dict(runtime.state.get("fssp_results") or {})
        fssp_results[inn] = response.model_dump()
        evidence = []
        if response.status == "success":
            total_count = sum(item.count for item in response.grouped)
            evidence = [
                {
                    "source": "fssp",
                    "source_url": None,
                    "file_path": None,
                    "page": None,
                    "locator": None,
                    "snippet": f"total_grouped_proceedings={total_count}",
                    "inn": inn,
                }
            ]
        return Command(
            update={
                "fssp_results": fssp_results,
                "evidence": _merge_dict_list(runtime.state.get("evidence"), evidence),
                "turn_validation": (
                    _with_issue(
                        runtime.state.get("turn_validation"),
                        TurnValidationIssue(
                            stage="counterparty_fssp_tool",
                            code="counterparty_fssp_failed",
                            message=response.error or f"counterparty_fssp_tool failed for INN {inn}.",
                            metadata={"inn": inn},
                        ),
                    )
                    if response.status == "failed"
                    else runtime.state.get("turn_validation")
                ),
                "turn_tool_usage": _append_tool_usage(
                    runtime.state.get("turn_tool_usage"),
                    {
                        "tool": "counterparty_fssp_tool",
                        "status": response.status,
                        "inn": inn,
                    },
                ),
                "semantic_dirty": True,
                "messages": [_tool_message(runtime, response.model_dump(), status="error" if response.status == "failed" else "success")],
            }
        )

    return [
        purchase_search_tool,
        open_source_fetch_tool,
        doc_search_tool,
        counterparty_scoring_tool,
        counterparty_fssp_tool,
    ]
