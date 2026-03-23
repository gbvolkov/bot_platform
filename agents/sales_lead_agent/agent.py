from __future__ import annotations

import json
import re
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware import after_agent, before_model, dynamic_prompt
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

from agents.llm_utils import get_llm
from agents.utils import ModelType, extract_text

from .prompts import build_system_prompt
from .schemas import (
    EvidenceItem,
    FactStatus,
    LeadAnswerContract,
    LeadAnswerItem,
    TaskUnderstandingResult,
    TurnValidationIssue,
    TurnValidationResult,
)
from .services import (
    ClassifierContractError,
    ClassifierExecutionError,
    CounterpartyClients,
    DocumentPreparationService,
    InternalClassifier,
    ProcurementQueryBuilder,
    PurchaseAdapter,
    RunWorkspaceManager,
)
from .settings import get_settings
from .state import SalesLeadAgentState
from .tools import SalesLeadAgentDependencies, build_sales_lead_tools


VALID_MODEL_SIZES = {"base", "mini", "nano"}
_INN_ONLY_RE = re.compile(r"^\d{10,12}$")


def _import_create_deep_agent():
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The `deepagents` package is not installed. Install dependencies in `.venv`."
        ) from exc
    return create_deep_agent


def _latest_human_message(state: SalesLeadAgentState) -> HumanMessage | None:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            return message
    return None


def _message_marker(state: SalesLeadAgentState, message: HumanMessage | None) -> str | None:
    if message is None:
        return None
    if getattr(message, "id", None):
        return str(message.id)
    human_messages = [
        item
        for item in (state.get("messages") or [])
        if isinstance(item, HumanMessage) or getattr(item, "type", "") == "human"
    ]
    return f"{len(human_messages)}:{extract_text(message)}"


def _build_turn_summary(state: SalesLeadAgentState, request_text: str) -> str:
    payload = {
        "current_request": request_text,
        "active_run_id": state.get("active_run_id"),
        "previous_answer": (state.get("normalized_final_answer") or {}).get("summary"),
        "known_inns": state.get("normalized_inns") or [],
        "known_companies": state.get("company_names") or [],
        "relevant_procurements": [
            {
                "registry_number": item.get("registry_number"),
                "purchase_title": item.get("purchase_title"),
                "customer_name": item.get("customer_name"),
            }
            for item in (state.get("procurement_hits") or [])[:10]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _default_next_step() -> str:
    return "Repeat the request after the transient tool/provider issue is resolved or narrow the request."


def _task_signature(result: TaskUnderstandingResult) -> str:
    return json.dumps(result.model_dump(), ensure_ascii=False, sort_keys=True)


def _build_turn_tool_requirements(result: TaskUnderstandingResult) -> dict[str, Any]:
    task_kind = result.task_kind
    requires_doc_search = task_kind in {"fact_lookup", "procurement_analysis"}
    requires_enrichment = task_kind in {"company_check", "comparison"}
    return {
        "task_kind": task_kind,
        "answer_type": result.answer_type,
        "purchase_search_required": bool(result.needs_purchase_search),
        "doc_search_required": requires_doc_search,
        "scoring_required": requires_enrichment,
        "fssp_required": requires_enrichment,
        "requested_company_inns": list(result.requested_company_inns),
        "comparison_targets": list(result.comparison_targets),
    }


def _assessment_payload(state: SalesLeadAgentState) -> dict[str, Any]:
    return {
        "task_understanding": state.get("task_understanding"),
        "procurement_hits": state.get("procurement_hits") or [],
        "open_source_hits": state.get("open_source_hits") or [],
        "last_doc_search_result": state.get("last_doc_search_result"),
        "scoring_results": state.get("scoring_results") or {},
        "fssp_results": state.get("fssp_results") or {},
        "evidence": state.get("evidence") or [],
        "missing_data": state.get("missing_data") or [],
    }


def _merge_strings(existing: list[str] | None, additions: list[str]) -> list[str]:
    merged = list(existing or [])
    for value in additions:
        if value and value not in merged:
            merged.append(value)
    return merged


def _extract_inn_candidates(values: list[str]) -> list[str]:
    candidates: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if _INN_ONLY_RE.fullmatch(normalized) and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _validation_from_state(state: SalesLeadAgentState) -> TurnValidationResult:
    payload = state.get("turn_validation")
    if isinstance(payload, TurnValidationResult):
        return payload
    if isinstance(payload, dict):
        try:
            return TurnValidationResult.model_validate(payload)
        except Exception:
            pass
    return TurnValidationResult()


def _clean_validation_payload() -> dict[str, Any]:
    return TurnValidationResult(status="clean", issues=[], manual_review_required=False).model_dump()


def _issue(
    *,
    stage: str,
    code: str,
    message: str,
    metadata: dict[str, str] | None = None,
) -> TurnValidationIssue:
    return TurnValidationIssue(
        stage=stage,
        code=code,
        message=message,
        metadata=metadata or {},
    )


def _with_issue(
    current: dict[str, Any] | TurnValidationResult | None,
    issue: TurnValidationIssue,
) -> dict[str, Any]:
    validation = (
        current
        if isinstance(current, TurnValidationResult)
        else TurnValidationResult.model_validate(current or {})
    )
    issues = list(validation.issues)
    issues.append(issue)
    status = "failed_verification" if any(item.severity == "error" for item in issues) else "partial"
    return TurnValidationResult(
        status=status,
        issues=issues,
        manual_review_required=True,
    ).model_dump()


def _fallback_answer_type(state: SalesLeadAgentState) -> str:
    task = state.get("task_understanding") or {}
    return str(task.get("answer_type") or "lead_list")


def _build_degraded_contract(
    state: SalesLeadAgentState,
    *,
    validation: dict[str, Any] | TurnValidationResult | None = None,
    structured_error: str | None = None,
) -> LeadAnswerContract:
    validation_payload = (
        validation.model_dump() if isinstance(validation, TurnValidationResult) else validation
    )
    validation_result = _validation_from_state(
        {**dict(state), "turn_validation": validation_payload or state.get("turn_validation")}
    )
    issues = list(validation_result.issues)
    if structured_error:
        issues.append(
            _issue(
                stage="finalization",
                code="structured_response_invalid",
                message=structured_error,
            )
        )

    summary_parts: list[str] = []
    if issues:
        summary_parts.append("The answer is partial because verification or runtime steps failed.")
    acquisition_error = state.get("last_acquisition_error") or {}
    if acquisition_error:
        tool_name = str(acquisition_error.get("tool") or "acquisition")
        error_text = str(acquisition_error.get("error") or "").strip()
        if error_text:
            summary_parts.append(f"{tool_name} reported: {error_text}")
    procurement_hits = state.get("procurement_hits") or []
    unclassified_hits = state.get("unclassified_procurement_hits") or []
    if procurement_hits:
        summary_parts.append(f"Verified procurement hits available: {len(procurement_hits)}.")
    if unclassified_hits:
        summary_parts.append(
            f"Unclassified procurement hits pending manual review: {len(unclassified_hits)}."
        )
    normalized_inns = state.get("normalized_inns") or []
    company_names = state.get("company_names") or []
    items: list[LeadAnswerItem] = []
    if procurement_hits:
        item_reasons = [issue.message for issue in issues[:2]]
        for raw_hit in procurement_hits[:10]:
            title = str(raw_hit.get("purchase_title") or "").strip() or None
            detail_url = str(raw_hit.get("detail_url") or "").strip() or None
            evidence = []
            if title:
                evidence.append(
                    EvidenceItem(
                        source="purchase",
                        source_url=detail_url,
                        file_path=None,
                        page=None,
                        locator=None,
                        snippet=title,
                    )
                )
            items.append(
                LeadAnswerItem(
                    company_name=str(raw_hit.get("customer_name") or "").strip() or None,
                    inn=None,
                    event_title=title,
                    source_url=detail_url,
                    region=str(raw_hit.get("region") or "").strip() or None,
                    amount_text=str(raw_hit.get("price_text") or "").strip() or None,
                    contacts=[],
                    scoring=None,
                    fssp=None,
                    priority="unknown",
                    reasons=item_reasons,
                    evidence=evidence,
                    fact_statuses=[
                        FactStatus(
                            fact_key="procurement_hit",
                            status="document",
                        )
                    ],
                )
            )
    elif normalized_inns:
        items.append(
            LeadAnswerItem(
                company_name=company_names[0] if company_names else None,
                inn=normalized_inns[0],
                priority="unknown",
                reasons=[issue.message for issue in issues[:3]],
                evidence=[],
                fact_statuses=[],
            )
        )
    if not summary_parts:
        summary_parts.append("The answer is partial because not enough verified data is available.")
    recommended_next_step = state.get("recommended_next_step") or _default_next_step()
    return LeadAnswerContract(
        answer_type=_fallback_answer_type(state),  # type: ignore[arg-type]
        summary=" ".join(summary_parts),
        items=items,
        missing_data=_merge_strings(
            state.get("missing_data"),
            [issue.code for issue in issues],
        ),
        recommended_next_step=recommended_next_step,
    )


def _degrade_existing_contract(
    contract: LeadAnswerContract,
    state: SalesLeadAgentState,
    *,
    validation: dict[str, Any] | TurnValidationResult | None = None,
) -> LeadAnswerContract:
    validation_payload = (
        validation.model_dump() if isinstance(validation, TurnValidationResult) else validation
    )
    validation_result = _validation_from_state(
        {**dict(state), "turn_validation": validation_payload or state.get("turn_validation")}
    )
    if not validation_result.issues:
        return contract

    prefix = "The answer is partial because verification or runtime steps failed."
    summary = contract.summary
    if prefix not in summary:
        additions = []
        unclassified_hits = state.get("unclassified_procurement_hits") or []
        if unclassified_hits:
            additions.append(
                f"Unclassified procurement hits pending manual review: {len(unclassified_hits)}."
            )
        detail = " ".join(additions)
        summary = " ".join(part for part in [prefix, detail, contract.summary] if part)

    return contract.model_copy(
        update={
            "summary": summary,
            "missing_data": _merge_strings(
                _merge_strings(contract.missing_data, state.get("missing_data") or []),
                [issue.code for issue in validation_result.issues],
            ),
            "recommended_next_step": contract.recommended_next_step
            or state.get("recommended_next_step")
            or _default_next_step(),
        }
    )


def _tool_usage_entries(state: SalesLeadAgentState) -> list[dict[str, Any]]:
    entries = state.get("turn_tool_usage")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _acquisition_attempt_entries(state: SalesLeadAgentState) -> list[dict[str, Any]]:
    entries = state.get("acquisition_attempts")
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return []


def _latest_acquisition_attempt(
    state: SalesLeadAgentState,
    *,
    tool_name: str,
) -> dict[str, Any] | None:
    for entry in reversed(_acquisition_attempt_entries(state)):
        if entry.get("tool") == tool_name:
            return entry
    return None


def _should_rebuild_from_state(
    state: SalesLeadAgentState,
    validation: TurnValidationResult,
) -> bool:
    if not validation.issues:
        return False
    acquisition_code_prefixes = (
        "required_purchase_search_",
        "purchase_search_",
        "open_source_fetch_",
    )
    for issue in validation.issues:
        if issue.code.startswith(acquisition_code_prefixes):
            return True
        if issue.stage in {"purchase_search_tool", "open_source_fetch_tool"}:
            return True
    return bool(state.get("last_acquisition_error"))


def _is_recoverable_provider_error(exc: Exception) -> bool:
    module_name = exc.__class__.__module__
    class_name = exc.__class__.__name__
    if isinstance(exc, StructuredOutputValidationError):
        return True
    if module_name.startswith(
        (
            "openai",
            "anthropic",
            "httpx",
            "langchain_openai",
            "langchain_anthropic",
        )
    ):
        return True
    return class_name in {"APIError", "APIConnectionError", "APITimeoutError", "RateLimitError"}


def _recover_model_exception(state: SalesLeadAgentState, exc: Exception) -> AIMessage:
    if isinstance(exc, StructuredOutputValidationError):
        code = "structured_output_validation_failed"
        missing_data_key = "final_structured_output"
    else:
        code = "final_model_provider_failed"
        missing_data_key = "final_model_provider"
    state["turn_validation"] = _with_issue(
        state.get("turn_validation"),
        _issue(
            stage="model_output",
            code=code,
            message=str(exc),
        ),
    )
    state["missing_data"] = _merge_strings(
        state.get("missing_data"),
        [missing_data_key],
    )
    state["recommended_next_step"] = state.get("recommended_next_step") or (
        "Retry the request after the final response model recovers or narrow the task scope."
    )
    state["structured_response"] = None
    return AIMessage(content="")


class RecoverModelFailuresMiddleware(AgentMiddleware):
    state_schema = SalesLeadAgentState
    tools: list[Any] = []

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except Exception as exc:
            if not _is_recoverable_provider_error(exc):
                raise
            return _recover_model_exception(request.state, exc)

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as exc:
            if not _is_recoverable_provider_error(exc):
                raise
            return _recover_model_exception(request.state, exc)


def _can_autorun_purchase_search(understanding: TaskUnderstandingResult) -> bool:
    if not understanding.needs_purchase_search:
        return False
    if understanding.search_url:
        return True
    if understanding.search_filters is None:
        return False
    return bool((understanding.search_filters.query_text or "").strip())


def _tool_used(
    usage: list[dict[str, Any]],
    *,
    tool_name: str,
    inn: str | None = None,
    acceptable_statuses: set[str] | None = None,
) -> bool:
    for entry in usage:
        if entry.get("tool") != tool_name:
            continue
        if inn is not None and entry.get("inn") != inn:
            continue
        if acceptable_statuses is not None and entry.get("status") not in acceptable_statuses:
            continue
        return True
    return False


def _required_enrichment_targets(
    contract: LeadAnswerContract,
    state: SalesLeadAgentState,
    requirements: dict[str, Any],
) -> tuple[list[str], list[str]]:
    contract_targets = _extract_inn_candidates([item.inn or "" for item in contract.items])
    requested_targets = _extract_inn_candidates(requirements.get("requested_company_inns") or [])
    task_kind = requirements.get("task_kind")
    if task_kind == "comparison":
        targets = list(requested_targets)
        unresolved_targets: list[str] = []
        name_to_inn = {
            str(item.company_name).strip().lower(): item.inn
            for item in contract.items
            if item.company_name and item.inn
        }
        for raw_target in requirements.get("comparison_targets") or []:
            target = str(raw_target or "").strip()
            if not target:
                continue
            if _INN_ONLY_RE.fullmatch(target):
                if target not in targets:
                    targets.append(target)
                continue
            mapped_inn = name_to_inn.get(target.lower())
            if mapped_inn and mapped_inn not in targets:
                targets.append(mapped_inn)
                continue
            unresolved_targets.append(target)
        if not targets and contract_targets:
            targets.extend(contract_targets)
        return targets, unresolved_targets

    if requested_targets:
        return requested_targets, []
    if contract_targets:
        return contract_targets[:1], []

    normalized_targets = _extract_inn_candidates(state.get("normalized_inns") or [])
    if task_kind == "company_check" and normalized_targets:
        return normalized_targets[:1], []
    return normalized_targets, []


def _tool_requirement_next_step(requirements: dict[str, Any]) -> str:
    task_kind = requirements.get("task_kind")
    if task_kind == "procurement_search":
        return "Run purchase_search_tool and keep only a procurement result backed by the current acquisition state."
    if task_kind in {"fact_lookup", "procurement_analysis"}:
        return "Run doc_search_tool against the active prepared corpus and cite exact snippets with page or locator."
    if task_kind == "company_check":
        return "Run counterparty_scoring_tool and counterparty_fssp_tool for the normalized company INN before finalizing the check."
    if task_kind == "comparison":
        return "Run counterparty_scoring_tool and counterparty_fssp_tool for each comparison target before finalizing the ranking."
    return _default_next_step()


def _enforce_tool_requirements(
    state: SalesLeadAgentState,
    contract: LeadAnswerContract,
    turn_validation: dict[str, Any] | TurnValidationResult | None,
) -> tuple[dict[str, Any] | TurnValidationResult | None, str | None]:
    requirements = state.get("turn_tool_requirements")
    if not isinstance(requirements, dict) or not requirements:
        return turn_validation, None

    usage = _tool_usage_entries(state)
    updated_validation = turn_validation
    recommended_next_step: str | None = None
    task_kind = str(requirements.get("task_kind") or "")

    if requirements.get("purchase_search_required"):
        purchase_attempt = _latest_acquisition_attempt(state, tool_name="purchase_search_tool")
        if purchase_attempt is None:
            updated_validation = _with_issue(
                updated_validation,
                _issue(
                    stage="tool_requirements",
                    code="required_purchase_search_missing",
                    message="This procurement answer requires purchase_search_tool before finalization.",
                    metadata={"task_kind": task_kind},
                ),
            )
            recommended_next_step = _tool_requirement_next_step(requirements)
        elif purchase_attempt.get("status") not in {"success", "partial"}:
            updated_validation = _with_issue(
                updated_validation,
                _issue(
                    stage="tool_requirements",
                    code="required_purchase_search_failed",
                    message="purchase_search_tool was attempted but did not return a usable acquisition result.",
                    metadata={"task_kind": task_kind},
                ),
            )
            recommended_next_step = _tool_requirement_next_step(requirements)

    if requirements.get("doc_search_required") and not _tool_used(usage, tool_name="doc_search_tool"):
        updated_validation = _with_issue(
            updated_validation,
            _issue(
                stage="tool_requirements",
                code="required_doc_search_missing",
                message="This answer requires doc_search_tool evidence before finalization.",
                metadata={"task_kind": task_kind},
            ),
        )
        recommended_next_step = _tool_requirement_next_step(requirements)
    elif requirements.get("doc_search_required") and not _tool_used(
        usage,
        tool_name="doc_search_tool",
        acceptable_statuses={"success"},
    ):
        updated_validation = _with_issue(
            updated_validation,
            _issue(
                stage="tool_requirements",
                code="required_doc_search_failed",
                message="doc_search_tool was attempted but did not produce a usable search result.",
                metadata={"task_kind": task_kind},
            ),
        )
        recommended_next_step = _tool_requirement_next_step(requirements)
    elif requirements.get("doc_search_required") and not any(
        entry.get("tool") == "doc_search_tool"
        and entry.get("status") == "success"
        and int(entry.get("match_count") or 0) > 0
        for entry in usage
    ):
        updated_validation = _with_issue(
            updated_validation,
            _issue(
                stage="tool_requirements",
                code="required_doc_search_evidence_missing",
                message="doc_search_tool ran but did not return any evidence snippets for this turn.",
                metadata={"task_kind": task_kind},
            ),
        )
        recommended_next_step = _tool_requirement_next_step(requirements)
    elif requirements.get("doc_search_required") and not any(
        evidence.source == "document" and bool(evidence.snippet.strip())
        for item in contract.items
        for evidence in item.evidence
    ):
        updated_validation = _with_issue(
            updated_validation,
            _issue(
                stage="tool_requirements",
                code="required_doc_search_evidence_not_rendered",
                message="Document-backed turns must include document evidence snippets in the final answer contract.",
                metadata={"task_kind": task_kind},
            ),
        )
        recommended_next_step = _tool_requirement_next_step(requirements)

    if requirements.get("scoring_required") or requirements.get("fssp_required"):
        targets, unresolved_targets = _required_enrichment_targets(contract, state, requirements)
        if not targets:
            updated_validation = _with_issue(
                updated_validation,
                _issue(
                    stage="tool_requirements",
                    code=f"{task_kind or 'enrichment'}_target_inn_missing",
                    message="Normalized INN is missing for the required enrichment step.",
                    metadata={"task_kind": task_kind},
                ),
            )
            recommended_next_step = _tool_requirement_next_step(requirements)
        for target in unresolved_targets:
            updated_validation = _with_issue(
                updated_validation,
                _issue(
                    stage="tool_requirements",
                    code="comparison_target_inn_missing",
                    message=f"Comparison target '{target}' does not have a resolved INN for enrichment.",
                    metadata={"task_kind": task_kind, "target": target},
                ),
            )
            recommended_next_step = _tool_requirement_next_step(requirements)
        for inn in targets:
            if requirements.get("scoring_required") and not _tool_used(usage, tool_name="counterparty_scoring_tool", inn=inn):
                updated_validation = _with_issue(
                    updated_validation,
                    _issue(
                        stage="tool_requirements",
                        code=f"{task_kind or 'enrichment'}_scoring_missing",
                        message=f"counterparty_scoring_tool was not attempted for INN {inn}.",
                        metadata={"task_kind": task_kind, "inn": inn},
                    ),
                )
                recommended_next_step = _tool_requirement_next_step(requirements)
            elif requirements.get("scoring_required") and not _tool_used(
                usage,
                tool_name="counterparty_scoring_tool",
                inn=inn,
                acceptable_statuses={"success"},
            ):
                updated_validation = _with_issue(
                    updated_validation,
                    _issue(
                        stage="tool_requirements",
                        code=f"{task_kind or 'enrichment'}_scoring_failed",
                        message=f"counterparty_scoring_tool failed for INN {inn}.",
                        metadata={"task_kind": task_kind, "inn": inn},
                    ),
                )
                recommended_next_step = _tool_requirement_next_step(requirements)
            if requirements.get("fssp_required") and not _tool_used(
                usage,
                tool_name="counterparty_fssp_tool",
                inn=inn,
            ):
                updated_validation = _with_issue(
                    updated_validation,
                    _issue(
                        stage="tool_requirements",
                        code=f"{task_kind or 'enrichment'}_fssp_missing",
                        message=f"counterparty_fssp_tool was not attempted for INN {inn}.",
                        metadata={"task_kind": task_kind, "inn": inn},
                    ),
                )
                recommended_next_step = _tool_requirement_next_step(requirements)
            elif requirements.get("fssp_required") and not _tool_used(
                usage,
                tool_name="counterparty_fssp_tool",
                inn=inn,
                acceptable_statuses={"success"},
            ):
                updated_validation = _with_issue(
                    updated_validation,
                    _issue(
                        stage="tool_requirements",
                        code=f"{task_kind or 'enrichment'}_fssp_failed",
                        message=f"counterparty_fssp_tool failed for INN {inn}.",
                        metadata={"task_kind": task_kind, "inn": inn},
                    ),
                )
                recommended_next_step = _tool_requirement_next_step(requirements)

    return updated_validation, recommended_next_step


def _render_contract(contract: LeadAnswerContract) -> str:
    lines = [contract.summary]
    for item in contract.items:
        head = " / ".join(
            [part for part in [item.company_name, item.inn, item.event_title] if part]
        )
        if head:
            lines.append(f"- {head}")
        if item.amount_text:
            lines.append(f"  Amount: {item.amount_text}")
        lines.append(f"  Priority: {item.priority}")
        if item.reasons:
            lines.append(f"  Reasons: {'; '.join(item.reasons)}")
        if item.fact_statuses:
            fact_statuses = "; ".join(
                f"{fact_status.fact_key}={fact_status.status}" if fact_status.fact_key else fact_status.status
                for fact_status in item.fact_statuses
            )
            lines.append(f"  Fact status: {fact_statuses}")
        for evidence in item.evidence[:3]:
            snippet = evidence.snippet.strip().replace("\n", " ")
            suffix = ""
            if evidence.page is not None:
                suffix = f" (page {evidence.page})"
            elif evidence.locator:
                suffix = f" ({evidence.locator})"
            source = evidence.file_path or evidence.source_url or evidence.source
            lines.append(f"  Evidence: {source}{suffix} - {snippet[:220]}")
    if contract.missing_data:
        lines.append("Missing data: " + "; ".join(contract.missing_data))
    if contract.recommended_next_step:
        lines.append("Next step: " + contract.recommended_next_step)
    return "\n".join(lines).strip()


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    *,
    model_size: str = "base",
    temperature: float = 0.1,
    checkpoint_saver: Any | None = None,
    streaming: bool = False,
    reasoning: str | None = None,
    max_tool_calls: int | None = 12,
    dependencies: SalesLeadAgentDependencies | None = None,
    **_: Any,
):
    if model_size not in VALID_MODEL_SIZES:
        choices = ", ".join(sorted(VALID_MODEL_SIZES))
        raise ValueError(f"Unknown model size '{model_size}'. Available values: {choices}")

    create_deep_agent = _import_create_deep_agent()
    settings = get_settings()
    deps = dependencies or SalesLeadAgentDependencies(
        workspace_manager=RunWorkspaceManager(settings),
        document_service=DocumentPreparationService(settings),
        classifier=InternalClassifier(provider=provider),
        purchase_adapter=PurchaseAdapter(
            settings=settings,
            query_builder=ProcurementQueryBuilder(settings),
        ),
        counterparty_clients=CounterpartyClients(settings),
        open_source_max_concurrency=settings.open_source_max_concurrency,
    )
    tools: Sequence[Any] = build_sales_lead_tools(deps)
    purchase_search_tool = next(tool for tool in tools if getattr(tool, "name", "") == "purchase_search_tool")

    @before_model(state_schema=SalesLeadAgentState)
    def prepare_runtime_state(state: SalesLeadAgentState, runtime) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        latest_human = _latest_human_message(state)
        latest_marker = _message_marker(state, latest_human)
        if latest_human is not None and latest_marker != state.get("last_processed_human_message_id"):
            request_text = extract_text(latest_human)
            updates.update(
                {
                    "turn_validation": _clean_validation_payload(),
                    "turn_tool_requirements": None,
                    "turn_tool_usage": [],
                    "task_understanding": None,
                    "structured_response": None,
                    "search_url": None,
                    "search_filters": None,
                    "acquisition_status": None,
                    "acquisition_attempts": [],
                    "last_acquisition_error": None,
                    "last_purchase_search_result": None,
                    "last_open_source_fetch_result": None,
                    "assessment": None,
                    "risk_verification": None,
                    "procurement_relevance": None,
                    "missing_data": [],
                    "recommended_next_step": None,
                    "current_query_signature": None,
                }
            )
            summary = _build_turn_summary(state, request_text)
            try:
                understanding = deps.classifier.classify_intent(summary=summary)
            except ClassifierExecutionError as exc:
                updates["turn_validation"] = _with_issue(
                    updates.get("turn_validation"),
                    _issue(
                        stage="intent_classifier",
                        code="intent_classifier_failed",
                        message=str(exc),
                    ),
                )
                updates["missing_data"] = _merge_strings(
                    updates.get("missing_data"),
                    ["intent_classification"],
                )
                updates["recommended_next_step"] = (
                    "Retry the request or narrow the task scope while the classifier issue is investigated."
                )
            else:
                normalized_inns = _merge_strings(
                    state.get("normalized_inns"),
                    _extract_inn_candidates(
                        list(understanding.requested_company_inns) + list(understanding.comparison_targets)
                    ),
                )
                updates["task_understanding"] = understanding.model_dump()
                updates["search_url"] = understanding.search_url
                updates["search_filters"] = (
                    understanding.search_filters.model_dump()
                    if understanding.search_filters is not None
                    else None
                )
                updates["missing_data"] = list(understanding.missing_data)
                updates["current_query_signature"] = _task_signature(understanding)
                updates["normalized_inns"] = normalized_inns
                updates["turn_tool_requirements"] = _build_turn_tool_requirements(understanding)
                if _can_autorun_purchase_search(understanding):
                    tool_runtime = SimpleNamespace(state={**dict(state), **updates})
                    tool_kwargs = (
                        understanding.search_filters.model_dump(exclude_none=True)
                        if understanding.search_filters is not None
                        else {}
                    )
                    command = purchase_search_tool.func(
                        runtime=tool_runtime,
                        search_url=understanding.search_url,
                        **tool_kwargs,
                    )
                    updates.update(command.update)
            updates["current_user_request"] = request_text
            updates["last_processed_human_message_id"] = latest_marker

        if state.get("semantic_dirty"):
            try:
                assessment = deps.classifier.assess_signals(signals=_assessment_payload(state))
            except ClassifierExecutionError as exc:
                updates["assessment"] = None
                updates["risk_verification"] = None
                updates["turn_validation"] = _with_issue(
                    updates.get("turn_validation") or state.get("turn_validation"),
                    _issue(
                        stage="assessment_classifier",
                        code="assessment_classifier_failed",
                        message=str(exc),
                    ),
                )
                updates["missing_data"] = _merge_strings(
                    updates.get("missing_data") or state.get("missing_data"),
                    ["semantic_assessment"],
                )
                updates["recommended_next_step"] = (
                    "Continue with documented tool outputs only or retry after the assessment service recovers."
                )
            else:
                updates["assessment"] = assessment.model_dump()
                updates["risk_verification"] = assessment.model_dump()
                updates["recommended_next_step"] = assessment.recommended_next_step
            updates["semantic_dirty"] = False

        return updates or None

    @dynamic_prompt
    def build_prompt(request) -> str:
        state: SalesLeadAgentState = request.state
        return build_system_prompt(dict(state))

    recover_model_failures = RecoverModelFailuresMiddleware()

    @after_agent(state_schema=SalesLeadAgentState)
    def finalize_answer(state: SalesLeadAgentState, runtime) -> dict[str, Any] | None:
        structured = state.get("structured_response")
        if structured is None:
            turn_validation = _with_issue(
                state.get("turn_validation"),
                _issue(
                    stage="finalization",
                    code="structured_response_missing",
                    message="The model returned no structured response.",
                ),
            )
            contract = _build_degraded_contract(
                state,
                validation=turn_validation,
                structured_error="missing structured_response",
            )
        else:
            try:
                contract = (
                    structured
                    if isinstance(structured, LeadAnswerContract)
                    else LeadAnswerContract.model_validate(structured)
                )
                turn_validation = state.get("turn_validation")
            except Exception as exc:
                turn_validation = _with_issue(
                    state.get("turn_validation"),
                    _issue(
                        stage="finalization",
                        code="structured_response_invalid",
                        message=str(exc),
                    ),
                )
                contract = _build_degraded_contract(
                    state,
                    validation=turn_validation,
                    structured_error=str(exc),
                )

        turn_validation, requirement_next_step = _enforce_tool_requirements(
            state,
            contract,
            turn_validation,
        )
        validation_state = _validation_from_state(
            {**dict(state), "turn_validation": turn_validation or state.get("turn_validation")}
        )
        if validation_state.issues:
            effective_state = dict(state)
            if requirement_next_step and not effective_state.get("recommended_next_step"):
                effective_state["recommended_next_step"] = requirement_next_step
            if _should_rebuild_from_state(effective_state, validation_state):
                contract = _build_degraded_contract(
                    effective_state,
                    validation=validation_state,
                )
            else:
                contract = _degrade_existing_contract(
                    contract,
                    effective_state,
                    validation=validation_state,
                )

        rendered = _render_contract(contract)
        messages = list(state.get("messages") or [])
        last_human_index = max(
            (idx for idx, message in enumerate(messages) if isinstance(message, HumanMessage)),
            default=-1,
        )
        remove_messages = []
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage) and index > last_human_index:
                if getattr(message, "id", None):
                    remove_messages.append(RemoveMessage(id=message.id))
                break
        return {
            "normalized_final_answer": contract.model_dump(),
            "rendered_answer": rendered,
            "missing_data": contract.missing_data,
            "recommended_next_step": contract.recommended_next_step,
            "turn_validation": turn_validation,
            "messages": [*remove_messages, AIMessage(content=rendered)],
        }

    llm = get_llm(
        model=model_size,
        provider=provider.value,
        temperature=temperature,
        streaming=streaming,
        reasoning=reasoning,
        max_tool_calls=max_tool_calls,
    )
    return create_deep_agent(
        name="sales_lead_agent",
        model=llm,
        tools=list(tools),
        system_prompt=build_system_prompt(),
        middleware=[build_prompt, prepare_runtime_state, finalize_answer, recover_model_failures],
        response_format=ProviderStrategy(schema=LeadAnswerContract),
        checkpointer=checkpoint_saver or MemorySaver(),
    )
