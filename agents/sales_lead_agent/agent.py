from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from agents.state.state import ConfigSchema
from agents.utils import ModelType
from bot_service.db import AsyncSessionFactory

from . import models as _lead_models  # noqa: F401
from .adapters import AdapterBundle
from .logic import (
    apply_enrichment_to_leads,
    build_export_attachment,
    build_leads_from_sources,
    build_task_plan,
    compose_response,
    deduplicate_leads,
    last_user_text,
    parse_datetime_filter,
    parse_request,
    refine_request_understanding,
    score_leads,
)
from .prompts import NO_RESULTS_MESSAGE_RU, STAGE_LABELS
from .schemas import LeadExportRequest, LeadFeedbackCreate
from .state import SalesLeadContext, SalesLeadState
from .storage import add_feedback, create_export, list_leads, persist_leads, resolve_feedback_target, search_lead_index


LOG = logging.getLogger(__name__)


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


def _emit_stage(stage: str, **payload: Any) -> None:
    writer = _safe_stream_writer()
    writer({"type": "sales_lead_stage", "stage": stage, **payload})


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        return configurable
    return {}


def _storage_filters(understanding: dict[str, Any]) -> dict[str, Any]:
    filters = understanding.get("filters") or {}
    return {
        "period_from": parse_datetime_filter(filters.get("period_from")),
        "period_to": parse_datetime_filter(filters.get("period_to")),
        "region": (filters.get("regions") or [None])[0],
        "priority": filters.get("priority"),
        "source_type": (filters.get("required_sources") or [None])[0],
        "inn": filters.get("inn"),
        "company_name": filters.get("company_name"),
        "procurement_id": filters.get("procurement_id"),
        "query_text": understanding.get("query_text"),
        "only_with_inn": bool(filters.get("only_with_inn")),
        "only_with_contacts": bool(filters.get("only_with_contacts")),
    }


async def request_understanding_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["request_understanding"])
    configurable = _configurable(config)
    attachments = list(state.get("attachments") or configurable.get("attachments") or [])
    text = last_user_text(state.get("messages") or [])
    understanding = parse_request(text, None, attachments)
    understanding = refine_request_understanding(understanding, attachments=attachments)
    return {
        "attachments": attachments,
        "request_understanding": understanding,
        "errors": list(state.get("errors") or []),
    }


async def task_planning_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del config, runtime
    _emit_stage(STAGE_LABELS["task_planning"])
    attachments = list(state.get("attachments") or [])
    understanding = dict(state.get("request_understanding") or {})
    return {"task_plan": build_task_plan(understanding, attachments)}


async def source_collection_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["source_collection"])
    understanding = dict(state.get("request_understanding") or {})
    task_plan = dict(state.get("task_plan") or {})
    configurable = _configurable(config)
    user_id = str(configurable.get("user_id") or "anonymous")
    existing_leads: list[dict[str, Any]] = []
    errors = list(state.get("errors") or [])
    if task_plan.get("use_existing_store"):
        try:
            async with AsyncSessionFactory() as session:
                existing = await list_leads(
                    session,
                    user_id=user_id,
                    **_storage_filters(understanding),
                    limit=20,
                )
                existing_leads = [item.model_dump(mode="json") for item in existing]
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Sales lead existing-store lookup failed: %s", exc)
            errors.append(f"Не удалось прочитать накопленные лиды: {exc}")

    bundle = AdapterBundle()
    source_hits: list[dict[str, Any]] = []
    if task_plan.get("collect_sources"):
        for source_kind in task_plan.get("source_priority") or []:
            if source_kind == "procurement":
                source_hits.extend(await bundle.procurement.search(understanding))
            elif source_kind == "open_source":
                source_hits.extend(await bundle.open_source.search(understanding))
    return {
        "existing_leads": existing_leads,
        "source_hits": source_hits,
        "errors": errors,
    }


async def document_resolution_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del config, runtime
    _emit_stage(STAGE_LABELS["document_resolution"])
    task_plan = dict(state.get("task_plan") or {})
    if not task_plan.get("resolve_documents"):
        return {"documents": []}
    bundle = AdapterBundle()
    documents = await bundle.documents.resolve(
        source_hits=list(state.get("source_hits") or []),
        attachments=list(state.get("attachments") or []),
        require_index=bool(task_plan.get("require_index")),
    )
    return {"documents": documents}


async def index_lookup_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["index_lookup"])
    task_plan = dict(state.get("task_plan") or {})
    if not task_plan.get("query_index"):
        return {"index_hits": []}

    understanding = dict(state.get("request_understanding") or {})
    configurable = _configurable(config)
    user_id = str(configurable.get("user_id") or "anonymous")
    errors = list(state.get("errors") or [])
    try:
        async with AsyncSessionFactory() as session:
            hits = await search_lead_index(
                session,
                user_id=user_id,
                query_text=str(understanding.get("query_text") or ""),
                **_storage_filters(understanding),
            )
        return {"index_hits": [item.model_dump(mode="json") for item in hits]}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Sales lead index lookup failed: %s", exc)
        errors.append(f"Не удалось выполнить поиск по индексу: {exc}")
        return {"index_hits": [], "errors": errors}


async def fact_extraction_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del config, runtime
    _emit_stage(STAGE_LABELS["fact_extraction"])
    understanding = dict(state.get("request_understanding") or {})
    existing_leads = list(state.get("existing_leads") or [])
    source_hits = list(state.get("source_hits") or [])
    documents = list(state.get("documents") or [])
    if not source_hits and not documents:
        return {"extracted_leads": existing_leads}
    extracted = build_leads_from_sources(
        understanding=understanding,
        source_hits=source_hits,
        documents=documents,
    )
    return {"extracted_leads": extracted or existing_leads}


async def normalization_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["normalization"])
    configurable = _configurable(config)
    user_id = str(configurable.get("user_id") or "anonymous")
    normalized = deduplicate_leads(list(state.get("extracted_leads") or []))
    for lead in normalized:
        lead["created_by_user_id"] = lead.get("created_by_user_id") or user_id
    return {"extracted_leads": normalized}


async def enrichment_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del config, runtime
    _emit_stage(STAGE_LABELS["enrichment"])
    task_plan = dict(state.get("task_plan") or {})
    lead_cards = list(state.get("extracted_leads") or [])
    if not task_plan.get("enrich") or not lead_cards:
        return {"extracted_leads": lead_cards}
    bundle = AdapterBundle()
    scoring_results: dict[str, dict[str, Any]] = {}
    fssp_results: dict[str, dict[str, Any]] = {}
    for inn in {str(item.get("inn")) for item in lead_cards if item.get("inn")}:
        scoring_results[inn] = await bundle.scoring.enrich(inn)
        fssp_results[inn] = await bundle.fssp.enrich(inn)
    return {"extracted_leads": apply_enrichment_to_leads(lead_cards, scoring_results, fssp_results)}


async def rule_scoring_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del config, runtime
    _emit_stage(STAGE_LABELS["rule_scoring"])
    understanding = dict(state.get("request_understanding") or {})
    lead_cards = list(state.get("extracted_leads") or [])
    if not lead_cards:
        return {"extracted_leads": []}
    return {
        "extracted_leads": score_leads(lead_cards, list(understanding.get("filters", {}).get("keywords") or []))
    }


async def persistence_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["persistence"])
    task_plan = dict(state.get("task_plan") or {})
    lead_cards = list(state.get("extracted_leads") or [])
    if not task_plan.get("persist") or not lead_cards:
        return {"persisted_leads": lead_cards}
    configurable = _configurable(config)
    user_id = str(configurable.get("user_id") or "anonymous")
    errors = list(state.get("errors") or [])
    try:
        async with AsyncSessionFactory() as session:
            persisted = await persist_leads(
                session,
                lead_cards=lead_cards,
                created_by_user_id=user_id,
            )
        return {"persisted_leads": [item.model_dump(mode="json") for item in persisted]}
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Sales lead persistence failed: %s", exc)
        errors.append(f"Не удалось сохранить результаты: {exc}")
        return {"persisted_leads": lead_cards, "errors": errors}


async def feedback_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["feedback"])
    task_plan = dict(state.get("task_plan") or {})
    if not task_plan.get("apply_feedback"):
        return {"feedback_result": None}

    understanding = dict(state.get("request_understanding") or {})
    filters = understanding.get("filters") or {}
    configurable = _configurable(config)
    user_id = str(configurable.get("user_id") or "anonymous")
    errors = list(state.get("errors") or [])
    try:
        async with AsyncSessionFactory() as session:
            target = await resolve_feedback_target(
                session,
                user_id=user_id,
                lead_id=filters.get("lead_id"),
                inn=filters.get("inn"),
                company_name=filters.get("company_name"),
                procurement_id=filters.get("procurement_id"),
            )
            if target is None:
                return {
                    "feedback_result": {
                        "saved": False,
                        "lead_id": filters.get("lead_id"),
                        "status": filters.get("feedback_status"),
                        "comment": filters.get("feedback_comment"),
                        "message": "Лид для обратной связи не найден.",
                    }
                }

            feedback = await add_feedback(
                session,
                lead_id=target.id,
                user_id=user_id,
                payload=LeadFeedbackCreate(
                    status=str(filters.get("feedback_status") or "manual_review"),
                    comment=filters.get("feedback_comment"),
                ),
            )
        return {"feedback_result": {"saved": True, **feedback.model_dump(mode="json")}}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Sales lead feedback failed: %s", exc)
        errors.append(f"Не удалось сохранить обратную связь: {exc}")
        return {
            "feedback_result": {
                "saved": False,
                "lead_id": filters.get("lead_id"),
                "status": filters.get("feedback_status"),
                "comment": filters.get("feedback_comment"),
                "message": str(exc),
            },
            "errors": errors,
        }


async def response_composition_node(
    state: SalesLeadState,
    config: RunnableConfig,
    runtime: Runtime[SalesLeadContext],
) -> Dict[str, Any]:
    del runtime
    _emit_stage(STAGE_LABELS["response_composition"])
    understanding = dict(state.get("request_understanding") or {})
    persisted_leads = list(
        state.get("persisted_leads") or state.get("extracted_leads") or state.get("existing_leads") or []
    )
    index_hits = list(state.get("index_hits") or [])
    feedback_result = state.get("feedback_result")
    errors = list(state.get("errors") or [])
    export_record: dict[str, Any] | None = None
    summary_export_record: dict[str, Any] | None = None
    configurable = _configurable(config)
    user_id = str(configurable.get("user_id") or "anonymous")
    if understanding.get("needs_export"):
        try:
            async with AsyncSessionFactory() as session:
                filters = understanding.get("filters") or {}
                export = await create_export(
                    session,
                    request=LeadExportRequest(
                        format="xlsx",
                        period_from=parse_datetime_filter(filters.get("period_from")),
                        period_to=parse_datetime_filter(filters.get("period_to")),
                        region=(filters.get("regions") or [None])[0],
                        priority=filters.get("priority"),
                        source_type=(filters.get("required_sources") or [None])[0],
                        only_with_inn=bool(filters.get("only_with_inn")),
                        only_with_contacts=bool(filters.get("only_with_contacts")),
                        limit=100,
                    ),
                    user_id=user_id,
                )
                summary_export = await create_export(
                    session,
                    request=LeadExportRequest(
                        format="txt",
                        period_from=parse_datetime_filter(filters.get("period_from")),
                        period_to=parse_datetime_filter(filters.get("period_to")),
                        region=(filters.get("regions") or [None])[0],
                        priority=filters.get("priority"),
                        source_type=(filters.get("required_sources") or [None])[0],
                        only_with_inn=bool(filters.get("only_with_inn")),
                        only_with_contacts=bool(filters.get("only_with_contacts")),
                        limit=100,
                    ),
                    user_id=user_id,
                )
            export_record = export.model_dump(mode="json")
            summary_export_record = summary_export.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Sales lead export failed: %s", exc)
            errors.append(f"Не удалось сформировать выгрузку: {exc}")

    response_text = compose_response(
        understanding=understanding,
        persisted_leads=persisted_leads,
        export_record=export_record,
        summary_export_record=summary_export_record,
        feedback_result=feedback_result,
        index_hits=index_hits,
        errors=errors,
    )
    if not persisted_leads and not index_hits and not feedback_result and not errors:
        response_text = NO_RESULTS_MESSAGE_RU

    attachment = build_export_attachment(export_record["path"]) if export_record else None
    summary_attachment = build_export_attachment(summary_export_record["path"]) if summary_export_record else None
    content: List[Dict[str, Any]] = [{"type": "text", "text": response_text}]
    if attachment:
        content.append(attachment)
    if summary_attachment:
        content.append(summary_attachment)

    return {
        "response_text": response_text,
        "export_record": export_record,
        "summary_export_record": summary_export_record,
        "errors": errors,
        "messages": [AIMessage(content=content)],
    }


def initialize_agent(
    provider: ModelType = ModelType.GPT,
    role: str = "default",
    use_platform_store: bool = False,
    notify_on_reload: bool = True,
    checkpoint_saver=None,
    locale: str = "ru",
    *,
    streaming: bool = True,
    **_kwargs: Any,
):
    del provider, role, notify_on_reload, locale, streaming, _kwargs
    memory = None if use_platform_store else checkpoint_saver or MemorySaver()

    builder = StateGraph(SalesLeadState, context_schema=SalesLeadContext, config_schema=ConfigSchema)
    builder.add_node("request_understanding", request_understanding_node)
    builder.add_node("task_planning", task_planning_node)
    builder.add_node("source_collection", source_collection_node)
    builder.add_node("document_resolution", document_resolution_node)
    builder.add_node("index_lookup", index_lookup_node)
    builder.add_node("fact_extraction", fact_extraction_node)
    builder.add_node("normalization", normalization_node)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("rule_scoring", rule_scoring_node)
    builder.add_node("persistence", persistence_node)
    builder.add_node("feedback", feedback_node)
    builder.add_node("response_composition", response_composition_node)

    builder.add_edge(START, "request_understanding")
    builder.add_edge("request_understanding", "task_planning")
    builder.add_edge("task_planning", "source_collection")
    builder.add_edge("source_collection", "document_resolution")
    builder.add_edge("document_resolution", "index_lookup")
    builder.add_edge("index_lookup", "fact_extraction")
    builder.add_edge("fact_extraction", "normalization")
    builder.add_edge("normalization", "enrichment")
    builder.add_edge("enrichment", "rule_scoring")
    builder.add_edge("rule_scoring", "persistence")
    builder.add_edge("persistence", "feedback")
    builder.add_edge("feedback", "response_composition")
    builder.add_edge("response_composition", END)

    graph = builder.compile(checkpointer=memory, debug=False, name="sales_lead_agent")
    return graph
