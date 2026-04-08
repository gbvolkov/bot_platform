from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import config
from agents.llm_utils import get_llm
from agents.state.state import ConfigSchema
from agents.structured_prompt_utils import provider_then_tool
from agents.utils import ModelType, build_internal_invoke_config, extract_text
from platform_utils.llm_logger import JSONFileTracer

from .documents import GazDocumentsClient
from .logic import (
    build_sales_context_baseline,
    clean_text,
    compute_missing_slots,
    infer_client_intent,
    infer_customer_temperature,
    merge_flags,
    merge_slots,
    prioritize_missing_slots,
)
from .locales import DEFAULT_LOCALE, resolve_locale
from .middleware import build_palimpsest_middleware
from .prompts import compose_prompt, get_prompt, get_text
from .schemas import TurnIntentExtractionResult
from .state import GazAgentState
from .tools import (
    build_branch_pack_tool,
    build_classify_problem_branch_tool,
    build_collect_product_snapshot_tool,
    build_compare_product_directions_tool,
    build_followup_pack_tool,
    build_query_pricing_bi_tool,
    build_read_material_tool,
    build_sales_catalog_overview_tool,
    build_sales_landscape_tool,
    build_search_sales_materials_tool,
    build_solution_shortlist_tool,
    build_web_search_tool,
    format_allowed_product_names,
    parse_allowed_product_names_env,
)

LOG = logging.getLogger(__name__)
StructuredOutputStrategy = Literal["auto", "provider", "tool", "provider_then_tool"]
_TEMPERATURE_ORDER = {"neutral": 0, "impatient": 1, "irritated": 2, "competitor_risk": 3}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GAZ_PRICING_DB_URL = f"sqlite:///{(_REPO_ROOT / 'data' / 'gaz-pricing' / 'gaz_pricing_norm.sqlite').as_posix()}"
_DEFAULT_GAZ_PRICING_PROMPT_CONTEXT_PATH = _REPO_ROOT / "agents" / "gaz_agent" / "pricing_bi_prompt_context.txt"
_DEFAULT_INTERNAL_RECURSION_LIMIT = 50
_PRICING_BI_THREAD_SUFFIX = ":pricing_bi"
_STAGE_BY_INTENT = {
    "recommendation": "RECOMMEND",
    "next_step": "FOLLOWUP",
    "compare": "RESEARCH",
    "specs": "RESEARCH",
    "materials": "RESEARCH",
    "objection": "RESEARCH",
}
_TOOL_PROMPT_KEYS = {
    "get_sales_catalog_overview": "tool_catalog",
    "get_sales_landscape": "tool_landscape",
    "compare_product_directions": "tool_compare_directions",
    "collect_product_snapshot": "tool_snapshot",
    "search_sales_materials": "tool_search",
    "read_material": "tool_read",
    "classify_problem_branch": "tool_branch_classify",
    "get_branch_pack": "tool_branch_pack",
    "build_solution_shortlist": "tool_shortlist",
    "build_followup_pack": "tool_followup",
    "query_pricing_bi": "tool_pricing_bi",
    "web_search": "tool_web_search",
}


def _safe_stream_writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None


class StreamWriterCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str=None, **kwargs):
        writer = _safe_stream_writer()
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "tool"
        writer({"type": "tool_start", "name": name})

    def on_tool_end(self, output=None, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_end"})

    def on_tool_error(self, error, **kwargs):
        writer = _safe_stream_writer()
        writer({"type": "tool_error", "error": str(error)})


def _emit_custom(event_type: str, **payload: Any) -> None:
    writer = _safe_stream_writer()
    writer({"type": event_type, **payload})



def _is_reset_requested(state: GazAgentState) -> bool:
    messages = state.get("messages") or []
    if not messages:
        return False
    last = messages[-1]
    if getattr(last, "type", None) != "human":
        return False
    content = getattr(last, "content", None)
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return isinstance(first, dict) and first.get("type") == "reset"



def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)



def _last_ai_text(messages: List[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return extract_text(message)
    return ""



def _append_trace(
    state: GazAgentState,
    *,
    stage_before: str,
    stage_after: str,
    allowed_tools: List[str],
    tool_calls: List[str],
    response_summary: str,
    policy_checks_passed: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    trace = list(state.get("trace") or [])
    trace.append(
        {
            "turn_index": len(trace) + 1,
            "stage_before": stage_before,
            "stage_after": stage_after,
            "current_client_intent": state.get("current_client_intent"),
            "source_strategy": state.get("source_strategy"),
            "source_reason": state.get("source_reason"),
            "mentioned_models": state.get("mentioned_models") or [],
            "requested_facts": state.get("requested_facts") or [],
            "research_layer": state.get("research_layer"),
            "slots_changed": state.get("slots") or {},
            "allowed_tools": allowed_tools,
            "tool_calls": tool_calls,
            "response_summary": response_summary,
            "runtime_warnings": state.get("runtime_warnings") or [],
            "tool_limit_hits": state.get("tool_limit_hits") or [],
            "loop_stop_reason": clean_text(state.get("sales_loop_guard_reason")),
            "policy_checks_passed": policy_checks_passed or [],
        }
    )
    return trace



def _build_middlewares(locale: str, summary_model: BaseChatModel) -> List[Any]:
    middleware: List[Any] = []
    if config.USE_ANONIMIZER:
        palimpsest_locale = "ru-RU" if locale == "ru" else "en-US"
        middleware.append(build_palimpsest_middleware(locale=palimpsest_locale))
    middleware.append(
        SummarizationMiddleware(
            model=summary_model,
            max_tokens_before_summary=80000,
            messages_to_keep=20,
            summary_prompt=get_prompt(locale, "summary"),
        )
    )
    return middleware



def _structured_response_format(schema: type[Any], strategy: StructuredOutputStrategy):
    if strategy == "provider":
        return ProviderStrategy(schema=schema)
    if strategy == "tool":
        return ToolStrategy(schema=schema)
    if strategy == "provider_then_tool":
        return schema
    if strategy == "auto":
        return schema
    raise ValueError(f"Unsupported structured output strategy: {strategy}")



def _structured_output_middleware(strategy: StructuredOutputStrategy) -> List[Any]:
    if strategy == "provider_then_tool":
        return [provider_then_tool]
    return []



def _max_temperature(first: str, second: str) -> str:
    return first if _TEMPERATURE_ORDER.get(first, 0) >= _TEMPERATURE_ORDER.get(second, 0) else second



def _latest_history_item(values: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    if not values:
        return {}
    item = values[-1]
    return dict(item or {})


def _compact_payload(state: GazAgentState) -> Dict[str, Any]:
    research_status = state.get("research_status") or {}
    latest_landscape = _latest_history_item(state.get("sales_digests"))
    latest_comparison = _latest_history_item(state.get("comparison_digests"))
    latest_snapshot = _latest_history_item(state.get("product_snapshots"))
    return {
        "locale": state.get("locale", DEFAULT_LOCALE),
        "stage": state.get("stage", "SELL"),
        "current_client_intent": state.get("current_client_intent"),
        "customer_temperature": state.get("customer_temperature"),
        "problem_summary": state.get("problem_summary") or "",
        "slots": state.get("slots") or {},
        "missing_slots": prioritize_missing_slots(state.get("missing_slots") or []),
        "intent_flags": state.get("intent_flags") or {},
        "clarification_allowed": bool(state.get("clarification_allowed")),
        "llm_retry_instruction": clean_text(state.get("llm_retry_instruction")),
        "provisional_recommendations": state.get("provisional_recommendations") or [],
        "mentioned_models": state.get("mentioned_models") or [],
        "requested_facts": state.get("requested_facts") or [],
        "source_strategy": state.get("source_strategy") or "docs_first",
        "source_reason": state.get("source_reason") or "",
        "runtime_warnings": state.get("runtime_warnings") or [],
        "active_branch": state.get("active_branch"),
        "branch_conflict": state.get("branch_conflict") or [],
        "docs_status": state.get("docs_status") or {},
        "sales_context_baseline": state.get("sales_context_baseline") or {},
        "research_layer": state.get("research_layer") or "portfolio_baseline",
        "allowed_tool_names": state.get("allowed_tool_names") or [],
        "search_query": state.get("search_query") or "",
        "allowed_material_ids": state.get("allowed_material_ids") or [],
        "research_status": {
            "estimated_remaining_cost": research_status.get("estimated_remaining_cost"),
            "candidate_count": research_status.get("candidate_count"),
            "queries": research_status.get("queries") or [],
            "documents_touched": research_status.get("documents_touched") or [],
            "rationale": research_status.get("rationale") or "",
            "last_composite_tool": research_status.get("last_composite_tool"),
            "read_attempts_by_candidate": state.get("read_attempts_by_candidate") or {},
            "search_key_count": len(state.get("search_keys_this_turn") or []),
            "read_focus_key_count": len(state.get("read_focus_keys_this_turn") or []),
            "tool_limit_hits": state.get("tool_limit_hits") or [],
            "sales_loop_guard_reason": clean_text(state.get("sales_loop_guard_reason")),
        },
        "material_candidates": [
            {
                "candidate_id": item.get("candidate_id"),
                "title": item.get("title"),
                "doc_kind": item.get("doc_kind"),
                "rationale": item.get("rationale"),
            }
            for item in state.get("material_candidates") or []
        ],
        "material_reads": [
            {
                "candidate_id": item.get("candidate_id"),
                "title": item.get("title"),
                "focus": item.get("focus"),
            }
            for item in state.get("material_reads") or []
        ],
        "sales_digest": {
            "topic": latest_landscape.get("topic"),
            "directions": [
                {
                    "title": item.get("title"),
                    "families": item.get("families") or [],
                    "main_characteristics": item.get("main_characteristics") or [],
                    "key_tradeoffs": item.get("key_tradeoffs") or [],
                }
                for item in latest_landscape.get("directions") or []
            ],
            "finance_options": latest_landscape.get("finance_options") or [],
            "recommended_next_narrowing": latest_landscape.get("recommended_next_narrowing"),
        },
        "comparison_digest": {
            "query": latest_comparison.get("query"),
            "products_compared": [
                {
                    "family_id": item.get("family_id"),
                    "label": item.get("label"),
                    "differentiators": item.get("differentiators") or [],
                }
                for item in latest_comparison.get("products_compared") or []
            ],
            "comparison_axes": latest_comparison.get("comparison_axes") or [],
            "high_level_differences": latest_comparison.get("high_level_differences") or [],
            "assumptions": latest_comparison.get("assumptions") or [],
        },
        "product_snapshot": {
            "query": latest_snapshot.get("query"),
            "dimensions_requested": latest_snapshot.get("dimensions_requested") or [],
            "products": [
                {
                    "family_id": item.get("family_id"),
                    "label": item.get("label"),
                    "facts": item.get("facts") or [],
                }
                for item in latest_snapshot.get("products") or []
            ],
            "value_ranges_or_baselines": latest_snapshot.get("value_ranges_or_baselines") or [],
            "assumptions": latest_snapshot.get("assumptions") or [],
        },
        "shortlist": state.get("shortlist") or [],
        "followup_pack": state.get("followup_pack") or {},
    }



def _turn_intent_payload(state: GazAgentState) -> Dict[str, Any]:
    return {
        "locale": state.get("locale", DEFAULT_LOCALE),
        "problem_summary": state.get("problem_summary") or "",
        "slots": state.get("slots") or {},
        "intent_flags": state.get("intent_flags") or {},
        "sales_context_baseline": state.get("sales_context_baseline") or {},
        "last_user_text": state.get("last_user_text") or "",
        "last_mentioned_models": state.get("mentioned_models") or [],
        "last_requested_facts": state.get("requested_facts") or [],
        "last_source_strategy": state.get("source_strategy") or "docs_first",
    }



def _retry_prompt_sections(state: GazAgentState) -> List[str]:
    instruction = clean_text(state.get("llm_retry_instruction"))
    if not instruction:
        return []
    return [f"RETRY INSTRUCTION:\n{instruction}"]



def _runtime_warning(stage: str, code: str, detail: str = "", attempt: Optional[int] = None) -> Dict[str, Any]:
    warning: Dict[str, Any] = {"stage": stage, "code": code}
    if detail:
        warning["detail"] = detail
    if attempt is not None:
        warning["attempt"] = attempt
    return warning



def _merge_runtime_warnings(state: GazAgentState, warnings: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    merged = list(state.get("runtime_warnings") or [])
    merged.extend(warnings or [])
    return merged



def _tool_call_id(value: Any) -> str:
    if isinstance(value, dict):
        return clean_text(value.get("id"))
    return clean_text(getattr(value, "id", ""))


def _sanitize_messages_for_followup(messages: List[Any]) -> tuple[List[Any], List[str]]:
    sanitized: List[Any] = []
    pruned_ids: List[str] = []
    idx = 0
    total = len(messages)
    while idx < total:
        message = messages[idx]
        if isinstance(message, ToolMessage):
            break
        if not isinstance(message, AIMessage):
            sanitized.append(message)
            idx += 1
            continue
        tool_calls = list(getattr(message, "tool_calls", []) or [])
        if not tool_calls:
            sanitized.append(message)
            idx += 1
            continue

        required_ids = [_tool_call_id(call) for call in tool_calls if _tool_call_id(call)]
        if not required_ids:
            break

        pending_ids = set(required_ids)
        segment: List[Any] = [message]
        cursor = idx + 1
        while cursor < total and isinstance(messages[cursor], ToolMessage):
            tool_message = messages[cursor]
            tool_call_id = clean_text(getattr(tool_message, "tool_call_id", ""))
            if tool_call_id not in pending_ids:
                break
            pending_ids.remove(tool_call_id)
            segment.append(tool_message)
            cursor += 1
            if not pending_ids:
                break

        if pending_ids:
            pruned_ids.extend(required_ids)
            break

        sanitized.extend(segment)
        idx = cursor
    deduped_pruned_ids = list(dict.fromkeys(pruned_ids))
    return sanitized, deduped_pruned_ids



def _derive_sales_loop_guard_reason(
    state: GazAgentState,
    warnings: List[Dict[str, Any]] | None,
    tool_limit_hits: List[Dict[str, Any]] | None,
) -> str:
    state_reason = clean_text(state.get("sales_loop_guard_reason"))
    if state_reason:
        return state_reason
    if tool_limit_hits:
        return clean_text((tool_limit_hits[-1] or {}).get("reason")) or "tool_limit_hit"
    codes = {clean_text(item.get("code")) for item in warnings or [] if clean_text(item.get("code"))}
    if "graph_recursion_limit" in codes:
        return "graph_recursion_limit"
    if "tool_limit_hit" in codes:
        return "tool_limit_hit"
    return ""



def _recover_checkpoint_result(agent: Any, config: RunnableConfig) -> Dict[str, Any]:
    get_state = getattr(agent, "get_state", None)
    if not callable(get_state):
        return {}
    snapshot = get_state(config)
    values = _to_dict(getattr(snapshot, "values", None))
    return values if isinstance(values, dict) else {}


def _invoke_agent_with_retry(
    agent,
    state: GazAgentState,
    config: RunnableConfig,
    runtime: Runtime | None,
    *,
    stage_name: str,
    extra_tags: List[str],
    retry_instruction: str,
    expects_structured: bool = False,
    non_retryable_exceptions: tuple[type[BaseException], ...] = (),
    stop_on_loop_guard: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any], str, List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []
    last_result: Dict[str, Any] = {}
    answer = ""
    for attempt in (1, 2):
        invoke_state: GazAgentState = dict(state)
        if attempt == 2:
            invoke_state["llm_retry_instruction"] = retry_instruction
        tags = list(extra_tags)
        if attempt == 2:
            tags.append("retry")
        internal_config = build_internal_invoke_config(config, extra_tags=tags)
        internal_config["recursion_limit"] = _DEFAULT_INTERNAL_RECURSION_LIMIT
        try:
            result = agent.invoke(invoke_state, config=internal_config, context=getattr(runtime, "context", None))
        except Exception as exc:
            LOG.warning("Agent invoke failed stage=%s attempt=%s error=%s", stage_name, attempt, exc)
            warning_code = "graph_recursion_limit" if isinstance(exc, GraphRecursionError) else "invoke_exception"
            warnings.append(_runtime_warning(stage_name, warning_code, str(exc), attempt))
            if isinstance(exc, GraphRecursionError):
                try:
                    recovered = _recover_checkpoint_result(agent, internal_config)
                except Exception as recovery_exc:
                    LOG.warning(
                        "Checkpoint recovery failed stage=%s attempt=%s error=%s",
                        stage_name,
                        attempt,
                        recovery_exc,
                    )
                    warnings.append(
                        _runtime_warning(
                            stage_name,
                            "checkpoint_recovery_failed",
                            str(recovery_exc),
                            attempt,
                        )
                    )
                else:
                    if recovered:
                        last_result = recovered
                        warnings.append(
                            _runtime_warning(
                                stage_name,
                                "checkpoint_recovered_after_recursion_limit",
                                attempt=attempt,
                            )
                        )
            if non_retryable_exceptions and isinstance(exc, non_retryable_exceptions):
                break
            continue
        last_result = dict(result or {})
        if expects_structured:
            structured = _to_dict(last_result.get("structured_response"))
            if structured:
                if attempt == 2:
                    warnings.append(_runtime_warning(stage_name, "structured_response_recovered_on_retry", attempt=attempt))
                return last_result, structured, "", warnings
            warnings.append(_runtime_warning(stage_name, "missing_structured_response", attempt=attempt))
            continue
        messages = list(last_result.get("messages") or [])
        answer = _last_ai_text(messages)
        if clean_text(answer):
            if attempt == 2:
                warnings.append(_runtime_warning(stage_name, "answer_recovered_on_retry", attempt=attempt))
            return last_result, {}, answer, warnings
        if stop_on_loop_guard:
            guard_reason = clean_text(last_result.get("sales_loop_guard_reason"))
            if guard_reason:
                warnings.append(_runtime_warning(stage_name, "sales_loop_guarded", guard_reason, attempt))
                break
        warnings.append(_runtime_warning(stage_name, "empty_answer", attempt=attempt))
    if last_result and not expects_structured and not clean_text(answer):
        answer = _last_ai_text(list(last_result.get("messages") or []))
    return last_result, {}, clean_text(answer), warnings

def _tool_contract_sections(locale: str, state: GazAgentState) -> List[str]:
    sections = [get_prompt(locale, "tool_rules")]
    prompt_keys: List[str] = ["tool_pricing_bi", "tool_web_search"]
    for tool_name in state.get("allowed_tool_names") or []:
        key = _TOOL_PROMPT_KEYS.get(tool_name)
        if key:
            prompt_keys.append(key)
    seen: set[str] = set()
    for key in prompt_keys:
        if key in seen:
            continue
        sections.append(get_prompt(locale, key))
        seen.add(key)
    return sections


def _source_ladder_prompt_sections(locale: str) -> List[str]:
    return [get_prompt(locale, "source_ladder_policy")]


def _allowed_family_prompt_sections(locale: str, allowed_family_ids: Optional[List[str]]) -> List[str]:
    labels = format_allowed_product_names(allowed_family_ids)
    if not labels:
        return []
    joined = ", ".join(labels)
    if locale == "ru":
        return [
            (
                "ОГРАНИЧЕНИЕ ПРОДУКТОВОГО СКОУПА:\n"
                f"В текущей конфигурации агента разрешено обсуждать только следующие продуктовые семейства: {joined}.\n"
                "Запрещено продвигать, рекламировать, рекомендовать к покупке, предлагать как решение или вести самостоятельное продуктовое обсуждение моделей и семейств вне этого списка.\n"
                "Внешние модели и семейства вне разрешённого списка можно использовать только как фактический контекст сравнения с разрешёнными семействами ГАЗ: допускается запрашивать и сообщать их характеристики, цены, сервисные условия и другие подтверждённые факты именно для сравнения.\n"
                "Если пользователь просит именно продвигать, советовать к покупке или подробно презентовать внешний продукт сам по себе, прямо скажите, что в текущей конфигурации агента такая информация недоступна."
            )
        ]
    return [
        (
            "PRODUCT SCOPE RESTRICTION:\n"
            f"In the current agent configuration, you may discuss only these product families: {joined}.\n"
            "You must not promote, recommend for purchase, pitch as a solution, or conduct standalone product-selling discussion for models or families outside this list.\n"
            "External models and families outside the allowed list may be used only as factual comparison context against the allowed GAZ families: it is permitted to retrieve and report their specifications, prices, service terms, and other confirmed facts strictly for comparison.\n"
            "If the user asks you to promote, recommend, or fully present an external product on its own, state plainly that this is unavailable in the current agent configuration."
        )
    ]



def _build_turn_intent_extractor(
    model: BaseChatModel,
    summary_model: BaseChatModel,
    locale: str,
    structured_output_strategy: StructuredOutputStrategy,
    allowed_family_ids: Optional[List[str]],
):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        return compose_prompt(
            locale,
            "turn_intent_extractor",
            _turn_intent_payload(request.state),
            [
                *_retry_prompt_sections(request.state),
                *_allowed_family_prompt_sections(locale, allowed_family_ids),
                *_source_ladder_prompt_sections(locale),
            ],
        )

    return create_agent(
        model=model,
        tools=None,
        middleware=[*_build_middlewares(locale, summary_model), prompt, *_structured_output_middleware(structured_output_strategy)],
        response_format=_structured_response_format(TurnIntentExtractionResult, structured_output_strategy),
        state_schema=GazAgentState,
    )



def _build_sales_response_agent(
    model: BaseChatModel,
    summary_model: BaseChatModel,
    locale: str,
    tools: List[Any],
    allowed_family_ids: Optional[List[str]],
):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        return compose_prompt(
            locale,
            "sales_response_agent",
            _compact_payload(request.state),
            [
                *_retry_prompt_sections(request.state),
                *_allowed_family_prompt_sections(locale, allowed_family_ids),
                *_source_ladder_prompt_sections(locale),
                *_tool_contract_sections(locale, request.state),
            ],
        )

    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            *_build_middlewares(locale, summary_model),
            prompt,
        ],
        state_schema=GazAgentState,
    )
def create_init_node(docs_client: GazDocumentsClient):
    def init_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        configuration = (config.get("configurable") or {}) if isinstance(config, dict) else {}
        locale = resolve_locale(configuration.get("locale") or state.get("locale") or DEFAULT_LOCALE)
        result: GazAgentState = {
            "locale": locale,
            "reset_done": False,
            "llm_retry_instruction": "",
            "runtime_warnings": [],
            "read_attempts_by_candidate": {},
            "read_focus_keys_this_turn": [],
            "search_keys_this_turn": [],
            "tool_limit_hits": [],
            "sales_loop_guard_reason": None,
            "research_layer": "portfolio_baseline",
            "source_strategy": state.get("source_strategy") or "docs_first",
            "source_reason": state.get("source_reason") or "",
            "mentioned_models": list(state.get("mentioned_models") or []),
            "requested_facts": list(state.get("requested_facts") or []),
        }
        slots = dict(state.get("slots") or {})
        user_role = clean_text(configuration.get("user_role"))
        if user_role and user_role != "default" and not clean_text(slots.get("decision_role")):
            slots["decision_role"] = user_role
        result["slots"] = slots
        if "intent_flags" not in state:
            result["intent_flags"] = {
                "requested_price": False,
                "requested_materials": False,
                "asks_for_recommendation": False,
                "requested_portfolio_overview": False,
                "requested_financing": False,
                "challenged_questions": False,
                "expressed_friction": False,
                "requested_comparison_table": False,
                "requested_specs": False,
                "requested_versions": False,
                "requested_competitor_comparison": False,
                "expressed_impatience": False,
                "requested_concrete_numbers": False,
                "threatened_competitor_switch": False,
            }
        if "stage" not in state:
            result["stage"] = "OPENING"
        messages = state.get("messages") or []
        last_user = next((message for message in reversed(messages) if getattr(message, "type", None) == "human"), None)
        result["last_user_text"] = extract_text(last_user) if last_user else None
        result["sales_context_baseline"] = build_sales_context_baseline(locale, slots, state.get("problem_summary") or "", state.get("current_client_intent") or "overview")
        if _is_reset_requested(state):
            result["reset_done"] = True
            return result
        try:
            status = docs_client.get_collection_status()
            result["docs_status"] = {
                "service_available": True,
                "collection_available": bool(status.get("available")),
                "doc_count": status.get("doc_count", 0),
            }
            if not result["docs_status"]["collection_available"]:
                result["runtime_warnings"] = _merge_runtime_warnings(
                    result,
                    [_runtime_warning("gaz:init", "docs_collection_unavailable")],
                )
        except Exception as exc:
            LOG.warning("GAZ docs status check failed: %s", exc)
            result["docs_status"] = {
                "service_available": False,
                "collection_available": False,
                "doc_count": 0,
            }
            result["runtime_warnings"] = _merge_runtime_warnings(
                result,
                [_runtime_warning("gaz:init", "docs_status_failed", str(exc))],
            )
        return result

    return init_node
def route_from_init(state: GazAgentState) -> str:
    if state.get("reset_done"):
        return "reset"
    if not state.get("greeted"):
        return "opening"
    return "turn_intent"



def reset_node(state: GazAgentState, config: RunnableConfig) -> GazAgentState:
    all_msg_ids = [message.id for message in state.get("messages") or [] if getattr(message, "id", None)]
    return {
        "messages": [RemoveMessage(id=message_id) for message_id in all_msg_ids],
        "stage": "OPENING",
        "greeted": False,
        "reset_done": True,
        "last_user_text": None,
        "slots": {},
        "missing_slots": [],
        "intent_flags": {
            "requested_price": False,
            "requested_materials": False,
            "asks_for_recommendation": False,
            "requested_portfolio_overview": False,
            "requested_financing": False,
            "challenged_questions": False,
            "expressed_friction": False,
            "requested_comparison_table": False,
            "requested_specs": False,
            "requested_versions": False,
            "requested_competitor_comparison": False,
            "expressed_impatience": False,
            "requested_concrete_numbers": False,
            "threatened_competitor_switch": False,
        },
        "problem_summary": "",
        "current_client_intent": None,
        "customer_temperature": None,
        "clarification_allowed": False,
        "provisional_recommendations": [],
        "mentioned_models": [],
        "requested_facts": [],
        "source_strategy": "docs_first",
        "source_reason": "",
        "active_branch": None,
        "branch_conflict": [],
        "search_query": None,
        "material_candidates": [],
        "material_reads": [],
        "allowed_material_ids": [],
        "sales_digests": [],
        "comparison_digests": [],
        "product_snapshots": [],
        "composite_tool_traces": [],
        "shortlist": [],
        "followup_pack": {},
        "sales_context_baseline": {},
        "research_layer": "portfolio_baseline",
        "research_status": {},
        "read_attempts_by_candidate": {},
        "read_focus_keys_this_turn": [],
        "search_keys_this_turn": [],
        "tool_limit_hits": [],
        "sales_loop_guard_reason": None,
        "allowed_tool_names": [],
        "tool_calls_this_turn": [],
        "trace": [],
        "llm_retry_instruction": "",
        "runtime_warnings": [],
    }



def opening_node(state: GazAgentState, config: RunnableConfig) -> GazAgentState:
    locale = resolve_locale(state.get("locale"))
    _emit_custom("gaz_stage", stage="OPENING")
    trace = _append_trace(
        state,
        stage_before="OPENING",
        stage_after="SELL",
        allowed_tools=[],
        tool_calls=[],
        response_summary="Opening message sent",
        policy_checks_passed=["sales_opening_sent"],
    )
    return {
        "messages": [AIMessage(content=get_text(locale, "opening_message"))],
        "greeted": True,
        "stage": "SELL",
        "trace": trace,
    }



def create_turn_intent_node(turn_intent_extractor, all_tool_names: List[str]):
    def turn_intent_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        last_user_text = clean_text(state.get("last_user_text"))
        if not last_user_text:
            return {
                "missing_slots": compute_missing_slots(state.get("slots") or {}),
                "allowed_tool_names": list(all_tool_names),
                "llm_retry_instruction": "",
                "runtime_warnings": [],
            }
        _result, structured, _answer, warnings = _invoke_agent_with_retry(
            turn_intent_extractor,
            state,
            config,
            runtime,
            stage_name="gaz:turn_intent",
            extra_tags=["gaz:turn_intent"],
            retry_instruction="Return only the structured turn-intent object. No prose. Fill only explicit facts and leave unknown fields empty.",
            expects_structured=True,
        )
        if not structured:
            current_flags = dict(state.get("intent_flags") or {})
            current_intent = state.get("current_client_intent") or infer_client_intent(current_flags, last_user_text)
            current_temperature = state.get("customer_temperature") or infer_customer_temperature(current_flags, last_user_text)
            problem_summary = state.get("problem_summary") or last_user_text
            return {
                "stage": _STAGE_BY_INTENT.get(current_intent, "SELL"),
                "problem_summary": problem_summary,
                "missing_slots": compute_missing_slots(state.get("slots") or {}),
                "current_client_intent": current_intent,
                "customer_temperature": current_temperature,
                "sales_context_baseline": build_sales_context_baseline(state.get("locale"), state.get("slots") or {}, problem_summary, current_intent),
                "research_layer": "portfolio_baseline",
                "tool_calls_this_turn": [],
                "mentioned_models": list(state.get("mentioned_models") or []),
                "requested_facts": list(state.get("requested_facts") or []),
                "source_strategy": state.get("source_strategy") or "docs_first",
                "source_reason": "turn intent extractor returned no structured output; using previous/default source strategy",
                "allowed_tool_names": list(all_tool_names),
                "read_attempts_by_candidate": {},
                "read_focus_keys_this_turn": [],
                "search_keys_this_turn": [],
                "tool_limit_hits": [],
                "sales_loop_guard_reason": None,
                "llm_retry_instruction": "",
                "runtime_warnings": _merge_runtime_warnings(state, warnings),
            }
        updated_slots = merge_slots(state.get("slots") or {}, _to_dict(structured.get("slot_updates")))
        updated_flags = merge_flags(state.get("intent_flags") or {}, _to_dict(structured.get("intent_flags")))
        inferred_intent = infer_client_intent(updated_flags, last_user_text)
        inferred_temperature = infer_customer_temperature(updated_flags, last_user_text)
        current_intent = structured.get("current_client_intent") or inferred_intent
        current_temperature = _max_temperature(structured.get("customer_temperature") or inferred_temperature, inferred_temperature)
        problem_summary = clean_text(structured.get("problem_summary_candidate")) or state.get("problem_summary") or last_user_text
        source_strategy = structured.get("source_strategy") or "docs_first"
        if source_strategy not in {"model_bi_first", "selection_docs_first", "docs_first"}:
            source_strategy = "docs_first"
        mentioned_models = [clean_text(item) for item in structured.get("mentioned_models") or [] if clean_text(item)]
        requested_facts = [clean_text(item) for item in structured.get("requested_facts") or [] if clean_text(item)]
        return {
            "stage": _STAGE_BY_INTENT.get(current_intent, "SELL"),
            "slots": updated_slots,
            "intent_flags": updated_flags,
            "problem_summary": problem_summary,
            "missing_slots": compute_missing_slots(updated_slots),
            "current_client_intent": current_intent,
            "customer_temperature": current_temperature,
            "mentioned_models": mentioned_models,
            "requested_facts": requested_facts,
            "source_strategy": source_strategy,
            "source_reason": clean_text(structured.get("source_reason")),
            "sales_context_baseline": build_sales_context_baseline(state.get("locale"), updated_slots, problem_summary, current_intent),
            "research_layer": "portfolio_baseline",
            "material_candidates": [],
            "material_reads": [],
            "allowed_material_ids": [],
            "shortlist": [],
            "followup_pack": {},
            "tool_calls_this_turn": [],
            "allowed_tool_names": list(all_tool_names),
            "read_attempts_by_candidate": {},
            "read_focus_keys_this_turn": [],
            "search_keys_this_turn": [],
            "tool_limit_hits": [],
            "sales_loop_guard_reason": None,
            "llm_retry_instruction": "",
            "runtime_warnings": _merge_runtime_warnings(state, warnings),
        }

    return turn_intent_node

def create_sales_response_node(sales_response_agent):
    def sales_response_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        _emit_custom(
            "gaz_stage",
            stage=state.get("stage") or "SELL",
            intent=state.get("current_client_intent"),
            source_strategy=state.get("source_strategy"),
        )
        result, _structured, answer, warnings = _invoke_agent_with_retry(
            sales_response_agent,
            state,
            config,
            runtime,
            stage_name="gaz:sales_response",
            extra_tags=[f"gaz:sales:{state.get('stage') or 'sell'}"],
            retry_instruction="Write one direct customer-facing answer now. Do not return empty output and do not mention internal issues.",
            expects_structured=False,
            non_retryable_exceptions=(GraphRecursionError,),
            stop_on_loop_guard=True,
        )
        merged_warnings = _merge_runtime_warnings(state, warnings)
        tool_limit_hits = list((result or {}).get("tool_limit_hits") or state.get("tool_limit_hits") or [])
        loop_stop_reason = _derive_sales_loop_guard_reason(result or state, warnings, tool_limit_hits)
        if clean_text(answer):
            result["runtime_warnings"] = merged_warnings
            result["tool_limit_hits"] = tool_limit_hits
            result["sales_loop_guard_reason"] = loop_stop_reason or result.get("sales_loop_guard_reason")
            result["llm_retry_instruction"] = ""
            return result

        if (result or {}).get("messages"):
            sanitized_messages, pruned_tool_call_ids = _sanitize_messages_for_followup(
                list(result.get("messages") or [])
            )
            recovered_answer = _last_ai_text(sanitized_messages)
            if pruned_tool_call_ids:
                merged_warnings = _merge_runtime_warnings(
                    {"runtime_warnings": merged_warnings},
                    [
                        _runtime_warning(
                            "gaz:sales_response",
                            "dangling_tool_calls_removed_before_fallback",
                            json.dumps(pruned_tool_call_ids, ensure_ascii=False),
                        )
                    ],
                )
            if clean_text(recovered_answer):
                return {
                    "messages": sanitized_messages,
                    "runtime_warnings": merged_warnings,
                    "tool_limit_hits": tool_limit_hits,
                    "sales_loop_guard_reason": loop_stop_reason or None,
                    "llm_retry_instruction": "",
                }

        if loop_stop_reason:
            LOG.warning("sales_response loop stopped; using deterministic fallback reason=%s", loop_stop_reason)
            merged_warnings = _merge_runtime_warnings(
                {"runtime_warnings": merged_warnings},
                [_runtime_warning("gaz:sales_response", "sales_response_deterministic_fallback", loop_stop_reason)],
            )
        locale = resolve_locale(state.get("locale"))
        merged_warnings.append(_runtime_warning("gaz:sales_response", "generic_continuation_message_used"))
        return {
            "messages": [AIMessage(content=get_text(locale, "continue_message"))],
            "runtime_warnings": merged_warnings,
            "tool_limit_hits": tool_limit_hits,
            "sales_loop_guard_reason": loop_stop_reason or None,
            "llm_retry_instruction": "",
        }

    return sales_response_node



def create_trace_finalize_node():
    def trace_finalize_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        messages = list(state.get("messages") or [])
        answer = _last_ai_text(messages)
        runtime_warnings = list(state.get("runtime_warnings") or [])
        stage_before = state.get("stage") or "SELL"
        trace_state: GazAgentState = dict(state)
        trace_state["runtime_warnings"] = runtime_warnings
        trace = _append_trace(
            trace_state,
            stage_before=stage_before,
            stage_after="SELL",
            allowed_tools=list(state.get("allowed_tool_names") or []),
            tool_calls=list(state.get("tool_calls_this_turn") or []),
            response_summary=answer,
            policy_checks_passed=["sales_answer_finalized_without_repair"],
        )
        result: GazAgentState = {
            "stage": "SELL",
            "trace": trace,
            "tool_calls_this_turn": [],
            "llm_retry_instruction": "",
            "runtime_warnings": runtime_warnings,
            "read_attempts_by_candidate": {},
            "read_focus_keys_this_turn": [],
            "search_keys_this_turn": [],
            "tool_limit_hits": [],
            "sales_loop_guard_reason": None,
        }
        return result

    return trace_finalize_node

def initialize_agent(
    provider: ModelType = ModelType.GPT,
    use_platform_store: bool = False,
    locale: str = DEFAULT_LOCALE,
    checkpoint_saver=None,
    *,
    streaming: bool = True,
    docs_collection: str = "gaz",
    structured_output_strategy: StructuredOutputStrategy = "provider_then_tool",
):
    locale_key = resolve_locale(locale)
    log_name = f"gaz_agent_{time.strftime('%Y%m%d%H%M')}"
    json_handler = JSONFileTracer(f"./logs/{log_name}")
    callback_handlers = [StreamWriterCallbackHandler(), json_handler]
    if config.LANGFUSE_URL:
        _ = Langfuse(
            public_key=config.LANGFUSE_PUBLIC,
            secret_key=config.LANGFUSE_SECRET,
            host=config.LANGFUSE_URL,
        )
        callback_handlers.append(CallbackHandler())

    memory = None if use_platform_store else checkpoint_saver or MemorySaver()
    base_url = os.environ.get("GAZ_DOCUMENTS_SERVICE_URL", "http://127.0.0.1:8081")
    docs_client = GazDocumentsClient(base_url=base_url, collection_id=docs_collection)
    pricing_db_url = os.environ.get("GAZ_PRICING_BI_DATABASE_URL", _DEFAULT_GAZ_PRICING_DB_URL)
    pricing_prompt_context_path = Path(
        os.environ.get("GAZ_PRICING_BI_PROMPT_CONTEXT_PATH") or str(_DEFAULT_GAZ_PRICING_PROMPT_CONTEXT_PATH)
    )
    if not pricing_prompt_context_path.is_absolute():
        pricing_prompt_context_path = (_REPO_ROOT / pricing_prompt_context_path).resolve()
    pricing_prompt_context = pricing_prompt_context_path.read_text(encoding="utf-8")
    allowed_family_ids = parse_allowed_product_names_env(os.environ.get("GAZ_AGENT_ALLOWED_PRODUCT_NAMES"))
    pricing_bi_init_context = {
        "database_url": pricing_db_url,
        "database_prompt_context": pricing_prompt_context,
        "return_files": False,
        "return_images": False,
    }

    from agents.bi_agent.bi_agent import initialize_agent as initialize_bi_agent

    mini_llm = get_llm(model="mini", provider=provider.value, temperature=0.0, streaming=streaming)
    base_llm = get_llm(model="base", provider=provider.value, temperature=0.2, streaming=streaming, reasoning="medium")
    summary_llm = get_llm(model="mini", provider=provider.value, temperature=0.0, streaming=False)
    pricing_bi_agent = initialize_bi_agent(
        provider=provider,
        use_platform_store=False,
        notify_on_reload=False,
        init_context=pricing_bi_init_context,
    )

    turn_intent_extractor = _build_turn_intent_extractor(
        mini_llm,
        summary_llm,
        locale_key,
        structured_output_strategy,
        allowed_family_ids,
    )

    tool_registry: Dict[str, Any] = {
        "get_sales_catalog_overview": build_sales_catalog_overview_tool(locale_key),
        "get_sales_landscape": build_sales_landscape_tool(locale_key, docs_client, allowed_family_ids),
        "compare_product_directions": build_compare_product_directions_tool(locale_key, docs_client, allowed_family_ids),
        "collect_product_snapshot": build_collect_product_snapshot_tool(locale_key, docs_client, allowed_family_ids),
        "search_sales_materials": build_search_sales_materials_tool(locale_key, docs_client, allowed_family_ids),
        "read_material": build_read_material_tool(locale_key, docs_client, allowed_family_ids),
        "classify_problem_branch": build_classify_problem_branch_tool(locale_key),
        "get_branch_pack": build_branch_pack_tool(locale_key, docs_client, allowed_family_ids),
        "build_solution_shortlist": build_solution_shortlist_tool(),
        "build_followup_pack": build_followup_pack_tool(),
        "query_pricing_bi": build_query_pricing_bi_tool(
            locale_key,
            pricing_bi_agent,
            dict(pricing_bi_init_context),
            _PRICING_BI_THREAD_SUFFIX,
            allowed_family_ids,
        ),
        "web_search": build_web_search_tool(locale_key, config.YA_API_KEY, config.YA_FOLDER_ID),
    }
    sales_response_agent = _build_sales_response_agent(
        base_llm,
        summary_llm,
        locale_key,
        list(tool_registry.values()),
        allowed_family_ids,
    )

    builder = StateGraph(GazAgentState, config_schema=ConfigSchema)
    builder.add_node("init", create_init_node(docs_client))
    builder.add_node("reset", reset_node)
    builder.add_node("opening", opening_node)
    builder.add_node("turn_intent", create_turn_intent_node(turn_intent_extractor, list(tool_registry.keys())))
    builder.add_node("sales_response", create_sales_response_node(sales_response_agent))
    builder.add_node("trace_finalize", create_trace_finalize_node())

    builder.add_edge(START, "init")
    builder.add_conditional_edges(
        "init",
        route_from_init,
        {
            "reset": "reset",
            "opening": "opening",
            "turn_intent": "turn_intent",
        },
    )
    builder.add_edge("reset", END)
    builder.add_edge("opening", END)
    builder.add_edge("turn_intent", "sales_response")
    builder.add_edge("sales_response", "trace_finalize")
    builder.add_edge("trace_finalize", END)

    graph = builder.compile(checkpointer=memory, debug=False, name="gaz_agent").with_config({"callbacks": callback_handlers})
    return graph
