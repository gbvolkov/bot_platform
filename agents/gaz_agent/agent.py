from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, SummarizationMiddleware, ToolCallLimitMiddleware, dynamic_prompt
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
from langgraph.types import interrupt

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
    build_allowed_tool_names,
    build_sales_context_baseline,
    clamp_answer_depth,
    clean_text,
    classify_branch,
    compute_missing_slots,
    derive_answer_depth,
    derive_research_layer,
    derive_work_mode,
    evaluate_hitl_gate,
    infer_client_intent,
    infer_customer_temperature,
    is_affirmative,
    is_negative,
    merge_flags,
    merge_slots,
    normalize_provisional_recommendations,
    prioritize_missing_slots,
    select_active_tool_names,
    update_provisional_recommendations,
)
from .locales import DEFAULT_LOCALE, resolve_locale
from .middleware import build_palimpsest_middleware
from .prompts import compose_prompt, get_prompt, get_text
from .schemas import AnswerPlanResult, PolicyValidationResult, TurnIntentExtractionResult
from .state import GazAgentState
from .tools import (
    build_branch_pack_tool,
    build_classify_problem_branch_tool,
    build_collect_product_snapshot_tool,
    build_compare_product_directions_tool,
    build_followup_pack_tool,
    build_read_material_tool,
    build_sales_catalog_overview_tool,
    build_sales_landscape_tool,
    build_search_sales_materials_tool,
    build_solution_shortlist_tool,
)

LOG = logging.getLogger(__name__)
StructuredOutputStrategy = Literal["auto", "provider", "tool", "provider_then_tool"]
_DEPTH_ORDER = {"broad": 0, "bounded": 1, "justified": 2, "deep_research": 3}
_TEMPERATURE_ORDER = {"neutral": 0, "impatient": 1, "irritated": 2, "competitor_risk": 3}
_VALID_BRANCH_HINTS = {
    "tco",
    "configuration",
    "comparison",
    "service_risk",
    "internal_approval",
    "passenger_route",
    "special_body",
    "special_conditions",
    "unknown_selection",
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


class SalesToolSelectionMiddleware(AgentMiddleware):
    def __init__(self, tool_registry: Dict[str, Any]) -> None:
        super().__init__()
        self._tool_registry = dict(tool_registry)

    def wrap_model_call(self, request, handler):
        state = request.state
        allowed_tool_names = list(state.get("allowed_tool_names") or [])
        if not allowed_tool_names:
            return handler(request)
        active_tool_names = select_active_tool_names(
            state.get("current_client_intent") or "overview",
            state.get("answer_depth") or "broad",
            state.get("stage") or "SELL",
            planned_tools=allowed_tool_names,
            has_sales_digest=bool(state.get("sales_digests")),
            has_comparison_digest=bool(state.get("comparison_digests")),
            has_product_snapshot=bool(state.get("product_snapshots")),
            has_material_candidates=bool(state.get("material_candidates")),
            has_material_reads=bool(state.get("material_reads")),
            has_branch_pack=bool((state.get("research_status") or {}).get("last_branch_pack")),
            has_shortlist=bool(state.get("shortlist")),
            has_followup=bool((state.get("followup_pack") or {}).get("documents")),
        )
        selected = [self._tool_registry[name] for name in active_tool_names if name in self._tool_registry]
        if not selected:
            return handler(request)
        return handler(request.override(tools=selected))



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



_QUESTION_POLICY_MARKERS = (
    "clarifying question",
    "more than one",
    "email twice",
    "same contact",
    "????????",
    "????????",
    "email",
)



def _is_question_policy_violation(violation: str) -> bool:
    lowered = clean_text(violation).lower()
    return bool(lowered) and any(marker in lowered for marker in _QUESTION_POLICY_MARKERS)



def _is_soft_question_validation(validation: Dict[str, Any]) -> bool:
    violations = [clean_text(item) for item in validation.get("violations") or [] if clean_text(item)]
    return bool(violations) and all(_is_question_policy_violation(item) for item in violations)



def _question_guidance(locale: str, validation: Dict[str, Any]) -> str:
    suggested = clean_text(validation.get("suggested_fix"))
    if suggested:
        return suggested
    return get_prompt(locale, "question_budget_guidance")



def _replace_last_ai_message(messages: List[Any], new_text: str) -> List[Any]:
    updated = list(messages)
    for index in range(len(updated) - 1, -1, -1):
        message = updated[index]
        if isinstance(message, AIMessage):
            replacement = AIMessage(content=new_text)
            message_id = getattr(message, "id", None)
            if message_id:
                return [RemoveMessage(id=message_id), replacement]
            return [replacement]
    return [AIMessage(content=new_text)]



def _validation_feedback_payload(validation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "violations": [clean_text(item) for item in validation.get("violations") or [] if clean_text(item)],
        "suggested_fix": clean_text(validation.get("suggested_fix")) or None,
    }



def _repair_answer(
    repair_agent,
    state: GazAgentState,
    config: RunnableConfig,
    runtime: Runtime | None,
    draft_answer: str,
    validation: Dict[str, Any],
) -> tuple[str, List[Dict[str, Any]]]:
    internal_config = build_internal_invoke_config(config, extra_tags=["gaz:repair"])
    repair_state: GazAgentState = dict(state)
    repair_state["draft_answer"] = draft_answer
    repair_state["validation_feedback"] = _validation_feedback_payload(validation)
    repair_state["validator_guidance"] = clean_text(validation.get("suggested_fix")) or repair_state.get("validator_guidance") or ""
    try:
        result = repair_agent.invoke(repair_state, config=internal_config, context=getattr(runtime, "context", None))
    except Exception as exc:
        LOG.warning("Repair invoke failed error=%s", exc)
        return "", [_runtime_warning("gaz:repair", "repair_path_failed", str(exc))]
    repaired_answer = _last_ai_text(list(result.get("messages") or []))
    if clean_text(repaired_answer):
        return repaired_answer, []
    return "", [_runtime_warning("gaz:repair", "empty_repair_answer")]



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
            "planned_answer_depth": state.get("planned_answer_depth"),
            "answer_depth": state.get("answer_depth"),
            "research_layer": state.get("research_layer"),
            "slots_changed": state.get("slots") or {},
            "allowed_tools": allowed_tools,
            "tool_calls": tool_calls,
            "response_summary": response_summary,
            "hitl_eligible": bool(state.get("hitl_eligible")),
            "needs_hitl_wait_confirmation": bool(state.get("needs_hitl_wait_confirmation")),
            "hitl_blocked_by_temperature": bool(state.get("hitl_blocked_by_temperature")),
            "hitl_blocked_by_first_turn_budget": bool(state.get("hitl_blocked_by_first_turn_budget")),
            "hitl_blocked_by_missing_prior_search": bool(state.get("hitl_blocked_by_missing_prior_search")),
            "hitl_trigger_kind": state.get("hitl_trigger_kind"),
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



def _max_depth(first: str, second: str) -> str:
    return first if _DEPTH_ORDER.get(first, 0) >= _DEPTH_ORDER.get(second, 0) else second



def _max_temperature(first: str, second: str) -> str:
    return first if _TEMPERATURE_ORDER.get(first, 0) >= _TEMPERATURE_ORDER.get(second, 0) else second



def _latest_history_item(values: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    if not values:
        return {}
    item = values[-1]
    return dict(item or {})


def _normalize_branch_hint(value: Any) -> Optional[str]:
    hint = clean_text(value)
    if hint in _VALID_BRANCH_HINTS:
        return hint
    return None



def _compact_payload(state: GazAgentState) -> Dict[str, Any]:
    research_status = state.get("research_status") or {}
    latest_landscape = _latest_history_item(state.get("sales_digests"))
    latest_comparison = _latest_history_item(state.get("comparison_digests"))
    latest_snapshot = _latest_history_item(state.get("product_snapshots"))
    return {
        "locale": state.get("locale", DEFAULT_LOCALE),
        "stage": state.get("stage", "SELL"),
        "current_client_intent": state.get("current_client_intent"),
        "answer_depth": state.get("answer_depth"),
        "customer_temperature": state.get("customer_temperature"),
        "problem_summary": state.get("problem_summary") or "",
        "slots": state.get("slots") or {},
        "missing_slots": prioritize_missing_slots(state.get("missing_slots") or []),
        "intent_flags": state.get("intent_flags") or {},
        "clarification_allowed": bool(state.get("clarification_allowed")),
        "llm_retry_instruction": clean_text(state.get("llm_retry_instruction")),
        "validator_guidance": clean_text(state.get("validator_guidance")),
        "provisional_recommendations": state.get("provisional_recommendations") or [],
        "validation_feedback": state.get("validation_feedback") or {},
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
    }



def _answer_plan_payload(state: GazAgentState) -> Dict[str, Any]:
    return {
        "locale": state.get("locale", DEFAULT_LOCALE),
        "problem_summary": state.get("problem_summary") or "",
        "slots": state.get("slots") or {},
        "intent_flags": state.get("intent_flags") or {},
        "missing_slots": prioritize_missing_slots(state.get("missing_slots") or []),
        "current_client_intent": state.get("current_client_intent"),
        "customer_temperature": state.get("customer_temperature"),
        "sales_context_baseline": state.get("sales_context_baseline") or {},
        "last_user_text": state.get("last_user_text") or "",
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



def _merge_tool_limit_hits(state: GazAgentState, hits: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    merged = list(state.get("tool_limit_hits") or [])
    merged.extend(hits or [])
    return merged



def _extract_tool_limit_hits(messages: List[Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = clean_text(getattr(message, "content", ""))
        if not content.startswith("Tool call limit exceeded."):
            continue
        tool_name = "__all__"
        match = re.search(r"'([^']+)'", content)
        if match:
            tool_name = match.group(1)
        hits.append({"tool_name": tool_name, "reason": "middleware_tool_call_limit"})
    return hits



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
        try:
            result = agent.invoke(invoke_state, config=internal_config, context=getattr(runtime, "context", None))
        except Exception as exc:
            LOG.warning("Agent invoke failed stage=%s attempt=%s error=%s", stage_name, attempt, exc)
            warning_code = "graph_recursion_limit" if isinstance(exc, GraphRecursionError) else "invoke_exception"
            warnings.append(_runtime_warning(stage_name, warning_code, str(exc), attempt))
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
            tool_limit_hits = _extract_tool_limit_hits(messages)
            guard_reason = clean_text(last_result.get("sales_loop_guard_reason"))
            if tool_limit_hits:
                warnings.append(
                    _runtime_warning(
                        stage_name,
                        "tool_limit_hit",
                        json.dumps(tool_limit_hits, ensure_ascii=False),
                        attempt,
                    )
                )
                break
            if guard_reason:
                warnings.append(_runtime_warning(stage_name, "sales_loop_guarded", guard_reason, attempt))
                break
        warnings.append(_runtime_warning(stage_name, "empty_answer", attempt=attempt))
    return last_result, {}, clean_text(answer), warnings

def _tool_contract_sections(locale: str, state: GazAgentState) -> List[str]:
    sections = [get_prompt(locale, "tool_rules")]
    for tool_name in state.get("allowed_tool_names") or []:
        key = _TOOL_PROMPT_KEYS.get(tool_name)
        if key:
            sections.append(get_prompt(locale, key))
    return sections



def _build_turn_intent_extractor(model: BaseChatModel, summary_model: BaseChatModel, locale: str, structured_output_strategy: StructuredOutputStrategy):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        return compose_prompt(locale, "turn_intent_extractor", _turn_intent_payload(request.state), _retry_prompt_sections(request.state))

    return create_agent(
        model=model,
        tools=None,
        middleware=[*_build_middlewares(locale, summary_model), prompt, *_structured_output_middleware(structured_output_strategy)],
        response_format=_structured_response_format(TurnIntentExtractionResult, structured_output_strategy),
        state_schema=GazAgentState,
    )



def _build_answer_planner(model: BaseChatModel, summary_model: BaseChatModel, locale: str, structured_output_strategy: StructuredOutputStrategy):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        return compose_prompt(locale, "answer_planner", _answer_plan_payload(request.state), _retry_prompt_sections(request.state))

    return create_agent(
        model=model,
        tools=None,
        middleware=[*_build_middlewares(locale, summary_model), prompt, *_structured_output_middleware(structured_output_strategy)],
        response_format=_structured_response_format(AnswerPlanResult, structured_output_strategy),
        state_schema=GazAgentState,
    )



def _build_policy_validator(model: BaseChatModel, summary_model: BaseChatModel, locale: str, structured_output_strategy: StructuredOutputStrategy):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        payload = _compact_payload(request.state)
        payload["draft_answer"] = request.state.get("draft_answer") or ""
        return compose_prompt(locale, "sales_validator", payload, _retry_prompt_sections(request.state))

    return create_agent(
        model=model,
        tools=None,
        middleware=[*_build_middlewares(locale, summary_model), prompt, *_structured_output_middleware(structured_output_strategy)],
        response_format=_structured_response_format(PolicyValidationResult, structured_output_strategy),
        state_schema=GazAgentState,
    )



def _build_sales_response_agent(model: BaseChatModel, summary_model: BaseChatModel, locale: str, tools: List[Any], tool_registry: Dict[str, Any]):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        return compose_prompt(locale, "sales_response_agent", _compact_payload(request.state), [*_retry_prompt_sections(request.state), *_tool_contract_sections(locale, request.state)])

    return create_agent(
        model=model,
        tools=tools,
        middleware=[
            *_build_middlewares(locale, summary_model),
            ToolCallLimitMiddleware(run_limit=8, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="compare_product_directions", run_limit=1, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="collect_product_snapshot", run_limit=1, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="get_sales_landscape", run_limit=1, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="search_sales_materials", run_limit=3, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="read_material", run_limit=4, exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="get_branch_pack", run_limit=1, exit_behavior="continue"),
            SalesToolSelectionMiddleware(tool_registry),
            prompt,
        ],
        state_schema=GazAgentState,
    )



def _build_sales_repair_agent(model: BaseChatModel, summary_model: BaseChatModel, locale: str):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        payload = _compact_payload(request.state)
        payload["draft_answer"] = request.state.get("draft_answer") or ""
        payload["validation_feedback"] = request.state.get("validation_feedback") or {}
        return compose_prompt(locale, "sales_repair_agent", payload, _retry_prompt_sections(request.state))

    return create_agent(
        model=model,
        tools=None,
        middleware=[*_build_middlewares(locale, summary_model), prompt],
        state_schema=GazAgentState,
    )



def _build_sales_continue_agent(model: BaseChatModel, summary_model: BaseChatModel, locale: str):
    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        return compose_prompt(locale, "sales_continue_agent", _compact_payload(request.state), _retry_prompt_sections(request.state))

    return create_agent(
        model=model,
        tools=None,
        middleware=[*_build_middlewares(locale, summary_model), prompt],
        state_schema=GazAgentState,
    )

def _validate_answer(validator, state: GazAgentState, config: RunnableConfig, runtime: Runtime | None, draft_answer: str) -> Dict[str, Any]:
    if not clean_text(draft_answer):
        return {
            "is_valid": True,
            "violations": [],
            "suggested_fix": None,
            "_runtime_warnings": [_runtime_warning("gaz:validator", "empty_draft_answer_skipped_validation")],
        }
    validator_state: GazAgentState = dict(state)
    validator_state["draft_answer"] = draft_answer
    _result, structured, _answer, warnings = _invoke_agent_with_retry(
        validator,
        validator_state,
        config,
        runtime,
        stage_name="gaz:validator",
        extra_tags=["gaz:validator"],
        retry_instruction="Return only the structured validation object. If uncertain, prefer is_valid=true with empty violations rather than prose.",
        expects_structured=True,
    )
    if not structured:
        return {
            "is_valid": True,
            "violations": [],
            "suggested_fix": None,
            "_runtime_warnings": warnings or [_runtime_warning("gaz:validator", "missing_structured_response_after_retry")],
        }
    if warnings:
        structured["_runtime_warnings"] = warnings
    return structured
def create_init_node(docs_client: GazDocumentsClient):
    def init_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        configuration = (config.get("configurable") or {}) if isinstance(config, dict) else {}
        locale = resolve_locale(configuration.get("locale") or state.get("locale") or DEFAULT_LOCALE)
        result: GazAgentState = {
            "locale": locale,
            "reset_done": False,
            "draft_answer": None,
            "validator_guidance": "",
            "validation_feedback": {},
            "llm_retry_instruction": "",
            "runtime_warnings": [],
            "read_attempts_by_candidate": {},
            "read_focus_keys_this_turn": [],
            "search_keys_this_turn": [],
            "tool_limit_hits": [],
            "sales_loop_guard_reason": None,
            "research_wait_approved": False,
            "research_wait_rejected": False,
            "research_layer": "portfolio_baseline",
            "planned_answer_depth": None,
            "hitl_eligible": False,
            "hitl_blocked_by_temperature": False,
            "hitl_blocked_by_first_turn_budget": False,
            "hitl_blocked_by_missing_prior_search": False,
            "hitl_trigger_kind": None,
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
        "planned_answer_depth": None,
        "answer_depth": None,
        "customer_temperature": None,
        "clarification_allowed": False,
        "provisional_recommendations": [],
        "hitl_eligible": False,
        "hitl_blocked_by_temperature": False,
        "hitl_blocked_by_first_turn_budget": False,
        "hitl_blocked_by_missing_prior_search": False,
        "hitl_trigger_kind": None,
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
        "needs_hitl_wait_confirmation": False,
        "hitl_reason": None,
        "research_wait_approved": False,
        "research_wait_rejected": False,
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



def create_turn_intent_node(turn_intent_extractor):
    def turn_intent_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        last_user_text = clean_text(state.get("last_user_text"))
        if not last_user_text:
            return {
                "missing_slots": compute_missing_slots(state.get("slots") or {}),
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
                "problem_summary": problem_summary,
                "missing_slots": compute_missing_slots(state.get("slots") or {}),
                "current_client_intent": current_intent,
                "customer_temperature": current_temperature,
                "sales_context_baseline": build_sales_context_baseline(state.get("locale"), state.get("slots") or {}, problem_summary, current_intent),
                "research_layer": "portfolio_baseline",
                "tool_calls_this_turn": [],
                "planned_answer_depth": None,
                "hitl_eligible": False,
                "hitl_blocked_by_temperature": False,
                "hitl_blocked_by_first_turn_budget": False,
                "hitl_blocked_by_missing_prior_search": False,
                "hitl_trigger_kind": None,
                "hitl_reason": None,
                "research_wait_approved": False,
                "research_wait_rejected": False,
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
        return {
            "slots": updated_slots,
            "intent_flags": updated_flags,
            "problem_summary": problem_summary,
            "missing_slots": compute_missing_slots(updated_slots),
            "current_client_intent": current_intent,
            "customer_temperature": current_temperature,
            "sales_context_baseline": build_sales_context_baseline(state.get("locale"), updated_slots, problem_summary, current_intent),
            "research_layer": "portfolio_baseline",
            "material_candidates": [],
            "material_reads": [],
            "allowed_material_ids": [],
            "shortlist": [],
            "followup_pack": {},
            "tool_calls_this_turn": [],
            "planned_answer_depth": None,
            "hitl_eligible": False,
            "hitl_blocked_by_temperature": False,
            "hitl_blocked_by_first_turn_budget": False,
            "hitl_blocked_by_missing_prior_search": False,
            "hitl_trigger_kind": None,
            "hitl_reason": None,
            "research_wait_approved": False,
            "research_wait_rejected": False,
            "read_attempts_by_candidate": {},
            "read_focus_keys_this_turn": [],
            "search_keys_this_turn": [],
            "tool_limit_hits": [],
            "sales_loop_guard_reason": None,
            "llm_retry_instruction": "",
            "runtime_warnings": _merge_runtime_warnings(state, warnings),
        }

    return turn_intent_node

def create_answer_plan_node(answer_planner, docs_client: GazDocumentsClient):
    def answer_plan_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        intent_flags = state.get("intent_flags") or {}
        last_user_text = clean_text(state.get("last_user_text"))
        _result, structured, _answer, warnings = _invoke_agent_with_retry(
            answer_planner,
            state,
            config,
            runtime,
            stage_name="gaz:answer_planner",
            extra_tags=["gaz:answer_planner"],
            retry_instruction="Return only the structured answer-plan object. Choose the minimum useful answer depth and do not return prose.",
            expects_structured=True,
        )

        current_intent = (structured.get("current_client_intent") if structured else None) or state.get("current_client_intent") or infer_client_intent(intent_flags, last_user_text)
        temperature = _max_temperature((structured.get("customer_temperature") if structured else None) or state.get("customer_temperature") or "neutral", infer_customer_temperature(intent_flags, last_user_text))

        branch_hint = _normalize_branch_hint(structured.get("branch_hint") if structured else None) or _normalize_branch_hint(state.get("active_branch"))
        classified_branch = None
        branch_conflict: List[str] = list(state.get("branch_conflict") or [])
        if structured and current_intent in {"recommendation", "next_step", "materials", "compare", "specs"}:
            classified_branch, branch_conflict, _reasoning = classify_branch(state.get("slots") or {}, state.get("problem_summary") or "")
            if not branch_hint and classified_branch and classified_branch != "unknown_selection":
                branch_hint = classified_branch

        research_status = dict(state.get("research_status") or {})
        has_prior_search = bool(research_status.get("has_prior_search")) or bool(state.get("material_candidates"))
        has_prior_read = bool(research_status.get("has_prior_read")) or bool(state.get("material_reads"))
        has_branch_basis = bool(
            (branch_hint and branch_hint != "unknown_selection")
            or (classified_branch and classified_branch != "unknown_selection")
            or research_status.get("last_branch_pack")
        )

        derived_depth = derive_answer_depth(current_intent, intent_flags, branch_hint)
        planned_answer_depth = _max_depth((structured.get("answer_depth") if structured else None) or derived_depth, derived_depth)
        answer_depth = clamp_answer_depth(
            current_intent,
            planned_answer_depth,
            has_prior_search=has_prior_search,
            has_prior_read=has_prior_read,
            has_branch_basis=has_branch_basis,
        )
        work_mode = derive_work_mode(current_intent, answer_depth)
        search_query = clean_text((structured.get("search_query") if structured else None)) or clean_text(state.get("problem_summary")) or last_user_text
        provisional = normalize_provisional_recommendations((structured.get("provisional_recommendations") if structured else []) or [])
        if structured and structured.get("should_offer_provisional_recommendations", True):
            if not provisional:
                provisional = list(state.get("provisional_recommendations") or [])
            if state.get("material_candidates"):
                provisional = update_provisional_recommendations(state.get("material_candidates") or [], provisional)
        else:
            provisional = list(state.get("provisional_recommendations") or [])

        hitl_gate = evaluate_hitl_gate(
            current_intent,
            temperature,
            intent_flags,
            research_status,
            research_wait_rejected=bool(state.get("research_wait_rejected")),
            has_material_candidates=bool(state.get("material_candidates")),
            has_material_reads=bool(state.get("material_reads")),
        )
        needs_hitl = False
        hitl_reason = None
        if structured and hitl_gate["needs_hitl_wait_confirmation"]:
            try:
                estimate = docs_client.estimate_research_cost(
                    query=search_query,
                    intended_depth=planned_answer_depth if planned_answer_depth in {"justified", "deep_research"} else answer_depth,
                    intent=current_intent,
                    families=provisional,
                    competitor=clean_text((state.get("slots") or {}).get("competitor")),
                )
                research_status.update(estimate)
                needs_hitl = bool(estimate.get("requires_hitl_wait_confirmation")) and not state.get("research_wait_rejected")
                hitl_reason = clean_text(estimate.get("rationale")) or None
            except Exception as exc:
                LOG.warning("Research cost estimation failed: %s", exc)
                warnings.append(_runtime_warning("gaz:answer_planner", "estimate_research_cost_failed", str(exc)))

        allowed_tool_names = build_allowed_tool_names(current_intent, answer_depth, work_mode)
        research_layer = derive_research_layer(
            has_sales_digest=bool(state.get("sales_digests")),
            has_comparison_digest=bool(state.get("comparison_digests")),
            has_product_snapshot=bool(state.get("product_snapshots")),
            has_material_candidates=bool(state.get("material_candidates")),
            has_material_reads=bool(state.get("material_reads")),
            has_branch_pack=bool(research_status.get("last_branch_pack")),
            has_shortlist=bool(state.get("shortlist")),
            has_followup=bool((state.get("followup_pack") or {}).get("documents")),
        )
        return {
            "stage": work_mode,
            "current_client_intent": current_intent,
            "planned_answer_depth": planned_answer_depth,
            "answer_depth": answer_depth,
            "customer_temperature": temperature,
            "clarification_allowed": bool(structured.get("clarification_allowed")) if structured else False,
            "provisional_recommendations": provisional,
            "sales_context_baseline": build_sales_context_baseline(state.get("locale"), state.get("slots") or {}, state.get("problem_summary") or search_query, current_intent),
            "search_query": search_query,
            "active_branch": branch_hint or (classified_branch if classified_branch != "unknown_selection" else None),
            "branch_conflict": branch_conflict,
            "research_status": research_status,
            "research_layer": research_layer,
            "hitl_eligible": bool(hitl_gate.get("hitl_eligible")),
            "hitl_blocked_by_temperature": bool(hitl_gate.get("hitl_blocked_by_temperature")),
            "hitl_blocked_by_first_turn_budget": bool(hitl_gate.get("hitl_blocked_by_first_turn_budget")),
            "hitl_blocked_by_missing_prior_search": bool(hitl_gate.get("hitl_blocked_by_missing_prior_search")),
            "hitl_trigger_kind": hitl_gate.get("hitl_trigger_kind"),
            "needs_hitl_wait_confirmation": needs_hitl,
            "hitl_reason": hitl_reason,
            "allowed_tool_names": allowed_tool_names,
            "llm_retry_instruction": "",
            "runtime_warnings": _merge_runtime_warnings(state, warnings),
        }

    return answer_plan_node


def route_after_answer_plan(state: GazAgentState) -> str:
    if state.get("needs_hitl_wait_confirmation"):
        return "hitl_wait"
    return "sales_response"



def hitl_wait_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
    locale = resolve_locale(state.get("locale"))
    trigger_kind = clean_text(state.get("hitl_trigger_kind")) or "deep_comparison_wait"
    question_key = f"{trigger_kind}_question"
    content_key = f"{trigger_kind}_content"
    payload = {
        "type": "choice",
        "question": get_text(locale, question_key),
        "content": get_text(locale, content_key),
    }
    user_response = interrupt(payload)
    user_response_text = str(user_response or "")
    approved = is_affirmative(user_response_text)
    rejected = is_negative(user_response_text)
    answer_depth = state.get("answer_depth") or "bounded"
    return {
        "messages": [HumanMessage(content=user_response_text)],
        "needs_hitl_wait_confirmation": False,
        "research_wait_approved": approved,
        "research_wait_rejected": rejected,
        "answer_depth": answer_depth,
        "stage": derive_work_mode(state.get("current_client_intent") or "overview", answer_depth),
        "research_status": {
            **dict(state.get("research_status") or {}),
            "wait_confirmation_response": user_response_text,
            "wait_confirmation_granted": approved,
        },
    }


def create_sales_response_node(sales_response_agent, sales_continue_agent):
    def sales_response_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        _emit_custom(
            "gaz_stage",
            stage=state.get("stage") or "SELL",
            intent=state.get("current_client_intent"),
            depth=state.get("answer_depth"),
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
        tool_limit_hits = _merge_tool_limit_hits(result or state, _extract_tool_limit_hits(list((result or {}).get("messages") or [])))
        loop_stop_reason = _derive_sales_loop_guard_reason(result or state, warnings, tool_limit_hits)
        if clean_text(answer):
            result["runtime_warnings"] = merged_warnings
            result["tool_limit_hits"] = tool_limit_hits
            result["sales_loop_guard_reason"] = loop_stop_reason or result.get("sales_loop_guard_reason")
            result["llm_retry_instruction"] = ""
            return result

        if loop_stop_reason:
            LOG.warning("sales_response loop stopped; handing off to sales_continue reason=%s", loop_stop_reason)
            merged_warnings = _merge_runtime_warnings(
                {"runtime_warnings": merged_warnings},
                [_runtime_warning("gaz:sales_response", "sales_response_fallback_to_continue", loop_stop_reason)],
            )

        continue_state: GazAgentState = dict(state)
        continue_state.update({k: v for k, v in (result or {}).items() if k != "messages"})
        if (result or {}).get("messages"):
            continue_state["messages"] = list(result.get("messages") or [])
        continue_state["runtime_warnings"] = merged_warnings
        continue_state["tool_limit_hits"] = tool_limit_hits
        continue_state["sales_loop_guard_reason"] = loop_stop_reason or clean_text(continue_state.get("sales_loop_guard_reason")) or None
        continue_state["llm_retry_instruction"] = (
            "Answer now from already collected evidence and the baseline context. Do not call more tools. Explicitly avoid inventing exact unsupported figures."
            if loop_stop_reason
            else "Write a short natural customer-facing answer now. Continue the conversation without mentioning internal issues."
        )
        continue_result, _structured, continue_answer, continue_warnings = _invoke_agent_with_retry(
            sales_continue_agent,
            continue_state,
            config,
            runtime,
            stage_name="gaz:sales_continue",
            extra_tags=["gaz:sales_continue"],
            retry_instruction="Write one short direct customer-facing answer now. Continue naturally from the conversation. Do not return empty output.",
            expects_structured=False,
        )
        combined_warnings = _merge_runtime_warnings(continue_state, continue_warnings)
        if clean_text(continue_answer):
            continue_result["runtime_warnings"] = combined_warnings
            continue_result["tool_limit_hits"] = tool_limit_hits
            continue_result["sales_loop_guard_reason"] = loop_stop_reason or continue_result.get("sales_loop_guard_reason")
            continue_result["llm_retry_instruction"] = ""
            return continue_result
        locale = resolve_locale(state.get("locale"))
        combined_warnings.append(_runtime_warning("gaz:sales_continue", "generic_continuation_message_used"))
        return {
            "messages": [AIMessage(content=get_text(locale, "continue_message"))],
            "runtime_warnings": combined_warnings,
            "tool_limit_hits": tool_limit_hits,
            "sales_loop_guard_reason": loop_stop_reason or None,
            "llm_retry_instruction": "",
        }

    return sales_response_node



def create_response_finalize_node(validator, repair_agent):
    def response_finalize_node(state: GazAgentState, config: RunnableConfig, runtime: Runtime) -> GazAgentState:
        locale = resolve_locale(state.get("locale"))
        messages = list(state.get("messages") or [])
        answer = _last_ai_text(messages)
        validation = _validate_answer(validator, state, config, runtime, answer)
        runtime_warnings = _merge_runtime_warnings(state, validation.pop("_runtime_warnings", []))
        validator_guidance = ""
        validation_feedback: Dict[str, Any] = {}
        policy_checks = ["sales_answer_validated"]
        if runtime_warnings:
            policy_checks.append("validator_nonblocking_warning_recorded")
        message_delta: List[Any] = []
        if not validation.get("is_valid", True):
            if _is_soft_question_validation(validation):
                validator_guidance = _question_guidance(locale, validation)
                validation_feedback = _validation_feedback_payload(validation)
                policy_checks.append("question_guidance_saved_for_next_turn")
            else:
                repaired_answer, repair_warnings = _repair_answer(repair_agent, state, config, runtime, answer, validation)
                runtime_warnings = _merge_runtime_warnings({"runtime_warnings": runtime_warnings}, repair_warnings)
                if clean_text(repaired_answer):
                    answer = clean_text(repaired_answer)
                    message_delta = _replace_last_ai_message(messages, answer)
                    validation = _validate_answer(validator, state, config, runtime, answer)
                    runtime_warnings = _merge_runtime_warnings({"runtime_warnings": runtime_warnings}, validation.pop("_runtime_warnings", []))
                    if validation.get("is_valid", True):
                        policy_checks.append("sales_answer_repaired_after_validation")
                    elif _is_soft_question_validation(validation):
                        validator_guidance = _question_guidance(locale, validation)
                        validation_feedback = _validation_feedback_payload(validation)
                        policy_checks.append("sales_answer_repaired_but_question_guidance_saved")
                    else:
                        validation_feedback = _validation_feedback_payload(validation)
                        policy_checks.append("sales_answer_repair_attempt_failed_but_sent_as_is")
                else:
                    validation_feedback = _validation_feedback_payload(validation)
                    policy_checks.append("sales_answer_kept_as_is_after_failed_repair_attempt")
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
            policy_checks_passed=policy_checks,
        )
        result: GazAgentState = {
            "stage": "SELL",
            "trace": trace,
            "tool_calls_this_turn": [],
            "draft_answer": None,
            "validator_guidance": validator_guidance,
            "validation_feedback": validation_feedback,
            "planned_answer_depth": None,
            "needs_hitl_wait_confirmation": False,
            "hitl_eligible": False,
            "hitl_blocked_by_temperature": False,
            "hitl_blocked_by_first_turn_budget": False,
            "hitl_blocked_by_missing_prior_search": False,
            "hitl_trigger_kind": None,
            "hitl_reason": None,
            "research_wait_approved": False,
            "research_wait_rejected": False,
            "llm_retry_instruction": "",
            "runtime_warnings": runtime_warnings,
            "read_attempts_by_candidate": {},
            "read_focus_keys_this_turn": [],
            "search_keys_this_turn": [],
            "tool_limit_hits": [],
            "sales_loop_guard_reason": None,
        }
        if message_delta:
            result["messages"] = message_delta
        return result

    return response_finalize_node

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

    mini_llm = get_llm(model="mini", provider=provider.value, temperature=0.0, streaming=streaming)
    base_llm = get_llm(model="base", provider=provider.value, temperature=0.2, streaming=streaming, reasoning="medium")
    summary_llm = get_llm(model="mini", provider=provider.value, temperature=0.0, streaming=False)

    turn_intent_extractor = _build_turn_intent_extractor(mini_llm, summary_llm, locale_key, structured_output_strategy)
    answer_planner = _build_answer_planner(mini_llm, summary_llm, locale_key, structured_output_strategy)
    validator = _build_policy_validator(mini_llm, summary_llm, locale_key, structured_output_strategy)
    repair_agent = _build_sales_repair_agent(base_llm, summary_llm, locale_key)
    sales_continue_agent = _build_sales_continue_agent(base_llm, summary_llm, locale_key)

    tool_registry: Dict[str, Any] = {
        "get_sales_catalog_overview": build_sales_catalog_overview_tool(locale_key),
        "get_sales_landscape": build_sales_landscape_tool(locale_key, docs_client),
        "compare_product_directions": build_compare_product_directions_tool(locale_key, docs_client),
        "collect_product_snapshot": build_collect_product_snapshot_tool(locale_key, docs_client),
        "search_sales_materials": build_search_sales_materials_tool(docs_client),
        "read_material": build_read_material_tool(docs_client),
        "classify_problem_branch": build_classify_problem_branch_tool(locale_key),
        "get_branch_pack": build_branch_pack_tool(docs_client),
        "build_solution_shortlist": build_solution_shortlist_tool(),
        "build_followup_pack": build_followup_pack_tool(),
    }
    sales_response_agent = _build_sales_response_agent(
        base_llm,
        summary_llm,
        locale_key,
        list(tool_registry.values()),
        tool_registry,
    )

    builder = StateGraph(GazAgentState, config_schema=ConfigSchema)
    builder.add_node("init", create_init_node(docs_client))
    builder.add_node("reset", reset_node)
    builder.add_node("opening", opening_node)
    builder.add_node("turn_intent", create_turn_intent_node(turn_intent_extractor))
    builder.add_node("answer_plan", create_answer_plan_node(answer_planner, docs_client))
    builder.add_node("hitl_wait", hitl_wait_node)
    builder.add_node("sales_response", create_sales_response_node(sales_response_agent, sales_continue_agent))
    builder.add_node("response_finalize", create_response_finalize_node(validator, repair_agent))

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
    builder.add_edge("turn_intent", "answer_plan")
    builder.add_conditional_edges(
        "answer_plan",
        route_after_answer_plan,
        {
            "hitl_wait": "hitl_wait",
            "sales_response": "sales_response",
        },
    )
    builder.add_edge("hitl_wait", "sales_response")
    builder.add_edge("sales_response", "response_finalize")
    builder.add_edge("response_finalize", END)

    graph = builder.compile(checkpointer=memory, debug=False, name="gaz_agent").with_config({"callbacks": callback_handlers})
    return graph
